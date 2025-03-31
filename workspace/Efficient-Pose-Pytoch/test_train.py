import argparse
import time
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from generators.linemod import LineModGenerator
from tqdm import tqdm

from losses import smooth_l1, focal, transformation_loss
from model import EfficientPose
import numpy as np

loss_weights = [1.0, 1.0, 0.02]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args(args):
    """
    Parse the arguments.
    """
    date_and_time = time.strftime("%d_%m_%Y_%H_%M_%S")
    parser = argparse.ArgumentParser(description = 'Simple EfficientPose training script.')
    subparsers = parser.add_subparsers(help = 'Arguments for specific dataset types.', dest = 'dataset_type')
    subparsers.required = True
    
    linemod_parser = subparsers.add_parser('linemod')
    linemod_parser.add_argument('linemod_path', help = 'Path to dataset directory (ie. /Datasets/Linemod_preprocessed).')
    linemod_parser.add_argument('--object-id', help = 'ID of the Linemod Object to train on', type = int, default = 8)
    
    occlusion_parser = subparsers.add_parser('occlusion')
    occlusion_parser.add_argument('occlusion_path', help = 'Path to dataset directory (ie. /Datasets/Linemod_preprocessed/).')

    suspension_parser = subparsers.add_parser('suspension')
    suspension_parser.add_argument('suspension_path', help = 'Path to dataset directory')

    parser.add_argument('--rotation-representation', help = 'Which representation of the rotation should be used. Choose from "axis_angle", "rotation_matrix" and "quaternion"', default = 'axis_angle')    

    parser.add_argument('--weights', help = 'File containing weights to init the model parameter')
    parser.add_argument('--freeze-backbone', help = 'Freeze training of backbone layers.', action = 'store_true')
    parser.add_argument('--no-freeze-bn', help = 'Do not freeze training of BatchNormalization layers.', action = 'store_true')

    parser.add_argument('--batch-size', help = 'Size of the batches.', default = 1, type = int)
    parser.add_argument('--lr', help = 'Learning rate', default = 5e-6, type = float)
    parser.add_argument('--no-color-augmentation', help = 'Do not use colorspace augmentation', action = 'store_true')
    parser.add_argument('--no-6dof-augmentation', help = 'Do not use 6DoF augmentation', action = 'store_true')
    parser.add_argument('--phi', help = 'Hyper parameter phi', default = 0, type = int, choices = (0, 1, 2, 3, 4, 5, 6))
    parser.add_argument('--gpu', help = 'Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--epochs', help = 'Number of epochs to train.', type = int, default = 500)
    parser.add_argument('--steps', help = 'Number of steps per epoch.', type = int, default = int(179 * 10))
    parser.add_argument('--snapshot-path', help = 'Path to store snapshots of models during training', default = os.path.join("checkpoints", date_and_time))
    parser.add_argument('--tensorboard-dir', help = 'Log directory for Tensorboard output', default = os.path.join("logs", date_and_time))
    parser.add_argument('--no-snapshots', help = 'Disable saving snapshots.', dest = 'snapshots', action = 'store_false')
    parser.add_argument('--no-evaluation', help = 'Disable per epoch evaluation.', dest = 'evaluation', action = 'store_false')
    parser.add_argument('--compute-val-loss', help = 'Compute validation loss during training', dest = 'compute_val_loss', action = 'store_true')
    parser.add_argument('--score-threshold', help = 'score threshold for non max suppresion', type = float, default = 0.5)
    parser.add_argument('--validation-image-save-path', help = 'path where to save the predicted validation images after each epoch', default = None)

    # Fit generator arguments
    parser.add_argument('--multiprocessing', help = 'Use multiprocessing in fit_generator.', action = 'store_true')
    parser.add_argument('--workers', help = 'Number of generator workers.', type = int, default = 4)
    parser.add_argument('--max-queue-size', help = 'Queue length for multiprocessing workers in fit_generator.', type = int, default = 10)
    
    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)

def main(args = None):
    print(time.time)
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # create the generators
    print("\nCreating the Generators...")
    train_generator, validation_generator = create_generators(args)

    print("Train Generator: ", train_generator)

    num_rotation_parameters = train_generator.get_num_rotation_parameters()
    num_classes = train_generator.num_classes()
    num_anchors = train_generator.num_anchors

    print("Number of rotation parameters: ", num_rotation_parameters)
    print("Number of classes: ", num_classes)
    print("Number of anchors: ", num_anchors)

    print(f"Total number of items in train_gen: {len(train_generator)}")

    print("\nBuilding the Model...")
    model = EfficientPose(phi=args.phi,
                          num_classes=num_classes,
                          num_anchors=num_anchors,
                          freeze_bn=not args.no_freeze_bn,
                          score_threshold=args.score_threshold,
                          num_rotation_parameters=num_rotation_parameters).to(DEVICE)
    
    print("\nLoading weights from h5 file...")
    model.load_h5('phi_0_linemod_best_ADD.h5')
    model = model.to(DEVICE)
    print("Done!")

    regression_loss = smooth_l1()
    classification_loss = focal()
    model_3d_points = train_generator.get_all_3d_model_points_array_for_loss()
    transformation = transformation_loss(model_3d_points_np = torch.tensor(model_3d_points).to(DEVICE))

    loss_fn = [classification_loss, regression_loss, transformation]

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    print("\nStart training...")

    for epoch in range(args.epochs):
        model.train()
        print ("\nStarting epoch number", epoch,"...")
        train_fn(train_generator, model, optimizer, loss_fn, scaler, args)
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
         # check accuracy
        print ('Validation ...')
        validate_model(validation_generator, model, loss_fn, args)

def train_fn(loader, model, optimizer, loss_fn, scaler,args):

    loop = tqdm(range(args.steps), desc="Training")
    for step in loop:
        img_components, annotations = loader[step % len(loader)]

        image_input = torch.from_numpy(img_components[0]).permute(0, 3, 1, 2).float().to(DEVICE)
        camera_input = torch.from_numpy(img_components[1]).float().to(DEVICE)
        camera_input = camera_input.unsqueeze(1)  

        classification_targets = torch.from_numpy(annotations[0]).float().squeeze(0).to(DEVICE).unsqueeze(0)
        regression_targets     = torch.from_numpy(annotations[1]).float().squeeze(0).to(DEVICE).unsqueeze(0)
        transformation_targets = torch.from_numpy(annotations[2]).float().squeeze(0).to(DEVICE).unsqueeze(0)

        targets = [classification_targets, regression_targets, transformation_targets]

        # print("image_input shape: ", image_input.shape)
        # print("camera_input shape: ", camera_input.shape)
        # print("classification_targets shape: ", classification_targets.shape)
        # print("regression_targets shape: ", regression_targets.shape)
        # print("transformation_targets shape: ", transformation_targets.shape)

        with torch.cuda.amp.autocast():
            predictions = model(image_input, camera_input)
            losses = []
            for i in range(3):
                losses.append(loss_fn[i](targets[i], predictions[i]))
            loss = losses[0] * loss_weights[0] + losses[1] * loss_weights[1] + losses[2] * loss_weights[2]

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update progress bar
        loop.set_postfix(loss=loss.item())

        
def validate_model(val_generator, model, loss_fn, args):
    model.eval()
    for i in tqdm(range(len(val_generator)), desc='Running detection network'):
        # Load and preprocess image
        raw_image = val_generator.load_image(i)
        image, scale = val_generator.preprocess_image(raw_image.copy())
        camera_matrix = val_generator.load_camera_matrix(i)
        camera_input = val_generator.get_camera_parameter_input(camera_matrix, scale, val_generator.translation_scale_norm)
        
        # Convert to PyTorch tensors and move to device
        image_tensor = torch.from_numpy(np.expand_dims(image, axis=0)).permute(0, 3, 1, 2).float().to(DEVICE)

        camera_tensor = torch.from_numpy(np.expand_dims(camera_input, axis=0)).float().to(DEVICE)
        camera_tensor = camera_tensor.unsqueeze(1)  # shape: [1, 1, 6]

        # print("image_tensor shape: ", image_tensor.shape)
        # print("camera_tensor shape: ", camera_tensor.shape)

        with torch.cuda.amp.autocast():
            predictions = model(image_tensor,camera_tensor)

            print("predictions: ", predictions)

            boxes = predictions[0][0]
            scores = predictions[1][0]
            labels = predictions[2][0]
            rotations = predictions[3][0]
            translations = predictions[4][0]

            print("boxes shape: ", boxes.shape)
            print("scores shape: ", scores.shape)
            print("labels shape: ", labels.shape)
            print("rotations shape: ", rotations.shape)
            print("translations shape: ", translations.shape)

        # img_components, annotations = val_generator[i % len(val_generator)]

        # image_input = torch.from_numpy(img_components[0]).permute(0, 3, 1, 2).float().to(DEVICE)
        # camera_input = torch.from_numpy(img_components[1]).float().to(DEVICE)
        # camera_input = camera_input.unsqueeze(1)  

        # classification_targets = torch.from_numpy(annotations[0]).float().squeeze(0).to(DEVICE).unsqueeze(0)
        # regression_targets     = torch.from_numpy(annotations[1]).float().squeeze(0).to(DEVICE).unsqueeze(0)
        # transformation_targets = torch.from_numpy(annotations[2]).float().squeeze(0).to(DEVICE).unsqueeze(0)

        # with torch.cuda.amp.autocast():
        #     predictions = model(image_input,camera_input)

        #     print("predictions: ", predictions)
        



def check_accuracy (validation_loader, model, loss_fn):
    loop = tqdm(validation_loader)
    total_loss = 0
    for batch_idx, (data, targets) in enumerate(loop):
        images = data[0].to(device=DEVICE)
        Ks = torch.unsqueeze(data[1].to(device=DEVICE),dim=1)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(images,Ks)
            losses = []
            for i in range(3):
                losses.append((loss_fn[i](targets[i].to(DEVICE),predictions[i])))
            loss = losses[0]*loss_weights[0] + losses[1]*loss_weights[1] + losses[2]*loss_weights[2]


        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])




def create_generators(args):
    """
    Create generators for training and validation.

    Args:
        args: parseargs object containing configuration for generators.
    Returns:
        The training and validation generators.
    """
    common_args = {
        'batch_size': args.batch_size,
        'phi': args.phi,
    }


    if args.dataset_type == 'linemod':
        train_generator = LineModGenerator(
            args.linemod_path,
            args.object_id,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = not args.no_color_augmentation,
            use_6DoF_augmentation = not args.no_6dof_augmentation,
            **common_args
        )

        validation_generator = LineModGenerator(
            args.linemod_path,
            args.object_id,
            train = False,
            shuffle_dataset = False,
            shuffle_groups = False,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = False,
            use_6DoF_augmentation = False,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator

if __name__ == '__main__':
    main()