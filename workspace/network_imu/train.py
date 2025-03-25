import argparse
import time
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from model import BuildEfficientPoseModel
from losses import smooth_l1, focal, transformation_loss
import json

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
    parser.add_argument('--steps', help = 'Number of steps per epoch.', type = int, default = int(150 * 10))
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
    """
    Train an EfficientPose model.

    Args:
        args: parseargs object containing configuration for the training procedure.
    """
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


    # create the model
    # --- Build the Model ---
    print("\nBuilding the Model...")
    # Create an instance of your model
    # Create model
    model = BuildEfficientPoseModel(phi=args.phi,
                                num_classes=num_classes,
                                num_anchors=num_anchors,
                                freeze_bn=not args.no_freeze_bn,
                                score_threshold=args.score_threshold,
                                num_rotation_parameters=num_rotation_parameters)

    # Move the ENTIRE model to the device at once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = model.to(device)  # This moves ALL parameters to the device


    if args.weights:
        if args.weights == 'imagenet':
            print("Using ImageNet pretrained weights (already loaded in model constructor).")
        else:
            print("Custom weight loading not implemented; using default weights.")

    # --- Freeze Backbone Layers if Specified ---
    if args.freeze_backbone:
        for param in model.feature_extractor.parameters():
            param.requires_grad = False

    # Set Up Optimizer and Loss Functions
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    print("Optimizer:", optimizer)
    # Instantiate loss functions from losses.py.
    criterion_reg = smooth_l1()  # Smooth L1 loss functor.
    criterion_cls = focal()      # Focal loss functor.
    # transformation_loss requires a numpy array with 3D model points.
    criterion_trans = transformation_loss(train_generator.get_all_3d_model_points_array_for_loss(), num_rotation_parameters)
    loss_weights = {'regression': 1.0, 'classification': 1.0, 'transformation': 0.02}


    # Training Loop.
    num_epochs = args.epochs
    steps_per_epoch = args.steps
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_generator):
            images, camera_params, targets = data



            # Convert images: if not a tensor, convert and ensure they are in NCHW order.
            if not isinstance(images, torch.Tensor):
                images = torch.from_numpy(images)
            if images.ndim == 3:
                images = images.unsqueeze(0)
            if images.ndim == 4 and images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)
            images = images.float()

            # Convert camera_params.
            if not isinstance(camera_params, torch.Tensor):
                camera_params = torch.from_numpy(camera_params)
            # Convert targets.
            for key in targets:
                if not isinstance(targets[key], torch.Tensor):
                    targets[key] = torch.from_numpy(targets[key])
                    
            images = images.to(device)
            camera_params = camera_params.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            print("Images: ", images.shape)

            optimizer.zero_grad()
            classification, bbox_regression, transformation = model(images, camera_params, inference=False)
            loss_reg = criterion_reg(bbox_regression, targets['regression'])
            loss_cls = criterion_cls(classification, targets['classification'])
            loss_trans = criterion_trans(transformation, targets['transformation'])
            loss = (loss_weights['regression'] * loss_reg +
                    loss_weights['classification'] * loss_cls +
                    loss_weights['transformation'] * loss_trans)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{steps_per_epoch}], Loss: {loss.item():.4f}")
            if i + 1 >= steps_per_epoch:
                break

        avg_loss = running_loss / steps_per_epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # Validation.
        if validation_generator is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for j, data in enumerate(validation_generator):
                    images, camera_params, targets = data
                    if not isinstance(images, torch.Tensor):
                        images = torch.from_numpy(images)
                    if images.ndim == 3:
                        images = images.unsqueeze(0)
                    if images.ndim == 4 and images.shape[-1] == 3:
                        images = images.permute(0, 3, 1, 2)
                    images = images.float()
                    if not isinstance(camera_params, torch.Tensor):
                        camera_params = torch.from_numpy(camera_params)
                    for key in targets:
                        if not isinstance(targets[key], torch.Tensor):
                            targets[key] = torch.from_numpy(targets[key])
                    images = images.to(device)
                    camera_params = camera_params.to(device)
                    targets = {k: v.to(device) for k, v in targets.items()}
                    classification, bbox_regression, transformation = model(images, camera_params, inference=False)
                    loss_reg = criterion_reg(bbox_regression, targets['regression'])
                    loss_cls = criterion_cls(classification, targets['classification'])
                    loss_trans = criterion_trans(transformation, targets['transformation'])
                    loss = (loss_weights['regression'] * loss_reg +
                            loss_weights['classification'] * loss_cls +
                            loss_weights['transformation'] * loss_trans)
                    val_loss += loss.item()
                avg_val_loss = val_loss / len(validation_generator)
                print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
            model.train()

        # Save Checkpoint.
        if args.snapshots:
            snapshot_dir = args.snapshot_path
            os.makedirs(snapshot_dir, exist_ok=True)
            checkpoint_path = os.path.join(snapshot_dir, f'phi_{args.phi}_{args.dataset_type}_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")


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
        from generators.linemod import LineModGenerator
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
    elif args.dataset_type == 'occlusion':
        from generators.occlusion import OcclusionGenerator
        train_generator = OcclusionGenerator(
            args.occlusion_path,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = not args.no_color_augmentation,
            use_6DoF_augmentation = not args.no_6dof_augmentation,
            **common_args
        )
        validation_generator = OcclusionGenerator(
            args.occlusion_path,
            train = False,
            shuffle_dataset = False,
            shuffle_groups = False,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = False,
            use_6DoF_augmentation = False,
            **common_args
        )
    elif args.dataset_type == 'suspension':
        from generators.suspension import SuspensionGenerator
        train_generator = SuspensionGenerator(
            args.suspension_path,
            # object_ids = read_class_json,
            # symmetric_objects = read_symm_list_ff,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = not args.no_color_augmentation,
            use_6DoF_augmentation = not args.no_6dof_augmentation,
            **common_args
        )
        validation_generator = SuspensionGenerator(
            args.suspension_path,
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