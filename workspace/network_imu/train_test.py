import argparse
import time
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from model import BuildEfficientPoseModel
from losses import focal, smooth_l1, transformation_loss
import json
from utils.visualization import draw_detections, draw_annotations
import cv2
import math
from tqdm import tqdm
import numpy as np

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


    print(f"Total number of items in train_gen: {len(train_generator)}")



    

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

    # Start training
    train(
        model=model,
        train_generator=train_generator,
        val_generator=validation_generator,
        args=args,
        device=device, 
        num_rotation_parameters=num_rotation_parameters
    )


def train(model, train_generator, val_generator, args, device=None, num_rotation_parameters=None):


    model_3d_points_np = train_generator.get_all_3d_model_points_array_for_loss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    print("Optimizer:", optimizer)

    # Instantiate loss functions from losses.py.
    focal_loss = focal()      # Focal loss functor.
    smooth_l1_loss = smooth_l1()  # Smooth L1 loss functor
    trans_loss = transformation_loss(model_3d_points_np, num_rotation_parameters)
    loss_weights = {'regression': 1.0, 'classification': 1.0, 'transformation': 0.02}


    for epoch in range(args.epochs):
        model.train()

        # Reset metrics
        total_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        total_transf_loss = 0.0

        for step in range(args.steps):
            img_components, annotations = train_generator[step % len(train_generator)]
            image_input = torch.from_numpy(img_components[0]).permute(0, 3, 1, 2).float().to(device)
            camera_input = torch.from_numpy(img_components[1]).float().to(device)

            # print("\nStep:", step)
            # print("Image input shape:", image_input.shape)
            # print("Camera input shape:", camera_input.shape)

            classification_targets = torch.from_numpy(annotations[0]).float().squeeze(0).to(device)
            classification_targets = classification_targets.unsqueeze(0)
            regression_targets     = torch.from_numpy(annotations[1]).float().squeeze(0).to(device)
            regression_targets     = regression_targets.unsqueeze(0)
            transformation_targets = torch.from_numpy(annotations[2]).float().squeeze(0).to(device)
            transformation_targets = transformation_targets.unsqueeze(0)


            # print("Classification targets shape:", classification_targets.shape)
            # print("Regression targets shape:", regression_targets.shape)
            # print("Transformation targets shape:", transformation_targets.shape)

            # Forward pass
            classification, bbox_regression, transformation = model(image_input, camera_input)

            # print("Classification shape:", classification.shape)
            # print("Bbox regression shape:", bbox_regression.shape)
            # print("Transformation shape:", transformation.shape)

            cls_loss = focal_loss(classification_targets, classification)
            reg_loss = smooth_l1_loss(regression_targets, bbox_regression)
            transformation_loss_val = trans_loss(transformation_targets, transformation)

            # print("-----> Classification loss:", cls_loss)
            # print("-----> Regression loss:", reg_loss)
            # print("-----> Transformation loss:", transformation_loss_val)

            # Total weighted loss
            loss = (loss_weights['classification'] * cls_loss + 
                    loss_weights['regression'] * reg_loss +
                    loss_weights['transformation'] * transformation_loss_val)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (same as in Keras version)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            total_transf_loss += transformation_loss_val.item()
            
            # Print progress every 10 steps
            if (step + 1) % 10 == 0:
                print(f"Step: {step+1}/{args.steps} - Loss: {loss.item():.4f}, "
                      f"Cls: {cls_loss.item():.4f}, Reg: {reg_loss.item():.4f}, "
                      f"Trans: {transformation_loss_val.item():.4f}")
        
        # Epoch summary
        avg_loss = total_loss / args.steps
        avg_cls_loss = total_cls_loss / args.steps
        avg_reg_loss = total_reg_loss / args.steps
        avg_trans_loss = total_transf_loss / args.steps
        
        print(f"Training - Loss: {avg_loss:.4f}, Cls: {avg_cls_loss:.4f}, "
              f"Reg: {avg_reg_loss:.4f}, Trans: {avg_trans_loss:.4f}")


        # Validation
        if val_generator is not None:
            model.eval()


            if args.validation_image_save_path:
                if args.dataset_type == 'linemod':
                    save_path = os.path.join(args.validation_image_save_path, f"object_{args.object_id}", f"epoch_{epoch+1}")
                else:
                    save_path = os.path.join(args.validation_image_save_path, f"epoch_{epoch+1}")
                
                os.makedirs(save_path, exist_ok=True)
            else:
                save_path = None

                
            validate_model(
                model, val_generator, focal_loss, smooth_l1_loss, trans_loss,
                loss_weights, device, args, epoch=epoch, save_path=save_path, 
            )
            

def validate_model(model, val_generator, focal_loss, smooth_l1_loss, trans_loss, 
                   loss_weights, device, args, epoch=0, save_path=None, score_threshold=0.1):
    """
    Validate the model on validation data.
    """
    model.eval()  # Set model to evaluation mode
    max_detections=50
    detection_count = 0
    # Process each image in the validation generator
    for i in tqdm(range(len(val_generator)), desc='Running detection network'):
        # Load and preprocess image
        raw_image = val_generator.load_image(i)
        image, scale = val_generator.preprocess_image(raw_image.copy())
        camera_matrix = val_generator.load_camera_matrix(i)
        camera_input = val_generator.get_camera_parameter_input(camera_matrix, scale, val_generator.translation_scale_norm)
        
        # Convert to PyTorch tensors and move to device
        image_tensor = torch.from_numpy(np.expand_dims(image, axis=0)).permute(0, 3, 1, 2).float().to(device)
        camera_tensor = torch.from_numpy(np.expand_dims(camera_input, axis=0)).float().to(device)

        # print("Image tensor shape:", image_tensor.shape)
        # print("Camera tensor shape:", camera_tensor.shape)

        detections = model(image_tensor, camera_tensor, inference=True)

        # Extract outputs from the model
        boxes = detections[0][0].cpu().numpy().astype(np.float32)  # Convert to float32
        scores = detections[1][0].cpu().numpy()
        labels = detections[2][0].cpu().numpy().astype(int)
        rotations = detections[3][0].cpu().numpy().astype(np.float32)  # Convert to float32 here
        translations = detections[4][0].cpu().numpy()

        # Now these will work without errors
        boxes /= scale
        rotations *= math.pi
        
        # Select indices which have a score above the threshold
        indices = np.where(scores > score_threshold)[0]

        detection_count += len(indices)
        
        # Select those scores
        scores = scores[indices]
                
        # No detections above threshold
        if len(scores) == 0:
            # Create empty arrays with correct shapes
            image_boxes = np.zeros((0, 4))
            image_rotations = np.zeros((0, rotations.shape[1]))
            image_translations = np.zeros((0, translations.shape[1]))
            image_scores = np.zeros((0,))
            image_labels = np.zeros((0,), dtype=np.int32)
            image_detections = np.zeros((0, 6))  # 4 + 1 + 1 (box, score, label)
        else:
            # Find the order with which to sort the scores
            scores_sort = np.argsort(-scores)[:max_detections]
            
            # Select detections
            image_boxes = boxes[indices[scores_sort], :]
            image_rotations = rotations[indices[scores_sort], :]
            image_translations = translations[indices[scores_sort], :]
            image_scores = scores[scores_sort]
            image_labels = labels[indices[scores_sort]]
            
            # Concatenate for storage
            image_detections = np.concatenate([
                image_boxes, 
                np.expand_dims(image_scores, axis=1), 
                np.expand_dims(image_labels, axis=1)
            ], axis=1)

        # Save visualization if requested
        if save_path is not None:
            try:
                # Make a copy and convert to BGR for OpenCV
                if raw_image.shape[2] == 3:  # RGB image
                    vis_image = cv2.cvtColor(raw_image.copy(), cv2.COLOR_RGB2BGR)
                else:
                    vis_image = raw_image.copy()
                
                # Draw ground truth annotations
                from utils.visualization import draw_annotations
                draw_annotations(
                    vis_image, 
                    val_generator.load_annotations(i), 
                    class_to_bbox_3D=val_generator.get_bbox_3d_dict(), 
                    camera_matrix=camera_matrix, 
                    label_to_name=val_generator.label_to_name
                )
                
                # Draw detections if any were found
                if len(image_boxes) > 0:
                    from utils.visualization import draw_detections
                    draw_detections(
                        vis_image, 
                        image_boxes, 
                        image_scores, 
                        image_labels, 
                        image_rotations, 
                        image_translations, 
                        class_to_bbox_3D=val_generator.get_bbox_3d_dict(), 
                        camera_matrix=camera_matrix, 
                        label_to_name=val_generator.label_to_name
                    )
                else:
                    print(f"No detections found for image {i}")
                
                img_save_path = os.path.join(save_path, f'epoch_{epoch}_img_{i}.jpg')
                cv2.imwrite(img_save_path, vis_image)

                
            except Exception as e:
                print(f"Error saving visualization for image {i}: {str(e)}")

    print(f"Total detections across all images: {detection_count}")

def calculate_pose_metrics(predictions, ground_truths, val_generator):
    print("Calculating pose metrics...")



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
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator


if __name__ == '__main__':
    main()

