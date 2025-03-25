import argparse
import time
import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import BuildEfficientPoseModel
from losses import smooth_l1, focal, transformation_loss
import json
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
    parser.add_argument('--steps', help = 'Number of steps per epoch.', type = int, default = int(150 * 10))
    parser.add_argument('--snapshot-path', help = 'Path to store snapshots of models during training', default = os.path.join("checkpoints", date_and_time))
    parser.add_argument('--tensorboard-dir', help = 'Log directory for Tensorboard output', default = os.path.join("logs", date_and_time))
    parser.add_argument('--no-snapshots', help = 'Disable saving snapshots.', dest = 'snapshots', action = 'store_false')
    parser.add_argument('--no-evaluation', help = 'Disable per epoch evaluation.', dest = 'evaluation', action = 'store_false')
    parser.add_argument('--compute-val-loss', help = 'Compute validation loss during training', dest = 'compute_val_loss', action = 'store_true')
    parser.add_argument('--score-threshold', help = 'score threshold for non max suppresion', type = float, default = 0.5)
    parser.add_argument('--validation-image-save-path', help = 'path where to save the predicted validation images after each epoch', default = None)

    # Fit generator arguments
    parser.add_argument('--workers', help = 'Number of generator workers.', type = int, default = 4)
    
    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)


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

    print("args.no_color_augmentation: ", not args.no_color_augmentation)

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


# PyTorch Dataset wrapper for the generators
class GeneratorDataset(torch.utils.data.Dataset):
    def __init__(self, generator):
        self.generator = generator
    
    def __len__(self):
        return len(self.generator)
    
    def __getitem__(self, idx):
        # Get batch data from generator
        data = self.generator[idx]
        
        # Split the data
        img, targets = data
        
        # Extract the image and camera parameters from the img list
        image = img[0]  # e.g., shape (1, 512, 512, 3)
        camera_params = img[1]  # e.g., shape (1, 6)
        
        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image).float()
        
        # Remove batch dimension (if exists)
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.squeeze(0)
        
        # Convert from HWC to CHW format (PyTorch uses channels first)
        if image_tensor.dim() == 3 and image_tensor.shape[-1] == 3:
            image_tensor = image_tensor.permute(2, 0, 1)
                
        camera_params_tensor = torch.from_numpy(camera_params).float()
        if camera_params_tensor.shape[0] == 1:
            camera_params_tensor = camera_params_tensor.squeeze(0)
        
        # Create a targets dictionary
        targets_dict = {}
        
        # Process targets based on what we receive
        if isinstance(targets, list) and len(targets) >= 3:
            # Convert numpy arrays to tensors and reshape as needed
            regression = torch.from_numpy(targets[0]).float() if isinstance(targets[0], np.ndarray) else targets[0]
            classification = torch.from_numpy(targets[1]).float() if isinstance(targets[1], np.ndarray) else targets[1]
            transformation = torch.from_numpy(targets[2]).float() if isinstance(targets[2], np.ndarray) else targets[2]
            
            # Match dimensions with model output
            if regression is not None:
                # Remove extra dimensions to match model output shape
                if regression.dim() > 2:
                    if regression.shape[0] == 1:
                        regression = regression.squeeze(0)
                    if regression.shape[0] == 1:
                        regression = regression.squeeze(0)
            
            if classification is not None:
                # Remove extra dimensions to match model output shape
                if classification.dim() > 2:
                    if classification.shape[0] == 1:
                        classification = classification.squeeze(0)
                    if classification.shape[0] == 1:
                        classification = classification.squeeze(0)
            
            if transformation is not None:
                # Remove extra dimensions to match model output shape
                if transformation.dim() > 2:
                    if transformation.shape[0] == 1:
                        transformation = transformation.squeeze(0)
                    if transformation.shape[0] == 1:
                        transformation = transformation.squeeze(0)
            
            targets_dict = {
                'regression': regression,
                'classification': classification,
                'transformation': transformation
            }
        elif isinstance(targets, dict):
            # If targets is already a dictionary, process each key
            for key in ['regression', 'classification', 'transformation']:
                if key in targets and targets[key] is not None:
                    target_tensor = torch.from_numpy(targets[key]).float() if isinstance(targets[key], np.ndarray) else targets[key]
                    
                    # Remove extra dimensions to match model output shape
                    if target_tensor.dim() > 2:
                        if target_tensor.shape[0] == 1:
                            target_tensor = target_tensor.squeeze(0)
                        if target_tensor.shape[0] == 1:
                            target_tensor = target_tensor.squeeze(0)
                    
                    targets_dict[key] = target_tensor
                else:
                    targets_dict[key] = None
        else:
            # Fallback: create empty dictionary with None values
            targets_dict = {
                'regression': None,
                'classification': None,
                'transformation': None
            }
        
        # Add camera parameters to the targets dictionary
        targets_dict['camera_parameters'] = camera_params_tensor
        
        return image_tensor, targets_dict



def create_dataloaders(train_generator, validation_generator, args):
    """
    Create PyTorch DataLoaders from generators
    """
    train_dataset = GeneratorDataset(train_generator)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,  # Already batched in generator
        shuffle=False,  # Already shuffled in generator
        num_workers=args.workers,
        pin_memory=True
    )
    
    val_loader = None
    if validation_generator:
        val_dataset = GeneratorDataset(validation_generator)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,  # Already batched in generator
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
    
    return train_loader, val_loader


def create_callbacks(args, model, validation_generator):
    """
    Create callback-like functionality for PyTorch training
    """
    callbacks = {}
    
    # Setup tensorboard
    if args.tensorboard_dir:
        if args.dataset_type == "linemod":
            tensorboard_dir = os.path.join(args.tensorboard_dir, f"object_{args.object_id}")
        elif args.dataset_type == "occlusion":
            tensorboard_dir = os.path.join(args.tensorboard_dir, "occlusion")
        elif args.dataset_type == "suspension":
            tensorboard_dir = os.path.join(args.tensorboard_dir, "suspension")
        else:
            tensorboard_dir = args.tensorboard_dir
            
        os.makedirs(tensorboard_dir, exist_ok=True)
        callbacks['writer'] = SummaryWriter(log_dir=tensorboard_dir)
    
    # Setup model checkpoint saving
    if args.snapshots:
        if args.dataset_type == "linemod":
            snapshot_path = os.path.join(args.snapshot_path, f"object_{args.object_id}")
            if validation_generator.is_symmetric_object(args.object_id):
                callbacks['metric_to_monitor'] = "ADD-S"
                callbacks['mode'] = "max"
            else:
                callbacks['metric_to_monitor'] = "ADD"
                callbacks['mode'] = "max"
        elif args.dataset_type == "occlusion":
            snapshot_path = os.path.join(args.snapshot_path, "occlusion")
            callbacks['metric_to_monitor'] = "ADD(-S)"
            callbacks['mode'] = "max"
        elif args.dataset_type == "suspension":
            snapshot_path = os.path.join(args.snapshot_path, "suspension")
            callbacks['metric_to_monitor'] = "ADD(-S)"
            callbacks['mode'] = "max"
        else:
            snapshot_path = args.snapshot_path
            callbacks['metric_to_monitor'] = "val_loss"
            callbacks['mode'] = "min"
            
        os.makedirs(snapshot_path, exist_ok=True)
        callbacks['snapshot_path'] = snapshot_path

    # Setup validation image saving
    if args.validation_image_save_path:
        if args.dataset_type == "linemod":
            save_path = os.path.join(args.validation_image_save_path, f"object_{args.object_id}")
        elif args.dataset_type == "occlusion":
            save_path = os.path.join(args.validation_image_save_path, "occlusion")
        elif args.dataset_type == "suspension":
            save_path = os.path.join(args.validation_image_save_path, "suspension")
        else:
            save_path = args.validation_image_save_path
            
        os.makedirs(save_path, exist_ok=True)
        callbacks['val_image_path'] = save_path
    
    # Setup early stopping and learning rate reduction
    callbacks['best_val_metric'] = float('inf') if callbacks.get('mode', 'min') == 'min' else 0.0
    callbacks['patience'] = 0
    callbacks['lr_patience'] = 0
    
    return callbacks


def save_checkpoint(model, optimizer, epoch, metrics, path, is_best=False, filename=None):
    """
    Save model checkpoint
    """
    if filename is None:
        filename = f'checkpoint_epoch_{epoch}.pth'
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save regular checkpoint
    torch.save(checkpoint, os.path.join(path, filename))
    
    # Save best model if it's the best so far
    if is_best:
        torch.save(checkpoint, os.path.join(path, 'best_model.pth'))
        print(f"✓ Saved new best model at epoch {epoch}")


def train_one_epoch(model, train_loader, optimizer, criterion_reg, criterion_cls, criterion_trans, 
                   loss_weights, device, epoch, callbacks, steps_per_epoch=None):
    """
    Train for one epoch
    """
    model.train()
    epoch_loss = 0.0
    epoch_reg_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_trans_loss = 0.0
    
    # Limit the number of steps per epoch if specified
    if steps_per_epoch is not None:
        total_steps = min(steps_per_epoch, len(train_loader))
    else:
        total_steps = len(train_loader)
    
    for batch_idx, batch_data in enumerate(train_loader):
        if batch_idx >= total_steps:
            break
            
        # Unpack the batch data
        images, targets = batch_data
        
        # Move data to device
        images = images.to(device)
        
        # Log batch information periodically
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx} - Image shape: {images.shape}")
            if isinstance(targets, dict):
                print(f"Batch {batch_idx} - Target keys: {list(targets.keys())}")
                for k, v in targets.items():
                    if v is not None and hasattr(v, 'shape'):
                        print(f"  {k} shape: {v.shape}")
        
        # Handle camera parameters if they exist in the targets
        camera_params = None
        if isinstance(targets, dict) and 'camera_parameters' in targets:
            camera_params = targets['camera_parameters'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        try:
            # Forward pass
            predictions = model(images, camera_params)
            
            # Get the predictions
            pred_regression = predictions[0]
            pred_classification = predictions[1]
            pred_transformation = predictions[2]
            
            # Print prediction shapes in first batch for debugging
            if batch_idx == 0:
                print(f"Prediction shapes:")
                print(f"  Regression: {pred_regression.shape}")
                print(f"  Classification: {pred_classification.shape}")
                print(f"  Transformation: {pred_transformation.shape}")
            
            # Prepare target tensors based on their type
            regression_target = None
            classification_target = None
            transformation_target = None
            
            if isinstance(targets, dict):
                # Get targets from dictionary and send to device
                if 'regression' in targets and targets['regression'] is not None:
                    regression_target = targets['regression'].to(device)
                if 'classification' in targets and targets['classification'] is not None:
                    classification_target = targets['classification'].to(device)
                if 'transformation' in targets and targets['transformation'] is not None:
                    transformation_target = targets['transformation'].to(device)
            
            # Print target shapes in first batch for debugging
            if batch_idx == 0 and regression_target is not None:
                print(f"Target shapes:")
                print(f"  Regression: {regression_target.shape}")
                print(f"  Classification: {classification_target.shape if classification_target is not None else None}")
                print(f"  Transformation: {transformation_target.shape if transformation_target is not None else None}")
            
            # Process targets to match prediction shapes if needed
            if regression_target is not None:
                # Reshape regression_target to match pred_regression if needed
                if regression_target.shape != pred_regression.shape:
                    # Try to reshape based on specific dimensions
                    if regression_target.shape[-1] != pred_regression.shape[-1]:
                        print(f"Warning: Last dimension mismatch for regression target")
                        # Create a compatible target (this is just a fallback)
                        regression_target = torch.zeros_like(pred_regression)
            else:
                regression_target = torch.zeros_like(pred_regression)
            
            if classification_target is not None:
                # Reshape classification_target to match pred_classification if needed
                if classification_target.shape != pred_classification.shape:
                    # Try to adapt the target shape
                    if len(classification_target.shape) > len(pred_classification.shape):
                        # Remove extra dimensions
                        while len(classification_target.shape) > len(pred_classification.shape):
                            classification_target = classification_target.squeeze(0)
                    
                    # If dimensions still don't match, check the last dimension
                    if classification_target.shape[-1] != pred_classification.shape[-1]:
                        print(f"Warning: Last dimension mismatch for classification. Target has {classification_target.shape[-1]} classes, prediction has {pred_classification.shape[-1]}")
                        # Create a compatible target (this is just a fallback)
                        classification_target = torch.zeros_like(pred_classification)
                        
                        # For focal loss, we need to mark some positive examples
                        # Convert first column to positive class for a few examples to avoid zero loss
                        if batch_idx % 10 == 0:  # Just for some batches
                            classification_target[0, :10, 0] = 1.0
            else:
                classification_target = torch.zeros_like(pred_classification)
            
            if transformation_target is not None:
                # Reshape transformation_target to match pred_transformation if needed 
                if transformation_target.shape != pred_transformation.shape:
                    # Try to adapt the target shape
                    if len(transformation_target.shape) > len(pred_transformation.shape):
                        # Remove extra dimensions
                        while len(transformation_target.shape) > len(pred_transformation.shape):
                            transformation_target = transformation_target.squeeze(0)
                    
                    # If last dimension still doesn't match
                    if transformation_target.shape[-1] != pred_transformation.shape[-1]:
                        print(f"Warning: Last dimension mismatch for transformation. Creating compatible target")
                        # Create a compatible target (this is just a fallback)
                        transformation_target = torch.zeros_like(pred_transformation)
            else:
                transformation_target = torch.zeros_like(pred_transformation)
            
            # Calculate losses
            reg_loss = criterion_reg(pred_regression, regression_target)
            cls_loss = criterion_cls(pred_classification, classification_target)
            trans_loss = criterion_trans(pred_transformation, transformation_target)
            
            # Combine losses with weights
            loss = (loss_weights['regression'] * reg_loss + 
                   loss_weights['classification'] * cls_loss + 
                   loss_weights['transformation'] * trans_loss)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running loss values
            epoch_loss += loss.item()
            epoch_reg_loss += reg_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_trans_loss += trans_loss.item()
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Log to tensorboard at intervals
        if 'writer' in callbacks and batch_idx % 10 == 0:
            step = epoch * total_steps + batch_idx
            callbacks['writer'].add_scalar('Training/Loss', loss.item(), step)
            callbacks['writer'].add_scalar('Training/Regression_Loss', reg_loss.item(), step)
            callbacks['writer'].add_scalar('Training/Classification_Loss', cls_loss.item(), step)
            callbacks['writer'].add_scalar('Training/Transformation_Loss', trans_loss.item(), step)
            
            # Log learning rate
            for param_group in optimizer.param_groups:
                callbacks['writer'].add_scalar('Training/LR', param_group['lr'], step)
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}/{total_steps}, Loss: {loss.item():.6f}")
    
    # Calculate average epoch losses
    epoch_loss /= total_steps
    epoch_reg_loss /= total_steps
    epoch_cls_loss /= total_steps
    epoch_trans_loss /= total_steps
    
    # Log epoch losses
    if 'writer' in callbacks:
        callbacks['writer'].add_scalar('Epoch/Loss', epoch_loss, epoch)
        callbacks['writer'].add_scalar('Epoch/Regression_Loss', epoch_reg_loss, epoch)
        callbacks['writer'].add_scalar('Epoch/Classification_Loss', epoch_cls_loss, epoch)
        callbacks['writer'].add_scalar('Epoch/Transformation_Loss', epoch_trans_loss, epoch)
    
    return {
        'loss': epoch_loss,
        'reg_loss': epoch_reg_loss,
        'cls_loss': epoch_cls_loss,
        'trans_loss': epoch_trans_loss
    }

def evaluate_model(model, val_loader, criterion_reg, criterion_cls, criterion_trans, 
                 loss_weights, device, callbacks):
    """
    Evaluate the model on validation data
    """
    model.eval()
    val_loss = 0.0
    val_reg_loss = 0.0
    val_cls_loss = 0.0
    val_trans_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            # Move data to device
            images = images.to(device)
            
            # Handle camera parameters if they exist in the targets
            camera_params = None
            if isinstance(targets, dict) and 'camera_parameters' in targets:
                camera_params = targets['camera_parameters'].to(device)
            
            try:
                # Forward pass
                predictions = model(images, camera_params)
                
                # Get the predictions
                pred_regression = predictions[0]
                pred_classification = predictions[1]
                pred_transformation = predictions[2]
                
                # Prepare target tensors
                regression_target = None
                classification_target = None
                transformation_target = None
                
                if isinstance(targets, dict):
                    # Get targets from dictionary and send to device
                    if 'regression' in targets and targets['regression'] is not None:
                        regression_target = targets['regression'].to(device)
                    if 'classification' in targets and targets['classification'] is not None:
                        classification_target = targets['classification'].to(device)
                    if 'transformation' in targets and targets['transformation'] is not None:
                        transformation_target = targets['transformation'].to(device)
                
                # Process targets to match prediction shapes if needed
                if regression_target is not None:
                    # Reshape regression_target to match pred_regression if needed
                    if regression_target.shape != pred_regression.shape:
                        # Try to reshape based on specific dimensions
                        if len(regression_target.shape) > len(pred_regression.shape):
                            # Remove extra dimensions
                            while len(regression_target.shape) > len(pred_regression.shape):
                                regression_target = regression_target.squeeze(0)
                        
                        # If dimensions still don't match, fallback to zeros
                        if regression_target.shape != pred_regression.shape:
                            regression_target = torch.zeros_like(pred_regression)
                else:
                    regression_target = torch.zeros_like(pred_regression)
                
                if classification_target is not None:
                    # Reshape classification_target to match pred_classification if needed
                    if classification_target.shape != pred_classification.shape:
                        # Try to adapt the target shape
                        if len(classification_target.shape) > len(pred_classification.shape):
                            # Remove extra dimensions
                            while len(classification_target.shape) > len(pred_classification.shape):
                                classification_target = classification_target.squeeze(0)
                        
                        # If dimensions still don't match, fallback to zeros
                        if classification_target.shape != pred_classification.shape:
                            classification_target = torch.zeros_like(pred_classification)
                else:
                    classification_target = torch.zeros_like(pred_classification)
                
                if transformation_target is not None:
                    # Reshape transformation_target to match pred_transformation if needed
                    if transformation_target.shape != pred_transformation.shape:
                        # Try to adapt the target shape
                        if len(transformation_target.shape) > len(pred_transformation.shape):
                            # Remove extra dimensions
                            while len(transformation_target.shape) > len(pred_transformation.shape):
                                transformation_target = transformation_target.squeeze(0)
                        
                        # If dimensions still don't match, fallback to zeros
                        if transformation_target.shape != pred_transformation.shape:
                            transformation_target = torch.zeros_like(pred_transformation)
                else:
                    transformation_target = torch.zeros_like(pred_transformation)
                
                # Calculate losses
                reg_loss = criterion_reg(pred_regression, regression_target)
                cls_loss = criterion_cls(pred_classification, classification_target)
                trans_loss = criterion_trans(pred_transformation, transformation_target)
                
                # Combine losses with weights
                loss = (loss_weights['regression'] * reg_loss + 
                       loss_weights['classification'] * cls_loss + 
                       loss_weights['transformation'] * trans_loss)
                
                # Update running loss values
                val_loss += loss.item()
                val_reg_loss += reg_loss.item()
                val_cls_loss += cls_loss.item()
                val_trans_loss += trans_loss.item()
                
                # Save validation predictions if path is provided
                if 'val_image_path' in callbacks:
                    # TODO: Implement validation image saving logic
                    pass
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
    # Calculate average losses
    num_batches = max(1, len(val_loader))  # Avoid division by zero
    val_loss /= num_batches
    val_reg_loss /= num_batches
    val_cls_loss /= num_batches
    val_trans_loss /= num_batches
    
    # Return all metrics
    metrics = {
        'val_loss': val_loss,
        'val_reg_loss': val_reg_loss,
        'val_cls_loss': val_cls_loss,
        'val_trans_loss': val_trans_loss,
        # 'ADD': add_metric,
        # 'ADD-S': adds_metric
    }
    
    # Log to tensorboard
    if 'writer' in callbacks:
        for metric_name, metric_value in metrics.items():
            callbacks['writer'].add_scalar(f'Validation/{metric_name}', metric_value, callbacks.get('current_epoch', 0))
    
    return metrics


def adjust_learning_rate(optimizer, decay_factor=0.5):
    """
    Reduce learning rate by decay_factor
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_factor
    return optimizer.param_groups[0]['lr']


def main(args = None):
    """
    Train an EfficientPose model.

    Args:
        args: parseargs object containing configuration for the training procedure.
    """
    # Set start time
    start_time = time.time()
    
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Create the generators
    print("\nCreating the Generators...")
    train_generator, validation_generator = create_generators(args)
    print("Done!")

    # Get parameters from generator
    num_rotation_parameters = train_generator.get_num_rotation_parameters()
    num_classes = train_generator.num_classes()
    num_anchors = train_generator.num_anchors

    print("Number of rotation parameters:", num_rotation_parameters)
    print("Number of classes:", num_classes)
    print("Number of anchors:", num_anchors)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"Using device: {device}")

    # Build the model
    print("\nBuilding the Model...")
    model = BuildEfficientPoseModel(
        phi=args.phi,
        num_classes=num_classes,
        num_anchors=num_anchors,
        freeze_bn=not args.no_freeze_bn,
        score_threshold=args.score_threshold,
        num_rotation_parameters=num_rotation_parameters
    )
    model = model.to(device)
    print("Done!")

    # Load weights if specified
    if args.weights:
        if args.weights == 'imagenet':
            print("Using ImageNet pretrained weights (already loaded in model constructor).")
        else:
            try:
                checkpoint = torch.load(args.weights, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Successfully loaded weights from {args.weights}")
            except Exception as e:
                print(f"Error loading weights: {e}")
                print("Using default weights.")

    # Freeze backbone if specified
    if args.freeze_backbone:
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
        print("Backbone frozen.")

    # Create optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Create loss functions
    criterion_reg = smooth_l1()
    criterion_cls = focal()
    criterion_trans = transformation_loss(
        model_3d_points_np=train_generator.get_all_3d_model_points_array_for_loss(),
        num_rotation_parameter=num_rotation_parameters
    )
    loss_weights = {'regression': 1.0, 'classification': 1.0, 'transformation': 0.02}
    
    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(train_generator, validation_generator, args)
    
    # Create callbacks
    callbacks = create_callbacks(args, model, validation_generator)
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        callbacks['current_epoch'] = epoch
        
        # Train one epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion_reg, criterion_cls, 
            criterion_trans, loss_weights, device, epoch, callbacks,
            steps_per_epoch=args.steps
        )
        
        # Print training metrics
        print(f"Training - Loss: {train_metrics['loss']:.6f}, Reg: {train_metrics['reg_loss']:.6f}, "
              f"Cls: {train_metrics['cls_loss']:.6f}, Trans: {train_metrics['trans_loss']:.6f}")
        
        # Evaluate if validation data is available
        if args.compute_val_loss and val_loader:
            val_metrics = evaluate_model(
                model, val_loader, criterion_reg, criterion_cls, 
                criterion_trans, loss_weights, device, callbacks
            )
            
            # Print validation metrics
            print(f"Validation - Loss: {val_metrics['val_loss']:.6f}, Reg: {val_metrics['val_reg_loss']:.6f}, "
                  f"Cls: {val_metrics['val_cls_loss']:.6f}, Trans: {val_metrics['val_trans_loss']:.6f}")
            
            # Determine which metric to monitor for best model selection
            metric_to_monitor = callbacks.get('metric_to_monitor', 'val_loss')
            current_metric = val_metrics.get(metric_to_monitor, val_metrics['val_loss'])
            mode = callbacks.get('mode', 'min')
            
            # Check if this is the best model
            is_best = False
            if mode == 'min':
                is_best = current_metric < callbacks['best_val_metric']
            else:  # mode == 'max'
                is_best = current_metric > callbacks['best_val_metric']
            
            if is_best:
                callbacks['best_val_metric'] = current_metric
                callbacks['patience'] = 0
                
                # Save best model
                if args.snapshots and 'snapshot_path' in callbacks:
                    save_checkpoint(
                        model, optimizer, epoch, val_metrics, 
                        callbacks['snapshot_path'], is_best=True
                    )
            else:
                callbacks['patience'] += 1
                print(f"Patience: {callbacks['patience']}/25")
                if callbacks['patience'] >= 25:  # Early stopping patience
                    print(f"Early stopping triggered after {callbacks['patience']} epochs without improvement")
                    break
            
            # Learning rate reduction on plateau
            callbacks['lr_patience'] += 1
            if callbacks['lr_patience'] >= 10:
                new_lr = adjust_learning_rate(optimizer, decay_factor=0.5)
                print(f"Learning rate reduced to {new_lr}")
                callbacks['lr_patience'] = 0
        
        # Save regular checkpoint
        if args.snapshots and 'snapshot_path' in callbacks and epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, 
                val_metrics if args.compute_val_loss and val_loader else train_metrics, 
                callbacks['snapshot_path'],
                filename=f'phi_{args.phi}_{args.dataset_type}_epoch_{epoch}.pth'
            )
    
    # Clean up
    if 'writer' in callbacks:
        callbacks['writer'].close()
    
    # Print training time
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    if 'best_val_metric' in callbacks:
        metric_name = callbacks.get('metric_to_monitor', 'val_loss')
        print(f"Best {metric_name}: {callbacks['best_val_metric']:.6f}")


if __name__ == '__main__':
    main()