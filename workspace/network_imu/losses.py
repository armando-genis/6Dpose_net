import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def focal(alpha=0.25, gamma=1.5):
    """
    Create a functor for computing the focal loss in PyTorch.

    Args:
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns:
        A functor that computes the focal loss using the alpha and gamma.
    """

    def _focal(y_true, y_pred):
        """
        Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args:
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns:
            The focal loss of y_pred w.r.t. y_true.
        """
        labels = y_true[:, :, :-1]
        # -1 for ignore, 0 for background, 1 for object
        anchor_state = y_true[:, :, -1]
        classification = y_pred

        # filter out "ignore" anchors
        indices = torch.where(anchor_state != -1)
        labels = labels[indices[0], indices[1]]
        classification = classification[indices[0], indices[1]]

        # compute the focal loss
        alpha_factor = torch.ones_like(labels) * alpha
        alpha_factor = torch.where(labels == 1, alpha_factor, 1 - alpha_factor)
        # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
        focal_weight = torch.where(labels == 1, 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma
        
        # Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
        # Using PyTorch's binary_cross_entropy for numerical stability
        cls_loss = torch.nn.functional.binary_cross_entropy(classification, labels, reduction='none')
        cls_loss = focal_weight * cls_loss

        # compute the normalizer: the number of positive anchors
        normalizer = torch.sum(anchor_state == 1).float()
        normalizer = torch.clamp(normalizer, min=1.0)

        return torch.sum(cls_loss) / normalizer

    return _focal


def smooth_l1(sigma=3.0):
    """
    Create a smooth L1 loss functor for PyTorch.
    
    Args:
        sigma: This argument defines the point where the loss changes from L2 to L1.
    
    Returns:
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.
        
        Args:
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).
            
        Returns:
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices = torch.where(anchor_state == 1)
        if len(indices[0]) == 0:
            # No positive anchors, return 0 loss
            return torch.tensor(0.0, device=y_true.device)
            
        regression = regression[indices[0], indices[1]]
        regression_target = regression_target[indices[0], indices[1]]

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = torch.abs(regression_diff)
        regression_loss = torch.where(
            regression_diff < 1.0 / sigma_squared,
            0.5 * sigma_squared * torch.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = torch.clamp(torch.tensor(indices[0].size(0), dtype=torch.float32, device=y_true.device), min=1.0)
        
        return torch.sum(regression_loss) / normalizer

    return _smooth_l1


def transformation_loss(model_3d_points_np, num_rotation_parameter):
    """
    Create a transformation loss functor as described in https://arxiv.org/abs/2011.04307
    Args:
        model_3d_points_np: numpy array containing the 3D model points of all classes for calculating the transformed point distances.
                           The shape is (num_classes, num_points, 3)
        num_rotation_parameter: The number of rotation parameters, usually 3 for axis angle representation
    Returns:
        A functor for computing the transformation loss given target data and predicted data.
    """
    # Convert numpy array to PyTorch tensor
    model_3d_points = torch.tensor(model_3d_points_np, dtype=torch.float32)
    
    def _transformation_loss(y_true, y_pred):
        """ Compute the transformation loss of y_pred w.r.t. y_true using the model_3d_points tensor.
        Args:
            y_true: Tensor from the generator of shape (B, N, num_rotation_parameter + num_translation_parameter + is_symmetric_flag + class_label + anchor_state).
                   num_rotation_parameter is 3 for axis angle representation and num_translation parameter is also 3
                   is_symmetric_flag is a Boolean indicating if the GT object is symmetric or not, used to calculate the correct loss
                   class_label is the class of the GT object, used to take the correct 3D model points from the model_3d_points tensor for the transformation
                   The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, num_rotation_parameter + num_translation_parameter).
        Returns:
            The transformation loss of y_pred w.r.t. y_true.
        """
        # Make sure model_3d_points is on the same device as the input tensors
        device = y_true.device
        model_3d_points_device = model_3d_points.to(device)
        num_points = model_3d_points_device.shape[1]
        
        # Separate target and state
        regression_rotation = y_pred[:, :, :num_rotation_parameter]
        regression_translation = y_pred[:, :, num_rotation_parameter:]
        regression_target_rotation = y_true[:, :, :num_rotation_parameter]
        regression_target_translation = y_true[:, :, num_rotation_parameter:-3]
        is_symmetric = y_true[:, :, -3]
        class_indices = y_true[:, :, -2]
        anchor_state = torch.round(y_true[:, :, -1]).to(torch.int32)
        
        # Filter out "ignore" anchors
        indices = torch.where(anchor_state == 1)
        if len(indices[0]) == 0:
            # No positive anchors, return 0 loss
            return torch.tensor(0.0, device=device)
            
        regression_rotation = regression_rotation[indices[0], indices[1]] * math.pi
        regression_translation = regression_translation[indices[0], indices[1]]
        
        regression_target_rotation = regression_target_rotation[indices[0], indices[1]] * math.pi
        regression_target_translation = regression_target_translation[indices[0], indices[1]]
        is_symmetric = is_symmetric[indices[0], indices[1]]
        is_symmetric = torch.round(is_symmetric).to(torch.int32)
        class_indices = class_indices[indices[0], indices[1]]
        class_indices = torch.round(class_indices).to(torch.int32)
        
        # Separate axis from angle
        axis_pred, angle_pred = separate_axis_from_angle(regression_rotation)
        axis_target, angle_target = separate_axis_from_angle(regression_target_rotation)
        
        # Rotate the 3D model points with target and predicted rotations
        # Select model points according to the class indices
        selected_model_points = torch.index_select(model_3d_points_device, 0, class_indices.long())
        
        # Expand dims of the rotation tensors to rotate all points via broadcasting
        axis_pred = axis_pred.unsqueeze(1)
        angle_pred = angle_pred.unsqueeze(1)
        axis_target = axis_target.unsqueeze(1)
        angle_target = angle_target.unsqueeze(1)
        
        # Also expand dims of the translation tensors
        regression_translation = regression_translation.unsqueeze(1)
        regression_target_translation = regression_target_translation.unsqueeze(1)
        
        # Transform points
        transformed_points_pred = rotate(selected_model_points, axis_pred, angle_pred) + regression_translation
        transformed_points_target = rotate(selected_model_points, axis_target, angle_target) + regression_target_translation
        
        # Distinguish between symmetric and asymmetric objects
        sym_indices = torch.where(is_symmetric == 1)[0]
        asym_indices = torch.where(is_symmetric != 1)[0]
        
        distances = []
        
        # Handle symmetric objects
        if len(sym_indices) > 0:
            sym_points_pred = transformed_points_pred[sym_indices].reshape(-1, num_points, 3)
            sym_points_target = transformed_points_target[sym_indices].reshape(-1, num_points, 3)
            sym_distances = calc_sym_distances(sym_points_pred, sym_points_target)
            distances.append(sym_distances)
        
        # Handle asymmetric objects
        if len(asym_indices) > 0:
            asym_points_pred = transformed_points_pred[asym_indices].reshape(-1, num_points, 3)
            asym_points_target = transformed_points_target[asym_indices].reshape(-1, num_points, 3)
            asym_distances = calc_asym_distances(asym_points_pred, asym_points_target)
            distances.append(asym_distances)
        
        # Combine distances
        if len(distances) > 0:
            all_distances = torch.cat(distances)
            loss = torch.mean(all_distances)
            # In case of NaN (no valid instances), replace with zero
            loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
            return loss
        else:
            # No valid instances
            return torch.tensor(0.0, device=device)
        
    return _transformation_loss

def separate_axis_from_angle(axis_angle_tensor):
    """ Separates the compact 3-dimensional axis_angle representation into rotation axis and angle
    Args:
        axis_angle_tensor: tensor with a shape of 3 in the last dimension.
    Returns:
        axis: Tensor of the same shape as the input axis_angle_tensor but containing only the rotation axis
        angle: Tensor with the last dimension as 1 containing the rotation angle
    """
    squared = torch.square(axis_angle_tensor)
    summed = torch.sum(squared, dim=-1)
    angle = torch.sqrt(summed).unsqueeze(-1)
    
    # Handle division by zero
    axis = torch.zeros_like(axis_angle_tensor)
    non_zero_mask = angle > 0
    if torch.any(non_zero_mask):
        axis = torch.where(
            angle > 0.0,
            axis_angle_tensor / angle,
            torch.zeros_like(axis_angle_tensor)
        )
    
    return axis, angle

def calc_sym_distances(sym_points_pred, sym_points_target):
    """ Calculates the average minimum point distance for symmetric objects
    Args:
        sym_points_pred: Tensor of shape (num_objects, num_3D_points, 3)
        sym_points_target: Tensor of shape (num_objects, num_3D_points, 3)
    Returns:
        Tensor of shape (num_objects) containing the average minimum point distance
    """
    sym_points_pred = sym_points_pred.unsqueeze(2)  # [num_obj, num_points, 1, 3]
    sym_points_target = sym_points_target.unsqueeze(1)  # [num_obj, 1, num_points, 3]
    
    # Calculate distances between all point pairs
    diff = sym_points_pred - sym_points_target  # [num_obj, num_points, num_points, 3]
    distances = torch.norm(diff, dim=-1)  # [num_obj, num_points, num_points]
    
    # Find minimum distance for each predicted point
    min_distances = torch.min(distances, dim=-1)[0]  # [num_obj, num_points]
    
    # Average over all points
    return torch.mean(min_distances, dim=-1)  # [num_obj]

def calc_asym_distances(asym_points_pred, asym_points_target):
    """ Calculates the average pairwise point distance for asymmetric objects
    Args:
        asym_points_pred: Tensor of shape (num_objects, num_3D_points, 3)
        asym_points_target: Tensor of shape (num_objects, num_3D_points, 3)
    Returns:
        Tensor of shape (num_objects) containing the average point distance
    """
    distances = torch.norm(asym_points_pred - asym_points_target, dim=-1)  # [num_obj, num_points]
    return torch.mean(distances, dim=-1)  # [num_obj]

def cross(vector1, vector2):
    """Computes the cross product between two 3D vectors.
    Args:
        vector1: Tensor of shape [..., 3]
        vector2: Tensor of shape [..., 3]
    Returns:
        Tensor of shape [..., 3] representing the cross product
    """
    v1_x, v1_y, v1_z = vector1[..., 0], vector1[..., 1], vector1[..., 2]
    v2_x, v2_y, v2_z = vector2[..., 0], vector2[..., 1], vector2[..., 2]
    
    n_x = v1_y * v2_z - v1_z * v2_y
    n_y = v1_z * v2_x - v1_x * v2_z
    n_z = v1_x * v2_y - v1_y * v2_x
    
    return torch.stack((n_x, n_y, n_z), dim=-1)

def dot(vector1, vector2, keepdims=True):
    """Computes the dot product between two vectors along the last dimension.
    Args:
        vector1: Tensor of shape [..., D]
        vector2: Tensor of shape [..., D]
        keepdims: If True, retains the last dimension with size 1
    Returns:
        Tensor of shape [..., 1] if keepdims=True, otherwise [...] 
    """
    result = torch.sum(vector1 * vector2, dim=-1)
    if keepdims:
        result = result.unsqueeze(-1)
    return result

def rotate(point, axis, angle):
    """Rotates 3D points using Rodrigues' rotation formula.
    Args:
        point: Tensor of shape [..., 3] representing 3D points
        axis: Tensor of shape [..., 3] representing normalized rotation axes
        angle: Tensor of shape [..., 1] representing rotation angles
    Returns:
        Tensor of shape [..., 3] representing rotated points
    """
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    # Calculate axis_dot_point
    axis_dot_point = dot(axis, point)
    
    # Apply Rodrigues' formula
    rotated = (point * cos_angle + 
              cross(axis, point) * sin_angle + 
              axis * axis_dot_point * (1.0 - cos_angle))
    
    return rotated