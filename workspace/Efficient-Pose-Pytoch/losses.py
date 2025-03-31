import torch
import torch.nn as nn
import math

def focal(alpha=0.25, gamma=1.5):
    """
    Create a functor for computing the focal loss.

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
        labels = torch.squeeze(y_true[:, :, :-1], dim =-1)
        # -1 for ignore, 0 for background, 1 for object
        anchor_state = y_true[:, :, -1]
        classification = torch.squeeze(y_pred, dim = -1)
        
        # filter out "ignore" anchors
        mask  = ~anchor_state.eq(-1)
        labels = torch.masked_select(labels,mask)
        classification = torch.masked_select(classification,mask)

        # compute the focal loss
        alpha_factor = torch.ones_like(labels) * alpha
        alpha_factor = torch.where(labels.eq(1), alpha_factor, 1 - alpha_factor)
        # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
        focal_weight = torch.where(labels.eq(1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma
        cls_loss = focal_weight * nn.BCEWithLogitsLoss()(labels, classification)
        
        return torch.mean(cls_loss)

    return _focal

def smooth_l1(sigma=3.0):
    """
    Create a smooth L1 loss functor.
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
        mask  = torch.unsqueeze(anchor_state.eq(1), dim = -1).repeat(1,1,4)
        regression = torch.masked_select(regression,mask)
        regression_target = torch.masked_select(regression_target,mask)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = torch.abs(regression_diff)
        regression_loss = torch.where(
            torch.lt(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * (regression_diff * regression_diff),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        return torch.mean(regression_loss)

    return _smooth_l1
    

def transformation_loss(model_3d_points_np, num_rotation_parameter = 3):
    """
    Create a transformation loss functor as described in https://arxiv.org/abs/2011.04307
    Args:
        model_3d_points_np: numpy array containing the 3D model points of all classes for calculating the transformed point distances.
                            The shape is (num_classes, num_points, 3)
        num_rotation_parameter: The number of rotation parameters, usually 3 for axis angle representation
    Returns:
        A functor for computing the transformation loss given target data and predicted data.
    """
    model_3d_points = model_3d_points_np
    num_points = model_3d_points.shape[1]
    
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
        # separate target and state
        regression_rotation = y_pred[:, :, :num_rotation_parameter]
        regression_translation = y_pred[:, :, num_rotation_parameter:]
        regression_target_rotation = y_true[:, :, :num_rotation_parameter]
        regression_target_translation = y_true[:, :, num_rotation_parameter:-3]
        is_symmetric = y_true[:, :, -3]
        class_indices = y_true[:, :, -2]
        anchor_state      = torch.round(y_true[:, :, -1]).int()
        
        # filter out "ignore" anchors
        mask = anchor_state.eq(1)
        rotation_mask = torch.unsqueeze(mask, dim=-1).repeat(1,1,num_rotation_parameter)
        translation_mask = torch.unsqueeze(mask, dim=-1).repeat(1,1,3)
        regression_rotation = torch.masked_select(regression_rotation,rotation_mask).reshape(-1,num_rotation_parameter) * math.pi
        regression_translation = torch.masked_select(regression_translation,translation_mask).reshape(-1,3)
        
        regression_target_rotation = torch.masked_select(regression_target_rotation,rotation_mask).reshape(-1,num_rotation_parameter) * math.pi
        regression_target_translation = torch.masked_select(regression_target_translation,translation_mask).reshape(-1,3)
        is_symmetric = torch.masked_select(is_symmetric,mask)
        is_symmetric = torch.round(is_symmetric).int()
        class_indices = torch.masked_select(class_indices,mask)
        class_indices = torch.round(class_indices).int()
        
        axis_pred, angle_pred = separate_axis_from_angle(regression_rotation)
        axis_target, angle_target = separate_axis_from_angle(regression_target_rotation)
        
        #rotate the 3d model points with target and predicted rotations        
        #select model points according to the class indices
        selected_model_points = torch.index_select(model_3d_points, 0, class_indices)
        #expand dims of the rotation tensors to rotate all points along the dimension via broadcasting
        axis_pred = torch.unsqueeze(axis_pred, dim = 1)
        angle_pred = torch.unsqueeze(angle_pred, dim = 1)
        axis_target = torch.unsqueeze(axis_target, dim = 1)
        angle_target = torch.unsqueeze(angle_target, dim = 1)
        
        #also expand dims of the translation tensors to translate all points along the dimension via broadcasting
        regression_translation = torch.unsqueeze(regression_translation, dim = 1)
        regression_target_translation = torch.unsqueeze(regression_target_translation, dim = 1)
        
        transformed_points_pred = rotate(selected_model_points, axis_pred, angle_pred) + regression_translation
        transformed_points_target = rotate(selected_model_points, axis_target, angle_target) + regression_target_translation
        
        #distinct between symmetric and asymmetric objects
        sym_mask = torch.unsqueeze(torch.unsqueeze(is_symmetric.eq(1),dim=-1),dim=-1).repeat(1,num_points,3)
        asym_mask = ~torch.unsqueeze(torch.unsqueeze(is_symmetric.eq(1),dim=-1),dim=-1).repeat(1,num_points,3)

        sym_points_pred = torch.reshape(torch.masked_select(transformed_points_pred, sym_mask), (-1, num_points, 3))
        asym_points_pred = torch.reshape(torch.masked_select(transformed_points_pred, asym_mask), (-1, num_points, 3))
        
        sym_points_target = torch.reshape(torch.masked_select(transformed_points_target, sym_mask), (-1, num_points, 3))
        asym_points_target = torch.reshape(torch.masked_select(transformed_points_target, asym_mask), (-1, num_points, 3))
        
        # # compute transformed point distances
        sym_distances = calc_sym_distances(sym_points_pred, sym_points_target)
        asym_distances = calc_asym_distances(asym_points_pred, asym_points_target)

        distances = torch.cat([sym_distances, asym_distances], dim = 0)
        
        loss = torch.mean(distances)
        #in case of no annotations the loss is nan => replace with zero
        loss = torch.nan_to_num(loss, nan=0.0)

        return loss
        
    return _transformation_loss

def separate_axis_from_angle(axis_angle_tensor):
    """ Separates the compact 3-dimensional axis_angle representation in the rotation axis and a rotation angle
        Args:
            axis_angle_tensor: tensor with a shape of 3 in the last dimension.
        Returns:
            axis: Tensor of the same shape as the input axis_angle_tensor but containing only the rotation axis and not the angle anymore
            angle: Tensor of the same shape as the input axis_angle_tensor except the last dimension is 1 and contains the rotation angle
        """
    squared = torch.square(axis_angle_tensor)
    summed = torch.sum(squared, dim = -1)
    angle = torch.unsqueeze(torch.sqrt(summed), axis = -1)
    
    axis = torch.div(axis_angle_tensor, angle)
    
    return axis, angle

def calc_sym_distances(sym_points_pred, sym_points_target):
    """ Calculates the average minimum point distance for symmetric objects
        Args:
            sym_points_pred: Tensor of shape (num_objects, num_3D_points, 3) containing all 3D model points transformed with the model's prediction
            sym_points_target: Tensor of shape (num_objects, num_3D_points, 3) containing all 3D model points transformed with the ground truth 6D pose
        Returns:
            Tensor of shape (num_objects) containing the average minimum point distance between both transformed 3D models
        """

    sym_points_pred = torch.unsqueeze(sym_points_pred, dim = 2)
    sym_points_target = torch.unsqueeze(sym_points_target, dim = 1)
    distances = torch.min(torch.norm(sym_points_pred - sym_points_target, dim = -1), dim = -1)

    return torch.mean(distances.values, dim = -1)


def calc_asym_distances(asym_points_pred, asym_points_target):
    """ Calculates the average pairwise point distance for asymmetric objects
        Args:
            asym_points_pred: Tensor of shape (num_objects, num_3D_points, 3) containing all 3D model points transformed with the model's prediction
            asym_points_target: Tensor of shape (num_objects, num_3D_points, 3) containing all 3D model points transformed with the ground truth 6D pose
        Returns:
            Tensor of shape (num_objects) containing the average point distance between both transformed 3D models
        """
    distances = torch.norm(asym_points_pred - asym_points_target, dim = -1)
    
    return torch.mean(distances, dim = -1)

#copied and adapted the following functions from tensorflow graphics source because they did not work with unknown shape
#https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/vector.py
def cross(vector1, vector2, name=None):
    """Computes the cross product between two tensors along an axis.
    Note:
        In the following, A1 to An are optional batch dimensions, which should be
        broadcast compatible.
    Args:
        vector1: A tensor of shape `[A1, ..., Ai = 3, ..., An]`, where the dimension
        i = axis represents a 3d vector.
        vector2: A tensor of shape `[A1, ..., Ai = 3, ..., An]`, where the dimension
        i = axis represents a 3d vector.
        axis: The dimension along which to compute the cross product.
        name: A name for this op which defaults to "vector_cross".
    Returns:
        A tensor of shape `[A1, ..., Ai = 3, ..., An]`, where the dimension i = axis
        represents the result of the cross product.
    """

    vector1_x = vector1[:, :, 0]
    vector1_y = vector1[:, :, 1]
    vector1_z = vector1[:, :, 2]
    vector2_x = vector2[:, :, 0]
    vector2_y = vector2[:, :, 1]
    vector2_z = vector2[:, :, 2]
    n_x = vector1_y * vector2_z - vector1_z * vector2_y
    n_y = vector1_z * vector2_x - vector1_x * vector2_z
    n_z = vector1_x * vector2_y - vector1_y * vector2_x
    return torch.cat((n_x, n_y, n_z), dim = -1)

#copied from https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/axis_angle.py
def rotate(point, axis, angle, name=None):
    r"""Rotates a 3d point using an axis-angle by applying the Rodrigues' formula.
    Rotates a vector $$\mathbf{v} \in {\mathbb{R}^3}$$ into a vector
    $$\mathbf{v}' \in {\mathbb{R}^3}$$ using the Rodrigues' rotation formula:
    $$\mathbf{v}'=\mathbf{v}\cos(\theta)+(\mathbf{a}\times\mathbf{v})\sin(\theta)
    +\mathbf{a}(\mathbf{a}\cdot\mathbf{v})(1-\cos(\theta)).$$
    Note:
        In the following, A1 to An are optional batch dimensions.
    Args:
        point: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
        represents a 3d point to rotate.
        axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
        represents a normalized axis.
        angle: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
        represents an angle.
        name: A name for this op that defaults to "axis_angle_rotate".
    Returns:
        A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
        a 3d point.
    Raises:
        ValueError: If `point`, `axis`, or `angle` are of different shape or if
        their respective shape is not supported.
    """

    cos_angle = torch.cos(angle)
    axis = axis.repeat(1,point.shape[1],1)
    axis_dot_point = axis * point
    return point * cos_angle + torch.cross(
        axis, point) * torch.sin(angle) + axis * axis_dot_point * (1.0 - cos_angle)   
