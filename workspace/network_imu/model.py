import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np
from anchor import anchors_for_shape
from layer import FilterDetections

MOMENTUM = 0.997
EPSILON = 1e-4

def get_scaled_parameters(phi):
    """
    Get all needed scaled parameters to build EfficientPose using PyTorch backbones.

    Args:
        phi (int): EfficientPose scaling hyperparameter (0 to 6)

    Returns:
       dict: Dictionary containing the scaled parameters.
    """
    # Scalable parameters
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    bifpn_widths = (64, 88, 112, 160, 224, 288, 384)
    bifpn_depths = (3, 4, 5, 6, 7, 7, 8)
    subnet_depths = (3, 3, 3, 4, 4, 4, 5)
    subnet_iteration_steps = (1, 1, 1, 2, 2, 2, 3)
    num_groups_gn = (4, 4, 7, 10, 14, 18, 24)  # Try to get 16 channels per group

    # Map phi values to PyTorch EfficientNet backbones using torchvision
    backbones = (
        models.efficientnet_b0,
        models.efficientnet_b1,
        models.efficientnet_b2,
        models.efficientnet_b3,
        models.efficientnet_b4,
        models.efficientnet_b5,
        models.efficientnet_b6
    )

    parameters = {
        "input_size": image_sizes[phi],
        "bifpn_width": bifpn_widths[phi],
        "bifpn_depth": bifpn_depths[phi],
        "subnet_depth": subnet_depths[phi],
        "subnet_num_iteration_steps": subnet_iteration_steps[phi],
        "num_groups_gn": num_groups_gn[phi],
        "backbone_class": backbones[phi]  # callable constructor, e.g., models.efficientnet_b0
    }

    return parameters

##########################################
# wBiFPNAdd: Weighted addition of features
##########################################

class wBiFPNAdd(nn.Module):
    def __init__(self, num_inputs, eps=1e-4):
        super().__init__()
        self.eps = eps
        # Learnable weights for each input
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)
    
    def forward(self, inputs):
        # Ensure non-negative weights and normalize
        w = F.relu(self.weights)
        norm = torch.sum(w) + self.eps
        norm_w = w / norm
        out = 0
        for i, inp in enumerate(inputs):
            out = out + norm_w[i] * inp
        return out

#############################################
# Custom BatchNormalization with freeze flag
#############################################
class BatchNormalization(nn.BatchNorm2d):
    """
    PyTorch version similar in functionality to Keras's BatchNormalization,
    with an option to freeze parameters (i.e. use running statistics always).
    """
    def __init__(self, num_features, freeze=False, **kwargs):
        super().__init__(num_features, **kwargs)
        self.freeze = freeze

    def forward(self, input):
        if self.freeze:
            # Force evaluation mode: do not update running stats.
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps
            )
        else:
            return super().forward(input)
        

##########################################
# SeparableConvBlock
##########################################
class SeparableConvBlock(nn.Module):
    """
    Builds a block with a depthwise separable convolution layer followed by a batch norm layer.
    """
    def __init__(self, num_channels, kernel_size, strides, name=None, freeze_bn=False, momentum=0.997, eps=1e-4):
        super().__init__()
        # Use padding to mimic 'same' padding (assumes kernel_size is odd)
        padding = kernel_size // 2

        # Depthwise convolution: groups == num_channels makes it depthwise.
        self.depthwise = nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size,
                                   stride=strides, padding=padding, groups=num_channels, bias=True)
        # Pointwise convolution: 1x1 conv to combine features.
        self.pointwise = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False)
        # Batch normalization (using your custom layer that supports freezing)
        self.bn = BatchNormalization(num_channels, freeze=freeze_bn, momentum=momentum, eps=eps)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x
    

        
#############################################
# Prepare backbone feature maps for BiFPN
#############################################
def prepare_feature_maps_for_BiFPN(C3, C4, C5, num_channels, freeze_bn):
    """
    Prepares the backbone feature maps for the first BiFPN layer.
    
    Args:
        C3, C4, C5 (torch.Tensor): Backbone feature maps.
        num_channels (int): Number of output channels to use in the BiFPN.
        freeze_bn (bool): If True, the BatchNormalization layers will be frozen.
    
    Returns:
        Tuple[torch.Tensor]: Processed feature maps (P3_in, P4_in_1, P4_in_2,
                                P5_in_1, P5_in_2, P6_in, P7_in)
    """
    # Process C3: apply 1x1 conv and BN.
    conv_p3 = nn.Conv2d(in_channels=C3.shape[1], out_channels=num_channels, kernel_size=1, bias=False)
    P3_in = conv_p3(C3)
    bn_p3 = BatchNormalization(num_features=num_channels, freeze=freeze_bn, momentum=MOMENTUM, eps=EPSILON)
    P3_in = bn_p3(P3_in)
    
    # Process C4: two branches.
    conv_p4_1 = nn.Conv2d(in_channels=C4.shape[1], out_channels=num_channels, kernel_size=1, bias=False)
    P4_in_1 = conv_p4_1(C4)
    bn_p4_1 = BatchNormalization(num_features=num_channels, freeze=freeze_bn, momentum=MOMENTUM, eps=EPSILON)
    P4_in_1 = bn_p4_1(P4_in_1)
    
    conv_p4_2 = nn.Conv2d(in_channels=C4.shape[1], out_channels=num_channels, kernel_size=1, bias=False)
    P4_in_2 = conv_p4_2(C4)
    bn_p4_2 = BatchNormalization(num_features=num_channels, freeze=freeze_bn, momentum=MOMENTUM, eps=EPSILON)
    P4_in_2 = bn_p4_2(P4_in_2)
    
    # Process C5: two branches.
    conv_p5_1 = nn.Conv2d(in_channels=C5.shape[1], out_channels=num_channels, kernel_size=1, bias=False)
    P5_in_1 = conv_p5_1(C5)
    bn_p5_1 = BatchNormalization(num_features=num_channels, freeze=freeze_bn, momentum=MOMENTUM, eps=EPSILON)
    P5_in_1 = bn_p5_1(P5_in_1)
    
    conv_p5_2 = nn.Conv2d(in_channels=C5.shape[1], out_channels=num_channels, kernel_size=1, bias=False)
    P5_in_2 = conv_p5_2(C5)
    bn_p5_2 = BatchNormalization(num_features=num_channels, freeze=freeze_bn, momentum=MOMENTUM, eps=EPSILON)
    P5_in_2 = bn_p5_2(P5_in_2)
    
    # Process P6: from C5.
    conv_p6 = nn.Conv2d(in_channels=C5.shape[1], out_channels=num_channels, kernel_size=1, bias=False)
    P6_in = conv_p6(C5)
    bn_p6 = BatchNormalization(num_features=num_channels, freeze=freeze_bn, momentum=MOMENTUM, eps=EPSILON)
    P6_in = bn_p6(P6_in)
    # Using a max-pooling layer to downsample. Here, kernel_size=3, stride=2, and padding=1 mimic 'same' padding.
    maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    P6_in = maxpool(P6_in)
    
    # Process P7: further downsample P6_in.
    P7_in = maxpool(P6_in)
    
    return P3_in, P4_in_1, P4_in_2, P5_in_1, P5_in_2, P6_in, P7_in


##########################################
# Single merge step used in both pathways
##########################################

def single_BiFPN_merge_step(feature_map_other_level, feature_maps_current_level, upsampling, num_channels,
                           idx_BiFPN_layer, node_idx, op_idx, freeze_bn=False, momentum=0.997, eps=1e-4):
    """
    Merges two feature maps of different levels in the BiFPN.
    
    Args:
        feature_map_other_level (torch.Tensor): Feature map from a different level.
        feature_maps_current_level (list of torch.Tensor): List of current-level feature map(s).
        upsampling (bool): Whether to upsample feature_map_other_level.
        num_channels (int): Number of channels used in the BiFPN.
        idx_BiFPN_layer (int): Index of the BiFPN layer (for naming purposes).
        node_idx, op_idx (int): Node and op indices (for naming, not used here).
    
    Returns:
        torch.Tensor: The merged feature map.
    """
    if upsampling:
        # Upsample to match spatial size of the current level.
        target_size = feature_maps_current_level[0].shape[-2:]
        feature_map_resampled = F.interpolate(feature_map_other_level, size=target_size, mode='nearest')
    else:
        # Downsample using max pooling.
        feature_map_resampled = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(feature_map_other_level)
    
    # Fuse the features using a weighted addition.
    num_inputs = len(feature_maps_current_level) + 1
    add_layer = wBiFPNAdd(num_inputs)
    merged_feature_map = add_layer(feature_maps_current_level + [feature_map_resampled])
    
    # Apply Swish activation.
    merged_feature_map = F.silu(merged_feature_map)
    
    # Apply a separable convolution block.
    # Remove 'padding' parameter since it's calculated in SeparableConvBlock
    sep_conv = SeparableConvBlock(
        num_channels=num_channels, 
        kernel_size=3, 
        strides=1,  # Changed from stride to strides to match your class
        name=f'fpn_cells/cell_{idx_BiFPN_layer}/fnode{node_idx}/op_after_combine{op_idx}',  # Added name parameter
        freeze_bn=freeze_bn, 
        momentum=momentum, 
        eps=eps
    )
    merged_feature_map = sep_conv(merged_feature_map)
    
    return merged_feature_map

##########################################
# Top-down pathway
##########################################

def top_down_pathway_BiFPN(input_feature_maps_top_down, num_channels, idx_BiFPN_layer, freeze_bn=False, momentum=0.997, eps=1e-4):
    """
    Computes the top-down-pathway in a single BiFPN layer
    
    Args:
        input_feature_maps_top_down: List containing the input feature maps of the BiFPN layer [P7, P6, P5, P4, P3]
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer
        freeze_bn: Whether to freeze batch normalization layers
        momentum: Momentum for batch normalization
        eps: Epsilon for batch normalization
    
    Returns:
       List with the output feature maps of the top-down-pathway [P7_out, P6_td, P5_td, P4_td, P3_out]
    """
    # P7 is passed through unchanged
    feature_map_P7 = input_feature_maps_top_down[0]
    output_top_down_feature_maps = [feature_map_P7]
    
    # Process levels from top to bottom (P7->P6->P5->P4->P3)
    for level in range(1, 5):
        merged_feature_map = single_BiFPN_merge_step(
            feature_map_other_level=output_top_down_feature_maps[-1],
            feature_maps_current_level=[input_feature_maps_top_down[level]],
            upsampling=True,
            num_channels=num_channels,
            idx_BiFPN_layer=idx_BiFPN_layer,
            node_idx=level - 1,
            op_idx=4 + level,
            freeze_bn=freeze_bn,
            momentum=momentum,
            eps=eps
        )
        
        output_top_down_feature_maps.append(merged_feature_map)
    
    return output_top_down_feature_maps

##########################################
# Bottom-up pathway
##########################################

def bottom_up_pathway_BiFPN(input_feature_maps_bottom_up, num_channels, idx_BiFPN_layer, freeze_bn=False, momentum=0.997, eps=1e-4):
    """
    Computes the bottom-up-pathway in a single BiFPN layer
    
    Args:
        input_feature_maps_bottom_up: List of lists containing feature maps for each level
                                     [[P3_out], [P4_in, P4_td], [P5_in, P5_td], [P6_in, P6_td], [P7_in]]
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer
        freeze_bn: Whether to freeze batch normalization layers
        momentum: Momentum for batch normalization
        eps: Epsilon for batch normalization
    
    Returns:
       List with the output feature maps of the bottom-up-pathway [P3_out, P4_out, P5_out, P6_out, P7_out]
    """
    # P3 is the first feature in the bottom-up pathway
    feature_map_P3 = input_feature_maps_bottom_up[0][0]
    output_bottom_up_feature_maps = [feature_map_P3]
    
    # Process levels from bottom to top (P3->P4->P5->P6->P7)
    for level in range(1, 5):
        merged_feature_map = single_BiFPN_merge_step(
            feature_map_other_level=output_bottom_up_feature_maps[-1],
            feature_maps_current_level=input_feature_maps_bottom_up[level],
            upsampling=False,
            num_channels=num_channels,
            idx_BiFPN_layer=idx_BiFPN_layer,
            node_idx=3 + level,
            op_idx=8 + level,
            freeze_bn=freeze_bn,
            momentum=momentum,
            eps=eps
        )
        
        output_bottom_up_feature_maps.append(merged_feature_map)
    
    return output_bottom_up_feature_maps

##########################################
# Build BiFPN layer
##########################################

def build_BiFPN_layer(features, num_channels, idx_BiFPN_layer, freeze_bn=False, momentum=0.997, eps=1e-4):
    """
    Builds a single layer of the bidirectional feature pyramid
    
    Args:
        features: List containing the feature maps of the previous BiFPN layer (P3, P4, P5, P6, P7) 
                 or the EfficientNet backbone feature maps of the different levels (C1, C2, C3, C4, C5)
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
        freeze_bn: Boolean indicating if the batch norm layers should be frozen during training
        momentum: Momentum parameter for batch normalization
        eps: Epsilon parameter for batch normalization
    
    Returns:
        Tuple of BiFPN feature maps (P3, P4, P5, P6, P7)
    """
    if idx_BiFPN_layer == 0:
        # For the first BiFPN layer, we use the backbone features
        # In PyTorch, we can't use the star unpacking like in TensorFlow, 
        # so we explicitly select C3, C4, C5
        C3, C4, C5 = features[2], features[3], features[4]
        
        # Prepare feature maps for the first BiFPN layer
        P3_in, P4_in_1, P4_in_2, P5_in_1, P5_in_2, P6_in, P7_in = prepare_feature_maps_for_BiFPN(
            C3, C4, C5, num_channels, freeze_bn
        )
    else:
        # For subsequent layers, use outputs from the previous BiFPN layer
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        
    # Top-down pathway
    input_feature_maps_top_down = [
        P7_in,
        P6_in,
        P5_in_1 if idx_BiFPN_layer == 0 else P5_in,
        P4_in_1 if idx_BiFPN_layer == 0 else P4_in,
        P3_in
    ]
    
    P7_in, P6_td, P5_td, P4_td, P3_out = top_down_pathway_BiFPN(
        input_feature_maps_top_down, 
        num_channels, 
        idx_BiFPN_layer,
        freeze_bn,
        momentum,
        eps
    )
    
    # Bottom-up pathway
    input_feature_maps_bottom_up = [
        [P3_out],
        [P4_in_2 if idx_BiFPN_layer == 0 else P4_in, P4_td],
        [P5_in_2 if idx_BiFPN_layer == 0 else P5_in, P5_td],
        [P6_in, P6_td],
        [P7_in]
    ]
    
    P3_out, P4_out, P5_out, P6_out, P7_out = bottom_up_pathway_BiFPN(
        input_feature_maps_bottom_up, 
        num_channels, 
        idx_BiFPN_layer,
        freeze_bn,
        momentum,
        eps
    )
    
    # Note: Following the original implementation, we return top-down feature maps for P4, P5, P6
    # This might be a bug in the original implementation as noted in the comment
    return P3_out, P4_td, P5_td, P6_td, P7_out

##########################################
# Build BiFPN
##########################################

def build_BiFPN(backbone_feature_maps, bifpn_depth, bifpn_width, freeze_bn, momentum=0.997, eps=1e-4):
    """
    Building the bidirectional feature pyramid as described in https://arxiv.org/abs/1911.09070
    
    Args:
        backbone_feature_maps: List containing the EfficientNet backbone feature maps of the different levels (C1, C2, C3, C4, C5)
        bifpn_depth: Number of BiFPN layers
        bifpn_width: Number of channels used in the BiFPN
        freeze_bn: Boolean indicating if the batch norm layers should be frozen during training
        momentum: Momentum parameter for batch normalization
        eps: Epsilon parameter for batch normalization
    
    Returns:
        fpn_feature_maps: Tuple of BiFPN feature maps of the different levels (P3, P4, P5, P6, P7)
    """
    fpn_feature_maps = backbone_feature_maps
    
    # Stack BiFPN layers
    for i in range(bifpn_depth):
        fpn_feature_maps = build_BiFPN_layer(
            fpn_feature_maps, 
            num_channels=bifpn_width, 
            idx_BiFPN_layer=i, 
            freeze_bn=freeze_bn,
            momentum=momentum,
            eps=eps
        )
        
    return fpn_feature_maps


##########################################
# BoxNet
##########################################

class BoxNet(nn.Module):
    def __init__(self, width, depth, num_anchors=9, freeze_bn=False, name="box_net"):
        super(BoxNet, self).__init__()
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = 4  # Bounding box coordinates (x, y, w, h)
        self.name = name

        # Create separable convolution layers
        self.convs = nn.ModuleList([
            SeparableConvBlock(num_channels=self.width, kernel_size=3, strides=1, freeze_bn=freeze_bn)
            for _ in range(self.depth)
        ])
        
        # Custom head implementation instead of using SeparableConvBlock
        # This creates a depthwise + pointwise convolution with the correct output channels
        self.head_depthwise = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.width,
            bias=True
        )
        
        self.head_pointwise = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.num_anchors * self.num_values,
            kernel_size=1,
            bias=True
        )
        
        # Batch normalization layers per level (P3-P7)
        self.bns = nn.ModuleList([
            nn.ModuleList([
                BatchNormalization(self.width, freeze=freeze_bn) for _ in range(5)  # Levels P3-P7
            ])
            for _ in range(self.depth)
        ])
        
        self.activation = nn.SiLU()  # Equivalent to TensorFlow's Swish activation
        self.level = 0

    def forward(self, feature, level):
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][level](feature)  # Select batch norm based on pyramid level
            feature = self.activation(feature)

        # Apply custom head (depthwise + pointwise convolution)
        outputs = self.head_depthwise(feature)
        outputs = self.head_pointwise(outputs)

        # Reshape output to match the expected format: (batch_size, -1, num_values)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()  # (B, H, W, num_anchors * 4)
        outputs = outputs.view(outputs.shape[0], -1, self.num_values)  # (B, num_boxes, 4)

        self.level += 1  # Track the level
        return outputs
    

##########################################
# ClassNet
##########################################

class ClassNet(nn.Module):
    def __init__(self, width, depth, num_classes=8, num_anchors=9, freeze_bn=False, name="class_net"):
        super(ClassNet, self).__init__()
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.name = name

        # Create separable convolution layers
        self.convs = nn.ModuleList([
            SeparableConvBlock(num_channels=self.width, kernel_size=3, strides=1, freeze_bn=freeze_bn)
            for _ in range(self.depth)
        ])
        
        # Prediction head - custom implementation to handle different output channels
        self.head_depthwise = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.width,
            bias=True
        )
        
        # Prior probability initialization for bias (equivalent to TensorFlow's PriorProbability)
        # Setting a negative bias to achieve ~0.01 probability after sigmoid
        bias_value = -np.log((1 - 0.01) / 0.01)
        self.head_pointwise = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.num_anchors * self.num_classes,
            kernel_size=1,
            bias=True
        )
        # Initialize bias to achieve desired prior probability
        nn.init.constant_(self.head_pointwise.bias, bias_value)
        
        # Batch normalization layers per level (P3-P7)
        self.bns = nn.ModuleList([
            nn.ModuleList([
                BatchNormalization(self.width, freeze=freeze_bn) for _ in range(5)  # Levels P3-P7
            ])
            for _ in range(self.depth)
        ])
        
        self.activation = nn.SiLU()  # Equivalent to TensorFlow's Swish activation
        self.activation_sigmoid = nn.Sigmoid()  # Final sigmoid activation
        self.level = 0

    def forward(self, feature, level):
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][level](feature)  # Select batch norm based on pyramid level
            feature = self.activation(feature)

        # Apply custom head (depthwise + pointwise convolution)
        outputs = self.head_depthwise(feature)
        outputs = self.head_pointwise(outputs)

        # Reshape output to match the expected format: (batch_size, -1, num_classes)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()  # (B, H, W, num_anchors * num_classes)
        outputs = outputs.view(outputs.shape[0], -1, self.num_classes)  # (B, num_boxes, num_classes)
        
        # Apply sigmoid activation for classification
        outputs = self.activation_sigmoid(outputs)

        self.level += 1  # Track the level
        return outputs
    
##########################################
# iteratively predict rotation and rotation subnet
##########################################
    
class IterativeRotationSubNet(nn.Module):
    def __init__(self, width, depth, num_values, num_iteration_steps, num_anchors=9, 
                 freeze_bn=False, use_group_norm=True, num_groups_gn=None, name="iterative_rotation_subnet"):
        super(IterativeRotationSubNet, self).__init__()
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = num_values
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        self.name = name
        
        # Concatenated input will have width + num_anchors*num_values channels
        # First convolution should handle this expanded input
        input_channels = self.width + self.num_anchors * self.num_values
        
        # First convolution layer handles the concatenated input
        self.first_conv_depthwise = nn.Conv2d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=input_channels,
            bias=True
        )
        
        self.first_conv_pointwise = nn.Conv2d(
            in_channels=input_channels,
            out_channels=self.width,  # Project back to width channels
            kernel_size=1,
            bias=True
        )
        
        # Remaining convolution layers (after the first one)
        self.convs = nn.ModuleList([
            SeparableConvBlock(
                num_channels=self.width, 
                kernel_size=3, 
                strides=1
            ) for _ in range(self.depth-1)  # One less because we have a separate first conv
        ])
        
        # Head layer
        self.head_depthwise = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.width,
            bias=True
        )
        
        self.head_pointwise = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.num_anchors * self.num_values,
            kernel_size=1,
            bias=True
        )
        
        # Normalization layers
        # Structure: [iteration_steps][depth][pyramid_levels]
        if self.use_group_norm:
            # First layer norm is for the output of first_conv
            self.first_norm_layers = nn.ModuleList([
                nn.ModuleList([
                    GroupNormalization(num_channels=self.width, groups=self.num_groups_gn)
                    for _ in range(5)  # 5 pyramid levels (P3-P7)
                ])
                for _ in range(self.num_iteration_steps)
            ])
            
            # Remaining norm layers
            self.norm_layers = nn.ModuleList([
                nn.ModuleList([
                    nn.ModuleList([
                        GroupNormalization(num_channels=self.width, groups=self.num_groups_gn)
                        for _ in range(5)  # 5 pyramid levels (P3-P7)
                    ])
                    for _ in range(self.depth-1)  # One less because we handle first layer separately
                ])
                for _ in range(self.num_iteration_steps)
            ])
        else:
            # First layer norm is for the output of first_conv
            self.first_norm_layers = nn.ModuleList([
                nn.ModuleList([
                    nn.BatchNorm2d(self.width, momentum=0.997, eps=1e-4)
                    for _ in range(5)  # 5 pyramid levels (P3-P7)
                ])
                for _ in range(self.num_iteration_steps)
            ])
            
            # Remaining norm layers
            self.norm_layers = nn.ModuleList([
                nn.ModuleList([
                    nn.ModuleList([
                        nn.BatchNorm2d(self.width, momentum=0.997, eps=1e-4)
                        for _ in range(5)  # 5 pyramid levels (P3-P7)
                    ])
                    for _ in range(self.depth-1)  # One less because we handle first layer separately
                ])
                for _ in range(self.num_iteration_steps)
            ])
        
        self.activation = nn.SiLU()  # Equivalent to TensorFlow's Swish
        
    def forward(self, inputs, level_py, iter_step_py):
        feature, level = inputs
        
        # First conv layer that handles the concatenated input
        feature = self.first_conv_depthwise(feature)
        feature = self.first_conv_pointwise(feature)
        feature = self.first_norm_layers[iter_step_py][level_py](feature)
        feature = self.activation(feature)
        
        # Remaining conv layers
        for i in range(self.depth-1):
            feature = self.convs[i](feature)
            feature = self.norm_layers[iter_step_py][i][level_py](feature)
            feature = self.activation(feature)
        
        # Apply head
        outputs = self.head_depthwise(feature)
        outputs = self.head_pointwise(outputs)
        
        return outputs


# RotationNet
class RotationNet(nn.Module):
    def __init__(self, width, depth, num_values, num_iteration_steps, num_anchors=9, 
                 freeze_bn=False, use_group_norm=True, num_groups_gn=None, name="rotation_net"):
        super(RotationNet, self).__init__()
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = num_values
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        self.name = name
        
        # Convolution layers
        self.convs = nn.ModuleList([
            SeparableConvBlock(
                num_channels=self.width, 
                kernel_size=3, 
                strides=1
            ) for _ in range(self.depth)
        ])
        
        # Initial rotation head
        self.initial_rotation_depthwise = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.width,
            bias=True
        )
        
        self.initial_rotation_pointwise = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.num_anchors * self.num_values,
            kernel_size=1,
            bias=True
        )
        
        # Normalization layers
        if self.use_group_norm:
            self.norm_layers = nn.ModuleList([
                nn.ModuleList([
                    GroupNormalization(num_channels=self.width, groups=self.num_groups_gn)
                    for _ in range(5)  # 5 pyramid levels (P3-P7)
                ])
                for _ in range(self.depth)
            ])
        else:
            self.norm_layers = nn.ModuleList([
                nn.ModuleList([
                    nn.BatchNorm2d(self.width, momentum=0.997, eps=1e-4)
                    for _ in range(5)  # 5 pyramid levels (P3-P7)
                ])
                for _ in range(self.depth)
            ])
        
        # Iterative subnet
        self.iterative_submodel = IterativeRotationSubNet(
            width=self.width,
            depth=self.depth - 1,
            num_values=self.num_values,
            num_iteration_steps=self.num_iteration_steps,
            num_anchors=self.num_anchors,
            freeze_bn=freeze_bn,
            use_group_norm=self.use_group_norm,
            num_groups_gn=self.num_groups_gn
        )
        
        self.activation = nn.SiLU()  # Equivalent to TensorFlow's Swish
        self.level = 0
        
    def forward(self, inputs):
        feature, level = inputs
        
        # Apply convolutional layers with normalization
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norm_layers[i][self.level](feature)
            feature = self.activation(feature)
        
        # Initial rotation prediction
        rotation = self.initial_rotation_depthwise(feature)
        rotation = self.initial_rotation_pointwise(rotation)
        
        # Iterative refinement
        for i in range(self.num_iteration_steps):
            # Concatenate feature and current rotation along channel dimension
            iterative_input = torch.cat([feature, rotation], dim=1)
            
            # Get delta rotation from iterative subnet
            delta_rotation = self.iterative_submodel([iterative_input, level], level_py=self.level, iter_step_py=i)
            
            # Update rotation
            rotation = rotation + delta_rotation
        
        # Reshape the output to [batch_size, -1, num_values]
        outputs = rotation.permute(0, 2, 3, 1).contiguous()  # (B, H, W, anchors*values)
        outputs = outputs.view(outputs.shape[0], -1, self.num_values)  # (B, num_boxes, values)
        
        self.level += 1
        return outputs
    

##########################################
# iteratively predict translation and translation subnet
##########################################
class IterativeTranslationSubNet(nn.Module):
    def __init__(self, width, depth, num_iteration_steps, num_anchors=9, 
                 freeze_bn=False, use_group_norm=True, num_groups_gn=None, name="iterative_translation_subnet"):
        super(IterativeTranslationSubNet, self).__init__()
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        self.name = name
        
        # Concatenated input will have width + num_anchors*3 channels (2 for xy, 1 for z)
        # First convolution should handle this expanded input
        input_channels = self.width + self.num_anchors * 3
        
        # First convolution layer handles the concatenated input
        self.first_conv_depthwise = nn.Conv2d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=input_channels,
            bias=True
        )
        
        self.first_conv_pointwise = nn.Conv2d(
            in_channels=input_channels,
            out_channels=self.width,  # Project back to width channels
            kernel_size=1,
            bias=True
        )
        
        # Remaining convolution layers (after the first one)
        self.convs = nn.ModuleList([
            SeparableConvBlock(
                num_channels=self.width, 
                kernel_size=3, 
                strides=1
            ) for _ in range(self.depth-1)  # One less because we have a separate first conv
        ])
        
        # Head layers for xy and z
        self.head_xy_depthwise = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.width,
            bias=True
        )
        
        self.head_xy_pointwise = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.num_anchors * 2,  # x, y coordinates
            kernel_size=1,
            bias=True
        )
        
        self.head_z_depthwise = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.width,
            bias=True
        )
        
        self.head_z_pointwise = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.num_anchors,  # z coordinate
            kernel_size=1,
            bias=True
        )
        
        # Normalization layers
        # Structure: [iteration_steps][depth][pyramid_levels]
        if self.use_group_norm:
            # First layer norm is for the output of first_conv
            self.first_norm_layers = nn.ModuleList([
                nn.ModuleList([
                    GroupNormalization(num_channels=self.width, groups=self.num_groups_gn)
                    for _ in range(5)  # 5 pyramid levels (P3-P7)
                ])
                for _ in range(self.num_iteration_steps)
            ])
            
            # Remaining norm layers
            self.norm_layers = nn.ModuleList([
                nn.ModuleList([
                    nn.ModuleList([
                        GroupNormalization(num_channels=self.width, groups=self.num_groups_gn)
                        for _ in range(5)  # 5 pyramid levels (P3-P7)
                    ])
                    for _ in range(self.depth-1)  # One less because we handle first layer separately
                ])
                for _ in range(self.num_iteration_steps)
            ])
        else:
            # First layer norm is for the output of first_conv
            self.first_norm_layers = nn.ModuleList([
                nn.ModuleList([
                    nn.BatchNorm2d(self.width, momentum=0.997, eps=1e-4)
                    for _ in range(5)  # 5 pyramid levels (P3-P7)
                ])
                for _ in range(self.num_iteration_steps)
            ])
            
            # Remaining norm layers
            self.norm_layers = nn.ModuleList([
                nn.ModuleList([
                    nn.ModuleList([
                        nn.BatchNorm2d(self.width, momentum=0.997, eps=1e-4)
                        for _ in range(5)  # 5 pyramid levels (P3-P7)
                    ])
                    for _ in range(self.depth-1)  # One less because we handle first layer separately
                ])
                for _ in range(self.num_iteration_steps)
            ])
        
        self.activation = nn.SiLU()  # Equivalent to TensorFlow's Swish
        
    def forward(self, inputs, level_py, iter_step_py):
        feature, level = inputs
        
        # First conv layer that handles the concatenated input
        feature = self.first_conv_depthwise(feature)
        feature = self.first_conv_pointwise(feature)
        feature = self.first_norm_layers[iter_step_py][level_py](feature)
        feature = self.activation(feature)
        
        # Remaining conv layers
        for i in range(self.depth-1):
            feature = self.convs[i](feature)
            feature = self.norm_layers[iter_step_py][i][level_py](feature)
            feature = self.activation(feature)
        
        # Apply heads
        outputs_xy = self.head_xy_depthwise(feature)
        outputs_xy = self.head_xy_pointwise(outputs_xy)
        
        outputs_z = self.head_z_depthwise(feature)
        outputs_z = self.head_z_pointwise(outputs_z)
        
        return outputs_xy, outputs_z


# TranslationNet
class TranslationNet(nn.Module):
    def __init__(self, width, depth, num_iteration_steps, num_anchors=9, 
                 freeze_bn=False, use_group_norm=True, num_groups_gn=None, name="translation_net"):
        super(TranslationNet, self).__init__()
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        self.name = name
        
        # Convolution layers
        self.convs = nn.ModuleList([
            SeparableConvBlock(
                num_channels=self.width, 
                kernel_size=3, 
                strides=1
            ) for _ in range(self.depth)
        ])
        
        # Initial translation heads for xy and z
        self.initial_translation_xy_depthwise = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.width,
            bias=True
        )
        
        self.initial_translation_xy_pointwise = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.num_anchors * 2,  # x, y coordinates
            kernel_size=1,
            bias=True
        )
        
        self.initial_translation_z_depthwise = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.width,
            bias=True
        )
        
        self.initial_translation_z_pointwise = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.num_anchors,  # z coordinate
            kernel_size=1,
            bias=True
        )
        
        # Normalization layers
        if self.use_group_norm:
            self.norm_layers = nn.ModuleList([
                nn.ModuleList([
                    GroupNormalization(num_channels=self.width, groups=self.num_groups_gn)
                    for _ in range(5)  # 5 pyramid levels (P3-P7)
                ])
                for _ in range(self.depth)
            ])
        else:
            self.norm_layers = nn.ModuleList([
                nn.ModuleList([
                    nn.BatchNorm2d(self.width, momentum=0.997, eps=1e-4)
                    for _ in range(5)  # 5 pyramid levels (P3-P7)
                ])
                for _ in range(self.depth)
            ])
        
        # Iterative subnet
        self.iterative_submodel = IterativeTranslationSubNet(
            width=self.width,
            depth=self.depth - 1,
            num_iteration_steps=self.num_iteration_steps,
            num_anchors=self.num_anchors,
            freeze_bn=freeze_bn,
            use_group_norm=self.use_group_norm,
            num_groups_gn=self.num_groups_gn
        )
        
        self.activation = nn.SiLU()  # Equivalent to TensorFlow's Swish
        self.level = 0
        
    def forward(self, inputs):
        feature, level = inputs
        
        # Apply convolutional layers with normalization
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norm_layers[i][self.level](feature)
            feature = self.activation(feature)
        
        # Initial translation prediction
        translation_xy = self.initial_translation_xy_depthwise(feature)
        translation_xy = self.initial_translation_xy_pointwise(translation_xy)
        
        translation_z = self.initial_translation_z_depthwise(feature)
        translation_z = self.initial_translation_z_pointwise(translation_z)
        
        # Iterative refinement
        for i in range(self.num_iteration_steps):
            # Concatenate feature, translation_xy and translation_z along channel dimension
            iterative_input = torch.cat([feature, translation_xy, translation_z], dim=1)
            
            # Get delta translation from iterative subnet
            delta_translation_xy, delta_translation_z = self.iterative_submodel(
                [iterative_input, level], 
                level_py=self.level, 
                iter_step_py=i
            )
            
            # Update translation
            translation_xy = translation_xy + delta_translation_xy
            translation_z = translation_z + delta_translation_z
        
        # Reshape xy output to [batch_size, -1, 2]
        outputs_xy = translation_xy.permute(0, 2, 3, 1).contiguous()  # (B, H, W, anchors*2)
        outputs_xy = outputs_xy.view(outputs_xy.shape[0], -1, 2)  # (B, num_boxes, 2)
        
        # Reshape z output to [batch_size, -1, 1]
        outputs_z = translation_z.permute(0, 2, 3, 1).contiguous()  # (B, H, W, anchors)
        outputs_z = outputs_z.view(outputs_z.shape[0], -1, 1)  # (B, num_boxes, 1)
        
        # Concatenate xy and z to get translation with shape [batch_size, -1, 3]
        outputs = torch.cat([outputs_xy, outputs_z], dim=-1)  # (B, num_boxes, 3)
        
        self.level += 1
        return outputs
    
##########################################
# GroupNormalization
##########################################
class GroupNormalization(nn.Module):
    """
    Group normalization layer for PyTorch.
    Equivalent to the TensorFlow implementation from TensorFlow Addons.
    
    Args:
        num_channels: Number of channels in the input tensor
        groups: Integer, the number of groups for Group Normalization.
            Can be in the range [1, num_channels]. The input dimension 
            must be divisible by the number of groups.
        eps: Small float added to variance to avoid dividing by zero.
        affine: If True, use learnable affine parameters (gamma, beta).
    """
    def __init__(self, num_channels, groups=2, eps=1e-5, affine=True):
        super(GroupNormalization, self).__init__()
        self.num_channels = num_channels
        self.groups = groups
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))  # gamma
            self.bias = nn.Parameter(torch.zeros(num_channels))   # beta
        
        # Make sure number of channels is divisible by groups
        assert num_channels % groups == 0, f"Number of channels ({num_channels}) must be divisible by groups ({groups})"
    
    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        batch_size, channels, height, width = x.size()
        
        # Reshape input to separate groups
        # [batch_size, groups, channels//groups, height, width]
        x = x.view(batch_size, self.groups, -1, height, width)
        
        # Normalize over all dimensions except channels
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back to original
        x = x.view(batch_size, channels, height, width)
        
        # Apply affine transformation
        if self.affine:
            x = x * self.weight.view(1, channels, 1, 1) + self.bias.view(1, channels, 1, 1)
        
        return x


##########################################
# subnet build 
##########################################

def build_subnets(num_classes, subnet_width, subnet_depth, subnet_num_iteration_steps, num_groups_gn, num_rotation_parameters, freeze_bn, num_anchors):
    # Instantiate BoxNet
    box_net = BoxNet(
        width=subnet_width,
        depth=subnet_depth,
        num_anchors=num_anchors,
        freeze_bn=freeze_bn,
        name = 'box_net'
    )

    # Instantiate ClassNet
    class_net = ClassNet(
        width=subnet_width,
        depth=subnet_depth,
        num_classes=num_classes,
        num_anchors=num_anchors,
        freeze_bn=freeze_bn,
        name = 'class_net'
    )

    # Instantiate RotationNet
    rotation_net = RotationNet(
        width=subnet_width,
        depth=subnet_depth,
        num_values=num_rotation_parameters,
        num_iteration_steps=subnet_num_iteration_steps,
        num_anchors=num_anchors,
        freeze_bn=freeze_bn,
        use_group_norm=True,
        num_groups_gn=num_groups_gn,
        name = 'rotation_net'
    )

    # Instantiate TranslationNet
    translation_net = TranslationNet(
        width=subnet_width,
        depth=subnet_depth,
        num_iteration_steps=subnet_num_iteration_steps,
        num_anchors=num_anchors,
        freeze_bn=freeze_bn,
        use_group_norm=True,
        num_groups_gn=num_groups_gn,
        name = 'translation_net'
    )


    return box_net, class_net, rotation_net, translation_net

class RegressTranslation(nn.Module):
    """
    PyTorch module for applying regression offset values to translation anchors 
    to get the 2D translation centerpoint and Tz.
    """
    
    def __init__(self):
        """Initializer for the RegressTranslation layer."""
        super(RegressTranslation, self).__init__()
    
    def forward(self, inputs):
        """
        Apply regression to translation anchors.
        
        Args:
            inputs: List containing [translation_anchors, regression_offsets]
                translation_anchors: tensor of shape (B, N, 3) where the last dimension is (x, y, stride)
                regression_offsets: tensor of shape (B, N, 3) where the last dimension is (dx, dy, tz)
                
        Returns:
            tensor of shape (B, N, 3) containing the transformed coordinates (x, y, tz)
        """
        translation_anchors, regression_offsets = inputs
        return self.translation_transform_inv(translation_anchors, regression_offsets)
    
    def translation_transform_inv(self, translation_anchors, deltas, scale_factors=None):
        """
        Applies the predicted 2D translation center point offsets (deltas) to the translation_anchors
        
        Args:
            translation_anchors: Tensor of shape (B, N, 3), where B is the batch size, 
                                 N the number of boxes and 3 values for (x, y, stride)
            deltas: Tensor of shape (B, N, 3). The first 2 values (dx, dy) are factors of 
                    the stride, and the third is Tz
            scale_factors: Optional scaling factors for the deltas
            
        Returns:
            A tensor of shape (B, N, 3) with the transformed coordinates (x, y, tz)
        """
        stride = translation_anchors[:, :, 2]
        
        if scale_factors is not None:
            x = translation_anchors[:, :, 0] + (deltas[:, :, 0] * scale_factors[0] * stride)
            y = translation_anchors[:, :, 1] + (deltas[:, :, 1] * scale_factors[1] * stride)
        else:
            x = translation_anchors[:, :, 0] + (deltas[:, :, 0] * stride)
            y = translation_anchors[:, :, 1] + (deltas[:, :, 1] * stride)
        
        Tz = deltas[:, :, 2]
        
        # Stack the predictions to form the final output
        pred_translations = torch.stack([x, y, Tz], dim=2)  # x,y 2D Image coordinates and Tz
        
        return pred_translations
    
class CalculateTxTy(nn.Module):
    """
    PyTorch module for calculating the Tx- and Ty-Components of the Translation vector 
    with a given 2D-point and the intrinsic camera parameters.
    """
    
    def __init__(self):
        """
        Initializer for a CalculateTxTy layer.
        """
        super(CalculateTxTy, self).__init__()
    
    def forward(self, inputs, fx, fy, px, py, tz_scale, image_scale):
        """
        Calculate the translation vector.
        
        Args:
            inputs: tensor of shape (B, N, 3) containing (x, y, tz)
            fx, fy, px, py: camera intrinsic parameters
            tz_scale: scaling factor for the z component
            image_scale: scaling factor for the image coordinates
            
        Returns:
            tensor of shape (B, N, 3) containing the translation vector (tx, ty, tz)
        """
        # Expand dimensions for proper broadcasting
        fx = fx.unsqueeze(-1)  # (B, 1)
        fy = fy.unsqueeze(-1)  # (B, 1)
        px = px.unsqueeze(-1)  # (B, 1)
        py = py.unsqueeze(-1)  # (B, 1)
        tz_scale = tz_scale.unsqueeze(-1)  # (B, 1)
        image_scale = image_scale.unsqueeze(-1)  # (B, 1)
        
        # Extract components from inputs
        x = inputs[:, :, 0] / image_scale  # Scale the x-coordinate
        y = inputs[:, :, 1] / image_scale  # Scale the y-coordinate
        tz = inputs[:, :, 2] * tz_scale    # Scale the z-component
        
        # Apply camera calibration parameters
        x = x - px
        y = y - py
        
        # Calculate tx and ty using the pinhole camera model
        tx = torch.mul(x, tz) / fx
        ty = torch.mul(y, tz) / fy
        
        # Stack the results to create the translation vector
        output = torch.stack([tx, ty, tz], dim=-1)
        
        return output
    
class RegressBoxes(nn.Module):
    """
    PyTorch module for applying regression offset values to anchor boxes to get the 2D bounding boxes.
    """
    def __init__(self):
        super(RegressBoxes, self).__init__()

    def forward(self, inputs):
        """
        Apply regression to anchor boxes.
        
        Args:
            inputs: List containing [anchors, regression]
                anchors: Tensor of shape (B, N, 4) containing the anchor boxes
                regression: Tensor of shape (B, N, 4) containing the offsets
        
        Returns:
            Tensor of shape (B, N, 4) containing the predicted boxes
        """
        anchors, regression = inputs
        return self.bbox_transform_inv(anchors, regression)
    
    def bbox_transform_inv(self, boxes, deltas, scale_factors=None):
        """
        Reconstructs the 2D bounding boxes using the anchor boxes and the predicted deltas 
        of the anchor boxes to the bounding boxes.
        
        Args:
            boxes: Tensor containing the anchor boxes with shape (..., 4)
            deltas: Tensor containing the offsets of the anchor boxes to the bounding boxes with shape (..., 4)
            scale_factors: Optional scaling factor for the deltas
            
        Returns:
            Tensor containing the reconstructed 2D bounding boxes with shape (..., 4)
        """
        # Calculate center, width, height of anchor boxes
        cxa = (boxes[..., 0] + boxes[..., 2]) / 2
        cya = (boxes[..., 1] + boxes[..., 3]) / 2
        wa = boxes[..., 2] - boxes[..., 0]
        ha = boxes[..., 3] - boxes[..., 1]
        
        # Extract regression values
        ty, tx, th, tw = deltas[..., 0], deltas[..., 1], deltas[..., 2], deltas[..., 3]
        
        # Apply scale factors if provided
        if scale_factors:
            ty = ty * scale_factors[0]
            tx = tx * scale_factors[1]
            th = th * scale_factors[2]
            tw = tw * scale_factors[3]
        
        # Apply transformations
        w = torch.exp(tw) * wa
        h = torch.exp(th) * ha
        cy = ty * ha + cya
        cx = tx * wa + cxa
        
        # Calculate box coordinates
        ymin = cy - h / 2.
        xmin = cx - w / 2.
        ymax = cy + h / 2.
        xmax = cx + w / 2.
        
        # Stack to create box tensor
        return torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    
class ClipBoxes(nn.Module):
    """
    PyTorch module that clips 2D bounding boxes so that they are inside the image
    """
    def __init__(self):
        super(ClipBoxes, self).__init__()
    
    def forward(self, inputs):
        """
        Clip boxes to image boundaries.
        
        Args:
            inputs: List containing [image, boxes]
                image: Tensor of shape (B, C, H, W)
                boxes: Tensor of shape (B, N, 4)
            
        Returns:
            Tensor of shape (B, N, 4) containing clipped boxes
        """
        image, boxes = inputs
        
        # Get image dimensions - note that in PyTorch images are (B, C, H, W)
        height = image.shape[2]
        width = image.shape[3]
        
        # Clip box coordinates to image boundaries
        x1 = torch.clamp(boxes[:, :, 0], 0, width - 1)
        y1 = torch.clamp(boxes[:, :, 1], 0, height - 1)
        x2 = torch.clamp(boxes[:, :, 2], 0, width - 1)
        y2 = torch.clamp(boxes[:, :, 3], 0, height - 1)
        
        # Stack to create the final tensor
        return torch.stack([x1, y1, x2, y2], dim=2)

##########################################
# Build EfficientPose Model
##########################################
def apply_subnets_to_feature_maps(box_net, class_net, rotation_net, translation_net, fpn_feature_maps, 
                                 image_input, camera_parameters, input_size, anchor_parameters=None):
    

    classification = []
    for i, feature in enumerate(fpn_feature_maps):
        classification.append(class_net(feature, i))
    classification = torch.cat(classification, dim=1)

    print(f"Classification shape: {classification.shape}")
    
    # Apply box network to feature maps and concatenate results
    bbox_regression = []
    for i, feature in enumerate(fpn_feature_maps):
        bbox_regression.append(box_net(feature, i))
    bbox_regression = torch.cat(bbox_regression, dim=1)

    print(f"Bbox regression shape: {bbox_regression.shape}")
    
    # Apply rotation network to feature maps and concatenate results
    rotation = []
    for i, feature in enumerate(fpn_feature_maps):
        rotation.append(rotation_net([feature, i]))
    rotation = torch.cat(rotation, dim=1)
    

    print(f"Rotation shape: {rotation.shape}")

    # Apply translation network to feature maps and concatenate results
    translation_raw = []
    for i, feature in enumerate(fpn_feature_maps):
        translation_raw.append(translation_net([feature, i]))
    translation_raw = torch.cat(translation_raw, dim=1)

    print(f"Translation raw shape: {translation_raw.shape}")

    anchors, translation_anchors = anchors_for_shape((input_size, input_size), anchor_params=anchor_parameters)
    translation_anchors_input = translation_anchors.unsqueeze(0)

    print(f"Anchors shape: {anchors.shape}")

    print(f"Translation anchors shape: {translation_anchors_input.shape}")

    # Apply regression to translation anchors
    regress_translation = RegressTranslation()
    translation_xy_Tz = regress_translation([translation_anchors_input, translation_raw])

    print(f"Translation xy Tz shape: {translation_xy_Tz.shape}")

    calculate_txty = CalculateTxTy()
    translation = calculate_txty(
        translation_xy_Tz,
        fx=camera_parameters[:, 0],
        fy=camera_parameters[:, 1],
        px=camera_parameters[:, 2],
        py=camera_parameters[:, 3],
        tz_scale=camera_parameters[:, 4],
        image_scale=camera_parameters[:, 5]
    )

    print(f"Translation shape: {translation.shape}")

    anchors_tensor = anchors.unsqueeze(0).to(translation_raw.device)

    print(f"Anchors tensor shape: {anchors_tensor.shape}")

    # Apply regression to get predicted bounding boxes
    regress_boxes = RegressBoxes()
    bboxes = regress_boxes([anchors_tensor, bbox_regression[..., :4]])

    print(f"Bboxes shape: {bboxes.shape}")

    # Clip bounding boxes to image boundaries
    clip_boxes = ClipBoxes()
    bboxes = clip_boxes([image_input, bboxes])

    print(f"Clipped bboxes shape: {bboxes.shape}")

    # Concatenate rotation and translation outputs to transformation output
    transformation = torch.cat([rotation, translation], dim=-1)

    print(f"Transformation shape: {transformation.shape}")

    return classification, bbox_regression, rotation, translation, bboxes, transformation

    
class BuildEfficientPoseModel(nn.Module):
    def __init__(self, phi, num_classes=8, num_anchors=9, freeze_bn=False, score_threshold=0.5, anchor_parameters=None, num_rotation_parameters=3, print_architecture=True):
        super(BuildEfficientPoseModel, self).__init__()

        # Select parameters according to the given phi
        assert phi in range(7)
        scaled_parameters = get_scaled_parameters(phi)
        self.phi = phi
        self.input_size = scaled_parameters["input_size"] 
        self.bifpn_width = scaled_parameters["bifpn_width"]
        self.bifpn_depth = scaled_parameters["bifpn_depth"]
        self.subnet_depth = scaled_parameters["subnet_depth"]
        self.subnet_num_iteration_steps = scaled_parameters["subnet_num_iteration_steps"]
        self.num_groups_gn = scaled_parameters["num_groups_gn"]
        self.backbone_class = scaled_parameters["backbone_class"]
        self.freeze_bn = freeze_bn
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_rotation_parameters = num_rotation_parameters
        self.score_threshold = score_threshold
        self.input_shape = torch.randn(1, 3, self.input_size, self.input_size)
        self.camera_parameters_input = torch.randn(1, 6)

        # print input_shape
        print(f"Input shape: {self.input_shape.shape}")
        # print camera_parameters_input
        print(f"Camera parameters input shape: {self.camera_parameters_input.shape}")
        
        self.feature_extractor = self._build_backbone()

        self.features = self.feature_extractor(self.input_shape)  # Extract features as a dictionary
        self.backbone_feature_maps = [self.features[node_name] for node_name in ["C1", "C2", "C3", "C4", "C5"]]

        # ✅ Build BiFPN
        self.fpn_feature_maps = build_BiFPN(
            self.backbone_feature_maps,
            bifpn_depth=self.bifpn_depth,
            bifpn_width=self.bifpn_width,
            freeze_bn=self.freeze_bn
        )
    
        # ✅ Build subnets
        box_net, class_net, rotation_net, translation_net = build_subnets(
            num_classes=self.num_classes,
            subnet_width=self.bifpn_width,
            subnet_depth=self.subnet_depth,
            subnet_num_iteration_steps=self.subnet_num_iteration_steps,
            num_groups_gn=self.num_groups_gn,
            num_rotation_parameters=self.num_rotation_parameters,
            freeze_bn=self.freeze_bn,
            num_anchors=self.num_anchors
        )

        self.box_net = box_net
        self.class_net = class_net
        self.rotation_net = rotation_net
        self.translation_net = translation_net

        # ✅ Apply subnets to feature maps
        classification, bbox_regression, rotation, translation, transformation, bboxes = apply_subnets_to_feature_maps(
            box_net, class_net, rotation_net, translation_net,
            self.fpn_feature_maps, self.input_shape, self.camera_parameters_input, self.input_size
        )

        self.classification = classification
        self.bbox_regression = bbox_regression
        self.rotation = rotation
        self.translation = translation
        self.transformation = transformation
        self.bboxes = bboxes

        # Create FilterDetections module
        self.filter_detections = FilterDetections(
            num_rotation_parameters = self.num_rotation_parameters,
            num_translation_parameters = 3,
            score_threshold = self.score_threshold 
        )

        self.filtered_detections = self.filter_detections([self.bboxes, self.classification, self.rotation, self.translation])
        

    def _build_backbone(self):
        """Build the EfficientNet backbone"""

        if self.phi == 0:
            from torchvision.models import EfficientNet_B0_Weights
            backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif self.phi == 1:
            from torchvision.models import EfficientNet_B1_Weights
            backbone = models.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
        elif self.phi == 2:
            from torchvision.models import EfficientNet_B2_Weights
            backbone = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        elif self.phi == 3:
            from torchvision.models import EfficientNet_B3_Weights
            backbone = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        elif self.phi == 4:
            from torchvision.models import EfficientNet_B4_Weights
            backbone = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        elif self.phi == 5:
            from torchvision.models import EfficientNet_B5_Weights
            backbone = models.efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        elif self.phi == 6:
            from torchvision.models import EfficientNet_B6_Weights
            backbone = models.efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)

        # Corrected layer names for feature extraction based on your printed architecture
        self.return_nodes = {
            "features.1": "C1",  # Low-level features
            "features.2": "C2",  # Mid-level features
            "features.3": "C3",  # Higher-level features
            "features.5": "C4",  # Deep features
            "features.7": "C5"   # Final feature map
        }

        # Create the feature extractor with the corrected layers
        feature_extractor = create_feature_extractor(backbone, return_nodes=self.return_nodes)
        return feature_extractor



    def forward(self, inference=False):
        print("Forward pass")

def build_EfficientPose(phi,
                        num_classes=8,
                        num_anchors=9,
                        freeze_bn=False,
                        score_threshold=0.5,
                        anchor_parameters=None,
                        num_rotation_parameters=3,
                        print_architecture=True):
    
        # Select parameters according to the given phi
        assert phi in range(7)
        scaled_parameters = get_scaled_parameters(phi)
        phi = phi
        input_size = scaled_parameters["input_size"] 
        bifpn_width = scaled_parameters["bifpn_width"]
        bifpn_depth = scaled_parameters["bifpn_depth"]
        subnet_depth = scaled_parameters["subnet_depth"]
        subnet_num_iteration_steps = scaled_parameters["subnet_num_iteration_steps"]
        num_groups_gn = scaled_parameters["num_groups_gn"]
        backbone_class = scaled_parameters["backbone_class"]
        freeze_bn = freeze_bn
        num_classes = num_classes
        num_anchors = num_anchors
        num_rotation_parameters = num_rotation_parameters
        score_threshold = score_threshold
        input_shape = torch.randn(1, 3, input_size, input_size)
        camera_parameters_input = torch.randn(1, 6)

        # print input_shape
        print(f"Input shape: {input_shape.shape}")
        # print camera_parameters_input
        print(f"Camera parameters input shape: {camera_parameters_input.shape}")
        
        feature_extractor = _build_backbone_two(phi)

        features = feature_extractor(input_shape)  # Extract features as a dictionary
        backbone_feature_maps = [features[node_name] for node_name in ["C1", "C2", "C3", "C4", "C5"]]

        # ✅ Build BiFPN
        fpn_feature_maps = build_BiFPN(
            backbone_feature_maps,
            bifpn_depth=bifpn_depth,
            bifpn_width=bifpn_width,
            freeze_bn=freeze_bn
        )
    
        # ✅ Build subnets
        box_net, class_net, rotation_net, translation_net = build_subnets(
            num_classes=num_classes,
            subnet_width=bifpn_width,
            subnet_depth=subnet_depth,
            subnet_num_iteration_steps=subnet_num_iteration_steps,
            num_groups_gn=num_groups_gn,
            num_rotation_parameters=num_rotation_parameters,
            freeze_bn=freeze_bn,
            num_anchors=num_anchors
        )

        # ✅ Apply subnets to feature maps
        classification, bbox_regression, rotation, translation, transformation, bboxes = apply_subnets_to_feature_maps(
            box_net, class_net, rotation_net, translation_net,
            fpn_feature_maps, input_shape, camera_parameters_input, input_size
        )

        efficientpose_train = nn.Sequential(
            box_net,
            class_net,
            rotation_net,
            translation_net
        )

        # Create FilterDetections module
        filter_detections = FilterDetections(
            num_rotation_parameters = num_rotation_parameters,
            num_translation_parameters = 3,
            score_threshold = score_threshold 
        )

        filtered_detections = filter_detections([bboxes, classification, rotation, translation])
        
        efficientpose_prediction = nn.Sequential(filtered_detections)

        return efficientpose_train, efficientpose_prediction

def _build_backbone_two(phi):
    """Build the EfficientNet backbone"""

    if phi == 0:
        from torchvision.models import EfficientNet_B0_Weights
        backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    elif phi == 1:
        from torchvision.models import EfficientNet_B1_Weights
        backbone = models.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
    elif phi == 2:
        from torchvision.models import EfficientNet_B2_Weights
        backbone = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    elif phi == 3:
        from torchvision.models import EfficientNet_B3_Weights
        backbone = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    elif phi == 4:
        from torchvision.models import EfficientNet_B4_Weights
        backbone = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    elif phi == 5:
        from torchvision.models import EfficientNet_B5_Weights
        backbone = models.efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
    elif phi == 6:
        from torchvision.models import EfficientNet_B6_Weights
        backbone = models.efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)

    # Corrected layer names for feature extraction based on your printed architecture
    return_nodes = {
        "features.1": "C1",  # Low-level features
        "features.2": "C2",  # Mid-level features
        "features.3": "C3",  # Higher-level features
        "features.5": "C4",  # Deep features
        "features.7": "C5"   # Final feature map
    }

    # Create the feature extractor with the corrected layers
    feature_extractor = create_feature_extractor(backbone, return_nodes=return_nodes)
    return feature_extractor

# add main function to test the function
if __name__ == "__main__":
    # # Test the function


    # build_EfficientPose
    phi = 1  # Select EfficientNet-B2
    num_rotation_parameters = 3
    num_classes = 1
    num_anchors = 9
    freeze_bn = True
    score_threshold = 0.7
    model = BuildEfficientPoseModel(phi, num_classes, num_anchors, freeze_bn, score_threshold, num_rotation_parameters)

    # Print feature maps in TensorFlow-like format
    print("\nBiFPN feature maps shape:")
    for i, fm in enumerate(model.fpn_feature_maps):
        print(f"Tensor(\"FPN{i+1}\", shape={tuple(fm.shape)}, dtype=float32)")


    # efficientpose_train, efficientpose_prediction = build_EfficientPose(phi, num_classes, num_anchors, freeze_bn, score_threshold, num_rotation_parameters)




