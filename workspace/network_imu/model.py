import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models.feature_extraction import create_feature_extractor

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





    
class BuildEfficientPoseModel(nn.Module):
    def __init__(self, phi, num_classes=8, num_anchors=9, freeze_bn=False, score_threshold=0.5, anchor_parameters=None, num_rotation_parameters=3, print_architecture=True):
        super(BuildEfficientPoseModel, self).__init__()

        # Select parameters according to the given phi
        assert phi in range(7)
        scaled_parameters = get_scaled_parameters(phi)
        self.phi = phi
        self.input_size = scaled_parameters["input_size"] 
        self.input_shape = (3, self.input_size, self.input_size)
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
        
        self.feature_extractor = self._build_backbone()

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



    def forward(self, x):
        """Returns feature maps in TensorFlow-like order"""
        features = self.feature_extractor(x)  # Extract features as a dictionary
        backbone_feature_maps = [features[node_name] for node_name in ["C1", "C2", "C3", "C4", "C5"]]

        # ✅ Build BiFPN
        fpn_feature_maps = build_BiFPN(
            backbone_feature_maps,
            bifpn_depth=self.bifpn_depth,
            bifpn_width=self.bifpn_width,
            freeze_bn=self.freeze_bn
        )

        return fpn_feature_maps



class EfficientPoseModel(nn.Module):
    def __init__(self, input_size=512, phi=0):
        super(EfficientPoseModel, self).__init__()
        
        # Get scaling parameters based on phi value
        self.input_size = input_size
        params = get_scaled_parameters(phi)
        self.bifpn_width = params['bifpn_width']
        self.bifpn_depth = params['bifpn_depth']
        
        # Backbone with proper downsampling pattern
        # These should match the TensorFlow downsampling pattern
        self.backbone_stage1 = nn.Conv2d(3, 32, kernel_size=3, stride=4, padding=1)  # 4× downsample
        self.backbone_stage2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 8× total
        self.backbone_stage3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 16× total
        self.backbone_stage4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 32× total
        self.backbone_stage5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # 64× total
        
        # Output head (example: keypoint regression)
        self.output_head = nn.Conv2d(self.bifpn_width, 17, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        # Extract features from backbone
        C1 = self.backbone_stage1(x)
        C2 = self.backbone_stage2(C1)
        C3 = self.backbone_stage3(C2)
        C4 = self.backbone_stage4(C3)
        C5 = self.backbone_stage5(C4)
        
        backbone_features = [C1, C2, C3, C4, C5]
        
        # Build BiFPN
        bifpn_features = build_BiFPN(
            backbone_features,
            bifpn_depth=self.bifpn_depth,
            bifpn_width=self.bifpn_width,
            freeze_bn=False
        )
        
        # Unpack BiFPN features - directly use P3 for output
        P3, _, _, _, _ = bifpn_features
        
        # Apply output head to P3
        outputs = self.activation(self.output_head(P3))
        
        return outputs
    
def count_parameters(model):
    """Count the total parameters, trainable parameters, and non-trainable parameters in a PyTorch model"""
    total_params = sum(p.numel() for p in model.parameters())
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    non_trainable_params = total_params - trainable_params
    
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
    
    return total_params, trainable_params, non_trainable_params


# add main function to test the function
if __name__ == "__main__":
    # # Test the function
    # phi = 0
    # scaled_parameters = get_scaled_parameters(phi)
    # print(scaled_parameters)
    # # Expected output: {'input_size': 512, 'bifpn_width': 64, 'bifpn_depth': 3, 'subnet_depth': 3, 'subnet_num_iteration_steps': 1, 'num_groups_gn': 4, 'backbone_class': <function efficientnet_b0 at 0x7f8c6b5e8d30>}


    # # Create dummy backbone feature maps with realistic shapes
    # batch_size = 2
    # C1 = torch.randn(batch_size, 32, 128, 128)
    # C2 = torch.randn(batch_size, 64, 64, 64)
    # C3 = torch.randn(batch_size, 128, 32, 32)
    # C4 = torch.randn(batch_size, 256, 16, 16)
    # C5 = torch.randn(batch_size, 512, 8, 8)

    # # Gather backbone features
    # backbone_features = [C1, C2, C3, C4, C5]

    # # Set BiFPN parameters
    # num_channels = 160
    # freeze_bn = True

    # # Build the first BiFPN layer
    # P3, P4, P5, P6, P7 = build_BiFPN_layer(
    #     backbone_features, 
    #     num_channels=num_channels, 
    #     idx_BiFPN_layer=0, 
    #     freeze_bn=freeze_bn
    # )

    # # Print output shapes
    # print("BiFPN Output Feature Maps:")
    # print(f"P3 shape: {P3.shape}")  # Should be [batch_size, num_channels, 32, 32]
    # print(f"P4 shape: {P4.shape}")  # Should be [batch_size, num_channels, 16, 16]
    # print(f"P5 shape: {P5.shape}")  # Should be [batch_size, num_channels, 8, 8]
    # print(f"P6 shape: {P6.shape}")  # Should be [batch_size, num_channels, 4, 4]
    # print(f"P7 shape: {P7.shape}")  # Should be [batch_size, num_channels, 2, 2]

    # # Create model
    # model = EfficientPoseModel(input_size=512)
    # print(model)

    # count_parameters(model)

    # # Test inference with a dummy image
    # dummy_input = torch.rand(1, 3, 512, 512)
    # predictions = model(dummy_input)
    # print(f"Prediction shape: {predictions.shape}")

    # build_EfficientPose
    phi = 3  # Select EfficientNet-B2
    model = BuildEfficientPoseModel(phi)

    # Create dummy input
    dummy_input = torch.randn(1, 3, model.input_size, model.input_size)

    # Get feature maps
    feature_maps = model(dummy_input)

    # Print feature maps in TensorFlow-like format
    print("\nBiFPN feature maps shape:")
    for i, fm in enumerate(feature_maps):
        print(f"Tensor(\"FPN{i+1}\", shape={tuple(fm.shape)}, dtype=float32)")