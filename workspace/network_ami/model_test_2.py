import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.ops as ops

#########################################
# Utility and helper functions/modules
#########################################

def get_scaled_parameters(phi):
    image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
    bifpn_widths = [64, 88, 112, 160, 224, 288, 384]
    bifpn_depths = [3, 4, 5, 6, 7, 7, 8]
    subnet_depths = [3, 3, 3, 4, 4, 4, 5]
    subnet_iteration_steps = [1, 1, 1, 2, 2, 2, 3]
    num_groups_gn = [4, 4, 7, 10, 14, 18, 24]
    
    # PyTorch backbone equivalents
    backbones = [
        models.efficientnet_b0,
        models.efficientnet_b1,
        models.efficientnet_b2,
        models.efficientnet_b3,
        models.efficientnet_b4,
        models.efficientnet_b5,
        models.efficientnet_b6
    ]
    
    parameters = {
        "input_size": image_sizes[phi],
        "bifpn_width": bifpn_widths[phi],
        "bifpn_depth": bifpn_depths[phi],
        "subnet_depth": subnet_depths[phi],
        "subnet_num_iteration_steps": subnet_iteration_steps[phi],
        "num_groups_gn": num_groups_gn[phi],
        "backbone_class": backbones[phi]
    }
    
    return parameters

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(SeparableConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.997, eps=1e-4)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.silu(x)  # Swish activation equivalent in PyTorch

class FeatureSplitter(nn.Module):
    """
    Takes the backbone's output (assumed to be 1280 channels from EfficientNet-B0)
    and produces multi-scale features.
    """
    def __init__(self, out_channels):
        super(FeatureSplitter, self).__init__()
        self.resample = nn.Conv2d(1280, out_channels, kernel_size=1)
        
    def forward(self, features):
        # features: [N, 1280, H, W]
        x = self.resample(features)  # now x is [N, out_channels, H, W]
        # Generate feature maps by progressively downsampling.
        P3 = x
        P4 = F.max_pool2d(x, kernel_size=2)
        P5 = F.max_pool2d(P4, kernel_size=2)
        P6 = F.max_pool2d(P5, kernel_size=2)
        P7 = F.max_pool2d(P6, kernel_size=2)
        return [P3, P4, P5, P6, P7]

#########################################
# Enhanced BiFPN
#########################################

class BiFPNLayer(nn.Module):
    def __init__(self, num_channels):
        super(BiFPNLayer, self).__init__()
        # Top-down convolution blocks
        self.conv_td = nn.ModuleList([SeparableConvBlock(num_channels, num_channels) for _ in range(5)])
        # Bottom-up convolution blocks
        self.conv_bu = nn.ModuleList([SeparableConvBlock(num_channels, num_channels) for _ in range(5)])
    
    def forward(self, features):
        # features: [P3, P4, P5, P6, P7]
        P3, P4, P5, P6, P7 = features
        
        # Top-down pathway
        P7_td = self.conv_td[4](P7)
        P6_td = self.conv_td[3](P6) + F.interpolate(P7_td, scale_factor=2, mode='bilinear', align_corners=False)
        P5_td = self.conv_td[2](P5) + F.interpolate(P6_td, scale_factor=2, mode='bilinear', align_corners=False)
        P4_td = self.conv_td[1](P4) + F.interpolate(P5_td, scale_factor=2, mode='bilinear', align_corners=False)
        P3_td = self.conv_td[0](P3) + F.interpolate(P4_td, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Bottom-up pathway
        P3_out = P3_td
        P4_out = self.conv_bu[1](P4_td) + P4_td + F.max_pool2d(P3_out, kernel_size=2)
        P5_out = self.conv_bu[2](P5_td) + P5_td + F.max_pool2d(P4_out, kernel_size=2)
        P6_out = self.conv_bu[3](P6_td) + P6_td + F.max_pool2d(P5_out, kernel_size=2)
        P7_out = self.conv_bu[4](P7_td) + P7_td + F.max_pool2d(P6_out, kernel_size=2)
        
        return [P3_out, P4_out, P5_out, P6_out, P7_out]

#########################################
# Subnetworks
#########################################

class BoxNet(nn.Module):
    """
    Box regression subnetwork.
    Predicts 4 offsets per anchor.
    """
    def __init__(self, in_channels, num_anchors, depth, width):
        super(BoxNet, self).__init__()
        layers_list = []
        for i in range(depth):
            inc = in_channels if i == 0 else width
            layers_list.append(SeparableConvBlock(inc, width))
        self.conv_layers = nn.Sequential(*layers_list)
        self.head = nn.Conv2d(width, num_anchors * 4, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.head(x)
        N = x.shape[0]
        x = x.permute(0, 2, 3, 1).contiguous().view(N, -1, 4)
        return x

class ClassNet(nn.Module):
    """
    Classification subnetwork.
    Predicts class probabilities for each anchor.
    """
    def __init__(self, in_channels, num_classes, num_anchors, depth, width):
        super(ClassNet, self).__init__()
        self.num_classes = num_classes
        layers_list = []
        for i in range(depth):
            inc = in_channels if i == 0 else width
            layers_list.append(SeparableConvBlock(inc, width))
        self.conv_layers = nn.Sequential(*layers_list)
        self.head = nn.Conv2d(width, num_anchors * num_classes, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.head(x)
        N = x.shape[0]
        x = x.permute(0, 2, 3, 1).contiguous().view(N, -1, self.num_classes)
        return torch.sigmoid(x)

class RotationNet(nn.Module):
    def __init__(self, in_channels, num_values, num_anchors, depth, width, num_iteration_steps):
        super(RotationNet, self).__init__()
        self.num_iteration_steps = num_iteration_steps
        self.num_values = num_values
        
        # Convolutional layers
        layers_list = []
        for i in range(depth):
            inc = in_channels if i == 0 else width
            layers_list.append(SeparableConvBlock(inc, width))
        self.conv_layers = nn.Sequential(*layers_list)
        
        # Initial rotation prediction
        self.initial_conv = nn.Conv2d(width, num_anchors * num_values, kernel_size=3, padding=1)
        
        # Iterative refinement
        self.iterative_conv = nn.Conv2d(width + num_anchors * num_values,
                                       num_anchors * num_values, kernel_size=3, padding=1)
        self.num_anchors = num_anchors

    def forward(self, x):
        features = self.conv_layers(x)
        rotation = self.initial_conv(features)
        
        # Iterative refinement steps
        for i in range(self.num_iteration_steps):
            iterative_input = torch.cat([features, rotation], dim=1)
            delta = self.iterative_conv(iterative_input)
            rotation = rotation + delta
            
        # Reshape to final output format
        N = rotation.shape[0]
        rotation = rotation.permute(0, 2, 3, 1).contiguous().view(N, -1, self.num_values)
        return rotation

class TranslationNet(nn.Module):
    def __init__(self, in_channels, num_anchors, depth, width, num_iteration_steps):
        super(TranslationNet, self).__init__()
        self.num_iteration_steps = num_iteration_steps
        
        # Shared feature extraction
        layers_list = []
        for i in range(depth):
            inc = in_channels if i == 0 else width
            layers_list.append(SeparableConvBlock(inc, width))
        self.conv_layers = nn.Sequential(*layers_list)
        
        # Initial translation prediction - xy and z separately
        self.initial_xy = nn.Conv2d(width, num_anchors * 2, kernel_size=3, padding=1)
        self.initial_z = nn.Conv2d(width, num_anchors, kernel_size=3, padding=1)
        
        # Iterative refinement
        self.iterative_conv = nn.Conv2d(width + num_anchors * 3, num_anchors * 3, kernel_size=3, padding=1)
        self.num_anchors = num_anchors

    def forward(self, x):
        features = self.conv_layers(x)
        
        # Initial predictions
        translation_xy = self.initial_xy(features)
        translation_z = self.initial_z(features)
        
        # Combine for iteration input
        combined = torch.cat([features, translation_xy, translation_z], dim=1)
        
        # Iterative refinement steps
        for i in range(self.num_iteration_steps):
            delta = self.iterative_conv(combined)
            
            # Split delta into xy and z components
            delta_xy = delta[:, :self.num_anchors*2, :, :]
            delta_z = delta[:, self.num_anchors*2:, :, :]
            
            # Update translation predictions
            translation_xy = translation_xy + delta_xy
            translation_z = translation_z + delta_z
            
            # Update combined features for next iteration
            combined = torch.cat([features, translation_xy, translation_z], dim=1)
        
        # Reshape outputs
        N = x.shape[0]
        translation_xy = translation_xy.permute(0, 2, 3, 1).contiguous().view(N, -1, 2)
        translation_z = translation_z.permute(0, 2, 3, 1).contiguous().view(N, -1, 1)
        
        # Combine into final translation output
        translation = torch.cat([translation_xy, translation_z], dim=2)
        return translation

#########################################
# Custom Post-processing Layers/Functions
#########################################

class ClipBoxes(nn.Module):
    """
    Clips 2D bounding boxes so that they lie within the image dimensions.
    """
    def forward(self, image, boxes):
        _, _, H, W = image.shape
        x1 = boxes[..., 0].clamp(0, W - 1)
        y1 = boxes[..., 1].clamp(0, H - 1)
        x2 = boxes[..., 2].clamp(0, W - 1)
        y2 = boxes[..., 3].clamp(0, H - 1)
        return torch.stack([x1, y1, x2, y2], dim=-1)

class RegressBoxes(nn.Module):
    """
    Applies regression deltas to anchor boxes.
    """
    def __init__(self, scale_factors=None):
        super(RegressBoxes, self).__init__()
        self.scale_factors = scale_factors

    def forward(self, anchors, regression):
        return bbox_transform_inv(anchors, regression, self.scale_factors)

class RegressTranslation(nn.Module):
    """
    Applies regression offsets to translation anchors.
    """
    def __init__(self, scale_factors=None):
        super(RegressTranslation, self).__init__()
        self.scale_factors = scale_factors

    def forward(self, translation_anchors, regression_offsets):
        return translation_transform_inv(translation_anchors, regression_offsets, self.scale_factors)

class CalculateTxTy(nn.Module):
    """
    Calculates Tx and Ty components from predicted translation offsets using camera intrinsics.
    """
    def __init__(self):
        super(CalculateTxTy, self).__init__()

    def forward(self, translation_xy_tz, camera_params):
        # camera_params: (batch_size, 6) [fx, fy, px, py, tz_scale, image_scale]
        batch_size = camera_params.shape[0]
        
        # Extract camera parameters
        fx = camera_params[:, 0].unsqueeze(1)  # (batch_size, 1)
        fy = camera_params[:, 1].unsqueeze(1)
        px = camera_params[:, 2].unsqueeze(1)
        py = camera_params[:, 3].unsqueeze(1)
        tz_scale = camera_params[:, 4].unsqueeze(1)
        image_scale = camera_params[:, 5].unsqueeze(1)
        
        # Apply to each box
        x = translation_xy_tz[..., 0] / image_scale
        y = translation_xy_tz[..., 1] / image_scale
        tz = translation_xy_tz[..., 2] * tz_scale
        
        x = x - px
        y = y - py
        tx = (x * tz) / fx
        ty = (y * tz) / fy
        
        return torch.stack([tx, ty, tz], dim=-1)

def bbox_transform_inv(boxes, deltas, scale_factors=None):
    """
    Reconstructs 2D boxes from anchors and regression deltas.
    boxes: (..., 4) in (x1, y1, x2, y2)
    deltas: (..., 4) [ty, tx, th, tw]
    """
    cxa = (boxes[..., 0] + boxes[..., 2]) / 2.0
    cya = (boxes[..., 1] + boxes[..., 3]) / 2.0
    wa = boxes[..., 2] - boxes[..., 0]
    ha = boxes[..., 3] - boxes[..., 1]
    ty = deltas[..., 0]
    tx = deltas[..., 1]
    th = deltas[..., 2]
    tw = deltas[..., 3]
    if scale_factors is not None:
        ty *= scale_factors[0]
        tx *= scale_factors[1]
        th *= scale_factors[2]
        tw *= scale_factors[3]
    w = torch.exp(tw) * wa
    h = torch.exp(th) * ha
    cy = ty * ha + cya
    cx = tx * wa + cxa
    ymin = cy - h / 2.0
    xmin = cx - w / 2.0
    ymax = cy + h / 2.0
    xmax = cx + w / 2.0
    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)

def translation_transform_inv(translation_anchors, deltas, scale_factors=None):
    """
    Applies regression deltas to translation anchors.
    translation_anchors: (num_boxes, 3)
    deltas: (num_boxes, 3)
    """
    stride = translation_anchors[..., -1]
    if scale_factors is not None:
        x = translation_anchors[..., 0] + (deltas[..., 0] * scale_factors[0] * stride)
        y = translation_anchors[..., 1] + (deltas[..., 1] * scale_factors[1] * stride)
    else:
        x = translation_anchors[..., 0] + (deltas[..., 0] * stride)
        y = translation_anchors[..., 1] + (deltas[..., 1] * stride)
    Tz = deltas[..., 2]
    return torch.stack([x, y, Tz], dim=-1)

def filter_detections(boxes, classification, rotation, translation,
                      num_rotation_parameters, num_translation_parameters=3,
                      class_specific_filter=True, nms=True, score_threshold=0.01,
                      max_detections=100, nms_threshold=0.5):
    """
    Filters detections using a score threshold and (optionally) NMS.
    """
    device = boxes.device
    if class_specific_filter:
        all_indices = []
        num_classes = classification.shape[1]
        for c in range(num_classes):
            scores = classification[:, c]
            indices = (scores > score_threshold).nonzero(as_tuple=False).squeeze(1)
            if indices.numel() == 0:
                continue
            filtered_boxes = boxes[indices]
            filtered_scores = scores[indices]
            if nms:
                keep = ops.nms(filtered_boxes, filtered_scores, nms_threshold)
                indices = indices[keep]
            all_indices.append((indices, torch.full((indices.shape[0],), c, dtype=torch.int64, device=device)))
        if len(all_indices) == 0:
            # Return empty (padded) tensors if no detection passes the threshold.
            boxes_out = -torch.ones(max_detections, 4, device=device)
            scores_out = -torch.ones(max_detections, device=device)
            labels_out = -torch.ones(max_detections, dtype=torch.int64, device=device)
            rotation_out = -torch.ones(max_detections, num_rotation_parameters, device=device)
            translation_out = -torch.ones(max_detections, num_translation_parameters, device=device)
            return boxes_out, scores_out, labels_out, rotation_out, translation_out

        indices_cat = torch.cat([item[0] for item in all_indices])
        labels_cat = torch.cat([item[1] for item in all_indices])
        scores_cat = classification[indices_cat, labels_cat]
        if scores_cat.numel() > max_detections:
            topk_scores, topk_idx = torch.topk(scores_cat, max_detections)
            indices_cat = indices_cat[topk_idx]
            labels_cat = labels_cat[topk_idx]
            scores_cat = topk_scores
        boxes_out = boxes[indices_cat]
        rotation_out = rotation[indices_cat]
        translation_out = translation[indices_cat]
        num_dets = scores_cat.shape[0]
        if num_dets < max_detections:
            pad = max_detections - num_dets
            boxes_out = torch.cat([boxes_out, -torch.ones(pad, 4, device=device)], dim=0)
            scores_cat = torch.cat([scores_cat, -torch.ones(pad, device=device)], dim=0)
            labels_cat = torch.cat([labels_cat, -torch.ones(pad, dtype=torch.int64, device=device)], dim=0)
            rotation_out = torch.cat([rotation_out, -torch.ones(pad, num_rotation_parameters, device=device)], dim=0)
            translation_out = torch.cat([translation_out, -torch.ones(pad, num_translation_parameters, device=device)], dim=0)
        return boxes_out, scores_cat, labels_cat, rotation_out, translation_out
    else:
        scores, labels = torch.max(classification, dim=1)
        indices = (scores > score_threshold).nonzero(as_tuple=False).squeeze(1)
        if indices.numel() == 0:
            boxes_out = -torch.ones(max_detections, 4, device=device)
            scores_out = -torch.ones(max_detections, device=device)
            labels_out = -torch.ones(max_detections, dtype=torch.int64, device=device)
            rotation_out = -torch.ones(max_detections, num_rotation_parameters, device=device)
            translation_out = -torch.ones(max_detections, num_translation_parameters, device=device)
            return boxes_out, scores_out, labels_out, rotation_out, translation_out
        filtered_boxes = boxes[indices]
        filtered_scores = scores[indices]
        if nms:
            keep = ops.nms(filtered_boxes, filtered_scores, nms_threshold)
            indices = indices[keep]
        scores = classification[indices, labels[indices]]
        if indices.numel() > max_detections:
            topk_scores, topk_idx = torch.topk(scores, max_detections)
            indices = indices[topk_idx]
            labels = labels[topk_idx]
            scores = topk_scores
        boxes_out = boxes[indices]
        rotation_out = rotation[indices]
        translation_out = translation[indices]
        num_dets = scores.shape[0]
        if num_dets < max_detections:
            pad = max_detections - num_dets
            boxes_out = torch.cat([boxes_out, -torch.ones(pad, 4, device=device)], dim=0)
            scores = torch.cat([scores, -torch.ones(pad, device=device)], dim=0)
            labels = torch.cat([labels, -torch.ones(pad, dtype=torch.int64, device=device)], dim=0)
            rotation_out = torch.cat([rotation_out, -torch.ones(pad, num_rotation_parameters, device=device)], dim=0)
            translation_out = torch.cat([translation_out, -torch.ones(pad, num_translation_parameters, device=device)], dim=0)
        return boxes_out, scores, labels, rotation_out, translation_out

class FilterDetections(nn.Module):
    """
    Wraps the filter_detections function.
    """
    def __init__(self, num_rotation_parameters, num_translation_parameters=3, nms=True,
                 class_specific_filter=True, nms_threshold=0.5, score_threshold=0.01,
                 max_detections=100):
        super(FilterDetections, self).__init__()
        self.num_rotation_parameters = num_rotation_parameters
        self.num_translation_parameters = num_translation_parameters
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections

    def forward(self, boxes, classification, rotation, translation):
        return filter_detections(boxes, classification, rotation, translation,
                                 self.num_rotation_parameters, self.num_translation_parameters,
                                 self.class_specific_filter, self.nms, self.score_threshold,
                                 self.max_detections, self.nms_threshold)

#########################################
# Dummy Anchor Generator
#########################################

def generate_anchors(image_size, num_boxes, anchor_parameters=None):
    """
    Dummy function returning random anchors.
    Replace with your actual anchor generation logic.
    """
    # You would use anchor_parameters here if provided
    # For now, just return random anchors
    anchors = torch.rand(num_boxes, 4) * image_size  # (num_boxes, 4)
    translation_anchors = torch.rand(num_boxes, 3) * image_size  # (num_boxes, 3)
    return anchors, translation_anchors

#########################################
# Main EfficientPose Model
#########################################

def build_EfficientPose(phi,
                        num_classes=8,
                        num_anchors=9,
                        freeze_bn=False,
                        score_threshold=0.5,
                        anchor_parameters=None,
                        num_rotation_parameters=3,
                        print_architecture=True):
    """
    Builds an EfficientPose model in PyTorch
    Args:
        phi: EfficientPose scaling hyperparameter phi
        num_classes: Number of classes,
        num_anchors: The number of anchors, usually 3 scales and 3 aspect ratios resulting in 3 * 3 = 9 anchors
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
        score_threshold: Minimum score threshold at which a prediction is not filtered out
        anchor_parameters: Struct containing anchor parameters. If None, default values are used.
        num_rotation_parameters: Number of rotation parameters, e.g. 3 for axis angle representation
        print_architecture: Boolean indicating if the model architecture should be printed or not
    
    Returns:
        efficientpose_train: EfficientPose model without NMS used for training
        efficientpose_prediction: EfficientPose model including NMS used for evaluating and inferencing
        all_modules: List of all modules in the EfficientPose model to load weights
    """
    # select parameters according to the given phi
    assert phi in range(7)
    scaled_parameters = get_scaled_parameters(phi)
    
    input_size = scaled_parameters["input_size"]
    bifpn_width = subnet_width = scaled_parameters["bifpn_width"]
    bifpn_depth = scaled_parameters["bifpn_depth"]
    subnet_depth = scaled_parameters["subnet_depth"]
    subnet_num_iteration_steps = scaled_parameters["subnet_num_iteration_steps"]
    num_groups_gn = scaled_parameters.get("num_groups_gn", 32)  # Default if not present
    backbone_fn = scaled_parameters["backbone_class"]
    
    # Create the models
    class EfficientPoseBase(nn.Module):
        def __init__(self):
            super(EfficientPoseBase, self).__init__()
            # Backbone
            backbone_model = backbone_fn(pretrained=True)
            self.backbone = backbone_model.features
            
            # Feature splitter to prepare for BiFPN
            self.feature_splitter = FeatureSplitter(bifpn_width)
            
            # BiFPN layers
            self.bifpn_layers = nn.ModuleList([
                BiFPNLayer(bifpn_width) for _ in range(bifpn_depth)
            ])
            
            # Subnets
            self.box_net = BoxNet(
                bifpn_width, num_anchors, subnet_depth, bifpn_width
            )
            self.class_net = ClassNet(
                bifpn_width, num_classes, num_anchors, subnet_depth, bifpn_width
            )
            self.rotation_net = RotationNet(
                bifpn_width, num_rotation_parameters, num_anchors,
                subnet_depth, bifpn_width, subnet_num_iteration_steps
            )
            self.translation_net = TranslationNet(
                bifpn_width, num_anchors, subnet_depth, 
                bifpn_width, subnet_num_iteration_steps
            )
            
            # Post-processing modules
            self.regress_boxes = RegressBoxes()
            self.clip_boxes = ClipBoxes()
            self.regress_translation = RegressTranslation()
            self.calculate_tx_ty = CalculateTxTy()
            
            # Anchor generation (would be defined elsewhere)
            self.generate_anchors = lambda size: generate_anchors(size, num_anchors)
        
        def extract_features(self, images):
            # Get backbone features
            backbone_features = self.backbone(images)
            
            # Split to multi-scale feature maps
            feature_maps = self.feature_splitter(backbone_features)
            
            # Pass through BiFPN
            for bifpn in self.bifpn_layers:
                feature_maps = bifpn(feature_maps)
                
            return feature_maps
        
        def apply_subnets(self, feature_maps, images, camera_params):
            # Apply subnet to each feature map
            box_outputs = []
            class_outputs = []
            rotation_outputs = []
            translation_outputs = []
            
            for feature_map in feature_maps:
                box_outputs.append(self.box_net(feature_map))
                class_outputs.append(self.class_net(feature_map))
                rotation_outputs.append(self.rotation_net(feature_map))
                translation_outputs.append(self.translation_net(feature_map))
            
            # Concatenate predictions from all feature levels
            boxes = torch.cat(box_outputs, dim=1)
            classifications = torch.cat(class_outputs, dim=1)
            rotations = torch.cat(rotation_outputs, dim=1)
            translations = torch.cat(translation_outputs, dim=1)
            
            # Generate anchors (this would be implemented elsewhere)
            batch_size = images.shape[0]
            image_shape = images.shape[2:]  # Height, Width
            anchors, translation_anchors = self.generate_anchors(input_size)
            anchors = anchors.to(images.device)
            translation_anchors = translation_anchors.to(images.device)
            
            # Apply regression to anchors
            boxes_regressed = self.regress_boxes(anchors, boxes)
            boxes_clipped = self.clip_boxes(images, boxes_regressed)
            
            translation_regressed = self.regress_translation(translation_anchors, translations)
            translation_final = self.calculate_tx_ty(translation_regressed, camera_params)
            
            # Combine rotation and translation for loss calculation
            transformation = torch.cat([rotations, translation_final], dim=-1)
            
            return classifications, boxes, rotations, translation_final, transformation, boxes_clipped
    
    # Training model without NMS
    class EfficientPoseTrain(EfficientPoseBase):
        def forward(self, inputs):
            images, camera_params = inputs
            
            feature_maps = self.extract_features(images)
            classifications, boxes, rotations, translations, transformation, boxes_clipped = self.apply_subnets(
                feature_maps, images, camera_params
            )
            
            # Return outputs in the same order as the TF model
            return classifications, boxes, transformation
    
    # Prediction model with NMS
    class EfficientPosePrediction(EfficientPoseBase):
        def __init__(self):
            super(EfficientPosePrediction, self).__init__()
            self.filter_detections = FilterDetections(
                num_rotation_parameters=num_rotation_parameters,
                num_translation_parameters=3,
                score_threshold=score_threshold
            )
        
        def forward(self, inputs):
            images, camera_params = inputs
            
            feature_maps = self.extract_features(images)
            classifications, boxes, rotations, translations, _, boxes_clipped = self.apply_subnets(
                feature_maps, images, camera_params
            )
            
            # Apply NMS filtering
            filtered_detections = self.filter_detections(
                boxes_clipped, classifications, rotations, translations
            )
            
            return filtered_detections
    
    # Create the two model variants
    efficientpose_train = EfficientPoseTrain()
    efficientpose_prediction = EfficientPosePrediction()
    
    # Create list of all modules for weight loading
    all_modules = []
    for model in [efficientpose_train, efficientpose_prediction]:
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                all_modules.append(module)
    
    if print_architecture:
        # Implementation of architecture printing function
        print(f"EfficientPose-{phi} Architecture:")
        print(efficientpose_train)
        print("\nBox Net:")
        print(efficientpose_train.box_net)
        print("\nClass Net:")
        print(efficientpose_train.class_net)
        print("\nRotation Net:")
        print(efficientpose_train.rotation_net)
        print("\nTranslation Net:")
        print(efficientpose_train.translation_net)
    
    return efficientpose_train, efficientpose_prediction, all_modules
#########################################
# Example usage / Testing
#########################################

if __name__ == '__main__':
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a dummy input image (batch size 1, 3 channels, 512x512) and dummy camera parameters.
    dummy_image = torch.randn(1, 3, 512, 512).to(device)
    dummy_camera_params = torch.randn(1, 6).to(device)

    # Instantiate the EfficientPose models
    efficientpose_train, efficientpose_prediction, all_modules = build_EfficientPose(
        phi=0, num_classes=8, num_anchors=9, num_rotation_parameters=3
    )
    
    # Move models to device
    efficientpose_train = efficientpose_train.to(device)
    efficientpose_prediction = efficientpose_prediction.to(device)
    
    # Set to evaluation mode
    efficientpose_prediction.eval()

    # Run inference (with torch.no_grad() to avoid tracking gradients).
    with torch.no_grad():
        # Create a tuple of inputs (image, camera params)
        inputs = (dummy_image, dummy_camera_params)
        filtered_detections = efficientpose_prediction(inputs)
        
    # Unpack the filtered detections
    boxes, scores, labels, rotations, translations = filtered_detections
    
    # Print out the detection results
    print("Final Detection Scores:")
    print(scores)
    print("Final Boxes:")
    print(boxes)
    print("Final Labels:")
    print(labels)
    print("Final Rotations:")
    print(rotations)
    print("Final Translations:")
    print(translations)

    # Also print shapes to verify dimensions
    print("\nOutput Shapes:")
    print("Scores:", scores.shape)
    print("Boxes:", boxes.shape)
    print("Labels:", labels.shape)
    print("Rotations:", rotations.shape)
    print("Translations:", translations.shape)
