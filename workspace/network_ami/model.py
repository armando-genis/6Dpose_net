import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.ops as ops

#########################################
# Utility and helper functions/modules
#########################################

def get_scaled_parameters(phi):
    """
    Returns scaled parameters for a given phi.
    """
    image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
    bifpn_widths = [64, 88, 112, 160, 224, 288, 384]
    bifpn_depths = [3, 4, 5, 6, 7, 7, 8]
    subnet_depths = [3, 3, 3, 4, 4, 4, 5]
    subnet_iteration_steps = [1, 1, 1, 2, 2, 2, 3]
    parameters = {
        "input_size": image_sizes[phi],
        "bifpn_width": bifpn_widths[phi],
        "bifpn_depth": bifpn_depths[phi],
        "subnet_depth": subnet_depths[phi],
        "subnet_num_iteration_steps": subnet_iteration_steps[phi]
    }
    return parameters


class SeparableConvBlock(nn.Module):
    """
    Implements a depthwise separable convolution block with BatchNorm and Swish (SiLU).
    """
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
        x = F.silu(x)  # Swish activation
        return x


class FeatureSplitter(nn.Module):
    """
    Converts the backbone's feature map into multiple scales with the desired channel dimension.
    """
    def __init__(self, out_channels):
        super(FeatureSplitter, self).__init__()
        # EfficientNet-B0 outputs 1280 channels; reduce them to out_channels.
        self.resample = nn.Conv2d(1280, out_channels, kernel_size=1)
        
    def forward(self, features):
        # features: [N, 1280, H, W]
        x = self.resample(features)  # now x is [N, out_channels, H, W]
        P3 = x
        P4 = F.max_pool2d(x, kernel_size=2)
        P5 = F.max_pool2d(P4, kernel_size=2)
        P6 = F.max_pool2d(P5, kernel_size=2)
        P7 = F.max_pool2d(P6, kernel_size=2)
        return [P3, P4, P5, P6, P7]

#########################################
# BiFPN
#########################################

class BiFPNLayer(nn.Module):
    """
    A simplified version of a single BiFPN layer.
    It takes a list of feature maps and fuses them via top-down and bottom-up paths.
    """
    def __init__(self, num_channels):
        super(BiFPNLayer, self).__init__()
        self.conv_P3 = SeparableConvBlock(num_channels, num_channels)
        self.conv_P4 = SeparableConvBlock(num_channels, num_channels)
        self.conv_P5 = SeparableConvBlock(num_channels, num_channels)
        self.conv_P6 = SeparableConvBlock(num_channels, num_channels)
        self.conv_P7 = SeparableConvBlock(num_channels, num_channels)

    def forward(self, features):
        # features = [P3, P4, P5, P6, P7]
        P3, P4, P5, P6, P7 = features
        P7_td = self.conv_P7(P7)
        P6_td = self.conv_P6(P6) + F.interpolate(P7_td, scale_factor=2, mode='nearest')
        P5_td = self.conv_P5(P5) + F.interpolate(P6_td, scale_factor=2, mode='nearest')
        P4_td = self.conv_P4(P4) + F.interpolate(P5_td, scale_factor=2, mode='nearest')
        P3_td = self.conv_P3(P3) + F.interpolate(P4_td, scale_factor=2, mode='nearest')
        return [P3_td, P4_td, P5_td, P6_td, P7_td]

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
    """
    Rotation estimation subnetwork with iterative refinement.
    Predicts rotation parameters for each anchor.
    """
    def __init__(self, in_channels, num_values, num_anchors, depth, width, num_iteration_steps):
        super(RotationNet, self).__init__()
        self.num_iteration_steps = num_iteration_steps
        self.num_values = num_values
        layers_list = []
        for i in range(depth):
            inc = in_channels if i == 0 else width
            layers_list.append(SeparableConvBlock(inc, width))
        self.conv_layers = nn.Sequential(*layers_list)
        self.initial_conv = nn.Conv2d(width, num_anchors * num_values, kernel_size=3, padding=1)
        self.iterative_conv = nn.Conv2d(width + num_anchors * num_values,
                                        num_anchors * num_values, kernel_size=3, padding=1)
        self.num_anchors = num_anchors

    def forward(self, x):
        features = self.conv_layers(x)
        rotation = self.initial_conv(features)
        for i in range(self.num_iteration_steps):
            iterative_input = torch.cat([features, rotation], dim=1)
            delta = self.iterative_conv(iterative_input)
            rotation = rotation + delta
        N = rotation.shape[0]
        rotation = rotation.permute(0, 2, 3, 1).contiguous().view(N, -1, self.num_values)
        return rotation


class TranslationNet(nn.Module):
    """
    Translation estimation subnetwork with iterative refinement.
    Predicts 3 values per anchor (e.g., x, y, and z offsets).
    """
    def __init__(self, in_channels, num_anchors, depth, width, num_iteration_steps):
        super(TranslationNet, self).__init__()
        self.num_iteration_steps = num_iteration_steps
        layers_list = []
        for i in range(depth):
            inc = in_channels if i == 0 else width
            layers_list.append(SeparableConvBlock(inc, width))
        self.conv_layers = nn.Sequential(*layers_list)
        self.initial_conv = nn.Conv2d(width, num_anchors * 3, kernel_size=3, padding=1)
        self.iterative_conv = nn.Conv2d(width + num_anchors * 3,
                                        num_anchors * 3, kernel_size=3, padding=1)
        self.num_anchors = num_anchors

    def forward(self, x):
        features = self.conv_layers(x)
        translation = self.initial_conv(features)
        for i in range(self.num_iteration_steps):
            iterative_input = torch.cat([features, translation], dim=1)
            delta = self.iterative_conv(iterative_input)
            translation = translation + delta
        N = translation.shape[0]
        translation = translation.permute(0, 2, 3, 1).contiguous().view(N, -1, 3)
        return translation

#########################################
# Custom Post-processing Layers/Functions
#########################################

# --- Begin Custom Layers Definitions ---

class ClipBoxes(nn.Module):
    """
    Clips 2D bounding boxes so that they are within the image.
    """
    def forward(self, image, boxes):
        # image: (N, C, H, W)
        _, _, H, W = image.shape
        x1 = boxes[..., 0].clamp(0, W - 1)
        y1 = boxes[..., 1].clamp(0, H - 1)
        x2 = boxes[..., 2].clamp(0, W - 1)
        y2 = boxes[..., 3].clamp(0, H - 1)
        return torch.stack([x1, y1, x2, y2], dim=-1)


class RegressBoxes(nn.Module):
    """
    Applies regression offsets to anchor boxes.
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
    Calculates Tx and Ty components based on camera intrinsics.
    """
    def __init__(self, fx=572.4114, fy=573.57043, px=325.2611, py=242.04899,
                 tz_scale=1000.0, image_scale=1.6666666666666667):
        super(CalculateTxTy, self).__init__()
        self.fx = fx
        self.fy = fy
        self.px = px
        self.py = py
        self.tz_scale = tz_scale
        self.image_scale = image_scale

    def forward(self, inputs):
        # inputs: (num_boxes, 3) [x, y, Tz]
        x = inputs[..., 0] / self.image_scale
        y = inputs[..., 1] / self.image_scale
        tz = inputs[..., 2] * self.tz_scale
        x = x - self.px
        y = y - self.py
        tx = (x * tz) / self.fx
        ty = (y * tz) / self.fy
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
    Filters detections using score threshold and NMS.
    boxes: (num_boxes, 4)
    classification: (num_boxes, num_classes)
    rotation: (num_boxes, num_rotation_parameters)
    translation: (num_boxes, num_translation_parameters)
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
            # Return empty (padded) tensors
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

# --- End Custom Layers Definitions ---

#########################################
# Dummy Anchor Generator
#########################################
def generate_anchors(image_size, num_boxes):
    """
    Dummy function that returns random anchors and translation anchors.
    In practice, replace with your anchor generation logic.
    """
    anchors = torch.rand(num_boxes, 4) * image_size  # (num_boxes, 4)
    translation_anchors = torch.rand(num_boxes, 3) * image_size  # (num_boxes, 3)
    return anchors, translation_anchors

#########################################
# Main EfficientPose model
#########################################

class EfficientPose(nn.Module):
    """
    Combines the backbone, BiFPN, and subnets.
    """
    def __init__(self, phi=0, num_classes=8, num_anchors=9, score_threshold=0.5, num_rotation_parameters=3):
        super(EfficientPose, self).__init__()
        # Get scaled parameters
        self.params = get_scaled_parameters(phi)
        self.input_size = self.params["input_size"]
        bifpn_width = self.params["bifpn_width"]
        bifpn_depth = self.params["bifpn_depth"]
        subnet_depth = self.params["subnet_depth"]
        subnet_num_iteration_steps = self.params["subnet_num_iteration_steps"]

        # Backbone using EfficientNet (using efficientnet_b0 as an example)
        if phi == 0:
            backbone_model = models.efficientnet_b0(pretrained=True)
        else:
            backbone_model = models.efficientnet_b0(pretrained=True)  # placeholder
        self.backbone = backbone_model.features

        # Feature splitter with channel resampling
        self.feature_splitter = FeatureSplitter(bifpn_width)

        # Stack BiFPN layers
        self.bifpn_layers = nn.ModuleList([BiFPNLayer(bifpn_width) for _ in range(bifpn_depth)])

        # Subnetworks
        self.box_net = BoxNet(bifpn_width, num_anchors, subnet_depth, bifpn_width)
        self.class_net = ClassNet(bifpn_width, num_classes, num_anchors, subnet_depth, bifpn_width)
        self.rotation_net = RotationNet(bifpn_width, num_rotation_parameters, num_anchors,
                                        subnet_depth, bifpn_width, subnet_num_iteration_steps)
        self.translation_net = TranslationNet(bifpn_width, num_anchors, subnet_depth, bifpn_width, subnet_num_iteration_steps)

        # Post-processing modules
        self.regress_boxes = RegressBoxes()          # Optionally, you can set scale_factors
        self.clip_boxes = ClipBoxes()
        self.regress_translation = RegressTranslation()  # Optionally, set scale_factors
        self.calculate_tx_ty = CalculateTxTy()
        self.filter_detections = FilterDetections(num_rotation_parameters=num_rotation_parameters)

        self.score_threshold = score_threshold

    def forward(self, image, camera_params=None):
        """
        image: [N, 3, H, W]
        camera_params: not used in this simplified example
        """
        # Backbone feature extraction
        features = self.backbone(image)
        # Split features into multiple scales
        feature_maps = self.feature_splitter(features)
        # Pass through stacked BiFPN layers
        for bifpn in self.bifpn_layers:
            feature_maps = bifpn(feature_maps)
        # Apply subnetworks
        box_outputs = [self.box_net(f) for f in feature_maps]
        class_outputs = [self.class_net(f) for f in feature_maps]
        rotation_outputs = [self.rotation_net(f) for f in feature_maps]
        translation_outputs = [self.translation_net(f) for f in feature_maps]

        # Concatenate predictions from all scales along the anchor dimension.
        boxes = torch.cat(box_outputs, dim=1)
        classifications = torch.cat(class_outputs, dim=1)
        rotations = torch.cat(rotation_outputs, dim=1)
        translations = torch.cat(translation_outputs, dim=1)

        # --- Post-processing ---
        # Generate dummy anchors (replace with your anchor generation)
        num_boxes = boxes.shape[1]
        anchors, translation_anchors = generate_anchors(self.input_size, num_boxes)
        anchors = anchors.to(image.device)
        translation_anchors = translation_anchors.to(image.device)

        # Apply regression deltas to anchors
        boxes_regressed = self.regress_boxes(anchors, boxes)
        boxes_clipped = self.clip_boxes(image, boxes_regressed)

        translation_regressed = self.regress_translation(translation_anchors, translations)
        translation_final = self.calculate_tx_ty(translation_regressed)

        # For filtering, typically this is done per image.
        final_boxes = []
        final_scores = []
        final_labels = []
        final_rotations = []
        final_translations = []
        batch_size = image.shape[0]
        for i in range(batch_size):
            b, s, l, r, t = self.filter_detections(
                boxes_clipped[i], classifications[i], rotations[i], translation_final[i]
            )
            final_boxes.append(b)
            final_scores.append(s)
            final_labels.append(l)
            final_rotations.append(r)
            final_translations.append(t)
        final_boxes = torch.stack(final_boxes, dim=0)
        final_scores = torch.stack(final_scores, dim=0)
        final_labels = torch.stack(final_labels, dim=0)
        final_rotations = torch.stack(final_rotations, dim=0)
        final_translations = torch.stack(final_translations, dim=0)

        return {
            'classifications': final_scores,  # final detection scores
            'boxes': final_boxes,               # final bounding boxes
            'rotations': final_rotations,       # final rotations
            'translations': final_translations  # final translations
        }

#########################################
# Example usage
#########################################

if __name__ == '__main__':
    import torch

    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a dummy input image (batch size 1, 3 channels, 512x512) and dummy camera parameters.
    dummy_image = torch.randn(1, 3, 512, 512).to(device)
    dummy_camera_params = torch.randn(1, 6).to(device)  # not used in this dummy

    # Instantiate the EfficientPose model with phi=0 and move it to the selected device.
    model = EfficientPose(phi=0, num_classes=8, num_anchors=9, num_rotation_parameters=3).to(device)
    model.eval()  # set to evaluation mode

    # Run inference (with torch.no_grad() to avoid tracking gradients).
    with torch.no_grad():
        outputs = model(dummy_image, dummy_camera_params)

    # Print out the detection results.
    print("Final Detection Scores (per image):")
    print(outputs['classifications'])
    print("Final Boxes (per image):")
    print(outputs['boxes'])
    print("Final Rotations (per image):")
    print(outputs['rotations'])
    print("Final Translations (per image):")
    print(outputs['translations'])

    # Also print shapes to verify dimensions.
    print("\nOutput Shapes:")
    print("Classifications:", outputs['classifications'].shape)
    print("Boxes:", outputs['boxes'].shape)
    print("Rotations:", outputs['rotations'].shape)
    print("Translations:", outputs['translations'].shape)
