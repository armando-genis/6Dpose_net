import torch


#####################################
# Anchor Parameters (Torch Version)
#####################################
class AnchorParameters:
    """
    Defines how anchors are generated.

    Args:
        sizes: List of sizes (one per feature level).
        strides: List of strides (one per feature level).
        ratios: Tuple/list of aspect ratios per location.
        scales: Tuple/list of scales per location.
    """
    def __init__(self, sizes=(32, 64, 128, 256, 512),
                 strides=(8, 16, 32, 64, 128),
                 ratios=(1, 0.5, 2),
                 scales=(2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.))):
        self.sizes = sizes
        self.strides = strides
        # Store as torch tensors (dtype float32)
        self.ratios = torch.tensor(ratios, dtype=torch.float32)
        self.scales = torch.tensor(scales, dtype=torch.float32)
    
    def num_anchors(self):
        return len(self.ratios) * len(self.scales)

# Default anchor parameters
AnchorParameters.default = AnchorParameters(
    sizes=[32, 64, 128, 256, 512],
    strides=[8, 16, 32, 64, 128],
    ratios=(1, 0.5, 2),
    scales=(2 ** 0, 2 ** (1.0/3.0), 2 ** (2.0/3.0))
)

#####################################
# Helper Functions
#####################################
def guess_shapes(image_shape, pyramid_levels):
    """
    Given an image shape and pyramid levels, guess the feature map shapes.
    
    Args:
        image_shape: Tuple (H, W, ...) or (H, W). Only the first two dimensions are used.
        pyramid_levels: List of ints (e.g. [3, 4, 5, 6, 7]).
    
    Returns:
        List of tuples (H_level, W_level) for each pyramid level.
    """
    # Use only the first two dimensions
    H, W = image_shape[:2]
    shapes = []
    for level in pyramid_levels:
        divisor = 2 ** level
        shapes.append(((H + divisor - 1) // divisor, (W + divisor - 1) // divisor))
    return shapes

def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate reference anchors by enumerating aspect ratios and scales w.r.t. a reference window.

    Args:
        base_size: The base size of the anchor.
        ratios: A torch tensor of aspect ratios.
        scales: A torch tensor of scales.
    
    Returns:
        anchors: A torch tensor of shape (num_anchors, 4) in (x1, y1, x2, y2) format.
    """
    if ratios is None:
        ratios = AnchorParameters.default.ratios
    if scales is None:
        scales = AnchorParameters.default.scales

    num_anchors = len(ratios) * len(scales)
    anchors = torch.zeros((num_anchors, 4), dtype=torch.float32)

    # First, compute the widths and heights for each combination.
    # Repeat each scale for each ratio.
    scales_repeated = scales.repeat_interleave(len(ratios))  # shape: (num_anchors,)
    # Compute preliminary widths and heights.
    anchors[:, 2] = base_size * scales_repeated
    anchors[:, 3] = base_size * scales_repeated
    areas = anchors[:, 2] * anchors[:, 3]
    
    # Tile the ratios (repeat the list once per scale)
    tiled_ratios = torch.tensor(ratios, dtype=torch.float32).repeat(len(scales))
    # Adjust widths and heights to match the desired aspect ratios.
    anchors[:, 2] = torch.sqrt(areas / tiled_ratios)
    anchors[:, 3] = anchors[:, 2] * tiled_ratios

    # Center the anchors at (0,0): compute x1,y1 and x2,y2.
    anchors[:, 0] = -0.5 * anchors[:, 2]
    anchors[:, 1] = -0.5 * anchors[:, 3]
    anchors[:, 2] = anchors[:, 0] + anchors[:, 2]
    anchors[:, 3] = anchors[:, 1] + anchors[:, 3]
    return anchors

def shift(feature_map_shape, stride, anchors):
    """
    Shift the base anchors over the entire feature map.

    Args:
        feature_map_shape: Tuple (H, W) of the feature map.
        stride: The stride of the feature map relative to the image.
        anchors: Base anchors (torch.Tensor of shape (A, 4)).
    
    Returns:
        all_anchors: A torch.Tensor of shape (K*A, 4) containing shifted anchors.
    """
    device = anchors.device
    H, W = feature_map_shape
    # Create grid of center positions (using 0.5 offset)
    shift_x = (torch.arange(0, W, device=device, dtype=torch.float32) + 0.5) * stride
    shift_y = (torch.arange(0, H, device=device, dtype=torch.float32) + 0.5) * stride
    # Use torch.meshgrid; note: indexing='ij' if supported in your version.
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    # Create shifts tensor of shape (K, 4)
    shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
    A = anchors.shape[0]
    K = shifts.shape[0]
    # Add shifts to anchors
    all_anchors = anchors.unsqueeze(0) + shifts.unsqueeze(1)  # shape: (K, A, 4)
    all_anchors = all_anchors.reshape(K * A, 4)
    return all_anchors

def translation_shift(feature_map_shape, stride, translation_anchors):
    """
    Shift the base translation anchors over the feature map.
    
    Args:
        feature_map_shape: Tuple (H, W) of the feature map.
        stride: The stride of the feature map.
        translation_anchors: Base translation anchors (torch.Tensor of shape (A, 2)).
    
    Returns:
        all_translation_anchors: A torch.Tensor of shape (K*A, 3) containing (x, y, stride).
    """
    device = translation_anchors.device
    H, W = feature_map_shape
    shift_x = (torch.arange(0, W, device=device, dtype=torch.float32) + 0.5) * stride
    shift_y = (torch.arange(0, H, device=device, dtype=torch.float32) + 0.5) * stride
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    shifts = torch.stack([shift_x, shift_y], dim=1)  # shape: (K, 2)
    A = translation_anchors.shape[0]
    K = shifts.shape[0]
    all_translation = translation_anchors.unsqueeze(0) + shifts.unsqueeze(1)  # shape: (K, A, 2)
    all_translation = all_translation.reshape(K * A, 2)
    # Append stride to each anchor, resulting in shape (K*A, 3)
    stride_tensor = torch.full((all_translation.shape[0], 1), stride, dtype=torch.float32, device=device)
    all_translation_anchors = torch.cat([all_translation, stride_tensor], dim=1)
    return all_translation_anchors

def anchors_for_shape(image_shape, pyramid_levels=None, anchor_params=None, shapes_callback=None):
    """
    Generate anchors for a given image shape using only PyTorch.
    
    Args:
        image_shape: Tuple, e.g. (height, width, channels) or (height, width). Only the first two dimensions are used.
        pyramid_levels: List of ints representing which pyramid levels to use (default: [3, 4, 5, 6, 7]).
        anchor_params: An instance of AnchorParameters (if None, default parameters are used).
        shapes_callback: Function to compute feature map shapes; if None, uses guess_shapes.
        
    Returns:
        anchors: A torch.Tensor of shape (N, 4) containing (x1, y1, x2, y2) for all anchors.
        translation_anchors: A torch.Tensor of shape (N, 3) containing (x, y, stride) for all anchors.
    """
    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]
    if anchor_params is None:
        anchor_params = AnchorParameters.default
    if shapes_callback is None:
        shapes_callback = guess_shapes

    # Compute feature map shapes for each pyramid level (using Python arithmetic)
    image_shapes = guess_shapes(image_shape, pyramid_levels)

    all_anchors = []
    all_translation_anchors = []
    for idx, level in enumerate(pyramid_levels):
        base_size = anchor_params.sizes[idx]
        # Generate base anchors (in (x1, y1, x2, y2) format)
        anchors = generate_anchors(base_size=base_size, ratios=anchor_params.ratios, scales=anchor_params.scales)
        # Create base translation anchors: zeros with shape (A, 2)
        num_anchors = anchors.shape[0]
        translation_anchors = torch.zeros((num_anchors, 2), dtype=torch.float32)
        stride = anchor_params.strides[idx]
        shifted_anchors = shift(image_shapes[idx], stride, anchors)
        shifted_translation_anchors = translation_shift(image_shapes[idx], stride, translation_anchors)
        all_anchors.append(shifted_anchors)
        all_translation_anchors.append(shifted_translation_anchors)
    
    # Concatenate results along the first dimension.
    all_anchors = torch.cat(all_anchors, dim=0)
    all_translation_anchors = torch.cat(all_translation_anchors, dim=0)
    return all_anchors, all_translation_anchors

# Example usage:
if __name__ == '__main__':
    # Example image shape (height, width, channels)
    image_shape = (512, 512, 3)
    anchors, translation_anchors = anchors_for_shape(image_shape)
    print("Anchors shape:", anchors.shape)                # e.g. (N, 4)
    print("Translation anchors shape:", translation_anchors.shape)  # e.g. (N, 3)
