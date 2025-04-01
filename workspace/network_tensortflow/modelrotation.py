from functools import reduce

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import models
from tensorflow.keras import backend
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6

from tensorflow.keras.initializers import Initializer
from initializers import PriorProbability

import numpy as np

# from files:
from layers import ClipBoxes, RegressBoxes, FilterDetections, EnhancedBiFPNAdd, wBiFPNAdd, BatchNormalization, RegressTranslation, CalculateTxTy, GroupNormalization, SpatialAttentionModule, RotationAttentionModule
from utils.anchors import anchors_for_shape

MOMENTUM = 0.997
EPSILON = 1e-4


def build_EfficientPose(phi,
                        num_classes = 8,
                        num_anchors = 9,
                        freeze_bn = False,
                        score_threshold = 0.5,
                        anchor_parameters = None,
                        num_rotation_parameters = 3,
                        print_architecture = True):

    #select parameters according to the given phi
    assert phi in range(7)
    scaled_parameters = get_scaled_parameters(phi)
    
    input_size = scaled_parameters["input_size"]
    input_shape = (input_size, input_size, 3)
    bifpn_width = subnet_width = scaled_parameters["bifpn_width"]
    bifpn_depth = scaled_parameters["bifpn_depth"]
    subnet_depth = scaled_parameters["subnet_depth"]
    subnet_num_iteration_steps = scaled_parameters["subnet_num_iteration_steps"]
    num_groups_gn = scaled_parameters["num_groups_gn"]
    backbone_class = scaled_parameters["backbone_class"]
    
    #input layers
    image_input = layers.Input(input_shape)
    camera_parameters_input = layers.Input((6,)) #camera parameters and image scale for calculating the translation vector from 2D x-, y-coordinates
    
    # Build EfficientNet backbone - adapted for TensorFlow 2.19
    # Create base model without the top classification layers
    base_model = backbone_class(include_top=False, 
                               weights='imagenet', 
                               input_tensor=image_input)
    
    # If freeze_bn is True, freeze batch normalization layers
    if freeze_bn:
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    
    # Extract feature maps based on the output names seen in the original implementation
    # Adjust these layer names based on TF 2.19's EfficientNet implementation
    C1 = base_model.get_layer('block1a_project_bn').output
    C2 = base_model.get_layer('block2b_add').output
    C3 = base_model.get_layer('block3b_add').output
    C4 = base_model.get_layer('block5c_add').output
    C5 = base_model.get_layer('block7a_project_bn').output
    
    backbone_feature_maps = [C1, C2, C3, C4, C5]
    
    print("EfficientNet feature maps:")
    for feature_map in backbone_feature_maps:
        print(feature_map)


    
    #build BiFPN
    fpn_feature_maps = build_BiFPN(backbone_feature_maps, bifpn_depth, bifpn_width, freeze_bn)

    if print_architecture:
        print("BiFPN feature maps:")
        for feature_map in fpn_feature_maps:
            print(feature_map)
            print()

    # print the type of the feature maps
    print("Type of feature maps:")
    for feature_map in fpn_feature_maps:
        print(type(feature_map))
        print()

    #build subnet
    class_net, box_net, rotation_net, translation_net  = build_subnets(num_classes,
                                                                      subnet_width,
                                                                      subnet_depth,
                                                                      subnet_num_iteration_steps,
                                                                      num_groups_gn,
                                                                      num_rotation_parameters,
                                                                      freeze_bn,
                                                                      num_anchors)
    
    classification, bbox_regression, rotation, translation, transformation, bboxes  = apply_subnets_to_feature_maps(box_net,
                                                                                                                    class_net,
                                                                                                                    rotation_net,
                                                                                                                    translation_net,
                                                                                                                    fpn_feature_maps,
                                                                                                                    image_input,
                                                                                                                    camera_parameters_input,
                                                                                                                    input_size,
                                                                                                                    anchor_parameters)
    # #get the EfficientPose model for training without NMS and the rotation and translation output combined in the transformation output because of the loss calculation
    efficientpose_train = models.Model(inputs = [image_input, camera_parameters_input], outputs = [classification, bbox_regression, transformation], name = 'efficientpose')

    filtered_detections = FilterDetections(num_rotation_parameters = num_rotation_parameters,
                                           num_translation_parameters = 3,
                                           name = 'filtered_detections',
                                           score_threshold = score_threshold
                                           )([bboxes, classification, rotation, translation])
    
    print("filtered detections:")
    print(filtered_detections)
    print()

    efficientpose_prediction = models.Model(inputs = [image_input, camera_parameters_input], outputs = filtered_detections, name = 'efficientpose_prediction')

    if print_architecture:
        print_models(efficientpose_train, box_net, class_net, rotation_net, translation_net)

    all_layers = list(set(efficientpose_train.layers + box_net.layers + class_net.layers + rotation_net.layers + translation_net.layers))

    return efficientpose_train, efficientpose_prediction, all_layers

def print_models(*models):
    """
    Print the model architectures
    Args:
        *models: Tuple containing all models that should be printed
    """
    for model in models:
        print("\n\n")
        model.summary()
        print("\n\n")

def get_scaled_parameters(phi):
    """
    Returns a dictionary of scaled parameters according to the phi value.
    """
    # Info tuples with scalable parameters
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    bifpn_widths = (64, 88, 112, 160, 224, 288, 384)
    bifpn_depths = (3, 4, 5, 6, 7, 7, 8)
    subnet_depths = (3, 3, 3, 4, 4, 4, 5)
    subnet_iteration_steps = (1, 1, 1, 2, 2, 2, 3)
    num_groups_gn = (4, 4, 7, 10, 14, 18, 24) # Try to get 16 channels per group
    backbones = (EfficientNetB0,
                 EfficientNetB1,
                 EfficientNetB2,
                 EfficientNetB3,
                 EfficientNetB4,
                 EfficientNetB5,
                 EfficientNetB6)
    
    parameters = {"input_size": image_sizes[phi],
                  "bifpn_width": bifpn_widths[phi],
                  "bifpn_depth": bifpn_depths[phi],
                  "subnet_depth": subnet_depths[phi],
                  "subnet_num_iteration_steps": subnet_iteration_steps[phi],
                  "num_groups_gn": num_groups_gn[phi],
                  "backbone_class": backbones[phi]}
    
    return parameters

def build_BiFPN(backbone_feature_maps, bifpn_depth, bifpn_width, freeze_bn):
    """
    Building the bidirectional feature pyramid as described in https://arxiv.org/abs/1911.09070
    Args:
        backbone_feature_maps: Sequence containing the EfficientNet backbone feature maps of the different levels (C1, C2, C3, C4, C5)
        bifpn_depth: Number of BiFPN layers
        bifpn_width: Number of channels used in the BiFPN
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
    
    Returns:
       fpn_feature_maps: Sequence of BiFPN layers of the different levels (P3, P4, P5, P6, P7)
    """
    fpn_feature_maps = backbone_feature_maps
    for i in range(bifpn_depth):
        fpn_feature_maps = build_BiFPN_layer(fpn_feature_maps, bifpn_width, i, freeze_bn=freeze_bn)
        
    return fpn_feature_maps


def build_BiFPN_layer(features, num_channels, idx_BiFPN_layer, freeze_bn=False):
    """
    Builds a single layer of the bidirectional feature pyramid
    Args:
        features: Sequence containing the feature maps of the previous BiFPN layer (P3, P4, P5, P6, P7) or the EfficientNet backbone feature maps of the different levels (C1, C2, C3, C4, C5)
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
    
    Returns:
       BiFPN layers of the different levels (P3, P4, P5, P6, P7)
    """
    if idx_BiFPN_layer == 0:
        _, _, C3, C4, C5 = features
        P3_in, P4_in_1, P4_in_2, P5_in_1, P5_in_2, P6_in, P7_in = prepare_feature_maps_for_BiFPN(C3, C4, C5, num_channels, freeze_bn)
    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        
    # Top down pathway
    input_feature_maps_top_down = [P7_in,
                                   P6_in,
                                   P5_in_1 if idx_BiFPN_layer == 0 else P5_in,
                                   P4_in_1 if idx_BiFPN_layer == 0 else P4_in,
                                   P3_in]
    
    P7_in, P6_td, P5_td, P4_td, P3_out = top_down_pathway_BiFPN(input_feature_maps_top_down, num_channels, idx_BiFPN_layer)
    
    # Bottom up pathway
    input_feature_maps_bottom_up = [[P3_out],
                                    [P4_in_2 if idx_BiFPN_layer == 0 else P4_in, P4_td],
                                    [P5_in_2 if idx_BiFPN_layer == 0 else P5_in, P5_td],
                                    [P6_in, P6_td],
                                    [P7_in]]
    
    P3_out, P4_out, P5_out, P6_out, P7_out = bottom_up_pathway_BiFPN(input_feature_maps_bottom_up, num_channels, idx_BiFPN_layer)
    
    # Note: The original implementation returns P3_out, P4_td, P5_td, P6_td, P7_out
    # This might be a bug as mentioned in the original comment, but keeping it for compatibility
    return P3_out, P4_td, P5_td, P6_td, P7_out


def top_down_pathway_BiFPN(input_feature_maps_top_down, num_channels, idx_BiFPN_layer):
    """
    Computes the top-down-pathway in a single BiFPN layer
    Args:
        input_feature_maps_top_down: Sequence containing the input feature maps of the BiFPN layer (P3, P4, P5, P6, P7)
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
    
    Returns:
       Sequence with the output feature maps of the top-down-pathway
    """
    feature_map_P7 = input_feature_maps_top_down[0]
    output_top_down_feature_maps = [feature_map_P7]
    
    for level in range(1, 5):
        merged_feature_map = single_BiFPN_merge_step(
            feature_map_other_level=output_top_down_feature_maps[-1],
            feature_maps_current_level=[input_feature_maps_top_down[level]],
            upsampling=True,
            num_channels=num_channels,
            idx_BiFPN_layer=idx_BiFPN_layer,
            node_idx=level - 1,
            op_idx=4 + level, 
            use_attention=True
        )
        
        output_top_down_feature_maps.append(merged_feature_map)
        
    return output_top_down_feature_maps

def bottom_up_pathway_BiFPN(input_feature_maps_bottom_up, num_channels, idx_BiFPN_layer):
    """
    Computes the bottom-up-pathway in a single BiFPN layer
    Args:
        input_feature_maps_bottom_up: Sequence containing a list of feature maps serving as input for each level of the BiFPN layer (P3, P4, P5, P6, P7)
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
    
    Returns:
       Sequence with the output feature maps of the bottom-up-pathway
    """
    feature_map_P3 = input_feature_maps_bottom_up[0][0]
    output_bottom_up_feature_maps = [feature_map_P3]
    
    for level in range(1, 5):
        merged_feature_map = single_BiFPN_merge_step(
            feature_map_other_level=output_bottom_up_feature_maps[-1],
            feature_maps_current_level=input_feature_maps_bottom_up[level],
            upsampling=False,
            num_channels=num_channels,
            idx_BiFPN_layer=idx_BiFPN_layer,
            node_idx=3 + level,
            op_idx=8 + level, 
            use_attention=True
        )
        
        output_bottom_up_feature_maps.append(merged_feature_map)
        
    return output_bottom_up_feature_maps

def single_BiFPN_merge_step(feature_map_other_level, feature_maps_current_level, upsampling, num_channels, idx_BiFPN_layer, node_idx, op_idx, use_attention=True):
    """
    Merges two feature maps of different levels in the BiFPN
    Args:
        feature_map_other_level: Input feature map of a different level. Needs to be resized before merging.
        feature_maps_current_level: Input feature map of the current level
        upsampling: Boolean indicating whether to upsample or downsample the feature map of the different level to match the shape of the current level
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
        node_idx, op_idx: Integers needed to set the correct layer names
    
    Returns:
       The merged feature map
    """
    if upsampling:
        feature_map_resampled = layers.UpSampling2D()(feature_map_other_level)
    else:
        feature_map_resampled = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(feature_map_other_level)
    
    # Combine feature maps using weighted addition
    # merged_feature_map = wBiFPNAdd(name=f'fpn_cells_cell_{idx_BiFPN_layer}_fnode{node_idx}_add')(feature_maps_current_level + [feature_map_resampled])
    merged_feature_map = EnhancedBiFPNAdd(
        use_softmax=True,
        name=f'fpn_cells_cell_{idx_BiFPN_layer}_fnode{node_idx}_add'
    )(feature_maps_current_level + [feature_map_resampled])
    
    # Apply activation
    merged_feature_map = layers.Activation(lambda x: tf.nn.swish(x))(merged_feature_map)
    
    # Apply separable convolution
    merged_feature_map = SeparableConvBlock(
        num_channels=num_channels,
        kernel_size=3,
        strides=1,
        name=f'fpn_cells_cell_{idx_BiFPN_layer}_fnode{node_idx}_op_after_combine{op_idx}'
    )(merged_feature_map)

    # Apply spatial attention
    if use_attention:
        merged_feature_map = SpatialAttentionModule(
            name=f'fpn_cells_cell_{idx_BiFPN_layer}_fnode{node_idx}_spatial_attention'
        )(merged_feature_map)

    return merged_feature_map


def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    """
    Builds a small block consisting of a depthwise separable convolution layer and a batch norm layer
    Args:
        num_channels: Number of channels used in the BiFPN
        kernel_size: Kernel size of the depthwise separable convolution layer
        strides: Stride of the depthwise separable convolution layer
        name: Name of the block
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
    
    Returns:
       The depthwise separable convolution block
    """
    
    f1 = layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, 
                               padding='same', use_bias=True, name=f'{name}_conv')
    f2 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name=f'{name}_bn')
    
    # Original implementation using reduce
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))


def prepare_feature_maps_for_BiFPN(C3, C4, C5, num_channels, freeze_bn):
    """
    Prepares the backbone feature maps for the first BiFPN layer
    Args:
        C3, C4, C5: The EfficientNet backbone feature maps of the different levels
        num_channels: Number of channels used in the BiFPN
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
    
    Returns:
       The prepared input feature maps for the first BiFPN layer
    """
    
    # Process P3 input - Replace / with _ in names
    P3_in = C3
    P3_in = layers.Conv2D(num_channels, kernel_size=1, padding='same', 
                         name='fpn_cells_cell_0_fnode3_resample_0_0_8_conv2d')(P3_in)
    P3_in = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, 
                              name='fpn_cells_cell_0_fnode3_resample_0_0_8_bn')(P3_in)
    
    # Process P4 input - need two versions for different pathways
    P4_in = C4
    P4_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same', 
                           name='fpn_cells_cell_0_fnode2_resample_0_1_7_conv2d')(P4_in)
    P4_in_1 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, 
                                name='fpn_cells_cell_0_fnode2_resample_0_1_7_bn')(P4_in_1)
    P4_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same', 
                           name='fpn_cells_cell_0_fnode4_resample_0_1_9_conv2d')(P4_in)
    P4_in_2 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, 
                                name='fpn_cells_cell_0_fnode4_resample_0_1_9_bn')(P4_in_2)
    
    # Process P5 input - need two versions for different pathways
    P5_in = C5
    P5_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same', 
                           name='fpn_cells_cell_0_fnode1_resample_0_2_6_conv2d')(P5_in)
    P5_in_1 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, 
                                name='fpn_cells_cell_0_fnode1_resample_0_2_6_bn')(P5_in_1)
    P5_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same', 
                           name='fpn_cells_cell_0_fnode5_resample_0_2_10_conv2d')(P5_in)
    P5_in_2 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, 
                                name='fpn_cells_cell_0_fnode5_resample_0_2_10_bn')(P5_in_2)
    
    # Create P6 input by pooling P5
    P6_in = layers.Conv2D(num_channels, kernel_size=1, padding='same', 
                         name='resample_p6_conv2d')(C5)
    P6_in = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, 
                              name='resample_p6_bn')(P6_in)
    P6_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', 
                               name='resample_p6_maxpool')(P6_in)
    
    # Create P7 input by pooling P6
    P7_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', 
                               name='resample_p7_maxpool')(P6_in)
    
    return P3_in, P4_in_1, P4_in_2, P5_in_1, P5_in_2, P6_in, P7_in


def build_subnets(num_classes, subnet_width, subnet_depth, subnet_num_iteration_steps, num_groups_gn, num_rotation_parameters, freeze_bn, num_anchors):
    
    class_net = ClassNet(subnet_width,
                          subnet_depth,
                          num_classes = num_classes,
                          num_anchors = num_anchors,
                          freeze_bn = freeze_bn,
                          name = 'class_net')
    
    box_net = BoxNet(subnet_width,
                        subnet_depth,
                        num_anchors = num_anchors,
                        freeze_bn = freeze_bn,
                        name = 'box_net')
    
    rotation_net = RotationNet(subnet_width,
                                subnet_depth,
                                num_values = num_rotation_parameters,
                                num_iteration_steps = subnet_num_iteration_steps,
                                num_anchors = num_anchors,
                                freeze_bn = freeze_bn,
                                use_group_norm = True,
                                num_groups_gn = num_groups_gn,
                                name = 'rotation_net')
    
    translation_net = TranslationNet(subnet_width,
                                subnet_depth,
                                num_iteration_steps = subnet_num_iteration_steps,
                                num_anchors = num_anchors,
                                freeze_bn = freeze_bn,
                                use_group_norm = True,
                                num_groups_gn = num_groups_gn,
                                name = 'translation_net')

    return class_net, box_net, rotation_net, translation_net

def apply_subnets_to_feature_maps(box_net, class_net, rotation_net, translation_net, fpn_feature_maps, image_input, camera_parameters_input, input_size, anchor_parameters):
    # Debug: Print feature map shapes
    print("FPN Feature Map Shapes:")
    for i, feature in enumerate(fpn_feature_maps):
        print(f"Level {i}: {feature.shape}")

    # Generate anchors
    anchors, translation_anchors = anchors_for_shape((input_size, input_size), anchor_params=anchor_parameters)
    print(f"Total anchors: {anchors.shape[0]}")
    
    # Apply ClassNet to each feature map level
    classifications = []
    for i, feature in enumerate(fpn_feature_maps):
        clf = class_net(feature, level=i)
        print(f"ClassNet output for level {i}: {clf.shape}")
        classifications.append(clf)
    
    classification = layers.Concatenate(axis=1, name='classification')(classifications)
    print("Final classification shape:", classification.shape)

    # Apply BoxNet
    bbox_regressions = []
    for i, feature in enumerate(fpn_feature_maps):
        bbox = box_net(feature, level=i)
        print(f"BoxNet output for level {i}: {bbox.shape}")
        bbox_regressions.append(bbox)
    
    bbox_regression = layers.Concatenate(axis=1, name='regression')(bbox_regressions)
    print("Final bbox regression shape:", bbox_regression.shape)

    # Apply RotationNet
    rotations = []
    for i, feature in enumerate(fpn_feature_maps):
        rot = rotation_net(feature, level=i)
        print(f"RotationNet output for level {i}: {rot.shape}")
        rotations.append(rot)
    
    rotation = layers.Concatenate(axis=1, name='rotation')(rotations)
    print("Final rotation shape:", rotation.shape)

    # Apply TranslationNet
    translation_raws = []
    for i, feature in enumerate(fpn_feature_maps):
        trans = translation_net(feature, level=i)
        print(f"TranslationNet output for level {i}: {trans.shape}")
        translation_raws.append(trans)
    
    translation_raw = layers.Concatenate(axis=1, name='translation_raw_outputs')(translation_raws)
    print("Final translation raw shape:", translation_raw.shape)
    
    # Process anchors
    translation_anchors_input = np.expand_dims(translation_anchors, axis=0)
    print("Translation anchors shape:", translation_anchors_input.shape)
    
    # Check for shape mismatch and slice if necessary
    if translation_raw.shape[1] != translation_anchors_input.shape[1]:
        print(f"WARNING: Shape mismatch between translation_raw ({translation_raw.shape[1]}) and translation_anchors ({translation_anchors_input.shape[1]})")
        # Slice the predictions to match anchors
        translation_raw = layers.Lambda(
            lambda x: x[:, :translation_anchors_input.shape[1], :],
            name='slice_translation_raw'
        )(translation_raw)
        print(f"After slicing, translation_raw shape: {translation_raw.shape}")
    
    # Continue with your existing code
    translation_xy_Tz = RegressTranslation(name='translation_regression')([translation_anchors_input, translation_raw])
    print("Translation xy Tz shape:", translation_xy_Tz.shape)
    
    translation = CalculateTxTy(name='translation')(
        translation_xy_Tz,
        fx=camera_parameters_input[:, 0],
        fy=camera_parameters_input[:, 1],
        px=camera_parameters_input[:, 2],
        py=camera_parameters_input[:, 3],
        tz_scale=camera_parameters_input[:, 4],
        image_scale=camera_parameters_input[:, 5]
    )
    print("Translation shape:", translation.shape)

    # apply predicted 2D bbox regression to anchors
    anchors_input = np.expand_dims(anchors, axis = 0)

    print("Anchors input shape: ", anchors_input.shape)

    bboxes = RegressBoxes(name='boxes')([anchors_input, bbox_regression[..., :4]])

    print("Bboxes shape: ", bboxes.shape)

    bboxes = ClipBoxes(name='clipped_boxes')([image_input, bboxes])

    print("Clipped bboxes shape: ", bboxes.shape)

    transformation = layers.Lambda(lambda input_list: tf.concat(input_list, axis = -1), name="transformation")([rotation, translation])

    print("Transformation shape: ", transformation.shape)

    
    return classification, bbox_regression, rotation, translation, transformation, bboxes

    

class ClassNet(models.Model):
    def __init__(self, width, depth, num_classes=8, num_anchors=9, freeze_bn=False, use_attention=True, **kwargs):
        super(ClassNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.use_attention = use_attention
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        
        # Update layer naming convention
        self.convs = [layers.SeparableConv2D(filters=self.width, 
                                           bias_initializer='zeros', 
                                           name=f'{self.name}_class_{i}', 
                                           **options) for i in range(self.depth)]
        
        self.head = layers.SeparableConv2D(filters=self.num_classes * self.num_anchors, 
                                         bias_initializer=PriorProbability(probability=0.01), 
                                         name=f'{self.name}_class_predict', 
                                         **options)

        # Update BatchNormalization naming 
        self.bns = [[BatchNormalization(freeze=freeze_bn, 
                                       momentum=MOMENTUM, 
                                       epsilon=EPSILON, 
                                       name=f'{self.name}_class_{i}_bn_{j}') for j in range(3, 8)] for i in range(self.depth)]
        
        # Pre-create attention modules
        if self.use_attention:
            self.attention_modules = []
            for i in range(5):  # Assuming 5 levels in FPN
                self.attention_modules.append(
                    SpatialAttentionModule(name=f'{self.name}_attention_{i}')
                )
        
        self.activation = layers.Activation(lambda x: tf.nn.swish(x))
        self.activation_sigmoid = layers.Activation('sigmoid')
        self.level = 0

    def build(self, input_shape):
        # Add empty build method to suppress warning
        super(ClassNet, self).build(input_shape)

    def call(self, feature, level=None, **kwargs):
        # Support both calling conventions for backwards compatibility
        if isinstance(feature, list):
            feature, level = feature
        elif level is None:
            level = self.level
            
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][level](feature)
            feature = self.activation(feature)

        # Add spatial attention before final prediction
        if self.use_attention:
            feature = self.attention_modules[level](feature)
            
        outputs = self.head(feature)
        
        # Get the spatial dimensions of the feature map
        shape = tf.shape(outputs)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        
        # Reshape to maintain proper dimensions based on feature map size
        outputs = tf.reshape(outputs, [batch_size, height * width * self.num_anchors, self.num_classes])
        outputs = self.activation_sigmoid(outputs)
        
        # Only increment the level if it wasn't explicitly provided
        if level == self.level:
            self.level += 1
            
        return outputs
    
class BoxNet(models.Model):
    def __init__(self, width, depth, num_anchors=9, freeze_bn=False, use_attention=True, **kwargs):
        super(BoxNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = 4  # x, y, width, height regression values
        self.use_attention = use_attention
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        
        # Update layer naming convention (replacing / with _)
        self.convs = [layers.SeparableConv2D(filters=self.width, 
                                           name=f'{self.name}_box_{i}', 
                                           **options) for i in range(self.depth)]
        
        self.head = layers.SeparableConv2D(filters=self.num_anchors * self.num_values, 
                                         name=f'{self.name}_box_predict', 
                                         **options)
        
        # Update batch normalization layer naming
        self.bns = [[BatchNormalization(freeze=freeze_bn, 
                                      momentum=MOMENTUM, 
                                      epsilon=EPSILON, 
                                      name=f'{self.name}_box_{i}_bn_{j}') for j in range(3, 8)] for i in range(self.depth)]
        
        # Pre-create attention modules
        if self.use_attention:
            self.attention_modules = []
            for i in range(5):  # Assuming 5 levels in FPN
                self.attention_modules.append(
                    SpatialAttentionModule(name=f'{self.name}_attention_{i}')
                )
        
        self.activation = layers.Activation(lambda x: tf.nn.swish(x))
        self.level = 0

    def build(self, input_shape):
        # Add empty build method to suppress warning
        super(BoxNet, self).build(input_shape)

    def call(self, feature, level=None, **kwargs):
        # Support both calling conventions for backwards compatibility
        if isinstance(feature, list):
            feature, level = feature
        elif level is None:
            level = self.level
            
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][level](feature)
            feature = self.activation(feature)

        # Add spatial attention before final prediction
        if self.use_attention:
            feature = self.attention_modules[level](feature)
            
        outputs = self.head(feature)
        
        # Get the spatial dimensions of the feature map
        shape = tf.shape(outputs)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        
        # Reshape to maintain proper dimensions based on feature map size
        outputs = tf.reshape(outputs, [batch_size, height * width * self.num_anchors, self.num_values])
        
        # Only increment the level if it wasn't explicitly provided
        if level == self.level:
            self.level += 1
            
        return outputs
    
class IterativeRotationSubNet(models.Model):
    def __init__(self, width, depth, num_values, num_iteration_steps, num_anchors=9, freeze_bn=False, use_group_norm=True, num_groups_gn=None, **kwargs):
        super(IterativeRotationSubNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = num_values
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        
        if backend.image_data_format() == 'channels_first':
            gn_channel_axis = 1
        else:
            gn_channel_axis = -1
            
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        
        # Update naming convention
        self.convs = [layers.SeparableConv2D(filters=width, 
                                           name=f'{self.name}_iterative_rotation_sub_{i}', 
                                           **options) for i in range(self.depth)]
        
        self.head = layers.SeparableConv2D(filters=self.num_anchors * self.num_values, 
                                         name=f'{self.name}_iterative_rotation_sub_predict', 
                                         **options)
        
        # Update normalization layer naming
        if self.use_group_norm:
            self.norm_layer = [[[GroupNormalization(groups=self.num_groups_gn, 
                                                  axis=gn_channel_axis, 
                                                  name=f'{self.name}_iterative_rotation_sub_{k}_{i}_gn_{j}') 
                               for j in range(3, 8)] for i in range(self.depth)] 
                             for k in range(self.num_iteration_steps)]
        else:
            self.norm_layer = [[[BatchNormalization(freeze=freeze_bn, 
                                                  momentum=MOMENTUM, 
                                                  epsilon=EPSILON, 
                                                  name=f'{self.name}_iterative_rotation_sub_{k}_{i}_bn_{j}') 
                               for j in range(3, 8)] for i in range(self.depth)] 
                             for k in range(self.num_iteration_steps)]

        self.activation = layers.Activation(lambda x: tf.nn.swish(x))

        if self.use_group_norm:  # or use a flag for attention if desired
            self.iter_sub_attention = RotationAttentionModule(name=f'{self.name}_iter_sub_attention')
        else:
            self.iter_sub_attention = None

    def build(self, input_shape):
        # Add empty build method to suppress warning
        super(IterativeRotationSubNet, self).build(input_shape)

    def call(self, feature, level_py=0, iter_step_py=0, **kwargs):
        # Support different calling conventions
        if isinstance(feature, list) and len(feature) >= 3:
            # Handle case where the inputs might be passed differently
            feature = feature[0]
            if 'level_py' not in kwargs:
                level_py = kwargs.get('level', level_py)
            if 'iter_step_py' not in kwargs:
                iter_step_py = kwargs.get('iter_step', iter_step_py)
        
        if self.iter_sub_attention is not None:
            feature = self.iter_sub_attention(feature)

        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norm_layer[iter_step_py][i][level_py](feature)
            feature = self.activation(feature)
            
        outputs = self.head(feature)
        
        # Get the spatial dimensions
        shape = tf.shape(outputs)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        
        # Reshape properly
        outputs = tf.reshape(outputs, [batch_size, height * width * self.num_anchors, self.num_values])
        
        return outputs
    
    
class RotationNet(models.Model):
    def __init__(self, width, depth, num_values, num_iteration_steps, num_anchors=9, freeze_bn=False, use_group_norm=True, num_groups_gn=None, use_attention=True, **kwargs):
        super(RotationNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = num_values
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        self.use_attention = use_attention
        
        if backend.image_data_format() == 'channels_first':
            channel_axis = 0
            gn_channel_axis = 1
        else:
            channel_axis = -1
            gn_channel_axis = -1
            
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        
        # Update naming convention
        self.convs = [layers.SeparableConv2D(filters=self.width, 
                                           name=f'{self.name}_rotation_{i}', 
                                           **options) for i in range(self.depth)]
        
        self.initial_rotation = layers.SeparableConv2D(filters=self.num_anchors * self.num_values, 
                                                     name=f'{self.name}_rotation_init_predict', 
                                                     **options)
    
        # Update normalization layer naming
        if self.use_group_norm:
            self.norm_layer = [[GroupNormalization(groups=self.num_groups_gn, 
                                                axis=gn_channel_axis, 
                                                name=f'{self.name}_rotation_{i}_gn_{j}') 
                              for j in range(3, 8)] for i in range(self.depth)]
        else:
            self.norm_layer = [[BatchNormalization(freeze=freeze_bn, 
                                                momentum=MOMENTUM, 
                                                epsilon=EPSILON, 
                                                name=f'{self.name}_rotation_{i}_bn_{j}') 
                              for j in range(3, 8)] for i in range(self.depth)]
        
        # Create the iterative subnet
        self.iterative_submodel = IterativeRotationSubNet(
            width=self.width,
            depth=self.depth - 1,
            num_values=self.num_values,
            num_iteration_steps=self.num_iteration_steps,
            num_anchors=self.num_anchors,
            freeze_bn=freeze_bn,
            use_group_norm=self.use_group_norm,
            num_groups_gn=self.num_groups_gn,
            name="iterative_rotation_subnet"
        )
        
        # Create a dedicated rotation attention module
        if self.use_attention:
            self.rotation_attention_module = RotationAttentionModule(name=f'{self.name}_rotation_attention')
            self.iter_rotation_attention_modules = []
            # Create one attention module per iteration step (adjust the number as needed)
            for i in range(5):  # Assuming 5 levels, as in the original design
                for j in range(self.num_iteration_steps):
                    self.iter_rotation_attention_modules.append(
                        RotationAttentionModule(name=f'{self.name}_iter_rotation_attention_{i}_{j}')
                    )

        self.activation = layers.Activation(lambda x: tf.nn.swish(x))
        self.level = 0
        self.add = layers.Add()
        self.concat = layers.Concatenate(axis=channel_axis)

    def build(self, input_shape):
        # Add empty build method to suppress warning
        super(RotationNet, self).build(input_shape)

    def call(self, feature, level=None, **kwargs):
        # Support both calling conventions
        if isinstance(feature, list):
            feature, level = feature
        elif level is None:
            level = self.level
            
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norm_layer[i][level](feature)
            feature = self.activation(feature)
            
        # Add spatial attention before initial rotation prediction
        if self.use_attention:
            attended_feature = self.rotation_attention_module(feature)
            rotation = self.initial_rotation(attended_feature)
        else:
            rotation = self.initial_rotation(feature)
        
        # Get the spatial dimensions
        shape = tf.shape(rotation)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        
        # Reshape properly for the iterative steps
        rotation_reshaped = tf.reshape(rotation, [batch_size, height, width, self.num_anchors * self.num_values])
        
        for i in range(self.num_iteration_steps):
            iterative_input = self.concat([feature, rotation_reshaped])

            # Optionally apply attention to iterative input
            if self.use_attention:
                iter_attention_idx = level * self.num_iteration_steps + i
                iterative_input = self.iter_rotation_attention_modules[iter_attention_idx](iterative_input)

            # Call iterative submodel
            delta_rotation = self.iterative_submodel(
                iterative_input, 
                level_py=level, 
                iter_step_py=i
            )
            
            # Get spatial dimensions of delta rotation
            delta_shape = tf.shape(delta_rotation)
            
            # Reshape delta_rotation to match rotation_reshaped
            delta_rotation_reshaped = tf.reshape(delta_rotation, [batch_size, height, width, self.num_anchors * self.num_values])
            
            # Add the delta to the rotation
            rotation_reshaped = self.add([rotation_reshaped, delta_rotation_reshaped])
        
        # Final reshape for output
        outputs = tf.reshape(rotation_reshaped, [batch_size, height * width * self.num_anchors, self.num_values])
        
        # Only increment the level if it wasn't explicitly provided
        if level == self.level:
            self.level += 1
            
        return outputs
        
    
class IterativeTranslationSubNet(models.Model):
    def __init__(self, width, depth, num_iteration_steps, num_anchors=9, freeze_bn=False, use_group_norm=True, num_groups_gn=None, **kwargs):
        super(IterativeTranslationSubNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        
        if backend.image_data_format() == 'channels_first':
            gn_channel_axis = 1
        else:
            gn_channel_axis = -1
            
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        
        # Update naming convention
        self.convs = [layers.SeparableConv2D(filters=self.width, 
                                           name=f'{self.name}_iterative_translation_sub_{i}', 
                                           **options) for i in range(self.depth)]
        
        self.head_xy = layers.SeparableConv2D(filters=self.num_anchors * 2, 
                                            name=f'{self.name}_iterative_translation_xy_sub_predict', 
                                            **options)
        
        self.head_z = layers.SeparableConv2D(filters=self.num_anchors, 
                                           name=f'{self.name}_iterative_translation_z_sub_predict', 
                                           **options)

        # Update normalization layer naming
        if self.use_group_norm:
            self.norm_layer = [[[GroupNormalization(groups=self.num_groups_gn, 
                                                  axis=gn_channel_axis, 
                                                  name=f'{self.name}_iterative_translation_sub_{k}_{i}_gn_{j}') 
                                for j in range(3, 8)] for i in range(self.depth)] 
                              for k in range(self.num_iteration_steps)]
        else:
            self.norm_layer = [[[BatchNormalization(freeze=freeze_bn, 
                                                  momentum=MOMENTUM, 
                                                  epsilon=EPSILON, 
                                                  name=f'{self.name}_iterative_translation_sub_{k}_{i}_bn_{j}') 
                                for j in range(3, 8)] for i in range(self.depth)] 
                              for k in range(self.num_iteration_steps)]

        self.activation = layers.Activation(lambda x: tf.nn.swish(x))

    def build(self, input_shape):
        # Add empty build method to suppress warning
        super(IterativeTranslationSubNet, self).build(input_shape)

    def call(self, feature, level_py=0, iter_step_py=0, **kwargs):
        # Support different calling conventions
        if isinstance(feature, list):
            # Handle case where the inputs might be passed differently
            feature = feature[0]
            if 'level_py' not in kwargs:
                level_py = kwargs.get('level', level_py)
            if 'iter_step_py' not in kwargs:
                iter_step_py = kwargs.get('iter_step', iter_step_py)
        
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norm_layer[iter_step_py][i][level_py](feature)
            feature = self.activation(feature)
            
        outputs_xy = self.head_xy(feature)
        outputs_z = self.head_z(feature)
        
        # Get the spatial dimensions
        shape = tf.shape(outputs_xy)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        
        # Reshape properly
        outputs_xy = tf.reshape(outputs_xy, [batch_size, height * width * self.num_anchors, 2])
        outputs_z = tf.reshape(outputs_z, [batch_size, height * width * self.num_anchors, 1])
        
        return outputs_xy, outputs_z


class TranslationNet(models.Model):
    def __init__(self, width, depth, num_iteration_steps, num_anchors=9, freeze_bn=False, use_group_norm=True, num_groups_gn=None, use_attention=True, **kwargs):
        super(TranslationNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        self.use_attention = use_attention
        
        if backend.image_data_format() == 'channels_first':
            channel_axis = 0
            gn_channel_axis = 1
        else:
            channel_axis = -1
            gn_channel_axis = -1
            
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        
        # Update naming convention
        self.convs = [layers.SeparableConv2D(filters=self.width, 
                                           name=f'{self.name}_translation_{i}', 
                                           **options) for i in range(self.depth)]
        
        self.initial_translation_xy = layers.SeparableConv2D(filters=self.num_anchors * 2, 
                                                           name=f'{self.name}_translation_xy_init_predict', 
                                                           **options)
        
        self.initial_translation_z = layers.SeparableConv2D(filters=self.num_anchors, 
                                                          name=f'{self.name}_translation_z_init_predict', 
                                                          **options)

        # Update normalization layer naming
        if self.use_group_norm:
            self.norm_layer = [[GroupNormalization(groups=self.num_groups_gn, 
                                                 axis=gn_channel_axis, 
                                                 name=f'{self.name}_translation_{i}_gn_{j}') 
                              for j in range(3, 8)] for i in range(self.depth)]
        else: 
            self.norm_layer = [[BatchNormalization(momentum=MOMENTUM, 
                                                 epsilon=EPSILON, 
                                                 name=f'{self.name}_translation_{i}_bn_{j}') 
                              for j in range(3, 8)] for i in range(self.depth)]
        
        # Create the iterative subnet
        self.iterative_submodel = IterativeTranslationSubNet(
            width=self.width,
            depth=self.depth - 1,
            num_iteration_steps=self.num_iteration_steps,
            num_anchors=self.num_anchors,
            freeze_bn=freeze_bn,
            use_group_norm=self.use_group_norm,
            num_groups_gn=self.num_groups_gn,
            name="iterative_translation_subnet"
        )
        
        # Pre-create attention modules
        if self.use_attention:
            self.attention_modules = []
            self.iter_attention_modules = []
            for i in range(5):  # Assuming 5 levels max in FPN
                self.attention_modules.append(
                    SpatialAttentionModule(name=f'{self.name}_attention_{i}')
                )
                for j in range(self.num_iteration_steps):
                    self.iter_attention_modules.append(
                        SpatialAttentionModule(name=f'{self.name}_iter_attention_{i}_{j}')
                    )

        self.activation = layers.Activation(lambda x: tf.nn.swish(x))
        self.level = 0
        self.add = layers.Add()
        self.concat = layers.Concatenate(axis=channel_axis)
        
        # Always use last axis after reshape for output concatenation
        self.concat_output = layers.Concatenate(axis=-1)

    def build(self, input_shape):
        # Add empty build method to suppress warning
        super(TranslationNet, self).build(input_shape)

    def call(self, feature, level=None, **kwargs):
        # Support both calling conventions
        if isinstance(feature, list):
            feature, level = feature
        elif level is None:
            level = self.level
            
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norm_layer[i][level](feature)
            feature = self.activation(feature)
            
        # Apply spatial attention before initial translation predictions
        if self.use_attention:
            attended_feature = self.attention_modules[level](feature)
            translation_xy = self.initial_translation_xy(attended_feature)
            translation_z = self.initial_translation_z(attended_feature)
        else:
            translation_xy = self.initial_translation_xy(feature)
            translation_z = self.initial_translation_z(feature)
        
        # Get the spatial dimensions
        shape = tf.shape(translation_xy)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        
        # Keep original shapes for the iterative steps
        translation_xy_reshaped = translation_xy
        translation_z_reshaped = translation_z
        
        for i in range(self.num_iteration_steps):
            iterative_input = self.concat([feature, translation_xy_reshaped, translation_z_reshaped])
            
            # Apply spatial attention for iterative refinement
            if self.use_attention:
                iter_attention_idx = level * self.num_iteration_steps + i
                iterative_input = self.iter_attention_modules[iter_attention_idx](iterative_input)

            # Call iterative submodel
            delta_translation_xy, delta_translation_z = self.iterative_submodel(
                iterative_input,
                level_py=level,
                iter_step_py=i
            )
            
            # Reshape deltas to match original spatial dimensions
            delta_xy_reshaped = tf.reshape(delta_translation_xy, [batch_size, height, width, self.num_anchors * 2])
            delta_z_reshaped = tf.reshape(delta_translation_z, [batch_size, height, width, self.num_anchors])
            
            # Add the deltas
            translation_xy_reshaped = self.add([translation_xy_reshaped, delta_xy_reshaped])
            translation_z_reshaped = self.add([translation_z_reshaped, delta_z_reshaped])
        
        # Final reshape for output
        outputs_xy = tf.reshape(translation_xy_reshaped, [batch_size, height * width * self.num_anchors, 2])
        outputs_z = tf.reshape(translation_z_reshaped, [batch_size, height * width * self.num_anchors, 1])
        
        # Concatenate for final output
        outputs = self.concat_output([outputs_xy, outputs_z])
        
        # Only increment the level if it wasn't explicitly provided
        if level == self.level:
            self.level += 1
            
        return outputs


if __name__ == '__main__':
    print("Model loaded")
    print("Tensorflow version: ", tf.__version__)

    phi = 0
    num_rotation_parameters = 3
    num_classes = 1
    num_anchors = 9
    freeze_bn = True
    score_threshold = 0.7

    print("\nBuilding the Model...")
    efficientpose_train, efficientpose_prediction, all_layers = build_EfficientPose(phi,
                                                                                    num_classes = num_classes,
                                                                                    num_anchors = num_anchors,
                                                                                    freeze_bn = not freeze_bn,
                                                                                    score_threshold = score_threshold,
                                                                                    num_rotation_parameters = num_rotation_parameters)
    
    # output: Tensorflow version:  2.19.0
