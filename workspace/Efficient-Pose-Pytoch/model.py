from functools import reduce
import h5py

import torch
import torch.nn as nn

import numpy as np

from EfficientNet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6
from EfficientNet import DEFAULT_BLOCKS_ARGS
from layers import wBiFPNAdd, RegressTranslation, CalculateTxTy, RegressBoxes, ClipBoxes
from utils.anchors import anchors_for_shape
from filter_detections import FilterDetections

class EfficientPose(nn.Module):
    def __init__(self,
                 phi,
                 num_classes = 8,
                 num_anchors = 9,
                 freeze_bn = False,
                 score_threshold = 0.5,
                 anchor_parameters = None,
                 num_rotation_parameters = 3,
                 print_architecture = True):
        """
        Builds an EfficientPose model
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
            A pytoch model instance.
        """
        super(EfficientPose, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.freeze_bn = freeze_bn
        self.score_threshold = score_threshold
        self.anchor_parameters = anchor_parameters
        self.num_rotation_parameters = num_rotation_parameters
        self.print_architecture = print_architecture

        #select parameters according to the given phi
        assert phi in range(7)
        scaled_parameters = get_scaled_parameters(phi)

        input_size = scaled_parameters["input_size"]
        self.input_shape = (3, input_size, input_size)
        bifpn_width = subnet_width = scaled_parameters["bifpn_width"]
        bifpn_depth = scaled_parameters["bifpn_depth"]
        subnet_depth = scaled_parameters["subnet_depth"]
        subnet_num_iteration_steps = scaled_parameters["subnet_num_iteration_steps"]
        num_groups_gn = scaled_parameters["num_groups_gn"]
        backbone_class = scaled_parameters["backbone_class"]

        #build EfficientNet backbone
        self.backbone = backbone_class(input_shape = self.input_shape)
        
        #build BiFPN
        self.BiFPN = BiFPN(bifpn_depth, bifpn_width, freeze_bn)

        #build subnets
        self.box_net, self.class_net, self.rotation_net, self.translation_net = build_subnets(num_classes,
                                                                                              subnet_width,
                                                                                              subnet_depth,
                                                                                              subnet_num_iteration_steps,
                                                                                              num_groups_gn,
                                                                                              num_rotation_parameters,
                                                                                              freeze_bn,
                                                                                              num_anchors)
        
        #adding missing module from apply_subnets_to_feature_maps
        self.RegressTranslation = RegressTranslation(name = 'translation_regression')
        self.CalculateTxTy = CalculateTxTy(name = 'translation')
        self.RegressBoxes = RegressBoxes(name='boxes')

        self.filter_detections = FilterDetections(
            num_rotation_parameters=num_rotation_parameters,
            num_translation_parameters=3,
            score_threshold=score_threshold,
            name='filtered_detections'
        )

    def apply_subnets_to_feature_maps(self, fpn_feature_maps, camera_parameters_input):
        """
        Applies the subnetworks to the BiFPN feature maps
        Args:
            fpn_feature_maps: Sequence of the BiFPN feature maps of the different levels (P3, P4, P5, P6, P7)
            camera_parameters_input: camera parameter input
        
        Returns:
        classification: Tensor containing the classification outputs for all anchor boxes. Shape (batch_size, num_anchor_boxes, num_classes)
        bbox_regression: Tensor containing the deltas of anchor boxes to the GT 2D bounding boxes for all anchor boxes. Shape (batch_size, num_anchor_boxes, 4)
        rotation: Tensor containing the rotation outputs for all anchor boxes. Shape (batch_size, num_anchor_boxes, num_rotation_parameters)
        translation: Tensor containing the translation outputs for all anchor boxes. Shape (batch_size, num_anchor_boxes, 3)
        transformation: Tensor containing the concatenated rotation and translation outputs for all anchor boxes. Shape (batch_size, num_anchor_boxes, num_rotation_parameters + 3)
                        Rotation and Translation are concatenated because the Keras Loss function takes only one GT and prediction tensor respectively as input but the transformation loss needs both
        bboxes: Tensor containing the 2D bounding boxes for all anchor boxes. Shape (batch_size, num_anchor_boxes, 4)
        """
        classification = [self.class_net([feature, i]) for i, feature in enumerate(fpn_feature_maps)]
        classification = torch.cat(classification, dim = 1)

        # print("Classification output shape: ", classification.shape)

        bbox_regression = [self.box_net([feature, i]) for i, feature in enumerate(fpn_feature_maps)]
        bbox_regression = torch.cat(bbox_regression, dim = 1)

        # print("Bbox regression output shape: ", bbox_regression.shape)
    
        rotation = [self.rotation_net([feature, i]) for i, feature in enumerate(fpn_feature_maps)]
        rotation = torch.cat(rotation, dim = 1)

        # print("Rotation output shape: ", rotation.shape)

        translation_raw = [self.translation_net([feature, i]) for i, feature in enumerate(fpn_feature_maps)]
        translation_raw = torch.cat(translation_raw, dim = 1)

        # print("Translation raw output shape: ", translation_raw.shape)

        #get anchors and apply predicted translation offsets to translation anchors
        anchors, translation_anchors = anchors_for_shape(self.input_shape[1:3], anchor_params = self.anchor_parameters)
        translation_anchors_input = torch.from_numpy(np.expand_dims(translation_anchors, axis = 0)).to(translation_raw.device)

        # print("Anchors shape: ", anchors.shape)
        # print("Translation anchors shape: ", translation_anchors_input.shape)


        translation_xy_Tz = self.RegressTranslation([translation_anchors_input, translation_raw])

        # print("Translation xy Tz shape: ", translation_xy_Tz.shape)

        translation = self.CalculateTxTy(translation_xy_Tz,
                                         fx = camera_parameters_input[:, :, 0],
                                         fy = camera_parameters_input[:, :, 1],
                                         px = camera_parameters_input[:, :, 2],
                                         py = camera_parameters_input[:, :, 3],
                                         tz_scale = camera_parameters_input[:, :, 4],
                                         image_scale = camera_parameters_input[:, :, 5])
        
        # print("Translation shape: ", translation.shape)

        # apply predicted 2D bbox regression to anchors
        anchors_input = torch.from_numpy(np.expand_dims(anchors, axis = 0)).to(bbox_regression.device)

        # print("Anchors input shape: ", anchors_input.shape)

        bboxes = self.RegressBoxes([anchors_input, bbox_regression[..., :4]])

        # print("Bboxes shape: ", bboxes.shape)

        bboxes = ClipBoxes([self.input_shape, bboxes])

        # print("Bboxes clipped shape: ", bboxes.shape)
    
        #concat rotation and translation outputs to transformation output to have a single output for transformation loss calculation

        transformation = torch.cat([rotation, translation], dim = -1)

        # print("Transformation shape: ", transformation.shape)

        return classification, bbox_regression, rotation, translation, transformation, bboxes

    def forward(self, input, camera_parameters_input):
        #build EfficientNet backbone
        backbone_feature_maps = self.backbone(input)
        #build BiFPN
        fpn_feature_maps = self.BiFPN(backbone_feature_maps)
        #apply subnets to feature maps
        classification, bbox_regression, rotation, translation, transformation, bboxes = self.apply_subnets_to_feature_maps(fpn_feature_maps, camera_parameters_input)


        if self.training:
            return classification, bbox_regression, transformation
        
        filtered_detections = self.filter_detections([bboxes, classification, rotation, translation])
        
        return filtered_detections

        


    def load_h5(self, weights_dir):
        self.backbone.load_h5(weights_dir)
        self.BiFPN.load_h5(weights_dir)
        self.box_net.load_h5(weights_dir)
        self.class_net.load_h5(weights_dir)
        self.rotation_net.load_h5(weights_dir)
        self.translation_net.load_h5(weights_dir)

        

def get_scaled_parameters(phi):
    """
    Get all needed scaled parameters to build EfficientPose
    Args:
        phi: EfficientPose scaling hyperparameter phi
    
    Returns:
       Dictionary containing the scaled parameters
    """
    #info tuples with scalable parameters
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    bifpn_widths = (64, 88, 112, 160, 224, 288, 384)
    bifpn_depths = (3, 4, 5, 6, 7, 7, 8)
    subnet_depths = (3, 3, 3, 4, 4, 4, 5)
    subnet_iteration_steps = (1, 1, 1, 2, 2, 2, 3)
    num_groups_gn = (4, 4, 7, 10, 14, 18, 24) #try to get 16 channels per group
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

class BiFPN(nn.Module):
    def __init__(
        self, bifpn_depth, bifpn_width, freeze_bn
    ):
        """
        Building the bidirectional feature pyramid as described in https://arxiv.org/abs/1911.09070
        Args:
            bifpn_depth: Number of BiFPN layer
            bifpn_width: Number of channels used in the BiFPN
            freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
        
        Returns:
        A pytorch BiFPN layers Sequence instance
        """
        super(BiFPN, self).__init__()
        self.bifpn_depth = bifpn_depth

        self.BiFPN_layers = nn.ModuleList()

        for i in range(bifpn_depth):
            self.BiFPN_layers.append(BiFPN_layer(bifpn_width, i, freeze_bn = freeze_bn))

    def forward(self, backbone_feature_maps):
        
        fpn_feature_maps = backbone_feature_maps
        for i in range(self.bifpn_depth):
            fpn_feature_maps = self.BiFPN_layers[i](fpn_feature_maps)
            
        return fpn_feature_maps

    def load_h5(self, weights_dir):
        for i in range(self.bifpn_depth):
            self.BiFPN_layers[i].load_h5(weights_dir)


class BiFPN_layer(nn.Module):
    def __init__(
        self, num_channels, idx_BiFPN_layer, freeze_bn = False
    ):
        super(BiFPN_layer, self).__init__()
        """
        Builds a single layer of the bidirectional feature pyramid
        Args:
            num_channels: Number of channels used in the BiFPN
            idx_BiFPN_layer: The index of the BiFPN layer to build
            freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
        
        Returns:
        A pytorch BiFPN layer instance
        """
        self.idx_BiFPN_layer = idx_BiFPN_layer

        if idx_BiFPN_layer == 0:
            self.prepare_feature_maps_for_BiFPN = prepare_feature_maps_for_BiFPN(num_channels, freeze_bn)
            
        self.top_down_pathway_BiFPN = top_down_pathway_BiFPN(num_channels, idx_BiFPN_layer)
        
        self.bottom_up_pathway_BiFPN = bottom_up_pathway_BiFPN(num_channels, idx_BiFPN_layer)
    
    def forward(self, features):

        if self.idx_BiFPN_layer == 0:
            _, _, C3, C4, C5 = features
            P3_in, P4_in_1, P4_in_2, P5_in_1, P5_in_2, P6_in, P7_in = self.prepare_feature_maps_for_BiFPN(C3, C4, C5)
        else:
            P3_in, P4_in, P5_in, P6_in, P7_in = features
        
        #top down pathway
        input_feature_maps_top_down = [P7_in,
                                       P6_in,
                                       P5_in_1 if self.idx_BiFPN_layer == 0 else P5_in,
                                       P4_in_1 if self.idx_BiFPN_layer == 0 else P4_in,
                                       P3_in]

        P7_in, P6_td, P5_td, P4_td, P3_out = self.top_down_pathway_BiFPN(input_feature_maps_top_down)

        #bottom up pathway
        input_feature_maps_bottom_up = [[P3_out],
                                        [P4_in_2 if self.idx_BiFPN_layer == 0 else P4_in, P4_td],
                                        [P5_in_2 if self.idx_BiFPN_layer == 0 else P5_in, P5_td],
                                        [P6_in, P6_td],
                                        [P7_in]]

        P3_out, P4_out, P5_out, P6_out, P7_out = self.bottom_up_pathway_BiFPN(input_feature_maps_bottom_up)

        return P3_out, P4_td, P5_td, P6_td, P7_out

    def load_h5(self, weights_dir):
        if self.idx_BiFPN_layer == 0:   
            self.prepare_feature_maps_for_BiFPN.load_h5(weights_dir)
        self.top_down_pathway_BiFPN.load_h5(weights_dir)
        self.bottom_up_pathway_BiFPN.load_h5(weights_dir)


class prepare_feature_maps_for_BiFPN(nn.Module):
    def __init__(
        self, num_channels, freeze_bn
    ):
        """
        Prepares the backbone feature maps for the first BiFPN layer
        Args:
            num_channels: Number of channels used in the BiFPN
            freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
        
        Returns:
        A pytorch instance of the first BiFPN input layer
        """
        
        super(prepare_feature_maps_for_BiFPN, self).__init__()

        blocks_args = DEFAULT_BLOCKS_ARGS
        input_layers_sizes = []
        for idx, block_args in enumerate(blocks_args):
            if idx < len(blocks_args) - 1 and blocks_args[idx + 1].strides == 2:
                input_layers_sizes.append(block_args.output_filters)
            elif idx == len(blocks_args) - 1:
                input_layers_sizes.append(block_args.output_filters)

        self.P3 = nn.Sequential(
            nn.Conv2d(input_layers_sizes[3-1], num_channels, 1, 1, 0),
            nn.BatchNorm2d(num_channels)
        )
        self.P4_1 = nn.Sequential(
            nn.Conv2d(input_layers_sizes[4-1], num_channels, 1, 1, 0),
            nn.BatchNorm2d(num_channels)
        )
        self.P4_2 = nn.Sequential(
            nn.Conv2d(input_layers_sizes[4-1], num_channels, 1, 1, 0),
            nn.BatchNorm2d(num_channels)
        )
        self.P5_1 = nn.Sequential(
            nn.Conv2d(input_layers_sizes[5-1], num_channels, 1, 1, 0),
            nn.BatchNorm2d(num_channels)
        )
        self.P5_2 = nn.Sequential(
            nn.Conv2d(input_layers_sizes[5-1], num_channels, 1, 1, 0),
            nn.BatchNorm2d(num_channels)
        )
        self.P6 = nn.Sequential(
            nn.Conv2d(input_layers_sizes[5-1], num_channels, 1, 1, 0),
            nn.BatchNorm2d(num_channels),
            nn.MaxPool2d(3,2,1)
        )
        self.P7 = nn.MaxPool2d(3,2,1)

    def forward (self, C3, C4, C5):
        P3_in = C3
        P3_in = self.P3(P3_in)
        
        P4_in = C4
        P4_in_1 = self.P4_1(P4_in)
        P4_in_2 = self.P4_2(P4_in)
        
        P5_in = C5
        P5_in_1 = self.P5_1(P5_in)
        P5_in_2 = self.P5_2(P5_in)
        
        P6_in = self.P6(C5)
        
        P7_in = self.P7 (P6_in)
        
        return P3_in, P4_in_1, P4_in_2, P5_in_1, P5_in_2, P6_in, P7_in

    def load_h5(self, weights_dir):
        f = h5py.File(weights_dir,'r')
        #P3
        self.P3[0].weight.data = torch.from_numpy(np.transpose(np.array(f['model_weights/fpn_cells/cell_0/fnode3/resample_0_0_8/conv2d/fpn_cells/cell_0/fnode3/resample_0_0_8/conv2d/kernel:0'])))
        self.P3[0].bias.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode3/resample_0_0_8/conv2d/fpn_cells/cell_0/fnode3/resample_0_0_8/conv2d/bias:0']))
        self.P3[1].running_mean.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode3/resample_0_0_8/bn/fpn_cells/cell_0/fnode3/resample_0_0_8/bn/moving_mean:0']))
        self.P3[1].running_var.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode3/resample_0_0_8/bn/fpn_cells/cell_0/fnode3/resample_0_0_8/bn/moving_variance:0']))
        self.P3[1].weight.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode3/resample_0_0_8/bn/fpn_cells/cell_0/fnode3/resample_0_0_8/bn/gamma:0']))
        self.P3[1].bias.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode3/resample_0_0_8/bn/fpn_cells/cell_0/fnode3/resample_0_0_8/bn/beta:0']))
        #P4_1
        self.P4_1[0].weight.data = torch.from_numpy(np.transpose(np.array(f['model_weights/fpn_cells/cell_0/fnode2/resample_0_1_7/conv2d/fpn_cells/cell_0/fnode2/resample_0_1_7/conv2d/kernel:0'])))
        self.P4_1[0].bias.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode2/resample_0_1_7/conv2d/fpn_cells/cell_0/fnode2/resample_0_1_7/conv2d/bias:0']))
        self.P4_1[1].running_mean.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode2/resample_0_1_7/bn/fpn_cells/cell_0/fnode2/resample_0_1_7/bn/moving_mean:0']))
        self.P4_1[1].running_var.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode2/resample_0_1_7/bn/fpn_cells/cell_0/fnode2/resample_0_1_7/bn/moving_variance:0']))
        self.P4_1[1].weight.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode2/resample_0_1_7/bn/fpn_cells/cell_0/fnode2/resample_0_1_7/bn/gamma:0']))
        self.P4_1[1].bias.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode2/resample_0_1_7/bn/fpn_cells/cell_0/fnode2/resample_0_1_7/bn/beta:0']))
        #P4_2
        self.P4_2[0].weight.data = torch.from_numpy(np.transpose(np.array(f['model_weights/fpn_cells/cell_0/fnode4/resample_0_1_9/conv2d/fpn_cells/cell_0/fnode4/resample_0_1_9/conv2d/kernel:0'])))
        self.P4_2[0].bias.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode4/resample_0_1_9/conv2d/fpn_cells/cell_0/fnode4/resample_0_1_9/conv2d/bias:0']))
        self.P4_2[1].running_mean.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode4/resample_0_1_9/bn/fpn_cells/cell_0/fnode4/resample_0_1_9/bn/moving_mean:0']))
        self.P4_2[1].running_var.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode4/resample_0_1_9/bn/fpn_cells/cell_0/fnode4/resample_0_1_9/bn/moving_variance:0']))
        self.P4_2[1].weight.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode4/resample_0_1_9/bn/fpn_cells/cell_0/fnode4/resample_0_1_9/bn/gamma:0']))
        self.P4_2[1].bias.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode4/resample_0_1_9/bn/fpn_cells/cell_0/fnode4/resample_0_1_9/bn/beta:0']))
        #P5_1
        self.P5_1[0].weight.data = torch.from_numpy(np.transpose(np.array(f['model_weights/fpn_cells/cell_0/fnode1/resample_0_2_6/conv2d/fpn_cells/cell_0/fnode1/resample_0_2_6/conv2d/kernel:0'])))
        self.P5_1[0].bias.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode1/resample_0_2_6/conv2d/fpn_cells/cell_0/fnode1/resample_0_2_6/conv2d/bias:0']))
        self.P5_1[1].running_mean.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode1/resample_0_2_6/bn/fpn_cells/cell_0/fnode1/resample_0_2_6/bn/moving_mean:0']))
        self.P5_1[1].running_var.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode1/resample_0_2_6/bn/fpn_cells/cell_0/fnode1/resample_0_2_6/bn/moving_variance:0']))
        self.P5_1[1].weight.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode1/resample_0_2_6/bn/fpn_cells/cell_0/fnode1/resample_0_2_6/bn/gamma:0']))
        self.P5_1[1].bias.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode1/resample_0_2_6/bn/fpn_cells/cell_0/fnode1/resample_0_2_6/bn/beta:0']))
        #P5_2
        self.P5_2[0].weight.data = torch.from_numpy(np.transpose(np.array(f['model_weights/fpn_cells/cell_0/fnode5/resample_0_2_10/conv2d/fpn_cells/cell_0/fnode5/resample_0_2_10/conv2d/kernel:0'])))
        self.P5_2[0].bias.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode5/resample_0_2_10/conv2d/fpn_cells/cell_0/fnode5/resample_0_2_10/conv2d/bias:0']))
        self.P5_2[1].running_mean.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode5/resample_0_2_10/bn/fpn_cells/cell_0/fnode5/resample_0_2_10/bn/moving_mean:0']))
        self.P5_2[1].running_var.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode5/resample_0_2_10/bn/fpn_cells/cell_0/fnode5/resample_0_2_10/bn/moving_variance:0']))
        self.P5_2[1].weight.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode5/resample_0_2_10/bn/fpn_cells/cell_0/fnode5/resample_0_2_10/bn/gamma:0']))
        self.P5_2[1].bias.data = torch.from_numpy(np.array(f['model_weights/fpn_cells/cell_0/fnode5/resample_0_2_10/bn/fpn_cells/cell_0/fnode5/resample_0_2_10/bn/beta:0']))
        #P6
        self.P6[0].weight.data = torch.from_numpy(np.transpose(np.array(f['model_weights/resample_p6/conv2d/resample_p6/conv2d/kernel:0'])))
        self.P6[0].bias.data = torch.from_numpy(np.array(f['model_weights/resample_p6/conv2d/resample_p6/conv2d/bias:0']))
        self.P6[1].running_mean.data = torch.from_numpy(np.array(f['model_weights/resample_p6/bn/resample_p6/bn/moving_mean:0']))
        self.P6[1].running_var.data = torch.from_numpy(np.array(f['model_weights/resample_p6/bn/resample_p6/bn/moving_variance:0']))
        self.P6[1].weight.data = torch.from_numpy(np.array(f['model_weights/resample_p6/bn/resample_p6/bn/gamma:0']))
        self.P6[1].bias.data = torch.from_numpy(np.array(f['model_weights/resample_p6/bn/resample_p6/bn/beta:0']))

        f.close()
        

class top_down_pathway_BiFPN(nn.Module):
    def __init__(
        self, num_channels, idx_BiFPN_layer
    ):
        """
        Computes the top-down-pathway in a single BiFPN layer
        Args:
            num_channels: Number of channels used in the BiFPN
            idx_BiFPN_layer: The index of the BiFPN layer to build
        
        Returns:
        A pytorch model with the output feature maps of the top-down-pathway
        """
        
        super(top_down_pathway_BiFPN, self).__init__()

        self.num_channels = num_channels
        self.idx_BiFPN_layer = idx_BiFPN_layer

        self.BiFPN_merge_steps = nn.ModuleList()

        for level in range(1, 5):
            self.BiFPN_merge_steps.append(single_BiFPN_merge_step(upsampling = True,
                                                                  num_channels = self.num_channels,
                                                                  idx_BiFPN_layer = self.idx_BiFPN_layer,
                                                                  node_idx = level - 1,
                                                                  op_idx = 4 + level))

    def forward(self, input_feature_maps_top_down):
        feature_map_P7 = input_feature_maps_top_down[0]
        output_top_down_feature_maps = [feature_map_P7]
        for level in range(1, 5):  
            merged_feature_map = self.BiFPN_merge_steps[level-1](feature_map_other_level = output_top_down_feature_maps[-1],
                                                                 feature_maps_current_level = input_feature_maps_top_down[level])
            output_top_down_feature_maps.append(merged_feature_map)
            
        return output_top_down_feature_maps

    def load_h5(self, weights_dir):
        for i in range (len(self.BiFPN_merge_steps)):
            self.BiFPN_merge_steps[i].load_h5(weights_dir)


class bottom_up_pathway_BiFPN(nn.Module):
    def __init__(
        self, num_channels, idx_BiFPN_layer
    ):
        """
        Computes the bottom-up-pathway in a single BiFPN layer
        Args:
            input_feature_maps_top_down: Sequence containing a list of feature maps serving as input for each level of the BiFPN layer (P3, P4, P5, P6, P7)
            num_channels: Number of channels used in the BiFPN
            idx_BiFPN_layer: The index of the BiFPN layer to build
        
        Returns:
        A pytorch model with the output feature maps of the bottom_up-pathway
        """
        
        super(bottom_up_pathway_BiFPN, self).__init__()

        self.num_channels = num_channels
        self.idx_BiFPN_layer = idx_BiFPN_layer

        self.BiFPN_merge_steps = nn.ModuleList()

        for level in range(1, 5):
            self.BiFPN_merge_steps.append(single_BiFPN_merge_step(upsampling = False,
                                                                  num_channels = self.num_channels,
                                                                  idx_BiFPN_layer = self.idx_BiFPN_layer,
                                                                  node_idx = 3 + level,
                                                                  op_idx = 8 + level))

    def forward(self, input_feature_maps_bottom_up):
        feature_map_P3 = input_feature_maps_bottom_up[0][0]
        output_bottom_up_feature_maps = [feature_map_P3]
        
        for level in range(1, 5):    
            merged_feature_map = self.BiFPN_merge_steps[level-1](feature_map_other_level = output_bottom_up_feature_maps[-1],
                                                                 feature_maps_current_level = input_feature_maps_bottom_up[level])

            output_bottom_up_feature_maps.append(merged_feature_map)
            
        return output_bottom_up_feature_maps

    def load_h5(self, weights_dir):
        for i in range (len(self.BiFPN_merge_steps)):
            self.BiFPN_merge_steps[i].load_h5(weights_dir)


class single_BiFPN_merge_step(nn.Module):
    def __init__(
        self, upsampling, num_channels, idx_BiFPN_layer, node_idx, op_idx
    ):
        """
        Merges two feature maps of different levels in the BiFPN
        Args:
            upsampling: Boolean indicating wheter to upsample or downsample the feature map of the different level to match the shape of the current level
            num_channels: Number of channels used in the BiFPN
            idx_BiFPN_layer: The index of the BiFPN layer to build
            node_idx, op_idx: Integers needed to set the correct layer names
        
        Returns:
        A pytorch feature merging block
        """
        super(single_BiFPN_merge_step, self).__init__()
        self.idx_BiFPN_layer = idx_BiFPN_layer
        self.node_idx = node_idx

        if upsampling:
            self.resample_feature_map = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.resample_feature_map = nn.MaxPool2d(kernel_size=3, stride=2, padding= 3//2)
        
        self.upsampling = upsampling

        if (upsampling):
            self.wBiFPNAdd = wBiFPNAdd(2)
        else:
            if (node_idx == 7):
                self.wBiFPNAdd = wBiFPNAdd(2)
            else :
                self.wBiFPNAdd = wBiFPNAdd(3)
        
        self.SeparableConvBlock = SeparableConvBlock(num_channels = num_channels,
                                                     kernel_size = 3,
                                                     name = f'fpn_cells/cell_{idx_BiFPN_layer}/fnode{node_idx}/op_after_combine{op_idx}')

    def forward (self, feature_map_other_level, feature_maps_current_level):
        
        feature_map_resampled = self.resample_feature_map(feature_map_other_level)

        if (self.upsampling):
            merged_feature_map = self.wBiFPNAdd(torch.stack([feature_map_resampled,feature_maps_current_level]))
        else:
            wBiFPNAdd_input = feature_maps_current_level
            wBiFPNAdd_input.append (feature_map_resampled)     
            merged_feature_map = self.wBiFPNAdd(torch.stack(wBiFPNAdd_input))

        merged_feature_map = nn.SiLU()(merged_feature_map)
        merged_feature_map = self.SeparableConvBlock(merged_feature_map)

        return merged_feature_map


    def load_h5(self, weights_dir):
        self.SeparableConvBlock.load_h5(weights_dir)
        f = h5py.File(weights_dir,'r')
        self.wBiFPNAdd.w.data = torch.from_numpy(np.array(f['model_weights'][f'fpn_cells/cell_{self.idx_BiFPN_layer}/fnode{self.node_idx}/add'][f'fpn_cells/cell_{self.idx_BiFPN_layer}/fnode{self.node_idx}/add'][f'fpn_cells/cell_{self.idx_BiFPN_layer}/fnode{self.node_idx}/add:0']))
        f.close()


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, name = ''):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=False, padding=kernel_size//2)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=1, bias=True)
        self.name = name

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

    def load_h5(self, weights_dir):
        f = h5py.File(weights_dir,'r')
        if 'net' in self.name:
            if 'iterative' in self.name:
                suffix = self.name.split('/')[0]+'/'
                name = self.name.replace(suffix,'',1)
                self.depthwise.weight.data = torch.from_numpy(np.transpose(np.array(f['model_weights'][name][suffix][name.split('/')[0]][name]['depthwise_kernel:0']), axes=[2,3,0,1]))
                self.pointwise.weight.data = torch.from_numpy(np.transpose(np.array(f['model_weights'][name][suffix][name.split('/')[0]][name]['pointwise_kernel:0'])))
                self.pointwise.bias.data = torch.from_numpy(np.array(f['model_weights'][name][suffix][name.split('/')[0]][name]['bias:0']))
            else:
                suffix = self.name.split('/')[0]+'/'
                self.depthwise.weight.data = torch.from_numpy(np.transpose(np.array(f['model_weights'][self.name.replace(suffix,'',1)][self.name]['depthwise_kernel:0']), axes=[2,3,0,1]))
                self.pointwise.weight.data = torch.from_numpy(np.transpose(np.array(f['model_weights'][self.name.replace(suffix,'',1)][self.name]['pointwise_kernel:0'])))
                self.pointwise.bias.data = torch.from_numpy(np.array(f['model_weights'][self.name.replace(suffix,'',1)][self.name]['bias:0']))
        else:
            self.depthwise.weight.data = torch.from_numpy(np.transpose(np.array(f['model_weights'][self.name]['conv'][self.name]['conv']['depthwise_kernel:0']), axes=[2,3,0,1]))
            self.pointwise.weight.data = torch.from_numpy(np.transpose(np.array(f['model_weights'][self.name]['conv'][self.name]['conv']['pointwise_kernel:0'])))
            self.pointwise.bias.data = torch.from_numpy(np.array(f['model_weights'][self.name]['conv'][self.name]['conv']['bias:0']))
        f.close()


class SeparableConvBlock(nn.Module):
    def __init__(
        self, num_channels, kernel_size, name = '', freeze_bn = False
    ):
        self.name = name
        super(SeparableConvBlock, self).__init__()
        self.f1 = SeparableConv2d(num_channels, num_channels, kernel_size, name = name)
        self.f2 = nn.BatchNorm2d(num_channels)

    def forward(self, input):
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (self.f1, self.f2))(input)

    def load_h5(self, weights_dir):
        self.f1.load_h5(weights_dir)
        f = h5py.File(weights_dir,'r')
        f['model_weights'][self.name]['bn'][self.name]['bn']
        self.f2.running_mean.data = torch.from_numpy(np.array(f['model_weights'][self.name]['bn'][self.name]['bn']['moving_mean:0']))
        self.f2.running_var.data = torch.from_numpy(np.array(f['model_weights'][self.name]['bn'][self.name]['bn']['moving_variance:0']))
        self.f2.weight.data = torch.from_numpy(np.array(f['model_weights'][self.name]['bn'][self.name]['bn']['gamma:0']))
        self.f2.bias.data = torch.from_numpy(np.array(f['model_weights'][self.name]['bn'][self.name]['bn']['beta:0']))
        f.close()

def build_subnets(num_classes, subnet_width, subnet_depth, subnet_num_iteration_steps, num_groups_gn, num_rotation_parameters, freeze_bn, num_anchors):
    """
    Builds the EfficientPose subnetworks
    Args:
        num_classes: Number of classes for the classification network output
        subnet_width: The number of channels used in the subnetwork layers
        subnet_depth: The number of layers used in the subnetworks
        subnet_num_iteration_steps: The number of iterative refinement steps used in the rotation and translation subnets
        num_groups_gn: The number of groups per group norm layer used in the rotation and translation subnets
        num_rotation_parameters: Number of rotation parameters, e.g. 3 for axis angle representation
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
        num_anchors: The number of anchors, usually 3 scales and 3 aspect ratios resulting in 3 * 3 = 9 anchors
    
    Returns:
       The subnetworks
    """
    box_net = BoxNet(subnet_width,
                     subnet_depth,
                     num_anchors = num_anchors,
                     freeze_bn = freeze_bn,
                     name = 'box_net')
    
    class_net = ClassNet(subnet_width,
                         subnet_depth,
                         num_classes = num_classes,
                         num_anchors = num_anchors,
                         freeze_bn = freeze_bn,
                         name = 'class_net')
    
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

    return box_net, class_net, rotation_net, translation_net   

class BoxNet(nn.Module):
    def __init__(self, width, depth, num_anchors = 9, freeze_bn = False, name=''):
        super(BoxNet, self).__init__()

        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = 4
        self.name = name

        input_shape = width
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.depth):
            self.convs.append(SeparableConv2d(input_shape, self.width, 3, name = f'{self.name}/{self.name}/box-{i}'))
            self.bns.append(nn.ModuleList())
            for j in range (3,8):
                self.bns[i].append(nn.BatchNorm2d(input_shape)),
            input_shape = width
        self.head = SeparableConv2d(input_shape, num_anchors*self.num_values, 3, name = f'{self.name}/{self.name}/box-predict')
        
    def forward (self, input):
        feature, level = input
        input_shape = feature.shape
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][level](feature)
            feature = nn.SiLU()(feature)
        output = self.head(feature)
        output = torch.reshape (output,(input_shape[0],-1,self.num_values))
        return output

    def load_h5(self, weights_dir):
        for i in range(self.depth):
            self.convs[i].load_h5(weights_dir)
        self.head.load_h5(weights_dir)
        
        f = h5py.File(weights_dir,'r')
        for i in range(self.depth):
            for j in range (3,8):
                if (j==3):
                    self.bns[i][j-3].running_mean.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/box-{i}-bn-{j}'][self.name][f'{self.name}/box-{i}-bn-{j}']['moving_mean:0']))
                    self.bns[i][j-3].running_var.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/box-{i}-bn-{j}'][self.name][f'{self.name}/box-{i}-bn-{j}']['moving_variance:0']))
                    self.bns[i][j-3].weight.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/box-{i}-bn-{j}'][self.name][f'{self.name}/box-{i}-bn-{j}']['gamma:0']))
                    self.bns[i][j-3].bias.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/box-{i}-bn-{j}'][self.name][f'{self.name}/box-{i}-bn-{j}']['beta:0']))
                else : 
                    self.bns[i][j-3].running_mean.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/box-{i}-bn-{j}'][self.name+f'_{j-3}'][f'{self.name}/box-{i}-bn-{j}']['moving_mean:0']))
                    self.bns[i][j-3].running_var.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/box-{i}-bn-{j}'][self.name+f'_{j-3}'][f'{self.name}/box-{i}-bn-{j}']['moving_variance:0']))
                    self.bns[i][j-3].weight.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/box-{i}-bn-{j}'][self.name+f'_{j-3}'][f'{self.name}/box-{i}-bn-{j}']['gamma:0']))
                    self.bns[i][j-3].bias.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/box-{i}-bn-{j}'][self.name+f'_{j-3}'][f'{self.name}/box-{i}-bn-{j}']['beta:0']))
        f.close()

        

class ClassNet(nn.Module):
    def __init__(self, width, depth, num_anchors = 9, num_classes = 8, freeze_bn = False, name=''):
        super(ClassNet, self).__init__()
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.name = name

        input_shape = width
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.depth):
            self.convs.append(SeparableConv2d(input_shape, self.width, 3, name = f'{self.name}/{self.name}/class-{i}'))
            self.bns.append(nn.ModuleList())
            for j in range (3,8):
                self.bns[i].append(nn.BatchNorm2d(input_shape)),
            input_shape = width
        self.head = SeparableConv2d(input_shape, self.num_classes * self.num_anchors, 3, name = f'{self.name}/{self.name}/class-predict')

    def forward (self, input):
        feature, level = input
        input_shape = feature.shape
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][level](feature)
            feature = nn.SiLU()(feature)
        output = self.head(feature)
        output = torch.reshape (output,(input_shape[0],-1,self.num_classes))
        output = nn.Sigmoid()(output)
        return output

    def load_h5(self, weights_dir):
        for i in range(self.depth):
            self.convs[i].load_h5(weights_dir)
        self.head.load_h5(weights_dir)
        
        f = h5py.File(weights_dir,'r')
        for i in range(self.depth):
            for j in range (3,8):
                if (j==3):
                    self.bns[i][j-3].running_mean.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/class-{i}-bn-{j}'][self.name][f'{self.name}/class-{i}-bn-{j}']['moving_mean:0']))
                    self.bns[i][j-3].running_var.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/class-{i}-bn-{j}'][self.name][f'{self.name}/class-{i}-bn-{j}']['moving_variance:0']))
                    self.bns[i][j-3].weight.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/class-{i}-bn-{j}'][self.name][f'{self.name}/class-{i}-bn-{j}']['gamma:0']))
                    self.bns[i][j-3].bias.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/class-{i}-bn-{j}'][self.name][f'{self.name}/class-{i}-bn-{j}']['beta:0']))
                else : 
                    self.bns[i][j-3].running_mean.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/class-{i}-bn-{j}'][self.name+f'_{j-3}'][f'{self.name}/class-{i}-bn-{j}']['moving_mean:0']))
                    self.bns[i][j-3].running_var.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/class-{i}-bn-{j}'][self.name+f'_{j-3}'][f'{self.name}/class-{i}-bn-{j}']['moving_variance:0']))
                    self.bns[i][j-3].weight.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/class-{i}-bn-{j}'][self.name+f'_{j-3}'][f'{self.name}/class-{i}-bn-{j}']['gamma:0']))
                    self.bns[i][j-3].bias.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/class-{i}-bn-{j}'][self.name+f'_{j-3}'][f'{self.name}/class-{i}-bn-{j}']['beta:0']))
        f.close()


class IterativeRotationSubNet(nn.Module):
    def __init__(self, width, depth, num_values, num_iteration_steps, num_anchors = 9, freeze_bn = False, use_group_norm = True, num_groups_gn = None, name=''):
        super(IterativeRotationSubNet, self).__init__()
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = num_values
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        self.name = name

        input_shape = num_anchors*num_values + width
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.depth):
            self.convs.append(SeparableConv2d(input_shape, self.width, 3, name = f'{self.name}/iterative-rotation-sub-{i}'))
            input_shape = width
        
        for k in range(num_iteration_steps):
            self.norms.append(nn.ModuleList())
            for i in range(self.depth):
                self.norms[k].append(nn.ModuleList()) 
                for j in range (3,8):
                    if (self.use_group_norm):
                        self.norms[k][i].append(nn.GroupNorm(self.num_groups_gn,input_shape)),
                    else:
                        self.norms[k][i].append(nn.BatchNorm2d(input_shape)),

        self.head = SeparableConv2d(input_shape, num_anchors*self.num_values, 3, name = f'{self.name}/iterative-rotation-sub-predict')

    def forward (self, inputs, **kwargs):
        feature, level = inputs
        level_py = kwargs["level_py"]
        iter_step_py = kwargs["iter_step_py"]
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norms[iter_step_py][i][level_py](feature)
            feature = nn.SiLU()(feature)
        outputs = self.head(feature)
        
        return outputs

    def load_h5(self, weights_dir):
        for i in range(self.depth):
            self.convs[i].load_h5(weights_dir)
        self.head.load_h5(weights_dir)

        f = h5py.File(weights_dir,'r')
        for k in range(self.num_iteration_steps):
            for i in range(self.depth):
                for j in range (3,8):
                    suffix = self.name.split('/')[0]
                    name = self.name.replace(suffix+'/','',1)
                    if (j-3):
                        suffix = f'{suffix}_{j-3}'
                    name = f'{name}/iterative-rotation-sub-{k}-{i}-gn-{j}'
                    self.norms[k][i][j-3].weight.data = torch.from_numpy(np.array(f['model_weights'][name][suffix][name.split('/')[0]][name]['gamma:0']))
                    self.norms[k][i][j-3].bias.data = torch.from_numpy(np.array(f['model_weights'][name][suffix][name.split('/')[0]][name]['beta:0']))
        f.close()


class RotationNet(nn.Module):
    def __init__(self, width, depth, num_values, num_iteration_steps, num_anchors = 9, freeze_bn = False, use_group_norm = True, num_groups_gn = None, name=''):
        super(RotationNet, self).__init__()
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = num_values
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        self.name = name

        input_shape = width
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.depth):
            self.convs.append(SeparableConv2d(input_shape, self.width, 3, name = f'{self.name}/{self.name}/rotation-{i}'))
            input_shape = width
            self.norms.append(nn.ModuleList())
            for j in range (3,8):
                if (self.use_group_norm):
                    self.norms[i].append(nn.GroupNorm(self.num_groups_gn,input_shape)),
                else:
                    self.norms[i].append(nn.BatchNorm2d(input_shape)),

        self.initial_rotation = SeparableConv2d(input_shape, self.num_anchors*self.num_values, 3, name = f'{self.name}/{self.name}/rotation-init-predict')

        self.iterative_submodel = IterativeRotationSubNet(width = self.width,
                                                          depth = self.depth - 1,
                                                          num_values = self.num_values,
                                                          num_iteration_steps = self.num_iteration_steps,
                                                          num_anchors = self.num_anchors,
                                                          freeze_bn = freeze_bn,
                                                          use_group_norm = self.use_group_norm,
                                                          num_groups_gn = self.num_groups_gn,
                                                          name = f'{self.name}/iterative_rotation_subnet')

    def forward (self, input):
        feature, level = input
        input_shape = feature.shape
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norms[i][level](feature)
            feature = nn.SiLU()(feature)

        rotation = self.initial_rotation(feature)

        for i in range(self.num_iteration_steps):
            iterative_input = torch.cat((feature, rotation), dim=1)
            delta_rotation = self.iterative_submodel([iterative_input, level], level_py = level, iter_step_py = i)
            rotation = rotation + delta_rotation
        
        output = torch.reshape (rotation,(input_shape[0],-1,self.num_values))

        return output

    def load_h5(self, weights_dir):
        for i in range(self.depth):
            self.convs[i].load_h5(weights_dir)
        self.initial_rotation.load_h5(weights_dir)
        self.iterative_submodel.load_h5(weights_dir)
        
        f = h5py.File(weights_dir,'r')
        for i in range(self.depth):
            for j in range (3,8):
                if (j==3):
                    self.norms[i][j-3].weight.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/rotation-{i}-gn-{j}'][self.name][f'{self.name}/rotation-{i}-gn-{j}']['gamma:0']))
                    self.norms[i][j-3].bias.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/rotation-{i}-gn-{j}'][self.name][f'{self.name}/rotation-{i}-gn-{j}']['beta:0']))
                else : 
                    self.norms[i][j-3].weight.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/rotation-{i}-gn-{j}'][self.name+f'_{j-3}'][f'{self.name}/rotation-{i}-gn-{j}']['gamma:0']))
                    self.norms[i][j-3].bias.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/rotation-{i}-gn-{j}'][self.name+f'_{j-3}'][f'{self.name}/rotation-{i}-gn-{j}']['beta:0']))
        f.close()


class IterativeTranslationSubNet(nn.Module):
    def __init__(self, width, depth, num_iteration_steps, num_anchors = 9, freeze_bn = False, use_group_norm = True, num_groups_gn = None, name=''):
        super(IterativeTranslationSubNet, self).__init__()
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        self.name = name

        input_shape = num_anchors*3 + width
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.depth):
            self.convs.append(SeparableConv2d(input_shape, self.width, 3, name = f'{self.name}/iterative-translation-sub-{i}'))
            input_shape = width
        
        for k in range(num_iteration_steps):
            self.norms.append(nn.ModuleList())
            for i in range(self.depth):
                self.norms[k].append(nn.ModuleList()) 
                for j in range (3,8):
                    if (self.use_group_norm):
                        self.norms[k][i].append(nn.GroupNorm(self.num_groups_gn,input_shape)),
                    else:
                        self.norms[k][i].append(nn.BatchNorm2d(input_shape)),

        self.head_xy = SeparableConv2d(input_shape, num_anchors*2, 3, name = f'{self.name}/iterative-translation-xy-sub-predict')
        self.head_z = SeparableConv2d(input_shape, num_anchors, 3, name = f'{self.name}/iterative-translation-z-sub-predict')

    def forward(self, inputs, **kwargs):
        feature, level = inputs
        level_py = kwargs["level_py"]
        iter_step_py = kwargs["iter_step_py"]
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norms[iter_step_py][i][level_py](feature)
            feature = nn.SiLU()(feature)
        outputs_xy = self.head_xy(feature)
        outputs_z = self.head_z(feature)

        return outputs_xy, outputs_z

    def load_h5(self, weights_dir):
        for i in range(self.depth):
            self.convs[i].load_h5(weights_dir)
        self.head_xy.load_h5(weights_dir)
        self.head_z.load_h5(weights_dir)

        f = h5py.File(weights_dir,'r')
        for k in range(self.num_iteration_steps):
            for i in range(self.depth):
                for j in range (3,8):
                    suffix = self.name.split('/')[0]
                    name = self.name.replace(suffix+'/','',1)
                    if (j-3):
                        suffix = f'{suffix}_{j-3}'
                    name = f'{name}/iterative-translation-sub-{k}-{i}-gn-{j}'
                    self.norms[k][i][j-3].weight.data = torch.from_numpy(np.array(f['model_weights'][name][suffix][name.split('/')[0]][name]['gamma:0']))
                    self.norms[k][i][j-3].bias.data = torch.from_numpy(np.array(f['model_weights'][name][suffix][name.split('/')[0]][name]['beta:0']))
        f.close()


class TranslationNet(nn.Module):
    def __init__(self, width, depth, num_iteration_steps, num_anchors = 9, freeze_bn = False, use_group_norm = True, num_groups_gn = None, name=''):
        super(TranslationNet, self).__init__()
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        self.name = name

        input_shape = width
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.depth):
            self.convs.append(SeparableConv2d(input_shape, self.width, 3, name = f'{self.name}/{self.name}/translation-{i}'))
            input_shape = width
            self.norms.append(nn.ModuleList())
            for j in range (3,8):
                if (self.use_group_norm):
                    self.norms[i].append(nn.GroupNorm(self.num_groups_gn,input_shape)),
                else:
                    self.norms[i].append(nn.BatchNorm2d(input_shape)),

        self.initial_translation_xy = SeparableConv2d(input_shape, self.num_anchors*2, 3, name = f'{self.name}/{self.name}/translation-xy-init-predict')
        self.initial_translation_z = SeparableConv2d(input_shape, self.num_anchors*1, 3, name = f'{self.name}/{self.name}/translation-z-init-predict')

        self.iterative_submodel = IterativeTranslationSubNet(width = self.width,
                                                             depth = self.depth - 1,
                                                             num_iteration_steps = self.num_iteration_steps,
                                                             num_anchors = self.num_anchors,
                                                             freeze_bn = freeze_bn,
                                                             use_group_norm= self.use_group_norm,
                                                             num_groups_gn = self.num_groups_gn,
                                                             name = f'{self.name}/iterative_translation_subnet')

    def forward(self, inputs, **kwargs):
        feature, level = inputs
        input_shape = feature.shape
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norms[i][level](feature)
            feature = nn.SiLU()(feature)
            
        translation_xy = self.initial_translation_xy(feature)
        translation_z = self.initial_translation_z(feature)
        
        for i in range(self.num_iteration_steps):
            iterative_input = torch.cat((feature, translation_xy, translation_z), dim=1)
            delta_translation_xy, delta_translation_z = self.iterative_submodel([iterative_input, level], level_py = level, iter_step_py = i)
            translation_xy = translation_xy + delta_translation_xy
            translation_z = translation_z + delta_translation_z
        
        outputs_xy = torch.reshape(translation_xy, (input_shape[0],-1,2))
        outputs_z = torch.reshape(translation_z, (input_shape[0],-1,1))
        outputs = torch.cat((outputs_xy, outputs_z), dim=-1)
        return outputs

    def load_h5(self, weights_dir):
        for i in range(self.depth):
            self.convs[i].load_h5(weights_dir)
        self.initial_translation_xy.load_h5(weights_dir)
        self.initial_translation_z.load_h5(weights_dir)
        self.iterative_submodel.load_h5(weights_dir)
        
        f = h5py.File(weights_dir,'r')
        for i in range(self.depth):
            for j in range (3,8):
                if (j==3):
                    self.norms[i][j-3].weight.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/translation-{i}-gn-{j}'][self.name][f'{self.name}/translation-{i}-gn-{j}']['gamma:0']))
                    self.norms[i][j-3].bias.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/translation-{i}-gn-{j}'][self.name][f'{self.name}/translation-{i}-gn-{j}']['beta:0']))
                else : 
                    self.norms[i][j-3].weight.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/translation-{i}-gn-{j}'][self.name+f'_{j-3}'][f'{self.name}/translation-{i}-gn-{j}']['gamma:0']))
                    self.norms[i][j-3].bias.data = torch.from_numpy(np.array(f['model_weights'][f'{self.name}/translation-{i}-gn-{j}'][self.name+f'_{j-3}'][f'{self.name}/translation-{i}-gn-{j}']['beta:0']))
        f.close()




if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device: ", DEVICE)

    phi = 0
    num_rotation_parameters = 3
    num_classes = 1
    num_anchors = 9
    freeze_bn = True
    score_threshold = 0.7

    scaled_parameters = get_scaled_parameters(phi)
    input_size = scaled_parameters["input_size"]
    subnet_width = scaled_parameters["bifpn_width"]
    subnet_depth = scaled_parameters["subnet_depth"]
    subnet_num_iteration_steps = scaled_parameters["subnet_num_iteration_steps"]
    num_groups_gn = scaled_parameters["num_groups_gn"]

    model = EfficientPose(phi,
                          num_classes = num_classes,
                          num_anchors = num_anchors,
                          freeze_bn = not freeze_bn,
                          score_threshold = score_threshold,
                          num_rotation_parameters = num_rotation_parameters).to(DEVICE)
    

    model.load_h5('phi_0_linemod_best_ADD.h5')
    model = model.to(DEVICE)

    input = torch.randn((4, 3, input_size, input_size)).to(DEVICE)
    k = torch.randn(4,1,6).to(DEVICE)

    print("Input shape: ", input.shape)
    print("Input k shape: ", k.shape)

    pred = model(input,k)
    for i in pred:
        print (i.shape)
