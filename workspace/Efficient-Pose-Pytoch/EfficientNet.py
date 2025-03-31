import collections
import h5py
import numpy as np

import string

import torch
import torch.nn as nn
import math

global all_abs_diff_sum
all_abs_diff_sum = 0

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])

BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=1, se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=2, se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=2, se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=2, se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=1, se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=2, se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=1, se_ratio=0.25)
]

def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)

def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""

    return int(math.ceil(depth_coefficient * repeats))

class CNNBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride, padding, groups=1, name = ''
    ):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.name = name
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU() # SiLU <-> Swish

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))

    def load_h5(self, weights_dir):
        f = h5py.File(weights_dir,'r')
        global all_abs_diff_sum
        test =  torch.tensor(self.cnn.weight.data.shape)
        if 'dw' in self.name:
            self.cnn.weight.data = torch.from_numpy(np.transpose(np.array(f['model_weights'][self.name+'conv'][self.name+'conv']['depthwise_kernel:0']), axes=[2,3,0,1]))
            self.bn.running_mean.data = torch.from_numpy(np.array(f['model_weights'][self.name.replace('dw','bn')][self.name.replace('dw','bn')]['moving_mean:0']))
            self.bn.running_var.data = torch.from_numpy(np.array(f['model_weights'][self.name.replace('dw','bn')][self.name.replace('dw','bn')]['moving_variance:0']))
            self.bn.weight.data = torch.from_numpy(np.array(f['model_weights'][self.name.replace('dw','bn')][self.name.replace('dw','bn')]['gamma:0']))
            self.bn.bias.data = torch.from_numpy(np.array(f['model_weights'][self.name.replace('dw','bn')][self.name.replace('dw','bn')]['beta:0']))
        else:
            self.cnn.weight.data = torch.from_numpy(np.transpose(np.array(f['model_weights'][self.name+'conv'][self.name+'conv']['kernel:0'])))
            self.bn.running_mean.data = torch.from_numpy(np.array(f['model_weights'][self.name+'bn'][self.name+'bn']['moving_mean:0']))
            self.bn.running_var.data = torch.from_numpy(np.array(f['model_weights'][self.name+'bn'][self.name+'bn']['moving_variance:0']))
            self.bn.weight.data = torch.from_numpy(np.array(f['model_weights'][self.name+'bn'][self.name+'bn']['gamma:0']))
            self.bn.bias.data = torch.from_numpy(np.array(f['model_weights'][self.name+'bn'][self.name+'bn']['beta:0']))
        all_abs_diff_sum = all_abs_diff_sum + torch.sum(torch.abs(test - torch.tensor(self.cnn.weight.data.shape)))
        f.close()

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim, name =''):
        super(SqueezeExcitation, self).__init__()
        self.name = name
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)

    def load_h5(self, weights_dir):
        f = h5py.File(weights_dir,'r')
        global all_abs_diff_sum
        test =  torch.tensor(self.se[1].weight.data.shape)
        self.se[1].weight.data = torch.from_numpy(np.transpose(np.array(f['model_weights'][self.name+'se_reduce'][self.name+'se_reduce']['kernel:0'])))
        self.se[1].bias.data = torch.from_numpy(np.array(f['model_weights'][self.name+'se_reduce'][self.name+'se_reduce']['bias:0']))
        all_abs_diff_sum = all_abs_diff_sum + torch.sum(torch.abs(test - torch.tensor(self.se[1].weight.data.shape)))
        test =  torch.tensor(self.se[3].weight.data.shape)
        self.se[3].weight.data = torch.from_numpy(np.transpose(np.array(f['model_weights'][self.name+'se_expand'][self.name+'se_expand']['kernel:0'])))
        self.se[3].bias.data = torch.from_numpy(np.array(f['model_weights'][self.name+'se_expand'][self.name+'se_expand']['bias:0']))
        all_abs_diff_sum = all_abs_diff_sum + torch.sum(torch.abs(test - torch.tensor(self.se[3].weight.data.shape)))
        f.close()

class mb_conv_block(nn.Module):
    """Mobile Inverted Residual Bottleneck."""

    def __init__(
            self,
            block_args,
            drop_rate=None,
            prefix='',
            freeze_bn=False,
    ):

        super(mb_conv_block, self).__init__()

        self.block_args = block_args
        self.survival_prob = 1 - drop_rate
        self.prefix = prefix
        self.freeze_bn = freeze_bn

        self.has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
        self.filters = self.block_args.input_filters * self.block_args.expand_ratio
        self.num_reduced_filters = max(1, int(
                self.block_args.input_filters * self.block_args.se_ratio
            ))

        #operations:
        self.expansion_phase = CNNBlock(self.block_args.input_filters, self.filters, 1, 1, 0, name=prefix+'expand_')
        self.depthwise_conv = CNNBlock(self.filters, self.filters, self.block_args.kernel_size, self.block_args.strides, self.block_args.kernel_size//2, groups=self.filters, name=prefix+'dw')
        self.squeezeNexcitation = SqueezeExcitation(self.filters, self.num_reduced_filters, name=self.prefix)
        self.output_phase = nn.Sequential(
            nn.Conv2d(self.filters,self.block_args.output_filters,1,bias=False),
            nn.BatchNorm2d(self.block_args.output_filters)
        )
        
    def forward(self, input):
        # Expansion phase
        if self.block_args.expand_ratio != 1:
            x = self.expansion_phase(input)
        else:
            x = input
        # Depthwise Convolution
        x = self.depthwise_conv(x)
        # Squeeze and Excitation phase
        if self.has_se:
            se_tensor = self.squeezeNexcitation(x)
            x = x * se_tensor

        #Output phase
        x = self.output_phase(x)
        if self.block_args.id_skip and self.block_args.strides == 1 and self.block_args.input_filters == self.block_args.output_filters:
            if self.training:
                binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
                x = torch.div(x, self.survival_prob) * binary_tensor
                x = x + input

        return x

    def load_h5(self, weights_dir):
        # Expansion phase
        if self.block_args.expand_ratio != 1:
            self.expansion_phase.load_h5(weights_dir)
        # Depthwise Convolution
        self.depthwise_conv.load_h5(weights_dir)
        # Squeeze and Excitation phase
        if self.has_se:
            self.squeezeNexcitation.load_h5(weights_dir)

        f = h5py.File(weights_dir,'r')
        global all_abs_diff_sum
        test =  torch.tensor(self.output_phase[0].weight.data.shape)
        self.output_phase[0].weight.data = torch.from_numpy(np.transpose(np.array(f['model_weights'][self.prefix+'project_conv'][self.prefix+'project_conv']['kernel:0'])))     
        self.output_phase[1].running_mean.data = torch.from_numpy(np.array(f['model_weights'][self.prefix+'project_bn'][self.prefix+'project_bn']['moving_mean:0']))
        self.output_phase[1].running_var.data = torch.from_numpy(np.array(f['model_weights'][self.prefix+'project_bn'][self.prefix+'project_bn']['moving_variance:0']))
        self.output_phase[1].weight.data = torch.from_numpy(np.array(f['model_weights'][self.prefix+'project_bn'][self.prefix+'project_bn']['gamma:0']))
        self.output_phase[1].bias.data = torch.from_numpy(np.array(f['model_weights'][self.prefix+'project_bn'][self.prefix+'project_bn']['beta:0']))
        all_abs_diff_sum = all_abs_diff_sum + torch.sum(torch.abs(test - torch.tensor(self.output_phase[0].weight.data.shape)))
        f.close()

class EfficientNet(nn.Module):
    def __init__(
        self,
        width_coefficient,
        depth_coefficient,
        default_resolution,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        depth_divisor=8,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        model_name='efficientnet',
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        freeze_bn=False
    ):
        """Instantiates the EfficientNet architecture using given scaling coefficients.
        
        # Arguments
            width_coefficient: float, scaling coefficient for network width.
            depth_coefficient: float, scaling coefficient for network depth.
            default_resolution: int, default input image size.
            dropout_rate: float, dropout rate before final classifier layer.
            drop_connect_rate: float, dropout rate at skip connections.
            depth_divisor: int.
            blocks_args: A list of BlockArgs to construct block modules.
            model_name: string, model name.
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: one of `None` (random initialization),
                'imagenet' (pre-training on ImageNet),
                or the path to the weights file to be loaded.
            input_tensor: optional Keras tensor
                (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False.
                It should have exactly 3 inputs channels.
            pooling: optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional layer.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional layer, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                    be applied.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
        # Returns
            A pytoch model instance.
        """
        super(EfficientNet, self).__init__()
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_resolution = default_resolution
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.depth_divisor = depth_divisor
        self.blocks_args = blocks_args
        self.model_name = model_name
        self.include_top = include_top
        self.weights = weights
        self.input_shape = input_shape
        self.pooling = pooling
        self.classes = classes
        self.freeze_bn = freeze_bn

        self.blocks = nn.ModuleList()
        for block in self.blocks_args:
            self.blocks.append(nn.ModuleList())
        
        #Build stem
        self.stem = CNNBlock(self.input_shape[0],round_filters(32, self.width_coefficient, self.depth_divisor),3,2,1, name = 'stem_')
        #Build blocks
        num_blocks_total = sum(block_args.num_repeat for block_args in self.blocks_args)
        block_num = 0
        for idx, block_args in enumerate(self.blocks_args):
            assert block_args.num_repeat > 0
        # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters,
                                            self.width_coefficient, self.depth_divisor),
                output_filters=round_filters(block_args.output_filters,
                                            self.width_coefficient, self.depth_divisor),
                num_repeat=round_repeats(block_args.num_repeat, self.depth_coefficient))

            # The first block needs to take care of stride and filter size increase.
            drop_rate = self.drop_connect_rate * float(block_num) / num_blocks_total
            self.blocks[idx].append(mb_conv_block(block_args, drop_rate=drop_rate, prefix='block{}a_'.format(idx + 1), freeze_bn=self.freeze_bn))
            block_num += 1
            block_num += 1
            if block_args.num_repeat > 1:
                # pylint: disable=protected-access
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1, 1])
                # pylint: enable=protected-access
                for bidx in range(block_args.num_repeat - 1):
                    drop_rate = self.drop_connect_rate * float(block_num) / num_blocks_total
                    block_prefix = 'block{}{}_'.format(
                        idx + 1,
                        string.ascii_lowercase[bidx + 1]
                    )
                    self.blocks[idx].append(mb_conv_block(block_args, drop_rate=drop_rate, prefix=block_prefix, freeze_bn=self.freeze_bn))

    def load_h5(self, weights_dir):
        self.stem.load_h5 (weights_dir)
        for i, block_args in enumerate(self.blocks_args):
            for j, block in enumerate(self.blocks[i]):
                self.blocks[i][j].load_h5(weights_dir)

    def forward (self, input):

        features = []

        x = self.stem(input)

        for idx, block_args in enumerate(self.blocks_args):
            for block in self.blocks[idx]:
                x = block(x)
            #saving the features
            if idx < len(self.blocks_args) - 1 and self.blocks_args[idx + 1].strides == 2:
                features.append(x)
            elif idx == len(self.blocks_args) - 1:
                features.append(x)
                
        return features

def EfficientNetB0(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,):
    return EfficientNet(1.0, 1.0, 224, 0.2,
                        model_name='efficientnet-b0',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes)


def EfficientNetB1(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000):
    return EfficientNet(1.0, 1.1, 240, 0.2,
                        model_name='efficientnet-b1',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes)


def EfficientNetB2(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000):
    return EfficientNet(1.1, 1.2, 260, 0.3,
                        model_name='efficientnet-b2',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes)


def EfficientNetB3(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000):
    return EfficientNet(1.2, 1.4, 300, 0.3,
                        model_name='efficientnet-b3',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes)

def EfficientNetB4(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000):
    return EfficientNet(1.4, 1.8, 380, 0.4,
                        model_name='efficientnet-b4',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes)


def EfficientNetB5(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000):
    return EfficientNet(1.6, 2.2, 456, 0.4,
                        model_name='efficientnet-b5',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes)

def EfficientNetB6(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000):
    return EfficientNet(1.8, 2.6, 528, 0.5,
                        model_name='efficientnet-b6',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes)

def EfficientNetB7(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000):
    return EfficientNet(2.0, 3.1, 600, 0.5,
                        model_name='efficientnet-b7',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes)


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = EfficientNetB0(input_shape = (3,512,512))
    model.load_h5('efficient_pose.h5')
    model = model.to(DEVICE)
    print (all_abs_diff_sum)

    input = torch.randn((1,3,512,512)).to(DEVICE)
    pred = model(input)
    for i in pred:
        print (i.shape)
