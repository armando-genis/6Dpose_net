from construct import Const
import torch
import torch.nn as nn

class wBiFPNAdd(nn.Module):
    """
    Layer that computes a weighted sum of BiFPN feature maps
    """
    def __init__(self, input_shape, epsilon=1e-4):
        super(wBiFPNAdd, self).__init__()
        self.input_shape = input_shape
        self.epsilon = epsilon
        self.w = nn.Parameter(data=torch.Tensor(input_shape), requires_grad=True)
        self.w.data.fill_(1/input_shape)
        self.relu = nn.ReLU()
    
    def forward(self, input):
        w = self.relu(self.w)
        output = 0
        for i in range(self.input_shape):
            output = output + input[i]*w[i]
        output = output/torch.sum(w)

        return output


def bbox_transform_inv(boxes, deltas, scale_factors = None):
    """
    Reconstructs the 2D bounding boxes using the anchor boxes and the predicted deltas of the anchor boxes to the bounding boxes
    Args:
        boxes: Tensor containing the anchor boxes with shape (..., 4)
        deltas: Tensor containing the offsets of the anchor boxes to the bounding boxes with shape (..., 4)
        scale_factors: optional scaling factor for the deltas
    Returns:
        Tensor containing the reconstructed 2D bounding boxes with shape (..., 4)

    """
    cxa = (boxes[..., 0] + boxes[..., 2]) / 2
    cya = (boxes[..., 1] + boxes[..., 3]) / 2
    wa = boxes[..., 2] - boxes[..., 0]
    ha = boxes[..., 3] - boxes[..., 1]
    ty, tx, th, tw = deltas[..., 0], deltas[..., 1], deltas[..., 2], deltas[..., 3]
    if scale_factors:
        ty *= scale_factors[0]
        tx *= scale_factors[1]
        th *= scale_factors[2]
        tw *= scale_factors[3]
    w = torch.exp(tw) * wa
    h = torch.exp(th) * ha
    cy = ty * ha + cya
    cx = tx * wa + cxa
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    return torch.cat([torch.unsqueeze(xmin,-1), torch.unsqueeze(ymin,-1), torch.unsqueeze(xmax,-1), torch.unsqueeze(ymax,-1)], axis=-1)


def translation_transform_inv(translation_anchors, deltas, scale_factors = None):
    """ Applies the predicted 2D translation center point offsets (deltas) to the translation_anchors

    Args
        translation_anchors : Tensor of shape (B, N, 3), where B is the batch size, N the number of boxes and 2 values for (x, y) +1 value with the stride.
        deltas: Tensor of shape (B, N, 3). The first 2 deltas (d_x, d_y) are a factor of the stride +1 with Tz.

    Returns
        A tensor of the same shape as translation_anchors, but with deltas applied to each translation_anchors and the last coordinate is the concatenated (untouched) Tz value from deltas.
    """

    stride  = translation_anchors[:, :, -1]

    if scale_factors:
        x = translation_anchors[:, :, 0] + (deltas[:, :, 0] * scale_factors[0] * stride)
        y = translation_anchors[:, :, 1] + (deltas[:, :, 1] * scale_factors[1] * stride)
    else:
        x = translation_anchors[:, :, 0] + (deltas[:, :, 0] * stride)
        y = translation_anchors[:, :, 1] + (deltas[:, :, 1] * stride)
    
    Tz = deltas[:, :, 2]

    pred_translations = torch.cat([torch.unsqueeze(x,-1), torch.unsqueeze(y,-1), torch.unsqueeze(Tz,-1)], dim = -1) #x,y 2D Image coordinates and Tz

    return pred_translations


def ClipBoxes(inputs):
    """
    Layer that clips 2D bounding boxes so that they are inside the image
    """
    shape, boxes = inputs
    height = shape[0]
    width = shape[1]
    x1 = torch.clamp(boxes[:, :, 0], min = 0, max = width - 1)
    y1 = torch.clamp(boxes[:, :, 1], min = 0, max = height - 1)
    x2 = torch.clamp(boxes[:, :, 2], min = 0, max = width - 1)
    y2 = torch.clamp(boxes[:, :, 3], min = 0, max = height - 1)

    return torch.concat([torch.unsqueeze(x1,-1), torch.unsqueeze(y1,-1), torch.unsqueeze(x2,-1), torch.unsqueeze(y2,-1)], axis=-1)

class RegressBoxes(nn.Module):
    """ 
    Pytorch layer for applying regression offset values to anchor boxes to get the 2D bounding boxes.
    """
    def __init__(self, *args, name =''):
        super(RegressBoxes, self).__init__()

    def forward(self, inputs):
        anchors, regression = inputs
        return bbox_transform_inv(anchors, regression)


class RegressTranslation(nn.Module):
    """ 
    Pytorch layer for applying regression offset values to translation anchors to get the 2D translation centerpoint and Tz.
    """

    def __init__(self, name=''):
        """
        Initializer for the RegressTranslation layer.
        """
        self.name = name
        super(RegressTranslation, self).__init__()

    def forward(self, inputs):
        translation_anchors, regression_offsets = inputs
        return translation_transform_inv(translation_anchors, regression_offsets)


class CalculateTxTy(nn.Module):
    """ 
    Pytorch layer for calculating the Tx- and Ty-Components of the Translationvector with a given 2D-point and the intrinsic camera parameters.
    """

    def __init__(self, name=''):
        """ 
        Initializer for an CalculateTxTy layer.
        """
        super(CalculateTxTy, self).__init__()

    def forward(self, inputs, fx = 572.4114, fy = 573.57043, px = 325.2611, py = 242.04899, tz_scale = 1000.0, image_scale = 1.6666666666666667):
        # Tx = (cx - px) * Tz / fx
        # Ty = (cy - py) * Tz / fy
        
        #fx = torch.unsqueeze(fx, dim = -1)
        #fy = torch.unsqueeze(fy, dim = -1)
        #px = torch.unsqueeze(px, dim = -1)
        #py = torch.unsqueeze(py, dim = -1)
        #tz_scale = torch.unsqueeze(tz_scale, dim = -1)
        #image_scale = torch.unsqueeze(image_scale, dim = -1)

        x = inputs[:, :, 0] / image_scale
        y = inputs[:, :, 1] / image_scale
        tz = inputs[:, :, 2] * tz_scale
        
        x = x - px
        y = y - py
        
        tx = x*tz/ fx
        ty = y*tz/ fy
        
        output = torch.cat([torch.unsqueeze(tx,-1), torch.unsqueeze(ty,-1), torch.unsqueeze(tz,-1)], dim = -1)
        
        return output
    
if __name__ == '__main__':
    input_shape = 4
    test = wBiFPNAdd(2)
    input1 = torch.arange(1, input_shape*2*2+1, dtype=torch.float32).view(1, input_shape, 2, 2)
    input2 = torch.arange(input_shape*2*2+1, input_shape*2*2*2+1, dtype=torch.float32).view(1, input_shape, 2, 2)
    print(test([input1,input2]))
