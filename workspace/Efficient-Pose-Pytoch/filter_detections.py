import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

def filter_detections(
    boxes,
    classification,
    rotation,
    translation,
    num_rotation_parameters,
    num_translation_parameters=3,
    class_specific_filter=True,
    nms=True,
    score_threshold=0.01,
    max_detections=100,
    nms_threshold=0.5,
):
    device = boxes.device
    
    def _filter_detections(scores, labels):
        # threshold based on score
        indices = torch.where(scores > score_threshold)[0]
        
        if nms and len(indices) > 0:
            # gather filtered boxes, scores
            filtered_boxes = boxes[indices]
            filtered_scores = scores[indices]
            
            # Debug: print max score before NMS
            if len(filtered_scores) > 0:
                print(f"Max score before NMS: {filtered_scores.max().item()}")
            
            # perform NMS - use torchvision's implementation
            if len(filtered_boxes) > 0:
                nms_indices = ops.nms(filtered_boxes, filtered_scores, nms_threshold)
                indices = indices[nms_indices]
            
        # add indices to list of all indices
        labels = labels[indices]
        
        return indices, labels
    
    # Debug: print overall max classification score
    max_class_score = classification.max()
    print(f"Max classification score: {max_class_score.item()}")
    
    if class_specific_filter:
        all_indices = []
        all_labels = []
        
        # perform per class filtering
        for c in range(classification.shape[1]):
            scores = classification[:, c]
            labels = torch.full((scores.shape[0],), c, dtype=torch.int64, device=device)
            indices, class_labels = _filter_detections(scores, labels)
            
            # Debug: print max score for this class
            if len(scores) > 0:
                print(f"Class {c}: Max score = {scores.max().item()}, Detections = {len(indices)}")
            
            if len(indices) > 0:
                all_indices.append(indices)
                all_labels.append(class_labels)
        
        # concatenate indices and labels
        if all_indices:
            indices = torch.cat(all_indices)
            labels = torch.cat(all_labels)
        else:
            indices = torch.tensor([], dtype=torch.int64, device=device)
            labels = torch.tensor([], dtype=torch.int64, device=device)
    else:
        scores, labels = classification.max(dim=1)
        indices, labels = _filter_detections(scores, labels)
    
    # Check if we have any valid detections
    if len(indices) > 0:
        # select top k
        scores = classification[indices, labels]
        
        # Sort by score
        scores, sort_idx = torch.sort(scores, descending=True)
        indices = indices[sort_idx]
        labels = labels[sort_idx]
        
        # select top k after sorting
        if len(scores) > max_detections:
            scores = scores[:max_detections]
            indices = indices[:max_detections]
            labels = labels[:max_detections]
        
        # get the final detections
        boxes_output = boxes[indices]
        scores_output = scores
        labels_output = labels
        rotation_output = rotation[indices]
        translation_output = translation[indices]
        
        # Debug: print final detection info
        print(f"Final detections: {len(scores)}, Max score: {scores.max().item() if len(scores) > 0 else 0}")
        
        # pad to fixed size if necessary
        pad_size = max_detections - len(scores)
        if pad_size > 0:
            # Pad boxes
            pad_boxes = torch.full((pad_size, 4), -1, dtype=boxes_output.dtype, device=device)
            boxes_output = torch.cat([boxes_output, pad_boxes], dim=0)
            
            # Pad scores
            pad_scores = torch.full((pad_size,), -1, dtype=scores_output.dtype, device=device)
            scores_output = torch.cat([scores_output, pad_scores], dim=0)
            
            # Pad labels
            pad_labels = torch.full((pad_size,), -1, dtype=labels_output.dtype, device=device)
            labels_output = torch.cat([labels_output, pad_labels], dim=0)
            
            # Pad rotation
            pad_rotation = torch.full((pad_size, num_rotation_parameters), -1, dtype=rotation_output.dtype, device=device)
            rotation_output = torch.cat([rotation_output, pad_rotation], dim=0)
            
            # Pad translation
            pad_translation = torch.full((pad_size, num_translation_parameters), -1, dtype=translation_output.dtype, device=device)
            translation_output = torch.cat([translation_output, pad_translation], dim=0)
    else:
        # No valid detections
        print("No detections found above threshold!")
        boxes_output = torch.full((max_detections, 4), -1, dtype=boxes.dtype, device=device)
        scores_output = torch.full((max_detections,), -1, dtype=torch.float32, device=device)
        labels_output = torch.full((max_detections,), -1, dtype=torch.int64, device=device)
        rotation_output = torch.full((max_detections, num_rotation_parameters), -1, dtype=rotation.dtype, device=device)
        translation_output = torch.full((max_detections, num_translation_parameters), -1, dtype=translation.dtype, device=device)
    
    return [boxes_output, scores_output, labels_output, rotation_output, translation_output]

class FilterDetections(nn.Module):
    """
    PyTorch module for filtering detections using score threshold and NMS.
    """
    
    def __init__(
        self,
        num_rotation_parameters,
        num_translation_parameters=3,
        nms=True,
        class_specific_filter=True,
        nms_threshold=0.5,
        score_threshold=0.01,
        max_detections=100,
        name=None
    ):
        """
        Filters detections using score threshold, NMS and selecting the top-k detections.
        
        Args:
            num_rotation_parameters: Number of rotation parameters, usually 3 for axis angle representation
            num_translation_parameters: Number of translation parameters, usually 3 
            nms: Flag to enable/disable NMS.
            class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold: Threshold used to prefilter the boxes with.
            max_detections: Maximum number of detections to keep.
            name: Optional name for the module.
        """
        super(FilterDetections, self).__init__()
        self.num_rotation_parameters = num_rotation_parameters
        self.num_translation_parameters = num_translation_parameters
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.name = name
    
    def forward(self, inputs):
        """
        Applies filtering to the input detections.
        
        Args:
            inputs: List of [boxes, classification, rotation, translation] tensors.
        
        Returns:
            List of [boxes, scores, labels, rotation, translation] tensors, with fixed shape.
        """
        boxes = inputs[0]
        classification = inputs[1]
        rotation = inputs[2]
        translation = inputs[3]
        
        # Process each batch item separately
        batch_size = boxes.shape[0]
        result_boxes = []
        result_scores = []
        result_labels = []
        result_rotation = []
        result_translation = []
        
        for i in range(batch_size):
            # Apply filter_detections on each batch item
            box_results = filter_detections(
                boxes[i],
                classification[i],
                rotation[i],
                translation[i],
                self.num_rotation_parameters,
                self.num_translation_parameters,
                nms=self.nms,
                class_specific_filter=self.class_specific_filter,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                nms_threshold=self.nms_threshold,
            )
            
            result_boxes.append(box_results[0])
            result_scores.append(box_results[1])
            result_labels.append(box_results[2])
            result_rotation.append(box_results[3])
            result_translation.append(box_results[4])
        
        # Stack results into batch tensors
        result_boxes = torch.stack(result_boxes, dim=0)
        result_scores = torch.stack(result_scores, dim=0)
        result_labels = torch.stack(result_labels, dim=0)
        result_rotation = torch.stack(result_rotation, dim=0)
        result_translation = torch.stack(result_translation, dim=0)
        
        return [result_boxes, result_scores, result_labels, result_rotation, result_translation]