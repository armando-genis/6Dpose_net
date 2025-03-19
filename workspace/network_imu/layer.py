import torchvision
import torch
import torch.nn as nn

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
    """
    Filter detections using the boxes and classification values.

    Args
        boxes: Tensor of shape (batch_size, num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification: Tensor of shape (batch_size, num_boxes, num_classes) containing the classification scores.
        rotation: Tensor of shape (batch_size, num_boxes, num_rotation_parameters) containing the rotations.
        translation: Tensor of shape (batch_size, num_boxes, 3) containing the translation vectors.
        num_rotation_parameters: Number of rotation parameters, usually 3 for axis angle representation
        num_translation_parameters: Number of translation parameters, usually 3 
        class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
        nms: Flag to enable/disable non maximum suppression.
        score_threshold: Threshold used to prefilter the boxes with.
        max_detections: Maximum number of detections to keep.
        nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.

    Returns
        A list of [boxes, scores, labels, rotation, translation] tensors after filtering.
        boxes is shaped (batch_size, max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (batch_size, max_detections) and contains the scores of the predicted class.
        labels is shaped (batch_size, max_detections) and contains the predicted label.
        rotation is shaped (batch_size, max_detections, num_rotation_parameters) and contains the rotations.
        translation is shaped (batch_size, max_detections, num_translation_parameters) and contains the translations.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """
    batch_size = boxes.shape[0]
    
    # Initialize lists to store the filtered detections for each batch
    all_filtered_boxes = []
    all_filtered_scores = []
    all_filtered_labels = []
    all_filtered_rotation = []
    all_filtered_translation = []
    
    # Process each image in the batch
    for batch_idx in range(batch_size):
        batch_boxes = boxes[batch_idx]
        batch_classification = classification[batch_idx]
        batch_rotation = rotation[batch_idx]
        batch_translation = translation[batch_idx]
        
        if class_specific_filter:
            all_indices = []
            # For each class, perform filtering based on score threshold
            for c in range(batch_classification.shape[1]):
                scores = batch_classification[:, c]
                score_mask = scores > score_threshold
                if not torch.any(score_mask):
                    continue
                
                # Get indices of boxes that pass score threshold
                indices = torch.nonzero(score_mask).squeeze(1)
                
                if nms:
                    # Get the filtered boxes and scores
                    filtered_boxes = batch_boxes[indices]
                    filtered_scores = scores[indices]
                    
                    # Apply NMS
                    keep_indices = torchvision.ops.nms(
                        filtered_boxes,
                        filtered_scores,
                        nms_threshold
                    )
                    
                    # Update indices to only include boxes that passed NMS
                    indices = indices[keep_indices]
                
                # Add class-specific information
                class_indices = torch.full((indices.size(0),), c, dtype=torch.int64, device=boxes.device)
                all_indices.append(torch.stack([indices, class_indices], dim=1))
            
            # Combine indices from all classes
            if all_indices:
                indices_with_class = torch.cat(all_indices, dim=0)
            else:
                # No detections, create empty tensors
                indices_with_class = torch.zeros((0, 2), dtype=torch.int64, device=boxes.device)
        else:
            # Get max scores and corresponding class indices
            scores, labels = torch.max(batch_classification, dim=1)
            
            # Filter by score threshold
            score_mask = scores > score_threshold
            if not torch.any(score_mask):
                indices_with_class = torch.zeros((0, 2), dtype=torch.int64, device=boxes.device)
            else:
                indices = torch.nonzero(score_mask).squeeze(1)
                labels = labels[indices]
                
                if nms:
                    # Get the filtered boxes and scores
                    filtered_boxes = batch_boxes[indices]
                    filtered_scores = scores[indices]
                    
                    # Apply NMS
                    keep_indices = torchvision.ops.nms(
                        filtered_boxes,
                        filtered_scores,
                        nms_threshold
                    )
                    
                    # Update indices and labels
                    indices = indices[keep_indices]
                    labels = labels[keep_indices]
                
                indices_with_class = torch.stack([indices, labels], dim=1)
        
        # Get the final scores
        if indices_with_class.shape[0] > 0:
            final_indices = indices_with_class[:, 0]
            final_labels = indices_with_class[:, 1]
            final_scores = torch.zeros(final_indices.shape[0], device=boxes.device)
            
            # Get the scores for each detected box (based on its class)
            for i in range(final_indices.shape[0]):
                final_scores[i] = batch_classification[final_indices[i], final_labels[i]]
            
            # Select top-k detections
            if final_scores.shape[0] > max_detections:
                topk_scores, topk_indices = torch.topk(final_scores, k=max_detections)
                final_scores = topk_scores
                final_indices = final_indices[topk_indices]
                final_labels = final_labels[topk_indices]
            
            # Gather the filtered detections
            filtered_boxes = batch_boxes[final_indices]
            filtered_rotation = batch_rotation[final_indices]
            filtered_translation = batch_translation[final_indices]
            
            # Pad if necessary
            pad_size = max_detections - final_scores.shape[0]
            if pad_size > 0:
                filtered_boxes = torch.cat([filtered_boxes, torch.full((pad_size, 4), -1, device=boxes.device)], dim=0)
                filtered_scores = torch.cat([final_scores, torch.full((pad_size,), -1, device=boxes.device)], dim=0)
                filtered_labels = torch.cat([final_labels, torch.full((pad_size,), -1, device=boxes.device, dtype=torch.int64)], dim=0)
                filtered_rotation = torch.cat([filtered_rotation, torch.full((pad_size, num_rotation_parameters), -1, device=boxes.device)], dim=0)
                filtered_translation = torch.cat([filtered_translation, torch.full((pad_size, num_translation_parameters), -1, device=boxes.device)], dim=0)
            else:
                filtered_scores = final_scores
                filtered_labels = final_labels
                
        else:
            # No detections, create empty padded tensors
            filtered_boxes = torch.full((max_detections, 4), -1, device=boxes.device)
            filtered_scores = torch.full((max_detections,), -1, device=boxes.device)
            filtered_labels = torch.full((max_detections,), -1, device=boxes.device, dtype=torch.int64)
            filtered_rotation = torch.full((max_detections, num_rotation_parameters), -1, device=boxes.device)
            filtered_translation = torch.full((max_detections, num_translation_parameters), -1, device=boxes.device)
        
        # Add to batch outputs
        all_filtered_boxes.append(filtered_boxes)
        all_filtered_scores.append(filtered_scores)
        all_filtered_labels.append(filtered_labels)
        all_filtered_rotation.append(filtered_rotation)
        all_filtered_translation.append(filtered_translation)
    
    # Stack batch outputs
    all_filtered_boxes = torch.stack(all_filtered_boxes, dim=0)
    all_filtered_scores = torch.stack(all_filtered_scores, dim=0)
    all_filtered_labels = torch.stack(all_filtered_labels, dim=0)
    all_filtered_rotation = torch.stack(all_filtered_rotation, dim=0)
    all_filtered_translation = torch.stack(all_filtered_translation, dim=0)
    
    return [all_filtered_boxes, all_filtered_scores, all_filtered_labels, all_filtered_rotation, all_filtered_translation]


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
            **kwargs
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
        """
        super(FilterDetections, self).__init__()
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.num_rotation_parameters = num_rotation_parameters
        self.num_translation_parameters = num_translation_parameters
    
    def forward(self, inputs):
        """
        Applies filtering to the detections.
        
        Args:
            inputs: List of [boxes, classification, rotation, translation] tensors.
                
        Returns:
            A list of [filtered_boxes, filtered_scores, filtered_labels, filtered_rotation, filtered_translation] tensors.
        """
        boxes, classification, rotation, translation = inputs
        
        # Apply filtering
        return filter_detections(
            boxes,
            classification,
            rotation,
            translation,
            num_rotation_parameters=self.num_rotation_parameters,
            num_translation_parameters=self.num_translation_parameters,
            class_specific_filter=self.class_specific_filter,
            nms=self.nms,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            nms_threshold=self.nms_threshold
        )