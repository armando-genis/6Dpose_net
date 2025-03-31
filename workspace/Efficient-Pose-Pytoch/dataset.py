import os
import cv2
from plyfile import PlyData
import math

from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils.anchors import anchors_for_shape, anchor_targets_bbox, AnchorParameters

class LineModDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 train, 
                 image_sizes = (512, 640, 768, 896, 1024, 1280, 1408), 
                 phi = 0, 
                 object_id=0):

        self.dataset_dir = dataset_dir
        self.is_symmetric = False
        self.rotation_parameter = 3
        self.translation_parameter = 3
        self.K = np.reshape([462.1379699707031, 0.0, 320.0, 0.0, 462.1379699707031, 240.0, 0.0, 0.0, 1.0],(3,3))
        self.translation_scale_norm = 1.0
        self.image_size = image_sizes[phi]
        self.object_id = object_id

        self.anchor_parameters = AnchorParameters.default
        self.anchors, self.translation_anchors = anchors_for_shape((self.image_size, self.image_size), anchor_params = self.anchor_parameters)
        self.num_anchors = self.anchor_parameters.num_anchors()

        #set the class and name dict for mapping each other
        self.class_to_name = {0: "object"}
        self.name_to_class = {"object": 0}
        self.name_to_mask_value = {"object": 255}
        self.object_ids_to_class_labels = {self.object_id: 0}
        self.class_labels_to_object_ids = {0: self.object_id}

        #get all train or test data examples from the dataset in the given split
        if train:
            data_file = os.path.join(self.dataset_dir, "train.txt")
        else:
            data_file = os.path.join(self.dataset_dir, "test.txt")

        with open(data_file) as fid:
            self.data_examples = [example.strip() for example in fid if example != ""]

        #parse files with infos about 3D BBox
        self.model_dict = dict()
        self.model_dict['diameter'] = 2.155
        self.model_dict['min_x'] = -1.175/2
        self.model_dict['min_y'] = -1.797/2
        self.model_dict['min_z'] = 0
        self.model_dict['size_x'] = 1.175
        self.model_dict['size_y'] = 1.797
        self.model_dict['size_z'] = 0.185

        self.all_models_dict = dict()
        self.all_models_dict[0] = self.model_dict
        #load the complete 3d model from the ply file
        self.model_3d_points = self.load_model_ply(path_to_ply_file = os.path.join(self.dataset_dir, "model.ply"))
        self.class_to_model_3d_points = {0: self.model_3d_points}
        self.name_to_model_3d_points = {"object": self.model_3d_points}

        #create dict with the class indices/names as keys and 3d model diameters as values
        self.class_to_model_3d_diameters, self.name_to_model_3d_diameters = self.create_model_3d_diameters_dict(self.all_models_dict, self.object_ids_to_class_labels, self.class_to_name)
        
        #create dict with the class indices/names as keys and model 3d bboxes as values
        self.class_to_model_3d_bboxes, self.name_to_model_3d_bboxes = self.create_model_3d_bboxes_dict(self.all_models_dict, self.object_ids_to_class_labels, self.class_to_name)

    def __len__(self):
        return len(self.data_examples)

    def create_model_3d_bboxes_dict(self, all_models_dict, object_ids_to_class_labels, class_to_name):
        """
        Creates two dictionaries which are mapping the class indices, respectively the class names to the 3D model cuboids
        Args:
            all_models_dict: Dictionary containing all 3D model's bboxes in the Linemod dataset format (min_x, min_y, min_z, size_x, size_y, size_z)
            object_ids_to_class_labels: Dictionary mapping the Linemod object ids to the EfficientPose classes
            class_to_name: Dictionary mapping the EfficientPose classes to their names
        Returns:
            Two dictionaries containing the EfficientPose class indices or the class names as keys and the 3D model cuboids as values
    
        """
        class_to_model_3d_bboxes = dict()
        name_to_model_3d_bboxes = dict()
        
        for object_id, class_label in object_ids_to_class_labels.items():
            model_bbox = all_models_dict[object_id]
            class_to_model_3d_bboxes[class_label] = model_bbox
            name_to_model_3d_bboxes[class_to_name[class_label]] = model_bbox
            
        return class_to_model_3d_bboxes, name_to_model_3d_bboxes

    def create_model_3d_diameters_dict(self, all_models_dict, object_ids_to_class_labels, class_to_name):
        """
       Creates two dictionaries containing the class idx and the model name as key and the 3D model diameters as values
        Args:
            all_models_dict: Dictionary containing all 3D model's bboxes and diameters in the Linemod dataset format
            object_ids_to_class_labels: Dictionary mapping the Linemod object ids to the EfficientPose classes
            class_to_name: Dictionary mapping the EfficientPose classes to their names
        Returns:
            Dictionary containing the class indices or the class names as keys and the 3D model diameters as values
    
        """
        class_to_model_3d_diameters = dict()
        name_to_model_3d_diameters = dict()
        
        for object_id, class_label in object_ids_to_class_labels.items():
            class_to_model_3d_diameters[class_label] = all_models_dict[object_id]["diameter"]
            name_to_model_3d_diameters[class_to_name[class_label]] = all_models_dict[object_id]["diameter"]
            
        return class_to_model_3d_diameters, name_to_model_3d_diameters

    def load_model_ply(self, path_to_ply_file):
        """
       Loads a 3D model from a plyfile
        Args:
            path_to_ply_file: Path to the ply file containing the object's 3D model
        Returns:
            points_3d: numpy array with shape (num_3D_points, 3) containing the x-, y- and z-coordinates of all 3D model points
    
        """
        model_data = PlyData.read(path_to_ply_file)
                                  
        vertex = model_data['vertex']
        points_3d = np.stack([vertex[:]['x'], vertex[:]['y'], vertex[:]['z']], axis = -1)
        
        return points_3d

    def __getitem__(self, index):
        """
        Pyorch sequence method for generating the input and annotation batches for EfficientPose.
        Args:
            index: The index of the element in the sequence
        Returns:
            inputs: List with the input batches for EfficientPose [batch_images, batch_camera_parameters]
            targets: List with the target batches for EfficientPose
        """
        image = self.load_image(index)
        annotation = self.load_annotation(index)
        camera_matrix_group = self.K

        image, annotation = self.preprocess_image_entry(image, annotation, camera_matrix_group)
        # compute network inputs
        inputs = self.compute_input(image, annotation)

        # compute network targets
        targets = self.compute_target(image, annotation)

        return inputs, targets

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        #load image and switch BGR to RGB
        base = os.path.join(self.dataset_dir,'JPEGImages')
        image = cv2.imread(os.path.join(base,self.data_examples[image_index]+'.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_annotation(self, image_index):
        """
        Load annotations for an image_index.
        """
        base = os.path.join(self.dataset_dir,'Linemod_preprocessed')
        #reading all the data
        f = open(os.path.join(base,self.data_examples[image_index]+'.txt'),'r')
        message = f.read()
        message = message.split(' ')[1:19]
        for i in range(len(message)):
            message[i] = float (message[i])
        rotation_matrix = np.reshape(message[0:9],(3,3))

        #init annotations in the correct base format. set number of annotations to one because linemod dataset only contains one annotation per image
        num_all_rotation_parameters = self.rotation_parameter + 2 #+1 for class id and +1 for is_symmetric flag

        annotations = {'labels': np.zeros((1,)),
                       'bboxes': np.zeros((1, 4)),
                       'rotations': np.zeros((1, num_all_rotation_parameters)),
                       'translations': np.zeros((1, self.translation_parameter)),
                       'translations_x_y_2D': np.zeros((1, 2))}
        
        #fill in the values
        #class label is always zero because there is only one possible object
        #get bbox from mask
        mask_base = os.path.join(self.dataset_dir,'masks')
        mask = cv2.imread(os.path.join(mask_base,self.data_examples[image_index]+'.jpg'))
        
        annotations["bboxes"][0, :], _ = self.get_bbox_from_mask(mask)
        #transform rotation into the needed representation
        annotations["rotations"][0, :-2] = self.rotation_mat_to_axis_angle(rotation_matrix)
        annotations["rotations"][0, -2] = float(self.is_symmetric)
        annotations["rotations"][0, -1] = float(0) #useless for linemod because there is only one object but neccessary to keep compatibility of the architecture with multi-object datasets
        
        annotations["translations"][0, :] = np.array(message[9:12])
        annotations["translations_x_y_2D"][0, :] = self.project_points_3D_to_2D(points_3D = np.zeros(shape = (1, 3)), #transform the object origin point which is the centerpoint
                                                                                    rotation_vector = self.rotation_mat_to_axis_angle(rotation_matrix),
                                                                                    translation_vector = np.array(message[9:12]),
                                                                                    camera_matrix = self.K)

        return annotations

    def get_bbox_from_mask(self, mask, mask_value = None):
        """ Computes the 2D bounding box from the input mask
        Args:
            mask: The segmentation mask of the image
            mask_value: The integer value of the object in the segmentation mask
        Returns:
            numpy array with shape (4,) containing the 2D bounding box
            Boolean indicating if the object is found in the given mask or not
        """
        if mask_value is None:
            seg = np.where(mask != 0)
        else:
            seg = np.where(mask == mask_value)
        
        #check if mask is empty
        if seg[0].size <= 0 or seg[1].size <= 0:
            return np.zeros((4,), dtype = np.float32), False
        min_x = np.min(seg[1])
        min_y = np.min(seg[0])
        max_x = np.max(seg[1])
        max_y = np.max(seg[0])
        
        return np.array([min_x, min_y, max_x, max_y], dtype = np.float32), True

    def rotation_mat_to_axis_angle(self, rotation_matrix):
        """
        Computes an axis angle rotation vector from a rotation matrix 
        Arguments:
            rotation_matrix: numpy array with shape (3, 3) containing the rotation
        Returns:
            axis_angle: numpy array with shape (3,) containing the rotation
        """
        axis_angle, jacobian = cv2.Rodrigues(rotation_matrix)
        
        return np.squeeze(axis_angle)

    def project_points_3D_to_2D(self, points_3D, rotation_vector, translation_vector, camera_matrix):
        """
        Transforms and projects the input 3D points onto the 2D image plane using the given rotation, translation and camera matrix    
        Arguments:
            points_3D: numpy array with shape (num_points, 3) containing 3D points (x, y, z)
            rotation_vector: numpy array containing the rotation vector with shape (3,)
            translation_vector: numpy array containing the translation vector with shape (3,)
            camera_matrix: numpy array with shape (3, 3) containing the intrinsic camera parameters
        Returns:
            points_2D: numpy array with shape (num_points, 2) with the 2D projections of the given 3D points
        """
        points_2D, jacobian = cv2.projectPoints(points_3D, rotation_vector, translation_vector, camera_matrix, None)
        points_2D = np.squeeze(points_2D)
    
        return points_2D

    def preprocess_image_entry(self, image, annotations, camera_matrix):
        """
        Preprocess image and its annotations.
        Args:
            image: The image to preprocess
            annotations: The annotations to preprocess
            camera_matrix: The camera matrix of the example
        Returns:
            image: The preprocessed image
            annotations: The preprocessed annotations
        """
        
        # preprocess and resize the image
        image, image_scale = self.preprocess_image(image)

        # apply resizing to annotations too
        annotations['bboxes'] *= image_scale
        
        #normalize rotation from [-pi, +pi] to [-1, +1]
        annotations['rotations'][:, :self.rotation_parameter] /= math.pi
        
        #apply resizing to translation 2D centerpoint
        annotations["translations_x_y_2D"] *= image_scale
        #concat rotation and translation annotations to transformation targets because keras accepts only a single prediction tensor in a loss function, so this is a workaround to combine them both in the loss function
        annotations['transformation_targets'] = np.concatenate([annotations["rotations"][:, :self.rotation_parameter], annotations['translations'], annotations["rotations"][:, self.rotation_parameter:]], axis = -1)
        
        annotations['camera_parameters'] = self.get_camera_parameter_input(camera_matrix, image_scale, self.translation_scale_norm)

        return image, annotations

    def preprocess_image(self, image):
        """
        Preprocess image
        Args:
            image: The image to preprocess
        Returns:
            image: The preprocessed image
            scale: The factor with which the image was scaled to match the EfficientPose input resolution
        """
        # image, RGB
        image_height, image_width = image.shape[:2]
        if image_height > image_width:
            scale = self.image_size / image_height
            resized_height = self.image_size
            resized_width = int(image_width * scale)
        else:
            scale = self.image_size / image_width
            resized_height = int(image_height * scale)
            resized_width = self.image_size

        image = cv2.resize(image, (resized_width, resized_height))
        image = image.astype(np.float32)
        image /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image -= mean
        image /= std
        pad_h = (self.image_size - resized_height)
        pad_w = self.image_size - resized_width
        image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')
        
        return image, scale

    def get_camera_parameter_input(self, camera_matrix, image_scale, translation_scale_norm):
        """
        Returns the input vector containing the needed intrinsic camera parameters, image scale and the translation_scale_norm
        Args:
            camera_matrix: numpy array with shape (3, 3) containing the intrinsic camera parameters
            image_scale: The factor with which the image was scaled to match the EfficientPose input resolution
            translation_scale_norm: factor to change units. EfficientPose internally works with meter and if your dataset unit is mm for example, then you need to set this parameter to 1000
        Returns:
            input_vector: numpy array of shape (6,) containing [fx, fy, px, py, translation_scale_norm, image_scale]
        """
        #input_vector = [fx, fy, px, py, translation_scale_norm, image_scale]
        input_vector = np.zeros((6,), dtype = np.float32)
        
        input_vector[0] = camera_matrix[0, 0]
        input_vector[1] = camera_matrix[1, 1]
        input_vector[2] = camera_matrix[0, 2]
        input_vector[3] = camera_matrix[1, 2]
        input_vector[4] = translation_scale_norm
        input_vector[5] = image_scale
        
        return input_vector

    def compute_input(self, image, annotation):
        """
        Compute input for the network using an image and the camera parameters from the annotations.

        Returns:
            List with the input for EfficientPose [image, camera_parameters]
        """
        image = np.array(image).astype(np.float32).transpose([2,0,1])
        #model needs also the camera parameters to compute the final translation vector
        camera_parameters = np.array(annotation['camera_parameters']).astype(np.float32)
        
        return [image, camera_parameters]

    def compute_target(self, image, annotation):
        """
        Compute target outputs for the network using images and their annotations.
        """

        targets = anchor_targets_bbox(
            self.anchors,
            image,
            annotation,
            num_classes=1,
            num_rotation_parameters = self.rotation_parameter + 2, #+1 for the is_symmetric flag and +1 for the class idx to choose the correct model 3d points
            num_translation_parameters = self.translation_parameter, #x,y in 2D and Tz
            translation_anchors = self.translation_anchors,
        )
        return targets

def get_model_3d_points_for_loss(all_model_points, points_for_shape_match_loss, flatten = True):
        """
        Creates and returns the numpy array with shape (points_for_shape_match_loss, 3) containing the 3D model points of a single object in the dataset.
        Subsamples 3D points if there are more than needed or use zero padding if there are less than needed.
        Args:
            class_to_model_3d_points: Dictionary mapping the object class to the object's 3D model points
            class_idx: The class index of the object
            points_for_shape_match_loss: The number of 3D points to use for each object
            flatten: Boolean indicating wheter to reshape the output array to a single dimension
        Returns:
            numpy array with shape (num_model_points, 3) or (num_model_points * 3,) containing the 3D model points (x, y, z) of an object
        """
        
        num_points = all_model_points.shape[0]
        
        if num_points == points_for_shape_match_loss:
            #just return the flattened array
            if flatten:
                return np.reshape(all_model_points, (-1,))
            else:
                return all_model_points
        elif num_points < points_for_shape_match_loss:
            #use zero padding
            points = np.zeros((points_for_shape_match_loss, 3))
            points[:num_points, :] = all_model_points
            if flatten:
                return np.reshape(points, (-1,))
            else:
                return points
        else:
            #return a subsample from all 3d points
            step_size = (num_points // points_for_shape_match_loss) - 1
            if step_size < 1:
                step_size = 1
            points = all_model_points[::step_size, :]
            if flatten:
                return np.reshape(points[:points_for_shape_match_loss, :], (-1, ))
            else:
                return points[:points_for_shape_match_loss, :]

def create_loaders (dataset_dir,
                    batch_size,
                    phi = 0,
                    object_id=0,
                    num_workers=4,
                    pin_memory=True):
    
    train_ds = LineModDataset(dataset_dir, True, phi=phi, object_id=object_id)

    test_ds = LineModDataset(dataset_dir, False, phi=phi, object_id=object_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    
    return train_loader, test_loader, np.expand_dims(get_model_3d_points_for_loss(train_ds.model_3d_points, 500, False), axis = 0)

if __name__ == '__main__':
    test = LineModDataset('/home/meyako/datasets/dataset_light',False)
    input ,output  =test.__getitem__(0)
    for i in output:
        print (i.shape)

