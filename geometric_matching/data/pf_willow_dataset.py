# ==============================================================================================================
# Generate a testing dataset from PF-WILLOW for geometric matching model
# Author: Ignacio Rocco
# Modification: Jingwei Qu
# Date: 18 May 2019
# ==============================================================================================================

import os
import torch
from torch.autograd import Variable
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geometric_matching.geotnf.transformation import GeometricTnf
from geometric_matching.util.net_util import roi_data
from scipy.misc import imread
from model.utils.blob import prep_im_for_blob

class PFWILLOWDataset(Dataset):
    """
    Proposal Flow WILLOW image pair dataset

    Args:
        csv_file (string): Path to the csv file with image names and transformations
        dataset_path (string): Directory with the images
        output_size (2-tuple): Desired output size
        transform (callable): Transformation for post-processing the training pair (eg. image normalization)

    Returns:
        Dict: {'source_image' & 'target_image': images for transformation,
               'source_im_size' & 'target_im_size': size of images
               'source_points' & 'target_points': coordinates of key points in images,
               'L_pck': PCK reference length
               'source_im' & 'target_im': images for object detection,
               'source_im_info' & 'target_im_info': image size & image scale ratio,
               'source_gt_boxes' & 'target_gt_boxes': coordinates of ground-truth bounding boxes, set to [1, 1, 1, 1, 1]
               'source_num_boxes' & 'target_num_boxes': number of ground-truth bounding boxes, set to 0}
    """

    def __init__(self, csv_file, dataset_path, output_size=(240, 240), transform=None):
        self.dataframe = pd.read_csv(csv_file)  # Read images data
        self.img_A_names = self.dataframe.iloc[:, 0]    # Get source image & target image name
        self.img_B_names = self.dataframe.iloc[:, 1]
        self.point_A_coords = self.dataframe.iloc[:, 2:22].values.astype('float')   # Get key points in source image and target image
        self.point_B_coords = self.dataframe.iloc[:, 22:].values.astype('float')
        self.dataset_path = dataset_path    # Path for reading images
        self.out_h, self.out_w = output_size
        self.transform = transform
        # Initialize an affine transformation to resize the image to (240, 240)
        self.affineTnf = GeometricTnf(geometric_model='affine', out_h=self.out_h, out_w=self.out_w, use_cuda=False)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # get pre-processed images
        image_A, im_size_A, im_A, im_info_A, gt_boxes_A, num_boxes_A = self.get_image(self.img_A_names, idx)
        image_B, im_size_B, im_B, im_info_B, gt_boxes_B, num_boxes_B = self.get_image(self.img_B_names, idx)

        # get pre-processed point coords
        point_A_coords = self.get_points(self.point_A_coords, idx)
        point_B_coords = self.get_points(self.point_B_coords, idx)

        # compute PCK reference length L_pck (equal to max bounding box side in image_A)
        L_pck = torch.Tensor([torch.max(point_A_coords.max(1)[0] - point_A_coords.min(1)[0])])

        sample = {'source_image': image_A, 'target_image': image_B,
                  'source_im_size': im_size_A, 'target_im_size': im_size_B,
                  'source_points': point_A_coords, 'target_points': point_B_coords,
                  'L_pck': L_pck,
                  'source_im': im_A, 'target_im': im_B,
                  'source_im_info': im_info_A, 'target_im_info': im_info_B,
                  'source_gt_boxes': gt_boxes_A, 'target_gt_boxes': gt_boxes_B,
                  'source_num_boxes': num_boxes_A, 'target_num_boxes': num_boxes_B}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self, img_name_list, idx):
        img_name = os.path.join(self.dataset_path, img_name_list[idx])
        image = io.imread(img_name)
        # If the image just has two channels, add one channel
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.concatenate((image, image, image), axis=2)

        # Get image size, (H, W, C)
        im_size = np.asarray(image.shape)

        # Get tensors of image, image_info (H, W, im_scale), ground-truth boxes, number of boxes for faster rcnn
        im, im_info, gt_boxes, num_boxes = roi_data(image)

        # Transform numpy to tensor, permute order of image to CHW
        image = torch.Tensor(image.astype(np.float32))
        image = image.permute(2, 0, 1)
        im_size = torch.Tensor(im_size)

        # Resize image using bilinear sampling with identity affine tnf
        image.requires_grad = False
        image = self.affineTnf(image.unsqueeze(0)).squeeze(0)

        return image, im_size, im, im_info, gt_boxes, num_boxes

    def get_points(self, point_coords_list, idx):
        point_coords = point_coords_list[idx, :].reshape(2, 10)

        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords)
        return point_coords
