# ==============================================================================================================
# Generate a testing dataset from PF-PASCAL for geometric matching model
# Author: Ignacio Rocco
# Modification: Jingwei Qu
# Date: 18 May 2019
# ==============================================================================================================

import torch
import os
from os.path import exists, join, basename
from skimage import io
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geometric_matching.geotnf.transformation import GeometricTnf
from geometric_matching.util.net_util import roi_data

class TestDataset(Dataset):
    """
    Proposal Flow PASCAL image pair dataset

    Args:
            csv_file (string): Path to the csv file with image names and transformations
            dataset_path (string): Directory with all the images
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

    def __init__(self, csv_file, dataset_path, output_size=(240, 240), normalize=None):
        self.dataframe = pd.read_csv(csv_file)  # Read images data
        self.dataset_path = dataset_path  # Path for reading images
        self.out_h, self.out_w = output_size
        self.normalize = normalize
        # Initialize an affine transformation to resize the image to (240, 240)
        self.affineTnf = GeometricTnf(geometric_model='affine', out_h=self.out_h, out_w=self.out_w, use_cuda=False)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        raise NotImplementedError

    def get_image(self, img_name_list, idx, flip=False):
        img_name = os.path.join(self.dataset_path, img_name_list[idx])
        # image = io.imread(img_name)
        image = cv2.imread(img_name)  # cv2: channel is BGR
        # If the image just has two channels, add one channel
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.concatenate((image, image, image), axis=2)

        # Flip horizontally
        if flip:
            image = image[:, ::-1, :]

        # Get image size, (H, W, C)
        im_size = np.asarray(image.shape)
        im_size = torch.Tensor(im_size.astype(np.float32))
        im_size.requires_grad = False

        im_info = im_size

        # Get tensors of image, image_info (H, W, im_scale), ground-truth boxes, number of boxes for faster rcnn
        # im, im_info, gt_boxes, num_boxes = roi_data(image, self.out_h)
        # im_info = torch.cat((im_size, im_info), 0)

        # Transform numpy to tensor, permute order of image to CHW
        image = image[:, :, ::-1]   # BGR -> RGB, due to cv2
        image = torch.Tensor(image.astype(np.float32))
        image = image.permute(2, 0, 1)  # For following normalization
        # Resize image using bilinear sampling with identity affine tnf
        image.requires_grad = False
        image = self.affineTnf(image_batch=image.unsqueeze(0)).squeeze(0)
        
        # return image, im, im_info, gt_boxes, num_boxes
        return image, im_info