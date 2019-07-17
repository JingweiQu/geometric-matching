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

class PFPASCALDataset(Dataset):
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

    def __init__(self, csv_file, dataset_path, output_size=(240, 240), normalize=None, category=None,
                 pck_procedure='scnet'):
        self.dataframe = pd.read_csv(csv_file)  # Read images data
        self.categories = self.dataframe.iloc[:, 2].values.astype('float')  # Get image category
        # Select image data according to given category
        if category is not None:
            cat_idx = np.nonzero(self.categories == category)[0]
            self.categories = self.categories[cat_idx]
            self.dataframe = self.dataframe.iloc[cat_idx, :]
        self.img_A_names = self.dataframe.iloc[:, 0]  # Get source image & target image name
        self.img_B_names = self.dataframe.iloc[:, 1]
        self.point_A_coords = self.dataframe.iloc[:, 3:5]   # Get key points in source image and target image
        self.point_B_coords = self.dataframe.iloc[:, 5:]
        self.dataset_path = dataset_path  # Path for reading images
        self.out_h, self.out_w = output_size
        self.normalize = normalize
        self.pck_procedure = pck_procedure
        # self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
        self.category_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                               'train', 'tvmonitor']
        # Initialize an affine transformation to resize the image to (240, 240)
        self.affineTnf = GeometricTnf(geometric_model='affine', out_h=self.out_h, out_w=self.out_w, use_cuda=False)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # get pre-processed images
        if self.normalize is not None:
            image_A, im_A, im_info_A, gt_boxes_A, num_boxes_A = self.get_image(img_name_list=self.img_A_names, idx=idx)
            image_B, im_B, im_info_B, gt_boxes_B, num_boxes_B = self.get_image(img_name_list=self.img_B_names, idx=idx)
        else:
            image_A, im_info_A, gt_boxes_A, num_boxes_A = self.get_image(img_name_list=self.img_A_names, idx=idx)
            image_B, im_info_B, gt_boxes_B, num_boxes_B = self.get_image(img_name_list=self.img_B_names, idx=idx)

        # get pre-processed point coords
        # point_A_coords.shape & point_B_coords.shape: (2, 20), including coordinates of key points, others are -1
        point_A_coords = self.get_points(point_coords_list=self.point_A_coords, idx=idx)
        point_B_coords = self.get_points(point_coords_list=self.point_B_coords, idx=idx)

        # Number of key points in image_A
        N_pts = torch.sum(torch.ne(point_A_coords[0, :], -1))

        # compute PCK reference length L_pck (equal to max bounding box side in image_A)
        if self.pck_procedure == 'pf':
            L_pck = torch.Tensor([torch.max(point_A_coords[:, :N_pts].max(1)[0] - point_A_coords[:, :N_pts].min(1)[0])])
        elif self.pck_procedure == 'scnet':
            # modification to follow the evaluation procedure of SCNet
            point_A_coords[0, 0:N_pts] = point_A_coords[0, 0:N_pts] * 224 / im_info_A[1]
            point_A_coords[1, 0:N_pts] = point_A_coords[1, 0:N_pts] * 224 / im_info_A[0]

            point_B_coords[0, 0:N_pts] = point_B_coords[0, 0:N_pts] * 224 / im_info_B[1]
            point_B_coords[1, 0:N_pts] = point_B_coords[1, 0:N_pts] * 224 / im_info_B[0]

            im_info_A[0:2] = torch.Tensor([224, 224])
            im_info_B[0:2] = torch.Tensor([224, 224])

            L_pck = torch.Tensor([224.0])

        sample = {'source_image': image_A, 'target_image': image_B,
                  'source_im_info': im_info_A, 'target_im_info': im_info_B,
                  'source_gt_boxes': gt_boxes_A, 'target_gt_boxes': gt_boxes_B,
                  'source_num_boxes': num_boxes_A, 'target_num_boxes': num_boxes_B,
                  'source_points': point_A_coords, 'target_points': point_B_coords,
                  'L_pck': L_pck}

        if self.normalize is not None:
            sample = {'source_image': image_A, 'target_image': image_B,
                      'source_im': im_A, 'target_im': im_B,
                      'source_im_info': im_info_A, 'target_im_info': im_info_B,
                      'source_gt_boxes': gt_boxes_A, 'target_gt_boxes': gt_boxes_B,
                      'source_num_boxes': num_boxes_A, 'target_num_boxes': num_boxes_B,
                      'source_points': point_A_coords, 'target_points': point_B_coords,
                      'L_pck': L_pck}
            sample = self.normalize(sample)

        return sample

    def get_image(self, img_name_list, idx):
        img_name = os.path.join(self.dataset_path, img_name_list[idx])
        # image = io.imread(img_name)
        image = cv2.imread(img_name)  # cv2: channel is BGR
        # If the image just has two channels, add one channel
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.concatenate((image, image, image), axis=2)

        # Get image size, (H, W, C)
        im_size = np.asarray(image.shape)
        im_size = torch.Tensor(im_size.astype(np.float32))
        im_size.requires_grad = False

        # Get tensors of image, image_info (H, W, im_scale), ground-truth boxes, number of boxes for faster rcnn
        im, im_info, gt_boxes, num_boxes = roi_data(image, self.out_h)
        im_info = torch.cat((im_size, im_info), 0)

        if self.normalize is not None:
            # Transform numpy to tensor, permute order of image to CHW
            image = torch.Tensor(image.astype(np.float32))
            image = image.permute(2, 0, 1)  # For following normalization
            # Resize image using bilinear sampling with identity affine tnf
            image.requires_grad = False
            image = self.affineTnf(image_batch=image.unsqueeze(0)).squeeze(0)
            return image, im, im_info, gt_boxes, num_boxes

        return im, im_info, gt_boxes, num_boxes

    def get_points(self, point_coords_list, idx):
        X = np.fromstring(point_coords_list.iloc[idx, 0], sep=';')
        Y = np.fromstring(point_coords_list.iloc[idx, 1], sep=';')
        Xpad = -np.ones(20)
        Xpad[:len(X)] = X
        Ypad = -np.ones(20)
        Ypad[:len(X)] = Y
        point_coords = np.concatenate((Xpad.reshape(1, 20), Ypad.reshape(1, 20)), axis=0)

        # make arrays float tensor for subsequent processing
        # point_coords.shape: (2, 20), including coordinates of key points, others are -1
        point_coords = torch.Tensor(point_coords)
        return point_coords