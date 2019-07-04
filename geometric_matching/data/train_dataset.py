# ==============================================================================================================
# Generate a synthetically training dataset from PF-PASCAL for geometric matching model
# Author: Jingwei Qu
# Date: 18 May 2019
# ==============================================================================================================

import torch
import os
from os.path import exists, join, basename
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geometric_matching.geotnf.transformation import GeometricTnf
from geometric_matching.util.net_util import roi_data

import matplotlib
import matplotlib.pyplot as plt

class TrainDataset(Dataset):
    """
    Synthetically training dataset with unsupervised training.

    Args:
            csv_file (string): Path to the csv file with image names and transformations
            dataset_path (string): Directory with all the images
            output_size (2-tuple): Desired output size
            transform (callable): Transformation for post-processing the training pair (eg. image normalization)

    Returns:
            Dict: {'source_image' & 'target_image': images for transformation,
                   'source_im_size' & 'target_im_size': size of images
                   'theta': transformation from source image to refer image (warped source image),
                   'source_im' & 'target_im': images for object detection,
                   'source_im_info' & 'target_im_info': image size & image scale ratio,
                   'source_gt_boxes' & 'target_gt_boxes': coordinates of ground-truth bounding boxes, set to [1, 1, 1, 1, 1]
                   'source_num_boxes' & 'target_num_boxes': number of ground-truth bounding boxes, set to 0}
    """

    def __init__(self, csv_file, dataset_path, output_size=(240, 240), geometric_model='affine', dataset_size=0,
                 transform=None, random_sample=False, random_t=0.5, random_s=0.5, random_alpha=1 / 6, random_t_tps=0.4):
        self.dataframe = pd.read_csv(csv_file)  # Read images data
        if dataset_size != 0:
            dataset_size = min((dataset_size, len(self.dataframe)))
            self.dataframe = self.dataframe.iloc[0:dataset_size, :]
        self.img_A_names = self.dataframe.iloc[:, 0]  # Get source image & target image name
        self.img_B_names = self.dataframe.iloc[:, 1]
        self.categories = self.dataframe.iloc[:, 2].values
        self.flips = self.dataframe.iloc[:, 3].values.astype('int')
        self.random_sample = random_sample
        if not self.random_sample:
            self.theta_array = self.dataframe.iloc[:, 4:].values.astype('float')  # Get ground-truth tps parameters
        self.dataset_path = dataset_path  # Path for reading images
        self.out_h, self.out_w = output_size
        self.geometric_model = geometric_model
        self.transform = transform
        self.random_t = random_t
        self.random_s = random_s
        self.random_alpha = random_alpha
        self.random_t_tps = random_t_tps
        # self.extension = '.jpg'
        # Initialize an affine transformation to resize the image to (240, 240)
        self.affineTnf = GeometricTnf(geometric_model='affine', out_h=self.out_h, out_w=self.out_w, use_cuda=False)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Read image, and get image information for fasterRCNN
        image_A, im_size_A, im_A, im_info_A, gt_boxes_A, num_boxes_A = self.get_image(img_name_list=self.img_A_names, idx=idx)
        image_B, im_size_B, im_B, im_info_B, gt_boxes_B, num_boxes_B = self.get_image(img_name_list=self.img_B_names, idx=idx)

        # Read theta
        if not self.random_sample:
            theta = self.theta_array[idx, :]

            # if self.geometric_model == 'affine':
                # reshape theta to 2x3 matrix [A|t] where
                # first row corresponds to X and second to Y
                # theta = theta[[3, 2, 5, 1, 0, 4]]
            if self.geometric_model == 'tps':
                theta = np.expand_dims(np.expand_dims(theta, 1), 2)
            if self.geometric_model == 'afftps':
                theta[[0, 1, 2, 3, 4, 5]] = theta[[3, 2, 5, 1, 0, 4]]
        else:
            if self.geometric_model == 'affine' or self.geometric_model == 'afftps':
                alpha = (np.random.rand(1) - 0.5) * 2 * np.pi * self.random_alpha
                theta_aff = np.random.rand(6)
                theta_aff[[2, 5]] = (theta_aff[[2, 5]] - 0.5) * 2 * self.random_t
                theta_aff[0] = (1 + (theta_aff[0] - 0.5) * 2 * self.random_s) * np.cos(alpha)
                theta_aff[1] = (1 + (theta_aff[1] - 0.5) * 2 * self.random_s) * (-np.sin(alpha))
                theta_aff[3] = (1 + (theta_aff[3] - 0.5) * 2 * self.random_s) * np.sin(alpha)
                theta_aff[4] = (1 + (theta_aff[4] - 0.5) * 2 * self.random_s) * np.cos(alpha)

            if self.geometric_model == 'tps' or self.geometric_model == 'afftps':
                theta_tps = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1])
                theta_tps = theta_tps + (np.random.rand(18) - 0.5) * 2 * self.random_t_tps

            if self.geometric_model == 'affine':
                theta = theta_aff
            elif self.geometric_model == 'tps':
                theta = theta_tps
            elif self.geometric_model == 'afftps':
                theta = np.concatenate((theta_aff, theta_tps))

        theta = torch.Tensor(theta.astype(np.float32))

        sample = {'source_image': image_A, 'target_image': image_B,
                  'source_im_size': im_size_A, 'target_im_size': im_size_B,
                  'theta_GT': theta,
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

        # h, w, c = image.shape
        # top = np.random.randint(h / 4)
        # bottom = int(3 * h / 4 + np.random.randint(h / 4))
        # left = np.random.randint(w / 4)
        # right = int(3 * w / 4 + np.random.randint(w / 4))
        # image = image[top:bottom, left:right, :]

        # Flip horizontally
        if self.flips[idx]:
            image = np.flip(image, axis=1)
            # file_name = basename(img_name)
            # io.imsave('/home/qujingwei/geometric-matching/geometric_matching/'+file_name, image)
            # print(img_name)

        im_size = np.asarray(image.shape)

        # Get tensors of image, image_info (H, W, im_scale), ground-truth boxes, number of boxes for faster rcnn
        im, im_info, gt_boxes, num_boxes = roi_data(image)

        # Transform numpy to tensor, permute order of image to CHW
        image = torch.Tensor(image.astype(np.float32))
        image = image.permute(2, 0, 1)
        im_size = torch.Tensor(im_size)

        # Resize image using bilinear sampling with identity affine tnf
        image.requires_grad = False
        image = self.affineTnf(image_batch=image.unsqueeze(0)).squeeze(0)

        return image, im_size, im, im_info, gt_boxes, num_boxes