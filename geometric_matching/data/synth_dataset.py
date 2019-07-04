# =================================================================================================================
# Generate a synthetically training dataset from PascalVOC2011 for geometric matching model
# Author: Ignacio Rocco
# Modification: Jingwei Qu
# Date: 19 April 2019
# =================================================================================================================

from __future__ import print_function, division
import torch
import os
from os.path import exists, join, basename
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
# from geotnf.transformation import GeometricTnf
from geometric_matching.geotnf.transformation import GeometricTnf
from geometric_matching.util.net_util import roi_data

class SynthDataset(Dataset):
    """

    Synthetically training dataset with strong supervision

    Args:
            csv_file (string): Path to the csv file with image names and transformations
            dataset_path (string): Directory with all the images
            output_size (2-tuple): Desired output size
            transform (callable): Transformation for post-processing the training pair (eg. image normalization)

    Returns:
            Dict: {'image': full dataset image, 'theta': desired transformation,
            'im', 'im_info', 'gt_boxes', 'num_boxes': image information for fasterRCNN}

    """

    def __init__(self, csv_file, dataset_path, output_size=(480, 640), geometric_model='tps', dataset_size=0,
                 transform=None, random_sample=False, random_t=0.5, random_s=0.5, random_alpha=1/6, random_t_tps=0.4):
        self.dataset_path = dataset_path  # Path for reading images
        self.out_h, self.out_w = output_size
        self.geometric_model = geometric_model
        self.transform = transform
        self.random_sample = random_sample
        self.random_t = random_t
        self.random_s = random_s
        self.random_alpha = random_alpha
        self.random_t_tps = random_t_tps
        self.dataframe = pd.read_csv(csv_file)  # Read images data
        if dataset_size != 0:
            dataset_size = min((dataset_size, len(self.dataframe)))
            self.dataframe = self.dataframe.iloc[0:dataset_size, :]
        self.img_names = self.dataframe.iloc[:, 0]  # Get image name
        if not self.random_sample:
            self.theta_array = self.dataframe.iloc[:, 1:].values.astype('float') # Get ground-truth tps parameters
        # Initialize an affine transformation to resize the image to (480, 640)
        self.affineTnf = GeometricTnf(geometric_model='affine', out_h=self.out_h, out_w=self.out_w, use_cuda=False)
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # read image
        img_name = os.path.join(self.dataset_path, self.img_names[idx])
        image = io.imread(img_name)
        # If the image just has two channels, add one channel
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.concatenate((image, image, image), axis=2)

        im, im_info, gt_boxes, num_boxes = roi_data(image)
        
        # read theta
        if not self.random_sample:
            theta = self.theta_array[idx, :]

            if self.geometric_model == 'affine':
                # reshape theta to 2x3 matrix [A|t] where 
                # first row corresponds to X and second to Y
                theta = theta[[3, 2, 5, 1, 0, 4]]
            if self.geometric_model == 'tps':
                theta = np.expand_dims(np.expand_dims(theta, 1), 2)
            if self.geometric_model == 'afftps':
                theta[[0, 1, 2, 3, 4, 5]] = theta[[3, 2, 5, 1, 0, 4]]
        else:
            if self.geometric_model == 'affine' or self.geometric_model == 'afftps':
                alpha = (np.random.rand(1) - 0.5) * 2 * np.pi * self.random_alpha
                theta_aff = np.random.rand(6)
                theta_aff[[2, 5]] = (theta_aff[[2, 5]] - 0.5) * 2 *self.random_t
                theta_aff[0] = (1 + (theta_aff[0] - 0.5) * 2 * self.random_s) * np.cos(alpha)
                theta_aff[1] = (1 + (theta_aff[1] - 0.5) * 2 * self.random_s) * (-np.sin(alpha))
                theta_aff[3] = (1 + (theta_aff[3] - 0.5) * 2 * self.random_s) * np.sin(alpha)
                theta_aff[4] = (1 + (theta_aff[4] - 0.5) * 2 * self.random_s) * np.cos(alpha)
            if self.geometric_model == 'tps' or self.geometric_model == 'afftps':
                theta_tps = np.array([-1 , -1 , -1 , 0 , 0 , 0 , 1 , 1 , 1 , -1 , 0 , 1 , -1 , 0 , 1 , -1 , 0 , 1])
                theta_tps = theta_tps + (np.random.rand(18) - 0.5) * 2 * self.random_t_tps

            if self.geometric_model == 'affine':
                theta = theta_aff
            elif self.geometric_model == 'tps':
                theta = theta_tps
            elif self.geometric_model == 'afftps':
                theta = np.concatenate((theta_aff, theta_tps))
            
        # Transform numpy to tensor, permute order of image to CHW
        # image = torch.Tensor(image.astype(np.float32))
        # theta = torch.Tensor(theta.astype(np.float32))
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)
        theta = torch.Tensor(theta)

        # image = image.transpose(1,2).transpose(0,1)

                
        # Resize image using bilinear sampling with identity affine tnf
        # if image.shape[1] != self.out_h or image.shape[2] != self.out_w:
            # image = self.affineTnf(Variable(image.unsqueeze(0),requires_grad=False)).data.squeeze(0)
            # image = self.affineTnf(image.unsqueeze(0)).data.squeeze(0)
        image.requires_grad = False
        image = self.affineTnf(image.unsqueeze(0)).squeeze(0)

        sample = {'image': image, 'theta': theta, 'im': im, 'im_info': im_info, 'gt_boxes': gt_boxes, 'num_boxes': num_boxes}
        
        if self.transform:
            sample = self.transform(sample)

        return sample