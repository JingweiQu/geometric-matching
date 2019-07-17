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

class WatchDataset(Dataset):

    def __init__(self, csv_file, dataset_path, output_size=(240, 240), normalize=None):
        self.dataframe = pd.read_csv(csv_file)  # Read images data
        self.img_A_names = self.dataframe.iloc[:, 0]  # Get source image & target image name
        self.img_B_names = self.dataframe.iloc[:, 1]
        self.flip_img_A = self.dataframe.iloc[:, 2].values.astype('int')
        self.dataset_path = dataset_path  # Path for reading images
        self.out_h, self.out_w = output_size
        self.normalize = normalize
        self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
        # Initialize an affine transformation to resize the image to (240, 240)
        self.affineTnf = GeometricTnf(geometric_model='affine', out_h=self.out_h, out_w=self.out_w, use_cuda=False)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # get pre-processed images
        flip_img_A = self.flip_img_A[idx]
        image_A = self.get_image(img_name_list=self.img_A_names, idx=idx, flip=flip_img_A)
        image_B = self.get_image(img_name_list=self.img_B_names, idx=idx)

        sample = {'source_image': image_A, 'target_image': image_B}

        if self.normalize is not None:
            sample = self.normalize(sample)

        return sample

    def get_image(self, img_name_list, idx, flip=False):
        img_name = os.path.join(self.dataset_path, img_name_list[idx])
        # image = io.imread(img_name)
        image = cv2.imread(img_name)
        # If the image just has two channels, add one channel
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.concatenate((image, image, image), axis=2)

        # Flip horizontally
        if flip:
            image = image[:, ::-1, :]

        # get image size
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