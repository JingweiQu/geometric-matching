# ==============================================================================================================
# Generate a synthetically training dataset from PF-PASCAL for geometric matching model
# Author: Jingwei Qu
# Date: 18 May 2019
# ==============================================================================================================

import torch
import os
from os.path import exists, join, basename
from skimage import io
# from scipy.misc import imread
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geometric_matching.geotnf.transformation import GeometricTnf
from geometric_matching.util.net_util import roi_data

class TrainDataset(Dataset):
    """
    Synthetically training dataset with unsupervised training.

    Args:
            csv_file (string): Path to the csv file with image names and transformations
            dataset_path (string): Directory with all the images
            output_size (2-tuple): Desired output size
            normalize (callable): Normalization for post-processing the training pair

    Returns:
            Dict: {'source_image' & 'target_image': images for transformation,
                   'source_im_size' & 'target_im_size': size of images
                   'theta': transformation from source image to refer image (warped source image),
                   'source_im' & 'target_im': images for object detection,
                   'source_im_info' & 'target_im_info': image size & image scale ratio,
                   'source_gt_boxes' & 'target_gt_boxes': coordinates of ground-truth bounding boxes, set to [1, 1, 1, 1, 1]
                   'source_num_boxes' & 'target_num_boxes': number of ground-truth bounding boxes, set to 0}
    """

    def __init__(self, csv_file, dataset_path, output_size=(240, 240), normalize=None, random_crop=False):
        self.dataframe = pd.read_csv(csv_file)  # Read images data
        self.img_A_names = self.dataframe.iloc[:, 0]  # Get source image & target image name
        self.img_B_names = self.dataframe.iloc[:, 1]
        self.flips = self.dataframe.iloc[:, 3].values.astype('int')
        self.dataset_path = dataset_path  # Path for reading images
        self.out_h, self.out_w = output_size
        self.normalize = normalize
        self.random_crop = random_crop
        # Initialize an affine transformation to resize the image to (240, 240)
        self.affineTnf = GeometricTnf(geometric_model='affine', out_h=self.out_h, out_w=self.out_w, use_cuda=False)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        flip = self.flips[idx]
        # Read image
        image_A = self.get_image(img_name_list=self.img_A_names, idx=idx, flip=flip)
        image_B = self.get_image(img_name_list=self.img_B_names, idx=idx, flip=flip)

        sample = {'source_image': image_A, 'target_image': image_B}

        sample = self.normalize(sample)

        return sample

    def get_image(self, img_name_list, idx, flip=False):
        img_name = os.path.join(self.dataset_path, img_name_list[idx])
        image = cv2.imread(img_name)  # cv2: channel is BGR
        # If the image just has two channels, add one channel
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.concatenate((image, image, image), axis=2)

        # do random crop
        if self.random_crop:
            h, w, c = image.shape
            top = np.random.randint(h / 4)
            bottom = int(3 * h / 4 + np.random.randint(h / 4))
            left = np.random.randint(w / 4)
            right = int(3 * w / 4 + np.random.randint(w / 4))
            image = image[top:bottom, left:right, :]

        # Flip horizontally
        if flip:
            image = image[:, ::-1, :]

        # Transform numpy to tensor, permute order of image to CHW
        image = image[:, :, ::-1]   # BGR -> RGB, due to cv2
        image = torch.Tensor(image.astype(np.float32))
        image = image.permute(2, 0, 1)  # For following normalization
        # Resize image using bilinear sampling with identity affine tnf
        image.requires_grad = False
        image = self.affineTnf(image_batch=image.unsqueeze(0)).squeeze(0)

        return image
