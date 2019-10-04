# ==============================================================================================================
# Generate a testing dataset from TSS for cosegmentation model
# Author: Ignacio Rocco
# Modification: Jingwei Qu
# Date: 02 Sep 2019
# ==============================================================================================================

import os
import torch
from torch.autograd import Variable
from skimage import io
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geometric_matching.geotnf.transformation import GeometricTnf

class TSSDataset(Dataset):
    """
    TSS image pair dataset (http://taniai.space/projects/cvpr16_dccs/)

    Args:
        csv_file (string): Path to the csv file with image names and annotation files.
        dataset_path (string): Directory with the images.
        output_size (2-tuple): Desired output size
        transform (callable): Transformation for post-processing the training pair (eg. image normalization)

    Returns:
        Dict: {'source_image' & 'target_image': images for transformation,
               'source_im_size' & 'target_im_size': size of images
               'flow_path': ground-truth flow relative path
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
        self.img_A_names = self.dataframe.iloc[:, 0]  # Get source image & target image name
        self.img_B_names = self.dataframe.iloc[:, 1]
        self.mask_A_names = self.dataframe.iloc[:, 2]
        self.mask_B_names = self.dataframe.iloc[:, 3]
        self.flow_direction = self.dataframe.iloc[:, 4].values.astype('int')
        self.flip_img_A = self.dataframe.iloc[:, 5].values.astype('int')
        self.pair_category = self.dataframe.iloc[:, 6].values.astype('int')
        # Initialize an affine transformation to resize the image to (240, 240)
        self.affineTnf = GeometricTnf(geometric_model='affine', out_h=self.out_h, out_w=self.out_w, use_cuda=False)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # get pre-processed images
        flip_img_A = self.flip_img_A[idx]
        image_A = self.get_image(img_name_list=self.img_A_names, idx=idx, flip=flip_img_A)
        image_B = self.get_image(img_name_list=self.img_B_names, idx=idx)

        # get gt-mask
        mask_A = self.get_mask(mask_name_list=self.mask_A_names, idx=idx, flip=flip_img_A)
        mask_B = self.get_mask(mask_name_list=self.mask_B_names, idx=idx)

        sample = {'source_image': image_A, 'target_image': image_B,
                  'source_mask': mask_A, 'target_mask': mask_B}

        sample = self.normalize(sample)

        return sample

    def get_image(self, img_name_list, idx, flip=False):
        img_name = os.path.join(self.dataset_path, img_name_list[idx])
        image = cv2.imread(img_name)  # cv2: channel is BGR
        # If the image just has two channels, add one channel
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.concatenate((image, image, image), axis=2)

        # Flip horizontally
        if flip:
            image = image[:, ::-1, :]

        # Transform numpy to tensor, permute order of image to CHW
        image = image[:, :, ::-1]  # BGR -> RGB, due to cv2
        image = torch.Tensor(image.astype(np.float32))
        image = image.permute(2, 0, 1)  # For following normalization
        # Resize image using bilinear sampling with identity affine tnf
        image.requires_grad = False
        image = self.affineTnf(image_batch=image.unsqueeze(0)).squeeze(0)

        return image

    def get_mask(self, mask_name_list, idx, flip=False):
        mask_name = os.path.join(self.dataset_path, mask_name_list[idx])
        mask = io.imread(mask_name)
        # mask = cv2.imread(mask_name)

        # Flip horizontally
        if flip:
            mask = mask[:, ::-1]

        # Transform numpy to tensor, permute order of image to CHW
        mask = torch.Tensor(mask.astype(np.float32)).unsqueeze(0)
        # mask = mask.permute(2, 0, 1)  # For following normalization
        # Resize image using bilinear sampling with identity affine tnf
        mask.requires_grad = False
        mask = self.affineTnf(image_batch=mask.unsqueeze(0)).squeeze(0)
        mask /= 255.0

        return mask
