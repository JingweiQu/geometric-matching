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
from geometric_matching.gm_data.test_dataset import TestDataset
from geometric_matching.geotnf.transformation import GeometricTnf
from geometric_matching.util.net_util import roi_data

class WatchDataset2(TestDataset):

    def __init__(self, csv_file, dataset_path, output_size=(240, 240), normalize=None):
        super(WatchDataset2, self).__init__(csv_file=csv_file, dataset_path=dataset_path, output_size=output_size, normalize=normalize)
        self.img_A_names = self.dataframe.iloc[:, 0]  # Get source image & target image name
        self.img_B_names = self.dataframe.iloc[:, 1]
        self.flip_img_A = self.dataframe.iloc[:, 2].values.astype('int')

    def __getitem__(self, idx):
        # get pre-processed images
        flip_img_A = self.flip_img_A[idx]
        if self.normalize is not None:
            image_A, im_A, im_info_A, gt_boxes_A, num_boxes_A = self.get_image(img_name_list=self.img_A_names, idx=idx, flip=flip_img_A)
            image_B, im_B, im_info_B, gt_boxes_B, num_boxes_B = self.get_image(img_name_list=self.img_B_names, idx=idx)
        else:
            image_A, im_info_A, gt_boxes_A, num_boxes_A = self.get_image(img_name_list=self.img_A_names, idx=idx, flip=flip_img_A)
            image_B, im_info_B, gt_boxes_B, num_boxes_B = self.get_image(img_name_list=self.img_B_names, idx=idx)

        sample = {'source_image': image_A, 'target_image': image_B,
                  'source_im_info': im_info_A, 'target_im_info': im_info_B,
                  'source_gt_boxes': gt_boxes_A, 'target_gt_boxes': gt_boxes_B,
                  'source_num_boxes': num_boxes_A, 'target_num_boxes': num_boxes_B}

        if self.normalize is not None:
            # sample = {'source_image': image_A, 'target_image': image_B,
            #           'source_im': im_A, 'target_im': im_B,
            #           'source_im_info': im_info_A, 'target_im_info': im_info_B,
            #           'source_gt_boxes': gt_boxes_A, 'target_gt_boxes': gt_boxes_B,
            #           'source_num_boxes': num_boxes_A, 'target_num_boxes': num_boxes_B}
            sample = {'source_image': image_A, 'target_image': image_B}
            sample = self.normalize(sample)

        return sample