# ==============================================================================================================
# Generate a testing dataset from TSS for geometric matching model
# Author: Ignacio Rocco
# Modification: Jingwei Qu
# Date: 08 June 2019
# ==============================================================================================================

import os
import torch
from torch.autograd import Variable
from skimage import io
import cv2
import pandas as pd
import numpy as np
from geometric_matching.gm_data.test_dataset import TestDataset
from geometric_matching.geotnf.transformation import GeometricTnf
from geometric_matching.geotnf.flow import read_flo_file
from geometric_matching.util.net_util import roi_data

class TSSDataset2(TestDataset):
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
        super(TSSDataset2, self).__init__(csv_file=csv_file, dataset_path=dataset_path, output_size=output_size, normalize=normalize)
        self.img_A_names = self.dataframe.iloc[:, 0]  # Get source image & target image name
        self.img_B_names = self.dataframe.iloc[:, 1]
        self.flow_direction = self.dataframe.iloc[:, 2].values.astype('int')
        self.flip_img_A = self.dataframe.iloc[:, 3].values.astype('int')
        self.pair_category = self.dataframe.iloc[:, 4].values.astype('int')

    def __getitem__(self, idx):
        # get pre-processed images
        flip_img_A = self.flip_img_A[idx]
        if self.normalize is not None:
            image_A, im_A, im_info_A, gt_boxes_A, num_boxes_A = self.get_image(img_name_list=self.img_A_names, idx=idx, flip=flip_img_A)
            image_B, im_B, im_info_B, gt_boxes_B, num_boxes_B = self.get_image(img_name_list=self.img_B_names, idx=idx)
        else:
            image_A, im_info_A, gt_boxes_A, num_boxes_A = self.get_image(img_name_list=self.img_A_names, idx=idx, flip=flip_img_A)
            image_B, im_info_B, gt_boxes_B, num_boxes_B = self.get_image(img_name_list=self.img_B_names, idx=idx)

        # get flow output path
        flow_path = self.get_GT_flow_relative_path(idx)

        sample = {'source_image': image_A, 'target_image': image_B,
                  'source_im_info': im_info_A, 'target_im_info': im_info_B,
                  'source_gt_boxes': gt_boxes_A, 'target_gt_boxes': gt_boxes_B,
                  'source_num_boxes': num_boxes_A, 'target_num_boxes': num_boxes_B,
                  'flow_path': flow_path}

        # # get ground-truth flow
        # flow = self.get_GT_flow(idx)

        # sample = {'source_image': image_A, 'target_image': image_B, 'source_im_size': im_size_A, 'target_im_size': im_size_B, 'flow_GT': flow}

        if self.normalize is not None:
            sample = {'source_image': image_A, 'target_image': image_B,
                      'source_im': im_A, 'target_im': im_B,
                      'source_im_info': im_info_A, 'target_im_info': im_info_B,
                      'source_gt_boxes': gt_boxes_A, 'target_gt_boxes': gt_boxes_B,
                      'source_num_boxes': num_boxes_A, 'target_num_boxes': num_boxes_B,
                      'flow_path': flow_path}
            sample = self.normalize(sample)

        return sample

    def get_GT_flow(self, idx):
        img_folder = os.path.dirname(self.img_A_names[idx])
        flow_dir = self.flow_direction[idx]
        flow_file = 'flow' + str(flow_dir) + '.flo'
        flow_file_path = os.path.join(self.dataset_path, img_folder, flow_file)

        flow = torch.FloatTensor(read_flo_file(flow_file_path))

        return flow

    def get_GT_flow_relative_path(self, idx):
        img_folder = os.path.dirname(self.img_A_names[idx])
        flow_dir = self.flow_direction[idx]
        flow_file = 'flow' + str(flow_dir) + '.flo'
        flow_file_path = os.path.join(img_folder, flow_file)

        return flow_file_path
