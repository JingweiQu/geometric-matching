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
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geometric_matching.geotnf.transformation import GeometricTnf
from geometric_matching.geotnf.flow import read_flo_file
from geometric_matching.util.net_util import roi_data

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

    def __init__(self, csv_file, dataset_path, output_size=(240,240), transform=None):
        self.dataframe = pd.read_csv(csv_file)  # Read images data
        self.img_A_names = self.dataframe.iloc[:, 0]    # Get source image & target image name
        self.img_B_names = self.dataframe.iloc[:, 1]
        self.flow_direction = self.dataframe.iloc[:, 2].values.astype('int')
        self.flip_img_A = self.dataframe.iloc[:, 3].values.astype('int')
        self.pair_category = self.dataframe.iloc[:, 4].values.astype('int')
        self.dataset_path = dataset_path
        self.out_h, self.out_w = output_size
        self.transform = transform
        # Initialize an affine transformation to resize the image to (240, 240)
        self.affineTnf = GeometricTnf(geometric_model='affine', out_h=self.out_h, out_w=self.out_w, use_cuda=False)
              
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # get pre-processed images
        flip_img_A = self.flip_img_A[idx]
        image_A, im_size_A, im_A, im_info_A, gt_boxes_A, num_boxes_A = self.get_image(img_name_list=self.img_A_names,
                                                                                      idx=idx, flip=flip_img_A)
        image_B, im_size_B, im_B, im_info_B, gt_boxes_B, num_boxes_B = self.get_image(img_name_list=self.img_B_names,
                                                                                      idx=idx)

        # get flow output path
        flow_path = self.get_GT_flow_relative_path(idx)

        sample = {'source_image': image_A, 'target_image': image_B,
                  'source_im_size': im_size_A, 'target_im_size': im_size_B,
                  'flow_path': flow_path,
                  'source_im': im_A, 'target_im': im_B,
                  'source_im_info': im_info_A, 'target_im_info': im_info_B,
                  'source_gt_boxes': gt_boxes_A, 'target_gt_boxes': gt_boxes_B,
                  'source_num_boxes': num_boxes_A, 'target_num_boxes': num_boxes_B}
        
        # # get ground-truth flow
        # flow = self.get_GT_flow(idx)
        
        # sample = {'source_image': image_A, 'target_image': image_B, 'source_im_size': im_size_A, 'target_im_size': im_size_B, 'flow_GT': flow}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self, img_name_list, idx, flip=False):
        img_name = os.path.join(self.dataset_path, img_name_list[idx])
        image = io.imread(img_name)
        # If the image just has two channels, add one channel
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.concatenate((image, image, image), axis=2)
            
        # Flip horizontally
        if flip:
            image = np.flip(image, axis=1)
            
        # get image size
        im_size = np.asarray(image.shape)

        # Get tensors of image, image_info (H, W, im_scale), ground-truth boxes, number of boxes for faster rcnn
        im, im_info, gt_boxes, num_boxes = roi_data(image)

        # Transform numpy to tensor, permute order of image to CHW
        image = torch.Tensor(image.astype(np.float32))
        image = image.permute(2, 0, 1)
        im_size = torch.Tensor(im_size.astype(np.float32))

        # Resize image using bilinear sampling with identity affine tnf
        image.requires_grad = False
        image = self.affineTnf(image_batch=image.unsqueeze(0)).squeeze(0)
        
        return image, im_size, im, im_info, gt_boxes, num_boxes

    def get_GT_flow(self,idx):
        img_folder = os.path.dirname(self.img_A_names[idx])
        flow_dir = self.flow_direction[idx]
        flow_file = 'flow'+str(flow_dir)+'.flo'
        flow_file_path = os.path.join(self.dataset_path, img_folder , flow_file)
        
        flow = torch.FloatTensor(read_flo_file(flow_file_path))

        return flow
    
    def get_GT_flow_relative_path(self, idx):
        img_folder = os.path.dirname(self.img_A_names[idx])
        flow_dir = self.flow_direction[idx]
        flow_file = 'flow' + str(flow_dir) + '.flo'
        flow_file_path = os.path.join(img_folder, flow_file)
        
        return flow_file_path
        