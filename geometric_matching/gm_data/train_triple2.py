# =================================================================================================================
# Generate a synthetically training triple using a given geometric transformation from PF-PASCAL
# Author: Jingwei Qu
# Date: 18 September 2019
# =================================================================================================================

from __future__ import print_function, division
import os
import sys
import numpy as np
import torch

from geometric_matching.geotnf.transformation_tps2 import GeometricTnf
from geometric_matching.util.net_util import roi_data
from geometric_matching.image.normalization import normalize_image

class TrainTriple(object):
    def __init__(self, geometric_model='tps', output_size=(240, 240), crop_factor=1.0, padding_factor=0.5, use_cuda=True, normalize=None):
        assert isinstance(output_size, (tuple))
        assert isinstance(crop_factor, (float))
        assert isinstance(padding_factor, (float))
        assert isinstance(use_cuda, (bool))
        self.out_h, self.out_w = output_size
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.use_cuda = use_cuda
        self.normalize = normalize
        # Initialize geometric transformation (tps or affine) to warp the image to form the training pair
        self.geometricTnf = GeometricTnf(geometric_model=geometric_model, out_h=self.out_h, out_w=self.out_w, use_cuda=self.use_cuda)

    def __call__(self, batch=None):

        """
            Generate a synthetically training triple:
            1. Use the given image pair as the source image and target image;
            2. Padding the source image, and wrap the padding image with the given transformation to generate the refer image;
            3. The training triple consists of {source image, target image, refer image, theta_GT}
            4. The input for fasterRCNN consists of {source_im, target_im, refer_im, source_im_info, target_im_info,
             refer_im_info, source_gt_boxes, target_gt_boxes, refer_gt_boxes, source_num_boxes, target_num_boxes,
             refer_num_boxes}.
        """

        # image_batch.shape: (batch_size, 3, H, W)
        # theta_batch.shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        # boxes.shape: (batch_size, 4), 4: (x_min, y_min, x_max, y_max)
        img_A_batch = batch['source_image']
        img_B_batch = batch['target_image']
        theta_batch = batch['theta_GT']

        # Generate symmetrically padded image for bigger sampling region to warp the source image
        padded_image_batch = self.symmetricImagePad(image_batch=img_A_batch, padding_factor=self.padding_factor)

        img_A_batch.requires_grad = False
        img_B_batch.requires_grad = False
        padded_image_batch.requires_grad = False
        theta_batch.requires_grad = False

        # Get the refer image by warping the padded image with the given transformation
        warped_image_batch = self.geometricTnf(image_batch=padded_image_batch, theta_batch=theta_batch,
                                               padding_factor=self.padding_factor, crop_factor=self.crop_factor)

        # img_A_batch.shape, img_B_batch.shape, warped_image_batch.shape: (batch_size, 3, 240, 240)
        # theta_batch.shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        # theta_batch.shape-affine: (batch_size, 2, 3)
        return {'source_image': img_A_batch, 'target_image': img_B_batch, 'refer_image': warped_image_batch,
                'source_im_info': batch['source_im_info'], 'target_im_info': batch['target_im_info'], 'refer_im_info': batch['source_im_info'],
                'theta_GT': theta_batch}

    def symmetricImagePad(self, image_batch, padding_factor):
        b, c, h, w = image_batch.size()
        pad_h, pad_w = int(h * padding_factor), int(w * padding_factor)
        # Use these four regions to perform symmetric padding for the image
        idx_pad_left = torch.LongTensor(range(pad_w - 1, -1, -1))
        idx_pad_right = torch.LongTensor(range(w - 1, w - pad_w - 1, -1))
        idx_pad_top = torch.LongTensor(range(pad_h - 1, -1, -1))
        idx_pad_bottom = torch.LongTensor(range(h - 1, h - pad_h - 1, -1))
        if self.use_cuda:
            idx_pad_left = idx_pad_left.cuda()
            idx_pad_right = idx_pad_right.cuda()
            idx_pad_top = idx_pad_top.cuda()
            idx_pad_bottom = idx_pad_bottom.cuda()
        # Symmetric padding for the image
        image_batch = torch.cat((image_batch.index_select(3, idx_pad_left), image_batch,
                                 image_batch.index_select(3, idx_pad_right)), 3)
        image_batch = torch.cat((image_batch.index_select(2, idx_pad_top), image_batch,
                                 image_batch.index_select(2, idx_pad_bottom)), 2)

        return image_batch