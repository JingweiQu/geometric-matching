# =================================================================================================================
# Generate a synthetically training triple using a given geometric transformation from PF-PASCAL
# Author: Jingwei Qu
# Date: 27 April 2019
# =================================================================================================================

from __future__ import print_function, division
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

from geometric_matching.geotnf.transformation import GeometricTnf, ComposedGeometricTnf
from geometric_matching.util.net_util import roi_data
from geometric_matching.image.normalization import normalize_image

class TrainDualTriple(object):
    def __init__(self, geometric_model='afftps', output_size=(240, 240), crop_factor=1.0, padding_factor=0.5, use_cuda=True, normalize=None):
        assert isinstance(output_size, (tuple))
        assert isinstance(crop_factor, (float))
        assert isinstance(padding_factor, (float))
        assert isinstance(use_cuda, (bool))
        self.out_h, self.out_w = output_size
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        # self.crop_layer = crop_layer
        self.use_cuda = use_cuda
        self.normalize = normalize
        # Initialize geometric transformation (tps or affine) to warp the image to form the training pair
        # self.affTnf = GeometricTnf(geometric_model='affine', out_h=self.out_h, out_w=self.out_w, use_cuda=self.use_cuda)
        # self.tpsTnf = GeometricTnf(geometric_model='tps', out_h=self.out_h, out_w=self.out_w, use_cuda=self.use_cuda)
        self.geometricTnf = ComposedGeometricTnf(padding_crop_factor=padding_factor * crop_factor, use_cuda=self.use_cuda)

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
        theta_aff = batch['theta_GT'][:, :6].contiguous()
        theta_tps = batch['theta_GT'][:, 6:]

        # Generate symmetrically padded image for bigger sampling region to warp the source image
        padded_image_batch = self.symmetricImagePad(image_batch=img_A_batch, padding_factor=self.padding_factor)

        img_A_batch.requires_grad = False
        img_B_batch.requires_grad = False
        padded_image_batch.requires_grad = False
        theta_aff.requires_grad = False
        theta_tps.requires_grad = False

        # Get the refer image by warping the padded image with the given transformation
        # warped_image_batch = self.affTnf(image_batch=padded_image_batch, theta_batch=theta_aff, padding_factor=0.5, crop_factor=self.crop_factor)
        # warped_image_batch = self.symmetricImagePad(image_batch=warped_image_batch, padding_factor=self.padding_factor)
        # warped_image_batch = self.tpsTnf(image_batch=warped_image_batch, theta_batch=theta_tps, padding_factor=0.5, crop_factor=self.crop_factor)
        # warped_image_aff = self.affTnf(image_batch=padded_image_batch, theta_batch=theta_aff, padding_factor=self.padding_factor, crop_factor=self.crop_factor)
        # warped_image_tps = self.tpsTnf(image_batch=padded_image_batch, theta_batch=theta_tps, padding_factor=self.padding_factor, crop_factor=self.crop_factor)

        # warped_image_batch = self.affTpsTnf(source_image=padded_image_batch, theta_aff=theta_aff, theta_aff_tps=theta_tps, use_cuda=self.use_cuda)
        warped_image_batch = self.geometricTnf(image_batch=padded_image_batch, theta_aff=theta_aff, theta_aff_tps=theta_tps)

        # Get the refer im for extracting rois
        tmp_image_batch = normalize_image(image=warped_image_batch, forward=False) * 255.0
        tmp_image_batch = tmp_image_batch.cpu().numpy().transpose((0, 2, 3, 1))
        tmp_image_batch = tmp_image_batch[:, :, :, ::-1]    # RGB -> BGR
        warped_im_batch = torch.zeros_like(warped_image_batch, dtype=torch.float32)
        for i in range(tmp_image_batch.shape[0]):
            warped_im_batch[i] = roi_data(tmp_image_batch[i])[0]
        warped_im_batch = warped_im_batch.cuda()
        warped_im_batch.requires_grad = False

        # img_A_batch.shape, img_B_batch.shape, warped_image_batch.shape: (batch_size, 3, 240, 240)
        # theta_batch.shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        # theta_batch.shape-affine: (batch_size, 2, 3)
        # return {'source_image': img_A_batch, 'target_image': img_B_batch, 'refer_image': warped_image_batch,
        #         'source_im': batch['source_im'], 'target_im': batch['target_im'], 'refer_im': warped_im_batch,
        #         'source_im_info': batch['source_im_info'], 'target_im_info': batch['target_im_info'], 'refer_im_info': batch['source_im_info'],
        #         'source_gt_boxes': batch['source_gt_boxes'], 'target_gt_boxes': batch['target_gt_boxes'], 'refer_gt_boxes': batch['source_gt_boxes'],
        #         'source_num_boxes': batch['source_num_boxes'], 'target_num_boxes': batch['target_num_boxes'], 'refer_num_boxes': batch['source_num_boxes'],
        #         'theta_aff_GT': theta_aff, 'theta_tps_GT': theta_tps}

        return {'source_image': img_A_batch, 'target_image': img_B_batch, 'refer_image': warped_image_batch,
                'source_im_info': batch['source_im_info'], 'target_im_info': batch['target_im_info'],
                'refer_im_info': batch['source_im_info'],
                'theta_aff_GT': theta_aff, 'theta_tps_GT': theta_tps}

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

    def affTpsTnf(self, source_image, theta_aff, theta_aff_tps, use_cuda=True):
        tpstnf = GeometricTnf(geometric_model='tps', out_h=240, out_w=240, use_cuda=use_cuda)
        sampling_grid_tps = tpstnf(image_batch=source_image, theta_batch=theta_aff_tps, padding_factor=0.5, crop_factor=1.0, return_sampling_grid=True)[1]
        X = sampling_grid_tps[:, :, :, 0].unsqueeze(3)
        Y = sampling_grid_tps[:, :, :, 1].unsqueeze(3)
        Xp = X * theta_aff[:, 0].unsqueeze(1).unsqueeze(2) + Y * theta_aff[:, 1].unsqueeze(1).unsqueeze(
            2) + theta_aff[:, 2].unsqueeze(1).unsqueeze(2)
        Yp = X * theta_aff[:, 3].unsqueeze(1).unsqueeze(2) + Y * theta_aff[:, 4].unsqueeze(1).unsqueeze(
            2) + theta_aff[:, 5].unsqueeze(1).unsqueeze(2)
        sampling_grid = torch.cat((Xp, Yp), 3)
        warped_image_batch = F.grid_sample(source_image, sampling_grid)

        return warped_image_batch