# =================================================================================================================
# Generate a synthetically training pair using a given geometric transformation from PascalVOC2011
# Author: Jingwei Qu
# Date: 19 April 2019
# =================================================================================================================

from __future__ import print_function, division
import os
import sys
import numpy as np
import torch
from geometric_matching.util.net_util import roi_data
from geometric_matching.geotnf.transformation import GeometricTnf

class SynthPairTnf(object):
    def __init__(self, use_cuda=True, geometric_model='tps', crop_factor=1.0, output_size=(240, 240),
                 padding_factor=0.5, crop_layer='image'):
        assert isinstance(use_cuda, (bool))
        assert isinstance(crop_factor, (float))
        assert isinstance(output_size, (tuple))
        assert isinstance(padding_factor, (float))
        self.use_cuda = use_cuda
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h, self.out_w = output_size
        # Initialize an affine transformation to resize the image to (240, 240)
        self.rescalingTnf = GeometricTnf('affine', self.out_h, self.out_w, use_cuda=self.use_cuda)
        # Initialize geometric transformation (tps or affine) to warp the image to form the training pair
        self.geometricTnf = GeometricTnf(geometric_model, self.out_h, self.out_w, use_cuda=self.use_cuda)
        self.crop_layer = crop_layer

    def __call__(self, batch, boxes):
        if self.crop_layer == 'pool4':
            return self.pool4_pair(batch)
        elif self.crop_layer == 'object':
            return self.object_pair(batch, boxes)
        elif self.crop_layer == 'image':
            return self.image_pair(batch)

    def image_pair(self, batch):
        image_batch = batch['image']
        theta_batch = batch['theta']
        if self.use_cuda:
            image_batch = image_batch.cuda()
            theta_batch = theta_batch.cuda()

        # Generate symmetrically padded image for bigger sampling region to warp the source image
        image_batch = self.symmetricImagePad(image_batch, self.padding_factor)

        image_batch.requires_grad = False
        theta_batch.requires_grad = False

        # Get the source image by resizing the given image
        cropped_image_batch = self.rescalingTnf(image_batch, None, self.padding_factor, self.crop_factor)

        # Get the target image by warping the padded image with the given transformation
        warped_image_batch = self.geometricTnf(image_batch, theta_batch, self.padding_factor, self.crop_factor)

        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch, 'theta_GT': theta_batch}

    def pool4_pair(self, batch):
        """

            Generate a synthetically training pair:
            1. Use the given image as the source image ;
            2. Padding the source image, and wrap the padding image with the given transformation to generate the target image;
            3. The training pair consists of {source image, target image, theta_GT}
            4. The input for fasterRCNN consists of {source_im, target_im, im_info, gt_boxes, num_boxes}

        """

        # image_batch.shape: (batch_size, 3, H, W)
        # theta_batch.shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        # theta_batch.shape-affine: (batch_size, 2, 3)
        image_batch = batch['image']
        theta_batch = batch['theta']
        if self.use_cuda:
            image_batch = image_batch.cuda()
            theta_batch = theta_batch.cuda()

        # Generate symmetrically padded image for bigger sampling region to warp the source image
        padded_image_batch = self.symmetricImagePad(image_batch, self.padding_factor)

        image_batch.requires_grad = False
        padded_image_batch.requires_grad = False
        theta_batch.requires_grad = False

        # Get the source image by resizing the given image
        resized_image_batch = self.rescalingTnf(image_batch)

        # Get the target image by warping the padded image with the given transformation
        warped_image_batch = self.geometricTnf(padded_image_batch, theta_batch, self.padding_factor, self.crop_factor)

        # Get the target im for extracting rois
        tmp_image_batch = warped_image_batch.clone()
        tmp_image_batch = tmp_image_batch.cpu().numpy().transpose((0, 2, 3, 1))
        warped_im_batch = torch.zeros_like(warped_image_batch, dtype=torch.float)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        for i in range(tmp_image_batch.shape[0]):
            tmp_image_batch[i] = std * tmp_image_batch[i] + mean
            tmp_image_batch[i] *= 255
            warped_im_batch[i], _, _, _ = roi_data(tmp_image_batch[i])

        # resized_image_batch.shape and warped_image_batch.shape: (batch_size, 3, out_h, out_w), such as (240, 240)
        # theta_batch.shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        # theta_batch.shape-affine: (batch_size, 2, 3)

        return {'source_image': resized_image_batch, 'target_image': warped_image_batch, 'theta_GT': theta_batch}, \
               {'source_im': batch['im'], 'target_im': warped_im_batch, 'im_info': batch['im_info'],
                'gt_boxes': batch['gt_boxes'], 'num_boxes': batch['num_boxes']}

    def object_pair(self, batch, boxes):
        """

            Generate a synthetically training pair (object):
            1. Use the given image and object bounding box to crop and resize object as the source image ;
            2. Padding the source image, and wrap the padding image with the given transformation to generate the target image;
            3. The training pair consists of {source image, target image, theta_GT}

        """

        # image_batch.shape: (batch_size, 3, H, W)
        # theta_batch.shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        # theta_batch.shape-affine: (batch_size, 2, 3)
        # boxes.shape: (batch_size, 4), 4: (x_min, y_min, x_max, y_max)
        image_batch = batch['image']
        theta_batch = batch['theta']
        if self.use_cuda:
            image_batch = image_batch.cuda()
            theta_batch = theta_batch.cuda()
            boxes = boxes.cuda()

        # Resize the image from (480, 640) to (240, 240)
        resized_image_batch = self.rescalingTnf(image_batch)

        # Crop and resize object on the image as the source image, (240, 240)
        croped_image_batch = torch.Tensor(resized_image_batch.shape).zero_()
        for i in range(boxes.shape[0]):
            # Crop if object is detected
            if torch.sum(boxes[i]).item() > 0:
                crop_object = resized_image_batch[i, :, int(boxes[i, 1]):int(boxes[i, 3]),
                              int(boxes[i, 0]):int(boxes[i, 2])].unsqueeze(0)
                croped_image_batch[i, :, :, :] = self.rescalingTnf(crop_object).squeeze()
            else:
                croped_image_batch[i, :, :, :] = resized_image_batch[i, :, :, :]

        if self.use_cuda:
            croped_image_batch = croped_image_batch.cuda()

        # Generate symmetrically padded image for bigger sampling region to warp the source image
        padded_image_batch = self.symmetricImagePad(croped_image_batch, self.padding_factor)

        croped_image_batch.requires_grad = False
        padded_image_batch.requires_grad = False
        theta_batch.requires_grad = False

        # Get the target image by warping the padded image with the given transformation
        warped_image_batch = self.geometricTnf(padded_image_batch, theta_batch, self.padding_factor, self.crop_factor)

        # cropped_image_batch.shape and warped_image_batch.shape: (batch_size, 3, out_h, out_w), such as (240, 240)
        # theta_batch.shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        # theta_batch.shape-affine: (batch_size, 2, 3)

        return {'source_image': croped_image_batch, 'target_image': warped_image_batch, 'theta_GT': theta_batch}

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