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

from geometric_matching.geotnf.transformation import GeometricTnf
from geometric_matching.util.net_util import roi_data
from geometric_matching.image.normalization import normalize_image

class TrainTriple(object):
    def __init__(self, geometric_model='affine', output_size=(240, 240), crop_factor=1.0, padding_factor=0.5, use_cuda=True, normalize=None):
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
        # Initialize an affine transformation to resize the cropped object to (240, 240)
        # if self.crop_layer == 'object':
        #     self.rescalingTnf = GeometricTnf(geometric_model='affine', out_h=self.out_h, out_w=self.out_w,
        #                                      use_cuda=self.use_cuda)
        # Initialize geometric transformation (tps or affine) to warp the image to form the training pair
        self.geometricTnf = GeometricTnf(geometric_model=geometric_model, out_h=self.out_h, out_w=self.out_w,
                                         use_cuda=self.use_cuda)
        # self.rescalingTnf = GeometricTnf(geometric_model='affine', out_h=self.out_h, out_w=self.out_w,
        #                                  use_cuda=self.use_cuda)

    # def __call__(self, batch=None, boxes_A=None, boxes_B=None):
    def __call__(self, batch=None):
        # if self.crop_layer == 'pool4':
        #     return self.pool4_triple(batch)
        # elif self.crop_layer == 'object':
        #     return self.object_triple(batch=batch, boxes_A=boxes_A, boxes_B=boxes_B)
        # elif self.crop_layer == 'image':
        #     return self.image_triple(batch)

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

        if self.normalize is not None:
            # Get the refer im for extracting rois
            tmp_image_batch = normalize_image(image=warped_image_batch, forward=False) * 255.0
            tmp_image_batch = tmp_image_batch.cpu().numpy().transpose((0, 2, 3, 1))
            tmp_image_batch = tmp_image_batch[:, :, :, ::-1]    # RGB -> BGR
            warped_im_batch = torch.zeros_like(warped_image_batch, dtype=torch.float32)
            for i in range(tmp_image_batch.shape[0]):
                warped_im_batch[i] = roi_data(tmp_image_batch[i])[0]
            warped_im_batch.requires_grad = False

            # return {'source_image': img_A_batch, 'target_image': img_B_batch,
            #         'source_im': batch['source_im'], 'target_im': batch['target_im'],
            #         'source_im_info': batch['source_im_info'], 'target_im_info': batch['target_im_info'],
            #         'source_gt_boxes': batch['source_gt_boxes'], 'target_gt_boxes': batch['target_gt_boxes'],
            #         'source_num_boxes': batch['source_num_boxes'], 'target_num_boxes': batch['target_num_boxes']}, \
            #        {'source_image': img_B_batch, 'target_image': warped_image_batch,
            #         'source_im': batch['target_im'], 'target_im': warped_im_batch,
            #         'source_im_info': batch['target_im_info'], 'target_im_info': batch['source_im_info'],
            #         'source_gt_boxes': batch['target_gt_boxes'], 'target_gt_boxes': batch['source_gt_boxes'],
            #         'source_num_boxes': batch['target_num_boxes'], 'target_num_boxes': batch['source_num_boxes']}, \
            #        theta_batch
            return {'source_image': img_A_batch, 'target_image': img_B_batch}, \
                   {'source_image': img_B_batch, 'target_image': warped_image_batch}, \
                   theta_batch

        # img_A_batch.shape, img_B_batch.shape, warped_image_batch.shape: (batch_size, 3, 240, 240)
        # theta_batch.shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        # theta_batch.shape-affine: (batch_size, 2, 3)
        return {'source_image': img_A_batch, 'target_image': img_B_batch,
                'source_im_info': batch['source_im_info'], 'target_im_info': batch['target_im_info'],
                'source_gt_boxes': batch['source_gt_boxes'], 'target_gt_boxes': batch['target_gt_boxes'],
                'source_num_boxes': batch['source_num_boxes'], 'target_num_boxes': batch['target_num_boxes']}, \
               {'source_image': img_B_batch, 'target_image': warped_image_batch,
                'source_im_info': batch['target_im_info'], 'target_im_info': batch['source_im_info'],
                'source_gt_boxes': batch['target_gt_boxes'], 'target_gt_boxes': batch['source_gt_boxes'],
                'source_num_boxes': batch['target_num_boxes'], 'target_num_boxes': batch['source_num_boxes']}, \
               theta_batch

        # return {'source_image': img_A_batch, 'target_image': img_B_batch, 'refer_image': warped_image_batch,
        #         'theta_GT': theta_batch,
        #         'source_im': batch['source_im'], 'target_im': batch['target_im'], 'refer_im': warped_im_batch,
        #         'source_im_info': batch['source_im_info'], 'target_im_info': batch['target_im_info'], 'refer_im_info': batch['source_im_info'],
        #         'source_gt_boxes': batch['source_gt_boxes'], 'target_gt_boxes': batch['target_gt_boxes'], 'refer_gt_boxes': batch['source_gt_boxes'],
        #         'source_num_boxes': batch['source_num_boxes'], 'target_num_boxes': batch['target_num_boxes'], 'refer_num_boxes': batch['source_num_boxes']}

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

'''
    def image_triple(self, batch):
        """

            Generate a synthetically training triple:
            1. Use the given image pair as the source image and target image;
            2. Padding the source image, and wrap the padding image with the given transformation to generate the refer image;
            3. The training triple consists of {source image, target image, refer image, theta_GT}.

        """

        # image_batch.shape: (batch_size, 3, H, W)
        # theta_batch.shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        # boxes.shape: (batch_size, 4), 4: (x_min, y_min, x_max, y_max)
        img_A_batch = batch['source_image']
        img_B_batch = batch['target_image']
        theta_batch = batch['theta']

        # if self.use_cuda:
        #     img_A_batch = img_A_batch.cuda()
        #     img_B_batch = img_B_batch.cuda()
        #     theta_batch = theta_batch.cuda()

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
                'theta_GT': theta_batch}
        # return {'source_image': img_A_batch, 'target_image': img_B_batch, 'refer_image': img_A_batch,
        #         'theta_GT': theta_batch}

    def pool4_triple(self, batch):
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
        # theta_batch.shape-affine: (batch_size, 2, 3)
        # boxes.shape: (batch_size, 4), 4: (x_min, y_min, x_max, y_max)
        img_A_batch = batch['source_image']
        img_B_batch = batch['target_image']
        theta_batch = batch['theta']

        # if self.use_cuda:
        #     img_A_batch = img_A_batch.cuda()
        #     img_B_batch = img_B_batch.cuda()
        #     theta_batch = theta_batch.cuda()

        # Generate symmetrically padded image for bigger sampling region to warp the source image
        padded_image_batch = self.symmetricImagePad(image_batch=img_A_batch, padding_factor=self.padding_factor)

        img_A_batch.requires_grad = False
        img_B_batch.requires_grad = False
        padded_image_batch.requires_grad = False
        theta_batch.requires_grad = False

        # Get the refer image by warping the padded image with the given transformation
        warped_image_batch = self.geometricTnf(image_batch=padded_image_batch, theta_batch=theta_batch,
                                               padding_factor=self.padding_factor, crop_factor=self.crop_factor)

        # Get the refer im for extracting rois
        tmp_image_batch = warped_image_batch.clone()
        tmp_image_batch = tmp_image_batch.cpu().numpy().transpose((0, 2, 3, 1))
        warped_im_batch = torch.zeros_like(warped_image_batch, dtype=torch.float)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        for i in range(tmp_image_batch.shape[0]):
            tmp_image_batch[i] = std * tmp_image_batch[i] + mean
            tmp_image_batch[i] *= 255
            warped_im_batch[i], _, _, _ = roi_data(tmp_image_batch[i])

        # img_A_batch.shape, img_B_batch.shape, warped_image_batch.shape: (batch_size, 3, 240, 240)
        # theta_batch.shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        # theta_batch.shape-affine: (batch_size, 2, 3)
        return {'source_image': img_A_batch, 'target_image': img_B_batch, 'refer_image': warped_image_batch,
                'theta_GT': theta_batch,
                'source_im': batch['source_im'], 'target_im': batch['target_im'], 'refer_im': warped_im_batch,
                'source_im_info': batch['source_im_info'], 'target_im_info': batch['target_im_info'],
                'refer_im_info': batch['source_im_info'],
                'source_gt_boxes': batch['source_gt_boxes'], 'target_gt_boxes': batch['target_gt_boxes'],
                'refer_gt_boxes': batch['source_gt_boxes'],
                'source_num_boxes': batch['source_num_boxes'], 'target_num_boxes': batch['target_num_boxes'],
                'refer_num_boxes': batch['source_num_boxes']}

    def object_triple(self, batch, boxes_A, boxes_B):
        """
            Generate a synthetically training triple (object):
            1. Use the given image pair and object bounding boxes to crop and resize objects as the source image and target image;
            2. Padding the source image, and wrap the padding image with the given transformation to generate the refer image;
            3. The training triple consists of {source image, target image, refer image, theta_GT}
        """

        # image_batch.shape: (batch_size, 3, H, W)
        # theta_batch.shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        # theta_batch.shape-affine: (batch_size, 2, 3)
        # boxes.shape: (batch_size, 4), 4: (x_min, y_min, x_max, y_max)
        img_A_batch = batch['source_image']
        img_B_batch = batch['target_image']
        theta_batch = batch['theta']

        # if self.use_cuda:
        #     img_A_batch = img_A_batch.cuda()
        #     img_B_batch = img_B_batch.cuda()
        #     theta_batch = theta_batch.cuda()
        #     boxes_A = boxes_A.cuda()
        #     boxes_B = boxes_B.cuda()

        # Crop and resize objects on the image pair as the source image and target image, (240, 240)
        obj_A_batch = self.crop_object(image_batch=img_A_batch, boxes=boxes_A)
        obj_B_batch = self.crop_object(image_batch=img_B_batch, boxes=boxes_B)

        # if self.use_cuda:
        #     obj_A_batch = obj_A_batch.cuda()
        #     obj_B_batch = obj_B_batch.cuda()

        # Generate symmetrically padded image for bigger sampling region to warp the source image
        padded_obj_batch = self.symmetricImagePad(image_batch=obj_A_batch, padding_factor=self.padding_factor)

        obj_A_batch.requires_grad = False
        obj_B_batch.requires_grad = False
        padded_obj_batch.requires_grad = False
        theta_batch.requires_grad = False

        # Get the refer image by warping the padded image with the given transformation
        warped_obj_batch = self.geometricTnf(image_batch=padded_obj_batch, theta_batch=theta_batch,
                                             padding_factor=self.padding_factor, crop_factor=self.crop_factor)

        # obj_A_batch.shape, obj_B_batch.shape, warped_obj_batch.shape: (batch_size, 3, 240, 240)
        # theta_batch.shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        # theta_batch.shape-affine: (batch_size, 2, 3)
        return {'source_image': obj_A_batch, 'target_image': obj_B_batch, 'refer_image': warped_obj_batch,
                'theta_GT': theta_batch}

    def crop_object(self, image_batch, boxes):
        # Crop object and resize object as (240, 240)
        croped_image_batch = torch.Tensor(image_batch.shape).zero_()
        if self.use_cuda and not croped_image_batch.is_cuda:
            croped_image_batch = croped_image_batch.cuda()
        for i in range(boxes.shape[0]):
            # Crop if object is detected
            if torch.sum(boxes[i]).item() > 0:
                croped_object = image_batch[i, :, int(boxes[i, 1]):int(boxes[i, 3]),
                              int(boxes[i, 0]):int(boxes[i, 2])].unsqueeze(0)
                croped_image_batch[i, :, :, :] = self.rescalingTnf(image_batch=croped_object).squeeze()
            else:
                croped_image_batch[i, :, :, :] = image_batch[i, :, :, :]

        return croped_image_batch
'''