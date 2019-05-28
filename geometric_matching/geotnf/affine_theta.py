# ====================================================================================================
# Obtain affine parameters between a image pair based on object detection of fasterRCNN
# Author: Jingwei Qu
# Date: 27 April 2019
# ====================================================================================================

import numpy as np
import torch
import cv2

from geometric_matching.geotnf.point_tnf import PointsToUnitCoords

class AffineTheta(object):
    """

        Use the coordinates of three corner points on bounding boxes (i.e. object detection of the two images) to
        compute affine parameters (translation and scale)

    """

    def __init__(self, use_cuda=True, original=False, image_size=240):
        self.use_cuda = use_cuda
        self.original = original
        self.image_size = image_size

    def __call__(self, boxes_s, boxes_t, source_im_size=None, target_im_size=None):
        batch_size = boxes_s.shape[0]

        if self.original:
            # Re-locate the bounding box of the object in the original image
            for i in range(batch_size):
                boxes_s[i, 0::2] *= (source_im_size[i, 1] / self.image_size)
                boxes_s[i, 1::2] *= (source_im_size[i, 0] / self.image_size)

                boxes_t[i, 0::2] *= (target_im_size[i, 1] / self.image_size)
                boxes_t[i, 1::2] *= (target_im_size[i, 0] / self.image_size)
        else:
            source_im_size = torch.Tensor(batch_size, 3)
            source_im_size[:, :2] = self.image_size
            source_im_size[:, 2] = 3
            target_im_size = source_im_size.clone()

        src_pts = torch.Tensor(batch_size, 2, 3)
        dst_pts = torch.Tensor(batch_size, 2, 3)
        theta_aff = torch.Tensor(batch_size, 2, 3)

        if self.use_cuda:
            src_pts = src_pts.cuda()
            dst_pts = dst_pts.cuda()
            source_im_size = source_im_size.cuda()
            target_im_size = target_im_size.cuda()
            boxes_s = boxes_s.cuda()
            boxes_t = boxes_t.cuda()
            theta_aff = theta_aff.cuda()

        # Select three corner points of bounding box
        src_pts[:, :, 0] = boxes_s[:, :2]
        src_pts[:, 0, 1] = boxes_s[:, 0]
        src_pts[:, 1, 1] = boxes_s[:, 3]
        src_pts[:, :, 2] = boxes_s[:, 2:]

        dst_pts[:, :, 0] = boxes_t[:, :2]
        dst_pts[:, 0, 1] = boxes_t[:, 0]
        dst_pts[:, 1, 1] = boxes_t[:, 3]
        dst_pts[:, :, 2] = boxes_t[:, 2:]

        src_pts_norm = PointsToUnitCoords(src_pts, source_im_size)
        dst_pts_norm = PointsToUnitCoords(dst_pts, target_im_size)

        src_pts_norm = src_pts_norm.permute(0, 2, 1)
        dst_pts_norm = dst_pts_norm.permute(0, 2, 1)

        # Compute affine parameters
        for i in range(batch_size):
            affine_mat = self.affine_theta(src_pts_norm[i], dst_pts_norm[i])
            theta_aff[i] = torch.Tensor(affine_mat)

        return theta_aff

    def affine_theta(self, src_pts, dst_pts):
        """

            Use the coordinates of three points to compute affine parameters (translation and scale)

        """
        # src_pts = np.array([[-1, -1], [1, -1], [1, 1]]).astype(np.float32)
        # dst_pts = np.array(mask_corners[0][0:3]).astype(np.float32)
        # affine_mat = cv2.getAffineTransform(src_pts, dst_pts)

        if self.use_cuda:
            src_pts = src_pts.cpu().numpy().astype(np.float32)
            dst_pts = dst_pts.cpu().numpy().astype(np.float32)
        else:
            src_pts = src_pts.numpy().astype(np.float32)
            dst_pts = dst_pts.numpy().astype(np.float32)

        affine_mat = cv2.getAffineTransform(dst_pts, src_pts)

        return affine_mat