import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os

from geometric_matching.gm_model.geometric_matching import GeometricMatching
from geometric_matching.gm_model.feature_correlation import FeatureCorrelation
from geometric_matching.gm_model.theta_regression import ThetaRegression
from geometric_matching.gm_model.feature_extraction import FeatureExtraction
# from geometric_matching.model.feature_extraction_2 import FeatureExtraction_2
from geometric_matching.gm_model.object_select import ObjectSelect
from geometric_matching.gm_model.feature_crop import FeatureCrop
from geometric_matching.geotnf.affine_theta import AffineTheta
from geometric_matching.geotnf.transformation import GeometricTnf

from lib.model.faster_rcnn.vgg16 import vgg16
from lib.model.faster_rcnn.resnet import resnet

# The entire architecture of the model
class DualGeometricMatching(GeometricMatching):
    """
    Predict geometric transformation parameters (TPS) between two images

    (1) Directly train with warped image pair; or
    (2) Fine-tune on pre-trained GeometricMatching model (TPS) with warped image pair

    image pair: source image: source image is warped by affine parameters computed by coordinates of object bounding box
    predicted by faster-rcnn
    target image: target image
    """
    def __init__(self,
                 aff_output_dim=6,
                 tps_output_dim=18,
                 crop_layer=None,
                 return_box=False,
                 use_cuda=True,
                 feature_extraction_cnn='resnet101',
                 feature_extraction_last_layer='',
                 fr_feature_size=15,
                 fr_kernel_sizes=[7, 5],
                 fr_channels=[128, 64],
                 normalize_features=True,
                 normalize_matches=True,
                 batch_normalization=True,
                 pytorch=False,
                 caffe=False,
                 return_correlation=False):
        super().__init__(output_dim=aff_output_dim,
                         feature_extraction_cnn=feature_extraction_cnn,
                         feature_extraction_last_layer=feature_extraction_last_layer,
                         fr_feature_size=fr_feature_size,
                         fr_kernel_sizes=fr_kernel_sizes,
                         fr_channels=fr_channels,
                         normalize_features=normalize_features,
                         normalize_matches=normalize_matches,
                         batch_normalization=batch_normalization,
                         pytorch=pytorch,
                         caffe=caffe,
                         return_correlation=return_correlation)
        self.AffTheta = AffineTheta(use_cuda=use_cuda)
        self.AffTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)
        self.crop_layer = crop_layer

        # Feature cropping on specific layer
        # self.FeatureCrop = FeatureCrop(crop_layer=crop_layer)
        # if self.crop_layer == 'conv1':
        #     self.FeatureExtraction_2 = FeatureExtraction_2(feature_extraction_cnn=feature_extraction_cnn,
        #                                                    first_layer='pool1', last_layer='pool4',
        #                                                    pretrained=pretrained)
        self.ThetaRegression2 = ThetaRegression(output_dim=tps_output_dim,
                                                feature_size=fr_feature_size,
                                                kernel_sizes=fr_kernel_sizes,
                                                channels=fr_channels,
                                                batch_normalization=batch_normalization)

    def forward(self, batch):
        # ===  STAGE 1 === #
        # Object detection
        box_info_A = self.FasterRCNN(im_data=batch['source_im'], im_info=batch['source_im_size'], gt_boxes=batch['source_gt_boxes'], num_boxes=batch['source_num_boxes'])[0:3]
        box_info_B = self.FasterRCNN(im_data=batch['target_im'], im_info=batch['target_im_size'], gt_boxes=batch['target_gt_boxes'], num_boxes=batch['target_num_boxes'])[0:3]
        box_A, box_B = self.ObjectSelect(box_info_A=box_info_A, im_info_A=batch['source_im_size'], box_info_B=box_info_B, im_info_B=batch['target_im_size'])

        # Warp source image with affine parameters
        theta_aff_det = self.AffTheta(boxes_s=box_A, boxes_t=box_B)
        source_image_warp_1 = self.AffTnf(image_batch=batch['source_image'], theta_batch=theta_aff_det)

        feature_A_warp_1 = self.FeatureExtraction(source_image_warp_1)
        feature_B = self.FeatureExtraction(batch['target_image'])

        correlation_aff = self.FeatureCorrelation(feature_A_warp_1, feature_B)

        theta_aff = self.ThetaRegression(correlation_aff)

        # ===  STAGE 2 === #
        # Feature extraction
        # feature_A and feature_B.shape: (batch_size, channels, h, w)
        source_image_warp_2 = self.AffTnf(image_batch=source_image_warp_1, theta_batch=theta_aff)
        feature_A_warp_2 = self.FeatureExtraction(source_image_warp_2)
        # feature_A = self.FeatureExtraction(batch['source_image'])
        # feature_B = self.FeatureExtraction(batch['target_image'])

        # Crop feature map on specific layer
        # feature_A = self.FeatureCrop(feature_A, box_A)
        # feature_B = self.FeatureCrop(feature_B, box_B)

        # if self.crop_layer == 'conv1':
        #     feature_A = self.FeatureExtraction_2(feature_A)
        #     feature_B = self.FeatureExtraction_2(feature_B)

        # Feature correlation
        # correlation.shape: (b, h * w, h, w), such as (225, 15, 15)
        correlation_tps = self.FeatureCorrelation(feature_A=feature_A_warp_2, feature_B=feature_B)

        # Theta (transformation parameters) regression
        # theta_aff_tps.shape: (batch_size, 18) for tps
        theta_aff_tps = self.ThetaRegression2(correlation_tps)

        if not self.return_correlation:
            if not self.return_box:
                return theta_aff_tps, theta_aff, theta_aff_det
            else:
                return theta_aff_tps, theta_aff, theta_aff_det, box_A, box_B
        else:
            return theta_aff_tps, theta_aff, theta_aff_det, correlation_tps, correlation_aff