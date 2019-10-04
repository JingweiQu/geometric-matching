import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os

from geometric_matching.gm_model.feature_correlation import FeatureCorrelation
from geometric_matching.gm_model.theta_regression import ThetaRegression
from geometric_matching.gm_model.feature_extraction import FeatureExtraction
# from geometric_matching.model.feature_extraction_2 import FeatureExtraction_2
from geometric_matching.geotnf.transformation import GeometricTnf

# The entire architecture of the model
class DualGeometricMatching(nn.Module):
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
                 use_cuda=True,
                 feature_extraction_cnn='resnet101',
                 feature_extraction_last_layer='',
                 fr_feature_size=15,
                 fr_kernel_sizes=[7, 5],
                 fr_channels=[128, 64],
                 fixed_blocks=3,
                 normalize_features=True,
                 normalize_matches=True,
                 batch_normalization=True,
                 crop_layer=None,
                 pytorch=False,
                 caffe=False,
                 return_correlation=False):
        super(DualGeometricMatching, self).__init__()
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.return_correlation = return_correlation
        # self.return_correlation = True
        self.AffTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)
        self.crop_layer = crop_layer

        # Feature extraction networks for two images
        self.FeatureExtraction = FeatureExtraction(feature_extraction_cnn=feature_extraction_cnn,
                                                   last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features,
                                                   fixed_blocks=fixed_blocks,
                                                   pytorch=pytorch,
                                                   caffe=caffe)

        # Matching layer based on normalized feature maps of two images
        self.FeatureCorrelation = FeatureCorrelation(shape='3D', normalization=normalize_matches)

        for param in self.parameters():
            param.requires_grad = False

        # Regression layer based on correlated feature map for predicting parameters of geometric transformation
        self.ThetaRegression = ThetaRegression(output_dim=aff_output_dim,
                                               feature_size=fr_feature_size,
                                               kernel_sizes=fr_kernel_sizes,
                                               channels=fr_channels,
                                               batch_normalization=batch_normalization)

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
        # feature_A and feature_B.shape: (batch_size, channels, h, w)
        feature_A = self.FeatureExtraction(batch['source_image'])
        feature_B = self.FeatureExtraction(batch['target_image'])

        correlation_aff = self.FeatureCorrelation(feature_A=feature_A, feature_B=feature_B)

        # Affine
        theta_aff = self.ThetaRegression(correlation_aff)

        # ===  STAGE 2 === #
        # Feature extraction
        # feature_A and feature_B.shape: (batch_size, channels, h, w)
        source_image_warp = self.AffTnf(image_batch=batch['source_image'], theta_batch=theta_aff)
        feature_A_warp = self.FeatureExtraction(source_image_warp)
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
        correlation_afftps = self.FeatureCorrelation(feature_A=feature_A_warp, feature_B=feature_B)

        # Theta (transformation parameters) regression
        # theta_afftps.shape: (batch_size, 18) for tps
        theta_afftps = self.ThetaRegression2(correlation_afftps)

        if not self.return_correlation:
            return theta_afftps, theta_aff,
        else:
            return theta_afftps, theta_aff, correlation_afftps, correlation_aff