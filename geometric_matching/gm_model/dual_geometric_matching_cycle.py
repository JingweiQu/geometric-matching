import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os

from geometric_matching.gm_model.feature_correlation import FeatureCorrelation
from geometric_matching.gm_model.theta_regression2 import ThetaRegression
from geometric_matching.gm_model.feature_extraction import FeatureExtraction
from geometric_matching.geotnf.transformation_tps import GeometricTnf

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
                 tps_output_dim=36,
                 use_cuda=True,
                 feature_extraction_cnn='resnet101',
                 feature_extraction_last_layer='',
                 fr_feature_size=15,
                 fr_kernel_sizes=[1, 7, 5, 3],
                 fr_channels=[256, 128, 64, 32],
                 fixed_blocks=3,
                 normalize_features=True,
                 normalize_matches=True,
                 batch_normalization=True,
                 pytorch=False,
                 caffe=False,
                 return_correlation=False):
        super(DualGeometricMatching, self).__init__()
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.return_correlation = return_correlation
        self.AffTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)

        # Feature extraction networks for two images
        self.FeatureExtraction = FeatureExtraction(feature_extraction_cnn=feature_extraction_cnn,
                                                   last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features,
                                                   fixed_blocks=fixed_blocks,
                                                   pytorch=pytorch,
                                                   caffe=caffe)

        # Matching layer based on normalized feature maps of two images
        self.FeatureCorrelation = FeatureCorrelation(shape='3D', normalization=normalize_matches)

        # Regression layer based on correlated feature map for predicting parameters of geometric transformation
        self.ThetaRegression = ThetaRegression(output_dim=aff_output_dim,
                                               feature_size=fr_feature_size,
                                               kernel_sizes=fr_kernel_sizes,
                                               channels=fr_channels,
                                               batch_normalization=batch_normalization)

        self.ThetaRegression2 = ThetaRegression(output_dim=tps_output_dim,
                                                feature_size=fr_feature_size,
                                                kernel_sizes=fr_kernel_sizes,
                                                channels=fr_channels,
                                                batch_normalization=batch_normalization)

    def forward(self, batch):
        if self.training:
            # ===  STAGE 1 === #
            # feature_A and feature_B.shape: (batch_size, channels, h, w)
            feature_A = self.FeatureExtraction(batch['source_image'])
            feature_B = self.FeatureExtraction(batch['target_image'])

            correlation_1_AB = self.FeatureCorrelation(feature_A=feature_A, feature_B=feature_B)
            correlation_1_BA = self.FeatureCorrelation(feature_A=feature_B, feature_B=feature_A)

            # Affine
            theta_1_AB = self.ThetaRegression(correlation_1_AB)
            theta_1_BA = self.ThetaRegression(correlation_1_BA)

            # ===  STAGE 2 === #
            # Feature extraction
            # feature_A and feature_B.shape: (batch_size, channels, h, w)
            source_image_warp = self.AffTnf(image_batch=batch['source_image'], theta_batch=theta_1_AB)
            target_image_warp = self.AffTnf(image_batch=batch['target_image'], theta_batch=theta_1_BA)

            feature_A_warp = self.FeatureExtraction(source_image_warp)
            feature_B_warp = self.FeatureExtraction(target_image_warp)

            # Feature correlation
            # correlation.shape: (b, h * w, h, w), such as (225, 15, 15)
            correlation_2_AB = self.FeatureCorrelation(feature_A_warp, feature_B)
            correlation_2_BA = self.FeatureCorrelation(feature_B_warp, feature_A)

            # Theta (transformation parameters) regression
            # theta_2.shape: (batch_size, 18 or 36) for tps
            theta_2_AB = self.ThetaRegression2(correlation_2_AB)
            theta_2_BA = self.ThetaRegression2(correlation_2_BA)

            if not self.return_correlation:
                return theta_2_AB, theta_1_AB, theta_2_BA, theta_1_BA
            else:
                return theta_2_AB, theta_1_AB, theta_2_BA, theta_1_BA, correlation_1_AB, correlation_1_BA, correlation_2_AB, correlation_2_BA
        else:
            # ===  STAGE 1 === #
            feature_A = self.FeatureExtraction(batch['source_image'])
            feature_B = self.FeatureExtraction(batch['target_image'])

            correlation_1 = self.FeatureCorrelation(feature_A=feature_A, feature_B=feature_B)

            theta_1 = self.ThetaRegression(correlation_1)

            # ===  STAGE 2 === #
            source_image_warp = self.AffTnf(image_batch=batch['source_image'], theta_batch=theta_1)
            feature_A_warp = self.FeatureExtraction(source_image_warp)

            correlation_2 = self.FeatureCorrelation(feature_A=feature_A_warp, feature_B=feature_B)

            theta_2 = self.ThetaRegression2(correlation_2)

            return theta_2, theta_1