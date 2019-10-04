import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os

from geometric_matching.gm_model.feature_correlation import FeatureCorrelation
from geometric_matching.gm_model.theta_regression2 import ThetaRegression
from geometric_matching.gm_model.feature_extraction import FeatureExtraction
# from geometric_matching.model.feature_extraction_2 import FeatureExtraction_2
from geometric_matching.gm_model.object_select import ObjectSelect
from geometric_matching.gm_model.feature_crop import FeatureCrop
from geometric_matching.geotnf.affine_theta import AffineTheta
from geometric_matching.geotnf.transformation import GeometricTnf

# from lib.model.faster_rcnn.vgg16 import vgg16

# The entire architecture of the model
class GeometricMatching(nn.Module):
    """
    Predict geometric transformation parameters (TPS) between two images
    """
    def __init__(self,
                 output_dim=6,
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
        super(GeometricMatching, self).__init__()
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.return_correlation = return_correlation
        # self.return_correlation = True

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
        self.ThetaRegression = ThetaRegression(output_dim=output_dim,
                                               feature_size=fr_feature_size,
                                               kernel_sizes=fr_kernel_sizes,
                                               channels=fr_channels,
                                               batch_normalization=batch_normalization)

    def forward(self, batch):
        if self.ThetaRegression.training:
            # Feature extraction
            # feature_A and feature_B.shape: (batch_size, channels, h, w)
            feature_A = self.FeatureExtraction(batch['source_image'])
            feature_B = self.FeatureExtraction(batch['target_image'])
            # feature_A_1, feature_A_2 = self.FeatureExtraction(batch['source_image'])
            # feature_B_1, feature_B_2 = self.FeatureExtraction(batch['target_image'])

            # feature_A_1 = F.interpolate(feature_A_1, size=(15, 15), mode='bilinear', align_corners=True)
            # feature_B_1 = F.interpolate(feature_B_1, size=(15, 15), mode='bilinear', align_corners=True)

            # feature_A = torch.cat((feature_A_2, feature_A_1), 1)
            # feature_B = torch.cat((feature_B_2, feature_B_1), 1)

            # Feature correlation
            # correlation.shape: (b, h * w, h, w), such as (225, 15, 15)
            correlation_AB = self.FeatureCorrelation(feature_A=feature_A, feature_B=feature_B)
            correlation_BA = self.FeatureCorrelation(feature_A=feature_B, feature_B=feature_A)

            # Theta (transformation parameters) regression
            # theta.shape: (batch_size, 18) for tps, (batch_size, 6) for affine
            theta_AB = self.ThetaRegression(correlation_AB)
            theta_BA = self.ThetaRegression(correlation_BA)

            # if not self.return_correlation:
            return theta_AB, theta_BA
        # else:
        #     return theta, feature_A, feature_B, correlation
        else:
            feature_A = self.FeatureExtraction(batch['source_image'])
            feature_B = self.FeatureExtraction(batch['target_image'])

            correlation = self.FeatureCorrelation(feature_A=feature_A, feature_B=feature_B)

            theta = self.ThetaRegression(correlation)

            return theta