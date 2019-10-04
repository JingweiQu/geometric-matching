import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os

from geometric_matching.gm_model.feature_correlation import FeatureCorrelation
from geometric_matching.gm_model.theta_regression2 import ThetaRegression
from geometric_matching.gm_model.feature_extraction import FeatureExtraction
from geometric_matching.geotnf.transformation import GeometricTnf

# The entire architecture of the model
class GeometricMatching(nn.Module):
    """
    Predict geometric transformation parameters between two images
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
            feature_R = self.FeatureExtraction(batch['refer_image'])

            # Feature correlation
            # correlation.shape: (b, h * w, h, w), such as (225, 15, 15)
            correlation_AB = self.FeatureCorrelation(feature_A, feature_B)
            correlation_BA = self.FeatureCorrelation(feature_B, feature_A)
            correlation_BR = self.FeatureCorrelation(feature_B, feature_R)
            correlation_RB = self.FeatureCorrelation(feature_R, feature_B)

            # Theta (transformation parameters) regression
            # theta.shape: (batch_size, 18) for tps, (batch_size, 6) for affine
            theta_AB = self.ThetaRegression(correlation_AB)
            theta_BA = self.ThetaRegression(correlation_BA)
            theta_BR = self.ThetaRegression(correlation_BR)
            theta_RB = self.ThetaRegression(correlation_RB)

            return theta_AB, theta_BA, theta_BR, theta_RB
        else:
            feature_A = self.FeatureExtraction(batch['source_image'])
            feature_B = self.FeatureExtraction(batch['target_image'])

            correlation = self.FeatureCorrelation(feature_A, feature_B)

            theta = self.ThetaRegression(correlation)

            return theta