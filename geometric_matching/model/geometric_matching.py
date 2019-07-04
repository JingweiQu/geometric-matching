import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os

from geometric_matching.model.feature_correlation import FeatureCorrelation
from geometric_matching.model.theta_regression import ThetaRegression
from geometric_matching.model.feature_extraction import FeatureExtraction
# from geometric_matching.model.feature_extraction_2 import FeatureExtraction_2
from geometric_matching.model.object_select import ObjectSelect
from geometric_matching.model.feature_crop import FeatureCrop
from geometric_matching.geotnf.affine_theta import AffineTheta
from geometric_matching.geotnf.transformation import GeometricTnf

from lib.model.faster_rcnn.vgg16 import vgg16

# The entire architecture of the model
class GeometricMatching(nn.Module):
    """
    Predict geometric transformation parameters (TPS) between two images
    """
    def __init__(self, output_dim=6, feature_extraction_cnn='vgg', feature_extraction_last_layer='',
                 return_correlation=False, fr_feature_size=15, fr_kernel_sizes=[7, 5], fr_channels=[128, 64],
                 normalize_features=True, normalize_matches=True, batch_normalization=True, train_fe=False,
                 use_cuda=True, pretrained=True):
        super(GeometricMatching, self).__init__()
        self.use_cuda = use_cuda
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.return_correlation = return_correlation

        # Feature extraction networks for two images
        self.FeatureExtraction = FeatureExtraction(train_fe=train_fe, feature_extraction_cnn=feature_extraction_cnn,
                                                   last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features, use_cuda=self.use_cuda,
                                                   pretrained=pretrained)

        # Matching layer based on normalized feature maps of two images
        self.FeatureCorrelation = FeatureCorrelation(shape='3D', normalization=normalize_matches)

        for param in self.parameters():
            param.requires_grad = False

        # Regression layer based on correlated feature map for predicting parameters of geometric transformation
        self.ThetaRegression = ThetaRegression(output_dim=output_dim, use_cuda=self.use_cuda,
                                               feature_size=fr_feature_size, kernel_sizes=fr_kernel_sizes,
                                               channels=fr_channels, batch_normalization=batch_normalization)

    def forward(self, batch):
        # Feature extraction
        # feature_A and feature_B.shape: (batch_size, channels, h, w)
        feature_A = self.FeatureExtraction(batch['source_image'])
        feature_B = self.FeatureExtraction(batch['target_image'])

        # Feature correlation
        # correlation.shape: (b, h * w, h, w), such as (225, 15, 15)
        correlation = self.FeatureCorrelation(feature_A=feature_A, feature_B=feature_B)

        # Theta (transformation parameters) regression
        # theta.shape: (batch_size, 18) for tps, (batch_size, 6) for affine
        theta = self.ThetaRegression(correlation)

        if not self.return_correlation:
            return theta
        else:
            return theta, correlation