import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os

from geometric_matching.gm_model.feature_extraction_coseg import FeatureExtraction
from geometric_matching.gm_model.feature_correlation import FeatureCorrelation
from geometric_matching.gm_model.mask_regression import MaskRegression

# The entire architecture of the model
class CoSegmentation(nn.Module):
    """
    Predict co-object mask between two images
    """
    def __init__(self,
                 feature_extraction_cnn='resnet101',
                 feature_extraction_last_layer='',
                 fixed_blocks=3,
                 normalize_features=True,
                 normalize_matches=True,
                 pytorch=False,
                 caffe=False,
                 return_correlation=False):
        super(CoSegmentation, self).__init__()
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.return_correlation = return_correlation

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

        # Regression layer based on correlated feature map for predicting co-object mask
        self.MaskRegression = MaskRegression()


    def forward(self, batch):
        # Feature extraction
        features_A_1, features_A_2, features_A_3, features_A_4 = self.FeatureExtraction(batch['source_image'])
        features_B_1, features_B_2, features_B_3, features_B_4 = self.FeatureExtraction(batch['target_image'])

        # feature_A = features_A[-1]
        # feature_B = features_B[-1]

        # Feature correlation
        # correlation.shape: (b, h * w, h, w), such as (225, 15, 15)
        correlation_AB = self.FeatureCorrelation(feature_A=features_A_4, feature_B=features_B_4)
        correlation_BA = self.FeatureCorrelation(feature_A=features_B_4, feature_B=features_A_4)

        feature_A = torch.cat((features_A_4, correlation_AB), 1)
        feature_B = torch.cat((features_B_4, correlation_BA), 1)

        mask_A = self.MaskRegression(feature_A, features_A_3, features_A_2, features_A_1)
        mask_B = self.MaskRegression(feature_B, features_B_3, features_B_2, features_B_1)

        return mask_A, mask_B