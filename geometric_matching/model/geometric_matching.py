import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import os

from geometric_matching.model.feature_correlation import FeatureCorrelation
from geometric_matching.model.theta_regression import ThetaRegression
from geometric_matching.model.feature_extraction import FeatureExtraction
from geometric_matching.model.feature_extraction_2 import FeatureExtraction_2

from lib.model.roi_layers import ROIPool

# The entire architecture of the model
class GeometricMatching(nn.Module):
    """
    Predict geometric transformation parameters (TPS) between two images
    """
    def __init__(self, geometric_model='tps', feature_extraction_cnn='vgg',
                 feature_extraction_last_layer='', return_correlation=False, fr_feature_size=15, fr_kernel_sizes=[7, 5],
                 fr_channels=[128, 64], normalize_features=True, normalize_matches=True,
                 batch_normalization=True, train_fe=False, use_cuda=True, pretrained=True, crop_layer='image',
                 init_identity=True):

        super(GeometricMatching, self).__init__()
        self.use_cuda = use_cuda
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.return_correlation = return_correlation
        self.crop_layer = crop_layer
        # Feature extraction networks for two images

        self.FeatureExtraction = FeatureExtraction(train_fe=train_fe, feature_extraction_cnn=feature_extraction_cnn,
                                                   last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features, use_cuda=self.use_cuda,
                                                   pretrained=pretrained)

        if self.crop_layer == 'pool4':
            self.RoIPool = ROIPool((15, 15), 1.0 / 16.0)
        if self.crop_layer == 'conv1':
            self.RoIPool = ROIPool((240, 240), 1.0 / 1.0)
            self.FeatureExtraction_2 = FeatureExtraction_2(feature_extraction_cnn=feature_extraction_cnn,
                                                           first_layer='pool1', last_layer='pool4',
                                                           pretrained=pretrained)

        # Matching layer based on normalized feature maps of two images
        self.FeatureCorrelation = FeatureCorrelation(shape='3D', normalization=normalize_matches)

        for param in self.parameters():
            param.requires_grad = False

        if geometric_model == 'affine':
            output_dim = 6
        elif geometric_model == 'tps':
            output_dim = 18
        # Regression layer based on correlated feature map for predicting parameters of geometric transformation
        self.ThetaRegression = ThetaRegression(output_dim=output_dim, use_cuda=self.use_cuda,
                                               feature_size=fr_feature_size, kernel_sizes=fr_kernel_sizes,
                                               channels=fr_channels, batch_normalization=batch_normalization,
                                               init_identity=init_identity)

    def forward(self, batch):
        # do feature extraction
        # feature_A and feature_B.shape: (batch_size, channels, h, w)
        feature_A = self.FeatureExtraction(batch['source_image'])
        feature_B = self.FeatureExtraction(batch['target_image'])

        if self.crop_layer == 'pool4' or self.crop_layer == 'conv1':
            # Use the object bounding box to crop on the feature map (pool4 of vgg16),
            # and resize as (batch_size, 512, 15, 15)
            for i in range(batch['source_box'].shape[0]):
                if torch.sum(batch['source_box'][i]).item() > 0 and torch.sum(batch['target_box'][i]).item() > 0:
                    box_A = torch.Tensor(1, 5).zero_().cuda()
                    box_A[0, 1:] = batch['source_box'][i]
                    feature_A[i, :, :, :] = self.RoIPool(feature_A[i, :, :, :].unsqueeze(0), box_A)
    
                    box_B = torch.Tensor(1, 5).zero_().cuda()
                    box_B[0, 1:] = batch['target_box'][i]
                    feature_B[i, :, :, :] = self.RoIPool(feature_B[i, :, :, :].unsqueeze(0), box_B)
            if self.crop_layer == 'conv1':
                feature_A = self.FeatureExtraction_2(feature_A)
                feature_A = self.FeatureExtraction_2(feature_A)

        # do feature correlation
        # correlation.shape: (b, h * w, h, w), such as (225, 15, 15)
        correlation = self.FeatureCorrelation(feature_A, feature_B)

        # do regression to tnf parameters theta
        # theta.shape: (batch_size, 18) for tps, (batch_size, 6) for affine
        theta = self.ThetaRegression(correlation)

        if not self.return_correlation:
            return theta
        else:
            return theta, correlation