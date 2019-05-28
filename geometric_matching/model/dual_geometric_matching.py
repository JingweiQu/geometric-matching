import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import os

from geometric_matching.model.geometric_matching import GeometricMatching
from geometric_matching.model.feature_correlation import FeatureCorrelation
from geometric_matching.model.theta_regression import ThetaRegression
from geometric_matching.model.feature_extraction import FeatureExtraction
from geometric_matching.model.feature_extraction_2 import FeatureExtraction_2
from geometric_matching.geotnf.transformation import GeometricTnf
from geometric_matching.geotnf.affine_theta import AffineTheta

from lib.model.roi_layers import ROIPool

class DualStageGeometricMatching(GeometricMatching):
    """
    Fine-tune on pre-trained GeometricMatching model (TPS) with warped image pair:
    source image: source image is warped by affine parameters computed by coordinates of object bounding box
    predicted by faster-rcnn
    target image: target image
    """
    def __init__(self, geometric_model='tps', fr_feature_size=15, fr_kernel_sizes=[7, 5], fr_channels=[128, 64],
                 feature_extraction_cnn='vgg', feature_extraction_last_layer='', return_correlation=False,
                 normalize_features=True, normalize_matches=True, batch_normalization=True, train_fe=False,
                 use_cuda=True, pretrained=True, crop_layer='image', init_identity=True):
        super(DualStageGeometricMatching, self).__init__(geometric_model=geometric_model,
                                                         fr_feature_size=fr_feature_size,
                                                         fr_kernel_sizes=fr_kernel_sizes, fr_channels=fr_channels,
                                                         feature_extraction_cnn=feature_extraction_cnn,
                                                         feature_extraction_last_layer=feature_extraction_last_layer,
                                                         return_correlation=return_correlation,
                                                         normalize_features=normalize_features,
                                                         normalize_matches=normalize_matches,
                                                         batch_normalization=batch_normalization, train_fe=train_fe,
                                                         use_cuda=use_cuda, pretrained=pretrained,
                                                         crop_layer=crop_layer, init_identity=init_identity)

        self.affine_theta = AffineTheta(use_cuda=use_cuda, original=False, image_size=240)
        self.geoTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)

    def forward(self, batch):
        # ===  STAGE 1 ===#
        # Warp source image with affine parameters
        theta_aff = self.affine_theta(boxes_s=batch['source_box'], boxes_t=batch['target_box'],
                                 source_im_size=None, target_im_size=None)
        batch['source_image'] = self.geoTnf(batch['source_image'], theta_aff)

        # ===  STAGE 2 ===#
        # feature extraction
        feature_src = self.FeatureExtraction(batch['source_image'])
        feature_tgt = self.FeatureExtraction(batch['target_image'])
        # feature correlation
        correlation = self.FeatureCorrelation(feature_src, feature_tgt)
        # regression to tnf parameters theta
        theta = self.FeatureRegression(correlation)

        if not self.return_correlation:
            return theta
        else:
            return theta, correlation
