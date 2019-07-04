from __future__ import print_function, division
import torch
import torch.nn as nn
from geometric_matching.model.featureL2norm import FeatureL2Norm

class FeatureCorrelation(nn.Module):
    """
    Correlation matching: per-pixel between per-pixel in two feature maps
    """
    def __init__(self, shape='3D', normalization=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.shape = shape
        self.ReLU = nn.ReLU()

    def forward(self, feature_A, feature_B):
        # feature_A and feature_B.shape: (batch_size, channels, h, w), such as (15, 15)
        b, c, h, w = feature_A.size()
        if self.shape == '3D':
            # reshape features for matrix multiplication
            # feature_A.shape becomes (b, c, h * w)
            # feature_B.shape becomes (b, h * w, c)
            feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
            feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
            # perform matrix mult.
            # bmm: batch matrix-matrix product, b * (h * w) * (h * w)
            feature_mul = torch.bmm(feature_B, feature_A)
            # correlation_tensor.shape: (b, h * w, h, w), such as (225, 15, 15)
            # correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
            correlation_tensor = feature_mul.view(b, h, w, h * w).permute(0, 3, 1, 2)
        elif self.shape == '4D':
            # reshape features for matrix multiplication
            feature_A = feature_A.view(b, c, h * w).transpose(1, 2)  # size [b,c,h*w]
            feature_B = feature_B.view(b, c, h * w)  # size [b,c,h*w]
            # perform matrix mult.
            feature_mul = torch.bmm(feature_A, feature_B)
            # indexed [batch,row_A,col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b, h, w, h, w).unsqueeze(1)

        # Normalize with ReLU
        if self.normalization:
            correlation_tensor = FeatureL2Norm(self.ReLU(correlation_tensor))

        return correlation_tensor