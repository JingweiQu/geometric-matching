import torch
from torch.autograd import Variable
import numpy as np
from geometric_matching.geotnf.transformation_tps import TpsGridGen

def normalize_axis(x, L):
    return (x - 1 - (L - 1) / 2) * 2 / (L - 1)

def unnormalize_axis(x, L):
    return x * (L - 1) / 2 + 1 + (L - 1) / 2

class PointTPS(object):
    """
    Class with functions for transforming a set of points with affine/tps transformations
    """

    def __init__(self, tps_grid_size=3, tps_reg_factor=0, use_cuda=True):
        self.use_cuda = use_cuda
        self.tpsTnf = TpsGridGen(grid_size=tps_grid_size, reg_factor=tps_reg_factor, use_cuda=self.use_cuda)

    def tpsPointTnf(self, theta, points):
        # points are expected in [B,2,N], where first row is X and second row is Y
        # reshape points for applying Tps transformation
        # points.shape: from (batch_size, 2, N) to (batch_size, 1, N, 2), N points with (x, y)
        points = points.unsqueeze(3).transpose(1, 3)
        # apply transformation
        warped_points = self.tpsTnf.apply_transformation(theta, points)
        # undo reshaping, warped_points.shape: from (batch_size, 1, N, 2) to (batch_size, 2, N)
        warped_points = warped_points.transpose(3, 1).squeeze(3)
        return warped_points

    def affPointTnf(self, theta, points):
        # theta_mat.shape: (batch_size, 2, 3)
        # points.shape: (batch_size, 2, N)
        theta_mat = theta.view(-1, 2, 3)
        # Scaling, rotate and shear
        warped_points = torch.bmm(theta_mat[:, :, :2], points)
        # Translate
        warped_points += theta_mat[:, :, 2].unsqueeze(2).expand_as(warped_points)
        # warped_points.shape: (batch_size, 2, N)
        return warped_points

def PointsToUnitCoords(P, im_size):
    """
    Compute normalized coordinates of points with given image size
    :param P: coordinates of points in image
    :param im_size: image size
    :return: normalized coordinate of points in range (-1, 1) * (-1, 1)
    """
    h, w = im_size[:, 0], im_size[:, 1]
    P_norm = P.clone()
    # normalize Y, horizontal axis is Y
    P_norm[:, 0, :] = normalize_axis(P[:, 0, :], w.unsqueeze(1).expand_as(P[:, 0, :]))
    # normalize X, vertical axis is X
    P_norm[:, 1, :] = normalize_axis(P[:, 1, :], h.unsqueeze(1).expand_as(P[:, 1, :]))
    return P_norm

def PointsToPixelCoords(P, im_size):
    """
    Compute coordinates of points in image with normalized coordinates
    :param P: normalized coordinates of points in range (-1, 1) * (-1, 1)
    :param im_size: image size
    :return: coordinates of points in image
    """
    h, w = im_size[:, 0], im_size[:, 1]
    P_norm = P.clone()
    # normalize Y
    P_norm[:, 0, :] = unnormalize_axis(P[:, 0, :], w.unsqueeze(1).expand_as(P[:, 0, :]))
    # normalize X
    P_norm[:, 1, :] = unnormalize_axis(P[:, 1, :], h.unsqueeze(1).expand_as(P[:, 1, :]))
    return P_norm