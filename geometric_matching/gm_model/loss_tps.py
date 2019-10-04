from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
from geometric_matching.geotnf.point_tps import PointTPS
from geometric_matching.geotnf.point_tnf import PointTnf

class TransformedGridLoss(nn.Module):
    """
    Compute loss by computing distances between
    (1) grid points transformed by ground-truth theta
    (2) grid points transformed by predicted theta_tr and theta_st
    """
    def __init__(self, geometric_model='tps', use_cuda=True, grid_size=20):
        super(TransformedGridLoss, self).__init__()
        self.geometric_model = geometric_model
        # define virtual grid of points to be transformed
        axis_coords = np.linspace(-1, 1, grid_size)
        self.N = grid_size * grid_size
        # X and Y.shape: (20, 20)
        X, Y = np.meshgrid(axis_coords, axis_coords)
        # X and Y.shape: (1, 1, 400), P.shape: (1, 2, 400)
        X = np.reshape(X, (1, 1, self.N))
        Y = np.reshape(Y, (1, 1, self.N))
        P = np.concatenate((X, Y), 1)
        # self.P = Variable(torch.FloatTensor(P), requires_grad=False)
        self.P = torch.Tensor(P.astype(np.float32))
        self.P.requires_grad = False
        self.pointTPS = PointTPS(use_cuda=use_cuda)
        self.pointTnf = PointTnf(use_cuda=use_cuda)
        if use_cuda:
            self.P = self.P.cuda()

    def forward(self, theta_st, theta_tr, theta_GT):
        # expand grid according to batch size
        # theta.shape: (batch_size, 18) for tps, (batch_size, 6) for affine
        # theta_GT.shape: (batch_size, 18, 1, 1) for tps, (batch_size, 6) for affine
        batch_size = theta_st.size()[0]
        # P.shape: (batch_size, 2, 400)
        P = self.P.expand(batch_size, 2, self.N)
        # compute transformed grid points using estimated and GT tnfs
        # P_prime and P_prime_GT.shape: (batch_size, 2, 400)
        P_prime = self.pointTPS.tpsPointTnf(theta_tr.unsqueeze(2).unsqueeze(3), P)
        P_prime = self.pointTPS.tpsPointTnf(theta_st.unsqueeze(2).unsqueeze(3), P_prime)
        P_prime_GT = self.pointTnf.tpsPointTnf(theta_GT, P)
        # compute MSE loss on transformed grid points
        loss = torch.sum(torch.pow(P_prime - P_prime_GT, 2), 1)
        loss = torch.mean(loss)
        return loss

class CycleLoss(nn.Module):
    def __init__(self, geometric_model='affine', use_cuda=True, grid_size=20):
        super(CycleLoss, self).__init__()
        self.geometric_model = geometric_model
        # define virtual grid of points to be transformed
        axis_coords = np.linspace(-1, 1, grid_size)
        self.N = grid_size * grid_size
        # X and Y.shape: (20, 20)
        X, Y = np.meshgrid(axis_coords, axis_coords)
        # X and Y.shape: (1, 1, 400), P.shape: (1, 2, 400)
        X = np.reshape(X, (1, 1, self.N))
        Y = np.reshape(Y, (1, 1, self.N))
        P = np.concatenate((X, Y), 1)
        # self.P = Variable(torch.FloatTensor(P), requires_grad=False)
        self.P = torch.Tensor(P.astype(np.float32))
        self.P.requires_grad = False
        self.pointTnf = PointTPS(use_cuda=use_cuda)
        if use_cuda:
            self.P = self.P.cuda()

    def forward(self, theta_AB, theta_BA):
        # expand grid according to batch size
        # theta.shape: (batch_size, 18) for tps, (batch_size, 6) for affine
        # theta_GT.shape: (batch_size, 18, 1, 1) for tps, (batch_size, 6) for affine
        batch_size = theta_AB.size()[0]
        # P.shape: (batch_size, 2, 400)
        P = self.P.expand(batch_size, 2, self.N)
        # compute transformed grid points using estimated and GT tnfs
        # P_prime and P_prime_GT.shape: (batch_size, 2, 400)
        P_prime = self.pointTnf.tpsPointTnf(theta_AB.unsqueeze(2).unsqueeze(3), P)
        P_prime = self.pointTnf.tpsPointTnf(theta_BA.unsqueeze(2).unsqueeze(3), P_prime)
        # compute MSE loss on transformed grid points
        loss = torch.sum(torch.pow(P_prime - P, 2), 1)
        loss = torch.mean(loss)
        return loss

# class JitterLoss(nn.Module):
#     """
#     Compute loss by computing distances between
#     (1) grid points transformed by ground-truth theta
#     (2) grid points transformed by predicted theta_tr and theta_st
#     """
#     def __init__(self, use_cuda=True, grid_size=3):
#         super(JitterLoss, self).__init__()
#         # axis_coords = np.linspace(-1, 1, grid_size)
#         self.N = grid_size * grid_size
#         # Grid scale is (-1, -1, 1, 1), a square with (x_min, y_min, x_max, y_max), the points is 3 * 3
#         # P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
#         # P_X.shape and P_Y.shape: (1, 9)
#         # P_X = np.reshape(P_X, (1, -1))  # size (1,N)
#         # P_Y = np.reshape(P_Y, (1, -1))  # size (1,N)
#         # self.P_X = torch.Tensor(P_X.astype(np.float32))
#         # self.P_Y = torch.Tensor(P_Y.astype(np.float32))
#
#         self.P_X = torch.FloatTensor([-0.8, -0.8, -0.8, 0, 0, 0, 0.8, 0.8, 0.8]).view(1, -1)
#         self.P_Y = torch.FloatTensor([-0.8, 0, 0.8, -0.8, 0, 0.8, -0.8, 0, 0.8]).view(1, -1)
#         if use_cuda:
#             self.P_X = self.P_X.cuda()
#             self.P_Y = self.P_Y.cuda()
#
#     def forward(self, theta_st, theta_tr):
#         # expand grid according to batch size
#         # theta.shape: (batch_size, 36) for tps
#         batch_size = theta_st.size()[0]
#         self.P_X = self.P_X.expand(batch_size, self.P_X.size()[1])
#         self.P_Y = self.P_Y.expand(batch_size, self.P_Y.size()[1])
#
#         Q_X_st = theta_st[:, 2 * self.N:3 * self.N]
#         Q_Y_st = theta_st[:, 3 * self.N:]
#         Q_X_tr = theta_tr[:, 2 * self.N:3 * self.N]
#         Q_Y_tr = theta_tr[:, 3 * self.N:]
#
#         # dist_squared_st = torch.pow(torch.pow(Q_X_st - self.P_X, 2) + torch.pow(Q_Y_st - self.P_Y, 2), 2)
#         # dist_squared_tr = torch.pow(torch.pow(Q_X_tr - self.P_X, 2) + torch.pow(Q_Y_tr - self.P_Y, 2), 2)
#
#         dist_squared_st = torch.pow(Q_X_st - self.P_X, 2) + torch.pow(Q_Y_st - self.P_Y, 2)
#         dist_squared_tr = torch.pow(Q_X_tr - self.P_X, 2) + torch.pow(Q_Y_tr - self.P_Y, 2)
#
#         dist_st_mean = torch.mean(dist_squared_st)
#         dist_tr_mean = torch.mean(dist_squared_tr)
#
#         loss = (dist_st_mean + dist_tr_mean) / 2
#         return loss

class JitterLoss(nn.Module):
    """
    Compute loss by computing distances between
    (1) grid points transformed by ground-truth theta
    (2) grid points transformed by predicted theta_tr and theta_st
    """
    def __init__(self, use_cuda=True, grid_size=20):
        super(JitterLoss, self).__init__()
        # define virtual grid of points to be transformed
        axis_coords = np.linspace(-1, 1, grid_size)
        self.N = grid_size * grid_size
        # X and Y.shape: (20, 20)
        X, Y = np.meshgrid(axis_coords, axis_coords)
        # X and Y.shape: (1, 1, 400), P.shape: (1, 2, 400)
        X = np.reshape(X, (1, 1, self.N))
        Y = np.reshape(Y, (1, 1, self.N))
        P = np.concatenate((X, Y), 1)
        # self.P = Variable(torch.FloatTensor(P), requires_grad=False)
        self.P = torch.Tensor(P.astype(np.float32))
        self.P.requires_grad = False
        self.pointTnf = PointTnf(use_cuda=use_cuda)
        # self.theta_norm = torch.FloatTensor([-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1]).view(1, -1)
        if use_cuda:
            self.P = self.P.cuda()
            # self.theta_norm = self.theta_norm.cuda()

    def forward(self, theta_st, theta_tr):
        # expand grid according to batch size
        # theta_st.shape & theta_tr.shape: (batch_size, 36) for tps
        batch_size = theta_st.size()[0]
        # P.shape: (batch_size, 2, 400)
        P = self.P.expand(batch_size, 2, self.N)
        # theta_norm = self.theta_norm.expand(batch_size, 18)

        P_prime_st = self.pointTnf.tpsPointTnf(theta_st[:, 18:].unsqueeze(2).unsqueeze(3), P)
        P_prime_tr = self.pointTnf.tpsPointTnf(theta_tr[:, 18:].unsqueeze(2).unsqueeze(3), P)

        loss_st = torch.sum(torch.pow(P_prime_st - P, 2), 1)
        loss_st = torch.mean(loss_st)

        loss_tr = torch.sum(torch.pow(P_prime_tr - P, 2), 1)
        loss_tr = torch.mean(loss_tr)

        loss = (loss_st + loss_tr) / 2
        return loss