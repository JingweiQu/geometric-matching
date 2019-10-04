from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
from geometric_matching.geotnf.point_tnf import PointTnf

class TransformedGridLoss(nn.Module):
    """
    Compute loss by computing distances between
    (1) grid points transformed by ground-truth theta
    (2) grid points transformed by predicted theta_tr and theta_st
    """
    def __init__(self, use_cuda=True, grid_size=20):
        super(TransformedGridLoss, self).__init__()
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
        if use_cuda:
            self.P = self.P.cuda()

    def forward(self, theta_aff_tps, theta_aff, theta_GT):
        # expand grid according to batch size
        # theta.shape: (batch_size, 18) for tps, (batch_size, 6) for affine
        # theta_GT.shape: (batch_size, 18, 1, 1) for tps, (batch_size, 6) for affine
        batch_size = theta_aff_tps.size()[0]
        # P.shape: (batch_size, 2, 400)
        P = self.P.expand(batch_size, 2, self.N)
        # compute transformed grid points using estimated and GT tnfs
        # P_prime and P_prime_GT.shape: (batch_size, 2, 400)
        P_prime = self.pointTnf.tpsPointTnf(theta_aff_tps.unsqueeze(2).unsqueeze(3), P)
        P_prime = self.pointTnf.affPointTnf(theta_aff, P_prime)
        P_prime_GT = self.pointTnf.tpsPointTnf(theta_GT, P)
        # compute MSE loss on transformed grid points
        loss = torch.sum(torch.pow(P_prime - P_prime_GT, 2), 1)
        loss = torch.mean(loss)
        return loss