from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from geometric_matching.geotnf.point_tnf import PointTnf

class TransformedGridLoss(nn.Module):
    """
    Compute loss by computing distances between
    (1) grid points transformed by ground-truth theta
    (2) grid points transformed by predicted theta_tr and theta_st
    """
    def __init__(self, geometric_model='affine', use_cuda=True, grid_size=20):
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
        if self.geometric_model == 'affine':
            P_prime = self.pointTnf.affPointTnf(theta_tr, P)
            P_prime = self.pointTnf.affPointTnf(theta_st, P_prime)
            P_prime_GT = self.pointTnf.affPointTnf(theta_GT, P)
        elif self.geometric_model == 'tps':
            P_prime = self.pointTnf.tpsPointTnf(theta_tr.unsqueeze(2).unsqueeze(3), P)
            P_prime = self.pointTnf.tpsPointTnf(theta_st.unsqueeze(2).unsqueeze(3), P_prime)
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
        # X and Y.shape: (1, 1, 400), P.shape: (1`, 2, 400)
        X = np.reshape(X, (1, 1, self.N))
        Y = np.reshape(Y, (1, 1, self.N))
        P = np.concatenate((X, Y), 1)
        # self.P = Variable(torch.FloatTensor(P), requires_grad=False)
        self.P = torch.Tensor(P.astype(np.float32))
        self.P.requires_grad = False
        self.pointTnf = PointTnf(use_cuda=use_cuda)
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
        if self.geometric_model == 'affine':
            P_prime = self.pointTnf.affPointTnf(theta_AB, P)
            P_prime = self.pointTnf.affPointTnf(theta_BA, P_prime)
        elif self.geometric_model == 'tps':
            P_prime = self.pointTnf.tpsPointTnf(theta_AB.unsqueeze(2).unsqueeze(3), P)
            P_prime = self.pointTnf.tpsPointTnf(theta_BA.unsqueeze(2).unsqueeze(3), P_prime)
        # compute MSE loss on transformed grid points
        loss = torch.sum(torch.pow(P_prime - P, 2), 1)
        loss = torch.mean(loss)
        return loss

class CosegLoss(nn.Module):
    def __init__(self, use_cuda=True):
        super(CosegLoss, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.FeatureExtraction = nn.Sequential(*list(resnet50.children())[:-1])
        for param in self.FeatureExtraction.parameters():
            param.requires_grad = False
        if use_cuda:
            self.FeatureExtraction.cuda()

    # def forward(self, mask_A, mask_B, image_A, image_B):
    #     mask_A = torch.sigmoid(mask_A)
    #     mask_B = torch.sigmoid(mask_B)
    #
    #     obj_A = torch.mul(image_A, mask_A)
    #     obj_B = torch.mul(image_B, mask_B)
    #
    #     bg_A = torch.mul(image_A, 1.0 - mask_A)
    #     bg_B = torch.mul(image_B, 1.0 - mask_B)
    #
    #     feature_obj_A = self.FeatureExtraction(obj_A)
    #     feature_obj_B = self.FeatureExtraction(obj_B)
    #
    #     feature_bj_A = self.FeatureExtraction(bg_A)
    #     feature_bj_B = self.FeatureExtraction(bg_B)
    #
    #     feature_obj_A = torch.squeeze(feature_obj_A)
    #     feature_obj_B = torch.squeeze(feature_obj_B)
    #     feature_bj_A = torch.squeeze(feature_bj_A)
    #     feature_bj_B = torch.squeeze(feature_bj_B)
    #
    #     d_inter = torch.mean(torch.pow(feature_obj_A - feature_obj_B, 2))
    #     d_intra = torch.max(torch.Tensor([0]).cuda(), 2 - (torch.mean(torch.pow(feature_obj_A - feature_bj_A, 2)) + torch.mean(torch.pow(feature_obj_B - feature_bj_B, 2))) / 2)
    #
    #     loss = d_inter + d_intra
    #
    #     return loss

    def forward(self, mask_A, mask_B, image_A, image_B):
        # mask_A = torch.sigmoid(mask_A)
        # mask_B = torch.sigmoid(mask_B)

        obj_A = torch.squeeze(self.FeatureExtraction(torch.mul(image_A, mask_A)))
        back_A = torch.squeeze(self.FeatureExtraction(torch.mul(image_A, 1.0 - mask_A)))

        obj_B = torch.squeeze(self.FeatureExtraction(torch.mul(image_B, mask_B)))
        back_B = torch.squeeze(self.FeatureExtraction(torch.mul(image_B, 1.0 - mask_B)))

        batch, dim = obj_A.size()
        pos = (torch.dist(obj_A, obj_B, p=2) ** 2) / dim / batch
        neg = torch.max(torch.Tensor([0]).cuda().squeeze(), 2 - ((torch.dist(obj_A, back_A, p=2) ** 2 + torch.dist(obj_B, back_B, p=2) ** 2) / dim / batch / 2))

        loss = pos + neg

        return loss