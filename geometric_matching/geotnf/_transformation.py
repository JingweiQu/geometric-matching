from __future__ import print_function, division
import os
import sys
from skimage import io
import pandas as pd
import numpy as np
import torch
from torch.nn.modules.module import Module
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F


class GeometricTnf(object):
    """
    
    Geometric transfromation to an image batch (wrapped in a PyTorch Variable)
    ( can be used with no transformation to perform bilinear resizing )        

    """
    def __init__(self, geometric_model='affine', out_h=240, out_w=240, use_cuda=True):
        self.out_h = out_h
        self.out_w = out_w
        self.use_cuda = use_cuda
        if geometric_model == 'affine':
            self.gridGen = AffineGridGen(out_h, out_w)
        elif geometric_model == 'tps':
            self.gridGen = TpsGridGen(out_h, out_w, use_cuda=use_cuda)
        # theta_identity.shape: (1, 2, 3), mainly use as affine transformation parameters for
        # (1) resize the image from original size to (480, 640) when initializing the dataset
        # (2) crop the image from (480. 640) to (240, 240) when generating the training image pairs
        self.theta_identity = torch.Tensor(np.expand_dims(np.array([[1, 0, 0], [0, 1, 0]]), 0).astype(np.float32))
        if use_cuda:
            self.theta_identity = self.theta_identity.cuda()

    def __call__(self, image_batch, theta_batch=None, padding_factor=1.0, crop_factor=1.0):
        # padding_factor and crop_factor are used for grid
        # image_batch.shape: (batch_size, 3, H, W)
        b, c, h, w = image_batch.size()
        # Use theta_identity as affine transformation parameters for
        # (1) resize the image from original size to (480, 640) when initializing the dataset
        # (2) crop the image from (480. 640) to (240, 240) when generating the training image pairs
        if theta_batch is None:
            theta_batch = self.theta_identity
            theta_batch = theta_batch.expand(b, 2, 3)
            theta_batch = Variable(theta_batch, requires_grad=False)

        # Generate the grid for geometric transformation (affine or tps) with the given theta (theta_batch)
        # theta is the parameters for geometric transformation from output image to input image
        # sampling_grid.shape is (batch_size, out_h, out_w, 2), such as (240, 240)
        # 2 includes coordinates (x, y) in the input image (image_batch)
        # For (x, y) in sampling_grid[i][j] (ignore batch dim):
        # use pixel value in (x, y) of the input image as the pixel value in (i, j) of the output image
        sampling_grid = self.gridGen(theta_batch)

        # rescale grid according to crop_factor and padding_factor
        # Rescale (x, y) in grid (i.e. coordinates in the input image) with crop_factor and padding_factor
        sampling_grid.data = sampling_grid.data * padding_factor * crop_factor
        # sample transformed image, warped_image_batch.shape: (batch_size, 3, out_h, out_h)
        # For (x, y) in sampling_grid[i][j] (ignore batch dim):
        # use pixel value in (x, y) of the image (image_batch) as the pixel value in (i, j) of the image (warped_image_batch)
        # (x, y) is float, use default bilinear interpolation to obtain the pixel value in (x, y)
        warped_image_batch = F.grid_sample(image_batch, sampling_grid)

        return warped_image_batch
    

# Generate a synthetically warped training pair using a given geometric transformation
class SynthPairTnf(object):
    """
    
    Generate a synthetically warped training pair using an affine transformation.
    
    """
    def __init__(self, use_cuda=True, geometric_model='affine', crop_factor=9/16, output_size=(240, 240), padding_factor=0.5):
        assert isinstance(use_cuda, (bool))
        assert isinstance(crop_factor, (float))
        assert isinstance(output_size, (tuple))
        assert isinstance(padding_factor, (float))
        self.use_cuda = use_cuda
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h, self.out_w = output_size
        # Use an identity and simple (just scaling) affine transformation to crop the input image as (240,240)
        self.rescalingTnf = GeometricTnf('affine', self.out_h, self.out_w, use_cuda=self.use_cuda)
        # Initialize geometric transformation (tps or affine) to warp the image to form the training pair
        self.geometricTnf = GeometricTnf(geometric_model, self.out_h, self.out_w, use_cuda=self.use_cuda)
        
    def __call__(self, batch):
        # image_batch.shape: (batch_size, 3, H, W), such as (480, 640)
        # theta_batch.shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        # theta_batch.shape-affine: (batch_size, 2, 3)
        # image_batch, theta_batch = batch['image'], batch['theta']
        images_A, images_B, theta_batch = batch['image_A'], batch['image_B'], batch['theta']

        if self.use_cuda:
            # image_batch = image_batch.cuda()
            images_A = images_A.cuda()
            images_B = images_B.cuda()
            theta_batch = theta_batch.cuda()
            
        # b, c, h, w = image_batch.size()
        b, c, h, w = images_A.size()
              
        # generate symmetrically padded image for bigger sampling region to warp the image
        # image_batch = self.symmetricImagePad(image_batch, self.padding_factor)
        images_A = self.symmetricImagePad(images_A, self.padding_factor)
        images_B = self.symmetricImagePad(images_B, self.padding_factor)
        
        # convert to variables
        # image_batch = Variable(image_batch, requires_grad=False)
        images_A = Variable(images_A, requires_grad=False)
        images_B = Variable(images_B, requires_grad=False)
        theta_batch = Variable(theta_batch, requires_grad=False)

        # get cropped image based on the padded image, identity (theta_identity) is used as no theta given
        # cropped_image_batch = self.rescalingTnf(image_batch, None, self.padding_factor, self.crop_factor)
        cropped_image_batch = self.rescalingTnf(images_A, None, self.padding_factor, self.crop_factor)
        # get transformed image based on the padded image, i.e. warped image
        # warped_image_batch = self.geometricTnf(image_batch, theta_batch, self.padding_factor, self.crop_factor)
        warped_image_batch = self.geometricTnf(images_B, theta_batch, self.padding_factor, self.crop_factor)

        # cropped_image_batch.shape and warped_image_batch.shape: (batch_size, 3, out_h, out_w), such as (240, 240)
        # theta_batch.shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        # theta_batch.shape-affine: (batch_size, 2, 3)
        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch, 'theta_GT': theta_batch}

    def symmetricImagePad(self, image_batch, padding_factor):
        b, c, h, w = image_batch.size()
        pad_h, pad_w = int(h*padding_factor), int(w*padding_factor)
        # Use these four regions to perform symmetric padding for the image
        idx_pad_left = torch.LongTensor(range(pad_w-1, -1, -1))
        idx_pad_right = torch.LongTensor(range(w-1, w-pad_w-1, -1))
        idx_pad_top = torch.LongTensor(range(pad_h-1, -1, -1))
        idx_pad_bottom = torch.LongTensor(range(h-1, h-pad_h-1, -1))
        if self.use_cuda:
                idx_pad_left = idx_pad_left.cuda()
                idx_pad_right = idx_pad_right.cuda()
                idx_pad_top = idx_pad_top.cuda()
                idx_pad_bottom = idx_pad_bottom.cuda()
        # Symmetric padding for the image
        image_batch = torch.cat((image_batch.index_select(3, idx_pad_left), image_batch,
                                 image_batch.index_select(3, idx_pad_right)), 3)
        image_batch = torch.cat((image_batch.index_select(2, idx_pad_top), image_batch,
                                 image_batch.index_select(2, idx_pad_bottom)), 2)
        return image_batch


# Generate the grid for affine transformation with the given theta
# theta is the parameters for affine transformation from output image to input image
# grid.shape is (batch_size, out_h, out_w, 2), such as (240, 240), 2 includes coordinates (x, y) in the input image
# For (x, y) in grid[i][j] (ignore batch dim):
# use pixel value in (x, y) of the input image as the pixel value in (i, j) of the output image
class AffineGridGen(Module):
    def __init__(self, out_h=240, out_w=240, out_ch = 3):
        super(AffineGridGen, self).__init__()        
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch
        
    def forward(self, theta):
        # theta.shape: (1, 2, 3) for affine, 6 parameters
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size, self.out_ch, self.out_h, self.out_w))
        return F.affine_grid(theta, out_size)


# Generate the grid for tps transformation with the given theta
# theta is the parameters for tps transformation from output image to input image
# grid.shape is (batch_size, out_h, out_w, 2), such as (240, 240), 2 includes coordinates (x, y) in the input image
# For (x, y) in grid[i][j] (ignore batch dim):
# use pixel value in (x, y) of the input image as the pixel value in (i, j) of the output image
class TpsGridGen(Module):
    def __init__(self, out_h=240, out_w=240, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # Create grid in numpy, i.e. self.grid_X and self.grid_Y
        # self.grid.shape: (out_h, out_w, 3)
        self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y, out_h)
        # Grid scale is (-1, -1, 1, 1), a square with (x_min, y_min, x_max, y_max), the points is out_h * out_w
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # self.grid_X, self.grid_Y: size [1, out_h, out_h, 1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X = Variable(self.grid_X, requires_grad=False)
        self.grid_Y = Variable(self.grid_Y, requires_grad=False)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # Initialize regular grid for control points P_i (self.P_X and self.P_Y), 3 * 3
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            # Grid scale is (-1, -1, 1, 1), a square with (x_min, y_min, x_max, y_max), the points is 3 * 3
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            # P_X.shape and P_Y.shape: (9, 1)
            P_X = np.reshape(P_X, (-1, 1))  # size (N,1)
            P_Y = np.reshape(P_Y, (-1, 1))  # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            # self.Li.shape: (1, 12, 12)
            self.Li = Variable(self.compute_L_inverse(P_X, P_Y).unsqueeze(0), requires_grad=False)
            # self.P_X.shape and self.P_Y.shape: (1, 1, 1, 1, 9)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_X = Variable(self.P_X, requires_grad=False)
            self.P_Y = Variable(self.P_Y, requires_grad=False)
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()

    def forward(self, theta):

        # Generate the grid (warped_grid) for tps transformation with the given theta
        # theta.shape: (batch_size, 18) for tps
        # self.grid_X, self.grid_Y: size [1, out_h, out_w, 1]
        # warped_grid.shape: (batch_size, out_h, out_w, 2)
        warped_grid = self.apply_transformation(theta, torch.cat((self.grid_X, self.grid_Y), 3))
        
        return warped_grid
    
    def compute_L_inverse(self, X, Y):
        # X.shape and Y.shape: (9, 1)
        N = X.size()[0] # num of points (along dim 0)
        # construct matrix K, Xmat.shape and Ymax.shape: (9. 9)
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        # Distance squared matrix, P_dist_squared.shape: (9, 9)
        P_dist_squared = torch.pow(Xmat-Xmat.transpose(0, 1), 2) + torch.pow(Ymat-Ymat.transpose(0, 1), 2)
        P_dist_squared[P_dist_squared==0] = 1 # make diagonal 1 to avoid NaN in log computation
        # K.shape: (9, 9), P.shape: (9, 3), L.shape: (12, 12)
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))
        # construct matrix L
        O = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((O, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1), torch.cat((P.transpose(0, 1), Z), 1)), 0)
        # Li is inverse matrix of L, Li.shape: (12, 12)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li

    # Generate the grid for tps transformation with the given theta and out_size (points)
    def apply_transformation(self, theta, points):
        # theta.shape: (batch_size, 18) for tps
        # points.shape: (batch_size, out_h, out_w, 2), for loss: (batch_size, 1, 400, 2)
        # theta.shape becomes (batch_size, 18, 1, 1)
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords  
        # and points[:,:,:,1] are the Y coords  
        
        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        # Q_X.shape and Q_Y.shape: (batch_size, 9, 1)
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)
        
        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]
        
        # repeat pre-defined control points along spatial dimensions of points to be transformed
        # P_X.shape and P_Y.shape: (1, out_h, out_w, 1, 9)
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))

        # TPS consists of an affine part and a non-linear part
        # compute weigths for non-linear part
        # W_X.shape and W_Y.shape: (batch_size, 9, 1)
        W_X = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_X)
        W_Y = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N], i.e. (batch_size, out_h, out_w, 1, 9)
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        # compute weights for affine part
        # A_X.shape and A_Y.shape: (batch_size, 3, 1)
        A_X = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_X)
        A_Y = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3], i.e. (batch_size, out_h, out_w, 1, 3)
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        
        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        # points_X_for_summation.shape and points_Y_for_summation.shape: (batch_size, H, W, 1, 9)
        points_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 0].size() + (1, self.N))
        points_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 1].size() + (1, self.N))
        if points_b == 1:
            delta_X = points_X_for_summation - P_X
            delta_Y = points_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation - P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation - P_Y.expand_as(points_Y_for_summation)
            
        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        # U: size [1,H,W,1,N], i.e.(1, out_h, out_w, 1, 9)
        dist_squared[dist_squared == 0] = 1 # avoid NaN in log computation
        U = torch.mul(dist_squared, torch.log(dist_squared))
        
        # expand grid in batch dimension if necessary
        # points_X_batch.shape and points_Y_batch.shape: (batch_size, out_h, out_w, 1)
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) + points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) + points_Y_batch.size()[1:])

        # points_X_prime.shape and points_Y_prime.shape: (batch_size, out_h, out_w, 1)
        points_X_prime = A_X[:, :, :, :, 0] + \
                       torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
                       torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
                       torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)
                    
        points_Y_prime = A_Y[:, :, :, :, 0] + \
                       torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
                       torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
                       torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)

        # Return grid.shape: (batch_size, out_h, out_w, 2)
        return torch.cat((points_X_prime, points_Y_prime), 3)