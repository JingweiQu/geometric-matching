# ========================================================================================
# Enlarge training pairs of PF-PASCAL
# Randomly generate ground-truth tps parameters for training pairs of PF-PASCAL
# Author: Jingwei Qu
# Date: 25 April 2019
# ========================================================================================

import os
import pandas as pd
import numpy as np
import copy
import torch

# def Generate_Rand(rand_range, n):
#     tmp = np.random.randint(rand_range, size=1)[0]
#     if tmp != n:
#         return tmp
#     else:
#         return Generate_Rand(rand_range, n)

def train_csv(source_file, target_file, subset):
    # dataset_classes = np.asarray(['aeroplane', 'bicycle', 'bird', 'boat',
    #                        'bottle', 'bus', 'car', 'cat', 'chair',
    #                        'cow', 'diningtable', 'dog', 'horse',
    #                        'motorbike', 'person', 'pottedplant',
    #                        'sheep', 'sofa', 'train', 'tvmonitor'])

    csv_data = pd.read_csv(source_file)
    img_A_names = csv_data.iloc[:, 0].values.tolist()
    img_B_names = csv_data.iloc[:, 1].values.tolist()
    category = csv_data.iloc[:, 2].values.tolist()
    flip = csv_data.iloc[:, 3].values.tolist()

    images_A = copy.deepcopy(img_A_names)
    images_B = copy.deepcopy(img_B_names)
    categories = copy.deepcopy(category)
    flips = copy.deepcopy(flip)

    # extend_prop = 7
    extend_prop = 15
    if subset == 'train':
        for i in range(extend_prop):
            images_A.extend(img_A_names)
            images_B.extend(img_B_names)
            categories.extend(category)
            flips.extend(flip)

    # Save training pairs
    dataframe = pd.DataFrame({'source_image': images_A, 'target_image': images_B, 'class': categories, 'flip': flips})
    dataframe.to_csv(target_file, index=False)

def gt_tps(csv_file=None, coor=1.0, random_t_tps=0.4):
    data = pd.read_csv(csv_file)
    num_pairs = len(data)

    theta = np.zeros((num_pairs, 18))
    for i in range(num_pairs):
        theta_tps = np.array([-coor, -coor, -coor, 0, 0, 0, coor, coor, coor, -coor, 0, coor, -coor, 0, coor, -coor, 0, coor])
        theta_tps = theta_tps + (np.random.rand(18) - 0.5) * 2 * random_t_tps
        theta[i, :] = theta_tps
        # if i == 0:
        #     print(theta_tps)

    frame_dict = {}
    for i in range(18):
        header = 't' + str(i+1)
        frame_dict[header] = theta[:, i]

    dataframe = pd.DataFrame(frame_dict)
    dataframe = pd.concat([data, dataframe], axis=1, sort=False)
    dataframe.to_csv(csv_file, index=False)

def compute_L_inverse(X, Y):
    # X.shape and Y.shape: (9, 1)
    N = X.size()[0]  # num of points (along dim 0)
    # construct matrix K, Xmat.shape and Ymax.shape: (9. 9)
    Xmat = X.expand(N, N)
    Ymat = Y.expand(N, N)
    # Distance squared matrix, P_dist_squared.shape: (9, 9)
    P_dist_squared = torch.pow(Xmat - Xmat.transpose(0, 1), 2) + torch.pow(Ymat - Ymat.transpose(0, 1), 2)
    P_dist_squared[P_dist_squared == 0] = 1  # make diagonal 1 to avoid NaN in log computation
    # P_dist_squared = P_dist_squared + 1e-6  # make diagonal 1 to avoid NaN in log computation
    # K.shape: (9, 9), P.shape: (9, 3), L.shape: (12, 12)
    K = torch.mul(P_dist_squared, torch.log(P_dist_squared))
    # construct matrix L
    O = torch.Tensor(N, 1).fill_(1)
    Z = torch.Tensor(3, 3).fill_(0)
    P = torch.cat((O, X, Y), 1)
    L = torch.cat((torch.cat((K, P), 1), torch.cat((P.transpose(0, 1), Z), 1)), 0)
    # Li is inverse matrix of L, Li.shape: (12, 12)
    Li = torch.inverse(L)
    return Li

def gt_tps2(csv_file=None, coor=1.0, random_t_tps=0.4):
    data = pd.read_csv(csv_file)
    num_pairs = len(data)

    grid_size = 3
    axis_coords = np.linspace(-1, 1, grid_size)
    N = grid_size * grid_size
    # Grid scale is (-1, -1, 1, 1), a square with (x_min, y_min, x_max, y_max), the points is 3 * 3
    P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
    # P_X.shape and P_Y.shape: (9, 1)
    P_X = np.reshape(P_X, (-1, 1))  # size (N,1)
    P_Y = np.reshape(P_Y, (-1, 1))  # size (N,1)
    P_X = torch.Tensor(P_X.astype(np.float32))
    P_Y = torch.Tensor(P_Y.astype(np.float32))
    # Li.shape: (12, 12)
    Li = compute_L_inverse(P_X, P_Y)

    theta = np.zeros((num_pairs, 24))
    for i in range(num_pairs):
        theta_tps = np.array([-coor, -coor, -coor, 0, 0, 0, coor, coor, coor, -coor, 0, coor, -coor, 0, coor, -coor, 0, coor])
        theta_tps = theta_tps + (np.random.rand(18) - 0.5) * 2 * random_t_tps
        theta_tps = np.reshape(theta_tps, (-1, 1))
        theta_tps = torch.Tensor(theta_tps.astype(np.float32))

        # Q_X.shape and Q_Y.shape: (9, 1)
        Q_X = theta_tps[:N, :]
        Q_Y = theta_tps[N:, :]

        # TPS consists of an affine part and a non-linear part
        # compute weigths for non-linear part
        # W_X.shape and W_Y.shape: (9, 1)
        W_X = torch.mm(Li[:N, :N], Q_X).squeeze(1)
        W_Y = torch.mm(Li[:N, :N], Q_Y).squeeze(1)
        # compute weights for affine part
        # A_X.shape and A_Y.shape: (3, 1)
        A_X = torch.mm(Li[N:, :N], Q_X).squeeze(1)
        A_Y = torch.mm(Li[N:, :N], Q_Y).squeeze(1)

        theta[i, :] = torch.cat((A_X, A_Y, W_X, W_Y)).numpy()

    frame_dict = {}
    header_aff = ['tx', 'a11', 'a12', 'ty', 'a21', 'a22']
    for i in range(24):
        if i < 6:
            frame_dict[header_aff[i]] = theta[:, i]
        elif i < 15:
            header = 'wx' + str(i-5)
            frame_dict[header] = theta[:, i]
        else:
            header = 'wy' + str(i-14)
            frame_dict[header] = theta[:, i]

    dataframe = pd.DataFrame(frame_dict)
    dataframe = pd.concat([data, dataframe], axis=1, sort=False)
    dataframe.to_csv(csv_file, index=False)

def gt_affine(csv_file=None, random_alpha=1/6, random_t=0.5, random_s=0.5):
    data = pd.read_csv(csv_file)
    num_pairs = len(data)

    theta = np.zeros((num_pairs, 6))
    for i in range(num_pairs):
        alpha = (np.random.rand(1) - 0.5) * 2 * np.pi * random_alpha
        theta_aff = np.random.rand(6)
        theta_aff[[2, 5]] = (theta_aff[[2, 5]] - 0.5) * 2 * random_t    # translation
        # scale & rotation
        theta_aff[0] = (1 + (theta_aff[0] - 0.5) * 2 * random_s) * np.cos(alpha)
        theta_aff[1] = (1 + (theta_aff[1] - 0.5) * 2 * random_s) * (-np.sin(alpha))
        theta_aff[3] = (1 + (theta_aff[3] - 0.5) * 2 * random_s) * np.sin(alpha)
        theta_aff[4] = (1 + (theta_aff[4] - 0.5) * 2 * random_s) * np.cos(alpha)
        theta[i, :] = theta_aff
        # if i == 0:
        #     print(theta_aff)

    frame_dict = {}
    header = ['A11', 'A12', 'tx', 'A21', 'A22', 'ty']
    for i in range(6):
        frame_dict[header[i]] = theta[:, i]

    dataframe = pd.DataFrame(frame_dict)
    dataframe = pd.concat([data, dataframe], axis=1, sort=False)
    dataframe.to_csv(csv_file, index=False)

def gt_afftps(csv_file=None, random_alpha=1/6, random_t=0.5, random_s=0.5, random_t_tps=0.4):
    data = pd.read_csv(csv_file)
    num_pairs = len(data)

    theta = np.zeros((num_pairs, 24))
    for i in range(num_pairs):
        alpha = (np.random.rand(1) - 0.5) * 2 * np.pi * random_alpha
        theta_aff = np.random.rand(6)
        theta_aff[[2, 5]] = (theta_aff[[2, 5]] - 0.5) * 2 * random_t  # translation
        # scale & rotation
        theta_aff[0] = (1 + (theta_aff[0] - 0.5) * 2 * random_s) * np.cos(alpha)
        theta_aff[1] = (1 + (theta_aff[1] - 0.5) * 2 * random_s) * (-np.sin(alpha))
        theta_aff[3] = (1 + (theta_aff[3] - 0.5) * 2 * random_s) * np.sin(alpha)
        theta_aff[4] = (1 + (theta_aff[4] - 0.5) * 2 * random_s) * np.cos(alpha)

        theta_tps = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1])
        theta_tps = theta_tps + (np.random.rand(18) - 0.5) * 2 * random_t_tps
        theta[i, :] = np.concatenate((theta_aff, theta_tps))

    frame_dict = {}
    header_aff = ['A11', 'A12', 'tx', 'A21', 'A22', 'ty']
    for i in range(6):
        frame_dict[header_aff[i]] = theta[:, i]

    for i in range(18):
        header_tps = 't' + str(i + 1)
        frame_dict[header_tps] = theta[:, i+6]

    dataframe = pd.DataFrame(frame_dict)
    dataframe = pd.concat([data, dataframe], axis=1, sort=False)
    dataframe.to_csv(csv_file, index=False)

geometric_model = 'tps'
source_file = '../training_data/PF-PASCAL/PF-dataset-PASCAL/train_pairs_pf_pascal.csv'
# target_file = '../training_data/PF-PASCAL/train_' + geometric_model + '0.3_PF-PASCAL.csv'
# target_file = '../training_data/PF-PASCAL/finetune_' + geometric_model + '_PF-PASCAL.csv'
target_file = '../training_data/PF-PASCAL/train_' + geometric_model + '_PF-PASCAL_24.csv'
train_csv(source_file, target_file, 'train')

# random_t_tps = 0.4
# random_t_tps = 0.3
# random_t_tps = 0.2
# random_t_tps = 0.1
if geometric_model == 'tps':
    # gt_tps(csv_file=target_file, coor=0.8, random_t_tps=0.3)
    gt_tps2(csv_file=target_file, coor=1.0, random_t_tps=0.4)
elif geometric_model == 'affine':
    gt_affine(csv_file=target_file)
elif geometric_model == 'afftps':
    gt_afftps(csv_file=target_file, random_alpha=1/6, random_t=0.3, random_s=0.3, random_t_tps=0.2)

print('Done!')