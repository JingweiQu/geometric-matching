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

    if subset == 'train':
        for i in range(7):
            images_A.extend(img_A_names)
            images_B.extend(img_B_names)
            categories.extend(category)
            flips.extend(flip)

    # Save training pairs
    dataframe = pd.DataFrame({'source_image': images_A, 'target_image': images_B, 'class': categories, 'flip': flips})
    dataframe.to_csv(target_file, index=False)

def gt_tps(csv_file=None, random_t_tps=0.4):
    data = pd.read_csv(csv_file)
    num_pairs = len(data)

    theta = np.zeros((num_pairs, 18))
    for i in range(num_pairs):
        theta_tps = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1])
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

source_file = '../training_data/PF-PASCAL/PF-dataset-PASCAL/train_pairs_pf_pascal.csv'
target_file = '../training_data/PF-PASCAL/train_aff_PF-PASCAL.csv'
train_csv(source_file, target_file, 'train')

# random_t_tps = 0.4
# random_t_tps = 0.3
random_t_tps = 0.2
# random_t_tps = 0.1
# gt_tps(target_file, random_t_tps)
gt_affine(csv_file=target_file)

print('Done!')