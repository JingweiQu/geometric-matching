# ====================================================================================================
# Test geometric matching model on benchmarks based on the object detection by faster rcnn
# Make an affine transformation before the predicted tps:
# (1) Use the coordinates of bounding boxes (i.e. object detection of the two images) as translation
# parameters of the affine
# (2) Resize the two objects to solve the scale transformation in the affine
# Author: Jingwei Qu
# Date: 27 April 2019
# ====================================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse
import pprint
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pdb
from PIL import Image
from collections import OrderedDict
from visdom import Visdom

from geometric_matching.arguments.arguments_setting import Arguments
from geometric_matching.gm_model.geometric_matching_cycle2 import GeometricMatching
from geometric_matching.image.normalization import *
from geometric_matching.gm_data.pf_willow_dataset import PFWILLOWDataset
from geometric_matching.gm_data.pf_pascal_dataset import PFPASCALDataset
from geometric_matching.gm_data.caltech_dataset import CaltechDataset
from geometric_matching.gm_data.tss_dataset import TSSDataset
from geometric_matching.util.net_util import *
from geometric_matching.util.test_watch import test_watch
from geometric_matching.util.test_watch_tps import test_watch as test_watch2
from geometric_matching.util.dataloader import default_collate
from geometric_matching.util.vis_watch_tps import *

if __name__ == '__main__':

    print('Test GeometricMatching model')

    ''' Load arguments '''
    args, arg_groups = Arguments(mode='test').parse()
    print('Arguments setting:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    ''' Initialize geometric matching model '''
    print('Initialize geometric matching model')
    do_aff = False
    do_tps = True
    pytorch = False
    caffe = False
    fixed_blocks = 3
    # Create geometric_matching model
    # Default: args.geometric_model - tps, args.feature_extraction_cnn - vgg
    output_dim_1 = 18
    output_dim_2 = 36
    # model_1 = GeometricMatching2(output_dim=output_dim_1, **arg_groups['model'], fixed_blocks=fixed_blocks, pytorch=pytorch, caffe=caffe)
    model_1 = GeometricMatching(output_dim=output_dim_2, **arg_groups['model'], fixed_blocks=fixed_blocks, pytorch=pytorch, caffe=caffe)
    model_2 = GeometricMatching(output_dim=output_dim_2, **arg_groups['model'], fixed_blocks=fixed_blocks, pytorch=pytorch, caffe=caffe)

    ''' Load trained geometric matching model '''
    GM_cp_name_1 = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, args.model_1)
    if not os.path.exists(GM_cp_name_1):
        raise Exception('There is no pre-trained geometric matching model, i.e. ' + GM_cp_name_1)
    print('Load geometric matching model {}'.format(GM_cp_name_1))
    GM_checkpoint_1 = torch.load(GM_cp_name_1, map_location=lambda storage, loc: storage)
    model_1.load_state_dict(GM_checkpoint_1['state_dict'])

    GM_cp_name_2 = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, args.model_2)
    if not os.path.exists(GM_cp_name_2):
        raise Exception('There is no pre-trained geometric matching model, i.e. ' + GM_cp_name_2)
    print('Load geometric matching model {}'.format(GM_cp_name_2))
    GM_checkpoint_2 = torch.load(GM_cp_name_2, map_location=lambda storage, loc: storage)
    model_2.load_state_dict(GM_checkpoint_2['state_dict'])

    print('Load geometric matching model successfully!')

    if args.cuda:
        model_1.cuda()
        model_2.cuda()

    args.test_dataset = 'TSS'
    ''' Set testing dataset & evaluation metric '''
    if args.test_dataset == 'PF-WILLOW':
        TestDataSet = PFWILLOWDataset
        metric = 'pck'
    elif args.test_dataset == 'PF-PASCAL':
        TestDataSet = PFPASCALDataset
        metric = 'pck'
    elif args.test_dataset == 'Caltech-101':
        TestDataSet = CaltechDataset
        metric = 'area'
    elif args.test_dataset == 'TSS':
        TestDataSet = TSSDataset
        metric = 'flow'
    collate_fn = default_collate

    # Set path of csv file including image names (source and target) and annotation
    csv_file_watch, test_dataset_path = get_dataset_csv(dataset_path=args.test_dataset_path, dataset=args.test_dataset, subset='watch')
    print(csv_file_watch)
    output_size = (args.image_size, args.image_size)
    normalize = NormalizeImageDict(['source_image', 'target_image'])
    # normalize = None
    dataset = TestDataSet(csv_file=csv_file_watch, dataset_path=test_dataset_path, output_size=output_size, normalize=normalize)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    ''' Test trained geometric matching model '''
    print('Test on {}'.format(args.test_dataset))
    # vis = Visdom(env='vsTPS')
    vis = Visdom()
    model_1.eval()
    model_2.eval()
    with torch.no_grad():
        # results_1, theta_1, theta_inver_1, _ = test_watch(model=model_1, metric=metric, dataset=dataset, dataloader=dataloader, do_aff=do_aff, do_tps=do_tps, args=args)
        results_1, theta_1, theta_inver_1, _ = test_watch2(model=model_1, metric=metric, dataset=dataset, dataloader=dataloader, do_aff=do_aff, do_tps=do_tps, args=args)
        results_2, theta_2, theta_inver_2, _ = test_watch2(model=model_2, metric=metric, dataset=dataset, dataloader=dataloader, do_aff=do_aff, do_tps=do_tps, args=args)
        # if args.test_dataset == 'PF-PASCAL' or args.test_dataset == 'PF-WILLOW':
        #     vis_pf(vis, dataloader, theta_1, theta_2, theta_inver_1, theta_inver_2, results_1, results_2, args.test_dataset, use_cuda=args.cuda)
        #     # vis_pf_2(vis, dataloader, theta_1, theta_2, theta_inver_1, theta_inver_2, results_1, results_2, args.test_dataset, use_cuda=args.cuda)
        # elif args.test_dataset == 'Caltech-101':
        #     vis_caltech(vis, dataloader, theta_1, theta_2, results_1, results_2, args.test_dataset, use_cuda=args.cuda)
        # elif args.test_dataset == 'TSS':
        #     vis_tss(vis, dataloader, theta_1, theta_2, csv_file_watch, args.test_dataset, use_cuda=args.cuda)
        # vis_control(vis, dataloader, theta_1, theta_2, args.test_dataset, use_cuda=args.cuda)
        vis_control2(vis, dataloader, theta_1, theta_2, args.test_dataset, use_cuda=args.cuda)

    print('Done!')