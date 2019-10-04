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
# from geometric_matching.gm_model.geometric_matching import GeometricMatching
# from geometric_matching.gm_model.geometric_matching_cycle import GeometricMatching
# from geometric_matching.gm_model.geometric_matching_cycle2 import GeometricMatching
from geometric_matching.gm_model.geometric_matching_cycle3 import GeometricMatching
from geometric_matching.image.normalization import *
from geometric_matching.gm_data.pf_willow_dataset import PFWILLOWDataset
from geometric_matching.gm_data.pf_pascal_dataset import PFPASCALDataset
from geometric_matching.gm_data.caltech_dataset import CaltechDataset
from geometric_matching.gm_data.tss_dataset import TSSDataset
from geometric_matching.util.net_util import *
from geometric_matching.util.test_fn import *
from geometric_matching.util.vis_feature import vis_feature
from geometric_matching.util.dataloader import default_collate

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
    do_aff = args.geometric_model == 'affine'
    do_tps = args.geometric_model == 'tps'
    pytorch = False
    caffe = False
    fixed_blocks = 3
    # Create geometric_matching model
    # Default: args.geometric_model - tps, args.feature_extraction_cnn - vgg
    if do_aff:
        output_dim = 6
    if do_tps:
        output_dim = 18
    model = GeometricMatching(output_dim=output_dim, **arg_groups['model'], fixed_blocks=fixed_blocks, pytorch=pytorch, caffe=caffe)

    ''' Load trained geometric matching model '''
    if do_aff:
        args.model = args.model_aff
    if do_tps:
        args.model = args.model_tps
    GM_cp_name = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, args.model)
    if not os.path.exists(GM_cp_name):
        raise Exception('There is no pre-trained geometric matching model, i.e. ' + GM_cp_name)
    print('Load geometric matching model {}'.format(GM_cp_name))
    GM_checkpoint = torch.load(GM_cp_name, map_location=lambda storage, loc: storage)
    model.load_state_dict(GM_checkpoint['state_dict'])

    # GM_checkpoint['state_dict'] = OrderedDict(
    #     [(k.replace('model', 'GM_base'), v) for k, v in GM_checkpoint['state_dict'].items()])
    # GM_checkpoint['state_dict'] = OrderedDict(
    #     [(k.replace('FeatureRegression', 'ThetaRegression'), v) for k, v in GM_checkpoint['state_dict'].items()])
    # model.load_state_dict(GM_checkpoint['state_dict'], strict=False)

    print('Load geometric matching model successfully!')

    if args.cuda:
        model.cuda()

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
    csv_file_test, test_dataset_path = get_dataset_csv(dataset_path=args.test_dataset_path, dataset=args.test_dataset, subset='test')
    # csv_file_test, test_dataset_path = get_dataset_csv(dataset_path=args.test_dataset_path, dataset=args.test_dataset, subset='watch')
    print(csv_file_test)
    output_size = (args.image_size, args.image_size)
    normalize = NormalizeImageDict(['source_image', 'target_image'])
    # normalize = None
    dataset = TestDataSet(csv_file=csv_file_test, dataset_path=test_dataset_path, output_size=output_size, normalize=normalize)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    ''' Test trained geometric matching model '''
    print('Test on {}'.format(args.test_dataset))
    model.eval()
    with torch.no_grad():
        results, _ = test_fn(model=model, metric=metric, batch_size=args.batch_size, dataset=dataset, dataloader=dataloader, do_aff=do_aff, do_tps=do_tps, args=args)

    print('Done!')