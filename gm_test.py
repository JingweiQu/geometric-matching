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

from geometric_matching.arguments.arguments_setting import Arguments
from geometric_matching.model.geometric_matching import GeometricMatching
from geometric_matching.model.dual_geometric_matching import DualGeometricMatching
from geometric_matching.image.normalization import *
from geometric_matching.data.pf_willow_dataset import PFWILLOWDataset
from geometric_matching.data.pf_pascal_dataset import PFPASCALDataset
from geometric_matching.data.caltech_dataset import CaltechDataset
from geometric_matching.data.tss_dataset import TSSDataset
from geometric_matching.util.net_util import *
from geometric_matching.util.test_fn import *
from geometric_matching.util.dataloader import default_collate

from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list

# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt

if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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

    # Config for fasterRCNN
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '20']
    args.cfg_file = 'cfgs/{}.yml'.format(args.net)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    # print('Config for faster-rcnn:')
    # pprint.pprint(cfg)

    ''' Initialize geometric matching model '''
    print('Initialize geometric matching model')
    if args.geometric_model == 'affine':
        output_dim = 6
    elif args.geometric_model == 'tps':
        output_dim = 18
    if args.dual:
        if args.net == 'vgg16':
            print('Initialize fasterRCNN module')
        else:
            print('FasterRCNN model is not defined')
            pdb.set_trace()
        pascal_category = np.asarray(['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                                      'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                                      'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])
        model = DualGeometricMatching(output_dim=output_dim, dataset_classes=pascal_category,
                                      class_agnostic=args.class_agnostic, use_cuda=args.cuda, pretrained=True,
                                      thresh=0.05, max_per_image=50, crop_layer=None, **arg_groups['model'])
    else:
        model = GeometricMatching(output_dim=output_dim, use_cuda=args.cuda, pretrained=True, **arg_groups['model'])

    ''' Set FasterRCNN module '''
    # if args.dual:
    #     RCNN_cp_dir = os.path.join(args.load_dir, args.net, args.dataset)
    #     RCNN_cp_name = os.path.join(RCNN_cp_dir, 'best_faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    #     print('Load fasterRCNN model {}'.format(RCNN_cp_name))
    #     if not os.path.exists(RCNN_cp_name):
    #         raise Exception('There is no pre-trained fasterRCNN model, i.e. {}'.format(RCNN_cp_name))
    #     RCNN_checkpoint = torch.load(RCNN_cp_name, map_location=lambda storage, loc: storage)
    #     for name, param in model.FasterRCNN.state_dict().items():
    #         model.FasterRCNN.state_dict()[name].copy_(RCNN_checkpoint['model'][name])
    #     # model.FasterRCNN.load_state_dict(RCNN_checkpoint['model'])
    #     if 'pooling_mode' in RCNN_checkpoint.keys():
    #         cfg.POOLING_MODE = RCNN_checkpoint['pooling_mode']
    #     print('Load fasterRCNN model successfully!')
    #     if args.cuda:
    #         cfg.CUDA = True

    ''' Set ThetaRegression module '''
    GM_cp_name = os.path.join(args.trained_models_dir, args.model)
    if not os.path.exists(GM_cp_name):
        raise Exception('There is no pre-trained geometric_matching model, i.e. ' + GM_cp_name)
    print('Load geometric matching model {}'.format(GM_cp_name))
    GM_checkpoint = torch.load(GM_cp_name, map_location=lambda storage, loc: storage)
    # GM_checkpoint['state_dict'] = OrderedDict(
    #     [(k.replace('vgg', 'model'), v) for k, v in GM_checkpoint['state_dict'].items()])
    # GM_checkpoint['state_dict'] = OrderedDict(
    #     [(k.replace('FeatureRegression', 'ThetaRegression'), v) for k, v in GM_checkpoint['state_dict'].items()])
    # for name, param in model.FeatureExtraction.state_dict().items():
    #     model.FeatureExtraction.state_dict()[name].copy_(GM_checkpoint['state_dict']['FeatureExtraction.' + name])
    # for name, param in model.ThetaRegression.state_dict().items():
    #     model.ThetaRegression.state_dict()[name].copy_(GM_checkpoint['state_dict']['ThetaRegression.' + name])
    # for name, param in model.FasterRCNN.state_dict().items():
    #     model.FasterRCNN.state_dict()[name].copy_(GM_checkpoint['state_dict']['FasterRCNN.' + name])
    model.load_state_dict(GM_checkpoint['state_dict'])
    print('Load geometric matching model successfully!')

    if args.cuda:
        model.cuda()

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
    output_size = (args.image_size, args.image_size)
    normalize = NormalizeImageDict(['source_image', 'target_image'])
    dataset = TestDataSet(csv_file=csv_file_test, dataset_path=test_dataset_path, output_size=output_size,
                          transform=normalize)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    ''' Test trained geometric matching model '''
    print('Test on {}'.format(args.test_dataset))
    with torch.no_grad():
        model.eval()
        results = test_fn(model=model, metric=metric, dataset=dataset, dataloader=dataloader, args=args)
    print('Done!')