# ====================================================================================================
# Test geometric matching model on PF-WILLOW dataset based on the object detection by faster rcnn
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

from geometric_matching.model.geometric_matching import GeometricMatching
from geometric_matching.image.normalization import *
from geometric_matching.data.pf_willow_dataset import PFWILLOWDataset
from geometric_matching.data.pf_pascal_dataset import PFPASCALDataset
from geometric_matching.util.net_util import *
from geometric_matching.util.test_fn import *
from geometric_matching.util.eval_util import *

from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list
from lib.model.faster_rcnn.vgg16 import vgg16

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a geometric matching network based on predicted rois by a faster rcnn')
    """ Arguments for the fasterRCNN """
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='pascal_voc_2011', type=str)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net', help='vgg16, res50, res101, res152', default='vgg16', type=str)
    parser.add_argument('--set', dest='set_cfgs', help='set config keys', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir', help='directory to load models', default="models", type=str)
    parser.add_argument('--cuda', dest='cuda', help='whether use CUDA', action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs', help='whether use multiple GPUs', action='store_true')
    parser.add_argument('--cag', dest='class_agnostic', help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling', default=0,
                        type=int)
    parser.add_argument('--checksession', dest='checksession', help='checksession to load model', default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch', help='checkepoch to load network', default=7, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint to load network', default=23079, type=int)

    """ Arguments for the geometric matching model """
    parser.add_argument('--num-epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='training batch size')
    parser.add_argument('--trained-models-dir', type=str,
                        default='geometric_matching/trained_models/PF-PASCAL/loss_1/image/identity/random_t_tps_0.4_a/',
                        help='Path to trained models folder')
    parser.add_argument('--feature-extraction-cnn', type=str, default='vgg',
                        help='Feature extraction architecture: vgg/resnet101')
    parser.add_argument('--testing-dataset', type=str, default='PF-WILLOW',
                        help='Dataset to use for testing: PF-WILLOW/PF-PASCAL')
    parser.add_argument('--testing-dataset-path', type=str, default='geometric_matching/testing_data',
                        help='Path to folder containing training dataset')
    parser.add_argument('--pck-alpha', type=float, default=0.1, help='pck margin factor alpha')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    ''' Parse arguments '''
    args = parse_args()

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)

    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    args.cfg_file = "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # print('Using config:')
    # pprint.pprint(cfg)

    ''' Load trained fasterRCNN model '''
    # 20 object classes in PascalVOC (plus '__background__')
    dataset_classes = np.asarray(['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                                  'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])
    if args.net == 'vgg16':
        fasterRCNN = vgg16(dataset_classes, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    # The directory for loading pre-trained faster rcnn for extracting rois
    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    load_name = os.path.join(input_dir,
                             'best_faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch,
                                                                    args.checkpoint))
    if not os.path.exists(load_name):
        raise Exception('There is no pre-trained model for extracting rois, i.e. ' + load_name)

    # Load parameters for the faster rcnn
    print("Load faster rcnn checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('Load faster rcnn model successfully!')

    if args.cuda:
        cfg.CUDA = True
        fasterRCNN.cuda()

    ''' Set testing dataset '''
    if args.testing_dataset == 'PF-WILLOW':
        TestDataSet = PFWILLOWDataset
    elif args.testing_dataset == 'PF-PASCAL':
        TestDataSet = PFPASCALDataset

    args.testing_dataset_path = os.path.join(args.testing_dataset_path, args.testing_dataset)
    csv_file_test = os.path.join(args.testing_dataset_path, 'test_' + args.testing_dataset + '.csv')
    output_size = (240, 240)
    normalize = NormalizeImageDict(['source_image', 'target_image'])
    dataset = TestDataSet(csv_file=csv_file_test, dataset_path=args.testing_dataset_path, output_size=output_size,
                          transform=normalize)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    ''' Set evaluation metric'''
    if args.testing_dataset == 'PF-WILLOW' or args.testing_dataset == 'PF-PASCAL':
        metric = 'pck'

    ''' Load trained geometric matching model '''
    # Crop object from image ('object'), feature map of vgg pool4 ('pool4'), feature map of vgg conv1 ('conv1'),
    # or no cropping ('image')
    crop_layer = 'image'
    # Feature extraction network: 1. pre-trained vgg on ImageNet; 2. fine-tuned vgg on PascalVOC2011
    pretrained = True
    # Initialize theta_regression module as identity mapping
    init_identity = True
    # Transform image with affine parameters before computing PCK
    with_affine = True
    # Create geometric_matching model
    print('Creating CNN model...')
    # Default: args.geometric_model - tps, args.feature_extraction_cnn - vgg
    model = GeometricMatching(use_cuda=args.cuda, geometric_model='tps',
                              feature_extraction_cnn=args.feature_extraction_cnn, pretrained=pretrained,
                              crop_layer=crop_layer, init_identity=init_identity)

    best_PCK_tps = 0
    most_correct_points_tps = 0
    best_epoch = 0
    start = time.time()

    # '''
    # for epoch in range(1, args.num_epochs + 1):
    for epoch in range(7, 8):
        # checkpoint_name = os.path.join(args.trained_models_dir,
        #                                str(epoch) + '_checkpoint_adam_tps_grid_loss_vgg.pth.tar')
        checkpoint_name = os.path.join(args.trained_models_dir,
                                       'best_' + str(epoch) + '_checkpoint_adam_tps_grid_loss_vgg.pth.tar')
        # checkpoint_name = os.path.join(args.trained_models_dir,
        #                                'best_pascal_checkpoint_adam_tps_grid_loss.pth.tar')
        # Load trained parameters for the geometric_matching model
        if not os.path.exists(checkpoint_name):
            raise Exception('There is no pre-trained geometric_matching model, i.e. ' + checkpoint_name)
        print('Load trained geometric_matching checkpoint {}'.format(checkpoint_name))
        checkpoint_gm = torch.load(checkpoint_name)

        checkpoint_gm['state_dict'] = OrderedDict(
            [(k.replace('vgg', 'model'), v) for k, v in
             checkpoint_gm['state_dict'].items()])
        checkpoint_gm['state_dict'] = OrderedDict(
            [(k.replace('FeatureRegression', 'ThetaRegression'), v) for k, v in
             checkpoint_gm['state_dict'].items()])

        model.load_state_dict(checkpoint_gm['state_dict'])
        print('Load trained geometric_matching model successfully!')

        if args.cuda:
            model.cuda()

        ''' Test the geometric matching model '''
        # Test model on PF-WILLOW dataset
        fasterRCNN.eval()
        model.eval()
        PCK_tps, total_correct_points_tps, total_points = test_pf_willow(model, fasterRCNN, dataloader,
                                                                         use_cuda=args.cuda, crop_layer=crop_layer,
                                                                         with_affine=with_affine)
        # stats = compute_metric(metric, model, fasterRCNN, dataset, dataloader, args)
        # if PCK_tps > best_PCK_tps:
        #     best_PCK_tps = PCK_tps
        #     most_correct_points_tps = total_correct_points_tps
        #     best_epoch = epoch
    # '''

    # stats = compute_metric(metric, model, dataset, dataloader, batch_tnf, batch_size, two_stage, do_aff, do_tps, args)

    end = time.time()
    # print(
    #     'Best epoch {:}\t\tBest PCK tps {:.4f}\t\tCorrect points {:}\tTotal points {:}\t\tTime cost (total) {:.4f}'.format(
    #         best_epoch, best_PCK_tps, most_correct_points_tps, total_points, end - start))
    print('Done!')