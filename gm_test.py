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
from geometric_matching.gm_model.geometric_matching import GeometricMatching
from geometric_matching.gm_model.dual_geometric_matching import DualGeometricMatching
from geometric_matching.image.normalization import *
from geometric_matching.gm_data.pf_willow_dataset2 import PFWILLOWDataset2
from geometric_matching.gm_data.pf_pascal_dataset2 import PFPASCALDataset2
from geometric_matching.gm_data.caltech_dataset2 import CaltechDataset2
from geometric_matching.gm_data.tss_dataset2 import TSSDataset2
from geometric_matching.gm_data.watch_dataset2 import WatchDataset2
from geometric_matching.gm_data.pf_willow_dataset import PFWILLOWDataset
from geometric_matching.gm_data.pf_pascal_dataset import PFPASCALDataset
from geometric_matching.gm_data.caltech_dataset import CaltechDataset
from geometric_matching.gm_data.tss_dataset import TSSDataset
from geometric_matching.util.net_util import *
from geometric_matching.util.test_fn import *
from geometric_matching.util.vis_feature import vis_feature
from geometric_matching.util.dataloader import default_collate

from model.utils.config import cfg, cfg_from_file, cfg_from_list

# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt

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
    # Flag for testing dual (affine+tps), affine, tps
    do_aff = args.model_aff != ''
    do_tps = args.model_tps != ''
    dual = args.model != '' or (do_aff and do_tps)
    # Create geometric_matching model
    if dual:
        # Crop object from image ('img'), feature map of vgg pool4 ('pool4'), feature map of vgg conv1 ('conv1'), or no cropping (None)
        # Feature extraction network: 1. pre-trained on ImageNet; 2. fine-tuned on PascalVOC2011, arg_groups['model']['pretrained'] = pretrained
        # 20 object classes in PascalVOC (plus '__background__')
        pascal_category = np.asarray(['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                                      'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                                      'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])
        model = DualGeometricMatching(aff_output_dim=6, tps_output_dim=18, dataset_classes=pascal_category,
                                      class_agnostic=args.class_agnostic, thresh=0.05, max_per_image=50, crop_layer=None,
                                      use_cuda=args.cuda, **arg_groups['model'], pytorch=False, caffe=False)
    else:
        if do_aff:
            output_dim = 6
        if do_tps:
            output_dim = 18
        model = GeometricMatching(output_dim=output_dim, **arg_groups['model'], pytorch=False, caffe=False)

    ''' Load trained geometric matching model '''
    # if (dual and args.model != '') or not dual:
    #     if do_aff:
    #         args.model = args.model_aff
    #     if do_tps:
    #         args.model = args.model_tps
    #     # GM_cp_name = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, args.model)
    #     GM_cp_name = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, 'fixed', args.model)
    #     if not os.path.exists(GM_cp_name):
    #         raise Exception('There is no pre-trained geometric matching model, i.e. ' + GM_cp_name)
    #     print('Load geometric matching model {}'.format(GM_cp_name))
    #     GM_checkpoint = torch.load(GM_cp_name, map_location=lambda storage, loc: storage)

        # GM_checkpoint['state_dict'] = OrderedDict(
        #     [(k.replace('model_1', 'GM_base_1'), v) for k, v in GM_checkpoint['state_dict'].items()])
        # GM_checkpoint['state_dict'] = OrderedDict(
        #     [(k.replace('model_2', 'GM_base_2'), v) for k, v in GM_checkpoint['state_dict'].items()])
        # for name, param in model.FeatureExtraction.state_dict().items():
        #     model.FeatureExtraction.state_dict()[name].copy_(
        #         GM_checkpoint['state_dict']['FeatureExtraction.' + name])
        # for name, param in model.ThetaRegression.state_dict().items():
        #     model.ThetaRegression.state_dict()[name].copy_(GM_checkpoint['state_dict']['ThetaRegression.' + name])
        # for name, param in model.ThetaRegression2.state_dict().items():
        #     model.ThetaRegression2.state_dict()[name].copy_(GM_checkpoint_tps['state_dict']['ThetaRegression.' + name])
        # model.load_state_dict(GM_checkpoint['state_dict'])

        # print('Load geometric matching model successfully!')

    if dual and args.model == '':
        ''' Set FasterRCNN module '''
        if args.net == 'vgg16' or args.net == 'res101':
            print('Initialize fasterRCNN module')
        else:
            print('FasterRCNN model is not defined')
            pdb.set_trace()
        # The directory for loading pre-trained fasterRCNN for extracting bounding box of objects
        RCNN_cp_dir = os.path.join(args.load_dir, args.net, args.dataset)
        RCNN_cp_name = os.path.join(RCNN_cp_dir, 'best_faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print('Load fasterRCNN model {}'.format(RCNN_cp_name))
        if not os.path.exists(RCNN_cp_name):
            raise Exception('There is no pre-trained fasterRCNN model, i.e. {}'.format(RCNN_cp_name))
        # Load pre-trained parameters for fasterRCNN
        RCNN_checkpoint = torch.load(RCNN_cp_name, map_location=lambda storage, loc: storage)
        for name, param in model.FasterRCNN.state_dict().items():
            model.FasterRCNN.state_dict()[name].copy_(RCNN_checkpoint['model'][name])
        # model.FasterRCNN.load_state_dict(RCNN_checkpoint['model'])
        for param in model.FasterRCNN.parameters():
            param.requires_grad = False
        if 'pooling_mode' in RCNN_checkpoint.keys():
            cfg.POOLING_MODE = RCNN_checkpoint['pooling_mode']
        print('Load fasterRCNN model successfully!')
        if args.cuda:
            cfg.CUDA = True

        ''' Set ThetaRegression module '''
        GM_cp_name_aff = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, args.model_aff)
        if not os.path.exists(GM_cp_name_aff):
            raise Exception('There is no pre-trained geometric matching affine model, i.e. ' + GM_cp_name_aff)
        print('Load geometric matching affine model {}'.format(GM_cp_name_aff))
        GM_checkpoint_aff = torch.load(GM_cp_name_aff, map_location=lambda storage, loc: storage)

        GM_cp_name_tps = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, args.model_tps)
        if not os.path.exists(GM_cp_name_aff):
            raise Exception('There is no pre-trained geometric matching tps model, i.e. ' + GM_cp_name_tps)
        print('Load geometric matching tps model {}'.format(GM_cp_name_tps))
        GM_checkpoint_tps = torch.load(GM_cp_name_tps, map_location=lambda storage, loc: storage)
        # GM_checkpoint['state_dict'] = OrderedDict(
        #     [(k.replace('vgg', 'model'), v) for k, v in GM_checkpoint['state_dict'].items()])
        # GM_checkpoint['state_dict'] = OrderedDict(
        #     [(k.replace('FeatureRegression', 'ThetaRegression'), v) for k, v in GM_checkpoint['state_dict'].items()])
        for name, param in model.FeatureExtraction.state_dict().items():
            model.FeatureExtraction.state_dict()[name].copy_(GM_checkpoint_aff['state_dict']['FeatureExtraction.' + name])
        for name, param in model.ThetaRegression.state_dict().items():
            model.ThetaRegression.state_dict()[name].copy_(GM_checkpoint_aff['state_dict']['ThetaRegression.' + name])
        for name, param in model.ThetaRegression2.state_dict().items():
            model.ThetaRegression2.state_dict()[name].copy_(GM_checkpoint_tps['state_dict']['ThetaRegression.' + name])
        print('Load geometric matching affine & tps model successfully!')

    if args.cuda:
        model.cuda()

    ''' Set testing dataset & evaluation metric '''
    if args.test_dataset == 'PF-WILLOW':
        # TestDataSet = PFWILLOWDataset
        TestDataSet = PFWILLOWDataset2
        metric = 'pck'
    elif args.test_dataset == 'PF-PASCAL':
        # TestDataSet = PFPASCALDataset
        TestDataSet = PFPASCALDataset2
        metric = 'pck'
    elif args.test_dataset == 'Caltech-101':
        # TestDataSet = CaltechDataset
        TestDataSet = CaltechDataset2
        metric = 'area'
    elif args.test_dataset == 'TSS':
        # TestDataSet = TSSDataset
        TestDataSet = TSSDataset2
        metric = 'flow'
    collate_fn = default_collate

    # Set path of csv file including image names (source and target) and annotation
    csv_file_test, test_dataset_path = get_dataset_csv(dataset_path=args.test_dataset_path, dataset=args.test_dataset, subset='test')
    print(csv_file_test)
    output_size = (args.image_size, args.image_size)
    normalize = NormalizeImageDict(['source_image', 'target_image'])
    # normalize = None
    dataset = TestDataSet(csv_file=csv_file_test, dataset_path=test_dataset_path, output_size=output_size, normalize=normalize)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Watching images dataset
    csv_file_watch = os.path.join(args.test_dataset_path, 'watch_images.csv')
    dataset_watch = WatchDataset2(csv_file=csv_file_watch, dataset_path=args.test_dataset_path, output_size=output_size,
                                  normalize=normalize)
    dataloader_watch = torch.utils.data.DataLoader(dataset_watch, batch_size=1, shuffle=False, num_workers=4)

    ''' Test trained geometric matching model '''
    print('Test on {}'.format(args.test_dataset))
    # vis = Visdom(env='feature')
    if args.test_dataset == 'PF-PASCAL':
        best_pck = float('-inf')
        best_epoch = 0
        for epoch in range(7, 51):
            if (dual and args.model != '') or not dual:
                if do_aff:
                    args.model = args.model_aff
                if do_tps:
                    args.model = args.model_tps
                GM_cp_name = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, 'fixed', str(epoch) + '_gm_tps_0.4.pth.tar')
                # GM_cp_name = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, '20190708', args.model)
                if not os.path.exists(GM_cp_name):
                    raise Exception('There is no pre-trained geometric matching model, i.e. ' + GM_cp_name)
                print('Load geometric matching model {}'.format(GM_cp_name))
                GM_checkpoint = torch.load(GM_cp_name, map_location=lambda storage, loc: storage)
                model.load_state_dict(GM_checkpoint['state_dict'])
                print('Load geometric matching model successfully!')
            model.eval()
            with torch.no_grad():
                results, _ = test_fn(model=model, metric=metric, dataset=dataset, dataloader=dataloader, dual=dual, do_aff=do_aff, do_tps=do_tps, args=args)

            if dual:
                pck = np.mean(results['aff_tps']['pck'])
            else:
                if do_aff:
                    pck = np.mean(results['aff']['pck'])
                elif do_tps:
                    pck = np.mean(results['tps']['pck'])

            if pck > best_pck:
                best_pck = pck
                best_epoch = epoch
        print('Best epoch: {}\t\tBest pck: {:.2%}'.format(best_epoch, best_pck))

    else:
        model.eval()
        with torch.no_grad():
            results = test_fn(model=model, metric=metric, dataset=dataset, dataloader=dataloader, dual=dual, do_aff=do_aff, do_tps=do_tps, args=args)
            # vis_feature(vis=vis, model=model, dataloader=dataloader_watch, use_cuda=args.cuda)

    print('Done!')