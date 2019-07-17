# ========================================================================================
# Train geometric matching model based on the object detection by fasterRCNN
# Author: Jingwei Qu
# Date: 05 Mar 2019
# ========================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse
import pprint
import time
import shutil
import pandas as pd

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pdb
from collections import OrderedDict
from visdom import Visdom
import cv2

from geometric_matching.arguments.arguments_setting import Arguments
from geometric_matching.util.net_util import get_dataset_csv, save_checkpoint, str_to_bool
from geometric_matching.gm_model.dual_geometric_matching import DualGeometricMatching
from geometric_matching.gm_model.geometric_matching import GeometricMatching
from geometric_matching.gm_model.loss import TransformedGridLoss
from geometric_matching.gm_data.train_dataset import TrainDataset
# from geometric_matching.gm_data.pf_pascal_dataset import PFPASCALDataset
# from geometric_matching.gm_data.watch_dataset import WatchDataset

from geometric_matching.gm_data.pf_pascal_dataset2 import PFPASCALDataset2
from geometric_matching.gm_data.watch_dataset2 import WatchDataset2

from geometric_matching.gm_data.train_triple import TrainTriple
from geometric_matching.image.normalization import NormalizeImageDict
from geometric_matching.util.train_fn import *
from geometric_matching.util.test_fn import *
from geometric_matching.geotnf.transformation import GeometricTnf
from geometric_matching.util.vis_fn import vis_fn

from model.utils.config import cfg, cfg_from_file, cfg_from_list

import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # print('Use GPU: {}-{}'.format(torch.cuda.get_device_name(torch.cuda.current_device()), torch.cuda.current_device()))

    print('Train GeometricMatching using weak supervision')

    ''' Load arguments '''
    args, arg_groups = Arguments(mode='train').parse()
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
    do_aff = args.geometric_model == 'affine'
    do_tps = args.geometric_model == 'tps'
    dual = (args.model_aff != '' and args.model_tps != '') or args.model != ''
    # resume = (args.model_aff != '') ^ (args.model_tps != '') ^ (args.model != '')
    pytorch = False
    caffe = False
    fixed_blocks = 2
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
                                      use_cuda=args.cuda, **arg_groups['model'], pytorch=pytorch, caffe=caffe)
    else:
        # Default: args.geometric_model - tps, args.feature_extraction_cnn - vgg
        if do_aff:
            output_dim = 6
        if do_tps:
            output_dim = 18
        model = GeometricMatching(output_dim=output_dim, **arg_groups['model'], fixed_blocks=fixed_blocks, pytorch=pytorch, caffe=caffe)

    ''' Set geometric matching model '''
    # if dual and not resume:
    if dual:
        # Train geometric matching model with computed affine, load pretrained model
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
        if not os.path.exists(GM_cp_name_tps):
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

    ''' Resume training geometric matching model '''
    # If resume training, load interrupted model
    # if resume:
    #     print('Resume training')
    #     if do_aff:
    #         args.model = args.model_aff
    #     if do_tps:
    #         args.model = args.model_tps
    #     GM_cp_name = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, args.model)
    #     if not os.path.exists(GM_cp_name):
    #         raise Exception('There is no pre-trained geometric matching model, i.e. ' + GM_cp_name)
    #     print('Load geometric matching model {}'.format(GM_cp_name))
    #     GM_checkpoint = torch.load(GM_cp_name, map_location=lambda storage, loc: storage)
    #     model.load_state_dict(GM_checkpoint['state_dict'])
    #     print('Load geometric matching model successfully!')

    print('Finetune')
    if do_aff:
        args.model = args.model_aff
    if do_tps:
        args.model = args.model_tps
    # GM_cp_name = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, args.model)
    GM_cp_name = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, 'fixed', args.model)
    if not os.path.exists(GM_cp_name):
        raise Exception('There is no pre-trained geometric matching model, i.e. ' + GM_cp_name)
    print('Load geometric matching model {}'.format(GM_cp_name))
    GM_checkpoint = torch.load(GM_cp_name, map_location=lambda storage, loc: storage)
    model.load_state_dict(GM_checkpoint['state_dict'])
    print('Load geometric matching model successfully!')

    if args.cuda:
        model.cuda()

    # Default is grid loss (as described in the CVPR 2017 paper)
    if args.use_mse_loss:
        print('Use MSE loss')
        loss = nn.MSELoss()
    else:
        print('Use grid loss')
        loss = TransformedGridLoss(geometric_model=args.geometric_model, use_cuda=args.cuda)

    # Optimizer
    # Only regression part needs training
    # optimizer = optim.Adam(model.ThetaRegression.parameters(), lr=args.lr)
    if dual:
        args.lr = 5e-8
        args.num_epochs = 10
        for param in model.FasterRCNN.parameters():
            param.requires_grad = False
    # optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr)
    optimizer = optim.Adam(model.ThetaRegression.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    print(args.lr)
    # if resume:
    #     optimizer.load_state_dict(GM_checkpoint['optimizer'])
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    ''' Set training dataset and validation dataset '''
    # Set path of csv files including image names (source and target) and pre-set random tps
    if args.geometric_model == 'tps':
        csv_file_train, train_dataset_path = get_dataset_csv(dataset_path=args.train_dataset_path, dataset=args.train_dataset, subset='train', random_t_tps=args.random_t_tps)
    elif args.geometric_model == 'affine':
        csv_file_train, train_dataset_path = get_dataset_csv(dataset_path=args.train_dataset_path, dataset=args.train_dataset, subset='train', geometric_model=args.geometric_model)
    # print(csv_file_train)
    csv_file_val, eval_dataset_path = get_dataset_csv(dataset_path=args.eval_dataset_path, dataset=args.eval_dataset, subset='val')
    print(csv_file_val)
    output_size = (args.image_size, args.image_size)
    # Train dataset
    normalize = NormalizeImageDict(['source_image', 'target_image'])
    # normalize = None
    print(args.random_t_tps)
    dataset = TrainDataset(csv_file=csv_file_train, dataset_path=train_dataset_path, output_size=output_size,
                           geometric_model=args.geometric_model, random_sample=args.random_sample, normalize=normalize,
                           random_t_tps=args.random_t_tps, random_crop=args.random_crop)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    triple_generation = TrainTriple(geometric_model=args.geometric_model, output_size=output_size, use_cuda=args.cuda, normalize=normalize)
    # Val dataset
    dataset_val = PFPASCALDataset2(csv_file=csv_file_val, dataset_path=eval_dataset_path, output_size=output_size, normalize=normalize)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Watching images dataset
    csv_file_watch = os.path.join(args.eval_dataset_path, 'watch_images.csv')
    dataset_watch = WatchDataset2(csv_file=csv_file_watch, dataset_path=args.eval_dataset_path, output_size=output_size, normalize=normalize)
    dataloader_watch = torch.utils.data.DataLoader(dataset_watch, batch_size=1, shuffle=False, num_workers=4)

    ''' Train and val geometric matching model '''
    # Define checkpoint name
    # checkpoint_suffix = '_' + args.feature_extraction_cnn
    if dual:
        checkpoint_suffix = '_aff_tps'
    else:
        checkpoint_suffix = '_' + args.geometric_model
    # checkpoint_suffix += '_' + args.train_dataset
    if args.geometric_model == 'tps':
        checkpoint_suffix += '_' + str(args.random_t_tps)
    # checkpoint_name = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, args.trained_models_fn + checkpoint_suffix + '.pth.tar')
    checkpoint_name = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, 'fixed', 'finetune_gm' + checkpoint_suffix + '.pth.tar')
    print('Checkpoint saving name: {}'.format(checkpoint_name))

    # Set visualization
    # vis = Visdom(env='third')
    # vis = Visdom(env='fourth')
    # vis = Visdom(env='fifth')
    # vis = Visdom(env='sixth')
    vis = Visdom(env='seventh')

    print('Starting training')
    train_loss = np.zeros(args.num_epochs)
    val_pck = np.zeros(args.num_epochs)
    best_val_pck = float('-inf')
    train_lr = np.zeros(args.num_epochs)
    train_time = np.zeros(args.num_epochs)
    val_time = np.zeros(args.num_epochs)
    best_epoch = 0
    # if resume:
    #     args.start_epoch = GM_checkpoint['epoch']
    #     best_val_pck = GM_checkpoint['best_val_pck']
    #     train_loss = GM_checkpoint['train_loss']
    #     val_pck = GM_checkpoint['val_pck']
    #     train_time = GM_checkpoint['train_time']
    #     val_time = GM_checkpoint['val_time']
    start = time.time()
    model.FeatureExtraction.eval()
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        # model.train()
        # model.FeatureExtraction.eval()
        # model.FeatureExtraction.GM_base[6].train()
        model.ThetaRegression.train()
        if dual:
            model.FasterRCNN.eval()
        train_loss[epoch-1], train_time[epoch-1] = train_fn(epoch=epoch, model=model, loss_fn=loss, optimizer=optimizer,
                                                            dataloader=dataloader, triple_generation=triple_generation,
                                                            geometric_model=args.geometric_model, dual=dual,
                                                            use_cuda=args.cuda, log_interval=100, vis=vis,
                                                            normalize=normalize)
        # model.eval()
        model.ThetaRegression.eval()
        results, val_time[epoch-1] = test_fn(model=model, metric='pck', dataset=dataset_val, dataloader=dataloader_val,
                                             dual=dual, do_aff=do_aff, do_tps=do_tps, args=args)

        if dual:
            val_pck[epoch - 1] = np.mean(results['aff_tps']['pck'])
        else:
            if do_aff:
                val_pck[epoch - 1] = np.mean(results['aff']['pck'])
            elif do_tps:
                val_pck[epoch - 1] = np.mean(results['tps']['pck'])

        train_lr[epoch - 1] = optimizer.param_groups[0]['lr']

        # Visualization
        if epoch % 5 == 0 or epoch == 1:
            vis_fn(vis=vis, model=model, train_loss=train_loss, val_pck=val_pck, train_lr=train_lr, epoch=epoch,
                   num_epochs=args.num_epochs, dataloader=dataloader_watch, geometric_model=args.geometric_model,
                   use_cuda=args.cuda, normalize=normalize)

        is_best = val_pck[epoch-1] > best_val_pck
        best_val_pck = max(val_pck[epoch-1] , best_val_pck)
        if is_best:
            best_epoch = epoch
        print('Save checkpoint...')
        # checkpoint_name = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, str(epoch) + '_gm' + checkpoint_suffix + '.pth.tar')
        # checkpoint_name = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, 'fixed', str(epoch) + '_gm' + checkpoint_suffix + '.pth.tar')
        # checkpoint_name = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, 'finetune', str(epoch) + '_gm' + checkpoint_suffix + '.pth.tar')
        # print('Checkpoint saving name: {}'.format(checkpoint_name))
        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_pck': best_val_pck,
            'train_loss': train_loss,
            'val_pck': val_pck,
            'train_time': train_time,
            'val_time': val_time,
        }, is_best, checkpoint_name)

        # Adjust learning rate every 10 epochs
        # scheduler.step()

    end = time.time()
    print('Best epoch: {}\t\tBest val pck: {:.2%}\t\tTime cost (total): {:.4f}'.format(best_epoch, best_val_pck, end - start))

    print('Done!')