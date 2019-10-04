# ========================================================================================
# Train geometric matching model based on the object detection by fasterRCNN
# Author: Jingwei Qu
# Date: 05 Mar 2019
# ========================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from visdom import Visdom
from collections import OrderedDict
import os
import time

from geometric_matching.arguments.arguments_setting import Arguments
from geometric_matching.gm_model.dual_geometric_matching import DualGeometricMatching
from geometric_matching.gm_model.geometric_matching_cycle import GeometricMatching
from geometric_matching.gm_model.loss import TransformedGridLoss, CycleLoss
from geometric_matching.gm_data.train_dataset import TrainDataset
from geometric_matching.gm_data.pf_pascal_dataset import PFPASCALDataset
from geometric_matching.gm_data.watch_dataset import WatchDataset

from geometric_matching.gm_data.train_triple import TrainTriple
from geometric_matching.image.normalization import NormalizeImageDict
from geometric_matching.util.train_fn_cycle import train_fn
from geometric_matching.util.test_fn import test_fn
from geometric_matching.util.vis_fn import vis_fn
from geometric_matching.util.test_watch import test_watch
from geometric_matching.util.net_util import get_dataset_csv, save_checkpoint

from model.utils.config import cfg, cfg_from_file, cfg_from_list

# matplotlib.use('Qt5Agg')

if __name__ == '__main__':
    print('Watch GeometricMatching Feature Map & Correlation')

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

    ''' Initialize geometric matching model '''
    print('Initialize geometric matching model')
    do_aff = args.geometric_model == 'affine'
    do_tps = args.geometric_model == 'tps'
    pytorch = True
    caffe = False
    fixed_blocks = 3
    # Create geometric_matching model
    # Default: args.geometric_model - tps, args.feature_extraction_cnn - vgg
    if do_aff:
        output_dim = 6
    if do_tps:
        output_dim = 18
    model = GeometricMatching(output_dim=output_dim, **arg_groups['model'], fixed_blocks=fixed_blocks, pytorch=pytorch, caffe=caffe)

    if args.cuda:
        model.cuda()

    # Default is grid loss (as described in the CVPR 2017 paper)
    if args.use_mse_loss:
        print('Use MSE loss')
        loss = nn.MSELoss()
    else:
        print('Use grid loss')
        loss = TransformedGridLoss(geometric_model=args.geometric_model, use_cuda=args.cuda)
        loss_cycle = CycleLoss(geometric_model=args.geometric_model, use_cuda=args.cuda)

    # Optimizer
    # Only regression part needs training
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr)
    print('Learning rate: {}'.format(args.lr))
    print('Num of epochs: {}'.format(args.num_epochs))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.requires_grad)

    ''' Set training dataset and validation dataset '''
    # Set path of csv files including image names (source and target) and pre-set random tps
    # if args.geometric_model == 'tps':
    #     print(args.random_t_tps)
    #     csv_file_train, train_dataset_path = get_dataset_csv(dataset_path=args.train_dataset_path, dataset=args.train_dataset, subset='train', geometric_model=args.geometric_model, random_t_tps=args.random_t_tps)
    # elif args.geometric_model == 'affine':
    # csv_file_train, train_dataset_path = get_dataset_csv(dataset_path=args.train_dataset_path, dataset=args.train_dataset, subset='train', geometric_model=args.geometric_model)
    csv_file_train, train_dataset_path = get_dataset_csv(dataset_path=args.train_dataset_path, dataset=args.train_dataset, subset='train', geometric_model=args.geometric_model)
    print('Train csv file: {}'.format(csv_file_train))
    csv_file_val, eval_dataset_path = get_dataset_csv(dataset_path=args.eval_dataset_path, dataset=args.eval_dataset, subset='val')
    print('Val csv file: {}'.format(csv_file_val))
    output_size = (args.image_size, args.image_size)
    # Train dataset
    normalize = NormalizeImageDict(['source_image', 'target_image'])
    # normalize = None
    print('Whether generate random gt transformation: {}'.format(args.random_sample))
    dataset = TrainDataset(csv_file=csv_file_train, dataset_path=train_dataset_path, output_size=output_size,
                           geometric_model=args.geometric_model, random_sample=args.random_sample, normalize=normalize,
                           random_t_tps=args.random_t_tps, random_crop=args.random_crop)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    triple_generation = TrainTriple(geometric_model=args.geometric_model, output_size=output_size, use_cuda=args.cuda, normalize=normalize)
    # Val dataset
    dataset_val = PFPASCALDataset(csv_file=csv_file_val, dataset_path=eval_dataset_path, output_size=output_size, normalize=normalize)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Watching images dataset
    # csv_file_watch = os.path.join(args.eval_dataset_path, 'watch_images.csv')
    # dataset_watch = WatchDataset(csv_file=csv_file_watch, dataset_path=args.eval_dataset_path, output_size=output_size, normalize=normalize)
    # dataloader_watch = torch.utils.data.DataLoader(dataset_watch, batch_size=1, shuffle=False, num_workers=4)
    csv_file_watch, watch_dataset_path = get_dataset_csv(dataset_path=args.eval_dataset_path, dataset=args.eval_dataset, subset='watch')
    print('Watch csv file: {}'.format(csv_file_watch))
    dataset_watch = PFPASCALDataset(csv_file=csv_file_watch, dataset_path=watch_dataset_path, output_size=output_size, normalize=normalize)
    dataloader_watch = torch.utils.data.DataLoader(dataset_watch, batch_size=1, shuffle=False, num_workers=4)

    ''' Train and val geometric matching model '''
    # Define checkpoint name
    # checkpoint_suffix = '_' + args.feature_extraction_cnn
    checkpoint_suffix = '_' + args.geometric_model
    # checkpoint_suffix += '_' + args.train_dataset
    # if args.geometric_model == 'tps':
    #     checkpoint_suffix += '_' + str(args.random_t_tps)
    # checkpoint_name = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, args.trained_models_fn + checkpoint_suffix + '.pth.tar')
    checkpoint_name = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, 'finetune_gm' + checkpoint_suffix + '_cycle.pth.tar')
    print('Checkpoint saving name: {}'.format(checkpoint_name))

    # Set visualization
    vis = Visdom(env='TPSCycle')

    print('Starting training')
    train_loss = np.zeros(args.num_epochs)
    val_pck = np.zeros(args.num_epochs)
    best_val_pck = float('-inf')
    train_lr = np.zeros(args.num_epochs)
    train_time = np.zeros(args.num_epochs)
    val_time = np.zeros(args.num_epochs)
    best_epoch = 0
    if args.resume:
        args.start_epoch = GM_checkpoint['epoch']
        best_val_pck = GM_checkpoint['best_val_pck']
        train_loss = GM_checkpoint['train_loss']
        val_pck = GM_checkpoint['val_pck']
        train_time = GM_checkpoint['train_time']
        val_time = GM_checkpoint['val_time']

    model.FeatureExtraction.eval()
    model.ThetaRegression.eval()
    with torch.no_grad():
        test_fn(model=model, metric='pck', batch_size=args.batch_size, dataset=dataset_val, dataloader=dataloader_val, do_aff=do_aff, do_tps=do_tps, args=args)
        results_watch, theta_watch, theta_watch_inver, _ = test_watch(model=model, metric='pck', batch_size=1,
                                                                      dataset=dataset_watch, dataloader=dataloader_watch,
                                                                      do_aff=do_aff, do_tps=do_tps, args=args)
        vis_fn(vis=vis, train_loss=train_loss, val_pck=val_pck, train_lr=train_lr, epoch=0, num_epochs=args.num_epochs,
               dataloader=dataloader_watch, theta=theta_watch, theta_inver=theta_watch_inver, results=results_watch,
               geometric_model=args.geometric_model, use_cuda=True)

    start = time.time()
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        # model.train()
        model.ThetaRegression.train()
        train_loss[epoch-1], train_time[epoch-1] = train_fn(epoch=epoch, model=model, loss_fn=loss, loss_cycle_fn=loss_cycle,
                                                            optimizer=optimizer, dataloader=dataloader,
                                                            triple_generation=triple_generation,
                                                            geometric_model=args.geometric_model, use_cuda=args.cuda,
                                                            log_interval=100, vis=vis)
        # model.eval()
        model.ThetaRegression.eval()
        results, val_time[epoch-1] = test_fn(model=model, metric='pck', batch_size=args.batch_size, dataset=dataset_val,
                                             dataloader=dataloader_val, do_aff=do_aff, do_tps=do_tps, args=args)

        if do_aff:
            val_pck[epoch - 1] = np.mean(results['aff']['pck'])
        elif do_tps:
            val_pck[epoch - 1] = np.mean(results['tps']['pck'])

        train_lr[epoch - 1] = optimizer.param_groups[0]['lr']

        # Visualization
        if epoch % 5 == 0 or epoch == 1:
            with torch.no_grad():
                results_watch, theta_watch, theta_watch_inver, _ = test_watch(model=model, metric='pck', batch_size=1,
                                                                              dataset=dataset_watch,
                                                                              dataloader=dataloader_watch,
                                                                              do_aff=do_aff, do_tps=do_tps, args=args)
                vis_fn(vis=vis, train_loss=train_loss, val_pck=val_pck, train_lr=train_lr, epoch=epoch,
                       num_epochs=args.num_epochs, dataloader=dataloader_watch, theta=theta_watch,
                       theta_inver=theta_watch_inver, results=results_watch, geometric_model=args.geometric_model, use_cuda=True)

        is_best = val_pck[epoch-1] > best_val_pck
        best_val_pck = max(val_pck[epoch-1] , best_val_pck)
        if is_best:
            best_epoch = epoch
        print('Save checkpoint...')

    end = time.time()
    print('Best epoch: {}\t\tBest val pck: {:.2%}\t\tTime cost (total): {:.4f}'.format(best_epoch, best_val_pck, end - start))

    print('Done!')