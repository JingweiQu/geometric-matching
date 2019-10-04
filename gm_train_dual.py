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
from geometric_matching.gm_model.loss_dual import TransformedGridLoss
from geometric_matching.gm_data.train_dataset import TrainDataset
from geometric_matching.gm_data.pf_pascal_dataset import PFPASCALDataset
from geometric_matching.gm_data.watch_dataset import WatchDataset

from geometric_matching.gm_data.train_dual_triple import TrainDualTriple
from geometric_matching.image.normalization import NormalizeImageDict
from geometric_matching.util.train_fn_dual import train_fn_dual
from geometric_matching.util.test_fn import test_fn
from geometric_matching.util.vis_fn_dual import vis_fn_dual
from geometric_matching.util.test_watch import test_watch
from geometric_matching.util.net_util import get_dataset_csv, save_checkpoint

# matplotlib.use('Qt5Agg')

if __name__ == '__main__':
    # print('Use GPU: {}-{}'.format(torch.cuda.get_device_name(torch.cuda.current_device()), torch.cuda.current_device()))

    print('Train DualGeometricMatching using weak supervision')

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

    ''' Initialize dual geometric matching model '''
    print('Initialize dual geometric matching model')
    # args.geometric_model = 'tps'
    # print(args.geometric_model)
    # Create dual_geometric_matching model
    # Crop object from image ('img'), feature map of vgg pool4 ('pool4'), feature map of vgg conv1 ('conv1'), or no cropping (None)
    # Feature extraction network: 1. pre-trained on ImageNet; 2. fine-tuned on PascalVOC2011, arg_groups['model']['pretrained'] = pretrained
    model = DualGeometricMatching(aff_output_dim=6, tps_output_dim=18, use_cuda=args.cuda, **arg_groups['model'],
                                  pytorch=False, caffe=False)

    ''' Set dual geometric matching model '''
    GM_cp_name_aff = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, args.model_aff)
    # GM_cp_name_aff = os.path.join(args.trained_models_dir, args.model_aff)
    if not os.path.exists(GM_cp_name_aff):
        raise Exception('There is no pre-trained geometric matching affine model, i.e. ' + GM_cp_name_aff)
    print('Load geometric matching affine model {}'.format(GM_cp_name_aff))
    GM_checkpoint_aff = torch.load(GM_cp_name_aff, map_location=lambda storage, loc: storage)
    # GM_checkpoint_aff['state_dict'] = OrderedDict(
    #     [(k.replace('model', 'GM_base'), v) for k, v in GM_checkpoint_aff['state_dict'].items()])
    # GM_checkpoint_aff['state_dict'] = OrderedDict(
    #     [(k.replace('FeatureRegression', 'ThetaRegression'), v) for k, v in GM_checkpoint_aff['state_dict'].items()])

    GM_cp_name_tps = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, args.model_tps)
    # GM_cp_name_tps = os.path.join(args.trained_models_dir, args.model_tps)
    if not os.path.exists(GM_cp_name_tps):
        raise Exception('There is no pre-trained geometric matching tps model, i.e. ' + GM_cp_name_tps)
    print('Load geometric matching tps model {}'.format(GM_cp_name_tps))
    GM_checkpoint_tps = torch.load(GM_cp_name_tps, map_location=lambda storage, loc: storage)
    # GM_checkpoint_tps['state_dict'] = OrderedDict(
    #     [(k.replace('FeatureRegression', 'ThetaRegression'), v) for k, v in GM_checkpoint_tps['state_dict'].items()])

    for name, param in model.FeatureExtraction.state_dict().items():
        # if 'num_batches_tracked' in name:
        #     print(name, param)
        #     continue
        model.FeatureExtraction.state_dict()[name].copy_(GM_checkpoint_aff['state_dict']['FeatureExtraction.' + name])
    for name, param in model.ThetaRegression.state_dict().items():
        # if 'num_batches_tracked' in name:
        #     print(name, param)
        #     continue
        model.ThetaRegression.state_dict()[name].copy_(GM_checkpoint_aff['state_dict']['ThetaRegression.' + name])
    for name, param in model.ThetaRegression2.state_dict().items():
        # if 'num_batches_tracked' in name:
        #     print(name, param)
        #     continue
        model.ThetaRegression2.state_dict()[name].copy_(GM_checkpoint_tps['state_dict']['ThetaRegression.' + name])
    print('Load geometric matching affine & tps model successfully!')

    ''' Resume training dual geometric matching model '''
    # If resume training, load interrupted model
    if args.resume:
        print('Resume training')
        GM_cp_name = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, args.model)
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
        # loss = TransformedGridLoss(geometric_model=args.geometric_model, use_cuda=args.cuda)
        loss = TransformedGridLoss(use_cuda=args.cuda)

    # Optimizer
    # Only regression part needs training
    # args.lr = 5e-8
    # args.num_epochs = 20
    # optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    print('Learning rate: {}'.format(args.lr))
    print('Num of epochs: {}'.format(args.num_epochs))
    if args.resume:
        optimizer.load_state_dict(GM_checkpoint['optimizer'])
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.requires_grad)

    ''' Set training dataset and validation dataset '''
    # Set path of csv files including image names (source and target) and pre-set random tps
    csv_file_train, train_dataset_path = get_dataset_csv(dataset_path=args.train_dataset_path, dataset=args.train_dataset, subset='train', geometric_model=args.geometric_model)
    print('Train csv file: {}'.format(csv_file_train))
    csv_file_val, eval_dataset_path = get_dataset_csv(dataset_path=args.eval_dataset_path, dataset=args.eval_dataset, subset='val')
    print('Val csv file: {}'.format(csv_file_val))
    output_size = (args.image_size, args.image_size)
    # Train dataset
    normalize = NormalizeImageDict(['source_image', 'target_image'])
    # normalize = None
    # print(args.random_t_tps)
    print('Whether generate random gt transformation: {}'.format(args.random_sample))
    dataset = TrainDataset(csv_file=csv_file_train, dataset_path=train_dataset_path, output_size=output_size,
                           geometric_model=args.geometric_model, random_sample=args.random_sample, normalize=normalize,
                           random_crop=args.random_crop)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    triple_generation = TrainDualTriple(geometric_model=args.geometric_model, output_size=output_size, use_cuda=args.cuda, normalize=normalize)
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
    checkpoint_suffix = '_' + args.geometric_model
    checkpoint_name = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, 'gm' + checkpoint_suffix + '.pth.tar')
    print('Checkpoint saving name: {}'.format(checkpoint_name))

    # Set visualization
    # vis = Visdom(env='Affine&TPSFinetuneOnRocco')
    vis = Visdom(env='AffTPS')
    # vis = Visdom()

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
    # Test pre-trained model before training
    model.ThetaRegression.eval()
    model.ThetaRegression2.eval()
    with torch.no_grad():
        test_fn(model=model, metric='pck', batch_size=args.batch_size, dataset=dataset_val, dataloader=dataloader_val, dual=True, args=args)
        results_watch, theta_watch, thetai_watch, _ = test_watch(model=model, metric='pck', batch_size=1,
                                                                 dataset=dataset_watch, dataloader=dataloader_watch,
                                                                 dual=True, args=args)
        vis_fn_dual(vis=vis, train_loss=train_loss, val_pck=val_pck, train_lr=train_lr, epoch=0,
                    num_epochs=args.num_epochs, dataloader=dataloader_watch, theta=theta_watch, thetai=thetai_watch,
                    results=results_watch, title='AffTPS', use_cuda=args.cuda)

    start = time.time()
    log_interval = 400
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        # model.train()
        model.ThetaRegression.train()
        model.ThetaRegression2.train()
        train_loss[epoch-1], train_time[epoch-1] = train_fn_dual(epoch=epoch, model=model, loss_fn=loss, optimizer=optimizer,
                                                                 dataloader=dataloader, triple_generation=triple_generation,
                                                                 use_cuda=args.cuda, log_interval=log_interval, vis=vis)
        # model.eval()
        model.ThetaRegression.eval()
        model.ThetaRegression2.eval()
        results, val_time[epoch-1] = test_fn(model=model, metric='pck', batch_size=args.batch_size, dataset=dataset_val,
                                             dataloader=dataloader_val, dual=True, args=args)

        val_pck[epoch - 1] = np.mean(results['afftps']['pck'])

        train_lr[epoch - 1] = optimizer.param_groups[0]['lr']

        # Visualization
        if epoch % 5 == 0 or epoch == 1:
            with torch.no_grad():
                results_watch, theta_watch, thetai_watch, _ = test_watch(model=model, metric='pck', batch_size=1,
                                                                         dataset=dataset_watch, dataloader=dataloader_watch,
                                                                         dual=True, args=args)
                vis_fn_dual(vis=vis, train_loss=train_loss, val_pck=val_pck, train_lr=train_lr, epoch=epoch,
                            num_epochs=args.num_epochs, dataloader=dataloader_watch, theta=theta_watch,
                            thetai=thetai_watch, results=results_watch, title='AffTPS', use_cuda=args.cuda)

        is_best = val_pck[epoch-1] > best_val_pck
        best_val_pck = max(val_pck[epoch-1] , best_val_pck)
        if is_best:
            best_epoch = epoch
        print('Save checkpoint...')
        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_pck': best_val_pck,
            'train_lr': train_lr,
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