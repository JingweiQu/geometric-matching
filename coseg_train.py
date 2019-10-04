# ========================================================================================
# Train cosegmentation model
# Author: Jingwei Qu
# Date: 01 Sep 2019
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

from geometric_matching.arguments.coseg_setting import Arguments
from geometric_matching.gm_model.cosegmentation import CoSegmentation
from geometric_matching.gm_model.loss import CosegLoss
from geometric_matching.gm_data.train_dataset_coseg import TrainDataset
from geometric_matching.gm_data.tss_dataset_coseg import TSSDataset

from geometric_matching.image.normalization import NormalizeImageDict
from geometric_matching.util.train_coseg import train_fn
from geometric_matching.util.test_coseg import test_fn
from geometric_matching.util.test_coseg_watch import test_watch
from geometric_matching.util.vis_coseg import vis_fn
from geometric_matching.util.net_util import get_dataset_csv, save_checkpoint

if __name__ == '__main__':

    print('Train CoSegmentation')

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

    ''' Initialize cosegmentation model '''
    print('Initialize cosegmentation model')
    pytorch = True
    caffe = False
    fixed_blocks = 3
    # Create cosegmentation model
    model = CoSegmentation(**arg_groups['model'], fixed_blocks=fixed_blocks, pytorch=pytorch, caffe=caffe)

    ''' Resume training cosegmentation model '''
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

    loss = CosegLoss(use_cuda=args.cuda)

    # Optimizer
    # Only regression part needs training
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr)
    print('Learning rate: {}'.format(args.lr))
    print('Num of epochs: {}'.format(args.num_epochs))
    if args.resume:
        optimizer.load_state_dict(GM_checkpoint['optimizer'])
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.requires_grad)

    ''' Set training dataset and validation dataset '''
    csv_file_train, train_dataset_path = get_dataset_csv(dataset_path=args.train_dataset_path, dataset=args.train_dataset, subset='train', geometric_model='coseg')
    print('Train csv file: {}'.format(csv_file_train))
    csv_file_val, eval_dataset_path = get_dataset_csv(dataset_path=args.eval_dataset_path, dataset=args.eval_dataset, subset='val')
    print('Val csv file: {}'.format(csv_file_val))
    output_size = (args.image_size, args.image_size)
    # Train dataset
    normalize = NormalizeImageDict(['source_image', 'target_image'])
    dataset = TrainDataset(csv_file=csv_file_train, dataset_path=train_dataset_path, output_size=output_size, normalize=normalize, random_crop=args.random_crop)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # Val dataset
    dataset_val = TSSDataset(csv_file=csv_file_val, dataset_path=eval_dataset_path, output_size=output_size, normalize=normalize)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Watching images dataset
    csv_file_watch, watch_dataset_path = get_dataset_csv(dataset_path=args.eval_dataset_path, dataset=args.eval_dataset, subset='watch')
    print('Watch csv file: {}'.format(csv_file_watch))
    dataset_watch = TSSDataset(csv_file=csv_file_watch, dataset_path=watch_dataset_path, output_size=output_size, normalize=normalize)
    dataloader_watch = torch.utils.data.DataLoader(dataset_watch, batch_size=1, shuffle=False, num_workers=4)

    ''' Train and val cosegmentation model '''
    # Define checkpoint name
    checkpoint_name = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, 'coseg.pth.tar')
    print('Checkpoint saving name: {}'.format(checkpoint_name))

    # Set visualization
    vis = Visdom(env='Coseg')

    print('Starting training')
    train_loss = np.zeros(args.num_epochs)
    val_iou = np.zeros(args.num_epochs)
    best_val_iou = float('-inf')
    train_lr = np.zeros(args.num_epochs)
    train_time = np.zeros(args.num_epochs)
    val_time = np.zeros(args.num_epochs)
    best_epoch = 0
    if args.resume:
        args.start_epoch = GM_checkpoint['epoch']
        best_val_iou = GM_checkpoint['best_val_iou']
        train_loss = GM_checkpoint['train_loss']
        val_iou = GM_checkpoint['val_iou']
        train_time = GM_checkpoint['train_time']
        val_time = GM_checkpoint['val_time']

    model.FeatureExtraction.eval()
    model.MaskRegression.eval()
    with torch.no_grad():
        test_fn(model=model, batch_size=args.batch_size, dataset=dataset_val, dataloader=dataloader_val, args=args)
        results_watch, masks_A, masks_B = test_watch(model=model, batch_size=1, dataset=dataset_watch,
                                                     dataloader=dataloader_watch, args=args)
        vis_fn(vis=vis, train_loss=train_loss, val_iou=val_iou, train_lr=train_lr, epoch=0, num_epochs=args.num_epochs,
               dataloader=dataloader_watch, results=results_watch, masks_A=masks_A, masks_B=masks_B, use_cuda=True)

    start = time.time()
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        model.MaskRegression.train()
        train_loss[epoch-1], train_time[epoch-1] = train_fn(epoch=epoch, model=model, loss_fn=loss, optimizer=optimizer,
                                                            dataloader=dataloader, use_cuda=args.cuda, log_interval=50,
                                                            vis=vis)

        model.MaskRegression.eval()
        results, val_time[epoch-1] = test_fn(model=model, batch_size=args.batch_size, dataset=dataset_val,
                                             dataloader=dataloader_val, args=args)

        val_iou[epoch - 1] = np.mean(results)

        train_lr[epoch - 1] = optimizer.param_groups[0]['lr']

        # Visualization
        # if epoch % 5 == 0 or epoch == 1:
        with torch.no_grad():
            results_watch, masks_A, masks_B = test_watch(model=model, batch_size=1, dataset=dataset_watch,
                                                         dataloader=dataloader_watch, args=args)
            vis_fn(vis=vis, train_loss=train_loss, val_iou=val_iou, train_lr=train_lr, epoch=epoch,
                   num_epochs=args.num_epochs, dataloader=dataloader_watch, results=results_watch,
                   masks_A=masks_A, masks_B=masks_B, use_cuda=True)

        is_best = val_iou[epoch-1] > best_val_iou
        best_val_iou = max(val_iou[epoch-1] , best_val_iou)
        if is_best:
            best_epoch = epoch
        print('Save checkpoint...')
        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_iou': best_val_iou,
            'train_loss': train_loss,
            'val_iou': val_iou,
            'train_time': train_time,
            'val_time': val_time,
        }, is_best, checkpoint_name)

    end = time.time()
    print('Best epoch: {}\t\tBest val IoU: {:.2%}\t\tTime cost (total): {:.4f}'.format(best_epoch, best_val_iou, end - start))

    print('Done!')