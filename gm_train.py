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

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pdb

from geometric_matching.util.net_util import save_checkpoint, str_to_bool
from geometric_matching.model.geometric_matching import GeometricMatching
# from geometric_matching.model.loss import TransformedGridLoss
# from geometric_matching.model.loss_new import TransformedGridLoss
from geometric_matching.data.synth_pair import SynthPairTnf
from geometric_matching.data.synth_dataset import SynthDataset
from geometric_matching.data.train_dataset import TrainDataset
from geometric_matching.data.train_pair import TrainPairTnf
from geometric_matching.image.normalization import NormalizeImageDict
from geometric_matching.util.train_val_fn import *
from geometric_matching.geotnf.transformation import GeometricTnf

from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list
from lib.model.faster_rcnn.vgg16 import vgg16

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Train a geometric matching network based on predicted rois by a faster rcnn')
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
    parser.add_argument('--checksession', dest='checksession', help='checksession to load model', default=1,
                        type=int)
    parser.add_argument('--checkepoch', dest='checkepoch', help='checkepoch to load network', default=7, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint to load network', default=23079, type=int)

    """ Arguments for the geometric matching model """
    # Training dataset parameters
    parser.add_argument('--training-dataset', type=str, default='PF-PASCAL',
                        help='Dataset to use for training: PF-PASCAL/PascalVOC2011')
    parser.add_argument('--training-dataset-path', type=str, default='geometric_matching/training_data',
                        help='Path to folder containing training dataset')
    parser.add_argument('--random-sample', type=str_to_bool, nargs='?', const=True, default=False,
                        help='sample random transformations')
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum constant')
    parser.add_argument('--num-epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='training batch size')
    parser.add_argument('--weight-decay', type=float, default=0, help='weight decay constant')
    parser.add_argument('--seed', type=int, default=2, help='Pseudo-RNG seed')
    # Model parameters
    parser.add_argument('--geometric-model', type=str, default='tps',
                        help='Geometric model to be regressed at output: affine or tps')
    parser.add_argument('--use-mse-loss', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Use MSE loss on tnf. parameters')
    parser.add_argument('--feature-extraction-cnn', type=str, default='vgg',
                        help='Feature extraction architecture: vgg/resnet101')
    parser.add_argument('--trained-models-dir', type=str, default='geometric_matching/trained_models',
                        help='Path to trained models folder')
    parser.add_argument('--trained-models-fn', type=str, default='checkpoint_adam', help='Trained model filename')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # print('Use GPU: {}-{}'.format(torch.cuda.get_device_name(torch.cuda.current_device()), torch.cuda.current_device()))

    ''' Initialize arguments '''
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

    ''' Initialize the fasterRCNN '''
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
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')

    if args.cuda:
        cfg.CUDA = True
        fasterRCNN.cuda()

    ''' Initialize the geometric matching model '''
    # Crop object from image ('object'), feature map of vgg pool4 ('pool4'), feature map of vgg conv1 ('conv1'),
    # or no cropping ('image')
    crop_layer = 'image'
    # Feature extraction network: 1. pre-trained vgg on ImageNet; 2. fine-tuned vgg on PascalVOC2011
    pretrained = True
    # Initialize theta_regression module as identity mapping
    init_identity = True
    # Create geometric_matching model
    print('Creating CNN model...')
    # Default: args.geometric_model - tps, args.feature_extraction_cnn - vgg
    model = GeometricMatching(use_cuda=args.cuda, geometric_model=args.geometric_model,
                              feature_extraction_cnn=args.feature_extraction_cnn, pretrained=pretrained,
                              crop_layer=crop_layer, init_identity=init_identity)
    if args.cuda:
        model.cuda()

    # Default is grid loss (as described in the CVPR 2017 paper)
    if args.use_mse_loss:
        print('Using MSE loss...')
        loss = nn.MSELoss()
    else:
        loss_type = 'loss_1'

        if loss_type == 'loss_1':
            from geometric_matching.model.loss_new import TransformedGridLoss
        elif loss_type == 'loss_2':
            from geometric_matching.model.loss import TransformedGridLoss

        print('Using grid loss...')
        loss = TransformedGridLoss(use_cuda=args.cuda, geometric_model=args.geometric_model)

    # Optimizer
    # Only regression part needs training
    optimizer = optim.Adam(model.ThetaRegression.parameters(), lr=args.lr)

    tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=args.cuda)

    ''' Initialize training dataset and validation dataset '''
    # Set path for pre-set random tps csv files for images
    args.training_dataset_path = os.path.join(args.training_dataset_path, args.training_dataset)
    csv_file_train = os.path.join(args.training_dataset_path, 'train_' + args.training_dataset + '.csv')
    csv_file_val = os.path.join(args.training_dataset_path, 'val_' + args.training_dataset + '.csv')
    random_t_tps = 0.4
    if args.training_dataset == 'PascalVOC2011':
        normalize = NormalizeImageDict(['image'])
        dataset = SynthDataset(geometric_model=args.geometric_model, csv_file=csv_file_train,
                               dataset_path=args.training_dataset_path, transform=normalize,
                               random_sample=args.random_sample)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        dataset_val = SynthDataset(geometric_model=args.geometric_model, csv_file=csv_file_val,
                                   dataset_path=args.training_dataset_path,
                                   transform=normalize, random_sample=args.random_sample)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=4)
        # Function for generating training pair
        pair_generation_tnf = SynthPairTnf(geometric_model=args.geometric_model, use_cuda=args.cuda,
                                           crop_layer=crop_layer)
        # Path for saving checkpoint
        checkpoint_path = os.path.join(args.trained_models_dir, args.training_dataset, crop_layer,
                                       (lambda x:'pretrained_vgg' if x==True else 'finetuned_vgg')(pretrained),
                                       (lambda x:'identity' if x==True else 'random')(init_identity))

    elif args.training_dataset == 'PF-PASCAL':
        normalize = NormalizeImageDict(['source_image', 'target_image'])
        dataset = TrainDataset(geometric_model=args.geometric_model, csv_file=csv_file_train,
                                  dataset_path=args.training_dataset_path, transform= normalize,
                                  random_sample=args.random_sample, random_t_tps=random_t_tps)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        dataset_val = TrainDataset(geometric_model=args.geometric_model, csv_file=csv_file_val,
                                      dataset_path=args.training_dataset_path, transform= normalize,
                                      random_sample=args.random_sample, random_t_tps=random_t_tps)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=4)

        pair_generation_tnf = TrainPairTnf(geometric_model=args.geometric_model, use_cuda=args.cuda,
                                              crop_layer=crop_layer)

        checkpoint_path = os.path.join(args.trained_models_dir, args.training_dataset, loss_type, crop_layer,
                                       (lambda x:'identity' if x==True else 'random')(init_identity))

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    ''' Train and val the geometric matching model '''
    best_val_loss = float("inf")
    best_epoch = 0

    print('Starting training...')

    start = time.time()
    for epoch in range(1, args.num_epochs + 1):
        if args.training_dataset == 'PascalVOC2011':
            if crop_layer == 'pool4':
                train_loss = train_pool4_synth(epoch, model, fasterRCNN, loss, optimizer, dataloader,
                                               pair_generation_tnf, log_interval=100, use_cuda=args.cuda)
                val_loss = val_pool4_synth(model, fasterRCNN, loss, dataloader_val, pair_generation_tnf,
                                           use_cuda=args.cuda)
            elif crop_layer == 'object':
                train_loss = train_object_synth(epoch, model, fasterRCNN, loss, optimizer, dataloader,
                                                pair_generation_tnf, log_interval=100, use_cuda=args.cuda)
                val_loss = val_object_synth(model, fasterRCNN, loss, dataloader_val, pair_generation_tnf,
                                            use_cuda=args.cuda)
            elif crop_layer == 'image':
                train_loss = train_image_synth(epoch, model, loss, optimizer, dataloader, pair_generation_tnf, tpsTnf,
                                               use_cuda=True, log_interval=100)
                val_loss = val_image_synth(model, loss, dataloader_val, pair_generation_tnf, use_cuda=True)

        elif args.training_dataset == 'PF-PASCAL':
            if crop_layer == 'pool4':
                print('Not ok!')
                # train_loss = train_image_synth(epoch, model, fasterRCNN, loss, optimizer, dataloader,
                #                                pair_generation_tnf, log_interval=100, use_cuda=args.cuda)
                # val_loss = val_image_synth(model, fasterRCNN, loss, dataloader_val, pair_generation_tnf,
                #                            use_cuda=args.cuda)
            elif crop_layer == 'object':
                if loss_type == 'loss_1':
                    train_loss = train_object_pfpascal_1(epoch, model, fasterRCNN, loss, optimizer, dataloader,
                                                         pair_generation_tnf, tpsTnf, log_interval=100,
                                                         use_cuda=args.cuda)
                    val_loss = val_object_pfpascal_1(model, fasterRCNN, loss, dataloader_val, pair_generation_tnf,
                                                     use_cuda=args.cuda)
                elif loss_type == 'loss_2':
                    train_loss = train_object_pfpascal_2(epoch, model, fasterRCNN, loss, optimizer, dataloader,
                                                         pair_generation_tnf, tpsTnf, log_interval=100, use_cuda=args.cuda)
                    val_loss = val_object_pfpascal_2(model, fasterRCNN, loss, dataloader_val, pair_generation_tnf, tpsTnf,
                                                     use_cuda=args.cuda)
            elif crop_layer == 'image':
                if loss_type == 'loss_1':
                    train_loss = train_image_pfpascal_1(epoch, model, loss, optimizer, dataloader, pair_generation_tnf,
                                                        tpsTnf, log_interval=100, use_cuda=args.cuda)
                    val_loss = val_image_pfpascal_1(model, loss, dataloader_val, pair_generation_tnf,
                                                    use_cuda=args.cuda)
                elif loss_type == 'loss_2':
                    train_loss = train_image_pfpascal_2(epoch, model, loss, optimizer, dataloader, pair_generation_tnf,
                                                        tpsTnf, log_interval=100, use_cuda=args.cuda)
                    val_loss = val_image_pfpascal_2(model, loss, dataloader_val, pair_generation_tnf, tpsTnf,
                                                    use_cuda=args.cuda)

        # Best loss
        is_best = val_loss < best_val_loss
        if is_best:
            best_epoch = epoch
        best_val_loss = min(val_loss, best_val_loss)
        # Name for saving trained model
        checkpoint_name = os.path.join(checkpoint_path, str(epoch) + '_' + args.trained_models_fn + '_' +
                                       args.geometric_model + '_grid_loss_' + args.feature_extraction_cnn + '.pth.tar')
        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'optimizer': optimizer.state_dict(),
        }, checkpoint_name)
    end = time.time()

    # best_checkpoint_name = os.path.join(checkpoint_path,
    #                                     str(best_epoch) + '_' + args.trained_models_fn + '_' + args.geometric_model +
    #                                     '_grid_loss_' + args.feature_extraction_cnn + '.pth.tar')
    # copy_name = os.path.join(checkpoint_path, 'best_' + str(best_epoch) + '_' + args.trained_models_fn + '_' +
    #                          args.geometric_model + '_grid_loss_' + args.feature_extraction_cnn + '.pth.tar')

    # Select the best model, and copy
    # shutil.copyfile(best_checkpoint_name, copy_name)
    print('Best epoch {:}\t\tBest val loss {:.4f}\t\tTime cost (total) {:.4f}'.format(best_epoch, best_val_loss, end - start))
    print('Done!')



































'''
    # The directory for loading pre-trained RoIFeature model and saving trained models
    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    # print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pair_generation_tnf = SynthPairTnf(geometric_model=args.geometric_model, use_cuda=args.cuda)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    gt_theta = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        theta = gt_theta.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    gt_theta = Variable(gt_theta)

    if args.cuda:
        cfg.CUDA = True

    geoRPN = GeometricRPN(geometric_model=args.geometric_model, use_cuda=args.cuda,
                          pretrained=True, classes=imdb.classes, class_agnostic=args.class_agnostic, select=True)

    # Default is grid loss (as described in the CVPR 2017 paper)
    if args.use_mse_loss:
        print('Using MSE loss...')
        loss_fn = nn.MSELoss()
    else:
        print('Using grid loss...')
        loss_fn = TransformedGridLoss(use_cuda=args.cuda, geometric_model=args.geometric_model)

    load_name = os.path.join(input_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading geoRPN.RoIFeature checkpoint %s" % (load_name))
    roi_checkpoint = torch.load(load_name)
    geoRPN.RoIFeature.load_state_dict(roi_checkpoint['model'])
    if 'pooling_mode' in roi_checkpoint.keys():
        cfg.POOLING_MODE = roi_checkpoint['pooling_mode']
    print("loaded geoRPN.RoIFeature checkpoint %s successfully!" % (load_name))

    if args.optimizer == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, geoRPN.parameters()), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, geoRPN.parameters()), lr=args.lr,
                              momentum=cfg.TRAIN.MOMENTUM)

    if args.mGPUs:
        print("Let's use {gpu_nums} GPUs!".format(gpu_nums=torch.cuda.device_count()))
        geoRPN = nn.DataParallel(geoRPN)

    if args.cuda:
        geoRPN.cuda()

    iters_per_epoch = int(train_size / args.batch_size)

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        geoRPN.train()
        train_loss = 0
        loss_temp = 0
        start = time.time()
        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            # im_data: the array data of images, shape (batch_size, channel, N, M), N: resized height, M: resized width
            # min(N, M) = 600, max(N, M) = max(Q, P) * im_scale
            # im_info: N, M, im_scale, shape (batch_size, 3)
            # im_scale = 600 / min(Q, P), Q and P are height and width of the original image
            # gt_boxes: ground truth bounding boxes of objects in images, shape (batch_size, 20, 5), 5 includes x_min, y_min, x_max, y_max, class
            # num_boxes: the ground-truth number of boxes in images, shape (batch_size,)
            data = next(data_iter)
            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])
            gt_theta.data.resize_(data[4].size()).copy_(data[4])
            # print('lalal')

            im_pair = pair_generation_tnf(im_data, gt_theta)

            geoRPN.zero_grad()
            optimizer.zero_grad()

            theta = geoRPN(im_pair, im_info, gt_boxes, num_boxes, select=True)
            # print(theta.shape)
            # print(im_pair['theta_GT'].shape)

            loss = loss_fn(theta, im_pair['theta_GT'])
            # print(loss)
            loss_temp += loss.item()
            train_loss += loss.item()

            # backward
            loss.backward()
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp))

                loss_temp = 0
                start = time.time()

        train_loss /= len(dataloader)
        print('Train set: Average loss: {:.4f}'.format(train_loss))

        save_name = os.path.join(output_dir, 'geoRPN_adam_tps_grid_loss_vgg16_{}_{}_{}.pth'.format(args.session, epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': geoRPN.module.state_dict() if args.mGPUs else geoRPN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))
'''