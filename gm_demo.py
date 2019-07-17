# ====================================================================================================
# Show results of geometric matching model on benchmarks based on the object detection by faster rcnn
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
from skimage import io
import cv2

from geometric_matching.arguments.arguments_setting import Arguments
from geometric_matching.gm_model.dual_geometric_matching import DualGeometricMatching
from geometric_matching.gm_model.geometric_matching import GeometricMatching
from geometric_matching.image.normalization import *
from geometric_matching.gm_data.pf_willow_dataset import PFWILLOWDataset
from geometric_matching.gm_data.pf_pascal_dataset import PFPASCALDataset
from geometric_matching.gm_data.caltech_dataset import CaltechDataset
from geometric_matching.gm_data.tss_dataset import TSSDataset
from geometric_matching.util.net_util import *
from geometric_matching.util.test_fn import *
from geometric_matching.geotnf.transformation import *
from geometric_matching.geotnf.point_tnf import *

from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list

# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def save_image(image, image_name):
    image = normalize_image(image, forward=False)
    image = image.permute(1, 2, 0).cpu().numpy()
    io.imsave(image_name, (image  * 255).astype(np.uint8))

def save_box(image, box, image_name):
    dataset_classes = np.asarray(['__background__',
                                  'aeroplane', 'bicycle', 'bird', 'boat',
                                  'bottle', 'bus', 'car', 'cat', 'chair',
                                  'cow', 'diningtable', 'dog', 'horse',
                                  'motorbike', 'person', 'pottedplant',
                                  'sheep', 'sofa', 'train', 'tvmonitor'])
    image = normalize_image(image, forward=False)
    image = image.permute(1, 2, 0).cpu().numpy()
    image = image[:, :, ::-1]
    image = (image * 255).astype(np.uint8)
    cv2.rectangle(image, (box[:, 0], box[:, 1]), (box[:, 2], box[:, 3]), (0, 0, 255), 4)
    cv2.putText(image, dataset_classes[int(box[:, 5])] + ' ' + '%.3f' % box[:, 4], (box[:, 0], box[:, 1]), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
    cv2.imwrite(image_name, image)

# def save_poly(image, points, image_name):
#     image = normalize_image(image, forward=False)
#     image = image.permute(1, 2, 0).cpu().numpy()
#     image = image[:, :, ::-1]
#     image = (image * 255).astype(np.uint8)
#     cv2.polylines(img=image, pts=np.int32(points), isClosed=True, color=(0, 255, 0), thickness=3)
#     cv2.imwrite(image_name, image)

def affTpsTnf(source_image, theta_aff, theta_aff_tps, use_cuda=True):
    tpstnf = GeometricTnf(geometric_model = 'tps',use_cuda=use_cuda)
    sampling_grid_tps = tpstnf(image_batch=source_image, theta_batch=theta_aff_tps, return_sampling_grid=True)[1]
    X = sampling_grid_tps[:, :, :, 0].unsqueeze(3)
    Y = sampling_grid_tps[:, :, :, 1].unsqueeze(3)
    Xp = X * theta_aff[:, 0, 0].unsqueeze(1).unsqueeze(2) + Y * theta_aff[:, 0, 1].unsqueeze(1).unsqueeze(2) + theta_aff[:, 0, 2].unsqueeze(1).unsqueeze(2)
    Yp = X * theta_aff[:, 1, 0].unsqueeze(1).unsqueeze(2) + Y * theta_aff[:, 1, 1].unsqueeze(1).unsqueeze(2) + theta_aff[:, 1, 2].unsqueeze(1).unsqueeze(2)
    sampling_grid = torch.cat((Xp, Yp), 3)
    warped_image_batch = F.grid_sample(source_image, sampling_grid)

    return warped_image_batch

if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    print('GeometricMatching model demo')

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
                                      thresh=0.05, max_per_image=50, crop_layer=None, return_box=True,
                                      **arg_groups['model'])
    else:
        model = GeometricMatching(output_dim=output_dim, use_cuda=args.cuda, pretrained=True, **arg_groups['model'])

    ''' Set FasterRCNN module '''
    if args.dual:
        RCNN_cp_dir = os.path.join(args.load_dir, args.net, args.dataset)
        RCNN_cp_name = os.path.join(RCNN_cp_dir, 'best_faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print('Load fasterRCNN model {}'.format(RCNN_cp_name))
        if not os.path.exists(RCNN_cp_name):
            raise Exception('There is no pre-trained fasterRCNN model, i.e. {}'.format(RCNN_cp_name))
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
    GM_cp_name = os.path.join(args.trained_models_dir, args.model)
    if not os.path.exists(GM_cp_name):
        raise Exception('There is no pre-trained geometric_matching model, i.e. ' + GM_cp_name)
    print('Load geometric matching model {}'.format(GM_cp_name))
    GM_checkpoint = torch.load(GM_cp_name, map_location=lambda storage, loc: storage)
    # GM_checkpoint['state_dict'] = OrderedDict(
    #     [(k.replace('vgg', 'model'), v) for k, v in GM_checkpoint['state_dict'].items()])
    # GM_checkpoint['state_dict'] = OrderedDict(
    #     [(k.replace('FeatureRegression', 'ThetaRegression'), v) for k, v in GM_checkpoint['state_dict'].items()])
    for name, param in model.FeatureExtraction.state_dict().items():
        model.FeatureExtraction.state_dict()[name].copy_(GM_checkpoint['state_dict']['FeatureExtraction.' + name])
    for name, param in model.ThetaRegression.state_dict().items():
        model.ThetaRegression.state_dict()[name].copy_(GM_checkpoint['state_dict']['ThetaRegression.' + name])
    # for name, param in model.FasterRCNN.state_dict().items():
    #     model.FasterRCNN.state_dict()[name].copy_(GM_checkpoint['state_dict']['FasterRCNN.' + name])
    # model.load_state_dict(GM_checkpoint['state_dict'])
    print('Load geometric matching model successfully!')

    if args.cuda:
        model.cuda()

    ''' Set demo dataset '''
    if args.test_dataset == 'PF-WILLOW':
        TestDataSet = PFWILLOWDataset
    elif args.test_dataset == 'PF-PASCAL':
        TestDataSet = PFPASCALDataset
    elif args.test_dataset == 'Caltech-101':
        TestDataSet = CaltechDataset
    elif args.test_dataset == 'TSS':
        TestDataSet = TSSDataset

    # Set path of csv file including image names (source and target) and annotation
    csv_file_test, test_dataset_path = get_dataset_csv(dataset_path=args.test_dataset_path, dataset=args.test_dataset, subset='test')
    output_size = (args.image_size, args.image_size)
    normalize = NormalizeImageDict(['source_image', 'target_image'])
    dataset = TestDataSet(csv_file=csv_file_test, dataset_path=test_dataset_path, output_size=output_size,
                          transform=normalize)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    ''' Demo trained geometric matching model '''
    print('Demo on {}'.format(args.test_dataset))
    if not model.return_box:
        results_dir = os.path.join('geometric_matching/demo_results/20190623', args.test_dataset)
    else:
        results_dir = os.path.join('geometric_matching/demo_results/20190623/with_box', args.test_dataset)
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    # tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=args.cuda)
    affTnf = GeometricTnf(geometric_model='affine', use_cuda=args.cuda)
    pt = PointTnf(use_cuda=args.cuda)
    with torch.no_grad():
        model.eval()
        for batch_idx, batch in enumerate(dataloader):
            # if batch_idx > 600:
            #     break
            # if batch_idx % 200 == 0:
            if batch_idx > 790:
                if args.cuda:
                    batch = batch_cuda(batch)

                theta_aff = None
                theta_tps = None
                theta_aff_tps = None

                if not model.return_box:
                    theta_aff_tps, theta_aff = model(batch)
                else:
                    theta_aff_tps, theta_aff, box_s, box_t = model(batch)
                warped_image_aff = affTnf(image_batch=batch['source_image'], theta_batch=theta_aff)
                warped_image_aff_tps = affTpsTnf(source_image=batch['source_image'], theta_aff=theta_aff,
                                                 theta_aff_tps=theta_aff_tps, use_cuda=args.cuda)
                if not model.return_box:
                    # warped_image_tps = tpsTnf(batch['source_image'], theta_tps)
                    save_image(batch['source_image'][0], os.path.join(results_dir, str(batch_idx) + '_source_image.jpg'))
                    save_image(batch['target_image'][0], os.path.join(results_dir, str(batch_idx) + '_target_image.jpg'))
                    # save_image(warped_image_tps[0], os.path.join(results_dir, str(batch_idx) + '_warped_image_tps.jpg'))
                    save_image(warped_image_aff[0], os.path.join(results_dir, str(batch_idx) + '_warped_image_aff.jpg'))
                    save_image(warped_image_aff_tps[0], os.path.join(results_dir, str(batch_idx) + '_warped_image_aff_tps.jpg'))
                else:
                    if torch.sum(box_s[:, 0:4]).item() > 0 and torch.sum(box_t[:, 0:4]).item() > 0:
                        save_box(batch['source_image'][0], box_s, os.path.join(results_dir, str(batch_idx) + '_source_image.jpg'))
                        save_box(batch['target_image'][0], box_t, os.path.join(results_dir, str(batch_idx) + '_target_image.jpg'))
                        save_box(warped_image_aff[0], box_t, os.path.join(results_dir, str(batch_idx) + '_warped_image_aff.jpg'))
                        save_box(warped_image_aff_tps[0], box_t, os.path.join(results_dir, str(batch_idx) + '_warped_image_aff_tps.jpg'))
                    else:
                        save_image(batch['source_image'][0], os.path.join(results_dir, str(batch_idx) + '_source_image.jpg'))
                        save_image(batch['target_image'][0], os.path.join(results_dir, str(batch_idx) + '_target_image.jpg'))
                        save_image(warped_image_aff[0], os.path.join(results_dir, str(batch_idx) + '_warped_image_aff.jpg'))
                        save_image(warped_image_aff_tps[0], os.path.join(results_dir, str(batch_idx) + '_warped_image_aff_tps.jpg'))


                    # im_size = torch.Tensor([[240, 240]]).cuda()
                    # target_box = torch.Tensor([[[box_t[0, 0], box_t[0, 0], box_t[0, 2], box_t[0, 2]],
                    #                            [box_t[0, 1], box_t[0, 3], box_t[0, 3], box_t[0, 1]]]]).cuda()

                    # warp points with estimated transformations
                    # target_box_norm = PointsToUnitCoords(P=target_box, im_size=im_size)
                    #
                    # warped_box_aff_norm = pt.affPointTnf(theta=theta_aff, points=target_box_norm)
                    # warped_box_aff = PointsToPixelCoords(P=warped_box_aff_norm, im_size=im_size)
                    # warped_box_aff = warped_box_aff.permute(2, 0, 1).cpu().numpy()
                    #
                    # warped_box_aff_tps_norm = pt.tpsPointTnf(theta=theta_aff_tps, points=target_box_norm)
                    # warped_box_aff_tps_norm = pt.affPointTnf(theta=theta_aff, points=warped_box_aff_tps_norm)
                    # warped_box_aff_tps = PointsToPixelCoords(P=warped_box_aff_tps_norm, im_size=im_size)
                    # warped_box_aff_tps = warped_box_aff_tps.permute(2, 0, 1).cpu().numpy()
                    #
                    # save_poly(warped_image_aff[0], warped_box_aff, os.path.join(results_dir, str(batch_idx) + '_warped_image_aff.jpg'))
                    # save_poly(warped_image_aff_tps[0], warped_box_aff_tps, os.path.join(results_dir, str(batch_idx) + '_warped_image_aff_tps.jpg'))

                # Show images
                # im_show_1(batch['source_image'][0], 'source_image', 1, 4, 1)
                # im_show_1(warped_image_tps[0], 'warped_image_tps', 1, 4, 2)
                # im_show_1(warped_image_aff[0], 'warped_image_aff', 1, 4, 2)
                # im_show_1(warped_image_aff_tps[0], 'warped_image_aff_tps', 1, 4, 3)
                # im_show_1(batch['target_image'][0], 'target_image', 1, 4, 4)
                # plt.show()
    print('Done!')