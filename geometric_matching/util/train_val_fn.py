# ========================================================================================
# Train and evaluate geometric matching model
# Author: Jingwei Qu
# Date: 05 Mar 2019
# ========================================================================================


from __future__ import print_function, division
import torch
from torch.autograd import Variable
import time

from geometric_matching.util.net_util import *

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# def show_images(tnf_batch, warped_image_tps):
#     # Show images
#     rows = 1
#     cols = 3
#     for i in range(tnf_batch['source_image'].shape[0]):
#         show_id = i
#
#         im_show_1(tnf_batch['source_image'][show_id], 'source_image', rows, cols, 1)
#
#         im_show_1(warped_image_tps[show_id], 'warped_image', rows, cols, 2)
#
#         im_show_1(tnf_batch['target_image'][show_id], 'target_image', rows, cols, 3)
#
#         plt.show()

def train_image_synth(epoch, model, loss_fn, optimizer, dataloader, pair_generation_tnf, tpsTnf, use_cuda=True,
                      log_interval=100):
    """

        Train the model with synthetically training pairs {source image, target image (warped source image), theta_GT}
        from PascalVOC2011.

    """

    model.train()
    train_loss = 0
    start = time.time()
    begin = time.time()
    for batch_idx, batch in enumerate(dataloader):
        ''' Move input batch to gpu '''
        # batch['source_image'].shape & batch['target_image'].shape: (batch_size, 3, 240, 240)
        # batch['theta'].shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        for k, v in batch.items():
            if use_cuda and batch[k].is_cuda == False:
                batch[k] = batch[k].cuda()

        ''' Get the training triple {source image, target image, refer image (warped source image), theta_GT}'''
        tnf_batch = pair_generation_tnf(batch, None)

        ''' Train the model '''
        optimizer.zero_grad()
        # Predict tps parameters between images
        # theta.shape: (batch_size, 18) for tps
        theta = model(tnf_batch)
        loss = loss_fn(theta, tnf_batch['theta_GT'])
        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy()
        if batch_idx % log_interval == 0:
            end = time.time()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}\t\tTime cost: {:.6f}'.format(
                epoch, batch_idx, len(dataloader),
                100. * batch_idx / len(dataloader), loss.item(), end - start))
            start = time.time()
        # warped_image_tps = tpsTnf(tnf_batch['source_image'], theta)
        # show_images(tnf_batch, warped_image_tps.detach())

    end = time.time()
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}\t\tTime cost: {:.4f}'.format(train_loss, end - begin))
    return train_loss


def val_image_synth(model, loss_fn, dataloader, pair_generation_tnf, use_cuda=True):
    """

        Val the model with synthetically training pairs {source image, target image (warped source image), theta_GT}
        from PascalVOC2011.

    """

    model.eval()
    val_loss = 0
    start = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            ''' Move input batch to gpu '''
            for k, v in batch.items():
                if use_cuda and batch[k].is_cuda == False:
                    batch[k] = batch[k].cuda()

            ''' Get the training triple {source object, target object, refer object (warped source object), theta_GT}'''
            tnf_batch = pair_generation_tnf(batch, None)

            ''' Val the model '''
            theta = model(tnf_batch)
            loss = loss_fn(theta, tnf_batch['theta_GT'])
            val_loss += loss.data.cpu().numpy()

    end = time.time()
    val_loss /= len(dataloader)
    print('Val set: Average loss: {:.4f}\t\tTime cost: {:.4f}'.format(val_loss, end - start))
    return val_loss

def show_images(tnf_batch, warped_image_tps, warped_image_tps_2, warped_image_tps_3):
    # Show images
    rows = 1
    cols = 3
    for i in range(tnf_batch['source_image'].shape[0]):
        show_id = i

        im_show_1(tnf_batch['source_image'][show_id], 'source_image', rows, cols, 1)

        # im_show_1(warped_image_tps[show_id], 'warped_image', rows, cols, 2)

        im_show_1(tnf_batch['target_image'][show_id], 'target_image', rows, cols, 2)

        # im_show_1(tnf_batch['target_image'][show_id], 'target_image', rows, cols, 4)

        # im_show_1(warped_image_tps_2[show_id], 'warped_image_2', rows, cols, 5)

        im_show_1(tnf_batch['refer_image'][show_id], 'refer_image', rows, cols, 3)

        # im_show_1(tnf_batch['source_image'][show_id], 'source_image', rows, cols, 7)

        # im_show_1(warped_image_tps_3[show_id], 'warped_image_3', rows, cols, 8)

        # im_show_1(tnf_batch['refer_image'][show_id], 'refer_image', rows, cols, 9)

        plt.show()

def train_image_pfpascal_1(epoch, model, loss_fn, optimizer, dataloader, pair_generation_tnf, tpsTnf, use_cuda=True,
                           log_interval=100):
    """

        Train the model with synthetically training triple:
        {source image, target image, refer image (warped source image), theta_GT} from PF-PASCAL.
        1. Train the transformation parameters theta_st from source image to target image;
        2. Train the transformation parameters theta_tr from target image to refer image;
        3. Combine theta_st and theta_st to obtain theta from source image to refer image, and compute loss between
        theta and theta_GT.

        :param tpsTnf: the transformation to warp source object with theta_st

    """
    model.train()
    train_loss = 0
    start = time.time()
    begin = time.time()
    for batch_idx, batch in enumerate(dataloader):
        ''' Move input batch to gpu '''
        # batch['source_image'].shape & batch['target_image'].shape: (batch_size, 3, 240, 240)
        # batch['theta'].shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        for k, v in batch.items():
            if use_cuda and batch[k].is_cuda == False:
                batch[k] = batch[k].cuda()

        ''' Get the training triple {source image, target image, refer image (warped source image), theta_GT}'''
        tnf_batch = pair_generation_tnf(batch, None, None)
        batch_1 = {'source_image': tnf_batch['source_image'], 'target_image': tnf_batch['target_image']}
        batch_2 = {'source_image': tnf_batch['target_image'], 'target_image': tnf_batch['refer_image']}

        ''' Train the model '''
        optimizer.zero_grad()
        # Predict tps parameters between images
        # theta.shape: (batch_size, 18) for tps
        theta_st = model(batch_1)  # from source image to target image
        theta_tr = model(batch_2)  # from target image to refer image
        loss = loss_fn(theta_st, theta_tr, tnf_batch['theta_GT'])
        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy()
        if batch_idx % log_interval == 0:
            end = time.time()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}\t\tTime cost: {:.6f}'.format(
                epoch, batch_idx, len(dataloader),
                100. * batch_idx / len(dataloader), loss.item(), end - start))
            start = time.time()
        warped_image_tps = tpsTnf(batch_1['source_image'], theta_st)
        warped_image_tps_2 = tpsTnf(batch_2['source_image'], theta_tr)
        warped_image_tps_3 = tpsTnf(warped_image_tps, theta_tr)
        show_images(tnf_batch, warped_image_tps.detach(), warped_image_tps_2.detach(), warped_image_tps_3.detach())

    end = time.time()
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}\t\tTime cost: {:.4f}'.format(train_loss, end - begin))
    return train_loss

def val_image_pfpascal_1(model, loss_fn, dataloader, pair_generation_tnf, use_cuda=True):
    """

        Train the model with synthetically training triple:
        {source image, target image, refer image (warped source image), theta_GT} from PF-PASCAL.
        1. Train the transformation parameters theta_st from source image to target image;
        2. Train the transformation parameters theta_tr from target image to refer image;
        3. Combine theta_st and theta_st to obtain theta from source image to refer image, and compute loss between
        theta and theta_GT.

        :param tpsTnf: the transformation to warp source object with theta_st

    """
    model.eval()
    val_loss = 0
    start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            ''' Move input batch to gpu '''
            for k, v in batch.items():
                if use_cuda and batch[k].is_cuda == False:
                    batch[k] = batch[k].cuda()

            ''' Get the training triple {source object, target object, refer object (warped source object), theta_GT}'''
            tnf_batch = pair_generation_tnf(batch, None, None)
            batch_1 = {'source_image': tnf_batch['source_image'], 'target_image': tnf_batch['target_image']}
            batch_2 = {'source_image': tnf_batch['target_image'], 'target_image': tnf_batch['refer_image']}

            ''' Val the model '''
            theta_st = model(batch_1)  # from source image to target image
            theta_tr = model(batch_2)  # from target image to refer image
            loss = loss_fn(theta_st, theta_tr, tnf_batch['theta_GT'])
            val_loss += loss.data.cpu().numpy()

    end = time.time()
    val_loss /= len(dataloader)
    print('Val set: Average loss: {:.4f}\t\tTime cost: {:.4f}'.format(val_loss, end - start))
    return val_loss

# def show_images(tnf_batch, warped_image_tps, warped_image_tps_2):
#     # Show images
#     rows = 2
#     cols = 3
#     for i in range(tnf_batch['source_image'].shape[0]):
#         show_id = i
#
#         im_show_1(tnf_batch['source_image'][show_id], 'source_image', rows, cols, 1)
#
#         im_show_1(warped_image_tps[show_id], 'warped_image', rows, cols, 2)
#
#         im_show_1(tnf_batch['target_image'][show_id], 'target_image', rows, cols, 3)
#
#         im_show_1(warped_image_tps[show_id], 'warped_image', rows, cols, 4)
#
#         im_show_1(warped_image_tps_2[show_id], 'warped_image_2', rows, cols, 5)
#
#         im_show_1(tnf_batch['refer_image'][show_id], 'refer_image', rows, cols, 6)
#
#         plt.show()

def train_image_pfpascal_2(epoch, model, loss_fn, optimizer, dataloader, pair_generation_tnf, tpsTnf, use_cuda=True,
                           log_interval=100):
    """

        Train the model with synthetically training triple:
        {source image, target image, refer image (warped source image), theta_GT} from PF-PASCAL.
        1. Train the transformation parameters theta_st from source image to target image;
        2. Warp source image with theta_st to obtain warped image;
        3. Train the transformation parameters theta from warped image to refer image;
        4. Compute loss between theta and theta_GT.

        :param tpsTnf: the transformation to warp source object with theta_st

    """
    model.train()
    train_loss = 0
    start = time.time()
    begin = time.time()
    for batch_idx, batch in enumerate(dataloader):
        ''' Move input batch to gpu '''
        # batch['source_image'].shape & batch['target_image'].shape: (batch_size, 3, 240, 240)
        # batch['theta'].shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        for k, v in batch.items():
            if use_cuda and batch[k].is_cuda == False:
                batch[k] = batch[k].cuda()

        ''' Get the training triple {source image, target image, refer image (warped source image), theta_GT}'''
        tnf_batch = pair_generation_tnf(batch, None, None)

        ''' Train the model '''
        optimizer.zero_grad()
        # Predict tps parameters between images
        # theta.shape: (batch_size, 18) for tps
        batch_1 = {'source_image': tnf_batch['source_image'], 'target_image': tnf_batch['target_image']}
        theta_st = model(batch_1)   # from source image to target image
        # Warp source image with theta_st, and make new training pair {warped image ,refer image}
        warped_image_tps = tpsTnf(batch_1['source_image'], theta_st)
        batch_2 = {'source_image': warped_image_tps, 'target_image': tnf_batch['refer_image']}
        theta = model(batch_2)   # from warped image to refer image
        loss = loss_fn(theta, tnf_batch['theta_GT'])
        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy()
        if batch_idx % log_interval == 0:
            end = time.time()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}\t\tTime cost: {:.6f}'.format(
                epoch, batch_idx, len(dataloader),
                100. * batch_idx / len(dataloader), loss.item(), end - start))
            start = time.time()
        # warped_image_tps_2 = tpsTnf(batch_2['source_image'], theta)
        # show_images(tnf_batch, warped_image_tps.detach(), warped_image_tps_2.detach())

    end = time.time()
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}\t\tTime cost: {:.4f}'.format(train_loss, end - begin))
    return train_loss

def val_image_pfpascal_2(model, loss_fn, dataloader, pair_generation_tnf, tpsTnf, use_cuda=True):
    """

        Val the model with synthetically training triple:
        {source image, target image, refer image (warped source image), theta_GT} from PF-PASCAL.
        1. Train the transformation parameters theta_st from source image to target image;
        2. Warp source image with theta_st to obtain warped image;
        3. Train the transformation parameters theta from warped image to refer image;
        4. Compute loss between theta and theta_GT.

        :param tpsTnf: the transformation to warp source object with theta_st

    """

    model.eval()
    val_loss = 0
    start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            ''' Move input batch to gpu '''
            for k, v in batch.items():
                if use_cuda and batch[k].is_cuda == False:
                    batch[k] = batch[k].cuda()

            ''' Get the training triple {source object, target object, refer object (warped source object), theta_GT}'''
            tnf_batch = pair_generation_tnf(batch, None, None)

            ''' Val the model '''
            batch_1 = {'source_image': tnf_batch['source_image'], 'target_image': tnf_batch['target_image']}
            theta_st = model(batch_1)  # from source image to target image
            warped_image_tps = tpsTnf(batch_1['source_image'], theta_st)
            batch_2 = {'source_image': warped_image_tps, 'target_image': tnf_batch['refer_image']}
            theta = model(batch_2)  # from warped image to refer image
            loss = loss_fn(theta, tnf_batch['theta_GT'])
            val_loss += loss.data.cpu().numpy()

    end = time.time()
    val_loss /= len(dataloader)
    print('Val set: Average loss: {:.4f}\t\tTime cost: {:.4f}'.format(val_loss, end - start))
    return val_loss

def train_object_pfpascal_2(epoch, model, fasterRCNN, loss_fn, optimizer, dataloader, pair_generation_tnf, tpsTnf,
                            use_cuda=True, log_interval=100):
    """

        Train the model with synthetically training triple:
        {source object, target object, refer object (warped source object), theta_GT} from PF-PASCAL.
        1. Train the transformation parameters theta_st from source object to target object;
        2. Warp source object with theta_st to obtain warped object;
        3. Train the transformation parameters theta from warped object to refer object;
        4. Compute loss between theta and theta_GT.

        :param tpsTnf: the transformation to warp source object with theta_st

    """
    fasterRCNN.eval()
    thresh = 0.05
    max_per_image = 50
    model.train()
    train_loss = 0
    start = time.time()
    begin = time.time()
    for batch_idx, batch in enumerate(dataloader):
        ''' Move input batch to gpu '''
        # batch['source_image'].shape & batch['target_image'].shape: (batch_size, 3, 240, 240)
        # batch['theta'].shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        for k, v in batch.items():
            if use_cuda and batch[k].is_cuda == False:
                batch[k] = batch[k].cuda()

        ''' Get the bounding box of the stand-out object in source image and target image'''
        with torch.no_grad():
            rois_s, cls_prob_s, bbox_pred_s, _, _, _, _, _ = fasterRCNN(batch['source_im'], batch['source_im_info'],
                                                                        batch['source_gt_boxes'],
                                                                        batch['source_num_boxes'])
            all_boxes_s = select_boxes(rois_s, cls_prob_s, bbox_pred_s, batch['source_im_info'], thresh, max_per_image)

            rois_t, cls_prob_t, bbox_pred_t, _, _, _, _, _ = fasterRCNN(batch['target_im'], batch['target_im_info'],
                                                                        batch['target_gt_boxes'],
                                                                        batch['target_num_boxes'])
            all_boxes_t = select_boxes(rois_t, cls_prob_t, bbox_pred_t, batch['target_im_info'], thresh, max_per_image)

            # Select the bounding box with the highest score in the source image
            # Select tht bounding box with the same class as the above box in the target image
            # If no selected bounding box, make empty box
            # boxes.shape: (batch_size, 4), 4: (x_min, y_min, x_max, y_max)
            boxes_s, boxes_t = select_box(all_boxes_s, all_boxes_t)

            boxes_s.requires_grad = False
            boxes_t.requires_grad = False

        ''' Get the training triple {source object, target object, refer object (warped source object), theta_GT}'''
        tnf_batch = pair_generation_tnf(batch, boxes_s, boxes_t)

        ''' Train the model '''
        optimizer.zero_grad()
        batch_1 = {'source_image': tnf_batch['source_image'], 'target_image': tnf_batch['target_image']}
        # Predict tps parameters between images
        # theta.shape: (batch_size, 18) for tps
        theta_st = model(batch_1)   # from source image to target image
        # Warp source image with theta_st, and make new training pair {warped image ,refer image}
        warped_image_tps = tpsTnf(batch_1['source_image'], theta_st)
        batch_2 = {'source_image': warped_image_tps, 'target_image': tnf_batch['refer_image']}
        theta = model(batch_2)   # from warped image to refer image
        loss = loss_fn(theta, tnf_batch['theta_GT'])
        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy()
        if batch_idx % log_interval == 0:
            end = time.time()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}\t\tTime cost: {:.6f}'.format(
                epoch, batch_idx, len(dataloader),
                100. * batch_idx / len(dataloader), loss.item(), end - start))
            start = time.time()

        # warped_image_tps_2 = tpsTnf(batch_2['source_image'], theta)
        # show_images(tnf_batch, warped_image_tps.detach(), warped_image_tps_2.detach())

    end = time.time()
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}\t\tTime cost: {:.4f}'.format(train_loss, end - begin))
    return train_loss

def val_object_pfpascal_2(model, fasterRCNN, loss_fn, dataloader, pair_generation_tnf, tpsTnf, use_cuda=True):
    """

        Val the model with synthetically training triple:
        {source object, target object, refer object (warped source object), theta_GT} from PF-PASCAL.
        1. Train the transformation parameters theta_st from source object to target object;
        2. Warp source object with theta_st to obtain warped object;
        3. Train the transformation parameters theta from warped object to refer object;
        4. Compute loss between theta and theta_GT.

        :param tpsTnf: the transformation to warp source object with theta_st

    """

    fasterRCNN.eval()
    thresh = 0.05
    max_per_image = 50
    model.eval()
    val_loss = 0
    start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            ''' Move input batch to gpu '''
            for k, v in batch.items():
                if use_cuda and batch[k].is_cuda == False:
                    batch[k] = batch[k].cuda()

            ''' Get the bounding box of the stand-out object in source image and target image'''
            rois_s, cls_prob_s, bbox_pred_s, _, _, _, _, _ = fasterRCNN(batch['source_im'], batch['source_im_info'],
                                                                        batch['source_gt_boxes'],
                                                                        batch['source_num_boxes'])
            all_boxes_s = select_boxes(rois_s, cls_prob_s, bbox_pred_s, batch['source_im_info'], thresh, max_per_image)

            rois_t, cls_prob_t, bbox_pred_t, _, _, _, _, _ = fasterRCNN(batch['target_im'], batch['target_im_info'],
                                                                        batch['target_gt_boxes'],
                                                                        batch['target_num_boxes'])
            all_boxes_t = select_boxes(rois_t, cls_prob_t, bbox_pred_t, batch['target_im_info'], thresh, max_per_image)

            boxes_s, boxes_t = select_box(all_boxes_s, all_boxes_t)

            ''' Get the training triple {source object, target object, refer object (warped source object), theta_GT}'''
            tnf_batch = pair_generation_tnf(batch, boxes_s, boxes_t)

            ''' Val the model '''
            batch_1 = {'source_image': tnf_batch['source_image'], 'target_image': tnf_batch['target_image']}
            theta_st = model(batch_1)  # from source image to target image
            warped_image_tps = tpsTnf(batch_1['source_image'], theta_st)
            batch_2 = {'source_image': warped_image_tps, 'target_image': tnf_batch['refer_image']}
            theta = model(batch_2)  # from warped image to refer image
            loss = loss_fn(theta, tnf_batch['theta_GT'])
            val_loss += loss.data.cpu().numpy()

    end = time.time()
    val_loss /= len(dataloader)
    print('Val set: Average loss: {:.4f}\t\tTime cost: {:.4f}'.format(val_loss, end - start))
    return val_loss

def train_object_pfpascal_1(epoch, model, fasterRCNN, loss_fn, optimizer, dataloader, pair_generation_tnf, tpsTnf,
                            use_cuda=True, log_interval=100):
    """

        Train the model with synthetically training triple:
        {source object, target object, refer object (warped source object), theta_GT} from PF-PASCAL.
        1. Train the transformation parameters theta_st from source object to target object;
        2. Train the transformation parameters theta_tr from target object to refer object;
        3. Combine theta_st and theta_st to obtain theta from source object to refer object, and compute loss between
        theta and theta_GT.

        :param pair_generation_tnf: the function for generating training triples of the model

    """

    fasterRCNN.eval()
    thresh = 0.05
    max_per_image = 50
    model.train()
    train_loss = 0
    start = time.time()
    begin = time.time()
    for batch_idx, batch in enumerate(dataloader):
        ''' Move input batch to gpu '''
        for k, v in batch.items():
            if use_cuda and batch[k].is_cuda == False:
                batch[k] = batch[k].cuda()

        ''' Get the bounding box of the stand-out object in source image and target image'''
        with torch.no_grad():
            rois_s, cls_prob_s, bbox_pred_s, _, _, _, _, _ = fasterRCNN(batch['source_im'], batch['source_im_info'],
                                                                        batch['source_gt_boxes'], batch['source_num_boxes'])
            all_boxes_s = select_boxes(rois_s, cls_prob_s, bbox_pred_s, batch['source_im_info'], thresh, max_per_image)

            rois_t, cls_prob_t, bbox_pred_t, _, _, _, _, _ = fasterRCNN(batch['target_im'], batch['target_im_info'],
                                                                        batch['target_gt_boxes'], batch['target_num_boxes'])
            all_boxes_t = select_boxes(rois_t, cls_prob_t, bbox_pred_t, batch['target_im_info'], thresh, max_per_image)

            # Select the bounding box with the highest score in the source image
            # Select tht bounding box with the same class as the above box in the target image
            # If no selected bounding box, make empty box
            # boxes.shape: (batch_size, 4), 4: (x_min, y_min, x_max, y_max)
            boxes_s, boxes_t = select_box(all_boxes_s, all_boxes_t)

            boxes_s.requires_grad = False
            boxes_t.requires_grad = False

        ''' Get the training triple {source object, target object, refer object (warped source object), theta_GT}'''
        tnf_batch = pair_generation_tnf(batch, boxes_s, boxes_t)

        batch_1 = {'source_image': tnf_batch['source_image'], 'target_image': tnf_batch['target_image']}
        batch_2 = {'source_image': tnf_batch['target_image'], 'target_image': tnf_batch['refer_image']}

        ''' Train the model '''
        optimizer.zero_grad()
        # Predict tps parameters between images
        # theta.shape: (batch_size, 18) for tps
        theta_st = model(batch_1)  # from source image to target image
        theta_tr = model(batch_2)  # from target image to refer image
        loss = loss_fn(theta_st, theta_tr, tnf_batch['theta_GT'])
        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy()
        if batch_idx % log_interval == 0:
            end = time.time()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}\t\tTime cost: {:.6f}'.format(
                epoch, batch_idx, len(dataloader),
                100. * batch_idx / len(dataloader), loss.item(), end - start))
            start = time.time()
        # warped_image_tps = tpsTnf(batch_1['source_image'], theta_st)
        # warped_image_tps_2 = tpsTnf(batch_2['source_image'], theta_tr)
        # warped_image_tps_3 = tpsTnf(warped_image_tps, theta_tr)
        # show_images(tnf_batch, warped_image_tps.detach(), warped_image_tps_2.detach(), warped_image_tps_3.detach())

    end = time.time()
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}\t\tTime cost: {:.4f}'.format(train_loss, end - begin))
    return train_loss

def val_object_pfpascal_1(model, fasterRCNN, loss_fn, dataloader, pair_generation_tnf, use_cuda=True):
    """

        Val the model with synthetically training triple:
        {source object, target object, refer object (warped source object), theta_GT} from PF-PASCAL.
        1. Train the transformation parameters theta_st from source object to target object;
        2. Train the transformation parameters theta_tr from target object to refer object;
        3. Combine theta_st and theta_st to obtain theta from source object to refer object, and compute loss between
        theta and theta_GT.

        :param pair_generation_tnf: the function for generating training triples of the model

    """

    fasterRCNN.eval()
    thresh = 0.05
    max_per_image = 50
    model.eval()
    val_loss = 0
    start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            ''' Move input batch to gpu '''
            for k, v in batch.items():
                if use_cuda and batch[k].is_cuda == False:
                    batch[k] = batch[k].cuda()

            ''' Get the bounding box of the stand-out object in source image and target image'''
            rois_s, cls_prob_s, bbox_pred_s, _, _, _, _, _ = fasterRCNN(batch['source_im'], batch['source_im_info'],
                                                                        batch['source_gt_boxes'],
                                                                        batch['source_num_boxes'])
            all_boxes_s = select_boxes(rois_s, cls_prob_s, bbox_pred_s, batch['source_im_info'], thresh,
                                       max_per_image)

            rois_t, cls_prob_t, bbox_pred_t, _, _, _, _, _ = fasterRCNN(batch['target_im'], batch['target_im_info'],
                                                                        batch['target_gt_boxes'],
                                                                        batch['target_num_boxes'])
            all_boxes_t = select_boxes(rois_t, cls_prob_t, bbox_pred_t, batch['target_im_info'], thresh,
                                       max_per_image)

            boxes_s, boxes_t = select_box(all_boxes_s, all_boxes_t)

            ''' Get the training triple {source object, target object, refer object (warped source object), theta_GT}'''
            tnf_batch = pair_generation_tnf(batch, boxes_s, boxes_t)

            batch_1 = {'source_image': tnf_batch['source_image'], 'target_image': tnf_batch['target_image']}
            batch_2 = {'source_image': tnf_batch['target_image'], 'target_image': tnf_batch['refer_image']}

            ''' Val the model '''
            theta_st = model(batch_1)  # from source image to target image
            theta_tr = model(batch_2)  # from target image to refer image
            loss = loss_fn(theta_st, theta_tr, tnf_batch['theta_GT'])
            val_loss += loss.data.cpu().numpy()

    end = time.time()
    val_loss /= len(dataloader)
    print('Val set: Average loss: {:.4f}\t\tTime cost: {:.4f}'.format(val_loss, end - start))
    return val_loss

def train_object_synth(epoch, model, fasterRCNN, loss_fn, optimizer, dataloader, pair_generation_tnf, use_cuda=True,
                 log_interval=100):
    """

        Train the model with synthetically training pairs {source object, target object (warped source object), theta_GT}
        from PascalVOC2011
        :param epoch: which epoch for training
        :param model: the geometric matching model
        :param fasterRCNN: the fatserRCNN model for detecting objects in images
        :param loss_fn: grid loss function
        :param optimizer: the optimizer for parameters of the model
        :param dataloader: dataloader for training set
        :param pair_generation_tnf: the function for generating training pairs of the model
        :param use_cuda: whether gpu is available
        :param log_interval: the interval for displaying running results
        :return: the training loss for this epoch

    """

    fasterRCNN.eval()
    thresh = 0.05
    max_per_image = 50
    model.train()
    train_loss = 0
    start = time.time()
    begin = time.time()
    for batch_idx, batch in enumerate(dataloader):
        ''' Move input batch to gpu '''
        # batch['image'].shape: (batch_size, 3, 480, 640)
        # batch['theta'].shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        # batch['theta'].shape-affine: (batch_size, 2, 3)
        for k, v in batch.items():
            if use_cuda and batch[k].is_cuda == False:
                batch[k] = batch[k].cuda()

        ''' Get the bounding box of the stand-out object in the image '''
        with torch.no_grad():
            # rois.shape: (batch_size, 300, 5), 5: (image_index_in_batch, x_min, y_min, x_max, y_max),
            # the coordinates is on the resized image (240*240), not the original image
            # cls_prob.shape: (batch_size, 300, n_classes), for PascalVOC n_classes=21
            # bbox_pred.shape: (batch_size, 300, 4 * n_classes), 4: (tx, ty, tw, th)
            rois, cls_prob, bbox_pred, _, _, _, _, _ = fasterRCNN(batch['im'], batch['im_info'],
                                                                  batch['gt_boxes'], batch['num_boxes'])
            # Compute and select bounding boxes for objects in the image
            all_boxes = select_boxes(rois, cls_prob, bbox_pred, batch['im_info'], thresh, max_per_image)
            # Select the bounding box with the highest score in the image
            # boxes.shape: (batch_size, 4), 4: (x_min, y_min, x_max, y_max)
            boxes = select_box_single(all_boxes)
            boxes.requires_grad = False

        ''' Get the training pair {source object, target object (warped source object), theta_GT}'''
        # tnf_batch['source_image'].shape and tnf_batch['target_image'].shape: (batch_size, 3, 240, 240)
        # tnf_batch['theta_GT'].shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        # tnf_batch['theta_GT'].shape-affine: (batch_size, 2, 3)
        tnf_batch = pair_generation_tnf(batch, boxes)

        '''
        # Show the bounding box with the highest score in the image
        for i in range(tnf_batch['source_image'].shape[0]):
            show_id = i

            ax = im_show_2(batch['im'][show_id], 'im', 2, 2, 1)
            show_boxes(ax, all_boxes[show_id])

            ax = im_show_2(batch['im'][show_id], 'im', 2, 2, 2)
            show_boxes(ax, boxes[show_id, :].reshape(1, -1))

            im_show_1(tnf_batch['source_image'][show_id], 'source_image', 2, 2, 3)

            im_show_1(tnf_batch['target_image'][show_id], 'target_image', 2, 2, 4)

            plt.show()
        '''

        ''' Train the model '''
        optimizer.zero_grad()
        # theta.shape: (batch_size, 18) for tps, (batch_size, 6) for affine
        theta = model(tnf_batch)
        loss = loss_fn(theta, tnf_batch['theta_GT'])
        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy()
        if batch_idx % log_interval == 0:
            end = time.time()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}\t\tTime cost: {:.6f}'.format(
                epoch, batch_idx, len(dataloader),
                100. * batch_idx / len(dataloader), loss.item(), end - start))
            start = time.time()
    end = time.time()
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}\t\tTime cost: {:.4f}'.format(train_loss, end - begin))
    return train_loss

def val_object_synth(model, fasterRCNN, loss_fn, dataloader, pair_generation_tnf, use_cuda=True):
    """

        Val the model with synthetically training pairs {source object, target object (warped source object), theta_GT}
        from PascalVOC2011.
        :param model: the geometric matching model
        :param fasterRCNN: the fatserRCNN model for detecting objects in images
        :param loss_fn: grid loss function
        :param dataloader: dataloader for validation set
        :param pair_generation_tnf: the function for generating training pairs of the model
        :param use_cuda: whether gpu is available
        :return: the validation loss for this epoch

    """

    fasterRCNN.eval()
    thresh = 0.05
    max_per_image = 50
    model.eval()
    val_loss = 0
    start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            ''' Move input batch to gpu '''
            for k, v in batch.items():
                if use_cuda and batch[k].is_cuda == False:
                    batch[k] = batch[k].cuda()

            ''' Get the bounding box of the stand-out object in the image '''
            rois, cls_prob, bbox_pred, _, _, _, _, _ = fasterRCNN(batch['im'], batch['im_info'],
                                                                  batch['gt_boxes'], batch['num_boxes'])
            all_boxes = select_boxes(rois, cls_prob, bbox_pred, batch['im_info'], thresh, max_per_image)
            boxes = select_box_single(all_boxes)

            ''' Get the training pair {source object, target object (warped source object), theta_GT}'''
            tnf_batch = pair_generation_tnf(batch, boxes)

            ''' Val the model '''
            theta = model(tnf_batch)
            loss = loss_fn(theta, tnf_batch['theta_GT'])
            val_loss += loss.data.cpu().numpy()

    end = time.time()
    val_loss /= len(dataloader)
    print('Val set: Average loss: {:.4f}\t\tTime cost: {:.4f}'.format(val_loss, end - start))
    return val_loss


def train_pool4_synth(epoch, model, fasterRCNN, loss_fn, optimizer, dataloader, pair_generation_tnf, use_cuda=True,
                log_interval=100):
    """

        Train the model with synthetically training pairs {source image, target image (warped source image), theta_GT,
        source box, target box} from PascalVOC2011.
        On the layer pool4 of vgg16, the feature map of the stand-out object on the image is cropped and resized with
        the bounding box.

    """

    fasterRCNN.eval()
    thresh = 0.05
    max_per_image = 50
    model.train()
    train_loss = 0
    start = time.time()
    begin = time.time()
    for batch_idx, batch in enumerate(dataloader):
        ''' Get the training pair {source image, target image (warped source image), theta_GT}, and input of fasterRCNN '''
        tnf_batch, roi_batch = pair_generation_tnf(batch, None)

        ''' Move input batch of fasterRCNN to gpu '''
        for k, v in roi_batch.items():
            if use_cuda and roi_batch[k].is_cuda == False:
                roi_batch[k] = roi_batch[k].cuda()

        ''' Get the bounding box of the stand-out object in source image and target image'''
        with torch.no_grad():
            rois_s, cls_prob_s, bbox_pred_s, _, _, _, _, _ = fasterRCNN(roi_batch['source_im'], roi_batch['im_info'],
                                                                        roi_batch['gt_boxes'], roi_batch['num_boxes'])
            all_boxes_s = select_boxes(rois_s, cls_prob_s, bbox_pred_s, roi_batch['im_info'], thresh, max_per_image)

            rois_t, cls_prob_t, bbox_pred_t, _, _, _, _, _ = fasterRCNN(roi_batch['target_im'], roi_batch['im_info'],
                                                                        roi_batch['gt_boxes'], roi_batch['num_boxes'])
            all_boxes_t = select_boxes(rois_t, cls_prob_t, bbox_pred_t, roi_batch['im_info'], thresh, max_per_image)

            # Select the bounding box with the highest score in the source image
            # Select tht bounding box with the same class as the above box in the target image
            # If no selected bounding box, make empty box
            # boxes_s, boxes_t.shape: (batch_size, 4), 4: (x_min, y_min, x_max, y_max)
            boxes_s, boxes_t = select_box(all_boxes_s, all_boxes_t)

            boxes_s.requires_grad = False
            boxes_t.requires_grad = False

            '''
            # Show the bounding box with the highest score in the image
            rows = 1
            cols = 2
            for i in range(tnf_batch['source_image'].shape[0]):
                subplot_idx = 1
                ax_1 = im_show_1(tnf_batch['source_image'][i], 'source_image', rows, cols, subplot_idx)
                show_boxes(ax_1, boxes_s[i, :].reshape(1, -1))
                subplot_idx += 1
                # ax_2 = im_show_1(warped_image_aff[0], 'warped_image_aff', rows, cols, subplot_idx)
                # subplot_idx += 1
                # ax_3 = im_show_1(warped_image_tps[show_id], 'warped_image_tps', rows, cols, subplot_idx)
                # subplot_idx += 1
                ax_4 = im_show_1(tnf_batch['target_image'][i], 'target_image', rows, cols, subplot_idx)
                show_boxes(ax_4, boxes_t[i, :].reshape(1, -1))

                plt.show()
            '''

        ''' Get the training pair {source image, target image (warped source image), theta_GT, source_box, target_box} '''
        # Add the bounding box of the stand-out object in source image and target image to the training pair
        tnf_batch['source_box'] = boxes_s
        tnf_batch['target_box'] = boxes_t

        ''' Move input batch to gpu '''
        for k, v in tnf_batch.items():
            if use_cuda and tnf_batch[k].is_cuda == False:
                tnf_batch[k] = tnf_batch[k].cuda()

        ''' Train the model '''
        optimizer.zero_grad()
        theta = model(tnf_batch)
        loss = loss_fn(theta, tnf_batch['theta_GT'])
        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy()
        if batch_idx % log_interval == 0:
            end = time.time()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}\t\tTime cost: {:.6f}'.format(
                epoch, batch_idx, len(dataloader),
                100. * batch_idx / len(dataloader), loss.item(), end - start))
            start = time.time()
    end = time.time()
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}\t\tTime cost: {:.4f}'.format(train_loss, end - begin))
    return train_loss


def val_pool4_synth(model, fasterRCNN, loss_fn, dataloader, pair_generation_tnf, use_cuda=True):
    """

        Val the model with synthetically training pairs {source image, target image (warped source image), theta_GT,
        source box, target box} from PascalVOC2011.
        On the layer pool4 of vgg16, the feature map of the stand-out object on the image is cropped and resized with
        the bounding box.

    """

    fasterRCNN.eval()
    thresh = 0.05
    max_per_image = 50
    model.eval()
    val_loss = 0
    start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            ''' Get the training pair {source image, target image (warped source image), theta_GT}, and input of fasterRCNN '''
            tnf_batch, roi_batch = pair_generation_tnf(batch, None)

            ''' Move input batch of fasterRCNN to gpu '''
            for k, v in roi_batch.items():
                if use_cuda and roi_batch[k].is_cuda == False:
                    roi_batch[k] = roi_batch[k].cuda()

            ''' Get the bounding box of the stand-out object in source image and target image'''
            rois_s, cls_prob_s, bbox_pred_s, _, _, _, _, _ = fasterRCNN(roi_batch['source_im'], roi_batch['im_info'],
                                                                        roi_batch['gt_boxes'], roi_batch['num_boxes'])
            all_boxes_s = select_boxes(rois_s, cls_prob_s, bbox_pred_s, roi_batch['im_info'], thresh, max_per_image)

            rois_t, cls_prob_t, bbox_pred_t, _, _, _, _, _ = fasterRCNN(roi_batch['target_im'], roi_batch['im_info'],
                                                                        roi_batch['gt_boxes'], roi_batch['num_boxes'])
            all_boxes_t = select_boxes(rois_t, cls_prob_t, bbox_pred_t, roi_batch['im_info'], thresh, max_per_image)

            boxes_s, boxes_t = select_box(all_boxes_s, all_boxes_t)

            ''' Get the training pair {source image, target image (warped source image), theta_GT, source_box, target_box} '''
            tnf_batch['source_box'] = boxes_s
            tnf_batch['target_box'] = boxes_t

            ''' Move input batch to gpu '''
            for k, v in tnf_batch.items():
                if use_cuda and tnf_batch[k].is_cuda == False:
                    tnf_batch[k] = tnf_batch[k].cuda()

            ''' Val the model '''
            theta = model(tnf_batch)
            loss = loss_fn(theta, tnf_batch['theta_GT'])
            val_loss += loss.data.cpu().numpy()

    end = time.time()
    val_loss /= len(dataloader)
    print('Val set: Average loss: {:.4f}\t\tTime cost: {:.4f}'.format(val_loss, end - start))
    return val_loss