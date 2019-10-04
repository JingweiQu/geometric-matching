# ========================================================================================
# Train and evaluate geometric matching model
# Author: Jingwei Qu
# Date: 01 June 2019
# ========================================================================================

from __future__ import print_function, division
import torch
import torchvision
import time
import cv2

from geometric_matching.geotnf.transformation_tps import GeometricTnf
from geometric_matching.util.net_util import *
from geometric_matching.image.normalization import normalize_image
from geometric_matching.geotnf.point_tnf import PointsToPixelCoords

import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def draw_grid(image, cp_norm):
    im_size = torch.Tensor([[240, 240]]).cuda()
    cp = PointsToPixelCoords(P=cp_norm, im_size=im_size)
    cp = cp.squeeze().cpu().numpy() + 50
    for j in range(9):
        cv2.drawMarker(image, (cp[0, j], cp[1, j]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)

    for j in range(2):
        for k in range(3):
            # vertical grid
            cv2.line(image, (cp[0, j + k * 3], cp[1, j + k * 3]), (cp[0, j + k * 3 + 1], cp[1, j + k * 3 + 1]), (0, 0, 255), 2, cv2.LINE_AA)
            # horizontal grid
            cv2.line(image, (cp[0, j * 3 + k], cp[1, j * 3 + k]), (cp[0, j * 3 + k + 3], cp[1, j * 3 + k + 3]), (0, 0, 255), 2, cv2.LINE_AA)

    return image

def add_watch(watch_images, watch_theta, batch_st, batch_tr, geoTnf, theta_st, theta_tr, k, l):
    warped_image_st = geoTnf(batch_st['source_image'][0].unsqueeze(0), theta_st[0].unsqueeze(0))
    warped_image_tr = geoTnf(batch_tr['source_image'][0].unsqueeze(0), theta_tr[0].unsqueeze(0))
    warped_image_sr = geoTnf(warped_image_st, theta_tr[0].unsqueeze(0))

    watch_images[k, :, 50:290, 50:290] = batch_st['source_image'][0]
    watch_images[k + 1, :, 50:290, 50:290] = warped_image_st.detach()
    watch_images[k + 2, :, 50:290, 50:290] = batch_st['target_image'][0]

    watch_images[k + 3, :, 50:290, 50:290] = batch_tr['source_image'][0]
    watch_images[k + 4, :, 50:290, 50:290] = warped_image_tr.detach()
    watch_images[k + 5, :, 50:290, 50:290] = batch_tr['target_image'][0]

    watch_images[k + 6, :, 50:290, 50:290] = batch_st['source_image'][0]
    watch_images[k + 7, :, 50:290, 50:290] = warped_image_sr.detach()
    watch_images[k + 8, :, 50:290, 50:290] = batch_tr['target_image'][0]

    watch_theta[l*2, :] = theta_st[0].detach()
    watch_theta[l*2+1, :] = theta_tr[0].detach()

    return watch_images, watch_theta

def train_fn(epoch, model, loss_fn, loss_cycle_fn, loss_jitter_fn, lambda_c, lambda_j, optimizer, dataloader,
             triple_generation, geometric_model='tps', use_cuda=True, log_interval=100, vis=None, show=False):
    """
        Train the model with synthetically training triple:
        {source image, target image, refer image (warped source image), theta_GT} from PF-PASCAL.
        1. Train the transformation parameters theta_st from source image to target image;
        2. Train the transformation parameters theta_tr from target image to refer image;
        3. Combine theta_st and theta_st to obtain theta from source image to refer image, and compute loss between
        theta and theta_GT.
    """

    geoTnf = GeometricTnf(geometric_model=geometric_model, use_cuda=use_cuda)
    epoch_loss = 0
    if (epoch % 5 == 0 or epoch == 1) and vis is not None:
        stride_images = len(dataloader) / 3
        group_size = 9
        watch_images = torch.ones(group_size * 4, 3, 340, 340).cuda()
        watch_theta = torch.zeros(8, 36).cuda()
        fnt = cv2.FONT_HERSHEY_COMPLEX
        stride_loss = len(dataloader) / 105
        iter_loss = np.zeros(106)
    begin = time.time()
    for batch_idx, batch in enumerate(dataloader):
        ''' Move input batch to gpu '''
        # batch['source_image'].shape & batch['target_image'].shape: (batch_size, 3, 240, 240)
        # batch['theta'].shape-tps: (batch_size, 18)-random or (batch_size, 18, 1, 1)-(pre-set from csv)
        if use_cuda:
            batch = batch_cuda(batch)

        ''' Get the training triple {source image, target image, refer image (warped source image), theta_GT}'''
        batch_triple = triple_generation(batch)

        ''' Train the model '''
        optimizer.zero_grad()
        loss = 0
        # Predict tps parameters between images
        # theta.shape: (batch_size, 18) for tps, theta.shape: (batch_size, 6) for affine
        batch_st = {'source_image': batch_triple['source_image'], 'target_image': batch_triple['target_image']}
        batch_tr = {'source_image': batch_triple['target_image'], 'target_image': batch_triple['refer_image']}
        theta_st, theta_ts = model(batch_st)  # from source image to target image
        theta_tr, theta_rt = model(batch_tr)  # from target image to refer image
        loss_match = loss_fn(theta_st=theta_st, theta_tr=theta_tr, theta_GT=batch_triple['theta_GT'])
        loss_cycle_st = loss_cycle_fn(theta_AB=theta_st, theta_BA=theta_ts)
        loss_cycle_ts = loss_cycle_fn(theta_AB=theta_ts, theta_BA=theta_st)
        loss_cycle_tr = loss_cycle_fn(theta_AB=theta_tr, theta_BA=theta_rt)
        loss_cycle_rt = loss_cycle_fn(theta_AB=theta_rt, theta_BA=theta_tr)
        loss_jitter = loss_jitter_fn(theta_st=theta_st, theta_tr=theta_tr)
        loss = loss_match + lambda_c * (loss_cycle_st + loss_cycle_ts + loss_cycle_tr + loss_cycle_rt) / 4 + lambda_j * loss_jitter
        # loss = loss_match + lambda_c * (loss_cycle_st + loss_cycle_ts + loss_cycle_tr + loss_cycle_rt) / 4
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if (batch_idx+1) % log_interval == 0:
            end = time.time()
            print('Train epoch: {} [{}/{} ({:.0%})]\t\tCurrent batch loss: {:.6f}\t\tTime cost ({} batches): {:.4f} s'
                  .format(epoch, batch_idx+1, len(dataloader), (batch_idx+1) / len(dataloader), loss.item(), batch_idx + 1, end - begin))

        if (epoch % 5 == 0 or epoch == 1) and vis is not None:
            if (batch_idx + 1) % stride_images == 0 or batch_idx == 0:
                watch_images, watch_theta = add_watch(watch_images, watch_theta, batch_st, batch_tr, geoTnf, theta_st, theta_tr, int((batch_idx + 1) / stride_images) * group_size, int((batch_idx + 1) / stride_images))

            # if batch_idx <= 19:
            #     watch_images, image_names = add_watch(watch_images, watch_theta, batch_st, batch_tr, geoTnf, theta_st, theta_tr, batch_idx * group_size, batch_idx)
            #     if batch_idx == 19:
            #         opts = dict(jpgquality=100, title='Epoch ' + str(epoch) + ' source warped_sr target warped_tr refer warped_sr')
            #         watch_images[:, :, 50:290, 50:290] = normalize_image(watch_images[:, :, 50:290, 50:290], forward=False)
            #         watch_images *= 255.0
            #         watch_images = watch_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            #         for i in range(watch_images.shape[0]):
            #             if i % group_size == 0:
            #                 cp_norm = watch_theta[int(i / group_size), :18].view(1, 2, -1)
            #                 watch_images[i] = draw_grid(watch_images[i], cp_norm)
            #
            #                 cp_norm = watch_theta[int(i / group_size), 18:].view(1, 2, -1)
            #                 watch_images[i + 1] = draw_grid(watch_images[i + 1], cp_norm)
            #
            #                 cp_norm = watch_theta[int(i / group_size) + 1, :18].view(1, 2, -1)
            #                 watch_images[i + 3] = draw_grid(watch_images[i + 3], cp_norm)
            #
            #                 cp_norm = watch_theta[int(i / group_size) + 1, 18:].view(1, 2, -1)
            #                 watch_images[i + 4] = draw_grid(watch_images[i + 4], cp_norm)
            #
            #         watch_images = torch.Tensor(watch_images.astype(np.float32))
            #         watch_images = watch_images.permute(0, 3, 1, 2)
            #         vis.image(torchvision.utils.make_grid(watch_images, nrow=3, padding=3), opts=opts)

            if (batch_idx + 1) % stride_loss == 0 or batch_idx == 0:
                iter_loss[int((batch_idx + 1) / stride_loss)] = epoch_loss / (batch_idx + 1)

        # watch_images = normalize_image(batch_tr['target_image'], forward=False) * 255.0
        # vis.images(watch_images, nrow=8, padding=3)

    end = time.time()

    # Visualize watch images & train loss
    if (epoch % 5 == 0 or epoch == 1) and vis is not None:
        opts = dict(jpgquality=100, title='Epoch ' + str(epoch) + ' source warped_sr target warped_tr refer warped_sr')
        watch_images[:, :, 50:290, 50:290] = normalize_image(watch_images[:, :, 50:290, 50:290], forward=False)
        watch_images *= 255.0
        watch_images = watch_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        for i in range(watch_images.shape[0]):
            if i % group_size == 0:
                cp_norm = watch_theta[int(i / group_size), :18].view(1, 2, -1)
                watch_images[i] = draw_grid(watch_images[i], cp_norm)

                cp_norm = watch_theta[int(i / group_size), 18:].view(1, 2, -1)
                watch_images[i + 1] = draw_grid(watch_images[i + 1], cp_norm)

                cp_norm = watch_theta[int(i / group_size) + 1, :18].view(1, 2, -1)
                watch_images[i + 3] = draw_grid(watch_images[i + 3], cp_norm)

                cp_norm = watch_theta[int(i / group_size) + 1, 18:].view(1, 2, -1)
                watch_images[i + 4] = draw_grid(watch_images[i + 4], cp_norm)

        watch_images = torch.Tensor(watch_images.astype(np.float32))
        watch_images = watch_images.permute(0, 3, 1, 2)
        vis.image(torchvision.utils.make_grid(watch_images, nrow=3, padding=3), opts=opts)

        opts_loss = dict(xlabel='Iterations (' + str(stride_loss) + ')',
                         ylabel='Loss',
                         title='GM ResNet101 ' + geometric_model + ' Training Loss in Epoch ' + str(epoch),
                         legend=['Loss'],
                         width=2000)
        vis.line(iter_loss, np.arange(106), opts=opts_loss)

    epoch_loss /= len(dataloader)
    print('Train set -- Average loss: {:.6f}\t\tTime cost: {:.4f}'.format(epoch_loss, end - begin))
    return epoch_loss, end - begin