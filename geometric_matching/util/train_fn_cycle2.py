# ========================================================================================
# Train and evaluate geometric matching model
# Author: Jingwei Qu
# Date: 02 October 2019
# ========================================================================================

from __future__ import print_function, division
import torch
import torchvision
import time
import cv2

from geometric_matching.geotnf.transformation import GeometricTnf
from geometric_matching.util.net_util import *
from geometric_matching.image.normalization import normalize_image

def add_watch(watch_images, image_names, batch, geoTnf, theta_st, theta_tr, k):
    warped_image_st = geoTnf(batch['source_image'][0].unsqueeze(0), theta_st[0].unsqueeze(0))
    warped_image_tr = geoTnf(batch['target_image'][0].unsqueeze(0), theta_tr[0].unsqueeze(0))
    warped_image_sr = geoTnf(warped_image_st, theta_tr[0].unsqueeze(0))

    watch_images[k, :, 0:240, :] = batch['source_image'][0]
    watch_images[k + 1, :, 0:240, :] = warped_image_st.detach()
    watch_images[k + 2, :, 0:240, :] = batch['target_image'][0]
    watch_images[k + 3, :, 0:240, :] = warped_image_tr.detach()
    watch_images[k + 4, :, 0:240, :] = batch['refer_image'][0]
    watch_images[k + 5, :, 0:240, :] = warped_image_sr.detach()

    image_names.append('Source')
    image_names.append('Warped_st')
    image_names.append('Target')
    image_names.append('Warped_tr')
    image_names.append('Refer')
    image_names.append('Warped_sr')

    return watch_images, image_names

def train_fn(epoch, model, loss_fn, loss_cycle_fn, lambda_c, optimizer, dataloader, triple_generation, geometric_model='tps', use_cuda=True,
             log_interval=100, vis=None):
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
        group_size = 6
        watch_images = torch.ones(group_size * 4, 3, 260, 240).cuda()
        image_names = list()
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
        theta_st, theta_ts, theta_tr, theta_rt = model(batch_triple)
        loss_match = loss_fn(theta_st=theta_st, theta_tr=theta_tr, theta_GT=batch_triple['theta_GT'])
        loss_cycle_st = loss_cycle_fn(theta_AB=theta_st, theta_BA=theta_ts)
        loss_cycle_ts = loss_cycle_fn(theta_AB=theta_ts, theta_BA=theta_st)
        loss_cycle_tr = loss_cycle_fn(theta_AB=theta_tr, theta_BA=theta_rt)
        loss_cycle_rt = loss_cycle_fn(theta_AB=theta_rt, theta_BA=theta_tr)
        loss = loss_match + lambda_c * (loss_cycle_st + loss_cycle_ts + loss_cycle_tr + loss_cycle_rt) / 4
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if (batch_idx+1) % log_interval == 0:
            end = time.time()
            print('Train epoch: {} [{}/{} ({:.0%})]\t\tCurrent batch loss: {:.6f}\t\tTime cost ({} batches): {:.4f} s'
                  .format(epoch, batch_idx+1, len(dataloader), (batch_idx+1) / len(dataloader), loss.item(), batch_idx + 1, end - begin))

        if (epoch % 5 == 0 or epoch == 1) and vis is not None:
            if (batch_idx + 1) % stride_images == 0 or batch_idx == 0:
                watch_images, image_names = add_watch(watch_images, image_names, batch_triple, geoTnf, theta_st,
                                                      theta_tr, int((batch_idx + 1) / stride_images) * group_size)

            # if batch_idx <= 3:
            #     watch_images, image_names = add_watch(watch_images, image_names, batch_triple, geoTnf, theta_st,
            #                                           theta_tr, batch_idx * group_size)
            #     if batch_idx == 3:
            #         opts = dict(jpgquality=100, title='Epoch ' + str(epoch) + ' source warped_sr target warped_tr refer warped_sr')
            #         watch_images[:, :, 0:240, :] = normalize_image(watch_images[:, :, 0:240, :], forward=False)
            #         watch_images *= 255.0
            #         watch_images = watch_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            #         for i in range(watch_images.shape[0]):
            #             cv2.putText(watch_images[i], image_names[i], (80, 255), fnt, 0.5, (0, 0, 0), 1)
            #         watch_images = torch.Tensor(watch_images.astype(np.float32))
            #         watch_images = watch_images.permute(0, 3, 1, 2)
            #         vis.image(torchvision.utils.make_grid(watch_images, nrow=group_size, padding=3), opts=opts)

            if (batch_idx + 1) % stride_loss == 0 or batch_idx == 0:
                iter_loss[int((batch_idx + 1) / stride_loss)] = epoch_loss / (batch_idx + 1)

    end = time.time()

    # Visualize watch images & train loss
    if (epoch % 5 == 0 or epoch == 1) and vis is not None:
        opts = dict(jpgquality=100, title='Epoch ' + str(epoch) + ' source warped_sr target warped_tr refer warped_sr')
        watch_images[:, :, 0:240, :] = normalize_image(watch_images[:, :, 0:240, :], forward=False)
        watch_images *= 255.0
        watch_images = watch_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        for i in range(watch_images.shape[0]):
            cv2.putText(watch_images[i], image_names[i], (80, 255), fnt, 0.5, (0, 0, 0), 1)
        watch_images = torch.Tensor(watch_images.astype(np.float32))
        watch_images = watch_images.permute(0, 3, 1, 2)
        vis.image(torchvision.utils.make_grid(watch_images, nrow=group_size, padding=3), opts=opts)

        opts_loss = dict(xlabel='Iterations (' + str(stride_loss) + ')',
                         ylabel='Loss',
                         title='GM ResNet101 ' + geometric_model + ' Training Loss in Epoch ' + str(epoch),
                         legend=['Loss'],
                         width=2000)
        vis.line(iter_loss, np.arange(106), opts=opts_loss)

    epoch_loss /= len(dataloader)
    print('Train set -- Average loss: {:.6f}\t\tTime cost: {:.4f}'.format(epoch_loss, end - begin))
    return epoch_loss, end - begin