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

from geometric_matching.util.net_util import *
from geometric_matching.image.normalization import normalize_image

def add_watch(watch_images, image_names, batch, mask_A, mask_B, k):
    watch_images[k, :, 0:240, :] = batch['source_image'][0]
    # watch_images[k + 1, :, 0:240, :] = torch.mul(batch['source_image'][0], mask_A[0].detach())
    # watch_images[k + 2, :, 0:240, :] = torch.mul(batch['source_image'][0], 1.0 - mask_A[0].detach())
    mask_A = mask_A[0].detach().gt(0.5).float()
    watch_images[k + 1, :, 0:240, :] = torch.mul(batch['source_image'][0], mask_A)
    watch_images[k + 2, :, 0:240, :] = torch.mul(batch['source_image'][0], 1.0 - mask_A)
    watch_images[k + 3, :, 0:240, :] = batch['target_image'][0]
    # watch_images[k + 4, :, 0:240, :] = torch.mul(batch['target_image'][0], mask_B[0].detach())
    # watch_images[k + 5, :, 0:240, :] = torch.mul(batch['target_image'][0], 1.0 - mask_B[0].detach())
    mask_B = mask_B[0].detach().gt(0.5).float()
    watch_images[k + 4, :, 0:240, :] = torch.mul(batch['target_image'][0], mask_B)
    watch_images[k + 5, :, 0:240, :] = torch.mul(batch['target_image'][0], 1.0 - mask_B)

    image_names.append('Source')
    image_names.append('Object')
    image_names.append('Background')
    image_names.append('Target')
    image_names.append('Object')
    image_names.append('Background')

    return watch_images, image_names

def train_fn(epoch, model, loss_fn, optimizer, dataloader, use_cuda=True, log_interval=100, vis=None):
    """
        Train cosegmentation model:
        {source image, target image} from PF-PASCAL.
        1. Train the co-object masks on source image and target image;
        2. Compute loss.
    """

    epoch_loss = 0
    # if (epoch % 5 == 0 or epoch == 1) and vis is not None:
    if vis is not None:
        stride_images = len(dataloader) / 4
        group_size = 6
        watch_images = torch.ones(group_size * 5, 3, 260, 240).cuda()
        image_names = list()
        fnt = cv2.FONT_HERSHEY_COMPLEX
        # means for normalize of caffe resnet and vgg
        # pixel_means = torch.Tensor(np.array([[[[102.9801, 115.9465, 122.7717]]]]).astype(np.float32)).cuda()
        stride_loss = len(dataloader) / 92
        iter_loss = np.zeros(93)
    begin = time.time()
    for batch_idx, batch in enumerate(dataloader):
        ''' Move input batch to gpu '''
        if use_cuda:
            batch = batch_cuda(batch)

        ''' Train the model '''
        optimizer.zero_grad()
        mask_A, mask_B = model(batch)
        loss = loss_fn(mask_A, mask_B, batch['source_image'], batch['target_image'])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if (batch_idx+1) % log_interval == 0:
            end = time.time()
            print('Train epoch: {} [{}/{} ({:.0%})]\t\tCurrent batch loss: {:.6f}\t\tTime cost ({} batches): {:.4f} s'
                  .format(epoch, batch_idx+1, len(dataloader), (batch_idx+1) / len(dataloader), loss.item(), batch_idx + 1, end - begin))

        # if (epoch % 5 == 0 or epoch == 1) and vis is not None:
        if vis is not None:
            if (batch_idx + 1) % stride_images == 0 or batch_idx == 0:
                watch_images, image_names = add_watch(watch_images, image_names, batch, mask_A, mask_B, int((batch_idx + 1) / stride_images) * group_size)

            # if batch_idx <= 4:
            #     watch_images, image_names = add_watch(watch_images, image_names, batch, mask_A, mask_B, batch_idx * group_size)
            #     if batch_idx == 4:
            #         opts = dict(jpgquality=100, title='Epoch ' + str(epoch) + ' image mask')
            #         watch_images[:, :, 0:240, :] = normalize_image(watch_images[:, :, 0:240, :], forward=False)
            #         watch_images *= 255.0
            #         watch_images = watch_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            #         for i in range(len(image_names)):
            #             cv2.putText(watch_images[i], image_names[i], (80, 255), fnt, 0.5, (0, 0, 0), 1)
            #         watch_images = torch.Tensor(watch_images.astype(np.float32))
            #         watch_images = watch_images.permute(0, 3, 1, 2)
            #         vis.image(torchvision.utils.make_grid(watch_images, nrow=group_size, padding=3), opts=opts)


            if (batch_idx + 1) % stride_loss == 0 or batch_idx == 0:
                iter_loss[int((batch_idx + 1) / stride_loss)] = epoch_loss / (batch_idx + 1)

    end = time.time()

    # Visualize watch images & train loss
    # if (epoch % 5 == 0 or epoch == 1) and vis is not None:
    if vis is not None:
        opts = dict(jpgquality=100, title='Epoch ' + str(epoch) + ' image mask')
        watch_images[:, :, 0:240, :] = normalize_image(watch_images[:, :, 0:240, :], forward=False)
        watch_images *= 255.0
        watch_images = watch_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        for i in range(watch_images.shape[0]):
            cv2.putText(watch_images[i], image_names[i], (80, 255), fnt, 0.5, (0, 0, 0), 1)
        watch_images = torch.Tensor(watch_images.astype(np.float32))
        watch_images = watch_images.permute(0, 3, 1, 2)
        vis.image(torchvision.utils.make_grid(watch_images, nrow=group_size, padding=3), opts=opts)
        # Un-normalize for caffe resnet and vgg
        # watch_images = watch_images.permute(0, 2, 3, 1) + pixel_means
        # watch_images = watch_images[:, :, :, [2, 1, 0]].permute(0, 3, 1, 2)

        opts_loss = dict(xlabel='Iterations (' + str(stride_loss) + ')',
                         ylabel='Loss',
                         title='CoSegmentation Training Loss in Epoch ' + str(epoch),
                         legend=['Loss'],
                         width=2000)
        vis.line(iter_loss, np.arange(93), opts=opts_loss)

    epoch_loss /= len(dataloader)
    print('Train set -- Average loss: {:.6f}\t\tTime cost: {:.4f}'.format(epoch_loss, end - begin))
    return epoch_loss, end - begin