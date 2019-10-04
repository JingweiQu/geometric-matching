import torch
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt

from geometric_matching.util.net_util import batch_cuda
from geometric_matching.image.normalization import normalize_image

def vis_fn(vis, train_loss, val_iou, train_lr, epoch, num_epochs, dataloader, results, masks_A, masks_B, use_cuda=True):
    # Visualize watch images
    group_size = 6
    watch_images = torch.ones(len(dataloader) * group_size, 3, 280, 240)
    if use_cuda:
        watch_images = watch_images.cuda()
    image_names = list()
    metrics = list()

    # Colors for keypoints
    cmap = plt.get_cmap('tab20')
    colors = list()
    for c in range(20):
        r = cmap(c)[0] * 255
        g = cmap(c)[1] * 255
        b = cmap(c)[2] * 255
        colors.append((b, g, r))
    fnt = cv2.FONT_HERSHEY_COMPLEX

    for batch_idx, batch in enumerate(dataloader):
        if use_cuda:
            batch = batch_cuda(batch)

        # Theta and theta_inver
        watch_images[batch_idx * group_size, :, 0:240, :] = batch['source_image']
        watch_images[batch_idx * group_size + 1, :, 0:240, :] = torch.mul(batch['source_image'], batch['source_mask'])
        # watch_images[batch_idx * group_size + 2, :, 0:240, :] = torch.mul(batch['source_image'], masks_A[batch_idx])
        mask_A = masks_A[batch_idx].gt(0.5).float()
        watch_images[batch_idx * group_size + 2, :, 0:240, :] = torch.mul(batch['source_image'], mask_A)
        watch_images[batch_idx * group_size + 3, :, 0:240, :] = batch['target_image']
        watch_images[batch_idx * group_size + 4, :, 0:240, :] = torch.mul(batch['target_image'], batch['target_mask'])
        # watch_images[batch_idx * group_size + 5, :, 0:240, :] = torch.mul(batch['target_image'], masks_B[batch_idx])
        mask_B = masks_B[batch_idx].gt(0.5).float()
        watch_images[batch_idx * group_size + 5, :, 0:240, :] = torch.mul(batch['target_image'], mask_B)


        image_names.append('Source')
        image_names.append('Mask_GT')
        image_names.append('Mask')
        image_names.append('Target')
        image_names.append('Mask_GT')
        image_names.append('Mask')

        metrics.append('')
        metrics.append('')
        metrics.append('IoU: {:.2%}'.format(float(results[batch_idx, 0])))
        metrics.append('')
        metrics.append('')
        metrics.append('IoU: {:.2%}'.format(float(results[batch_idx, 1])))

    opts = dict(jpgquality=100, title='Epoch ' + str(epoch) + ' image mask_gt mask')
    # Un-normalize for caffe resnet and vgg
    # watch_images = watch_images.permute(0, 2, 3, 1) + pixel_means
    # watch_images = watch_images[:, :, :, [2, 1, 0]].permute(0, 3, 1, 2)
    # watch_images = normalize_image(watch_images, forward=False) * 255.0
    watch_images[:, :, 0:240, :] = normalize_image(watch_images[:, :, 0:240, :], forward=False)
    watch_images *= 255.0
    watch_images = watch_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    for i in range(watch_images.shape[0]):
        pos_name = (80, 255)
        if (i + 1) % group_size == 3 or (i + 1) % group_size == 0:
            pos_iou = (70, 275)
        else:
            pos_iou = (0, 0)
        cv2.putText(watch_images[i], image_names[i], pos_name, fnt, 0.5, (0, 0, 0), 1)
        cv2.putText(watch_images[i], metrics[i], pos_iou, fnt, 0.5, (0, 0, 0), 1)

    watch_images = torch.Tensor(watch_images.astype(np.float32))
    watch_images = watch_images.permute(0, 3, 1, 2)
    vis.image(torchvision.utils.make_grid(watch_images, nrow=6, padding=3), opts=opts)

    if epoch == num_epochs:
        epochs = np.arange(1, num_epochs+1)
        # Visualize train loss
        opts_loss = dict(xlabel='Epoch',
                    ylabel='Loss',
                    title='CoSegmentation Training Loss',
                    legend=['Loss'],
                    width=2000)
        vis.line(train_loss, epochs, opts=opts_loss)

        # Visualize val pck
        opts_pck = dict(xlabel='Epoch',
                    ylabel='Val PCK',
                    title='CoSegmentation Val IoU',
                    legend=['PCK'],
                    width=2000)
        vis.line(val_iou, epochs, opts=opts_pck)

        # Visualize train lr
        opts_lr = dict(xlabel='Epoch',
                       ylabel='Learning Rate',
                       title='CoSegmentation Training Learning Rate',
                       legend=['LR'],
                       width=2000)
        vis.line(train_lr, epochs, opts=opts_lr)