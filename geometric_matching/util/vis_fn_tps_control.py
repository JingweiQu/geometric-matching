import torch
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt

from geometric_matching.util.net_util import batch_cuda
from geometric_matching.geotnf.transformation_tps import GeometricTnf
from geometric_matching.image.normalization import normalize_image
from geometric_matching.geotnf.point_tnf import PointsToPixelCoords

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

def vis_fn(vis, train_loss, val_pck, train_lr, epoch, num_epochs, dataloader, theta, thetai, results,
           geometric_model='tps', use_cuda=True):
    geoTnf = GeometricTnf(geometric_model='tps', use_cuda=use_cuda)

    group_size = 3
    watch_images = torch.ones(len(dataloader) * group_size, 3, 340, 340)
    if use_cuda:
        watch_images = watch_images.cuda()

    # means for normalize of caffe resnet and vgg
    # pixel_means = torch.Tensor(np.array([[[[102.9801, 115.9465, 122.7717]]]]).astype(np.float32))
    for batch_idx, batch in enumerate(dataloader):
        if use_cuda:
            batch = batch_cuda(batch)

        # Theta and thetai
        theta_batch = theta['tps'][batch_idx].unsqueeze(0)

        # Warped image
        warped_image = geoTnf(batch['source_image'], theta_batch)

        watch_images[batch_idx * group_size, :, 50:290, 50:290] = batch['source_image']
        watch_images[batch_idx * group_size + 1, :, 50:290, 50:290] = warped_image
        watch_images[batch_idx * group_size + 2, :, 50:290, 50:290] = batch['target_image']


    opts = dict(jpgquality=100, title='Epoch ' + str(epoch) + ' source warped target')
    # Un-normalize for caffe resnet and vgg
    # watch_images = watch_images.permute(0, 2, 3, 1) + pixel_means
    # watch_images = watch_images[:, :, :, [2, 1, 0]].permute(0, 3, 1, 2)
    # watch_images = normalize_image(watch_images, forward=False) * 255.0
    watch_images[:, :, 50:290, 50:290] = normalize_image(watch_images[:, :, 50:290, 50:290], forward=False)
    watch_images *= 255.0
    watch_images = watch_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    for i in range(watch_images.shape[0]):
        if i % group_size == 0:
            cp_norm = theta['tps'][int(i / group_size)][:18].view(1, 2, -1)
            watch_images[i] = draw_grid(watch_images[i], cp_norm)

            cp_norm = theta['tps'][int(i / group_size)][18:].view(1, 2, -1)
            watch_images[i + 1] = draw_grid(watch_images[i + 1], cp_norm)

    watch_images = torch.Tensor(watch_images.astype(np.float32))
    watch_images = watch_images.permute(0, 3, 1, 2)
    vis.image(torchvision.utils.make_grid(watch_images, nrow=3, padding=3), opts=opts)

    if epoch == num_epochs:
        if geometric_model == 'affine':
            sub_str = 'Affine'
        elif geometric_model == 'tps':
            sub_str = 'TPS'
        epochs = np.arange(1, num_epochs+1)
        # Visualize train loss
        opts_loss = dict(xlabel='Epoch',
                    ylabel='Loss',
                    title='GM ResNet101 ' + sub_str + ' Training Loss',
                    legend=['Loss'],
                    width=2000)
        vis.line(train_loss, epochs, opts=opts_loss)

        # Visualize val pck
        opts_pck = dict(xlabel='Epoch',
                    ylabel='Val PCK',
                    title='GM ResNet101 ' + sub_str + ' Val PCK',
                    legend=['PCK'],
                    width=2000)
        vis.line(val_pck, epochs, opts=opts_pck)

        # Visualize train lr
        opts_lr = dict(xlabel='Epoch',
                       ylabel='Learning Rate',
                       title='GM ResNet101 ' + sub_str + ' Training Learning Rate',
                       legend=['LR'],
                       width=2000)
        vis.line(train_lr, epochs, opts=opts_lr)