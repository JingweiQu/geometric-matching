import torch
import numpy as np
from geometric_matching.util.net_util import batch_cuda
from geometric_matching.geotnf.transformation import GeometricTnf
from geometric_matching.image.normalization import normalize_image

def vis_fn(vis, model, train_loss, val_pck, train_lr, epoch, num_epochs, dataloader, geometric_model='tps', use_cuda=True, normalize=None):
    # Visualize watch images
    geoTnf = GeometricTnf(geometric_model=geometric_model, use_cuda=use_cuda)
    watch_images = torch.Tensor(27, 3, 240, 240)
    if normalize is None:
        pixel_means = torch.Tensor(np.array([[[[102.9801, 115.9465, 122.7717]]]]).astype(np.float32))
    for batch_idx, batch in enumerate(dataloader):
        if use_cuda:
            batch = batch_cuda(batch)
        theta = model(batch)
        warped_image = geoTnf(batch['source_image'], theta).squeeze(0)
        watch_images[batch_idx * 3] = batch['source_image']
        watch_images[batch_idx * 3 + 1] = warped_image
        watch_images[batch_idx * 3 + 2] = batch['target_image']
    opts = dict(jpgquality=100, title='Epoch ' + str(epoch) + ' source warped target')
    if normalize is None:
        watch_images = watch_images.permute(0, 2, 3, 1) + pixel_means
        watch_images = watch_images[:, :, :, [2, 1, 0]].permute(0, 3, 1, 2)
    else:
        watch_images = normalize_image(watch_images, forward=False) * 255.0
    vis.images(watch_images, nrow=9, padding=3, opts=opts)

    if epoch == num_epochs:
        epochs = np.arange(1, num_epochs+1)
        # Visualize train loss
        opts_loss = dict(xlabel='Epoch',
                    ylabel='Loss',
                    title='GM ResNet101 ' + geometric_model + ' Training Loss',
                    legend=['Loss'],
                    width=2000)
        vis.line(train_loss, epochs, opts=opts_loss)

        # Visualize val pck
        opts_pck = dict(xlabel='Epoch',
                    ylabel='Val PCK',
                    title='GM ResNet101 ' + geometric_model + ' Val PCK',
                    legend=['PCK'],
                    width=2000)
        vis.line(val_pck, epochs, opts=opts_pck)

        # Visualize train lr
        opts_lr = dict(xlabel='Epoch',
                       ylabel='Learning Rate',
                       title='GM ResNet101 ' + geometric_model + ' Training Learning Rate',
                       legend=['LR'],
                       width=2000)
        vis.line(train_lr, epochs, opts=opts_lr)