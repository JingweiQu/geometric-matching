import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from geometric_matching.util.net_util import batch_cuda
from geometric_matching.geotnf.transformation import GeometricTnf
from geometric_matching.image.normalization import normalize_image

def vis_feature(vis, model, dataloader, use_cuda=True):
    # Visualize feature map of watch image
    h = 40
    num = 1024
    id = 4
    for batch_idx, batch in enumerate(dataloader):
        if use_cuda:
            batch = batch_cuda(batch)
        theta, feature_A, feature_B, correlation = model(batch)
        if batch_idx == id:
            break
    watch_feature_A = F.interpolate(feature_A, size=(h, h), mode='bilinear', align_corners=True).transpose(0, 1)[0:num, :, :, :]
    watch_feature_B = F.interpolate(feature_B, size=(h, h), mode='bilinear', align_corners=True).transpose(0, 1)[0:num, :, :, :]

    opts = dict(jpgquality=100, title='source image')
    image_A = normalize_image(batch['source_image'][0], forward=False) * 255.0
    vis.image(image_A, opts=opts)

    nrow = 32
    padding = 3
    opts = dict(jpgquality=100, title='feature map A')
    # vis.images(watch_feature_A * 255.0, nrow=nrow, padding=padding, opts=opts)
    vis.image(torchvision.utils.make_grid(watch_feature_A * 255.0, nrow=nrow, padding=padding), opts=opts)
    # vis.image(watch_feature_A[0], opts=opts)

    opts = dict(jpgquality=100, title='target image')
    image_B = normalize_image(batch['target_image'][0], forward=False) * 255.0
    vis.image(image_B, opts=opts)

    opts = dict(jpgquality=100, title='feature map B')
    vis.image(torchvision.utils.make_grid(watch_feature_B * 255.0, nrow=nrow, padding=padding), opts=opts)
    # vis.images(watch_feature_B * 255.0, nrow=nrow, padding=padding, opts=opts)
    # vis.image(watch_feature_B[0], opts=opts)

    # opts = dict(title='correlation')
    # vis.heatmap(correlation[0, 0, :, :], opts=opts)