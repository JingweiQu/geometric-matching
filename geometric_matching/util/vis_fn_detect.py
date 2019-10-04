import torch
import torchvision
import numpy as np
from geometric_matching.util.net_util import batch_cuda
from geometric_matching.geotnf.transformation import GeometricTnf
from geometric_matching.image.normalization import normalize_image
from geometric_matching.util.net_util import *

def vis_fn_detect(vis, model, faster_rcnn, aff_theta, train_loss, val_pck, train_lr, epoch, num_epochs, dataloader, use_cuda=True):
    # Visualize watch images
    affTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)
    tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=use_cuda)
    watch_images = torch.Tensor(len(dataloader) * 5, 3, 240, 240)
    # means for normalize of caffe resnet and vgg
    # pixel_means = torch.Tensor(np.array([[[[102.9801, 115.9465, 122.7717]]]]).astype(np.float32))
    for batch_idx, batch in enumerate(dataloader):
        if use_cuda:
            batch = batch_cuda(batch)

        box_info_s = faster_rcnn(im_data=batch['source_im'], im_info=batch['source_im_info'][:, 3:],
                                 gt_boxes=batch['source_gt_boxes'], num_boxes=batch['source_num_boxes'])[0:3]
        box_info_t = faster_rcnn(im_data=batch['target_im'], im_info=batch['target_im_info'][:, 3:],
                                 gt_boxes=batch['target_gt_boxes'], num_boxes=batch['target_num_boxes'])[0:3]
        all_box_s = select_boxes(rois=box_info_s[0], cls_prob=box_info_s[1], bbox_pred=box_info_s[2], im_infos=batch['source_im_info'][:, 3:])
        all_box_t = select_boxes(rois=box_info_t[0], cls_prob=box_info_t[1], bbox_pred=box_info_t[2], im_infos=batch['target_im_info'][:, 3:])
        box_s, box_t = select_box_st(all_box_s, all_box_t)
        theta_det = aff_theta(boxes_s=box_s, boxes_t=box_t)
        theta_aff_tps, theta_aff = model(batch, theta_det)

        warped_image_1 = affTnf(batch['source_image'], theta_det)
        warped_image_2 = affTnf(warped_image_1, theta_aff)
        warped_image_3 = tpsTnf(warped_image_2, theta_aff_tps)
        watch_images[batch_idx * 5] = batch['source_image'][0]
        watch_images[batch_idx * 5 + 1] = warped_image_1[0]
        watch_images[batch_idx * 5 + 2] = warped_image_2[0]
        watch_images[batch_idx * 5 + 3] = warped_image_3[0]
        watch_images[batch_idx * 5 + 4] = batch['target_image'][0]

    opts = dict(jpgquality=100, title='Epoch ' + str(epoch) + ' source warped target')
    # Un-normalize for caffe resnet and vgg
    # watch_images = watch_images.permute(0, 2, 3, 1) + pixel_means
    # watch_images = watch_images[:, :, :, [2, 1, 0]].permute(0, 3, 1, 2)
    watch_images = normalize_image(watch_images, forward=False) * 255.0
    vis.image(torchvision.utils.make_grid(watch_images, nrow=5, padding=5), opts=opts)
    # vis.images(watch_images, nrow=5, padding=3, opts=opts)

    if epoch == num_epochs:
        epochs = np.arange(1, num_epochs+1)
        # Visualize train loss
        opts_loss = dict(xlabel='Epoch',
                    ylabel='Loss',
                    title='GM ResNet101 Detect&Affine&TPS Training Loss',
                    legend=['Loss'],
                    width=2000)
        vis.line(train_loss, epochs, opts=opts_loss)

        # Visualize val pck
        opts_pck = dict(xlabel='Epoch',
                    ylabel='Val PCK',
                    title='GM ResNet101 Detect&Affine&TPS Val PCK',
                    legend=['PCK'],
                    width=2000)
        vis.line(val_pck, epochs, opts=opts_pck)

        # Visualize train lr
        opts_lr = dict(xlabel='Epoch',
                       ylabel='Learning Rate',
                       title='GM ResNet101 Detect&Affine&TPS Training Learning Rate',
                       legend=['LR'],
                       width=2000)
        vis.line(train_lr, epochs, opts=opts_lr)