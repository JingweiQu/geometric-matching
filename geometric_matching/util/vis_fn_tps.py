import torch
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt

from geometric_matching.util.net_util import batch_cuda
from geometric_matching.geotnf.transformation_tps import GeometricTnf
from geometric_matching.image.normalization import normalize_image
from geometric_matching.geotnf.point_tps import PointTPS, PointsToUnitCoords, PointsToPixelCoords

def pck(source_points, warped_points, dataset_name='PF-PASCAL', alpha=0.1):
    batch_size = source_points.size(0)
    pck = torch.zeros((batch_size))
    num_pts = torch.zeros((batch_size))
    correct_index = -torch.ones((batch_size, 20))
    for idx in range(batch_size):
        p_src = source_points[idx, :]
        p_wrp = warped_points[idx, :]
        if dataset_name == 'PF-WILLOW':
            L_pck = torch.Tensor([torch.max(p_src.max(1)[0] - p_src.min(1)[0])]).cuda()
        elif dataset_name == 'PF-PASCAL':
            L_pck = torch.Tensor([224.0]).cuda()
        N_pts = torch.sum(torch.ne(p_src[0, :], -1) * torch.ne(p_src[1, :], -1))
        num_pts[idx] = N_pts
        point_distance = torch.pow(torch.sum(torch.pow(p_src[:, :N_pts] - p_wrp[:, :N_pts], 2), 0), 0.5)
        L_pck_mat = L_pck.expand_as(point_distance)
        correct_points = torch.le(point_distance, L_pck_mat * alpha)
        C_pts = torch.sum(correct_points)
        correct_index[idx, :C_pts] = torch.nonzero(correct_points).view(-1)
        pck[idx] = torch.mean(correct_points.float())

    # batch_size is 1
    if batch_size == 1:
        pck = pck[0].item()
        num_pts = int(num_pts[0].item())
        correct_index = correct_index.squeeze().cpu().numpy().astype(np.int8)
        correct_index = correct_index[np.where(correct_index > -1)]

    return pck, correct_index, num_pts

def relocate(points, image_size):
    batch_size = points.size(0)
    for i in range(batch_size):
        points[i, 0, :] = points[i, 0, :] * 240 / image_size[i, 1]
        points[i, 1, :] = points[i, 1, :] * 240 / image_size[i, 0]
    return points

def swap(source, target):
    tmp = source
    source = target
    target = tmp

    return source, target

def vis_fn(vis, train_loss, val_pck, train_lr, epoch, num_epochs, dataloader, theta, thetai, results,
           geometric_model='tps', use_cuda=True):
    # Visualize watch images
    if geometric_model == 'tps':
        geoTnf = GeometricTnf(geometric_model='tps', use_cuda=use_cuda)
    elif geometric_model == 'affine':
        geoTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)

    pt = PointTPS(use_cuda=use_cuda)

    group_size = 3
    watch_images = torch.ones(len(dataloader) * group_size, 3, 280, 240)
    watch_keypoints = -torch.ones(len(dataloader) * group_size, 2, 20)
    if use_cuda:
        watch_images = watch_images.cuda()
        watch_keypoints = watch_keypoints.cuda()
    num_points = np.ones(len(dataloader) * group_size).astype(np.int8)
    correct_index = list()
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

    theta, thetai = swap(theta, thetai)
    # means for normalize of caffe resnet and vgg
    # pixel_means = torch.Tensor(np.array([[[[102.9801, 115.9465, 122.7717]]]]).astype(np.float32))
    for batch_idx, batch in enumerate(dataloader):
        if use_cuda:
            batch = batch_cuda(batch)

        batch['source_image'], batch['target_image'] = swap(batch['source_image'], batch['target_image'])
        batch['source_im_info'], batch['target_im_info'] = swap(batch['source_im_info'], batch['target_im_info'])
        batch['source_points'], batch['target_points'] = swap(batch['source_points'], batch['target_points'])

        # Theta and thetai
        if geometric_model == 'tps':
            theta_batch = theta['tps'][batch_idx].unsqueeze(0)
            theta_batch_inver = thetai['tps'][batch_idx].unsqueeze(0)
        elif geometric_model == 'affine':
            theta_batch = theta['aff'][batch_idx].unsqueeze(0)
            theta_batch_inver = thetai['aff'][batch_idx].unsqueeze(0)

        # Warped image
        warped_image = geoTnf(batch['source_image'], theta_batch)

        watch_images[batch_idx * group_size, :, 0:240, :] = batch['source_image']
        watch_images[batch_idx * group_size + 1, :, 0:240, :] = warped_image
        watch_images[batch_idx * group_size + 2, :, 0:240, :] = batch['target_image']

        # Warped keypoints
        source_im_size = batch['source_im_info'][:, 0:3]
        target_im_size = batch['target_im_info'][:, 0:3]

        source_points = batch['source_points']
        target_points = batch['target_points']

        source_points_norm = PointsToUnitCoords(P=source_points, im_size=source_im_size)
        target_points_norm = PointsToUnitCoords(P=target_points, im_size=target_im_size)

        if geometric_model == 'tps':
            warped_points_norm = pt.tpsPointTnf(theta=theta_batch_inver, points=source_points_norm)
        elif geometric_model == 'affine':
            warped_points_norm = pt.affPointTnf(theta=theta_batch_inver, points=source_points_norm)

        warped_points = PointsToPixelCoords(P=warped_points_norm, im_size=target_im_size)
        _, index_correct, N_pts = pck(target_points, warped_points)

        watch_keypoints[batch_idx * group_size, :, :N_pts] = relocate(batch['source_points'], source_im_size)[:, :, :N_pts]
        watch_keypoints[batch_idx * group_size + 1, :, :N_pts] = relocate(warped_points, target_im_size)[:, :, :N_pts]
        watch_keypoints[batch_idx * group_size + 2, :, :N_pts] = relocate(batch['target_points'], target_im_size)[:, :, :N_pts]

        num_points[batch_idx * group_size:batch_idx * group_size + group_size] = N_pts

        correct_index.append(np.arange(N_pts))
        correct_index.append(index_correct)
        correct_index.append(np.arange(N_pts))

        image_names.append('Source')
        if geometric_model == 'tps':
            image_names.append('TPS')
        elif geometric_model == 'affine':
            image_names.append('Affine')
        image_names.append('Target')

        metrics.append('')
        if geometric_model == 'tps':
            metrics.append('PCK: {:.2%}'.format(float(results['tps']['pck'][batch_idx])))
        elif geometric_model == 'affine':
            metrics.append('PCK: {:.2%}'.format(float(results['aff']['pck'][batch_idx])))
        metrics.append('')

    opts = dict(jpgquality=100, title='Epoch ' + str(epoch) + ' source warped target')
    # Un-normalize for caffe resnet and vgg
    # watch_images = watch_images.permute(0, 2, 3, 1) + pixel_means
    # watch_images = watch_images[:, :, :, [2, 1, 0]].permute(0, 3, 1, 2)
    # watch_images = normalize_image(watch_images, forward=False) * 255.0
    watch_images[:, :, 0:240, :] = normalize_image(watch_images[:, :, 0:240, :], forward=False)
    watch_images *= 255.0
    watch_images = watch_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    watch_keypoints = watch_keypoints.cpu().numpy()

    for i in range(watch_images.shape[0]):
        pos_name = (80, 255)
        if (i + 1) % group_size == 1 or (i + 1) % group_size == 0:
            pos_pck = (0, 0)
        else:
            pos_pck = (70, 275)
        cv2.putText(watch_images[i], image_names[i], pos_name, fnt, 0.5, (0, 0, 0), 1)
        cv2.putText(watch_images[i], metrics[i], pos_pck, fnt, 0.5, (0, 0, 0), 1)
        if (i + 1) % group_size == 0:
            for j in range(num_points[i]):
                cv2.drawMarker(watch_images[i], (watch_keypoints[i, 0, j], watch_keypoints[i, 1, j]), colors[j],
                               cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
        else:
            for j in correct_index[i]:
                cv2.drawMarker(watch_images[i], (watch_keypoints[i, 0, j], watch_keypoints[i, 1, j]), colors[j],
                               cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)
                cv2.drawMarker(watch_images[i],
                               (watch_keypoints[i + (group_size - 1) - (i % group_size), 0, j], watch_keypoints[i + (group_size - 1) - (i % group_size), 1, j]),
                               colors[j], cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)

    watch_images = torch.Tensor(watch_images.astype(np.float32))
    watch_images = watch_images.permute(0, 3, 1, 2)
    vis.image(torchvision.utils.make_grid(watch_images, nrow=6, padding=3), opts=opts)

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
'''        
def vis_fn(vis, model, train_loss, val_pck, train_lr, epoch, num_epochs, dataloader, geometric_model='tps', use_cuda=True):
    # Visualize watch images
    geoTnf = GeometricTnf(geometric_model=geometric_model, use_cuda=use_cuda)
    watch_images = torch.Tensor(27, 3, 240, 240)
    # means for normalize of caffe resnet and vgg
    # pixel_means = torch.Tensor(np.array([[[[102.9801, 115.9465, 122.7717]]]]).astype(np.float32)).cuda()
    for batch_idx, batch in enumerate(dataloader):
        if use_cuda:
            batch = batch_cuda(batch)

        theta = model(batch)

        warped_image = geoTnf(batch['source_image'], theta).squeeze(0)
        watch_images[batch_idx * 3] = batch['source_image']
        watch_images[batch_idx * 3 + 1] = warped_image
        watch_images[batch_idx * 3 + 2] = batch['target_image']

    opts = dict(jpgquality=100, title='Epoch ' + str(epoch) + ' source warped target')
    # Un-normalize for caffe resnet and vgg
    # watch_images = watch_images.permute(0, 2, 3, 1) + pixel_means
    # watch_images = watch_images[:, :, :, [2, 1, 0]].permute(0, 3, 1, 2)
    watch_images = normalize_image(watch_images, forward=False) * 255.0
    vis.image(torchvision.utils.make_grid(watch_images, nrow=6, padding=3), opts=opts)
    # vis.images(watch_images, nrow=9, padding=3, opts=opts)

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
'''