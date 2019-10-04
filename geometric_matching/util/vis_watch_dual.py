import torch
import torchvision
from torchvision import transforms
import numpy as np
import PIL
from PIL import ImageDraw
from PIL import ImageFont
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt

from geometric_matching.util.net_util import batch_cuda
from geometric_matching.geotnf.transformation import GeometricTnf
from geometric_matching.image.normalization import normalize_image
from geometric_matching.util.net_util import *
from geometric_matching.geotnf.point_tnf import PointTnf, PointsToUnitCoords, PointsToPixelCoords

def pck(source_points, warped_points, dataset_name, alpha=0.1):
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

def vis_pf(vis, dataloader, theta, theta_weak, theta_inver, theta_weak_inver, results, results_weak, dataset_name, use_cuda=True):
    # Visualize watch images
    affTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)
    tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=use_cuda)
    pt = PointTnf(use_cuda=use_cuda)

    watch_images = torch.ones(len(dataloader) * 6, 3, 280, 240)
    watch_keypoints = -torch.ones(len(dataloader) * 6, 2, 20)
    if use_cuda:
        watch_images = watch_images.cuda()
        watch_keypoints = watch_keypoints.cuda()
    num_points = np.ones(len(dataloader) * 6).astype(np.int8)
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

    # means for normalize of caffe resnet and vgg
    # pixel_means = torch.Tensor(np.array([[[[102.9801, 115.9465, 122.7717]]]]).astype(np.float32))
    for batch_idx, batch in enumerate(dataloader):
        if use_cuda:
            batch = batch_cuda(batch)

        # Theta and theta_inver
        theta_aff = theta['aff'][batch_idx].unsqueeze(0)
        theta_aff_tps = theta['aff_tps'][batch_idx].unsqueeze(0)
        theta_weak_aff = theta_weak['aff'][batch_idx].unsqueeze(0)
        theta_weak_aff_tps = theta_weak['aff_tps'][batch_idx].unsqueeze(0)

        theta_aff_inver = theta_inver['aff'][batch_idx].unsqueeze(0)
        theta_aff_tps_inver = theta_inver['aff_tps'][batch_idx].unsqueeze(0)
        theta_weak_aff_inver = theta_weak_inver['aff'][batch_idx].unsqueeze(0)
        theta_weak_aff_tps_inver = theta_weak_inver['aff_tps'][batch_idx].unsqueeze(0)

        # Warped image
        warped_aff = affTnf(batch['source_image'], theta_aff)
        warped_aff_tps = tpsTnf(warped_aff, theta_aff_tps)
        warped_weak_aff = affTnf(batch['source_image'], theta_weak_aff)
        warped_weak_aff_tps = tpsTnf(warped_weak_aff, theta_weak_aff_tps)

        watch_images[batch_idx * 6, :, 0:240, :] = batch['source_image']
        watch_images[batch_idx * 6 + 1, :, 0:240, :] = warped_aff
        watch_images[batch_idx * 6 + 2, :, 0:240, :] = warped_aff_tps
        watch_images[batch_idx * 6 + 3, :, 0:240, :] = batch['target_image']
        watch_images[batch_idx * 6 + 4, :, 0:240, :] = warped_weak_aff
        watch_images[batch_idx * 6 + 5, :, 0:240, :] = warped_weak_aff_tps

        # Warped keypoints
        source_im_size = batch['source_im_info'][:, 0:3]
        target_im_size = batch['target_im_info'][:, 0:3]

        source_points = batch['source_points']
        target_points = batch['target_points']

        source_points_norm = PointsToUnitCoords(P=source_points, im_size=source_im_size)
        target_points_norm = PointsToUnitCoords(P=target_points, im_size=target_im_size)

        warped_points_aff_norm = pt.affPointTnf(theta=theta_aff_inver, points=source_points_norm)
        warped_points_aff = PointsToPixelCoords(P=warped_points_aff_norm, im_size=target_im_size)
        pck_aff, index_aff, N_pts = pck(target_points, warped_points_aff, dataset_name)
        warped_points_aff = relocate(warped_points_aff, target_im_size)

        warped_points_aff_tps_norm = pt.tpsPointTnf(theta=theta_aff_tps_inver, points=source_points_norm)
        warped_points_aff_tps_norm = pt.affPointTnf(theta=theta_aff_inver, points=warped_points_aff_tps_norm)
        warped_points_aff_tps = PointsToPixelCoords(P=warped_points_aff_tps_norm, im_size=target_im_size)
        pck_aff_tps, index_aff_tps, _ = pck(target_points, warped_points_aff_tps, dataset_name)
        warped_points_aff_tps = relocate(warped_points_aff_tps, target_im_size)

        warped_points_weak_aff_norm = pt.affPointTnf(theta=theta_weak_aff_inver, points=source_points_norm)
        warped_points_weak_aff = PointsToPixelCoords(P=warped_points_weak_aff_norm, im_size=target_im_size)
        pck_weak_aff, index_weak_aff, _ = pck(target_points, warped_points_weak_aff, dataset_name)
        warped_points_weak_aff = relocate(warped_points_weak_aff, target_im_size)

        warped_points_weak_aff_tps_norm = pt.tpsPointTnf(theta=theta_weak_aff_tps_inver, points=source_points_norm)
        warped_points_weak_aff_tps_norm = pt.affPointTnf(theta=theta_weak_aff_inver, points=warped_points_weak_aff_tps_norm)
        warped_points_weak_aff_tps = PointsToPixelCoords(P=warped_points_weak_aff_tps_norm, im_size=target_im_size)
        pck_weak_aff_tps, index_weak_aff_tps, _ = pck(target_points, warped_points_weak_aff_tps, dataset_name)
        warped_points_weak_aff_tps = relocate(warped_points_weak_aff_tps, target_im_size)

        watch_keypoints[batch_idx * 6, :, :N_pts] = relocate(batch['source_points'], source_im_size)[:, :, :N_pts]
        watch_keypoints[batch_idx * 6 + 1, :, :N_pts] = warped_points_aff[:, :, :N_pts]
        watch_keypoints[batch_idx * 6 + 2, :, :N_pts] = warped_points_aff_tps[:, :, :N_pts]
        watch_keypoints[batch_idx * 6 + 3, :, :N_pts] = relocate(batch['target_points'], target_im_size)[:, :, :N_pts]
        watch_keypoints[batch_idx * 6 + 4, :, :N_pts] = warped_points_weak_aff[:, :, :N_pts]
        watch_keypoints[batch_idx * 6 + 5, :, :N_pts] = warped_points_weak_aff_tps[:, :, :N_pts]

        num_points[batch_idx * 6:batch_idx * 6 + 6] = N_pts

        correct_index.append(np.arange(N_pts))
        correct_index.append(index_aff)
        correct_index.append(index_aff_tps)
        correct_index.append(np.arange(N_pts))
        correct_index.append(index_weak_aff)
        correct_index.append(index_weak_aff_tps)

        image_names.append('Source')
        image_names.append('Aff')
        image_names.append('Aff_tps')
        image_names.append('Target')
        image_names.append('Rocco_aff')
        image_names.append('Rocco_aff_tps')

        metrics.append('')
        metrics.append('PCK: {:.2%}'.format(pck_aff))
        metrics.append('PCK: {:.2%}'.format(pck_aff_tps))
        metrics.append('')
        metrics.append('PCK: {:.2%}'.format(pck_weak_aff))
        metrics.append('PCK: {:.2%}'.format(pck_weak_aff_tps))

    opts = dict(jpgquality=100, title=dataset_name)
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
        if (i + 1) % 6 == 1 or (i + 1) % 6 == 4:
            pos_pck = (0, 0)
        else:
            pos_pck = (70, 275)
        cv2.putText(watch_images[i], image_names[i], pos_name, fnt, 0.5, (0, 0, 0), 1)
        cv2.putText(watch_images[i], metrics[i], pos_pck, fnt, 0.5, (0, 0, 0), 1)
        if (i + 1) % 6 == 4:
            for j in range(num_points[i]):
                cv2.drawMarker(watch_images[i], (watch_keypoints[i, 0, j], watch_keypoints[i, 1, j]), colors[j], cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)
        else:
            for j in correct_index[i]:
                cv2.drawMarker(watch_images[i], (watch_keypoints[i, 0, j], watch_keypoints[i, 1, j]), colors[j], cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
                cv2.drawMarker(watch_images[i], (watch_keypoints[i + 3 - (i % 6), 0, j], watch_keypoints[i + 3 - (i % 6), 1, j]), colors[j], cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)

    watch_images = torch.Tensor(watch_images.astype(np.float32))
    watch_images = watch_images.permute(0, 3, 1, 2)
    vis.image(torchvision.utils.make_grid(watch_images, nrow=3, padding=3), opts=opts)

def vis_caltech(vis, dataloader, theta, theta_weak, results, results_weak, title, use_cuda=True):
    # Visualize watch images
    affTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)
    tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=use_cuda)
    watch_images = torch.ones(len(dataloader) * 6, 3, 280, 240)
    if use_cuda:
        watch_images = watch_images.cuda()
    image_names = list()
    lt_acc = list()
    iou = list()
    fnt = cv2.FONT_HERSHEY_COMPLEX
    # means for normalize of caffe resnet and vgg
    # pixel_means = torch.Tensor(np.array([[[[102.9801, 115.9465, 122.7717]]]]).astype(np.float32))
    for batch_idx, batch in enumerate(dataloader):
        if use_cuda:
            batch = batch_cuda(batch)

        theta_aff = theta['aff'][batch_idx].unsqueeze(0)
        theta_aff_tps = theta['aff_tps'][batch_idx].unsqueeze(0)
        theta_weak_aff = theta_weak['aff'][batch_idx].unsqueeze(0)
        theta_weak_aff_tps = theta_weak['aff_tps'][batch_idx].unsqueeze(0)

        # Warped image
        warped_aff = affTnf(batch['source_image'], theta_aff)
        warped_aff_tps = tpsTnf(warped_aff, theta_aff_tps)
        warped_weak_aff = affTnf(batch['source_image'], theta_weak_aff)
        warped_weak_aff_tps = tpsTnf(warped_weak_aff, theta_weak_aff_tps)

        watch_images[batch_idx * 6, :, 0:240, :] = batch['source_image']
        watch_images[batch_idx * 6 + 1, :, 0:240, :] = warped_aff
        watch_images[batch_idx * 6 + 2, :, 0:240, :] = warped_aff_tps
        watch_images[batch_idx * 6 + 3, :, 0:240, :] = batch['target_image']
        watch_images[batch_idx * 6 + 4, :, 0:240, :] = warped_weak_aff
        watch_images[batch_idx * 6 + 5, :, 0:240, :] = warped_weak_aff_tps

        image_names.append('Source')
        image_names.append('Aff')
        image_names.append('Aff_tps')
        image_names.append('Target')
        image_names.append('Rocco_aff')
        image_names.append('Rocco_aff_tps')

        lt_acc.append('')
        lt_acc.append('LT-ACC: {:.2f}'.format(float(results['aff']['label_transfer_accuracy'][batch_idx])))
        lt_acc.append('LT-ACC: {:.2f}'.format(float(results['aff_tps']['label_transfer_accuracy'][batch_idx])))
        lt_acc.append('')
        lt_acc.append('LT-ACC: {:.2f}'.format(float(results_weak['aff']['label_transfer_accuracy'][batch_idx])))
        lt_acc.append('LT-ACC: {:.2f}'.format(float(results_weak['aff_tps']['label_transfer_accuracy'][batch_idx])))

        iou.append('')
        iou.append('IoU: {:.2f}'.format(float(results['aff']['intersection_over_union'][batch_idx])))
        iou.append('IoU: {:.2f}'.format(float(results['aff_tps']['intersection_over_union'][batch_idx])))
        iou.append('')
        iou.append('IoU: {:.2f}'.format(float(results_weak['aff']['intersection_over_union'][batch_idx])))
        iou.append('IoU: {:.2f}'.format(float(results_weak['aff_tps']['intersection_over_union'][batch_idx])))

    opts = dict(jpgquality=100, title=title)
    # Un-normalize for caffe resnet and vgg
    # watch_images = watch_images.permute(0, 2, 3, 1) + pixel_means
    # watch_images = watch_images[:, :, :, [2, 1, 0]].permute(0, 3, 1, 2)
    # watch_images = normalize_image(watch_images, forward=False) * 255.0
    watch_images[:, :, 0:240, :] = normalize_image(watch_images[:, :, 0:240, :], forward=False)
    watch_images *= 255.0
    watch_images = watch_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    for i in range(watch_images.shape[0]):
        pos_name = (80, 255)
        if (i + 1) % 6 == 1 or (i + 1) % 6 == 4:
            pos_lt_ac = (0, 0)
            pos_iou = (0, 0)
        else:
            pos_lt_ac = (10, 275)
            pos_iou = (140, 275)
        cv2.putText(watch_images[i], image_names[i], pos_name, fnt, 0.5, (0, 0, 0), 1)
        cv2.putText(watch_images[i], lt_acc[i], pos_lt_ac, fnt, 0.5, (0, 0, 0), 1)
        cv2.putText(watch_images[i], iou[i], pos_iou, fnt, 0.5, (0, 0, 0), 1)

    watch_images = torch.Tensor(watch_images.astype(np.float32))
    watch_images = watch_images.permute(0, 3, 1, 2)
    vis.image(torchvision.utils.make_grid(watch_images, nrow=3, padding=5), opts=opts)

def vis_tss(vis, dataloader, theta, theta_weak, csv_file, title, use_cuda=True):
    # Visualize watch images
    dataframe = pd.read_csv(csv_file)
    scores_aff = dataframe.iloc[:, 5]
    scores_aff_tps = dataframe.iloc[:, 6]
    scores_weak_aff = dataframe.iloc[:, 7]
    scores_weak_aff_tps = dataframe.iloc[:, 8]
    affTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)
    tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=use_cuda)
    watch_images = torch.ones(int(len(dataloader) / 2 * 6), 3, 280, 240)
    watch_images_inver = torch.ones(int(len(dataloader) / 2 * 6), 3, 280, 240)
    if use_cuda:
        watch_images = watch_images.cuda()
        watch_images_inver = watch_images_inver.cuda()
    image_names = list()
    image_names_inver = list()
    flow = list()
    flow_inver = list()
    fnt = cv2.FONT_HERSHEY_COMPLEX
    # means for normalize of caffe resnet and vgg
    # pixel_means = torch.Tensor(np.array([[[[102.9801, 115.9465, 122.7717]]]]).astype(np.float32))
    for batch_idx, batch in enumerate(dataloader):
        if use_cuda:
            batch = batch_cuda(batch)

        theta_aff = theta['aff'][batch_idx].unsqueeze(0)
        theta_aff_tps = theta['aff_tps'][batch_idx].unsqueeze(0)
        theta_weak_aff = theta_weak['aff'][batch_idx].unsqueeze(0)
        theta_weak_aff_tps = theta_weak['aff_tps'][batch_idx].unsqueeze(0)

        # Warped image
        warped_aff = affTnf(batch['source_image'], theta_aff)
        warped_aff_tps = tpsTnf(warped_aff, theta_aff_tps)
        warped_weak_aff = affTnf(batch['source_image'], theta_weak_aff)
        warped_weak_aff_tps = tpsTnf(warped_weak_aff, theta_weak_aff_tps)

        if (batch_idx + 1) % 2 != 0:
            watch_images[int(batch_idx / 2 * 6), :, 0:240, :] = batch['source_image']
            watch_images[int(batch_idx / 2 * 6) + 1, :, 0:240, :] = warped_aff
            watch_images[int(batch_idx / 2 * 6) + 2, :, 0:240, :] = warped_aff_tps
            watch_images[int(batch_idx / 2 * 6) + 3, :, 0:240, :] = batch['target_image']
            watch_images[int(batch_idx / 2 * 6) + 4, :, 0:240, :] = warped_weak_aff
            watch_images[int(batch_idx / 2 * 6) + 5, :, 0:240, :] = warped_weak_aff_tps

            image_names.append('Source')
            image_names.append('Aff')
            image_names.append('Aff_tps')
            image_names.append('Target')
            image_names.append('Rocco_aff')
            image_names.append('Rocco_aff_tps')

            flow.append('')
            flow.append('Flow: {:.3f}'.format(scores_aff[batch_idx]))
            flow.append('Flow: {:.3f}'.format(scores_aff_tps[batch_idx]))
            flow.append('')
            flow.append('Flow: {:.3f}'.format(scores_weak_aff[batch_idx]))
            flow.append('Flow: {:.3f}'.format(scores_weak_aff_tps[batch_idx]))
        else:
            watch_images_inver[int((batch_idx - 1) / 2 * 6), :, 0:240, :] = batch['source_image']
            watch_images_inver[int((batch_idx - 1) / 2 * 6) + 1, :, 0:240, :] = warped_aff
            watch_images_inver[int((batch_idx - 1) / 2 * 6) + 2, :, 0:240, :] = warped_aff_tps
            watch_images_inver[int((batch_idx - 1) / 2 * 6) + 3, :, 0:240, :] = batch['target_image']
            watch_images_inver[int((batch_idx - 1) / 2 * 6) + 4, :, 0:240, :] = warped_weak_aff
            watch_images_inver[int((batch_idx - 1) / 2 * 6) + 5, :, 0:240, :] = warped_weak_aff_tps

            image_names_inver.append('Source')
            image_names_inver.append('Aff')
            image_names_inver.append('Aff_tps')
            image_names_inver.append('Target')
            image_names_inver.append('Rocco_aff')
            image_names_inver.append('Rocco_aff_tps')

            flow_inver.append('')
            flow_inver.append('Flow: {:.3f}'.format(scores_aff[batch_idx]))
            flow_inver.append('Flow: {:.3f}'.format(scores_aff_tps[batch_idx]))
            flow_inver.append('')
            flow_inver.append('Flow: {:.3f}'.format(scores_weak_aff[batch_idx]))
            flow_inver.append('Flow: {:.3f}'.format(scores_weak_aff_tps[batch_idx]))

    opts = dict(jpgquality=100, title=title)
    # Un-normalize for caffe resnet and vgg
    # watch_images = watch_images.permute(0, 2, 3, 1) + pixel_means
    # watch_images = watch_images[:, :, :, [2, 1, 0]].permute(0, 3, 1, 2)
    # watch_images = normalize_image(watch_images, forward=False) * 255.0
    def draw_image(images, names, flows):
        images[:, :, 0:240, :] = normalize_image(images[:, :, 0:240, :], forward=False)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        for i in range(images.shape[0]):
            pos_name = (80, 255)
            if (i + 1) % 6 == 1 or (i + 1) % 6 == 4:
                pos_lt_ac = (0, 0)
                pos_flow = (0, 0)
            else:
                pos_flow = (70, 275)
            cv2.putText(images[i], names[i], pos_name, fnt, 0.5, (0, 0, 0), 1)
            cv2.putText(images[i], flows[i], pos_flow, fnt, 0.5, (0, 0, 0), 1)

        images = torch.Tensor(images.astype(np.float32))
        images = images.permute(0, 3, 1, 2)

        return images

    watch_images = draw_image(images=watch_images, names=image_names, flows=flow)
    watch_images_inver = draw_image(images=watch_images_inver, names=image_names_inver, flows=flow_inver)
    vis.image(torchvision.utils.make_grid(watch_images, nrow=3, padding=5), opts=opts)
    # vis.image(torchvision.utils.make_grid(watch_images_inver, nrow=3, padding=5), opts=opts)

def swap(source, target):
    tmp = source
    source = target
    target = tmp

    return source, target

def vis_pf_2(vis, dataloader, theta, theta_weak, theta_inver, theta_weak_inver, results, results_weak, dataset_name, use_cuda=True):
    # Visualize watch images
    affTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)
    tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=use_cuda)
    pt = PointTnf(use_cuda=use_cuda)

    watch_images = torch.ones(len(dataloader) * 6, 3, 280, 240)
    watch_keypoints = -torch.ones(len(dataloader) * 6, 2, 20)
    if use_cuda:
        watch_images = watch_images.cuda()
        watch_keypoints = watch_keypoints.cuda()
    num_points = np.ones(len(dataloader) * 6).astype(np.int8)
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

    theta, theta_inver = swap(theta, theta_inver)
    theta_weak, theta_weak_inver = swap(theta_weak, theta_weak_inver)
    # means for normalize of caffe resnet and vgg
    # pixel_means = torch.Tensor(np.array([[[[102.9801, 115.9465, 122.7717]]]]).astype(np.float32))
    for batch_idx, batch in enumerate(dataloader):
        if use_cuda:
            batch = batch_cuda(batch)

        batch['source_image'], batch['target_image'] = swap(batch['source_image'], batch['target_image'])
        batch['source_im_info'], batch['target_im_info'] = swap(batch['source_im_info'], batch['target_im_info'])
        batch['source_points'], batch['target_points'] = swap(batch['source_points'], batch['target_points'])

        # Theta and theta_inver
        theta_aff = theta['aff'][batch_idx].unsqueeze(0)
        theta_aff_tps = theta['aff_tps'][batch_idx].unsqueeze(0)
        theta_weak_aff = theta_weak['aff'][batch_idx].unsqueeze(0)
        theta_weak_aff_tps = theta_weak['aff_tps'][batch_idx].unsqueeze(0)

        theta_aff_inver = theta_inver['aff'][batch_idx].unsqueeze(0)
        theta_aff_tps_inver = theta_inver['aff_tps'][batch_idx].unsqueeze(0)
        theta_weak_aff_inver = theta_weak_inver['aff'][batch_idx].unsqueeze(0)
        theta_weak_aff_tps_inver = theta_weak_inver['aff_tps'][batch_idx].unsqueeze(0)

        # Warped image
        warped_aff = affTnf(batch['source_image'], theta_aff)
        warped_aff_tps = tpsTnf(warped_aff, theta_aff_tps)
        warped_weak_aff = affTnf(batch['source_image'], theta_weak_aff)
        warped_weak_aff_tps = tpsTnf(warped_weak_aff, theta_weak_aff_tps)

        watch_images[batch_idx * 6, :, 0:240, :] = batch['source_image']
        watch_images[batch_idx * 6 + 1, :, 0:240, :] = warped_aff
        watch_images[batch_idx * 6 + 2, :, 0:240, :] = warped_aff_tps
        watch_images[batch_idx * 6 + 3, :, 0:240, :] = batch['target_image']
        watch_images[batch_idx * 6 + 4, :, 0:240, :] = warped_weak_aff
        watch_images[batch_idx * 6 + 5, :, 0:240, :] = warped_weak_aff_tps

        # Warped keypoints
        source_im_size = batch['source_im_info'][:, 0:3]
        target_im_size = batch['target_im_info'][:, 0:3]

        source_points = batch['source_points']
        target_points = batch['target_points']

        source_points_norm = PointsToUnitCoords(P=source_points, im_size=source_im_size)
        target_points_norm = PointsToUnitCoords(P=target_points, im_size=target_im_size)

        warped_points_aff_norm = pt.affPointTnf(theta=theta_aff_inver, points=source_points_norm)
        warped_points_aff = PointsToPixelCoords(P=warped_points_aff_norm, im_size=target_im_size)
        _, index_aff, N_pts = pck(target_points, warped_points_aff, dataset_name)
        warped_points_aff = relocate(warped_points_aff, target_im_size)

        warped_points_aff_tps_norm = pt.tpsPointTnf(theta=theta_aff_tps_inver, points=source_points_norm)
        warped_points_aff_tps_norm = pt.affPointTnf(theta=theta_aff_inver, points=warped_points_aff_tps_norm)
        warped_points_aff_tps = PointsToPixelCoords(P=warped_points_aff_tps_norm, im_size=target_im_size)
        _, index_aff_tps, _ = pck(target_points, warped_points_aff_tps, dataset_name)
        warped_points_aff_tps = relocate(warped_points_aff_tps, target_im_size)

        warped_points_weak_aff_norm = pt.affPointTnf(theta=theta_weak_aff_inver, points=source_points_norm)
        warped_points_weak_aff = PointsToPixelCoords(P=warped_points_weak_aff_norm, im_size=target_im_size)
        _, index_weak_aff, _ = pck(target_points, warped_points_weak_aff, dataset_name)
        warped_points_weak_aff = relocate(warped_points_weak_aff, target_im_size)

        warped_points_weak_aff_tps_norm = pt.tpsPointTnf(theta=theta_weak_aff_tps_inver, points=source_points_norm)
        warped_points_weak_aff_tps_norm = pt.affPointTnf(theta=theta_weak_aff_inver, points=warped_points_weak_aff_tps_norm)
        warped_points_weak_aff_tps = PointsToPixelCoords(P=warped_points_weak_aff_tps_norm, im_size=target_im_size)
        _, index_weak_aff_tps, _ = pck(target_points, warped_points_weak_aff_tps, dataset_name)
        warped_points_weak_aff_tps = relocate(warped_points_weak_aff_tps, target_im_size)

        watch_keypoints[batch_idx * 6, :, :N_pts] = relocate(batch['source_points'], source_im_size)[:, :, :N_pts]
        watch_keypoints[batch_idx * 6 + 1, :, :N_pts] = warped_points_aff[:, :, :N_pts]
        watch_keypoints[batch_idx * 6 + 2, :, :N_pts] = warped_points_aff_tps[:, :, :N_pts]
        watch_keypoints[batch_idx * 6 + 3, :, :N_pts] = relocate(batch['target_points'], target_im_size)[:, :, :N_pts]
        watch_keypoints[batch_idx * 6 + 4, :, :N_pts] = warped_points_weak_aff[:, :, :N_pts]
        watch_keypoints[batch_idx * 6 + 5, :, :N_pts] = warped_points_weak_aff_tps[:, :, :N_pts]

        num_points[batch_idx * 6:batch_idx * 6 + 6] = N_pts

        correct_index.append(np.arange(N_pts))
        correct_index.append(index_aff)
        correct_index.append(index_aff_tps)
        correct_index.append(np.arange(N_pts))
        correct_index.append(index_weak_aff)
        correct_index.append(index_weak_aff_tps)

        image_names.append('Source')
        image_names.append('Aff')
        image_names.append('Aff_tps')
        image_names.append('Target')
        image_names.append('Rocco_aff')
        image_names.append('Rocco_aff_tps')

        metrics.append('')
        metrics.append('PCK: {:.2%}'.format(float(results['aff']['pck'][batch_idx])))
        metrics.append('PCK: {:.2%}'.format(float(results['aff_tps']['pck'][batch_idx])))
        metrics.append('')
        metrics.append('PCK: {:.2%}'.format(float(results_weak['aff']['pck'][batch_idx])))
        metrics.append('PCK: {:.2%}'.format(float(results_weak['aff_tps']['pck'][batch_idx])))

    opts = dict(jpgquality=100, title=dataset_name)
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
        if (i + 1) % 6 == 1 or (i + 1) % 6 == 4:
            pos_pck = (0, 0)
        else:
            pos_pck = (70, 275)
        cv2.putText(watch_images[i], image_names[i], pos_name, fnt, 0.5, (0, 0, 0), 1)
        cv2.putText(watch_images[i], metrics[i], pos_pck, fnt, 0.5, (0, 0, 0), 1)
        if (i + 1) % 6 == 4:
            for j in range(num_points[i]):
                cv2.drawMarker(watch_images[i], (watch_keypoints[i, 0, j], watch_keypoints[i, 1, j]), colors[j], cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
        else:
            for j in correct_index[i]:
                cv2.drawMarker(watch_images[i], (watch_keypoints[i, 0, j], watch_keypoints[i, 1, j]), colors[j], cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)
                cv2.drawMarker(watch_images[i], (watch_keypoints[i + 3 - (i % 6), 0, j], watch_keypoints[i + 3 - (i % 6), 1, j]), colors[j], cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)

    watch_images = torch.Tensor(watch_images.astype(np.float32))
    watch_images = watch_images.permute(0, 3, 1, 2)
    vis.image(torchvision.utils.make_grid(watch_images, nrow=3, padding=3), opts=opts)