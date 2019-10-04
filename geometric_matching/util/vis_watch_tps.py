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
from geometric_matching.geotnf.transformation_tps import GeometricTnf as GeometricTnf2
from geometric_matching.image.normalization import normalize_image
from geometric_matching.util.net_util import *
from geometric_matching.geotnf.point_tnf import PointTnf, PointsToUnitCoords, PointsToPixelCoords
from geometric_matching.geotnf.point_tps import PointTPS

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

def vis_pf(vis, dataloader, theta_1, theta_2, theta_inver_1, theta_inver_2, results_1, results_2, dataset_name, use_cuda=True):
    # Visualize watch images
    tpsTnf_1 = GeometricTnf(geometric_model='tps', use_cuda=use_cuda)
    tpsTnf_2 = GeometricTnf2(geometric_model='tps', use_cuda=use_cuda)
    pt_1 = PointTnf(use_cuda=use_cuda)
    pt_2 = PointTPS(use_cuda=use_cuda)

    group_size = 4
    watch_images = torch.ones(len(dataloader) * group_size, 3, 280, 240)
    watch_keypoints = -torch.ones(len(dataloader) * group_size, 2, 20)
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
        theta_tps_1 = theta_1['tps'][batch_idx].unsqueeze(0)
        theta_tps_2 = theta_2['tps'][batch_idx].unsqueeze(0)

        thetai_tps_1 = theta_inver_1['tps'][batch_idx].unsqueeze(0)
        thetai_tps_2 = theta_inver_2['tps'][batch_idx].unsqueeze(0)

        # Warped image
        warped_tps_1 = tpsTnf_1(batch['source_image'], theta_tps_1)
        warped_tps_2 = tpsTnf_2(batch['source_image'], theta_tps_2)

        watch_images[batch_idx * group_size, :, 0:240, :] = batch['source_image']
        watch_images[batch_idx * group_size + 1, :, 0:240, :] = warped_tps_1
        watch_images[batch_idx * group_size + 2, :, 0:240, :] = warped_tps_2
        watch_images[batch_idx * group_size + 3, :, 0:240, :] = batch['target_image']

        # Warped keypoints
        source_im_size = batch['source_im_info'][:, 0:3]
        target_im_size = batch['target_im_info'][:, 0:3]

        source_points = batch['source_points']
        target_points = batch['target_points']

        source_points_norm = PointsToUnitCoords(P=source_points, im_size=source_im_size)
        target_points_norm = PointsToUnitCoords(P=target_points, im_size=target_im_size)

        warped_points_tps_norm_1 = pt_1.tpsPointTnf(theta=thetai_tps_1, points=source_points_norm)
        warped_points_tps_1 = PointsToPixelCoords(P=warped_points_tps_norm_1, im_size=target_im_size)
        pck_tps_1, index_tps_1, N_pts = pck(target_points, warped_points_tps_1, dataset_name)
        warped_points_tps_1 = relocate(warped_points_tps_1, target_im_size)

        warped_points_tps_norm_2 = pt_2.tpsPointTnf(theta=thetai_tps_2, points=source_points_norm)
        warped_points_tps_2 = PointsToPixelCoords(P=warped_points_tps_norm_2, im_size=target_im_size)
        pck_tps_2, index_tps_2, _ = pck(target_points, warped_points_tps_2, dataset_name)
        warped_points_tps_2 = relocate(warped_points_tps_2, target_im_size)

        watch_keypoints[batch_idx * group_size, :, :N_pts] = relocate(batch['source_points'], source_im_size)[:, :, :N_pts]
        watch_keypoints[batch_idx * group_size + 1, :, :N_pts] = warped_points_tps_1[:, :, :N_pts]
        watch_keypoints[batch_idx * group_size + 2, :, :N_pts] = warped_points_tps_2[:, :, :N_pts]
        watch_keypoints[batch_idx * group_size + 3, :, :N_pts] = relocate(batch['target_points'], target_im_size)[:, :, :N_pts]

        num_points[batch_idx * group_size:batch_idx * group_size + group_size] = N_pts

        correct_index.append(np.arange(N_pts))
        correct_index.append(index_tps_1)
        correct_index.append(index_tps_2)
        correct_index.append(np.arange(N_pts))

        image_names.append('Source')
        image_names.append('TPS')
        image_names.append('TPS_Jitter')
        image_names.append('Target')

        metrics.append('')
        metrics.append('PCK: {:.2%}'.format(pck_tps_1))
        metrics.append('PCK: {:.2%}'.format(pck_tps_2))
        metrics.append('')

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
        if (i + 1) % group_size == 1 or (i + 1) % group_size == 0:
            pos_pck = (0, 0)
        else:
            pos_pck = (70, 275)
        cv2.putText(watch_images[i], image_names[i], pos_name, fnt, 0.5, (0, 0, 0), 1)
        cv2.putText(watch_images[i], metrics[i], pos_pck, fnt, 0.5, (0, 0, 0), 1)
        if (i + 1) % group_size == 0:
            for j in range(num_points[i]):
                cv2.drawMarker(watch_images[i], (watch_keypoints[i, 0, j], watch_keypoints[i, 1, j]), colors[j], cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)
        else:
            for j in correct_index[i]:
                cv2.drawMarker(watch_images[i], (watch_keypoints[i, 0, j], watch_keypoints[i, 1, j]), colors[j], cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
                cv2.drawMarker(watch_images[i], (watch_keypoints[i + (group_size - 1) - (i % group_size), 0, j], watch_keypoints[i + (group_size - 1) - (i % group_size), 1, j]), colors[j], cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)

    watch_images = torch.Tensor(watch_images.astype(np.float32))
    watch_images = watch_images.permute(0, 3, 1, 2)
    vis.image(torchvision.utils.make_grid(watch_images, nrow=4, padding=5), opts=opts)

def vis_caltech(vis, dataloader, theta_1, theta_2, results_1, results_2, dataset_name, use_cuda=True):
    # Visualize watch images
    tpsTnf_1 = GeometricTnf(geometric_model='tps', use_cuda=use_cuda)
    tpsTnf_2 = GeometricTnf2(geometric_model='tps', use_cuda=use_cuda)

    group_size = 4
    watch_images = torch.ones(len(dataloader) * group_size, 3, 280, 240)
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

        theta_tps_1 = theta_1['tps'][batch_idx].unsqueeze(0)
        theta_tps_2 = theta_2['tps'][batch_idx].unsqueeze(0)

        # Warped image
        warped_tps_1 = tpsTnf_1(batch['source_image'], theta_tps_1)
        warped_tps_2 = tpsTnf_2(batch['source_image'], theta_tps_2)

        watch_images[batch_idx * group_size, :, 0:240, :] = batch['source_image']
        watch_images[batch_idx * group_size + 1, :, 0:240, :] = warped_tps_1
        watch_images[batch_idx * group_size + 2, :, 0:240, :] = warped_tps_2
        watch_images[batch_idx * group_size + 3, :, 0:240, :] = batch['target_image']

        image_names.append('Source')
        image_names.append('TPS')
        image_names.append('TPS_Jitter')
        image_names.append('Target')

        lt_acc.append('')
        lt_acc.append('LT-ACC: {:.2f}'.format(float(results_1['tps']['label_transfer_accuracy'][batch_idx])))
        lt_acc.append('LT-ACC: {:.2f}'.format(float(results_2['tps']['label_transfer_accuracy'][batch_idx])))
        lt_acc.append('')

        iou.append('')
        iou.append('IoU: {:.2f}'.format(float(results_1['tps']['intersection_over_union'][batch_idx])))
        iou.append('IoU: {:.2f}'.format(float(results_2['tps']['intersection_over_union'][batch_idx])))
        iou.append('')

    opts = dict(jpgquality=100, title=dataset_name)
    # Un-normalize for caffe resnet and vgg
    # watch_images = watch_images.permute(0, 2, 3, 1) + pixel_means
    # watch_images = watch_images[:, :, :, [2, 1, 0]].permute(0, 3, 1, 2)
    # watch_images = normalize_image(watch_images, forward=False) * 255.0
    watch_images[:, :, 0:240, :] = normalize_image(watch_images[:, :, 0:240, :], forward=False)
    watch_images *= 255.0
    watch_images = watch_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    for i in range(watch_images.shape[0]):
        pos_name = (80, 255)
        if (i + 1) % group_size == 1 or (i + 1) % group_size == 0:
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
    vis.image(torchvision.utils.make_grid(watch_images, nrow=4, padding=5), opts=opts)

def vis_tss(vis, dataloader, theta_1, theta_2, csv_file, dataset_name, use_cuda=True):
    # Visualize watch images
    dataframe = pd.read_csv(csv_file)
    scores_tps_1 = dataframe.iloc[:, 5]
    scores_tps_2 = dataframe.iloc[:, 6]
    tpsTnf_1 = GeometricTnf(geometric_model='tps', use_cuda=use_cuda)
    tpsTnf_2 = GeometricTnf2(geometric_model='tps', use_cuda=use_cuda)

    group_size = 4
    watch_images = torch.ones(int(len(dataloader) / 2 * group_size), 3, 280, 240)
    watch_images_inver = torch.ones(int(len(dataloader) / 2 * group_size), 3, 280, 240)
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

        theta_tps_1 = theta_1['tps'][batch_idx].unsqueeze(0)
        theta_tps_2 = theta_2['tps'][batch_idx].unsqueeze(0)

        # Warped image
        warped_tps_1 = tpsTnf_1(batch['source_image'], theta_tps_1)
        warped_tps_2 = tpsTnf_2(batch['source_image'], theta_tps_2)

        if (batch_idx + 1) % 2 != 0:
            watch_images[int(batch_idx / 2 * group_size), :, 0:240, :] = batch['source_image']
            watch_images[int(batch_idx / 2 * group_size) + 1, :, 0:240, :] = warped_tps_1
            watch_images[int(batch_idx / 2 * group_size) + 2, :, 0:240, :] = warped_tps_2
            watch_images[int(batch_idx / 2 * group_size) + 3, :, 0:240, :] = batch['target_image']

            image_names.append('Source')
            image_names.append('TPS')
            image_names.append('TPS_Jitter')
            image_names.append('Target')

            flow.append('')
            flow.append('Flow: {:.3f}'.format(scores_tps_1[batch_idx]))
            flow.append('Flow: {:.3f}'.format(scores_tps_2[batch_idx]))
            flow.append('')
        else:
            watch_images_inver[int((batch_idx - 1) / 2 * group_size), :, 0:240, :] = batch['source_image']
            watch_images_inver[int((batch_idx - 1) / 2 * group_size) + 1, :, 0:240, :] = warped_tps_1
            watch_images_inver[int((batch_idx - 1) / 2 * group_size) + 2, :, 0:240, :] = warped_tps_2
            watch_images_inver[int((batch_idx - 1) / 2 * group_size) + 3, :, 0:240, :] = batch['target_image']

            image_names_inver.append('Source')
            image_names_inver.append('TPS')
            image_names_inver.append('TPS_Jitter')
            image_names_inver.append('Target')

            flow_inver.append('')
            flow_inver.append('Flow: {:.3f}'.format(scores_tps_1[batch_idx]))
            flow_inver.append('Flow: {:.3f}'.format(scores_tps_2[batch_idx]))
            flow_inver.append('')

    opts = dict(jpgquality=100, title=dataset_name)
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
            if (i + 1) % group_size == 1 or (i + 1) % group_size == 0:
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
    vis.image(torchvision.utils.make_grid(watch_images, nrow=4, padding=5), opts=opts)
    # vis.image(torchvision.utils.make_grid(watch_images_inver, nrow=3, padding=5), opts=opts)

def swap(source, target):
    tmp = source
    source = target
    target = tmp

    return source, target

def vis_pf_2(vis, dataloader, theta_1, theta_2, theta_inver_1, theta_inver_2, results_1, results_2, dataset_name, use_cuda=True):
    # Visualize watch images
    tpsTnf_1 = GeometricTnf(geometric_model='tps', use_cuda=use_cuda)
    tpsTnf_2 = GeometricTnf2(geometric_model='tps', use_cuda=use_cuda)
    pt_1 = PointTnf(use_cuda=use_cuda)
    pt_2 = PointTPS(use_cuda=use_cuda)

    group_size = 4
    watch_images = torch.ones(len(dataloader) * group_size, 3, 280, 240)
    watch_keypoints = -torch.ones(len(dataloader) * group_size, 2, 20)
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

    theta_1, theta_inver_1 = swap(theta_1, theta_inver_1)
    theta_2, theta_inver_2 = swap(theta_2, theta_inver_2)
    # means for normalize of caffe resnet and vgg
    # pixel_means = torch.Tensor(np.array([[[[102.9801, 115.9465, 122.7717]]]]).astype(np.float32))
    for batch_idx, batch in enumerate(dataloader):
        if use_cuda:
            batch = batch_cuda(batch)

        batch['source_image'], batch['target_image'] = swap(batch['source_image'], batch['target_image'])
        batch['source_im_info'], batch['target_im_info'] = swap(batch['source_im_info'], batch['target_im_info'])
        batch['source_points'], batch['target_points'] = swap(batch['source_points'], batch['target_points'])

        # Theta and theta_inver
        theta_tps_1 = theta_1['tps'][batch_idx].unsqueeze(0)
        theta_tps_2 = theta_2['tps'][batch_idx].unsqueeze(0)

        thetai_tps_1 = theta_inver_1['tps'][batch_idx].unsqueeze(0)
        thetai_tps_2 = theta_inver_2['tps'][batch_idx].unsqueeze(0)

        # Warped image
        warped_tps_1 = tpsTnf_1(batch['source_image'], theta_tps_1)
        warped_tps_2 = tpsTnf_2(batch['source_image'], theta_tps_2)

        watch_images[batch_idx * group_size, :, 0:240, :] = batch['source_image']
        watch_images[batch_idx * group_size + 1, :, 0:240, :] = warped_tps_1
        watch_images[batch_idx * group_size + 2, :, 0:240, :] = warped_tps_2
        watch_images[batch_idx * group_size + 3, :, 0:240, :] = batch['target_image']

        # Warped keypoints
        source_im_size = batch['source_im_info'][:, 0:3]
        target_im_size = batch['target_im_info'][:, 0:3]

        source_points = batch['source_points']
        target_points = batch['target_points']

        source_points_norm = PointsToUnitCoords(P=source_points, im_size=source_im_size)
        target_points_norm = PointsToUnitCoords(P=target_points, im_size=target_im_size)

        warped_points_tps_norm_1 = pt_1.tpsPointTnf(theta=thetai_tps_1, points=source_points_norm)
        warped_points_tps_1 = PointsToPixelCoords(P=warped_points_tps_norm_1, im_size=target_im_size)
        _, index_tps_1, N_pts = pck(target_points, warped_points_tps_1, dataset_name)
        warped_points_tps_1 = relocate(warped_points_tps_1, target_im_size)

        warped_points_tps_norm_2 = pt_2.tpsPointTnf(theta=thetai_tps_2, points=source_points_norm)
        warped_points_tps_2 = PointsToPixelCoords(P=warped_points_tps_norm_2, im_size=target_im_size)
        _, index_tps_2, _ = pck(target_points, warped_points_tps_2, dataset_name)
        warped_points_tps_2 = relocate(warped_points_tps_2, target_im_size)

        watch_keypoints[batch_idx * group_size, :, :N_pts] = relocate(batch['source_points'], source_im_size)[:, :, :N_pts]
        watch_keypoints[batch_idx * group_size + 1, :, :N_pts] = warped_points_tps_1[:, :, :N_pts]
        watch_keypoints[batch_idx * group_size + 2, :, :N_pts] = warped_points_tps_2[:, :, :N_pts]
        watch_keypoints[batch_idx * group_size + 3, :, :N_pts] = relocate(batch['target_points'], target_im_size)[:, :, :N_pts]

        num_points[batch_idx * group_size:batch_idx * group_size + group_size] = N_pts

        correct_index.append(np.arange(N_pts))
        correct_index.append(index_tps_1)
        correct_index.append(index_tps_2)
        correct_index.append(np.arange(N_pts))

        image_names.append('Source')
        image_names.append('TPS')
        image_names.append('TPS_Jitter')
        image_names.append('Target')

        metrics.append('')
        metrics.append('PCK: {:.2%}'.format(float(results_1['tps']['pck'][batch_idx])))
        metrics.append('PCK: {:.2%}'.format(float(results_2['tps']['pck'][batch_idx])))
        metrics.append('')

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
        if (i + 1) % group_size == 1 or (i + 1) % group_size == 0:
            pos_pck = (0, 0)
        else:
            pos_pck = (70, 275)
        cv2.putText(watch_images[i], image_names[i], pos_name, fnt, 0.5, (0, 0, 0), 1)
        cv2.putText(watch_images[i], metrics[i], pos_pck, fnt, 0.5, (0, 0, 0), 1)
        if (i + 1) % group_size == 0:
            for j in range(num_points[i]):
                cv2.drawMarker(watch_images[i], (watch_keypoints[i, 0, j], watch_keypoints[i, 1, j]), colors[j], cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
        else:
            for j in correct_index[i]:
                cv2.drawMarker(watch_images[i], (watch_keypoints[i, 0, j], watch_keypoints[i, 1, j]), colors[j], cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)
                cv2.drawMarker(watch_images[i], (watch_keypoints[i + (group_size - 1) - (i % group_size), 0, j], watch_keypoints[i + (group_size - 1) - (i % group_size), 1, j]), colors[j], cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)

    watch_images = torch.Tensor(watch_images.astype(np.float32))
    watch_images = watch_images.permute(0, 3, 1, 2)
    vis.image(torchvision.utils.make_grid(watch_images, nrow=4, padding=5), opts=opts)

def vis_control(vis, dataloader, theta_1, theta_2, dataset_name, use_cuda=True):
    # Visualize watch images
    tpsTnf_1 = GeometricTnf(geometric_model='tps', use_cuda=use_cuda)
    tpsTnf_2 = GeometricTnf2(geometric_model='tps', use_cuda=use_cuda)

    group_size = 5
    watch_images = torch.ones(len(dataloader) * group_size, 3, 340, 340)
    if use_cuda:
        watch_images = watch_images.cuda()

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
        theta_tps_1 = theta_1['tps'][batch_idx].unsqueeze(0)
        theta_tps_2 = theta_2['tps'][batch_idx].unsqueeze(0)

        # Warped image
        warped_tps_1 = tpsTnf_1(batch['source_image'], theta_tps_1)
        warped_tps_2 = tpsTnf_2(batch['source_image'], theta_tps_2)

        watch_images[batch_idx * group_size, :, 50:290, 50:290] = batch['source_image']
        watch_images[batch_idx * group_size + 1, :, 50:290, 50:290] = warped_tps_1
        watch_images[batch_idx * group_size + 2, :, 50:290, 50:290] = batch['source_image']
        watch_images[batch_idx * group_size + 3, :, 50:290, 50:290] = warped_tps_2
        watch_images[batch_idx * group_size + 4, :, 50:290, 50:290] = batch['target_image']


    opts = dict(jpgquality=100, title=dataset_name)
    watch_images[:, :, 50:290, 50:290] = normalize_image(watch_images[:, :, 50:290, 50:290], forward=False)
    watch_images *= 255.0
    watch_images = watch_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    im_size = torch.Tensor([[240, 240]]).cuda()
    for i in range(watch_images.shape[0]):
        if i % group_size == 0:
            cp_norm = theta_1['tps'][int(i/group_size)].view(1, 2, -1)
            cp = PointsToPixelCoords(P=cp_norm, im_size=im_size)
            cp = cp.squeeze().cpu().numpy() + 50
            for j in range(9):
                cv2.drawMarker(watch_images[i], (cp[0, j], cp[1, j]), (0, 0, 255), cv2.MARKER_TILTED_CROSS , 12, 2, cv2.LINE_AA)

            for j in range(2):
                for k in range(3):
                    # vertical grid
                    cv2.line(watch_images[i], (cp[0, j + k * 3], cp[1, j + k * 3]), (cp[0, j + k * 3 + 1], cp[1, j + k * 3 + 1]), (0, 0, 255), 2, cv2.LINE_AA)
                    # horizontal grid
                    cv2.line(watch_images[i], (cp[0, j * 3 + k], cp[1, j * 3 + k]), (cp[0, j * 3 + k + 3], cp[1, j * 3 + k + 3]), (0, 0, 255), 2, cv2.LINE_AA)

        if i % group_size == 1:
            cp_norm = torch.Tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1]).cuda().view(1, 2, -1)
            cp = PointsToPixelCoords(P=cp_norm, im_size=im_size)
            cp = cp.squeeze().cpu().numpy() + 50
            for j in range(9):
                cv2.drawMarker(watch_images[i], (cp[0, j], cp[1, j]), (0, 0, 255), cv2.MARKER_TILTED_CROSS , 12, 2, cv2.LINE_AA)

            for j in range(1):
                for k in range(3):
                    # vertical grid
                    cv2.line(watch_images[i], (cp[0, j + k * 3], cp[1, j + k * 3]), (cp[0, j + k * 3 + 1], cp[1, j + k * 3 + 1]), (0, 0, 255), 2, cv2.LINE_AA)
                    # horizontal grid
                    cv2.line(watch_images[i], (cp[0, j * 3 + k], cp[1, j * 3 + k]), (cp[0, j * 3 + k + 3], cp[1, j * 3 + k + 3]), (0, 0, 255), 2, cv2.LINE_AA)

        if i % group_size == 2:
            cp_norm = theta_2['tps'][int(i/group_size)][:18].view(1, 2, -1)
            cp = PointsToPixelCoords(P=cp_norm, im_size=im_size)
            cp = cp.squeeze().cpu().numpy() + 50
            for j in range(9):
                cv2.drawMarker(watch_images[i], (cp[0, j], cp[1, j]), (0, 0, 255), cv2.MARKER_TILTED_CROSS , 12, 2, cv2.LINE_AA)

            for j in range(2):
                for k in range(3):
                    # vertical grid
                    cv2.line(watch_images[i], (cp[0, j + k * 3], cp[1, j + k * 3]), (cp[0, j + k * 3 + 1], cp[1, j + k * 3 + 1]), (0, 0, 255), 2, cv2.LINE_AA)
                    # horizontal grid
                    cv2.line(watch_images[i], (cp[0, j * 3 + k], cp[1, j * 3 + k]), (cp[0, j * 3 + k + 3], cp[1, j * 3 + k + 3]), (0, 0, 255), 2, cv2.LINE_AA)

        if i % group_size == 3:
            cp_norm = theta_2['tps'][int(i/group_size)][18:].view(1, 2, -1)
            cp = PointsToPixelCoords(P=cp_norm, im_size=im_size)
            cp = cp.squeeze().cpu().numpy() + 50
            for j in range(9):
                cv2.drawMarker(watch_images[i], (cp[0, j], cp[1, j]), (0, 0, 255), cv2.MARKER_TILTED_CROSS , 12, 2, cv2.LINE_AA)

            for j in range(2):
                for k in range(3):
                    # vertical grid
                    cv2.line(watch_images[i], (cp[0, j + k * 3], cp[1, j + k * 3]), (cp[0, j + k * 3 + 1], cp[1, j + k * 3 + 1]), (0, 0, 255), 2, cv2.LINE_AA)
                    # horizontal grid
                    cv2.line(watch_images[i], (cp[0, j * 3 + k], cp[1, j * 3 + k]), (cp[0, j * 3 + k + 3], cp[1, j * 3 + k + 3]), (0, 0, 255), 2, cv2.LINE_AA)

    watch_images = torch.Tensor(watch_images.astype(np.float32))
    watch_images = watch_images.permute(0, 3, 1, 2)
    vis.image(torchvision.utils.make_grid(watch_images, nrow=5, padding=5), opts=opts)

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

def vis_control2(vis, dataloader, theta_1, theta_2, dataset_name, use_cuda=True):
    # Visualize watch images
    tpsTnf = GeometricTnf2(geometric_model='tps', use_cuda=use_cuda)

    group_size = 5
    watch_images = torch.ones(len(dataloader) * group_size, 3, 340, 340)
    if use_cuda:
        watch_images = watch_images.cuda()

    # means for normalize of caffe resnet and vgg
    # pixel_means = torch.Tensor(np.array([[[[102.9801, 115.9465, 122.7717]]]]).astype(np.float32))
    for batch_idx, batch in enumerate(dataloader):
        if use_cuda:
            batch = batch_cuda(batch)

        # Theta and theta_inver
        theta_tps_1 = theta_1['tps'][batch_idx].unsqueeze(0)
        theta_tps_2 = theta_2['tps'][batch_idx].unsqueeze(0)

        # Warped image
        warped_tps_1 = tpsTnf(batch['source_image'], theta_tps_1)
        warped_tps_2 = tpsTnf(batch['source_image'], theta_tps_2)

        watch_images[batch_idx * group_size, :, 50:290, 50:290] = batch['source_image']
        watch_images[batch_idx * group_size + 1, :, 50:290, 50:290] = warped_tps_1
        watch_images[batch_idx * group_size + 2, :, 50:290, 50:290] = batch['source_image']
        watch_images[batch_idx * group_size + 3, :, 50:290, 50:290] = warped_tps_2
        watch_images[batch_idx * group_size + 4, :, 50:290, 50:290] = batch['target_image']


    opts = dict(jpgquality=100, title=dataset_name)
    watch_images[:, :, 50:290, 50:290] = normalize_image(watch_images[:, :, 50:290, 50:290], forward=False)
    watch_images *= 255.0
    watch_images = watch_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    im_size = torch.Tensor([[240, 240]]).cuda()
    for i in range(watch_images.shape[0]):
        if i % group_size < 4:
            if i % group_size == 0:
                cp_norm = theta_1['tps'][int(i / group_size)][:18].view(1, 2, -1)

            if i % group_size == 1:
                cp_norm = theta_1['tps'][int(i / group_size)][18:].view(1, 2, -1)

            if i % group_size == 2:
                cp_norm = theta_2['tps'][int(i / group_size)][:18].view(1, 2, -1)

            if i % group_size == 3:
                cp_norm = theta_2['tps'][int(i / group_size)][18:].view(1, 2, -1)
            watch_images[i] = draw_grid(watch_images[i], cp_norm)

    watch_images = torch.Tensor(watch_images.astype(np.float32))
    watch_images = watch_images.permute(0, 3, 1, 2)
    vis.image(torchvision.utils.make_grid(watch_images, nrow=5, padding=5), opts=opts)