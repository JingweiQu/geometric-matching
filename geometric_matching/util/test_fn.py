# ========================================================================================
# Test geometric matching model
# Author: Jingwei Qu
# Date: 01 May 2019
# ========================================================================================

import torch
import time

from geometric_matching.util.net_util import *
from geometric_matching.geotnf.transformation import GeometricTnf
from geometric_matching.geotnf.affine_theta import AffineTheta
from geometric_matching.geotnf.point_tnf import *
from geometric_matching.data.pf_willow_pair import pf_willow_pair

def test_pf_willow(model, fasterRCNN, dataloader, use_cuda=True, crop_layer='image', with_affine=True):
    # Instantiate point transformer
    pt = PointTnf(use_cuda=use_cuda)

    # Instantiate image transformers
    tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=use_cuda)
    affTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)
    rescalingTnf = GeometricTnf(geometric_model='affine', out_h=240, out_w=240, use_cuda=use_cuda)
    affine_theta = AffineTheta(use_cuda=use_cuda, original=False, image_size=240)

    print('Computing PCK...')
    fasterRCNN.eval()
    thresh = 0.05
    max_per_image = 50
    model.eval()
    total_correct_points_tps = 0
    total_points = 0
    start = time.time()
    begin = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            ''' Move input batch to gpu '''
            if use_cuda:
                for k, v in batch.items():
                    batch[k] = batch[k].cuda()
            batch_size = batch['source_image'].shape[0]

            if with_affine:
                ''' Get the bounding box of the stand-out object in source image and target image'''
                # rois.shape: (batch_size, 300, 5), 5: (image_index_in_batch, x_min, y_min, x_max, y_max),
                # the coordinates is on the resized image (240*240), not the original image
                # cls_prob.shape: (batch_size, 300, n_classes), for PascalVOC n_classes=21
                # bbox_pred.shape: (batch_size, 300, 4 * n_classes), 4: (tx, ty, tw, th)
                rois_s, cls_prob_s, bbox_pred_s, _, _, _, _, _ = fasterRCNN(batch['source_im'], batch['source_im_info'],
                                                                            batch['source_gt_boxes'],
                                                                            batch['source_num_boxes'])
                # Compute and select bounding boxes for objects in the image
                all_boxes_s = select_boxes(rois_s, cls_prob_s, bbox_pred_s, batch['source_im_info'], thresh, max_per_image)

                rois_t, cls_prob_t, bbox_pred_t, _, _, _, _, _ = fasterRCNN(batch['target_im'], batch['target_im_info'],
                                                                            batch['target_gt_boxes'],
                                                                            batch['target_num_boxes'])
                all_boxes_t = select_boxes(rois_t, cls_prob_t, bbox_pred_t, batch['target_im_info'], thresh, max_per_image)

                # Select the bounding box with the highest score in the source image
                # Select tht bounding box with the same class as the above box in the target image
                # If no selected bounding box, make empty box
                # boxes_s, boxes_t.shape: (batch_size, 4), 4: (x_min, y_min, x_max, y_max)
                boxes_s, boxes_t = select_box(all_boxes_s, all_boxes_t)

                # batch = pf_willow_pair(batch, boxes_s, boxes_t, rescalingTnf)

                if use_cuda:
                    boxes_s = boxes_s.cuda()
                    boxes_t = boxes_t.cuda()

            ''' Get the testing pair {source image, target image} '''
            # Compute affine parameters based on object detection of fasterRCNN
            if with_affine:
                theta_aff = affine_theta(boxes_s=boxes_s, boxes_t=boxes_t, source_im_size=None, target_im_size=None)
                source_image = batch['source_image'].clone()
                batch['source_image'] = affTnf(batch['source_image'], theta_aff)

            tnf_batch = {'source_image': batch['source_image'], 'target_image': batch['target_image']}
            if crop_layer == 'pool4' or crop_layer == 'conv1':
                tnf_batch['source_box'] = boxes_s
                tnf_batch['target_box'] = boxes_t

            ''' Test the model '''
            theta = model(tnf_batch)

            source_im_size = batch['source_im_size']
            target_im_size = batch['target_im_size']

            source_points = batch['source_points']
            target_points = batch['target_points']

            # Warp points with estimated transformations
            target_points_norm = PointsToUnitCoords(target_points, target_im_size)
            # warped_points_tps_norm = pt.affPointTnf(theta_aff, target_points_norm)  # Affine
            # warped_points_tps_norm = pt.tpsPointTnf(theta, warped_points_tps_norm)  # TPS
            warped_points_tps_norm = pt.tpsPointTnf(theta, target_points_norm)  # TPS
            if with_affine:
                warped_points_tps_norm = pt.affPointTnf(theta_aff, warped_points_tps_norm)  # Affine
            warped_points_tps = PointsToPixelCoords(warped_points_tps_norm, source_im_size)

            correct_points_tps, num_points = correct_keypoints(source_points, warped_points_tps, batch['L_pck'])
            total_correct_points_tps += correct_points_tps
            total_points += num_points

            end = time.time()
            print('Batch: [{}/{} ({:.0f}%)]\t\tTime cost {:.4f}'.format(
                batch_idx, len(dataloader), 100. * batch_idx / len(dataloader), end - start))
            start = time.time()

            # warped_image_tps = tpsTnf(source_image, theta)
            # warped_image = tpsTnf(tnf_batch['source_image'], theta)

            '''
            # Show images
            rows = 1
            cols = 3
            for i in range(tnf_batch['source_image'].shape[0]):
                show_id = i

                source_point = source_points[show_id, :, :].cpu().numpy()
                source_size = source_im_size[show_id, :].cpu().numpy()

                target_point = target_points[show_id, :, :].cpu().numpy()
                target_size = target_im_size[show_id, :].cpu().numpy()

                warped_point_tps = warped_points_tps[show_id, :, :].cpu().numpy()

                source_point[0, :] = source_point[0, :] * (240 / source_size[1])
                source_point[1, :] = source_point[1, :] * (240 / source_size[0])

                target_point[0, :] = target_point[0, :] * (240 / target_size[1])
                target_point[1, :] = target_point[1, :] * (240 / target_size[0])

                warped_point_tps[0, :] = warped_point_tps[0, :] * (240 / source_size[1])
                warped_point_tps[1, :] = warped_point_tps[1, :] * (240 / source_size[0])

                # ax = im_show_1(source_image[show_id], 'source_image', rows, cols, 1)
                # show_boxes(ax, all_boxes_s[show_id])
                # ax.scatter(source_point[0, :], source_point[1, :], c='b', marker='o', s=30, zorder=2)
                # ax.scatter(warped_point_tps[0, :], warped_point_tps[1, :], c='g', marker='x', s=30, zorder=2)

                ax = im_show_1(tnf_batch['source_image'][show_id], 'source_image', rows, cols, 1)
                
                # im_show_1(warped_image_tps[show_id], 'tps', rows, cols, 3)

                ax = im_show_1(warped_image[show_id], 'tps', rows, cols, 2)

                ax = im_show_1(tnf_batch['target_image'][show_id], 'target_image', rows, cols, 3)
                # show_boxes(ax, all_boxes_t[show_id])
                # show_boxes(ax, boxes_t[show_id, :].reshape(1, -1))
                # ax.scatter(target_point[0, :], target_point[1, :], c='g', marker='x', s=30, zorder=2)

                plt.show()
            '''

        end = time.time()
        PCK_tps = total_correct_points_tps / total_points
        print('PCK tps {:}\t\tCorrect points {:}\tTotal points {:}\t\tTotal time cost {:.4f}'.format(
            PCK_tps, total_correct_points_tps, total_points, end - begin))
        return PCK_tps, total_correct_points_tps, total_points