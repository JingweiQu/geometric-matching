import torch
import numpy as np
import os
import time
from skimage import draw
from geometric_matching.geotnf.flow import th_sampling_grid_to_np_flow, write_flo_file
import torch.nn.functional as F
from geometric_matching.gm_data.pf_pascal_dataset import PFPASCALDataset
# from geometric_matching.gm_data.caltech_dataset import CaltechDataset
from geometric_matching.geotnf.point_tps2 import PointTPS, PointsToUnitCoords, PointsToPixelCoords
from geometric_matching.geotnf.affine_theta import AffineTheta
from geometric_matching.util.net_util import *
import matplotlib
import matplotlib.pyplot as plt

def test_fn(model=None, faster_rcnn=None, aff_theta=None, metric='pck', batch_size=1, dataset=None, dataloader=None, detect=False, dual=False, do_aff=False, do_tps=False, args=None):
    # Initialize results
    N = len(dataset)
    results = {}
    # decide which results should be computed aff/tps/aff+tps
    if do_aff:
        results['aff']={}
    if do_tps:
        results['tps'] = {}
    if dual:
        results['aff'] = {}
        results['afftps'] = {}
    if detect:
        results['det'] = {}
        results['det_aff']={}
        results['det_aff_tps'] = {}
    # choose metric function and metrics to compute
    if metric == 'pck':
        metrics = ['pck']
        metric_fun = pck_metric
    elif metric == 'area':
        metrics = ['label_transfer_accuracy',
                   'intersection_over_union',
                   'localization_error']
        metric_fun = area_metrics
    elif metric == 'flow':
        metrics = ['flow']
        metric_fun = flow_metrics
    # elif metric == 'pascal_parts':
    #     metrics = ['intersection_over_union', 'pck']
        # metric_fun = pascal_parts_metrics
    # elif metric == 'dist':
    #     metrics = ['dist']
        # metric_fun = point_dist_metric

    # initialize vector for storing results for each metric
    for key in results.keys():
        for metric in metrics:
            results[key][metric] = np.zeros((N, 1))

    # Compute
    begin = time.time()
    for batch_idx, batch in enumerate(dataloader):
        if args.cuda:
            batch = batch_cuda(batch)
        batch_start_idx = batch_size * batch_idx
        batch_end_idx = np.minimum(batch_start_idx + batch_size, N)
        # current_batch_size = batch['source_im_info'].size(0)
        # batch_start_idx = current_batch_size * batch_idx

        theta_det = None
        theta_aff = None
        theta_tps = None
        theta_afftps = None

        # theta_aff is predicted by geometric model, theta_aff_det is computed by detection results of faster rcnn
        if detect:
            box_info_s = faster_rcnn(im_data=batch['source_im'], im_info=batch['source_im_info'][:, 3:],
                                     gt_boxes=batch['source_gt_boxes'], num_boxes=batch['source_num_boxes'])[0:3]
            box_info_t = faster_rcnn(im_data=batch['target_im'], im_info=batch['target_im_info'][:, 3:],
                                     gt_boxes=batch['target_gt_boxes'], num_boxes=batch['target_num_boxes'])[0:3]
            all_box_s = select_boxes(rois=box_info_s[0], cls_prob=box_info_s[1], bbox_pred=box_info_s[2], im_infos=batch['source_im_info'][:, 3:])
            all_box_t = select_boxes(rois=box_info_t[0], cls_prob=box_info_t[1], bbox_pred=box_info_t[2], im_infos=batch['target_im_info'][:, 3:])
            box_s, box_t = select_box_st(all_box_s, all_box_t)
            theta_det = aff_theta(boxes_s=box_s, boxes_t=box_t)
            theta_afftps, theta_aff = model(batch, theta_det)
        if dual:
            theta_afftps, theta_aff = model(batch)
        if do_aff:
            theta_aff = model(batch)
        if do_tps:
            theta_tps = model(batch)

        results = metric_fun(batch, batch_start_idx, theta_det, theta_aff, theta_tps, theta_afftps, results, args)

        end = time.time()
        print('Batch: [{}/{} ({:.0%})]\t\tTime cost ({} batches): {:.4f} s'.format(batch_idx+1, len(dataloader), (batch_idx+1) / len(dataloader), batch_idx + 1, end - begin))
    end = time.time()
    print('Dataset time cost: {:.4f} s'.format(end - begin))

    # Print results
    if metric == 'flow':
        print('Flow files have been saved to ' + args.flow_output_dir)
        return results, end - begin

    for key in results.keys():
        print('=== Results ' + key + ' ===')
        for metric in metrics:
            # print per-class brakedown for PFPascal, or caltech
            # if isinstance(dataset, PFPASCALDataset) or isinstance(dataset, CaltechDataset):
            if isinstance(dataset, PFPASCALDataset):
                N_cat = int(np.max(dataset.categories))  # Number of categories in dataset (PF-PASCAL or Caltech-101)
                for c in range(N_cat):
                    cat_idx = np.nonzero(dataset.categories == c + 1)[0]  # Compute indices of current category
                    # print('{}: {:.2%}'.format(dataset.category_names[c].ljust(15), np.mean(results[key][metric][cat_idx])))
                    print('{}: {:.4}'.format(dataset.category_names[c].ljust(15), np.mean(results[key][metric][cat_idx])))

            # print mean value
            values = results[key][metric]
            good_idx = np.flatnonzero((values != -1) * ~np.isnan(values))
            print('Total: {}'.format(values.size))
            print('Valid: {}'.format(good_idx.size))
            filtered_values = values[good_idx]
            # print('{}: {:.2%}'.format(metric, np.mean(filtered_values)))
            print('{}: {:.4}'.format(metric, np.mean(filtered_values)))

        print('\n')

    return results, end - begin

def pck(source_points, warped_points, L_pck, alpha=0.1):
    # compute precentage of correct keypoints
    batch_size = source_points.size(0)
    pck = torch.zeros((batch_size))
    for i in range(batch_size):
        p_src = source_points[i, :]
        p_wrp = warped_points[i, :]
        # Compute the number of key points in source image
        N_pts = torch.sum(torch.ne(p_src[0, :], -1) * torch.ne(p_src[1, :], -1))
        point_distance = torch.pow(torch.sum(torch.pow(p_src[:, :N_pts] - p_wrp[:, :N_pts], 2), 0), 0.5)
        L_pck_mat = L_pck[i].expand_as(point_distance)
        correct_points = torch.le(point_distance, L_pck_mat * alpha)
        pck[i] = torch.mean(correct_points.float())
    return pck

def pck_metric(batch, batch_start_idx, theta_det, theta_aff, theta_tps, theta_afftps, results, args):
    alpha = args.pck_alpha
    do_det = theta_det is not None
    do_aff = theta_aff is not None
    do_tps = theta_tps is not None
    do_aff_tps = theta_afftps is not None

    source_im_size = batch['source_im_info'][:, 0:3]
    target_im_size = batch['target_im_info'][:, 0:3]

    source_points = batch['source_points']
    target_points = batch['target_points']

    # Instantiate point transformer
    pt = PointTPS(use_cuda=args.cuda, tps_reg_factor=args.tps_reg_factor)
    # pt = PointTPS(use_cuda=args.cuda)

    # warp points with estimated transformations
    target_points_norm = PointsToUnitCoords(P=target_points, im_size=target_im_size)

    if do_det:
        # Affine transformation only based on object detection
        warped_points_det_norm = pt.affPointTnf(theta=theta_det, points=target_points_norm)
        warped_points_det = PointsToPixelCoords(P=warped_points_det_norm, im_size=source_im_size)

    if do_aff:
        # do affine only
        warped_points_aff_norm = pt.affPointTnf(theta=theta_aff, points=target_points_norm)
        if do_det:
            warped_points_aff_norm = pt.affPointTnf(theta=theta_det, points=warped_points_aff_norm)
        warped_points_aff = PointsToPixelCoords(P=warped_points_aff_norm, im_size=source_im_size)

    if do_tps:
        # do tps only
        warped_points_tps_norm = pt.tpsPointTnf(theta=theta_tps, points=target_points_norm)
        warped_points_tps = PointsToPixelCoords(P=warped_points_tps_norm, im_size=source_im_size)

    if do_aff_tps:
        # do tps+affine
        warped_points_aff_tps_norm = pt.tpsPointTnf(theta=theta_afftps, points=target_points_norm)
        warped_points_aff_tps_norm = pt.affPointTnf(theta=theta_aff, points=warped_points_aff_tps_norm)
        if do_det:
            warped_points_aff_tps_norm = pt.affPointTnf(theta=theta_det, points=warped_points_aff_tps_norm)
        warped_points_aff_tps = PointsToPixelCoords(P=warped_points_aff_tps_norm, im_size=source_im_size)

    L_pck = batch['L_pck']

    current_batch_size = batch['source_im_info'].size(0)
    indices = range(batch_start_idx, batch_start_idx + current_batch_size)

    # import pdb; pdb.set_trace()
    if do_det:
        pck_det = pck(source_points, warped_points_det, L_pck, alpha)

    if do_aff:
        pck_aff = pck(source_points, warped_points_aff, L_pck, alpha)

    if do_tps:
        pck_tps = pck(source_points, warped_points_tps, L_pck, alpha)

    if do_aff_tps:
        pck_aff_tps = pck(source_points, warped_points_aff_tps, L_pck, alpha)

    if do_det:
        results['det']['pck'][indices] = pck_det.unsqueeze(1).cpu().numpy()
    if do_aff:
        if do_det:
            key = 'det_aff'
        else:
            key = 'aff'
        results[key]['pck'][indices] = pck_aff.unsqueeze(1).cpu().numpy()
    if do_tps:
        results['tps']['pck'][indices] = pck_tps.unsqueeze(1).cpu().numpy()
    if do_aff_tps:
        if do_det:
            key = 'det_aff_tps'
        else:
            key = 'afftps'
        results[key]['pck'][indices] = pck_aff_tps.unsqueeze(1).cpu().numpy()

    return results

def area_metrics(batch, batch_start_idx, theta_det, theta_aff, theta_tps, theta_afftps, results, args):
    do_det = theta_det is not None
    do_aff = theta_aff is not None
    do_tps = theta_tps is not None
    do_aff_tps = theta_afftps is not None

    batch_size = batch['source_im_info'].size(0)

    pt = PointTPS(use_cuda=args.cuda)

    for b in range(batch_size):
        # Get H, W of source and target image
        h_src = int(batch['source_im_info'][b, 0].cpu().numpy())
        w_src = int(batch['source_im_info'][b, 1].cpu().numpy())
        h_tgt = int(batch['target_im_info'][b, 0].cpu().numpy())
        w_tgt = int(batch['target_im_info'][b, 1].cpu().numpy())

        # Transform annotated polygon to mask using given coordinates of key points
        # target_mask_np.shape: (h_tgt, w_tgt), target_mask.shape: (1, 1, h_tgt, w_tgt)
        target_mask_np, target_mask = poly_str_to_mask(poly_x_str=batch['target_polygon'][0][b],
                                                       poly_y_str=batch['target_polygon'][1][b], out_h=h_tgt,
                                                       out_w=w_tgt, use_cuda=args.cuda)
        source_mask_np, source_mask = poly_str_to_mask(poly_x_str=batch['source_polygon'][0][b],
                                                       poly_y_str=batch['source_polygon'][1][b], out_h=h_src,
                                                       out_w=w_src, use_cuda=args.cuda)

        # Generate grid for warping
        grid_X, grid_Y = np.meshgrid(np.linspace(-1, 1, w_tgt), np.linspace(-1, 1, h_tgt))
        # grid_X, grid_Y.shape: (1, h_tgt, w_tgt, 1)
        grid_X = torch.Tensor(grid_X.astype(np.float32)).unsqueeze(0).unsqueeze(3)
        grid_Y = torch.Tensor(grid_Y.astype(np.float32)).unsqueeze(0).unsqueeze(3)
        grid_X.requires_grad = False
        grid_Y.requires_grad = False
        if args.cuda:
            grid_X = grid_X.cuda()
            grid_Y = grid_Y.cuda()
        # Reshape to vector, grid_X_vec, grid_Y_vec.shape: (1, 1, h_tgt * w_tgt)
        grid_X_vec = grid_X.view(1, 1, -1)
        grid_Y_vec = grid_Y.view(1, 1, -1)
        # grid_XY_vec.shape: (1, 2, h_tgt * w_tgt)
        grid_XY_vec = torch.cat((grid_X_vec, grid_Y_vec), 1)

        # Transform vector of points to grid
        def pointsToGrid(x, h_tgt=h_tgt, w_tgt=w_tgt):
            return x.contiguous().view(1, 2, h_tgt, w_tgt).permute(0, 2, 3, 1)

        idx = batch_start_idx + b

        if do_det:
            grid_det = pointsToGrid(pt.affPointTnf(theta=theta_det[b, :].unsqueeze(0), points=grid_XY_vec))
            warped_mask_det = F.grid_sample(source_mask, grid_det)
            flow_det = th_sampling_grid_to_np_flow(source_grid=grid_det, h_src=h_src, w_src=w_src)

            results['det']['intersection_over_union'][idx] = intersection_over_union(warped_mask=warped_mask_det, target_mask=target_mask)
            results['det']['label_transfer_accuracy'][idx] = label_transfer_accuracy(warped_mask=warped_mask_det, target_mask=target_mask)
            results['det']['localization_error'][idx] = localization_error(source_mask_np=source_mask_np, target_mask_np=target_mask_np, flow_np=flow_det)

        if do_aff:
            if do_det:
                key = 'det_aff'
                grid_aff = pointsToGrid(pt.affPointTnf(theta=theta_det[b, :].unsqueeze(0),
                                                       points=pt.affPointTnf(theta=theta_aff[b, :].unsqueeze(0),
                                                                             points=grid_XY_vec)))
            else:
                key = 'aff'
                grid_aff = pointsToGrid(pt.affPointTnf(theta=theta_aff[b, :].unsqueeze(0), points=grid_XY_vec))
            warped_mask_aff = F.grid_sample(source_mask, grid_aff)
            flow_aff = th_sampling_grid_to_np_flow(source_grid=grid_aff, h_src=h_src, w_src=w_src)

            results[key]['intersection_over_union'][idx] = intersection_over_union(warped_mask=warped_mask_aff, target_mask=target_mask)
            results[key]['label_transfer_accuracy'][idx] = label_transfer_accuracy(warped_mask=warped_mask_aff, target_mask=target_mask)
            results[key]['localization_error'][idx] = localization_error(source_mask_np=source_mask_np, target_mask_np=target_mask_np, flow_np=flow_aff)

        if do_tps:
            # Get sampling grid with predicted TPS parameters, grid_tps.shape: (1, h_tgt, w_tgt, 2)
            grid_tps = pointsToGrid(pt.tpsPointTnf(theta=theta_tps[b, :].unsqueeze(0), points=grid_XY_vec))
            warped_mask_tps = F.grid_sample(source_mask, grid_tps)  # Sampling source_mask with warped grid
            # Transform sampling grid to flow
            flow_tps = th_sampling_grid_to_np_flow(source_grid=grid_tps, h_src=h_src, w_src=w_src)

            results['tps']['intersection_over_union'][idx] = intersection_over_union(warped_mask=warped_mask_tps, target_mask=target_mask)
            results['tps']['label_transfer_accuracy'][idx] = label_transfer_accuracy(warped_mask=warped_mask_tps, target_mask=target_mask)
            results['tps']['localization_error'][idx] = localization_error(source_mask_np=source_mask_np, target_mask_np=target_mask_np, flow_np=flow_tps)

        if do_aff_tps:
            if do_det:
                key = 'det_aff_tps'
                grid_aff_tps = pointsToGrid(pt.affPointTnf(theta=theta_det[b, :].unsqueeze(0),
                                                           points=pt.affPointTnf(theta=theta_aff[b,:].unsqueeze(0),
                                                                                 points=pt.tpsPointTnf(theta=theta_afftps[b,:].unsqueeze(0),
                                                                                                       points=grid_XY_vec))))
            else:
                key = 'afftps'
                grid_aff_tps = pointsToGrid(pt.affPointTnf(theta=theta_aff[b, :].unsqueeze(0), points=pt.tpsPointTnf(theta=theta_afftps[b, :].unsqueeze(0), points=grid_XY_vec)))
            warped_mask_aff_tps = F.grid_sample(source_mask, grid_aff_tps)
            flow_aff_tps = th_sampling_grid_to_np_flow(source_grid=grid_aff_tps, h_src=h_src, w_src=w_src)

            results[key]['intersection_over_union'][idx] = intersection_over_union(warped_mask=warped_mask_aff_tps, target_mask=target_mask)
            results[key]['label_transfer_accuracy'][idx] = label_transfer_accuracy(warped_mask=warped_mask_aff_tps, target_mask=target_mask)
            results[key]['localization_error'][idx] = localization_error(source_mask_np=source_mask_np, target_mask_np=target_mask_np, flow_np=flow_aff_tps)

    return results

def flow_metrics(batch, batch_start_idx, theta_det, theta_aff, theta_tps, theta_afftps, results, args):
    result_path = args.flow_output_dir

    do_det = theta_det is not None
    do_aff = theta_aff is not None
    do_tps = theta_tps is not None
    do_aff_tps = theta_afftps is not None

    pt = PointTPS(use_cuda=args.cuda)

    batch_size = batch['source_im_info'].size(0)
    for b in range(batch_size):
        # Get H, W of source and target image
        h_src = int(batch['source_im_info'][b, 0].cpu().numpy())
        w_src = int(batch['source_im_info'][b, 1].cpu().numpy())
        h_tgt = int(batch['target_im_info'][b, 0].cpu().numpy())
        w_tgt = int(batch['target_im_info'][b, 1].cpu().numpy())

        # Generate grid for warping
        grid_X, grid_Y = np.meshgrid(np.linspace(-1, 1, w_tgt), np.linspace(-1, 1, h_tgt))
        # grid_X, grid_Y.shape: (1, h_tgt, w_tgt, 1)
        grid_X = torch.Tensor(grid_X.astype(np.float32)).unsqueeze(0).unsqueeze(3)
        grid_Y = torch.Tensor(grid_Y.astype(np.float32)).unsqueeze(0).unsqueeze(3)
        grid_X.requires_grad = False
        grid_Y.requires_grad = False
        if args.cuda:
            grid_X = grid_X.cuda()
            grid_Y = grid_Y.cuda()
        # Reshape to vector, grid_X_vec, grid_Y_vec.shape: (1, 1, h_tgt * w_tgt)
        grid_X_vec = grid_X.view(1, 1, -1)
        grid_Y_vec = grid_Y.view(1, 1, -1)
        # grid_XY_vec.shape: (1, 2, h_tgt * w_tgt)
        grid_XY_vec = torch.cat((grid_X_vec, grid_Y_vec), 1)

        # Transform vector of points to grid
        def pointsToGrid(x, h_tgt=h_tgt, w_tgt=w_tgt):
            return x.contiguous().view(1, 2, h_tgt, w_tgt).permute(0, 2, 3, 1)

        idx = batch_start_idx + b

        if do_det:
            grid_det = pointsToGrid(pt.affPointTnf(theta=theta_det[b,:].unsqueeze(0), points=grid_XY_vec))
            flow_det = th_sampling_grid_to_np_flow(source_grid=grid_det, h_src=h_src, w_src=w_src)
            flow_det_path = os.path.join(result_path, 'det', batch['flow_path'][b])
            create_file_path(flow_det_path)
            write_flo_file(flow_det, flow_det_path)

        if do_aff:
            if do_det:
                key = 'det_aff'
                grid_aff = pointsToGrid(pt.affPointTnf(theta=theta_det[b, :].unsqueeze(0),
                                                       points=pt.affPointTnf(theta=theta_aff[b, :].unsqueeze(0), points=grid_XY_vec)))
            else:
                key = 'aff'
                grid_aff = pointsToGrid(pt.affPointTnf(theta=theta_aff[b, :].unsqueeze(0), points=grid_XY_vec))
            flow_aff = th_sampling_grid_to_np_flow(source_grid=grid_aff, h_src=h_src, w_src=w_src)
            flow_aff_path = os.path.join(result_path, key, batch['flow_path'][b])
            create_file_path(flow_aff_path)
            write_flo_file(flow_aff, flow_aff_path)

        if do_tps:
            # Get sampling grid with predicted TPS parameters, grid_tps.shape: (1, h_tgt, w_tgt, 2)
            grid_tps = pointsToGrid(pt.tpsPointTnf(theta=theta_tps[b, :].unsqueeze(0), points=grid_XY_vec))
            # Transform sampling grid to flow
            flow_tps = th_sampling_grid_to_np_flow(source_grid=grid_tps, h_src=h_src, w_src=w_src)
            flow_tps_path = os.path.join(result_path, 'tps', batch['flow_path'][b])
            create_file_path(flow_tps_path)
            write_flo_file(flow_tps, flow_tps_path)

        if do_aff_tps:
            if do_det:
                key = 'det_aff_tps'
                grid_aff_tps = pointsToGrid(pt.affPointTnf(theta=theta_det[b,:].unsqueeze(0),
                                                           points=pt.affPointTnf(theta=theta_aff[b,:].unsqueeze(0),
                                                                                 points=pt.tpsPointTnf(theta=theta_afftps[b,:].unsqueeze(0),
                                                                                                       points=grid_XY_vec))))
            else:
                key = 'afftps'
                grid_aff_tps = pointsToGrid(pt.affPointTnf(theta=theta_aff[b, :].unsqueeze(0), points=pt.tpsPointTnf(theta=theta_afftps[b, :].unsqueeze(0), points=grid_XY_vec)))
            flow_aff_tps = th_sampling_grid_to_np_flow(source_grid=grid_aff_tps,h_src=h_src,w_src=w_src)
            flow_aff_tps_path = os.path.join(result_path, key, batch['flow_path'][b])
            create_file_path(flow_aff_tps_path)
            write_flo_file(flow_aff_tps, flow_aff_tps_path)

        idx = batch_start_idx+b

    return results

def poly_to_mask(vertex_row_coords, vertex_col_coords, shape):
    """ Transform annotated polygon to mask using given coordinates of key points """
    # Get coordinates of pixels within polygon
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    # Use coordinates of pixels within polygon to generate mask
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    # plt.imshow(mask)
    # plt.show()
    return mask

def poly_str_to_mask(poly_x_str, poly_y_str, out_h, out_w, use_cuda=True):
    """ Generate mask using given coordinates of key points on polygon """
    polygon_x = np.fromstring(poly_x_str, sep=',')
    polygon_y = np.fromstring(poly_y_str, sep=',')
    # mask_np.shape: (out_h, out_w)
    mask_np = poly_to_mask(vertex_col_coords=polygon_x, vertex_row_coords=polygon_y, shape=[out_h, out_w])
    # mask = Variable(torch.FloatTensor(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0))
    # mask.shape: (1, 1, out_h, out_w)
    mask = torch.Tensor(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    if use_cuda:
        mask = mask.cuda()
    return mask_np, mask

def intersection_over_union(warped_mask, target_mask):
    # relative_part_weight = torch.sum(torch.sum(target_mask.data.gt(0.5).float(), 2, True), 3, True) / torch.sum(target_mask.data.gt(0.5).float())
    # part_iou = torch.sum(torch.sum((warped_mask.data.gt(0.5) & target_mask.data.gt(0.5)).float(), 2, True), 3, True) / torch.sum(torch.sum((warped_mask.data.gt(0.5) | target_mask.data.gt(0.5)).float(), 2, True), 3, True)
    relative_part_weight = torch.sum(torch.sum(target_mask.gt(0.5).float(), 2, True), 3, True) / torch.sum(target_mask.gt(0.5).float())
    part_iou = torch.sum(torch.sum((warped_mask.gt(0.5) & target_mask.gt(0.5)).float(), 2, True), 3, True) / torch.sum(torch.sum((warped_mask.gt(0.5) | target_mask.gt(0.5)).float(), 2, True), 3, True)
    weighted_iou = torch.sum(torch.mul(relative_part_weight, part_iou)).item()
    return weighted_iou

def label_transfer_accuracy(warped_mask, target_mask):
    # return torch.mean((warped_mask.data.gt(0.5) == target_mask.data.gt(0.5)).double()).item()
    return torch.mean((warped_mask.gt(0.5) == target_mask.gt(0.5)).double()).item()

def localization_error(source_mask_np, target_mask_np, flow_np):
    # target_mask_np.shape: (h_tgt, w_tgt)
    h_tgt, w_tgt = target_mask_np.shape[0], target_mask_np.shape[1]
    h_src, w_src = source_mask_np.shape[0], source_mask_np.shape[1]

    # initial pixel positions x1,y1 in target image
    x1, y1 = np.meshgrid(range(1, w_tgt + 1), range(1, h_tgt + 1))
    # sampling pixel positions x2,y2
    x2 = x1 + flow_np[:, :, 0]
    y2 = y1 + flow_np[:, :, 1]

    # compute in-bound coords for each image
    in_bound = (x2 >= 1) & (x2 <= w_src) & (y2 >= 1) & (y2 <= h_src)
    row, col = np.where(in_bound)
    # Coordinates of in-bound in target image (warp)
    row_1 = y1[row, col].flatten().astype(np.int) - 1
    col_1 = x1[row, col].flatten().astype(np.int) - 1
    # Coordinates of in-bound in source image
    row_2 = y2[row, col].flatten().astype(np.int) - 1
    col_2 = x2[row, col].flatten().astype(np.int) - 1

    # compute relative positions based on objects
    target_loc_x, target_loc_y = obj_ptr(target_mask_np)
    source_loc_x, source_loc_y = obj_ptr(source_mask_np)
    # Relative positions after warping
    x1_rel = target_loc_x[row_1, col_1]
    y1_rel = target_loc_y[row_1, col_1]
    # Relative positions in source image
    x2_rel = source_loc_x[row_2, col_2]
    y2_rel = source_loc_y[row_2, col_2]

    # Compute localization error based on differences of relative positions between source image and after warping
    loc_err = np.mean(np.abs(x1_rel - x2_rel) + np.abs(y1_rel - y2_rel))

    return loc_err

def obj_ptr(mask):
    # computes images of normalized coordinates around bounding box
    # kept function name from DSP code
    h, w = mask.shape[0], mask.shape[1]
    y, x = np.where(mask > 0.5) # Get coordinates of foreground object
    # Get bounding box of object
    left = np.min(x)
    right = np.max(x)
    top = np.min(y)
    bottom = np.max(y)
    fg_width = right - left + 1
    fg_height = bottom - top + 1
    # Only position of foreground object have values [0, 1]
    x_image, y_image = np.meshgrid(range(1, w + 1), range(1, h + 1))
    x_image = (x_image - left) / fg_width
    y_image = (y_image - top) / fg_height
    return x_image, y_image
