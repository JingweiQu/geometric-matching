import torch
import numpy as np
import os
import time
from geometric_matching.util.net_util import *

def test_fn(model=None, batch_size=1, dataset=None, dataloader=None, args=None):
    # Initialize results
    N = len(dataset)
    # initialize vector for storing results for each metric
    results = np.zeros((N, 2))

    # Compute
    begin = time.time()
    for batch_idx, batch in enumerate(dataloader):
        if args.cuda:
            batch = batch_cuda(batch)
        batch_start_idx = batch_size * batch_idx

        mask_A, mask_B = model(batch)

        results = area_metrics(batch, batch_start_idx, mask_A, mask_B, results)

        end = time.time()
        print('Batch: [{}/{} ({:.0%})]\t\tTime cost ({} batches): {:.4f} s'.format(batch_idx+1, len(dataloader), (batch_idx+1) / len(dataloader), batch_idx + 1, end - begin))
    end = time.time()
    print('Dataset time cost: {:.4f} s'.format(end - begin))

    # Print results
    print('=== Results IoU ===')

    # print mean value
    values = results
    values = np.reshape(values, (-1))
    good_idx = np.flatnonzero((values != -1) * ~np.isnan(values))
    print('Total: {}'.format(values.size))
    print('Valid: {}'.format(good_idx.size))
    filtered_values = values[good_idx]
    print('IoU: {:.4}'.format(np.mean(filtered_values)))
    print('\n')

    return results, end - begin

def area_metrics(batch, batch_start_idx, mask_A, mask_B, results):
    batch_size = batch['source_image'].size(0)

    for b in range(batch_size):
        # Transform annotated polygon to mask using given coordinates of key points
        # target_mask.shape: (1, 1, 240, 240)
        source_mask = batch['source_mask'][b]
        target_mask = batch['target_mask'][b]

        idx = batch_start_idx + b

        results[idx, 0] = intersection_over_union(mask=mask_A[b].unsqueeze(0), mask_gt=source_mask.unsqueeze(0))
        results[idx, 1] = intersection_over_union(mask=mask_B[b].unsqueeze(0), mask_gt=target_mask.unsqueeze(0))

    return results

def intersection_over_union(mask, mask_gt):
    # relative_part_weight = torch.sum(torch.sum(target_mask.data.gt(0.5).float(), 2, True), 3, True) / torch.sum(target_mask.data.gt(0.5).float())
    # part_iou = torch.sum(torch.sum((warped_mask.data.gt(0.5) & target_mask.data.gt(0.5)).float(), 2, True), 3, True) / torch.sum(torch.sum((warped_mask.data.gt(0.5) | target_mask.data.gt(0.5)).float(), 2, True), 3, True)
    relative_part_weight = torch.sum(torch.sum(mask_gt.gt(0.5).float(), 2, True), 3, True) / torch.sum(mask_gt.gt(0.5).float())
    part_iou = torch.sum(torch.sum((mask.gt(0.5) & mask_gt.gt(0.5)).float(), 2, True), 3, True) / torch.sum(torch.sum((mask.gt(0.5) | mask_gt.gt(0.5)).float(), 2, True), 3, True)
    weighted_iou = torch.sum(torch.mul(relative_part_weight, part_iou)).item()
    return weighted_iou