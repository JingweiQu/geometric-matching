# ========================================================================================
# 1. Generate image pairs (objects) of PF-WILLOW dataset
# 2. Re-locate key points on the objects,
# 3. Rescale the objects as input image size (240, 240)
# Author: Jingwei Qu
# Date: 05 Mar 2019
# ========================================================================================

import torch
import numpy as np


def pf_willow_pair(batch, boxes_s, boxes_t, rescalingTnf):
    '''
    Generate object pairs of PF-WILLOW dataset, re-compute key points on the objects, and rescale image size as object size
    '''

    # '''
    for i in range(boxes_s.shape[0]):
        # Crop if object is detected
        if torch.sum(boxes_s[i]).item() > 0 and torch.sum(boxes_t[i]).item() > 0:
            crop_object_A = batch['source_image'][i, :, int(boxes_s[i, 1]):int(boxes_s[i, 3]),
                          int(boxes_s[i, 0]):int(boxes_s[i, 2])].unsqueeze(0)
            batch['source_image'][i, :, :, :] = rescalingTnf(crop_object_A).squeeze()

            crop_object_B = batch['target_image'][i, :, int(boxes_t[i, 1]):int(boxes_t[i, 3]),
                            int(boxes_t[i, 0]):int(boxes_t[i, 2])].unsqueeze(0)
            batch['target_image'][i, :, :, :] = rescalingTnf(crop_object_B).squeeze()
    # '''

    # '''
    boxes_s_o = boxes_s.clone()
    boxes_t_o = boxes_t.clone()
    source_points = batch['source_points'].clone()
    target_points = batch['target_points'].clone()
    source_im_sizes = batch['source_im_size'].clone()
    target_im_sizes = batch['target_im_size'].clone()

    for i in range(boxes_s_o.shape[0]):
        if torch.sum(boxes_s_o[i]).item() > 0 and torch.sum(boxes_t_o[i]).item() > 0:
            # Locate object box on the original image
            boxes_s_o[i, 0::2] *= (batch['source_im_size'][i, 1] / 240)
            boxes_s_o[i, 1::2] *= (batch['source_im_size'][i, 0] / 240)
            # Locate key points on the object (in the original image)
            source_points[i, 0, :] -= boxes_s_o[i, 0]
            source_points[i, 1, :] -= boxes_s_o[i, 1]
            # Compute the original size of the object
            source_im_sizes[i, 1] = int(boxes_s_o[i, 2]) - int(boxes_s_o[i, 0])
            source_im_sizes[i, 0] = int(boxes_s_o[i, 3]) - int(boxes_s_o[i, 1])

            boxes_t_o[i, 0::2] *= (batch['target_im_size'][i, 1] / 240)
            boxes_t_o[i, 1::2] *= (batch['target_im_size'][i, 0] / 240)
            target_points[i, 0, :] -= boxes_t_o[i, 0]
            target_points[i, 1, :] -= boxes_t_o[i, 1]
            target_im_sizes[i, 1] = int(boxes_t_o[i, 2]) - int(boxes_t_o[i, 0])
            target_im_sizes[i, 0] = int(boxes_t_o[i, 3]) - int(boxes_t_o[i, 1])

    batch['source_points'] = source_points
    batch['target_points'] = target_points
    batch['source_im_size'] = source_im_sizes
    batch['target_im_size'] = target_im_sizes
    # '''

    return batch