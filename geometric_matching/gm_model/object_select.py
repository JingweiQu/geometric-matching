import torch
import torch.nn as nn

from geometric_matching.util.net_util import select_boxes, select_box

class ObjectSelect(nn.Module):
    def __init__(self, thresh=0.05, max_per_image=50):
        super().__init__()
        self.thresh = thresh
        self.max_per_image = max_per_image

    def forward(self, box_info_A, im_info_A, box_info_B, im_info_B):
        all_box_A = select_boxes(rois=box_info_A[0], cls_prob=box_info_A[1], bbox_pred=box_info_A[2],
                                   im_infos=im_info_A, thresh=self.thresh, max_per_image=self.max_per_image)
        all_box_B = select_boxes(rois=box_info_B[0], cls_prob=box_info_B[1], bbox_pred=box_info_B[2],
                                   im_infos=im_info_B, thresh=self.thresh, max_per_image=self.max_per_image)
        box_A, box_B = select_box(all_boxes_s=all_box_A, all_boxes_t=all_box_B)
        box_A.requires_grad = False
        box_B.requires_grad = False

        return box_A, box_B


