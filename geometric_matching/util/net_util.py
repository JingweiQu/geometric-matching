# ========================================================================================
# Functions used for geometric matching model
# Author: Jingwei Qu
# Date: 05 Mar 2019
# ========================================================================================

import shutil
import torch
from torch.autograd import Variable
import os
from os import makedirs, remove
from os.path import exists, join, basename, dirname
import errno
import argparse
import collections
import numpy as np
import xml.etree.ElementTree as ET

from lib.model.utils.blob import prep_im_for_blob_2
from lib.model.rpn.bbox_transform import *
from lib.model.roi_layers import nms

from geometric_matching.image.normalization import normalize_image
from geometric_matching.util.dataloader import default_collate

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

class BatchTensorToVars(object):
    """ Convert tensors in dict batch to vars """
    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda

    def __call__(self, batch):
        batch_var = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and not self.use_cuda:
                batch_var[key] = Variable(value, requires_grad=False)
            elif isinstance(value, torch.Tensor) and self.use_cuda:
                batch_var[key] = Variable(value, requires_grad=False).cuda()
            else:
                batch_var[key] = value
        return batch_var

def collate_custom(batch):
    """ Custom collate function for the Dataset class
     * It doesn't convert numpy arrays to stacked-tensors, but rather combines them in a list
     * This is useful for processing annotations of different sizes
    """
    # this case will occur in first pass, and will convert a
    # list of dictionaries (returned by the threads by sampling dataset[idx])
    # to a unified dictionary of collated values
    if isinstance(batch[0], collections.Mapping):
        return {key: collate_custom([d[key] for d in batch]) for key in batch[0]}
    # these cases will occur in recursion
    elif torch.is_tensor(batch[0]): # for tensors, use standrard collating function
        return default_collate(batch)
    else: # for other types (i.e. lists), return as is
        return batch

def create_file_path(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def Softmax1D(x,dim):
    x_k = torch.max(x,dim)[0].unsqueeze(dim)
    x -= x_k.expand_as(x)
    exp_x = torch.exp(x)
    return torch.div(exp_x,torch.sum(exp_x,dim).unsqueeze(dim).expand_as(x))

# def save_checkpoint(state, is_best, file):
def save_checkpoint(state, file):
    """ Save checkpoint of the trained model """
    model_dir = dirname(file)
    model_fn = basename(file)
    # make dir if needed (should be non-empty)
    if model_dir != '' and not exists(model_dir):
        makedirs(model_dir)
    torch.save(state, file)
    # Select the best model, and copy
    # if is_best:
    #     shutil.copyfile(file, join(model_dir, 'best_' + model_fn))

def str_to_bool(v):
    """ Transform string to bool """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def expand_dim(tensor,dim,desired_dim_len):
    sz = list(tensor.size())
    sz[dim]=desired_dim_len
    return tensor.expand(tuple(sz))

def roi_data(image):
    """ Prepare the input of faster rcnn for detecting objects """
    # flip the channel, since the original one using cv2
    # rgb -> bgr
    image = image[:, :, ::-1]

    # Pixel mean values (BGR order) as a (1, 1, 3) array
    # We use the same pixel mean for all networks even though it's not exactly what
    # they were trained with
    pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
    image, im_scale = prep_im_for_blob_2(image, pixel_means, target_size=240)
    im_info = np.array([image.shape[0], image.shape[1], im_scale], dtype=np.float32)

    # numpy to tensor
    image = torch.Tensor(image)
    image = image.permute(2, 0, 1)
    im_info = torch.Tensor(im_info)
    gt_boxes = torch.Tensor([1, 1, 1, 1, 1])
    num_boxes = 0

    return image, im_info, gt_boxes, num_boxes


def select_boxes(rois, cls_prob, bbox_pred, im_infos, thresh=0.05, max_per_image=50):
    """ Select bounding boxes of objects from the predicted results of faster rcnn """
    n_classes = cls_prob.shape[2]
    all_boxes = []
    for i in range(rois.shape[0]):
        boxes = rois[i, :, 1:5].view(1, -1, 4)
        scores = cls_prob[i, :, :].view(1, -1, n_classes)
        box_deltas = bbox_pred[i, :, :].view(1, -1, 4 * n_classes)
        im_info = im_infos[i, :].view(1, 3)
        # Normalize boxes deltas by a mean and std
        bbox_normalize_means = (0.0, 0.0, 0.0, 0.0)
        bbox_normalize_stds = (0.1, 0.1, 0.2, 0.2)
        box_deltas = box_deltas.view(-1, 4) * torch.Tensor(bbox_normalize_stds).cuda() \
                     + torch.Tensor(bbox_normalize_means).cuda()
        # 21 is the number of classed in pascal voc datasets
        box_deltas = box_deltas.view(1, -1, 4 * n_classes)

        # Compute predicted boxes by predicted rois and the corresponding box deltas
        # Clip borders of predicted boxes if they cross the border of the resized image
        pred_boxes = bbox_transform_inv(boxes, box_deltas, batch_size=1)
        padding = 5
        pred_boxes[:, :, 0::4] -= padding
        pred_boxes[:, :, 1::4] -= padding
        pred_boxes[:, :, 2::4] += padding
        pred_boxes[:, :, 3::4] += padding
        pred_boxes = clip_boxes(pred_boxes, im_info, batch_size=1)

        # pred_boxes.shape: (300, 4 * n_classes)
        # scores.shape: (300, n_classes)
        pred_boxes = pred_boxes.squeeze()
        scores = scores.squeeze()

        all_box = []

        for j in range(1, 21):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, dim=0, descending=True)
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                # Concatenate boxes coordinates and class scores
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]

                # Non-maximum suppression (suppress boxes with IoU >= 0.3) for selecting predicted boxes for each object class in the image
                keep = nms(cls_boxes[order, :], cls_scores[order], 0.3)
                cls_dets = cls_dets[keep.view(-1).long()]

                # Concatenate the class ids of boxes
                class_id = torch.ones(cls_dets.shape[0], 1, dtype=torch.float).cuda() * j
                cls_dets = torch.cat((cls_dets, class_id), dim=1)
                # Add each box
                for i in range(cls_dets.shape[0]):
                    all_box.append(cls_dets[i, :].cpu().detach().numpy())

        all_box = np.array(all_box)
        if all_box.shape[0] != 0:
            # all_box = np.concatenate((all_box, all_box), axis=0)
            # Rank all boxes based on scores in the descending order
            index = all_box[:, 4].argsort()[::-1]
            all_box = all_box[index, :]
            # Limit to max_per_image detections *over all classes*
            if all_box.shape[0] > max_per_image:
                all_box = all_box[:max_per_image, :]

            # all_box.shape: (num_boxes, 6), 6: (x_min, y_min, x_max, y_max, score, class_id)
            # print(all_box)
            # print(all_box.shape[0])
            # print('\n')
        all_boxes.append(all_box)

    return all_boxes

def select_box(all_boxes_s, all_boxes_t):
    """ Select the object bounding box from the selected bounding boxes """
    boxes_s = torch.Tensor(len(all_boxes_s), 4).zero_() - 1
    boxes_t = torch.Tensor(len(all_boxes_t), 4).zero_() - 1
    for j in range(len(all_boxes_s)):
        box_s = np.ones(4, dtype=np.float) * -1
        box_t = np.ones(4, dtype=np.float) * -1
        all_box_s = all_boxes_s[j]
        all_box_t = all_boxes_t[j]
        if all_box_s.shape[0] != 0 and all_box_t.shape[0] != 0:
            class_s = all_box_s[:, 5].astype(np.int32)
            class_ids = all_box_t[:, 5].astype(np.int32)
            for i in range(class_s.shape[0]):
                keep = np.where(class_ids == class_s[i])[0]
                if keep.size != 0:
                    box_s = all_box_s[i, :4]
                    box_t = all_box_t[keep[0], :4]
                    break

        boxes_s[j, :] = torch.Tensor(box_s)
        boxes_t[j, :] = torch.Tensor(box_t)

    return boxes_s, boxes_t

'''
def select_box(all_boxes_s, all_boxes_t):
    """ Select the object bounding box from the selected bounding boxes """
    boxes_s = torch.Tensor(len(all_boxes_s), 4).zero_() - 1
    boxes_t = torch.Tensor(len(all_boxes_t), 4).zero_() - 1
    for j in range(len(all_boxes_s)):
        box_s = np.ones(4, dtype=np.float) * -1
        box_t = np.ones(4, dtype=np.float) * -1
        all_box_s = all_boxes_s[j]
        all_box_t = all_boxes_t[j]
        if all_box_s.shape[0] != 0 and all_box_t.shape[0] != 0:
            areas_s = (all_box_s[:, 2] - all_box_s[:, 0]) * (all_box_s[:, 3] - all_box_s[:, 1])
            max_box_idx = np.where(areas_s == np.max(areas_s))[0][0]
            class_s = all_box_s[max_box_idx, 5].astype(np.int32)
            class_ids = all_box_t[:, 5].astype(np.int32)
            keep = np.where(class_ids == class_s)[0]
            if keep.size != 0:
                box_s = all_box_s[max_box_idx, :4]
                box_t = all_box_t[keep[0], :4]

        boxes_s[j, :] = torch.Tensor(box_s)
        boxes_t[j, :] = torch.Tensor(box_t)

    return boxes_s, boxes_t
'''


def select_box_single(all_boxes):
    """ Select the object bounding box with the highest score """
    boxes = torch.ones(len(all_boxes), 4, dtype=torch.float) * -1
    for j in range(len(all_boxes)):
        box = np.ones(4, dtype=np.float) * -1
        all_box = all_boxes[j]
        if all_box.shape[0] != 0:
            box = all_box[0, :4]

        boxes[j, :] = torch.Tensor(box)

    return boxes


def numpy_image(image):
    """ Transform image tensor to image numpy """
    image = image.permute(1, 2, 0).cpu().numpy()
    mean = np.array([102.9801, 115.9465, 122.7717])
    image += mean
    image = image[:, :, ::-1]
    image = image / 255

    return image


def im_show_1(image, title, rows, cols, index):
    """ Show image (transfer tensor to numpy first) """
    # image = image.permute(1, 2, 0).cpu().numpy()
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # image = std * image + mean
    image = normalize_image(image, forward=False)
    image = image.permute(1, 2, 0).cpu().numpy()
    ax = plt.subplot(rows, cols, index)
    ax.set_title(title)
    ax.imshow(image.clip(0, 1))

    return ax


def im_show_2(image, title, rows, cols, index):
    """ Show image (transfer tensor to numpy first) """
    image = numpy_image(image)
    ax = plt.subplot(rows, cols, index)
    ax.set_title(title)
    ax.imshow(image.clip(0, 1))

    return ax


def show_boxes(ax, boxes):
    """ Show bounding boxes on the image """
    # Object classes in PascalVOC
    dataset_classes = np.asarray(['__background__',
                                  'aeroplane', 'bicycle', 'bird', 'boat',
                                  'bottle', 'bus', 'car', 'cat', 'chair',
                                  'cow', 'diningtable', 'dog', 'horse',
                                  'motorbike', 'person', 'pottedplant',
                                  'sheep', 'sofa', 'train', 'tvmonitor'])

    if boxes.shape[0] > 0:
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0

        if boxes.shape[1] == 6:
            for i in range(boxes.shape[0]):
                rect = plt.Rectangle((boxes[i, 0], boxes[i, 1]), widths[i], heights[i], fill=False, edgecolor='r', linewidth=2)
                ax.add_patch(rect)
                plt.text(boxes[i, 0], boxes[i, 1], dataset_classes[int(boxes[i, 5])] + ' ' + '%.3f' % boxes[i, 4], fontsize=16, color='y')
                # break
        elif boxes.shape[1] == 4:
            rect = plt.Rectangle((boxes[:, 0], boxes[:, 1]), widths, heights, fill=False, edgecolor='r', linewidth=2)
            ax.add_patch(rect)
    else:
        print('No object is detected')


def correct_keypoints(source_points, warped_points, L_pck, alpha=0.1):
    """ Compute PCK """
    # Compute correct key points
    point_distance = torch.pow(torch.sum(torch.pow(source_points-warped_points, 2), 1), 0.5).squeeze(1)
    L_pck_mat = L_pck.expand_as(point_distance)
    correct_points = torch.le(point_distance, L_pck_mat*alpha)
    num_of_correct_points = torch.sum(correct_points)
    num_of_points = correct_points.numel()
    return num_of_correct_points.item(), num_of_points


def parse_xml(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects