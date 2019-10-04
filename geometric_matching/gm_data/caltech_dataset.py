# ==============================================================================================================
# Generate a testing dataset from Caltech-101 for geometric matching model
# Author: Ignacio Rocco
# Modification: Jingwei Qu
# Date: 01 June 2019
# ==============================================================================================================

import os
import torch
from skimage import io
import cv2
import pandas as pd
import numpy as np
from geometric_matching.gm_data.test_dataset import TestDataset

class CaltechDataset(TestDataset):
    """
    Caltech-101 image pair dataset

    Args:
            csv_file (string): Path to the csv file with image names and transformations
            dataset_path (string): Directory with all the images
            output_size (2-tuple): Desired output size
            transform (callable): Transformation for post-processing the training pair (eg. image normalization)

    Returns:
        Dict: {'source_image' & 'target_image': images for transformation,
               'source_im_size' & 'target_im_size': size of images
               'source_points' & 'target_points': coordinates of key points in images,
               'L_pck': PCK reference length
               'source_im' & 'target_im': images for object detection,
               'source_im_info' & 'target_im_info': image size & image scale ratio,
               'source_gt_boxes' & 'target_gt_boxes': coordinates of ground-truth bounding boxes, set to [1, 1, 1, 1, 1]
               'source_num_boxes' & 'target_num_boxes': number of ground-truth bounding boxes, set to 0}
    """

    def __init__(self, csv_file, dataset_path, output_size=(240, 240), normalize=None):
        super(CaltechDataset, self).__init__(csv_file=csv_file, dataset_path=dataset_path, output_size=output_size, normalize=normalize)
        self.img_A_names = self.dataframe.iloc[:, 0]  # Get source image & target image name
        self.img_B_names = self.dataframe.iloc[:, 1]
        self.categories = self.dataframe.iloc[:, 2].values.astype('float')  # Get image category
        self.annot_A_str = self.dataframe.iloc[:, 3:5]
        self.annot_B_str = self.dataframe.iloc[:, 5:]
        self.category_names = ['Faces', 'Faces_easy', 'Leopards', 'Motorbikes', 'accordion', 'airplanes', 'anchor',
                               'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus',
                               'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone',
                               'chair', 'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile',
                               'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly',
                               'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo',
                               'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill',
                               'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo',
                               'ketch', 'lamp', 'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah',
                               'metronome', 'minaret', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon',
                               'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner',
                               'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish',
                               'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella',
                               'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']

    def __getitem__(self, idx):
        # get pre-processed images
        # image_A, im_A, im_info_A, gt_boxes_A, num_boxes_A = self.get_image(img_name_list=self.img_A_names, idx=idx)
        # image_B, im_B, im_info_B, gt_boxes_B, num_boxes_B = self.get_image(img_name_list=self.img_B_names, idx=idx)
        image_A, im_info_A = self.get_image(img_name_list=self.img_A_names, idx=idx)
        image_B, im_info_B = self.get_image(img_name_list=self.img_B_names, idx=idx)

        # get pre-processed point coords
        annot_A = self.get_points(point_coords_list=self.annot_A_str, idx=idx)
        annot_B = self.get_points(point_coords_list=self.annot_B_str, idx=idx)

        # sample = {'source_image': image_A, 'target_image': image_B,
        #           'source_im': im_A, 'target_im': im_B,
        #           'source_im_info': im_info_A, 'target_im_info': im_info_B,
        #           'source_gt_boxes': gt_boxes_A, 'target_gt_boxes': gt_boxes_B,
        #           'source_num_boxes': num_boxes_A, 'target_num_boxes': num_boxes_B,
        #           'source_polygon': annot_A, 'target_polygon': annot_B}

        sample = {'source_image': image_A, 'target_image': image_B,
                  'source_im_info': im_info_A, 'target_im_info': im_info_B,
                  'source_polygon': annot_A, 'target_polygon': annot_B}

        sample = self.normalize(sample)

        return sample

    def get_points(self, point_coords_list, idx):
        point_coords_x = point_coords_list[point_coords_list.columns[0]][idx]
        point_coords_y = point_coords_list[point_coords_list.columns[1]][idx]

        return (point_coords_x, point_coords_y)