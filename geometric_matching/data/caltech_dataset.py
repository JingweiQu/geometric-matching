# ==============================================================================================================
# Generate a testing dataset from Caltech-101 for geometric matching model
# Author: Ignacio Rocco
# Modification: Jingwei Qu
# Date: 01 June 2019
# ==============================================================================================================

import os
import torch
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geometric_matching.geotnf.transformation import GeometricTnf
from geometric_matching.util.net_util import roi_data

class CaltechDataset(Dataset):
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

    def __init__(self, csv_file, dataset_path, output_size=(240,240), transform=None):
        self.dataframe = pd.read_csv(csv_file)  # Read images data
        self.img_A_names = self.dataframe.iloc[:, 0] # Get source image & target image name
        self.img_B_names = self.dataframe.iloc[:, 1]
        self.categories = self.dataframe.iloc[:, 2].values.astype('float') # Get image category
        self.annot_A_str = self.dataframe.iloc[:, 3:5]
        self.annot_B_str = self.dataframe.iloc[:, 5:]
        self.dataset_path = dataset_path  # Path for reading images
        self.out_h, self.out_w = output_size
        self.transform = transform
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
        # Initialize an affine transformation to resize the image to (240, 240)
        self.affineTnf = GeometricTnf(geometric_model='affine', out_h=self.out_h, out_w=self.out_w, use_cuda=False)
              
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # get pre-processed images
        image_A, im_size_A, im_A, im_info_A, gt_boxes_A, num_boxes_A = self.get_image(img_name_list=self.img_A_names,
                                                                                      idx=idx)
        image_B, im_size_B, im_B, im_info_B, gt_boxes_B, num_boxes_B = self.get_image(img_name_list=self.img_B_names,
                                                                                      idx=idx)

        # get pre-processed point coords
        annot_A = self.get_points(point_coords_list=self.annot_A_str, idx=idx)
        annot_B = self.get_points(point_coords_list=self.annot_B_str, idx=idx)
                        
        sample = {'source_image': image_A, 'target_image': image_B,
                  'source_im_size': im_size_A, 'target_im_size': im_size_B,
                  'source_polygon': annot_A, 'target_polygon': annot_B,
                  'source_im': im_A, 'target_im': im_B,
                  'source_im_info': im_info_A, 'target_im_info': im_info_B,
                  'source_gt_boxes': gt_boxes_A, 'target_gt_boxes': gt_boxes_B,
                  'source_num_boxes': num_boxes_A, 'target_num_boxes': num_boxes_B}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self, img_name_list, idx):
        img_name = os.path.join(self.dataset_path, img_name_list[idx])
        image = io.imread(img_name)
        # If the image just has two channels, add one channel
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.concatenate((image, image, image), axis=2)

        # Get image size, (H, W, C)
        im_size = np.asarray(image.shape)

        # Get tensors of image, image_info (H, W, im_scale), ground-truth boxes, number of boxes for faster rcnn
        im, im_info, gt_boxes, num_boxes = roi_data(image)

        # Transform numpy to tensor, permute order of image to CHW
        image = torch.Tensor(image.astype(np.float32))
        image = image.permute(2, 0, 1)
        im_size = torch.Tensor(im_size)

        # Resize image using bilinear sampling with identity affine tnf
        image.requires_grad = False
        image = self.affineTnf(image.unsqueeze(0)).squeeze(0)

        return image, im_size, im, im_info, gt_boxes, num_boxes
    
    def get_points(self, point_coords_list, idx):
        point_coords_x = point_coords_list[point_coords_list.columns[0]][idx]
        point_coords_y = point_coords_list[point_coords_list.columns[1]][idx]

        return (point_coords_x, point_coords_y)