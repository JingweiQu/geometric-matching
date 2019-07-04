import torch
import torch.nn as nn

from geometric_matching.geotnf.transformation import GeometricTnf
# from geometric_matching.model.feature_extraction_2 import FeatureExtraction_2

from lib.model.roi_layers import ROIPool

class FeatureCrop(nn.Module):
    def __init__(self, crop_layer='img'):
        super().__init__()
        self.crop_layer = crop_layer
        if self.crop_layer == 'img':
            self.affineTnf = GeometricTnf(geometric_model='affine')
        elif self.crop_layer == 'pool4':
            self.RoIPool = ROIPool((15, 15), 1.0 / 16.0)
        # elif self.crop_layer == 'conv1':
        #     self.RoIPool = ROIPool((240, 240), 1.0 / 1.0)

    def forward(self, feature_batch, box_batch):
        if self.crop_layer is None:
            return feature_batch

        for i in range(box_batch.shape[0]):
            if torch.sum(box_batch[i]).item() > 0:
                # Crop object and resize object as (240, 240) if object is detected
                if self.crop_layer == 'img':
                    feature_batch[i, :, :, :] = self.rescalingTnf(image_batch=feature_batch[i, :, int(box_batch[i, 1]):int(box_batch[i, 3]), int(box_batch[i, 0]):int(box_batch[i, 2])].unsqueeze(0)).squeeze()
                # Crop object feature map and resize as (15, 15) if object is detected
                elif self.crop_layer == 'pool4':
                    feature_batch[i, :, :, :] = self.RoIPool(feature_batch[i, :, :, :].unsqueeze(0), torch.cat((torch.Tensor([0]).cuda(), box_batch[i]), 0).unsqueeze(0))

        return feature_batch
