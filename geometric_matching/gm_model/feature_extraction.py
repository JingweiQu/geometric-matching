import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from geometric_matching.gm_model.featureL2norm import FeatureL2Norm

# Extract feature maps of two images with the pre-trained model (VGG16 or ResNet101)
class FeatureExtraction(nn.Module):
    """
    Extract feature maps of images by VGG16 or ResNet101
    """

    def __init__(self, feature_extraction_cnn='resnet101', normalization=True, last_layer='', fixed_blocks=3, pytorch=False, caffe=False):
        super(FeatureExtraction, self).__init__()
        self.normalization = normalization
        if feature_extraction_cnn == 'vgg16':
            if pytorch:
                """ Use pytorch pre-trained vgg """
                vgg = models.vgg16(pretrained=True)
            else:
                vgg = models.vgg16()
            if caffe:
                self.model_path = 'geometric_matching/trained_models/vgg16_caffe.pth'
                print('Load pretrained vgg16 weights from {}'.format(self.model_path))
                state_dict = torch.load(self.model_path)
                vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

            # keep feature extraction network up to indicated layer
            vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
                                  'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
                                  'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
                                  'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                                  'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5']
            if last_layer == '':
                last_layer = 'pool4'
            last_layer_idx = vgg_feature_layers.index(last_layer)
            # Feature extraction networks consists of layers before pool4 (including pool4) of vgg16
            self.GM_base = nn.Sequential(*list(vgg.features.children())[:last_layer_idx + 1])

            # else:
            #     """ Use PascalVOC2011 fine-tuned vgg """
            #     # Load pre-trained vgg16 from faster rcnn (pre-trained on ImageNet, fine-tuned on PascalVOC2011)
            #     self.model_path = 'models/vgg16/pascal_voc_2011/best_faster_rcnn_1_7_23079.pth'
            #     self.model = models.vgg16()
            #     trained_state_dict = torch.load(self.model_path)['model']
            #     new_state_dict = self.model.state_dict().copy()
            #
            #     new_list = list(self.model.state_dict().keys())
            #     trained_list = list(trained_state_dict.keys())
            #
            #     # The first 6 keys of trained_state_dict belong to RPN
            #     for i in range(len(self.model.features.state_dict())):
            #         new_state_dict[new_list[i]] = trained_state_dict[trained_list[i+6]]
            #
            #     self.model.load_state_dict(new_state_dict)
            #
            #     # Only keep the features part of vgg
            #     vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
            #                           'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
            #                           'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
            #                           'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
            #                           'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5']
            #     if last_layer == '':
            #         last_layer = 'pool4'
            #     last_layer_idx = vgg_feature_layers.index(last_layer)
            #     # Feature extraction networks consists of layers before pool1 of vgg16
            #     self.model = nn.Sequential(*list(self.model.features.children())[:last_layer_idx + 1])

        if feature_extraction_cnn == 'resnet101':
            if pytorch:
                resnet = models.resnet101(pretrained=True)
            else:
                resnet = models.resnet101()
            if caffe:
                self.model_path = 'geometric_matching/trained_models/resnet101_caffe.pth'
                print('Load pretrained resnet101 weights from {}'.format(self.model_path))
                state_dict = torch.load(self.model_path)
                resnet.load_state_dict({k:v for k, v in state_dict.items() if k in resnet.state_dict()})

            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            # first_layer = 'layer1'
            # first_layer_idx = resnet_feature_layers.index(first_layer)
            if last_layer == '':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [resnet.conv1,
                                  resnet.bn1,
                                  resnet.relu,
                                  resnet.maxpool,
                                  resnet.layer1,
                                  resnet.layer2,
                                  resnet.layer3,
                                  resnet.layer4]

            self.GM_base = nn.Sequential(*resnet_module_list[:last_layer_idx + 1])  # Get the feature maps of resnet101
            # self.GM_base_1 = nn.Sequential(*resnet_module_list[:first_layer_idx + 1])  # Get the feature maps of resnet101
            # self.GM_base_2 = nn.Sequential(*resnet_module_list[first_layer_idx + 1:last_layer_idx + 1])  # Get the feature maps of resnet101

            # Fix blocks
            # for param in self.GM_base[0].parameters():
            #     param.requires_grad = False
            # for param in self.GM_base[1].parameters():
            #     param.requires_grad = False
            # if fixed_blocks >= 3:
            #     for param in self.GM_base[6].parameters():
            #         param.requires_grad = False
            # if fixed_blocks >= 2:
            #     for param in self.GM_base[5].parameters():
            #         param.requires_grad = False
            # if fixed_blocks >= 1:
            #     for param in self.GM_base[4].parameters():
            #         param.requires_grad = False


    def forward(self, image_batch):
        # image_batch.shape: (batch_size, 3, H, W), such as (240, 240)
        features = self.GM_base(image_batch)
        # features_1 = self.GM_base_1(image_batch)
        # features_2 = self.GM_base_2(features_1)
        # Normalize feature maps
        if self.normalization:
            features = FeatureL2Norm(features)
            # features_1 = FeatureL2Norm(features_1)
            # features_2 = FeatureL2Norm(features_2)
        return features
        # return features_1, features_2