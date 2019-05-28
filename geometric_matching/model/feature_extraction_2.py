import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models


# Extract feature maps of two images with the pre-trained model (VGG16 or ResNet101)
class FeatureExtraction_2(torch.nn.Module):
    def __init__(self, use_cuda=True, feature_extraction_cnn='vgg', first_layer='', last_layer='', pretrained=True):
        super(FeatureExtraction_2, self).__init__()
        if feature_extraction_cnn == 'vgg':
            if pretrained:
                """Use ImageNet pre-trained vgg"""
                self.model = models.vgg16(pretrained=pretrained)
                # keep feature extraction network up to indicated layer
                vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
                                      'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
                                      'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
                                      'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                                      'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5']
                if first_layer == '':
                    first_layer = 'pool1'
                if last_layer == '':
                    last_layer = 'pool4'
                first_layer_idx = vgg_feature_layers.index(first_layer)
                last_layer_idx = vgg_feature_layers.index(last_layer)
                # Feature extraction networks consists of layers before pool4 (including pool4) of vgg16
                self.model = nn.Sequential(*list(self.model.features.children())[first_layer_idx:last_layer_idx + 1])

            else:
                """Use PascalVOC2011 fine-tuned vgg"""
                # Load pre-trained vgg16 from faster rcnn
                self.model_path = 'models/vgg16/pascal_voc_2011/faster_rcnn_1_7_23079.pth'
                self.model = models.vgg16()
                trained_state_dict = torch.load(self.model_path)['model']
                new_state_dict = self.model.state_dict().copy()

                new_list = list(self.model.state_dict().keys())
                trained_list = list(trained_state_dict.keys())

                # The first 6 keys of trained_state_dict belong to RPN
                for i in range(len(self.model.features.state_dict())):
                    new_state_dict[new_list[i]] = trained_state_dict[trained_list[i+6]]

                self.model.load_state_dict(new_state_dict)

                # Only keep the features part of vgg
                vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
                                      'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
                                      'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
                                      'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                                      'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5']
                if first_layer == '':
                    first_layer = 'pool1'
                if last_layer == '':
                    last_layer = 'pool4'
                first_layer_idx = vgg_feature_layers.index(first_layer)
                last_layer_idx = vgg_feature_layers.index(last_layer)
                # Feature extraction networks consists of layers after pool1 and before pool4 (both included) of vgg16
                self.model = nn.Sequential(*list(self.model.features.children())[first_layer_idx:last_layer_idx+1])

        if feature_extraction_cnn == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer == '':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]

            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx + 1])  # Get the feature maps of resnet101
        # freeze parameters, directly use the pre-trained model without finetuning
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # move to GPU
        # if use_cuda:
        #     self.model.cuda()

    def forward(self, image_batch):
        # print(image_batch.shape)
        # image_batch.shape: (batch_size, 3, H, W), such as (240, 240)
        return self.model(image_batch)