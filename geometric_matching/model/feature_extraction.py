import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from geometric_matching.model.featureL2norm import FeatureL2Norm

# Extract feature maps of two images with the pre-trained model (VGG16 or ResNet101)
class FeatureExtraction(nn.Module):
    """
    Extract feature maps of images by VGG16 or ResNet101
    """

    def __init__(self, train_fe=False, feature_extraction_cnn='vgg', normalization=True, last_layer='', use_cuda=True,
                 pretrained=True):
        super(FeatureExtraction, self).__init__()
        self.normalization = normalization
        if feature_extraction_cnn == 'vgg':
            if pretrained:
                """ Use ImageNet pre-trained vgg """
                self.model = models.vgg16(pretrained=pretrained)
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
                self.model = nn.Sequential(*list(self.model.features.children())[:last_layer_idx + 1])

            else:
                """ Use PascalVOC2011 fine-tuned vgg """
                # Load pre-trained vgg16 from faster rcnn (pre-trained on ImageNet, fine-tuned on PascalVOC2011)
                self.model_path = 'models/vgg16/pascal_voc_2011/best_faster_rcnn_1_7_23079.pth'
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
                if last_layer == '':
                    last_layer = 'pool4'
                last_layer_idx = vgg_feature_layers.index(last_layer)
                # Feature extraction networks consists of layers before pool1 of vgg16
                self.model = nn.Sequential(*list(self.model.features.children())[:last_layer_idx + 1])

        if feature_extraction_cnn == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
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
        if feature_extraction_cnn == 'resnet101_v2':
            self.model = models.resnet101(pretrained=pretrained)
            # keep feature extraction network up to pool4 (last layer - 7)
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if feature_extraction_cnn == 'densenet201':
            self.model = models.densenet201(pretrained=pretrained)
            # keep feature extraction network up to denseblock3
            # self.model = nn.Sequential(*list(self.model.features.children())[:-3])
            # keep feature extraction network up to transitionlayer2
            self.model = nn.Sequential(*list(self.model.features.children())[:-4])

        '''
        # freeze parameters, directly use the pre-trained model without finetuning
        if not train_fe:
            for param in self.model.parameters():
                param.requires_grad = False
        # move to GPU
        if use_cuda:
            self.model.cuda()
        '''

    def forward(self, image_batch):
        # image_batch.shape: (batch_size, 3, H, W), such as (240, 240)
        features = self.model(image_batch)
        # Normalize feature maps
        if self.normalization:
            features = FeatureL2Norm(features)
        return features