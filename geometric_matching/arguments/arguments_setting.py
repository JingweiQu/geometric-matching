# ========================================================================================
# Parameters for training and testing geometric model
# Author: Jingwei Qu
# Date: 05 Mar 2019
# ========================================================================================

import argparse
from geometric_matching.util.net_util import str_to_bool

class Arguments():
    def __init__(self,mode='train'):
        self.parser = argparse.ArgumentParser(description='GeometricMatching Arguments')
        self.add_base_parameters()
        self.add_faster_rcnn_parmaters()
        self.add_model_parameters()
        if mode=='train':
            self.add_train_parameters()
            self.add_training_dataset_parameters()
            self.add_loss_parameters()
        elif mode=='test':
            self.add_test_parameters()
            self.add_testing_dataset_parameters()
        
    def add_base_parameters(self):
        base_params = self.parser.add_argument_group('base')
        base_params.add_argument('--cuda', dest='cuda', help='Whether use CUDA', action='store_true')
        base_params.add_argument('--mGPUs', dest='mGPUs', help='Whether use multiple GPUs', action='store_true')
        # model_params.add_argument('--model', type=str, default='', help='Pre-trained model filename')
        # model_params.add_argument('--model-aff', type=str, default='', help='Trained affine model filename')
        # model_params.add_argument('--model-tps', type=str, default='', help='Trained TPS model filename')

    def add_faster_rcnn_parmaters(self):
        """ Faster-RCNN parameters """
        rcnn_params = self.parser.add_argument_group('rcnn')
        rcnn_params.add_argument('--dataset', dest='dataset', help='training dataset', default='pascal_voc_2011', type=str)
        rcnn_params.add_argument('--cfg', dest='cfg_file', help='optional config file', default='cfgs/vgg16.yml', type=str)
        rcnn_params.add_argument('--net', dest='net', help='vgg16, res50, res101, res152', default='vgg16', type=str)
        rcnn_params.add_argument('--set', dest='set_cfgs', help='set config keys', default=None, nargs=argparse.REMAINDER)
        rcnn_params.add_argument('--load_dir', dest='load_dir', help='directory to load models', default="models", type=str)
        rcnn_params.add_argument('--cag', dest='class_agnostic', help='whether perform class_agnostic bbox regression', action='store_true')
        rcnn_params.add_argument('--parallel_type', dest='parallel_type', help='which part of model to parallel, 0: all, 1: model before roi pooling', default=0, type=int)
        rcnn_params.add_argument('--checksession', dest='checksession', help='checksession to load model', default=1, type=int)
        rcnn_params.add_argument('--checkepoch', dest='checkepoch', help='checkepoch to load network', default=7, type=int)
        rcnn_params.add_argument('--checkpoint', dest='checkpoint', help='checkpoint to load network', default=23079, type=int)

    def add_model_parameters(self):
        """ Model parameters """
        model_params = self.parser.add_argument_group('model')
        model_params.add_argument('--image-size', type=int, default=240, help='Image input size')
        model_params.add_argument('--geometric-model', type=str, default='tps', help='Geometric model to be regressed at output: affine or tps')
        model_params.add_argument('--use-mse-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use MSE loss on tnf. parameters')
        model_params.add_argument('--feature-extraction-cnn', type=str, default='vgg', help='Feature extraction CNN model architecture: vgg/resnet101')
        model_params.add_argument('--feature-extraction-last-layer', type=str, default='', help='Feature extraction CNN last layer')
        model_params.add_argument('--fr-feature-size', type=int, default=15, help='Feature map input size for feat.reg. conv layers')
        model_params.add_argument('--fr-kernel-sizes', nargs='+', type=int, default=[7,5], help='Kernels sizes in feat.reg. conv layers')
        model_params.add_argument('--fr-channels', nargs='+', type=int, default=[128,64], help='Channels in feat. reg. conv layers')

    def add_train_parameters(self):
        """ Training parameters """
        train_params = self.parser.add_argument_group('train')
        # Optimization parameters
        train_params.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        train_params.add_argument('--momentum', type=float, default=0.9, help='Momentum constant')
        train_params.add_argument('--num-epochs', type=int, default=10, help='Number of training epochs')
        train_params.add_argument('--batch-size', type=int, default=16, help='Training batch size')
        train_params.add_argument('--weight-decay', type=float, default=0, help='Weight decay constant')
        train_params.add_argument('--seed', type=int, default=1, help='Pseudo-RNG seed')
        # Trained model parameters
        train_params.add_argument('--trained-models-dir', type=str, default='geometric_matching/trained_models', help='Path to trained models folder')
        train_params.add_argument('--trained-models-fn', type=str, default='checkpoint_adam', help='Trained model filename')
        # Parts of model to train
        train_params.add_argument('--train-fe', type=str_to_bool, nargs='?', const=True, default=True, help='Train feature extraction')
        train_params.add_argument('--train-fr', type=str_to_bool, nargs='?', const=True, default=True, help='Train feature regressor')
        train_params.add_argument('--train-bn', type=str_to_bool, nargs='?', const=True, default=True, help='Train batch-norm layers')
        train_params.add_argument('--fe-finetune-params',  nargs='+', type=str, default=[''], help='String indicating the F.Ext params to finetune')
        train_params.add_argument('--update-bn-buffers', type=str_to_bool, nargs='?', const=True, default=False, help='Update batch norm running mean and std')

    def add_training_dataset_parameters(self):
        """ Training dataset parameters """
        dataset_params = self.parser.add_argument_group('dataset')
        # Training dataset
        dataset_params.add_argument('--training-dataset', type=str, default='PF-PASCAL', help='Dataset to use for training: PF-PASCAL/PascalVOC2011')
        dataset_params.add_argument('--training-dataset-path', type=str, default='geometric_matching/training_data', help='Path to folder containing training dataset')
        dataset_params.add_argument('--train-dataset-size', type=int, default=0, help='Training dataset size limitation')
        dataset_params.add_argument('--random-sample', type=str_to_bool, nargs='?', const=True, default=False, help='Sample random transformations')
        dataset_params.add_argument('--random-t', type=float, default=0.5, help='Random transformation translation')
        dataset_params.add_argument('--random-s', type=float, default=0.5, help='Random transformation translation')
        dataset_params.add_argument('--random-alpha', type=float, default=1/6, help='Random transformation translation')
        dataset_params.add_argument('--random-t-tps', type=float, default=0.4, help='Random transformation translation')
        # Eval dataset parameters for early stopping
        dataset_params.add_argument('--eval-dataset', type=str, default='PF-PASCAL', help='Validation dataset used for early stopping')
        dataset_params.add_argument('--eval-dataset-path', type=str, default='', help='Path to validation dataset used for early stopping')
        dataset_params.add_argument('--eval-dataset-size', type=int, default=0, help='Eval dataset size limitation')
        dataset_params.add_argument('--categories', nargs='+', type=int, default=0, help='indices of categories for training/eval')
        dataset_params.add_argument('--eval-metric', type=str, default='pck', help='pck/distance')
        dataset_params.add_argument('--pck-alpha', type=float, default=0.1, help='pck margin factor alpha')

    def add_loss_parameters(self):
        """ Parameters of loss """
        loss_params = self.parser.add_argument_group('loss')
        loss_params.add_argument('--tps-grid-size', type=int, default=3, help='Tps grid size')
        loss_params.add_argument('--tps-reg-factor', type=float, default=0.2, help='Tps regularization factor')
        
    def add_test_parameters(self):
        """ Testing parameters """
        test_params = self.parser.add_argument_group('test')
        test_params.add_argument('--num-epochs', type=int, default=10, help='Number of training epochs')
        test_params.add_argument('--batch-size', type=int, default=16, help='Testing batch size')
        test_params.add_argument('--trained-models-dir', type=str, default='geometric_matching/trained_models/PF-PASCAL/loss_1/image/identity/random_t_tps_0.4_b/', help='Path to trained models folder')
        test_params.add_argument('--feature-extraction-cnn', type=str, default='vgg', help='Feature extraction architecture: vgg/resnet101')

    def add_testing_dataset_parameters(self):
        """ Testing dataset parameters """
        dataset_params = self.parser.add_argument_group('dataset')
        dataset_params.add_argument('--testing-dataset', type=str, default='PF-PASCAL', help='Dataset to use for testing: PF-WILLOW/PF-PASCAL/caltech/tss')
        dataset_params.add_argument('--testing-dataset-path', type=str, default='geometric_matching/testing_data', help='Path to folder containing training dataset')
        dataset_params.add_argument('--eval-metric', type=str, default='pck', help='pck/distance')
        dataset_params.add_argument('--pck-alpha', type=float, default=0.1, help='pck margin factor alpha')
        dataset_params.add_argument('--tps-reg-factor', type=float, default=0.0, help='regularisation factor for tps tnf')
        # dataset_params.add_argument('--flow-output-dir', type=str, default='results/', help='flow output dir')

    def parse(self, arg_str=None):
        if arg_str is None:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(arg_str.split())
        arg_groups = {}
        for group in self.parser._action_groups:
            group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
            arg_groups[group.title]=group_dict
        return args, arg_groups

        