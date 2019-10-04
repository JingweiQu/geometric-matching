# ========================================================================================
# Parameters for training and testing geometric model
# Author: Jingwei Qu
# Date: 05 Mar 2019
# ========================================================================================

import argparse
from geometric_matching.util.net_util import str_to_bool


class Arguments():
    def __init__(self, mode='train'):
        self.parser = argparse.ArgumentParser(description='GeometricMatching Arguments')
        self.add_base_parameters()
        self.add_model_parameters()
        if mode == 'train':
            self.add_train_parameters()
            self.add_train_dataset_parameters()
            self.add_loss_parameters()
        elif mode == 'test':
            self.add_test_parameters()
            self.add_test_dataset_parameters()

    def add_base_parameters(self):
        base_params = self.parser.add_argument_group('base')
        base_params.add_argument('--cuda', dest='cuda', help='Whether use CUDA', action='store_true')
        base_params.add_argument('--mGPUs', dest='mGPUs', help='Whether use multiple GPUs', action='store_true')
        base_params.add_argument('--geometric_model', dest='geometric_model', type=str, default='tps',
                                 help='Geometric model to be regressed at output: affine or tps')
        base_params.add_argument('--image_size', dest='image_size', type=int, default=240, help='Image input size')
        base_params.add_argument('--trained_models_dir', dest='trained_models_dir', type=str,
                                 default='geometric_matching/trained_models/PF-PASCAL',
                                 help='Path to trained models folder')
        base_params.add_argument('--seed', dest='seed', type=int, default=1, help='Pseudo-RNG seed')
        base_params.add_argument('--model', dest='model', type=str, default='', help='Pre-trained model filename')

    def add_model_parameters(self):
        """ Model parameters """
        model_params = self.parser.add_argument_group('model')
        model_params.add_argument('--feature_extraction_cnn', dest='feature_extraction_cnn', type=str, default='resnet101',
                                  help='Feature extraction CNN model architecture: vgg16/resnet101')
        model_params.add_argument('--feature_extraction_last_layer', dest='feature_extraction_last_layer', type=str,
                                  default='', help='Feature extraction CNN last layer')

    def add_train_parameters(self):
        """ Train parameters """
        train_params = self.parser.add_argument_group('train')
        # Optimization parameters
        train_params.add_argument('--lr', dest='lr', type=float, default=0.001, help='Learning rate')
        train_params.add_argument('--lr_decay_step', dest='lr_decay_step',
                                  help='Step to do learning rate decay, unit is epoch', default=10, type=int)
        train_params.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', help='learning rate decay ratio',
                                  default=0.1)
        train_params.add_argument('--momentum', dest='momentum', type=float, default=0.9, help='Momentum constant')
        train_params.add_argument('--start_epoch', dest='start_epoch', type=int, default=1,
                                  help='Start epoch for resuming training')
        train_params.add_argument('--num_epochs', dest='num_epochs', type=int, default=50,
                                  help='Number of training epochs')
        train_params.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Training batch size')
        train_params.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.0005,
                                  help='Weight decay constant')
        # train_params.add_argument('--seed', type=int, default=1, help='Pseudo-RNG seed')
        train_params.add_argument('--use_mse_loss', dest='use_mse_loss', type=str_to_bool, nargs='?', const=True,
                                  default=False, help='Use MSE loss on tnf. parameters')
        # Trained model parameters
        train_params.add_argument('--resume', dest='resume', help='Whether resume interrupted training',
                                  action='store_true')
        # train_params.add_argument('--trained-models-dir', type=str, default='geometric_matching/trained_models', help='Path to trained models folder')
        # train_params.add_argument('--trained-models-fn', type=str, default='checkpoint_adam', help='Trained model filename')
        # Parts of model to train
        train_params.add_argument('--train_fe', dest='train_fe', type=str_to_bool, nargs='?', const=True, default=True,
                                  help='Train feature extraction')
        train_params.add_argument('--train_fr', dest='train_fr', type=str_to_bool, nargs='?', const=True, default=True,
                                  help='Train feature regressor')
        train_params.add_argument('--train_bn', dest='train_bn', type=str_to_bool, nargs='?', const=True, default=True,
                                  help='Train batch-norm layers')
        train_params.add_argument('--fe_finetune_params', dest='fe_finetune_params', nargs='+', type=str, default=[''],
                                  help='String indicating the F.Ext params to finetune')
        train_params.add_argument('--update_bn_buffers', dest='update_bn_buffers', type=str_to_bool, nargs='?',
                                  const=True, default=False, help='Update batch norm running mean and std')

    def add_train_dataset_parameters(self):
        """ Train dataset parameters """
        dataset_params = self.parser.add_argument_group('dataset')
        # Training dataset
        dataset_params.add_argument('--train_dataset', dest='train_dataset', type=str, default='PF-PASCAL',
                                    help='Dataset to use for training: PF-PASCAL/PascalVOC2011')
        dataset_params.add_argument('--train_dataset_path', dest='train_dataset_path', type=str,
                                    default='geometric_matching/training_data',
                                    help='Path to folder containing training dataset')
        dataset_params.add_argument('--train_dataset_size', dest='train_dataset_size', type=int, default=0,
                                    help='Training dataset size limitation')
        dataset_params.add_argument('--random_sample', dest='random_sample', type=str_to_bool, nargs='?', const=True,
                                    default=False, help='Sample random transformations')
        dataset_params.add_argument('--random_t', dest='random_t', type=float, default=0.5,
                                    help='Random transformation translation')
        dataset_params.add_argument('--random_s', dest='random_s', type=float, default=0.5,
                                    help='Random transformation translation')
        dataset_params.add_argument('--random_alpha', dest='random_alpha', type=float, default=1 / 6,
                                    help='Random transformation translation')
        dataset_params.add_argument('--random_t_tps', dest='random_t_tps', type=float, default=0.4,
                                    help='Random transformation translation')
        dataset_params.add_argument('--random_crop', dest='random_crop', type=str_to_bool, nargs='?', const=True,
                                    default=False, help='Use random crop augmentation')
        # Eval dataset parameters for early stopping
        dataset_params.add_argument('--eval_dataset', dest='eval_dataset', type=str, default='TSS',
                                    help='Validation dataset used for early stopping')
        dataset_params.add_argument('--eval_dataset_path', dest='eval_dataset_path', type=str,
                                    default='geometric_matching/testing_data',
                                    help='Path to validation dataset used for early stopping')
        dataset_params.add_argument('--eval_dataset_size', dest='eval_dataset_size', type=int, default=0,
                                    help='Eval dataset size limitation')
        dataset_params.add_argument('--categories', dest='categories', nargs='+', type=int, default=0,
                                    help='Indices of categories for eval')
        dataset_params.add_argument('--eval_metric', dest='eval_metric', type=str, default='pck', help='pck/distance')
        dataset_params.add_argument('--pck_alpha', dest='pck_alpha', type=float, default=0.1,
                                    help='pck margin factor alpha')

    def add_loss_parameters(self):
        """ Parameters of loss """
        loss_params = self.parser.add_argument_group('loss')
        loss_params.add_argument('--tps_grid_size', dest='tps_grid_size', type=int, default=3, help='Tps grid size')
        loss_params.add_argument('--tps_reg_factor', dest='tps_reg_factor', type=float, default=0.2,
                                 help='Tps regularization factor')

    def add_test_parameters(self):
        """ Test parameters """
        test_params = self.parser.add_argument_group('test')
        # test_params.add_argument('--num-epochs', type=int, default=10, help='Number of training epochs')
        test_params.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='Testing batch size')

    def add_test_dataset_parameters(self):
        """ Test dataset parameters """
        dataset_params = self.parser.add_argument_group('dataset')
        dataset_params.add_argument('--test_dataset', dest='test_dataset', type=str, default='PF-WILLOW',
                                    help='Dataset to use for testing: PF-WILLOW/PF-PASCAL/Caltech-101/TSS')
        dataset_params.add_argument('--test_dataset_path', dest='test_dataset_path', type=str,
                                    default='geometric_matching/testing_data',
                                    help='Path to folder containing training dataset')
        dataset_params.add_argument('--categories', dest='categories', nargs='+', type=int, default=0,
                                    help='Indices of categories for testing')
        dataset_params.add_argument('--eval_metric', dest='eval_metric', type=str, default='pck', help='pck/distance')
        dataset_params.add_argument('--pck_alpha', dest='pck_alpha', type=float, default=0.1,
                                    help='pck margin factor alpha')
        dataset_params.add_argument('--flow_output_dir', dest='flow_output_dir', type=str,
                                    default='geometric_matching/tss_results/resnet101/', help='flow output dir')
        dataset_params.add_argument('--tps_reg_factor', dest='tps_reg_factor', type=float, default=0.0,
                                    help='regularisation factor for tps tnf')

    def parse(self, arg_str=None):
        if arg_str is None:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(arg_str.split())
        arg_groups = {}
        for group in self.parser._action_groups:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            arg_groups[group.title] = group_dict
        return args, arg_groups