import shutil
import torch
from torch.autograd import Variable
from os import makedirs, remove
from os.path import exists, join, basename, dirname
import argparse
from image.normalization import normalize_image

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


class BatchTensorToVars(object):
    """Convert tensors in dict batch to vars
    """
    def __init__(self, use_cuda=True):
        self.use_cuda=use_cuda
        
    def __call__(self, batch):
        batch_var = {}
        for key,value in batch.items():
            batch_var[key] = Variable(value,requires_grad=False)
            if self.use_cuda:
                batch_var[key] = batch_var[key].cuda()
            
        return batch_var


def save_checkpoint(state, is_best, file):
    model_dir = dirname(file)
    model_fn = basename(file)
    # make dir if needed (should be non-empty)
    if model_dir != '' and not exists(model_dir):
        makedirs(model_dir)
    torch.save(state, file)
    # Select the best model, and copy
    if is_best:
        shutil.copyfile(file, join(model_dir, 'best_' + model_fn))


def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
