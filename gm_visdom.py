import torch
import torchvision
import numpy as np
import os
from visdom import Visdom

from geometric_matching.arguments.arguments_setting import Arguments

def loss_line(vis, checkpoint, title):
    index = np.where(checkpoint['train_loss'] != 0)[0][-1]
    train_loss = checkpoint['train_loss'][0:index + 1]
    epochs = np.arange(1, index + 2)
    opts = dict(xlabel='Epoch',
                ylabel='Loss',
                title=title,
                legend=['Loss'])
    vis.line(train_loss, epochs, opts=opts)


if __name__ == '__main__':
    args = Arguments(mode='train').parse()[0]

    vis = Visdom()

    weak_cp_name = '/home/qujingwei/weakalign/trained_models/weakalign_resnet101_affine_tps.pth.tar'
    if not os.path.exists(weak_cp_name):
        raise Exception('There is no pre-trained weakly end-to-end model, i.e. ' + weak_cp_name)
    print('Load weakly end-to-end model {}'.format(weak_cp_name))
    weak_cp = torch.load(weak_cp_name, map_location=lambda storage, loc: storage)

    cnngeo_cp_name_aff = '/home/qujingwei/weakalign/trained_models/cnngeo_resnet101_affine.pth.tar'
    if not os.path.exists(cnngeo_cp_name_aff):
        raise Exception('There is no pre-trained cnn_geo affine model, i.e. ' + cnngeo_cp_name_aff)
    print('Load cnn_geo affine model {}'.format(cnngeo_cp_name_aff))
    cnngeo_cp_aff = torch.load(cnngeo_cp_name_aff, map_location=lambda storage, loc: storage)

    cnngeo_cp_name_tps = '/home/qujingwei/weakalign/trained_models/cnngeo_resnet101_tps.pth.tar'
    if not os.path.exists(cnngeo_cp_name_tps):
        raise Exception('There is no pre-trained cnn_geo tps model, i.e. ' + cnngeo_cp_name_tps)
    print('Load cnn_geo tps model {}'.format(cnngeo_cp_name_tps))
    cnngeo_cp_tps= torch.load(cnngeo_cp_name_tps, map_location=lambda storage, loc: storage)

    GM_cp_name_aff = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, args.model_aff)
    if not os.path.exists(GM_cp_name_aff):
        raise Exception('There is no pre-trained geometric matching affine model, i.e. ' + GM_cp_name_aff)
    print('Load geometric matching affine model {}'.format(GM_cp_name_aff))
    GM_checkpoint_aff = torch.load(GM_cp_name_aff, map_location=lambda storage, loc: storage)

    GM_cp_name_tps = os.path.join(args.trained_models_dir, args.feature_extraction_cnn, args.model_tps)
    if not os.path.exists(GM_cp_name_tps):
        raise Exception('There is no pre-trained geometric matching tps model, i.e. ' + GM_cp_name_tps)
    print('Load geometric matching tps model {}'.format(GM_cp_name_tps))
    GM_checkpoint_tps = torch.load(GM_cp_name_tps, map_location=lambda storage, loc: storage)

    loss_line(vis, weak_cp, 'Weakly End-to-end Training Loss')
    loss_line(vis, cnngeo_cp_aff, 'CNNGEO ResNet101 Affine Training Loss')
    loss_line(vis, cnngeo_cp_tps, 'CNNGEO ResNet101 TPS Training Loss')
    loss_line(vis, GM_checkpoint_aff, 'GM ResNet101 Affine Training Loss')
    loss_line(vis, GM_checkpoint_tps, 'GM ResNet101 TPS Training Loss')

