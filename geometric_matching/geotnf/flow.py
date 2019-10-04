import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from geometric_matching.geotnf.point_tnf import normalize_axis, unnormalize_axis
from visdom import Visdom
import matplotlib.colors as cl
import matplotlib.pyplot as plt

UNKNOWN_FLOW_THRESH = 1e7

def read_flow(filename):
    """
    read optical flow data from flow file
    :param filename: name of the flow file
    :return: optical flow data in numpy array
    """
    if filename.endswith('.flo'):
        flow = read_flo_file(filename)
    else:
        raise Exception('Invalid flow file format!')

    return flow

def show_flow(filename):
    """
    visualize optical flow map using matplotlib
    :param filename: optical flow file
    :return: None
    """
    flow = read_flow(filename)
    img = flow_to_image(flow)
    plt.imshow(img)
    plt.show()

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def visualize_flow(flow, mode='Y'):
    """
    this function visualize the input flow
    :param flow: input flow in array
    :param mode: choose which color mode to visualize the flow (Y: Ccbcr, RGB: RGB color)
    :return: None
    """
    if mode == 'Y':
        # Ccbcr color wheel
        img = flow_to_image(flow)
        # plt.imshow(img)
        # plt.show()
        # vis = Visdom()
        # vis.matplot(plt)
    elif mode == 'RGB':
        (h, w) = flow.shape[0:2]
        du = flow[:, :, 0]
        dv = flow[:, :, 1]
        valid = flow[:, :, 2]
        max_flow = max(np.max(du), np.max(dv))
        img = np.zeros((h, w, 3), dtype=np.float64)
        # angle layer
        img[:, :, 0] = np.arctan2(dv, du) / (2 * np.pi)
        # magnitude layer, normalized to 1
        img[:, :, 1] = np.sqrt(du * du + dv * dv) * 8 / max_flow
        # phase layer
        img[:, :, 2] = 8 - img[:, :, 1]
        # clip to [0,1]
        small_idx = img[:, :, 0:3] < 0
        large_idx = img[:, :, 0:3] > 1
        img[small_idx] = 0
        img[large_idx] = 1
        # convert to rgb
        img = cl.hsv_to_rgb(img)
        # remove invalid point
        img[:, :, 0] = img[:, :, 0] * valid
        img[:, :, 1] = img[:, :, 1] * valid
        img[:, :, 2] = img[:, :, 2] * valid
        # show
        # plt.imshow(img)
        # plt.show()
        # vis = Visdom()
        # vis.matplot(plt)

    # return None
    return img

def read_flo_file(filename,verbose=False):
    """
    Read from .flo optical flow file (Middlebury format)
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    
    adapted from https://github.com/liruoteng/OpticalFlowToolkit/
    
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        raise TypeError('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        if verbose:
            print("Reading %d x %d flow file in .flo format" % (h, w))
        data2d = np.fromfile(f, np.float32, count=int(2 * w * h))
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
    f.close()
    return data2d

def write_flo_file(flow, filename):
    """
    Write optical flow in Middlebury .flo format
    
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    
    from https://github.com/liruoteng/OpticalFlowToolkit/
    
    """
    # forcing conversion to float32 precision
    flow = flow.astype(np.float32)
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()

def warp_image(image, flow):
    """
    Warp image (np.ndarray, shape=[h_src,w_src,3]) with flow (np.ndarray, shape=[h_tgt,w_tgt,2])
    
    """
    h_src,w_src=image.shape[0],image.shape[1]
    sampling_grid_torch = np_flow_to_th_sampling_grid(flow, h_src, w_src)
    image_torch = Variable(torch.FloatTensor(image.astype(np.float32)).transpose(1,2).transpose(0,1).unsqueeze(0))
    warped_image_torch = F.grid_sample(image_torch, sampling_grid_torch)
    warped_image = warped_image_torch.data.squeeze(0).transpose(0,1).transpose(1,2).numpy().astype(np.uint8)
    return warped_image

def np_flow_to_th_sampling_grid(flow,h_src,w_src,use_cuda=False):
    h_tgt,w_tgt=flow.shape[0],flow.shape[1]
    grid_x,grid_y = np.meshgrid(range(1,w_tgt+1),range(1,h_tgt+1))
    disp_x=flow[:,:,0]
    disp_y=flow[:,:,1]
    source_x=grid_x+disp_x
    source_y=grid_y+disp_y
    source_x_norm=normalize_axis(source_x,w_src) 
    source_y_norm=normalize_axis(source_y,h_src) 
    sampling_grid=np.concatenate((np.expand_dims(source_x_norm,2),
                                  np.expand_dims(source_y_norm,2)),2)
    sampling_grid_torch = Variable(torch.FloatTensor(sampling_grid).unsqueeze(0))
    if use_cuda:
        sampling_grid_torch = sampling_grid_torch.cuda()
    return sampling_grid_torch

def th_sampling_grid_to_np_flow(source_grid, h_src, w_src):
    """ Transform sampling grid to flow """
    # Flow describes displacements of coordinates of each pixel (x and y)
    # remove batch dimension, source_grid.shape: (h_tgt, w_tgt, 2)
    source_grid = source_grid.squeeze(0)
    # get mask
    in_bound_mask = (source_grid[:, :, 0] > -1) & (source_grid[:, :, 0] < 1) & (source_grid[:, :, 1] > -1) & (source_grid[:, :, 1] < 1)
    in_bound_mask = in_bound_mask.cpu().numpy()
    # convert coords
    h_tgt, w_tgt = source_grid.size(0), source_grid.size(1)
    source_x_norm = source_grid[:, :, 0]
    source_y_norm = source_grid[:, :, 1]
    source_x = unnormalize_axis(source_x_norm, w_src)
    source_y = unnormalize_axis(source_y_norm, h_src)
    # source_x = source_x.data.cpu().numpy()
    # source_y = source_y.data.cpu().numpy()
    source_x = source_x.cpu().numpy()
    source_y = source_y.cpu().numpy()
    # Generate original coordinates of pixels, grid_x, grid_y.shape: (h_tgt, w_tgt)
    grid_x, grid_y = np.meshgrid(range(1, w_tgt + 1), range(1, h_tgt + 1))
    disp_x = source_x - grid_x
    disp_y = source_y - grid_y
    # apply mask
    disp_x = disp_x * in_bound_mask + 1e10 * (1 - in_bound_mask)
    disp_y = disp_y * in_bound_mask + 1e10 * (1 - in_bound_mask)
    # flow.shape: (h_tgt, w_tgt, 2)
    flow = np.concatenate((np.expand_dims(disp_x, 2), np.expand_dims(disp_y, 2)), 2)
    # flow = np.concatenate((flow, np.expand_dims(in_bound_mask, 2)), 2)

    # vis = Visdom()
    # vis.heatmap(in_bound_mask)
    # tmp_x = np.reshape(source_x, (-1, 1))
    # tmp_y = np.reshape(source_y, (-1, 1))
    # source_xy = np.concatenate((tmp_x, tmp_y), 1)
    # vis.mesh(source_xy, opts=dict(opacity=0.3))
    # grid_X_vec = np.reshape(grid_x, (-1, 1))
    # grid_Y_vec = np.reshape(grid_y, (-1, 1))
    # grid_XY_vec = np.concatenate((grid_X_vec, grid_Y_vec), 1)
    # vis.mesh(grid_XY_vec, opts=dict(opacity=0.3))
    # flow_img = visualize_flow(flow, 'Y')
    # vis.image(flow_img.transpose((2, 0, 1)))

    return flow

