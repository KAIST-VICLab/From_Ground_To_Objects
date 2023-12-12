# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import torch
import numpy as np
import random
import torch.nn.functional as F
import yaml

import matplotlib.pyplot as plt


width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351
width_to_focal[1226] = 707.0912
width_to_focal[1280] = 738.2355 # focal lenght upscaled

width_to_baseline = dict()
width_to_baseline[1242] = 0.9982 * 0.54
width_to_baseline[1241] = 0.9848 * 0.54
width_to_baseline[1224] = 1.0144 * 0.54
width_to_baseline[1238] = 0.9847 * 0.54
width_to_baseline[1226] = 0.9765 * 0.54
width_to_baseline[1280] = 0.54

def get_point_cloud(img, disp):
    b, c, h, w = disp.shape

    # Set camera parameters
    # focal = width_to_focal[w]
    cx = w / 2
    cy = h / 2
    # baseline = width_to_baseline[w]

    # Get depth from disparity
    # z = focal * baseline / (disp + 0.0001)
    z = 1 / (disp + 0.0001)

    # Make normalized grid
    i_tetha = torch.zeros(b, 2, 3).cuda()
    i_tetha[:, 0, 0] = 1
    i_tetha[:, 1, 1] = 1
    grid = F.affine_grid(i_tetha, [b, c, h, w])
    grid = (grid + 1) / 2

    # Get horizontal and vertical pixel coordinates
    u = grid[:,:,:,0].unsqueeze(1) * w
    v = grid[:,:,:,1].unsqueeze(1) * h

    # Get X, Y world coordinates
    # x = ((u - cx) / focal) * z
    # y = ((v - cy) / focal) * z
    x = ((u - cx) / cx) * z
    y = ((v - cy) / cy) * z

    # Cap coordinates
    # z[z < 0] = 0
    # z[z > 10] = 10

    mask = torch.ones_like(x)
    mask[z > 10] = 0
    mask[z < 0] = 0
    mask[y > 10] = 0
    mask[y < -5] = 0
    mask[x > 10] = 0
    mask[x < -10] = 0

    xyz_rgb = torch.cat([x, z, -y, img], 1)
    xyz_rgb = xyz_rgb.view(1, 6, h*w)
    mask = mask.view(1, 1, h*w)
    filtered_xyz_rgb = []
    for i in range(h*w):
        if mask[0, 0, i] == 0: continue
        filtered_xyz_rgb.append(xyz_rgb[0, :, i])

    filtered_xyz_rgb = torch.stack(filtered_xyz_rgb, dim=1).unsqueeze(0)
    return filtered_xyz_rgb

# Saves pointcloud in .ply format for visualizing
# I recommend blender for visualization
def save_point_cloud(pc, file_name):
    _, vertex_no = pc.shape

    with open(file_name, 'w+') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(vertex_no))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar diffuse_red\n')
        f.write('property uchar diffuse_green\n')
        f.write('property uchar diffuse_blue\n')
        f.write('end_header\n')
        for i in range(vertex_no):
            f.write('{:f} {:f} {:f} {:d} {:d} {:d}\n'.format(pc[0, i], pc[1, i], pc[2, i],
                                                             int(pc[3, i]), int(pc[4, i]), int(pc[5, i])))


_ERROR_COLORMAP = plt.get_cmap('jet', 256)  # for plotting
def error_colormap(inputs):
    return _ERROR_COLORMAP(inputs)[:, :, :3]

_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting
def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def seed_all(seed):
    if not seed:
        seed = 1

    # print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path, default_path=None, inherit_from=None):
    ''' Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)

    # Check if we should inherit from a config
    # inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v