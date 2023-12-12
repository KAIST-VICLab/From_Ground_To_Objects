# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


def visual_feature(features, stage):
    feature_map = features.squeeze(0).cpu()
    n, h, w = feature_map.size()
    print(h, w)
    list_mean = []
    # sum_feature_map = torch.sum(feature_map,0)
    sum_feature_map, _ = torch.max(feature_map, 0)
    for i in range(n):
        list_mean.append(torch.mean(feature_map[i]))

    sum_mean = sum(list_mean)
    feature_map_weighted = torch.ones([n, h, w])
    for i in range(n):
        feature_map_weighted[i, :, :] = (torch.mean(feature_map[i]) / sum_mean) * feature_map[i, :, :]
    sum_feature_map_weighted = torch.sum(feature_map_weighted, 0)
    plt.imshow(sum_feature_map)
    # plt.savefig('feature_viz/{}_stage.png'.format(a))
    plt.savefig('feature_viz/decoder_{}.png'.format(stage))
    plt.imshow(sum_feature_map_weighted)
    # plt.savefig('feature_viz/{}_stage_weighted.png'.format(a))
    plt.savefig('feature_viz/decoder_{}_weighted.png'.format(stage))


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp

    return scaled_disp, depth

def depth_to_disp(depth, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    depth = depth.clamp(min=min_depth, max=max_depth)
    scaled_disp = 1 / depth
    disp = (scaled_disp - min_disp) / (max_disp - min_disp)
    return disp

def depth_to_scaled_disp(depth, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    depth = depth.clamp(min=min_depth, max=max_depth)
    disp = 1 / depth

    return disp


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M

def rotation_from_parameters(axisangle, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)

    if invert:
        R = R.transpose(1, 2)

    return R

def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T

def get_zero_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1

    return T

def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def coords_to_normals(coords):
    """Calculate surface normals using first order finite-differences.
    https://github.com/voyleg/perceptual-depth-sr/
    Parameters
    ----------
    coords : array_like
        Coordinates of the points (**, 3, h, w).
    Returns
    -------
    normals : torch.Tensor
        Surface normals (**, 3, h, w).
    """
    coords = torch.as_tensor(coords)
    if coords.ndim < 4:
        coords = coords[None]

    dxdu = coords[..., 0, :, 1:] - coords[..., 0, :, :-1]
    dydu = coords[..., 1, :, 1:] - coords[..., 1, :, :-1]
    dzdu = coords[..., 2, :, 1:] - coords[..., 2, :, :-1]
    dxdv = coords[..., 0, 1:, :] - coords[..., 0, :-1, :]
    dydv = coords[..., 1, 1:, :] - coords[..., 1, :-1, :]
    dzdv = coords[..., 2, 1:, :] - coords[..., 2, :-1, :]

    dxdu = torch.nn.functional.pad(dxdu, (0, 1),       mode='replicate')
    dydu = torch.nn.functional.pad(dydu, (0, 1),       mode='replicate')
    dzdu = torch.nn.functional.pad(dzdu, (0, 1),       mode='replicate')

    # pytorch cannot just do `dxdv = torch.nn.functional.pad(dxdv, (0, 0, 0, 1), mode='replicate')`, so
    dxdv = torch.cat([dxdv, dxdv[..., -1:, :]], dim=-2)
    dydv = torch.cat([dydv, dydv[..., -1:, :]], dim=-2)
    dzdv = torch.cat([dzdv, dzdv[..., -1:, :]], dim=-2)

    n_x = dydv * dzdu - dydu * dzdv
    n_y = dzdv * dxdu - dzdu * dxdv
    n_z = dxdv * dydu - dxdu * dydv

    n = torch.stack([n_x, n_y, n_z], dim=-3)
    n = torch.nn.functional.normalize(n, dim=-3)
    return n

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        b = depth.size(0)
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords[:b])
        cam_points = depth.view(b, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones[:b]], 1)
        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        b = points.size(0)
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P[:b], points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(b, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2

        return pix_coords


class Rotation3D(nn.Module):
    def __init__(self, height, width, eps=1e-7):
        super(Rotation3D, self).__init__()

        self.height = height
        self.width = width
        self.eps = eps
        self.pix_grid = self.get_pix_grid(self.width, self.height)

    def get_pix_grid(self, w, h):
        meshgrid = np.meshgrid(range(w), range(h), indexing='xy')
        pix_loc = np.stack(meshgrid, axis=0).astype(np.float32)
        pix_loc = torch.tensor(pix_loc, requires_grad=False)
        pix_loc = torch.unsqueeze(torch.stack([pix_loc[0].view(-1), pix_loc[1].view(-1)], 0), 0)
        return pix_loc

    def forward(self, K, invK, R):
        b = R.size(0)
        P = torch.matmul(K, R)[:, :3, :3]

        ones = torch.ones(1, 1, self.height * self.width).to(R.device)
        curr = self.pix_grid.clone()
        curr = curr.to(R.device)
        curr = torch.cat([curr, ones], 1)
        curr = curr.repeat(b, 1, 1)

        back_curr = torch.matmul(invK[:, :3, :3], curr)
        rot_back_curr = torch.matmul(P, back_curr)

        pix_coords = rot_back_curr[:, :2, :] / (rot_back_curr[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(b, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2

        return pix_coords

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

# former one
# def get_inst_aware_smooth_loss(disp, img, map, bd_x, bd_y):
#     grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
#     grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
#     grad_img_y[map[:, :-1, :].unsqueeze(1) == 1] = 0
#
#     gravity_mask_y = 100 * map[:, :-1, :] + 1 * (1 - map[:, :-1, :])
#     gravity_mask_y *= bd_y
#     grad_disp_y *= gravity_mask_y.unsqueeze(1)
#
#     grad_disp_y *= torch.exp(-grad_img_y)
#
#     grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
#     grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
#     grad_img_x[map[:, :, -1].unsqueeze(1) == 1] = 0
#     grad_disp_x *= bd_x.unsqueeze(1)
#     grad_disp_x *= torch.exp(-grad_img_x)
#
#     return grad_disp_y.mean() + grad_disp_x.mean()

def get_inst_aware_smooth_loss(disp, img, gds_weight, inst_mask, bd_mask):
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
    grad_img_y[inst_mask[:, :, :-1, :] == 1] = 0

    gravity_mask_y = gds_weight * inst_mask[:, :, :-1, :] + 1 * (1 - inst_mask[:, :, :-1, :])
    gravity_mask_y *= bd_mask[:, :, :-1, :]
    grad_disp_y *= gravity_mask_y
    grad_disp_y *= torch.exp(-grad_img_y)

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    img_x = img * (1 - inst_mask + bd_mask)
    grad_img_x = torch.mean(torch.abs(img_x[:, :, :, :-1] - img_x[:, :, :, 1:]), 1, keepdim=True)
    grad_disp_x *= torch.exp(-grad_img_x)

    return grad_disp_y.mean() + grad_disp_x.mean()

def get_simple_inst_aware_smooth_loss(disp, img, gds_weight, inst_mask):
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
    grad_img_y[inst_mask[:, :, :-1, :] == 1] = 0

    gravity_mask_y = gds_weight * inst_mask[:, :, :-1, :] + 1 * (1 - inst_mask[:, :, :-1, :])
    grad_disp_y *= gravity_mask_y
    grad_disp_y *= torch.exp(-grad_img_y)

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_disp_x *= torch.exp(-grad_img_x)

    return grad_disp_y.mean() + grad_disp_x.mean()


def get_inst_aware_smooth_loss_2(disp, img, map, bd_x, bd_y):
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
    grad_img_y[map[:, :-1, :].unsqueeze(1) == 1] = 0

    gravity_mask_y = 1000 * map[:, :-1, :] + 1 * (1 - map[:, :-1, :])
    gravity_mask_y *= bd_y
    grad_disp_y *= gravity_mask_y.unsqueeze(1)

    grad_disp_y *= torch.exp(-grad_img_y)

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_x[map[:, :, -1].unsqueeze(1) == 1] = 0
    grad_disp_x *= bd_x.unsqueeze(1)
    grad_disp_x *= torch.exp(-grad_img_x)

    return grad_disp_y.mean() + grad_disp_x.mean()


def get_inst_aware_smooth_loss_3(disp, img, map, bd_x, bd_y):
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
    grad_img_y[map[:, :-1, :].unsqueeze(1) == 1] = 0

    gravity_mask_y = 1000 * map[:, :-1, :] + 1 * (1 - map[:, :-1, :])
    gravity_mask_y *= bd_y
    grad_disp_y *= gravity_mask_y.unsqueeze(1)

    grad_disp_y *= torch.exp(-grad_img_y)

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_x[map[:, :, -1].unsqueeze(1) == 1] = 0
    grad_disp_x *= bd_x.unsqueeze(1)
    grad_disp_x *= torch.exp(-grad_img_x)

    return grad_disp_y.mean() + grad_disp_x.mean()

def get_unedge_smooth_loss(disp):

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    return grad_disp_x.mean() + grad_disp_y.mean()

def get_mask_smooth_loss(disp, img, mask):

    disp *= mask
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    loss_grad_disp_x = grad_disp_x.sum() / (mask.sum() + 1e-5)
    loss_grad_disp_y = grad_disp_y.sum() / (mask.sum() + 1e-5)

    return loss_grad_disp_x + loss_grad_disp_y

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def get_normals_smooth_loss(normals, img, mask):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(normals[:, :, :, :-1] - normals[:, :, :, 1:])
    grad_disp_y = torch.abs(normals[:, :, :-1, :] - normals[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    grad_disp_x = F.pad(grad_disp_x, (0, 1, 0, 0))
    grad_disp_y = F.pad(grad_disp_y, (0, 0, 0, 1))

    loss_grad_disp_x = (grad_disp_x * mask).sum() / (mask.sum() + 1e-5)
    loss_grad_disp_y = (grad_disp_y * mask).sum() / (mask.sum() + 1e-5)

    return loss_grad_disp_x + loss_grad_disp_y

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


class BackprojectDepthSR(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepthSR, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords),
            requires_grad=False)

        self.ones = nn.Parameter(
            torch.ones(self.batch_size, 1, self.height * self.width),
                       requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(
            torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def forward(self, depth, inv_K):
        b = depth.size(0)
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords[:b])
        cam_points = depth.view(b, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones[:b]], 1).reshape(b, 4, self.height, self.width)

        return cam_points


class ScaleRecovery(nn.Module):
    """Layer to estimate scale through dense geometrical constrain
    """
    def __init__(self, batch_size, height, width):
        super(ScaleRecovery, self).__init__()
        self.backproject_depth = BackprojectDepthSR(batch_size, height, width)
        self.batch_size = batch_size
        self.height = height
        self.width = width

    # derived from https://github.com/zhenheny/LEGO
    def get_surface_normal(self, cam_points, nei=1):
        cam_points_ctr  = cam_points[:, :-1, nei:-nei, nei:-nei]
        cam_points_x0   = cam_points[:, :-1, nei:-nei, 0:-(2*nei)]
        cam_points_y0   = cam_points[:, :-1, 0:-(2*nei), nei:-nei]
        cam_points_x1   = cam_points[:, :-1, nei:-nei, 2*nei:]
        cam_points_y1   = cam_points[:, :-1, 2*nei:, nei:-nei]
        cam_points_x0y0 = cam_points[:, :-1, 0:-(2*nei), 0:-(2*nei)]
        cam_points_x0y1 = cam_points[:, :-1, 2*nei:, 0:-(2*nei)]
        cam_points_x1y0 = cam_points[:, :-1, 0:-(2*nei), 2*nei:]
        cam_points_x1y1 = cam_points[:, :-1, 2*nei:, 2*nei:]

        vector_x0   = cam_points_x0   - cam_points_ctr
        vector_y0   = cam_points_y0   - cam_points_ctr
        vector_x1   = cam_points_x1   - cam_points_ctr
        vector_y1   = cam_points_y1   - cam_points_ctr
        vector_x0y0 = cam_points_x0y0 - cam_points_ctr
        vector_x0y1 = cam_points_x0y1 - cam_points_ctr
        vector_x1y0 = cam_points_x1y0 - cam_points_ctr
        vector_x1y1 = cam_points_x1y1 - cam_points_ctr

        normal_0 = F.normalize(torch.cross(vector_x0,   vector_y0,   dim=1), dim=1).unsqueeze(0)
        normal_1 = F.normalize(torch.cross(vector_x1,   vector_y1,   dim=1), dim=1).unsqueeze(0)
        normal_2 = F.normalize(torch.cross(vector_x0y0, vector_x0y1, dim=1), dim=1).unsqueeze(0)
        normal_3 = F.normalize(torch.cross(vector_x1y0, vector_x1y1, dim=1), dim=1).unsqueeze(0)

        normals = torch.cat((normal_0, normal_1, normal_2, normal_3), dim=0).mean(0)
        normals = F.normalize(normals, dim=1)

        refl = nn.ReflectionPad2d(nei)
        normals = refl(normals)

        return normals

    def get_ground_mask(self, cam_points, normal_map, threshold=5):
        b, _, h, w = normal_map.size()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        threshold = math.cos(math.radians(threshold))
        ones, zeros = torch.ones(b, 1, h, w).cuda(), torch.zeros(b, 1, h, w).cuda()
        vertical = torch.cat((zeros, ones, zeros), dim=1)

        cosine_sim = cos(normal_map, vertical).unsqueeze(1)
        vertical_mask = (cosine_sim > threshold) | (cosine_sim < -threshold)

        y = cam_points[:,1,:,:].unsqueeze(1)
        ground_mask = vertical_mask.masked_fill(y <= 0, False)

        return ground_mask

    def forward(self, depth, K, real_cam_height=1):
        inv_K = torch.inverse(K)

        cam_points = self.backproject_depth(depth, inv_K)
        surface_normal = self.get_surface_normal(cam_points.detach())
        ground_mask = self.get_ground_mask(cam_points.detach(), surface_normal.detach())

        cam_heights = (cam_points[:,:-1,:,:] * surface_normal.detach()).sum(1).abs().unsqueeze(1)
        cam_heights_masked = torch.masked_select(cam_heights, ground_mask.detach())
        cam_height = torch.median(cam_heights_masked).unsqueeze(0)
        # cam_height = (cam_heights * ground_mask).sum() / (ground_mask.sum() + 1e-7)
        scale = torch.reciprocal(cam_height).mul_(real_cam_height)
        return scale
