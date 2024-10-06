# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import cv2
import numpy as np
import argparse
from PIL import Image
import time
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from utils import readlines, colormap, error_colormap, load_config
import datasets
import networks
from skimage.morphology import square, dilation
from networks.layers import transformation_from_parameters, disp_to_depth, BackprojectDepth, Project3D
import tqdm

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

STEREO_SCALE_FACTOR = 5.4

def save_errors(opt, idx, gt, pred, mask):
    abs_rel = np.abs(gt - pred) / (gt + 1e-5)
    abs_rel *= mask

    abs_rel = error_colormap(abs_rel)
    mask = mask[:, :, np.newaxis]
    mask = np.concatenate([mask, mask, mask], axis=2)
    abs_rel[mask == 0] = 1

    im = Image.fromarray(np.rint(255 * abs_rel).astype(np.uint8))
    im.save(os.path.join(opt['path']['log_dir'], 'abs_rel', '{:010d}.png'.format(idx)))

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def generate_images_pred(input_data, frames_to_load, output, depth, backproject_depth, project_3d):
    for frame_id in frames_to_load[1:]:
        T = input_data[('relative_pose', frame_id)].cpu()
        cam_points = backproject_depth(depth.cpu(), input_data[("inv_K", 0)])
        pix_coords = project_3d(cam_points, input_data[("K", 0)], T)

        output[("color", frame_id, 0)] = F.grid_sample(input_data[("color", frame_id, 0)],
                                                       pix_coords, padding_mode="border",
                                                       align_corners=True)


def log(opt, input_data, frames_to_load, outputs, i):
    for j in range(opt['system']['batch_size']):
        if j >= input_data[("color", 0, 0)].size(0): return
        idx = i * opt['system']['batch_size'] + j

        if opt['eval']['save_color']:
            im = input_data[("color", 0, 0)][j].cpu().numpy()
            im = Image.fromarray(np.rint(255 * im.transpose(1, 2, 0)).astype(np.uint8))
            im.save(os.path.join(opt['path']['log_dir'], 'color', '{:010d}.png'.format(idx)))

        if opt['eval']['save_disp']:
            disp = colormap(outputs[("disp", 0)][j, 0])
            im = Image.fromarray(np.rint(255 * disp.transpose(1, 2, 0)).astype(np.uint8))
            im.save(os.path.join(opt['path']['log_dir'], 'disp', '{:010d}.png'.format(idx)))


def evaluate(opt):
    start = time.time()
    # Set gpu
    device = torch.device('cuda:{}'.format(opt['system']['gpu']) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device', torch.cuda.current_device())

    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = opt['eval']['min_depth']
    MAX_DEPTH = opt['eval']['max_depth']

    # set number of frames to load
    frames_to_load = [0]
    for idx in range(-1, -1 - opt['eval']['num_matching_frames'], -1):
        if idx not in frames_to_load:
            frames_to_load.append(idx)

    log_dir = os.path.join(opt['path']['log_dir'], opt['loading']['load_weights_folder'].split('/')[2],
                               opt['dataset']['eval_split'])
    opt['path']['log_dir'] = log_dir

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if opt['eval']['save_disp'] and not os.path.exists(os.path.join(log_dir, 'disp')):
        os.makedirs(os.path.join(log_dir, 'disp'))
    if opt['eval']['save_color'] and not os.path.exists(os.path.join(log_dir, 'color')):
        os.makedirs(os.path.join(log_dir, 'color'))
    if opt['eval']['save_error'] and not os.path.exists(os.path.join(log_dir, 'abs_rel')):
        os.makedirs(os.path.join(log_dir, 'abs_rel'))
    if opt['eval']['save_gt_disp'] and not os.path.exists(os.path.join(log_dir, 'gt_disp')):
        os.makedirs(os.path.join(log_dir, 'gt_disp'))

    backproject_depth = {}

    for scale in opt['single_depth_model']['scales']:
        h = opt['eval']['height'] // (2 ** scale)
        w = opt['eval']['width'] // (2 ** scale)

        backproject_depth[scale] = BackprojectDepth(opt['system']['batch_size'], h, w)
        backproject_depth[scale].cuda()

    if opt['eval']['ext_disp_to_eval'] is None:

        load_weights_folder = os.path.expanduser(opt['loading']['load_weights_folder'])

        assert os.path.isdir(load_weights_folder), \
            "Cannot find a folder at {}".format(load_weights_folder)

        print("-> Loading weights from {}".format(load_weights_folder))

        if opt['eval']['eval_mono']:
            encoder_path = os.path.join(opt['loading']['load_weights_folder'], "mono_encoder.pth")
            decoder_path = os.path.join(opt['loading']['load_weights_folder'], "mono_depth.pth")
            print("Eval mono encoder")
            encoder_dict = torch.load(encoder_path, map_location="cuda:{}".format(opt['system']['gpu']))

        try:
            HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
        except KeyError:
            print('No "height" or "width" keys found in the encoder state_dict, resorting to '
                  'using command line values!')
            HEIGHT, WIDTH = opt['eval']['height'], opt['eval']['width']

        try:
            min_depth_bin, max_depth_bin = encoder_dict['min_depth_bin'], encoder_dict['max_depth_bin']
            print("max_depth_bin: {} / min_depth_bin: {}".format(max_depth_bin, min_depth_bin))
        except KeyError:
            print('No "min_depth_bin" or "max_depth_bin" keys found in the encoder state_dict, resorting to '
                  'using command line values!')

        # evaluation dataset
        filenames = readlines(os.path.join(opt['path']['splits_dir'], opt['dataset']['eval_split'],
                                           "test_files.txt"))

        if opt['dataset']['eval_split'] == 'cityscapes':
            dataset = datasets.CityscapesEvalDataset(filenames[:], opt,
                                                     HEIGHT, WIDTH,
                                                     frames_to_load, 4,
                                                     is_train=False)
        else:
            dataset = datasets.KITTIRAWDataset(opt['path']['data_path'], filenames[:],
                                               HEIGHT, WIDTH,
                                               frames_to_load, 4,
                                               is_train=False)
        dataloader = DataLoader(dataset, opt['system']['batch_size'], shuffle=False,
                                num_workers=opt['system']['num_workers'],
                                pin_memory=True, drop_last=False)

        # setup models
        if opt['eval']['eval_mono']:
            if opt['single_depth_model']['encoder_type'] == 'Resnet':
                encoder = networks.ResnetEncoder(num_layers=opt['single_depth_model']['num_layers'],
                                                 pretrained=True)
            elif opt['single_depth_model']['encoder_type'] == 'ViT':
                encoder = networks.mpvit_small()
                encoder.num_ch_enc = [64, 128, 216, 288, 288]
            else:
                print("Wrong single frame depth model name !!")

            if opt['single_depth_model']['decoder_type'] == 'base':
                depth_decoder = networks.DepthDecoder(encoder.num_ch_enc,
                                                      opt=opt['single_depth_model'],
                                                      scales=opt['single_depth_model']['scales'])
            elif opt['single_depth_model']['decoder_type'] == 'ViT':
                depth_decoder = networks.DepthDecoderViT(
                                    ch_enc=encoder.num_ch_enc,
                                    opt=opt['single_depth_model'],
                                    backproject_depth=backproject_depth
                                )
            else:
                print("Wrong single frame depth model name !!")

            model_dict = encoder.state_dict()
            encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

            decoder_dict = torch.load(decoder_path, map_location="cuda:{}".format(opt['system']['gpu']))
            mono_model_dict = depth_decoder.state_dict()
            depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in mono_model_dict})

            encoder.eval()
            depth_decoder.eval()

            if torch.cuda.is_available():
                encoder.cuda()
                depth_decoder.cuda()

        pred_disps = []
        if opt['eval']['eval_object']:
            object_masks = []

        print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))

        # do inference
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(dataloader)):
                input_color = data[('color', 0, 0)]
                if torch.cuda.is_available():
                    input_color = input_color.cuda()
                for key, ipt in data.items():
                    if isinstance(ipt, list):
                        continue
                    data[key] = ipt.cuda()

                if opt['dataset']['eval_split'] == 'cityscapes':
                    if opt['eval']['eval_object']:
                        for bi in range(input_color.size(0)):
                            object_masks.append(data["doj_mask"][bi])
                elif opt['eval']['eval_object']:
                    for bi in range(input_color.size(0)):
                        object_masks.append(data[("instance_map", 0)][bi].cpu().numpy())

                mono_feats = encoder(input_color)
                output = depth_decoder(mono_feats)

                pred_disp, pred_depth = disp_to_depth(output[("disp", 0)], opt['eval']['min_depth'],
                                                      opt['eval']['max_depth'])
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)

                if opt['eval']['save']:
                    log(opt, data, frames_to_load, output, i)

        pred_disps = np.concatenate(pred_disps)

        print('finished predicting!')

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt['eval']['ext_disp_to_eval']))
        pred_disps = np.load(opt['eval']['ext_disp_to_eval'])

        if opt['eval']['eval_eigen_to_benchmark']:
            eigen_to_benchmark_ids = np.load(
                os.path.join(opt['path']['splits_dir'], "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt['dataset']['eval_split'] == 'cityscapes':
        print('loading cityscapes gt depths individually due to their combined size!')
        gt_depths = os.path.join(opt['path']['splits_dir'], opt['dataset']['eval_split'], "gt_depths")
    else:
        gt_path = os.path.join(opt['path']['splits_dir'], opt['dataset']['eval_split'], "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")
    print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    if opt['eval']['eval_object']:
        object_errors = []
    for i in tqdm.tqdm(range(pred_disps.shape[0])):

        if opt['dataset']['eval_split'] == 'cityscapes':
            gt_id = i
            gt_depth = np.load(os.path.join(gt_depths, str(gt_id).zfill(3) + '_depth.npy'))
            gt_height, gt_width = gt_depth.shape[:2]
            # crop ground truth to remove ego car -> this has happened in the dataloader for input
            # images
            gt_height = int(round(gt_height * 0.75))
            gt_depth = gt_depth[:gt_height]

        else:
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = np.squeeze(pred_disps[i])
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt['dataset']['eval_split'] == 'cityscapes':
            # when evaluating cityscapes, we centre crop to the middle 50% of the image.
            # Bottom 25% has already been removed - so crop the sides and the top here
            gt_depth = gt_depth[256:, 192:1856]
            pred_depth = pred_depth[256:, 192:1856]

        if opt['dataset']['eval_split'] == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
            object_mask = np.squeeze(object_masks[i])
            object_mask = cv2.resize(object_mask, (gt_width, gt_height))

        elif opt['dataset']['eval_split'] == 'cityscapes':
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            object_mask = F.interpolate(object_masks[i].unsqueeze(0), [gt_height, gt_width])
            object_mask = object_mask[0][0][256:, 192:1856].cpu().numpy()

        else:
            mask = gt_depth > 0


        pred_depth *= opt['eval']['pred_depth_scale_factor']
        if not opt['eval']['disable_median_scaling']:
            ratio = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth[mask], pred_depth[mask]))

        if opt['eval']['eval_object']:
            doj_mask = np.logical_and(mask, object_mask)
            if doj_mask.sum() == 0:
                continue
            object_errors.append(compute_errors(gt_depth[doj_mask], pred_depth[doj_mask]))

        if opt['eval']['save_error']:
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            save_errors(opt, i, gt_depth, pred_depth, mask)

        if opt['eval']['save_gt_disp']:
            gt_disp = 1 / (gt_depth + 1e-4)
            gt_depth_mask = np.ones_like(gt_depth)
            gt_depth_mask[gt_depth == 0] = 0
            gt_disp[gt_depth == 0] = 0
            disp = colormap(gt_disp)
            gt_depth_mask = gt_depth_mask[np.newaxis]
            disp = disp * gt_depth_mask + 1 * (1 - gt_depth_mask)
            im = Image.fromarray(np.rint(255 * disp.transpose(1, 2, 0)).astype(np.uint8))
            im.save(os.path.join(opt['path']['log_dir'], 'gt_disp', str(i).zfill(3) + '.png'))

    if not opt['eval']['disable_median_scaling']:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\nMetrics on whole image region\n " + ("{:>8} | " * 7).format("abs_rel",
                                           "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

    if opt['eval']['eval_object']:
        mean_errors = np.array(object_errors).mean(0)

        print("\nMetrics on dynamic object region\n  " + ("{:>8} | " * 7).format("abs_rel",
                                                        "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

    print("Time cost for test: {}".format(time.time() - start))
    print("\n-> Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file.')
    args = parser.parse_args()
    cfg = load_config(args.config, './configs/default.yaml')

    evaluate(cfg)
