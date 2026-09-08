#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from submodules.lpipsPyTorch import LPIPS
import json,logging
from tqdm import tqdm
from argparse import ArgumentParser
import torch.nn.functional as F
from torch.autograd import Variable

from math import exp
def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(os.path.join(renders_dir,fname))#renders_dir / fname
        gt = Image.open(os.path.join(gt_dir,fname))#gt_dir / fname
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    lpips_criterion = LPIPS('vgg', '0.1').cuda()

    for test_dir in model_paths:
        print("test_dir:", test_dir)
        full_dict = {}
        per_view_dict = {}
        full_dict_polytopeonly = {}
        per_view_dict_polytopeonly = {}
        print("")
        full_dict[test_dir] = {
            "PSNR": 0.0,
            "SSIM": 0.0,
            "LPIPS": 0.0,
            "MAE": 0.0,
            'frames': 0.0,
            }

        for scene_dir in os.listdir(test_dir):

            scene_path = os.path.join(test_dir, scene_dir)
            if os.path.isdir(scene_path):
                print("scene_dir:", scene_dir)
                logging.info(f"scene_dir:{scene_dir}")

                full_dict[scene_dir] = {}
                per_view_dict[scene_dir] = {}
                full_dict_polytopeonly[scene_dir] = {}
                per_view_dict_polytopeonly[scene_dir] = {}

                scene_base_name = os.path.basename(scene_dir)
                print("scene_name:", scene_base_name)
                logging.info(f"scene_name:{scene_base_name}")

                full_dict[scene_dir][scene_base_name] = {}
                per_view_dict[scene_dir][scene_base_name] = {}
                full_dict_polytopeonly[scene_dir][scene_base_name] = {}
                per_view_dict_polytopeonly[scene_dir][scene_base_name] = {}

                gt_dir = os.path.join(scene_path, "gt")
                renders_dir = os.path.join(scene_path, "render")
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                maes = []
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_criterion(renders[idx], gts[idx]))
                    maes.append(l1_loss(renders[idx], gts[idx]))
                print(f"scene_dir:{scene_dir}")
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean().item(), ".5"))
                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean().item(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean().item(), ".5"))
                print("  MAE: {:>12.7f}".format(torch.tensor(maes).mean().item(), ".5"))
                print("")
                logging.info(f"PSNR:{torch.tensor(psnrs).mean().item()}   SSIM:{torch.tensor(ssims).mean().item()}   LPIPS:{torch.tensor(lpipss).mean().item()}    MAE:{torch.tensor(maes).mean().item()}")

                full_dict[scene_dir][scene_base_name].update({
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item(),
                    "MAE": torch.tensor(maes).mean().item()
                })

                per_view_dict[scene_dir][scene_base_name].update({
                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                    "MAE": {name: mae for mae, name in zip(torch.tensor(maes).tolist(), image_names)}
                })

                with open(os.path.join(scene_path, "results.json"), 'w') as fp:
                    json.dump(full_dict[scene_dir], fp, indent=True)
                with open(os.path.join(scene_path, "per_view.json"), 'w') as fp:
                    json.dump(per_view_dict[scene_dir], fp, indent=True)

                full_dict[test_dir]["PSNR"] += torch.tensor(psnrs).mean().item()*len(renders)
                full_dict[test_dir]["SSIM"] += torch.tensor(ssims).mean().item()*len(renders)
                full_dict[test_dir]["LPIPS"] += torch.tensor(lpipss).mean().item()*len(renders)
                full_dict[test_dir]["MAE"] += torch.tensor(maes).mean().item()*len(renders)
                full_dict[test_dir]["frames"] += len(renders)
                renders.clear()
                gts.clear()
                torch.cuda.empty_cache()

        full_dict[test_dir]["PSNR"] /= full_dict[test_dir]["frames"]
        full_dict[test_dir]["SSIM"] /= full_dict[test_dir]["frames"]
        full_dict[test_dir]["LPIPS"] /= full_dict[test_dir]["frames"]
        full_dict[test_dir]["MAE"] /= full_dict[test_dir]["frames"]
        print(f"----------------test_dir:{test_dir}-----------------")
        print("  PSNR : {:>12.7f}".format(full_dict[test_dir]["PSNR"], ".5"))
        print("  SSIM : {:>12.7f}".format(full_dict[test_dir]["SSIM"], ".5"))
        print("  LPIPS: {:>12.7f}".format(full_dict[test_dir]["LPIPS"], ".5"))
        print("  MAE: {:>12.7f}".format(full_dict[test_dir]["MAE"], ".5"))
        with open(os.path.join(test_dir, "results.json"), 'w') as fp:
            json.dump(full_dict[test_dir], fp, indent=True)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
