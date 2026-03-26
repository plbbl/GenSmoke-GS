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
import json
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, USE_FASTERGS_RASTERIZER
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.gaussian_model import build_scaling_rotation
from plyfile import PlyData, PlyElement
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def save_radius_filtered_ply(src_path, dst_path, radius_max=30.0):
    ply_data = PlyData.read(src_path)
    vertices = ply_data["vertex"].data
    xyz = np.stack((vertices["x"], vertices["y"], vertices["z"]), axis=1)
    radius = np.linalg.norm(xyz, axis=1)
    filtered_vertices = vertices[radius < radius_max]

    filtered_ply = PlyData(
        [PlyElement.describe(filtered_vertices, "vertex")],
        text=ply_data.text,
        byte_order=ply_data.byte_order,
        comments=ply_data.comments,
        obj_info=ply_data.obj_info,
    )
    filtered_ply.write(dst_path)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if dataset.cap_max == -1:
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]

        if USE_FASTERGS_RASTERIZER:
            image, densification_info = render_pkg["render"], render_pkg["densification_info"]
        else:
            image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss = loss + args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
        loss = loss + args.scale_reg * torch.abs(gaussians.get_scaling).mean()


        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}","Num": f"{gaussians._xyz.shape[0]:d}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                point_cloud_dir = os.path.join(scene.model_path, "point_cloud", f"iteration_{iteration}")
                src_ply = os.path.join(point_cloud_dir, "point_cloud.ply")
                dst_ply = os.path.join(point_cloud_dir, "point_cloud_r_lt_30.ply")
                save_radius_filtered_ply(src_ply, dst_ply, radius_max=30.0)

            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                gaussians.relocate_gs(dead_mask=dead_mask)
                gaussians.add_new_gs(cap_max=args.cap_max)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                actual_covariance = L @ L.transpose(1, 2)

                def op_sigmoid(x, k=100, x0=0.995):
                    return 1 / (1 + torch.exp(-k * (x - x0)))

                noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*args.noise_lr*xyz_lr
                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                gaussians._xyz.add_(noise)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            opacity_values = scene.gaussians.get_opacity.detach().view(-1).float().cpu().numpy()
            try:
                tb_writer.add_histogram("scene/opacity_histogram", opacity_values, iteration)
            except TypeError:
                tb_writer.add_scalar("scene/opacity_mean", float(opacity_values.mean()), iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def get_cli_overrides(argv):
    keys = set()
    for i, tok in enumerate(argv):
        if not tok.startswith("--"):
            continue
        key = tok[2:]
        if "=" in key:
            key = key.split("=", 1)[0]
        keys.add(key.replace("-", "_"))
    return keys

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    if args.config is not None:
        config = load_config(args.config)
        cli_keys = get_cli_overrides(sys.argv[1:])
        for key, value in config.items():
            if key not in cli_keys:
                setattr(args, key, value)

    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    print("\nTraining complete.")
