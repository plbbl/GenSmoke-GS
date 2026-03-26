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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from types import SimpleNamespace
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def _make_zoom_view(view, offset):
    c2w = torch.inverse(view.world_view_transform)
    forward = c2w[2, :3]
    forward = forward / (torch.norm(forward) + 1e-8)

    c2w_zoom = c2w.clone()
    c2w_zoom[3, :3] = c2w[3, :3] + forward * offset
    w2c_zoom = torch.inverse(c2w_zoom)
    full_proj_zoom = (w2c_zoom.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)

    return SimpleNamespace(
        image_width=view.image_width,
        image_height=view.image_height,
        FoVx=view.FoVx,
        FoVy=view.FoVy,
        world_view_transform=w2c_zoom,
        full_proj_transform=full_proj_zoom,
        camera_center=c2w_zoom[3, :3],
        image_name=view.image_name,
    )

def _save_video_mp4(frame_dir, video_path, fps):
    try:
        from torchvision.io import write_video
    except Exception as e:
        print(f"[Warning] Skip mp4 export ({video_path}), torchvision video backend unavailable: {e}")
        return

    frame_files = sorted(
        [f for f in os.listdir(frame_dir) if f.endswith(".png")]
    )
    if len(frame_files) == 0:
        return

    frames = []
    for fname in frame_files:
        frame_path = os.path.join(frame_dir, fname)
        frame = torchvision.io.read_image(frame_path)  # uint8, CxHxW
        frames.append(frame.permute(1, 2, 0))  # HxWxC
    video_tensor = torch.stack(frames, dim=0)  # TxHxWxC uint8
    try:
        write_video(video_path, video_tensor, fps=fps)
    except Exception as e:
        print(f"[Warning] Skip mp4 export ({video_path}), failed to encode video: {e}")

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, zoom_frames, zoom_distance, zoom_sign, zoom_fps, save_mp4):
    render_root = os.path.join(model_path, name, "ours_{}".format(iteration), "zoom_renders")
    video_root = os.path.join(model_path, name, "ours_{}".format(iteration), "zoom_videos")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_root, exist_ok=True)
    makedirs(video_root, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        frame_dir = os.path.join(render_root, "{0:05d}".format(idx))
        makedirs(frame_dir, exist_ok=True)

        for frame_idx in range(zoom_frames):
            alpha = frame_idx / max(zoom_frames - 1, 1)
            offset = zoom_sign * zoom_distance * alpha
            zoom_view = _make_zoom_view(view, offset)
            rendering = render(zoom_view, gaussians, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(frame_dir, '{0:05d}'.format(frame_idx) + ".png"))

        if save_mp4:
            video_path = os.path.join(video_root, "{0:05d}.mp4".format(idx))
            _save_video_mp4(frame_dir, video_path, zoom_fps)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, zoom_frames : int, zoom_distance : float, zoom_sign : float, zoom_fps : int, save_mp4 : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # if not skip_train:
        #      render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, zoom_frames, zoom_distance, zoom_sign, zoom_fps, save_mp4)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--zoom_frames", default=60, type=int)
    parser.add_argument("--zoom_distance", default=0.3, type=float)
    parser.add_argument("--zoom_sign", default=1.0, type=float)
    parser.add_argument("--zoom_fps", default=30, type=int)
    parser.add_argument("--no_save_mp4", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.zoom_frames,
        args.zoom_distance,
        args.zoom_sign,
        args.zoom_fps,
        not args.no_save_mp4,
    )
