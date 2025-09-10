import os
import math
import random
import time
import torch
import numpy as np
from einops import rearrange

from typing import Any
from PIL import Image
from torchvision.utils import save_image
from typing import List, Optional, Any, cast

from .spuv.ops import get_projection_matrix, get_mvp_matrix 
from .spuv.camera import get_c2w 
from .spuv.mesh_utils import load_mesh_only, vertex_transform 
from .spuv.nvdiffrast_utils import render_xyz_from_mesh, rasterize_geometry_maps, render_normal_from_mesh 
from .spuv.rasterize import NVDiffRasterizerContext 
from .model.utils.feature_baking import bake_image_feature_to_uv
from .pipeline.weighter import Weighter 
from .pipeline.outpainter import OutpainterPipe
from .utils.video import render_video 
from .utils.misc import process_image 
from .utils.pipe import mv_sync_cfg_generation 
from .utils.voronoi import voronoi_solve
from .utils.renderer import position_to_depth, normalize_depth, generate_ray_image, rotate_c2w

from pathlib import Path


# utility
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class FlexPainterNode:
    """
    ComfyUI Node: FlexPainter
    Generates textures for a mesh using arbitrary ComfyUI models (SDXL, Flux, Krea, etc.)
    Accepts positive/negative embeddings like KSampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
   
                "mesh_path": ("STRING", {"default": ""}),
               
                "render_azim": ("FLOAT", {"default": -1, "min": -100.0, "max": 100.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgb_image",)
    FUNCTION = "run"
    CATEGORY = "FlexPainter"

    def run(self, mesh_path: str,render_azim: float):

        device = "cuda"
        result_root = "./flex_results"

        # Rasterizer context
        ctx = NVDiffRasterizerContext("cuda", device)

        # Camera setup
        camera_poses = [(15.0, 0.0), (15.0, 90.0), (15.0, 180.0), (15.0, 270)]
        camera_dist_scalar = 20 / 9
        fovy = math.radians(30)

        # Load and transform mesh
        mesh = load_mesh_only(mesh_path, device)
        mesh = vertex_transform(mesh, mesh_scale=0.5)

        heles = torch.tensor([pose[0] for pose in camera_poses], device=device)
        if render_azim < 0 or render_azim >= 360:
            render_azim = random.uniform(0, 360)
        azims = torch.tensor([(pose[1] + render_azim) % 360 for pose in camera_poses], device=device)
        camera_dist = torch.tensor(camera_dist_scalar, device=device).repeat(len(heles))
        c2w = get_c2w(azims, heles, camera_dist)
        proj = get_projection_matrix(fovy, 1, 0.1, 1000.0).to(device)
        mvp = get_mvp_matrix(c2w, proj)

        # Render depth maps
        resolution = 512
        texture_size = 1024
        #uv_position, uv_normals, uv_mask = rasterize_geometry_maps(ctx, mesh, texture_size, texture_size)
        xyz, mask = render_xyz_from_mesh(ctx, mesh, mvp, resolution, resolution)

        depth = position_to_depth(xyz, c2w)
        inv_depth = normalize_depth(depth, mask).permute(0, 3, 1, 2)  # (views, C, H, W)

        # Save 2x2 multi-view depth images
        mesh_dir_name = os.path.splitext(os.path.basename(mesh_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(result_root, mesh_dir_name, timestamp)
        ensure_dir(result_dir)

        depth_paths = []
        for i, d in enumerate(inv_depth):
            # Convert to 0-255 grayscale for saving
            d_img = (d[0].cpu().numpy() * 255.0).astype("uint8")
            img = Image.fromarray(d_img).convert("L")
            out_path = os.path.join(result_dir, f"depth_{i}.png")
            img.save(out_path)
            depth_paths.append(out_path)

        # Optional: return 2x2 grid path
        mv_imgs = [Image.open(p).convert("L") for p in depth_paths]
        w, h = mv_imgs[0].size
        grid = Image.new("L", (2 * w, 2 * h), color=0)
        for img, pos in zip(mv_imgs, [(0,0),(w,0),(0,h),(w,h)]):
            grid.paste(img.resize((w,h)), pos)
        mv_grid_path = os.path.join(result_dir, "mv_depth_grid.png")
        grid.save(mv_grid_path)

        # Return depth paths (grid first, then individual views)
        import torchvision.transforms as T
        to_tensor = T.ToTensor()
        grid_tensor = to_tensor(grid).to(torch.float32)  # shape: [1, H, W]
        grid_tensor_rgb = grid_tensor.unsqueeze(0)  # shape [1, 3, H, W]
        return grid_tensor_rgb
    
class ContinueFromRGBNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb_image": ("IMAGE",),           # the RGB output from Node 2
                "mesh_path": ("STRING", {"default": ""}),
                "positive": ("STRING", {"default": ""}),
                "render_azim": ("FLOAT", {"default": -1, "min": -100.0, "max": 100.0}),
                "sample_steps": ("INT", {"default": 30}),
                "resolution": ("INT", {"default": 512}),
                "texture_size": ("INT", {"default": 1024}),
                "render_ele": ("FLOAT", {"default": 15.0}),
                "frame_num": ("INT", {"default": 90}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("mv_gallery", "uv_pred", "uv_paint", "video_path")
    FUNCTION = "run"
    CATEGORY = "FlexPainter"

    def run(self, rgb_image, mesh_path: str, positive: str, render_azim: float, sample_steps: int, resolution: int, texture_size: int, render_ele:float, frame_num:int):

        device = "cuda"
        dtype = torch.float16

        # Assuming this script is somewhere inside the ComfyUI repo
        COMFYUI_ROOT = Path(__file__).parent.parent.parent  # adjust if script is deeper inside folders
        MODELS_DIR = COMFYUI_ROOT / "models"

        #generator = torch.Generator(device=device).manual_seed(1)

        # Working params
        result_root = "./flex_results"
        #sample_steps = 30
        #resolution = 512
        #texture_size = 1024
        #mixing_step = 10
        #image_strength = 0.3
        #frame_num = 90
        #render_ele = 15.0

        # Rasterizer
        ctx = NVDiffRasterizerContext("cuda", device)

        # Camera setup
        camera_poses = [(15.0, 0.0), (15.0, 90.0), (15.0, 180.0), (15.0, 270)]
        camera_dist_scalar = 20 / 9
        fovy = math.radians(30)

        # Load mesh
        mesh = load_mesh_only(mesh_path, device)
        mesh = vertex_transform(mesh, mesh_scale=0.5)

        heles = torch.tensor([pose[0] for pose in camera_poses], device=device)
        if render_azim < 0 or render_azim >= 360:
            render_azim = random.uniform(0, 360)
        azims = torch.tensor([(pose[1] + render_azim) % 360 for pose in camera_poses], device=device)
        camera_dist = torch.tensor(camera_dist_scalar, device=device).repeat(len(heles))
        c2w = get_c2w(azims, heles, camera_dist)
        proj = get_projection_matrix(fovy, 1, 0.1, 1000.0).to(device)
        mvp = get_mvp_matrix(c2w, proj)

        # Weighter + Outpainter
        weighter = Weighter(texture_size, resolution, device)

        weighter_model = os.path.join(MODELS_DIR, "FlexPainter", "weighternet", "model.safetensors")
        weighter.load_weights(weighter_model)



        weighter = cast(Any, weighter)
        assert weighter is not None, 'Weighter not initialized'

        outpainter_model = os.path.join(MODELS_DIR, "FlexPainter", "outpainter", "texgen_v1.ckpt")
        outpainter = OutpainterPipe(device, dtype)
        outpainter.load_weights(outpainter_model)
        
        # Result dir
        mesh_dir_name = os.path.splitext(os.path.basename(mesh_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(result_root, mesh_dir_name, timestamp)
        ensure_dir(result_dir)

        with torch.no_grad():
            # Geometry maps
            uv_position, uv_normals, uv_mask = rasterize_geometry_maps(ctx, mesh, texture_size, texture_size)
            xyz, mask = render_xyz_from_mesh(ctx, mesh, mvp, resolution, resolution)

            depth = None  # Depth already used in previous node
            inv_depth = None  # Not needed here

            renderer = {"ctx": ctx, "mesh": mesh, "mvps": mvp}
            weighter.texture_size = texture_size
            weighter.render_size = resolution
            weighter.device = device
            weighter.preprocess(renderer)
            # If batch dimension exists, squeeze it




            # If PIL, convert to tensor
            if isinstance(rgb_image, Image.Image):
                rgb_tensor = torch.tensor(np.array(rgb_image)/255.0, dtype=torch.float32).permute(2,0,1).unsqueeze(0)  # [1, C, H, W]
            else:
                rgb_tensor = rgb_image.unsqueeze(0) if rgb_image.ndim == 3 else rgb_image  # [1, C, H, W]

            rgb_tensor = rgb_tensor.to(device)

            # rgb_tensor shape: [1, H, W, C]  (from PIL -> np.array -> torch.tensor)
            rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)  # [1, C, H, W]

            # Now split 2x2 grid into 4 separate images
            images = rearrange(rgb_tensor[0], 'c (row h) (col w) -> (row col) c h w', row=2, col=2)
            images = images.to(torch.float32)

            # Save and prepare multi-view images
            mv_paths: list[str] = []
            images_white: list[torch.Tensor] = []

            for i in range(len(images)):
                out_path = os.path.join(result_dir, f'rgb_{i}.png')
                save_image(images[i], out_path)
                images[i] = images[i] * mask[i].permute(2, 0, 1)
                mv_paths.append(out_path)
                img_white = process_image((images[i] * 255.0).cpu().numpy())
                img_white_t = torch.tensor((img_white / 255.0)).permute(2, 0, 1)
                images_white.append(img_white_t)
            images_white = torch.stack(images_white).to(device=device)

            # Build a 2x2 grid image for a single-file download
            try:
                mv_imgs = [Image.open(p).convert('RGB') for p in mv_paths]
                w, h = mv_imgs[0].size
                grid = Image.new('RGB', (2 * w, 2 * h), color=(255, 255, 255))
                positions = [(0, 0), (w, 0), (0, h), (w, h)]
                for img, pos in zip(mv_imgs, positions):
                    grid.paste(img.resize((w, h)), pos)
                mv_grid_path = os.path.join(result_dir, 'mv_grid.png')
                grid.save(mv_grid_path)
            except Exception:
                mv_grid_path = mv_paths[0] if len(mv_paths) > 0 else None
                   
            clear_cuda_cache()

            # Prepare features for baking and weighter
            normal = render_normal_from_mesh(ctx, mesh, mvp, resolution, resolution)
            rays_d = generate_ray_image(mvp, resolution, resolution)
            rays_d = rotate_c2w(rays_d)
            score = torch.sum(normal * rays_d, dim=-1, keepdim=True)
            score = torch.abs(score)

            feature = torch.cat([images.permute(0, 2, 3, 1), rays_d, score, images_white.permute(0, 2, 3, 1)], dim=-1)
            uv_position_, uv_normal, uv_mask_ = rasterize_geometry_maps(ctx, mesh, texture_size, texture_size)
            image_info = {"mvp_mtx": mvp.unsqueeze(0), "rgb": feature.unsqueeze(0)}
            uv_bakes, uv_bake_masks = bake_image_feature_to_uv(ctx, [mesh], image_info, uv_position_)
            uv_bakes = uv_bakes.view(-1, feature.shape[-1], texture_size, texture_size)
            uv_bake_masks = uv_bake_masks.view(-1, 1, texture_size, texture_size)
            uv_bake_mask = uv_bake_masks.sum(dim=0, keepdim=True) > 0

            uv_bakes_white_masks = (uv_bakes[:, 7:] != 0).any(dim=1, keepdim=True).float()
            uv_bake_white_mask = uv_bakes_white_masks.sum(dim=0, keepdim=True) > 0
            final_mask = torch.bitwise_xor(uv_bake_white_mask, uv_bake_mask).float()

            uv_pred = weighter(uv_bakes[:, :3], uv_bake_masks, torch.tensor([0]).to(device))
            uv_position = uv_position_.permute(0, 3, 1, 2)
            uv_mask = uv_mask_.float().permute(0, 3, 1, 2)
            uv_pred_white_bg = uv_pred * final_mask + 1 - final_mask
            uv_pred = uv_pred * final_mask 

            image_final_mask = torch.bitwise_xor(images_white[:, :1].bool(), mask.permute(0, 3, 1, 2).bool()).float()
            images = images * image_final_mask

            # Outpaint to get final texture map
            final_res = outpainter(
                [mesh], positive, images, uv_pred, final_mask, uv_mask, uv_position,
                sample_steps, 3.5, (0.0, 1.0), 0.0
            )
            final_res_white_bg = final_res * uv_mask + 1 - uv_mask
            final_res = voronoi_solve(final_res.squeeze(0).permute(1, 2, 0), uv_mask.squeeze(), device=device)
            final_res = final_res.permute(2, 0, 1).unsqueeze(0)

            # Save UV related outputs
            for i in range(len(images)):
                save_image(uv_bakes[i, :3], os.path.join(result_dir, f'uv_bakes_{i}.png'))
            save_image(uv_pred_white_bg, os.path.join(result_dir, 'uv_pred.png'))
            save_image(final_res, os.path.join(result_dir, 'uv_final_res.png'))
            save_image(final_res_white_bg, os.path.join(result_dir, 'uv_paint.png'))
            save_image(final_mask, os.path.join(result_dir, 'final_mask.png'))

            # Clear CUDA cache after UV prediction
            clear_cuda_cache()

            # Yield Stage 2 partial: uv_pred ready
            uv_pred_path = os.path.join(result_dir, 'uv_pred.png')

            # Render and save video
            video_path = os.path.join(result_dir, 'video.mp4')
            render_video(frame_num, render_ele, camera_dist, fovy, device, ctx, mesh, final_res.squeeze(0), resolution, torch.tensor([1, 1, 1]), video_path)

                # Clear CUDA cache after video rendering
            clear_cuda_cache()

            # Final yield: uv_paint and video ready
            uv_paint_path = os.path.join(result_dir, 'uv_paint.png')

                # Clear CUDA cache after video rendering


            # Final yield: uv_paint and video ready
            uv_paint_path = os.path.join(result_dir, 'uv_paint.png')
       
        return mv_grid_path, uv_pred_path, uv_paint_path, video_path    


NODE_CLASS_MAPPINGS = {
    "FlexPainter": FlexPainterNode,
    "ContinueFromRGBNode": ContinueFromRGBNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlexPainter": "FlexPainter Depth Generator",
    "ContinueFromRGBNode": "FlexPainter Texture Generator",
}
