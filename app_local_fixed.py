import gc
import os
import uuid
from pathlib import Path

# Enable CPU fallback for MPS operations not yet implemented
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# Disable memory limit to allow full GPU usage
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
# Try to limit allocations
os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'

import numpy as np
import gradio as gr
import torch
from decord import cpu, VideoReader
from diffusers.training_utils import set_seed
import torch.nn.functional as F
import imageio
from kornia.filters import canny
from kornia.morphology import dilation

from third_party import MoGe
from geometrycrafter import (
    GeometryCrafterDiffPipeline,
    GeometryCrafterDetermPipeline,
    PMapAutoencoderKLTemporalDecoder,
    UNetSpatioTemporalConditionModelVid2vid
)

from utils.glb_utils import pmap_to_glb
from utils.disp_utils import pmap_to_disp

examples = [
    # process_length: int,
    # max_res: int,
    # num_inference_steps: int,
    # guidance_scale: float,
    # window_size: int,
    # decode_chunk_size: int,
    # overlap: int,
    ["examples/video1.mp4", 30, 512, 5, 1.0, 30, 4, 10],  # Reduced settings
    ["examples/video2.mp4", 30, 512, 5, 1.0, 30, 4, 10],
    ["examples/video3.mp4", 30, 512, 5, 1.0, 30, 4, 10],
    ["examples/video4.mp4", 30, 512, 5, 1.0, 30, 4, 10],
]

# Global variables to cache models
_models_cache = {
    'initialized': False,
    'pipe': None,
    'point_map_vae': None,
    'prior_model': None,
    'unet': None
}

def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    torch.mps.empty_cache()
    torch.mps.synchronize()

def initialize_models():
    """Initialize models lazily with memory management"""
    if _models_cache['initialized']:
        return _models_cache['pipe'], _models_cache['point_map_vae'], _models_cache['prior_model']
    
    print("Initializing models with memory optimization...")
    cleanup_memory()
    
    model_type = 'determ'  # Use deterministic model for lower memory
    cache_dir = 'workspace/cache'
    
    try:
        # Load UNet
        print("Loading UNet...")
        unet = UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
            'TencentARC/GeometryCrafter',
            subfolder='unet_determ',  # Use deterministic model
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            cache_dir=cache_dir
        ).requires_grad_(False).to("mps", dtype=torch.float32)
        cleanup_memory()
        
        # Load VAE
        print("Loading Point Map VAE...")
        point_map_vae = PMapAutoencoderKLTemporalDecoder.from_pretrained(
            'TencentARC/GeometryCrafter',
            subfolder='point_map_vae',
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            cache_dir=cache_dir
        ).requires_grad_(False).to("mps", dtype=torch.float32)
        cleanup_memory()
        
        # Load MoGe
        print("Loading MoGe prior model...")
        prior_model = MoGe(
            cache_dir=cache_dir,
        ).requires_grad_(False).to('mps', dtype=torch.float32)
        cleanup_memory()
        
        # Create pipeline with error handling for large VAE
        print("Creating pipeline...")
        
        try:
            # Try to load the full pipeline
            pipe = GeometryCrafterDetermPipeline.from_pretrained(
                'stabilityai/stable-video-diffusion-img2vid-xt',
                unet=unet,
                torch_dtype=torch.float32,
                variant="fp16",
                cache_dir=cache_dir,
                low_cpu_mem_usage=True
            ).to("mps")
        except Exception as e:
            print(f"Failed to load full pipeline: {e}")
            print("Trying alternative initialization...")
            # If it fails, load with CPU first then move to MPS
            pipe = GeometryCrafterDetermPipeline.from_pretrained(
                'stabilityai/stable-video-diffusion-img2vid-xt',
                unet=unet,
                torch_dtype=torch.float32,
                variant="fp16",
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
                device_map="cpu"  # Load on CPU first
            )
            # Move components to MPS selectively
            pipe.unet = pipe.unet.to("mps")
            if hasattr(pipe, 'vae') and pipe.vae is not None:
                pipe.vae = pipe.vae.to("mps")
            pipe = pipe.to("mps")
        
        # Enable memory optimizations
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"Xformers not available: {e}")
        
        pipe.enable_attention_slicing(slice_size=1)  # Most aggressive slicing
        
        # Cache models
        _models_cache['initialized'] = True
        _models_cache['pipe'] = pipe
        _models_cache['point_map_vae'] = point_map_vae
        _models_cache['prior_model'] = prior_model
        _models_cache['unet'] = unet
        
        cleanup_memory()
        print("Models initialized successfully")
        
        return pipe, point_map_vae, prior_model
        
    except Exception as e:
        print(f"Error initializing models: {e}")
        cleanup_memory()
        raise

def read_video_frames(video_path, process_length, max_res):
    print("==> processing video: ", video_path)
    vid = VideoReader(video_path, ctx=cpu(0))
    fps = vid.get_avg_fps()
    print("==> original video shape: ", (len(vid), *vid.get_batch([0]).shape[1:]))
    original_height, original_width = vid.get_batch([0]).shape[1:3]
    
    # Enforce maximum resolution
    max_res = min(max_res, 512)  # Cap at 512 for memory
    
    if max(original_height, original_width) > max_res:
        scale = max_res / max(original_height, original_width)
        original_height, original_width = round(original_height * scale), round(original_width * scale)
    else:
        scale = 1.0
    
    height = round(original_height * scale / 64) * 64
    width = round(original_width * scale / 64) * 64
    vid = VideoReader(video_path, ctx=cpu(0), width=original_width, height=original_height)
    
    # Limit process length
    process_length = min(process_length, 30, len(vid))  # Cap at 30 frames
    
    frames_idx = list(range(0, process_length))
    print(f"==> final processing shape: {len(frames_idx), height, width}")
    
    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0
    return frames, height, width, fps

def compute_edge_mask(depth: torch.Tensor, edge_dilation_radius: int):
    magnitude, edges = canny(depth[None, None, :, :], low_threshold=0.4, high_threshold=0.5)
    magnitude = magnitude[0, 0]
    edges = edges[0, 0]
    mask = (edges > 0).float()
    mask = dilation(mask[None, None, :, :], torch.ones((edge_dilation_radius,edge_dilation_radius), device=mask.device))
    return mask[0, 0] > 0.5

@torch.inference_mode()
def infer_geometry(
    video: str,
    process_length: int,
    max_res: int,
    num_inference_steps: int,
    guidance_scale: float,
    window_size: int,
    decode_chunk_size: int,
    overlap: int,
    downsample_ratio: float = 1.0,
    remove_edge: bool = True,
    save_folder: str = os.path.join('workspace', 'GeometryCrafterApp'),
):
    try:
        # Initialize models lazily
        pipe, point_map_vae, prior_model = initialize_models()
        
        run_id = str(uuid.uuid4())
        set_seed(42)
        
        # Apply memory constraints
        max_res = min(max_res, 512)
        process_length = min(process_length, 30)
        window_size = min(window_size, 30)
        decode_chunk_size = min(decode_chunk_size, 4)
        
        print(f"Processing with: max_res={max_res}, frames={process_length}, window={window_size}, chunk={decode_chunk_size}")

        frames, height, width, fps = read_video_frames(video, process_length, max_res)
        aspect_ratio = width / height
        assert 0.5 <= aspect_ratio and aspect_ratio <= 2.0
        
        cleanup_memory()
        
        frames_tensor = torch.tensor(frames.astype("float32"), device='mps').float().permute(0, 3, 1, 2)
        window_size = min(window_size, len(frames))
        if window_size == len(frames): 
            overlap = 0

        print("Running inference...")
        point_maps, valid_masks = pipe(
            frames_tensor,
            point_map_vae,
            prior_model,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            window_size=window_size,
            decode_chunk_size=decode_chunk_size,
            overlap=overlap,
            force_projection=True,
            force_fixed_focal=True,
            low_memory_usage=True  # Enable low memory mode
        )
        
        # Move to CPU immediately
        frames_tensor = frames_tensor.cpu()
        point_maps = point_maps.cpu()
        valid_masks = valid_masks.cpu()

        os.makedirs(save_folder, exist_ok=True)

        cleanup_memory()
        
        output_npz_path = Path(save_folder, run_id, f'point_maps.npz')
        output_npz_path.parent.mkdir(exist_ok=True)
            
        np.savez_compressed(
            output_npz_path,
            point_map=point_maps.cpu().numpy().astype(np.float16),
            mask=valid_masks.cpu().numpy().astype(np.bool_)
        )

        output_disp_path = Path(save_folder, run_id, f'disp.mp4')
        output_disp_path.parent.mkdir(exist_ok=True)
            
        colored_disp = pmap_to_disp(point_maps, valid_masks)
        imageio.mimsave(
            output_disp_path, (colored_disp*255).cpu().numpy().astype(np.uint8), fps=fps, macro_block_size=1)

        # downsample for visualization
        if downsample_ratio > 1.0:
            H, W = point_maps.shape[1:3]
            H, W = round(H / downsample_ratio), round(W / downsample_ratio)
            point_maps = F.interpolate(point_maps.permute(0,3,1,2), (H, W)).permute(0,2,3,1)
            frames = F.interpolate(frames_tensor.cpu(), (H, W)).permute(0,2,3,1)
            valid_masks = F.interpolate(valid_masks.float()[:, None], (H, W))[:, 0] > 0.5
        else:
            H, W = point_maps.shape[1:3]
            frames = frames_tensor.cpu().permute(0,2,3,1)
        
        if remove_edge:
            for i in range(len(valid_masks)):
                edge_mask = compute_edge_mask(point_maps[i, :, :, 2], 3)
                valid_masks[i] = valid_masks[i] & (~edge_mask)

        indices = np.linspace(0, len(point_maps)-1, 6)
        indices = np.round(indices).astype(np.int32)

        mesh_seqs, frame_seqs = [], []

        for index in indices:
            valid_mask = valid_masks[index].cpu().numpy()
            point_map = point_maps[index].cpu().numpy()
            frame = frames[index].cpu().numpy()    
            output_glb_path = Path(save_folder, run_id, f'{index:04}.glb')
            output_glb_path.parent.mkdir(exist_ok=True)
            glbscene = pmap_to_glb(point_map, valid_mask, frame)
            glbscene.export(file_obj=output_glb_path)
            mesh_seqs.append(output_glb_path)
            frame_seqs.append(index)
        
        cleanup_memory()
        
        return [
            gr.Model3D(value=mesh_seqs[idx], label=f"Frame: {frame_seqs[idx]}") for idx in range(len(frame_seqs))
        ] + [ 
            gr.Video(value=output_disp_path, label="Disparity", interactive=False), 
            gr.DownloadButton("Download Npz File", value=output_npz_path)
        ]
    except Exception as e:
        cleanup_memory()
        print(f"Error in inference: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(str(e))

def build_demo():
    with gr.Blocks(analytics_enabled=False) as gradio_demo:
        gr.HTML(
            """
            <div align='center'> 
                <h1> GeometryCrafter: Memory-Optimized Version for MPS </h1> 
                <p style='font-size:16px'>Running with reduced settings for Apple Silicon compatibility</p>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_video = gr.Video(
                    label="Input Video",
                    sources=['upload']
                )
                with gr.Row(equal_height=False):
                    with gr.Accordion("Settings (Optimized for MPS)", open=True):
                        process_length = gr.Slider(
                            label="process length (frames)",
                            minimum=10,
                            maximum=30,
                            value=20,
                            step=1,
                        )
                        max_res = gr.Slider(
                            label="max resolution",
                            minimum=256,
                            maximum=512,
                            value=384,
                            step=64,
                        )
                        num_denoising_steps = gr.Slider(
                            label="num denoising steps",
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                        )
                        guidance_scale = gr.Slider(
                            label="cfg scale",
                            minimum=1.0,
                            maximum=1.2,
                            value=1.0,
                            step=0.1,
                        )
                        window_size = gr.Slider(
                            label="window size",
                            minimum=10,
                            maximum=30,
                            value=20,
                            step=5,
                        )
                        decode_chunk_size = gr.Slider(
                            label="decode chunk size",
                            minimum=1,
                            maximum=4,
                            value=2,
                            step=1,
                        )
                        overlap = gr.Slider(
                            label="overlap",
                            minimum=0,
                            maximum=10,
                            value=5,
                            step=1,
                        )
                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                output_disp_video = gr.Video(
                    label="Disparity",
                    interactive=False
                )
                download_btn = gr.DownloadButton("Download Npz File")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                output_point_map0 = gr.Model3D(
                    label="Point Map 0",
                    clear_color=[1.0, 1.0, 1.0, 1.0],
                    interactive=False
                )
            with gr.Column(scale=1):
                output_point_map1 = gr.Model3D(
                    label="Point Map 1",
                    clear_color=[1.0, 1.0, 1.0, 1.0],
                    interactive=False
                )
            with gr.Column(scale=1):
                output_point_map2 = gr.Model3D(
                    label="Point Map 2",
                    clear_color=[1.0, 1.0, 1.0, 1.0],
                    interactive=False
                )
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                output_point_map3 = gr.Model3D(
                    label="Point Map 3",
                    clear_color=[1.0, 1.0, 1.0, 1.0],
                    interactive=False
                )
            with gr.Column(scale=1):
                output_point_map4 = gr.Model3D(
                    label="Point Map 4",
                    clear_color=[1.0, 1.0, 1.0, 1.0],
                    interactive=False
                )
            with gr.Column(scale=1):
                output_point_map5 = gr.Model3D(
                    label="Point Map 5",
                    clear_color=[1.0, 1.0, 1.0, 1.0],
                    interactive=False
                )
        
        gr.Examples(
            examples=examples,
            fn=infer_geometry,
            inputs=[
                input_video,
                process_length,
                max_res,
                num_denoising_steps,
                guidance_scale,
                window_size,
                decode_chunk_size,
                overlap,
            ],
            outputs=[
                output_point_map0, output_point_map1, output_point_map2,
                output_point_map3, output_point_map4, output_point_map5,
                output_disp_video, download_btn
            ],            
        )

        generate_btn.click(
            fn=infer_geometry,
            inputs=[
                input_video,
                process_length,
                max_res,
                num_denoising_steps,
                guidance_scale,
                window_size,
                decode_chunk_size,
                overlap,
            ],
            outputs=[
                output_point_map0, output_point_map1, output_point_map2,
                output_point_map3, output_point_map4, output_point_map5,
                output_disp_video, download_btn
            ],
        )

    return gradio_demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=False)