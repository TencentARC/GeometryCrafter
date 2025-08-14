import gc
import os
import uuid
from pathlib import Path

# Enable CPU fallback for MPS operations not yet implemented
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

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
    ["examples/video1.mp4", 60, 640, 5, 1.0, 110, 8, 25],
    ["examples/video2.mp4", 60, 640, 5, 1.0, 110, 8, 25],
    ["examples/video3.mp4", 60, 640, 5, 1.0, 110, 8, 25],
    ["examples/video4.mp4", 60, 640, 5, 1.0, 110, 8, 25],
]

model_type = 'diff'
cache_dir = 'workspace/cache'

unet = UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
    'TencentARC/GeometryCrafter',
    subfolder='unet_diff' if model_type == 'diff' else 'unet_determ',
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32,
    cache_dir=cache_dir
).requires_grad_(False).to("mps", dtype=torch.float32)
point_map_vae = PMapAutoencoderKLTemporalDecoder.from_pretrained(
    'TencentARC/GeometryCrafter',
    subfolder='point_map_vae',
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32,
    cache_dir=cache_dir
).requires_grad_(False).to("mps", dtype=torch.float32)
prior_model = MoGe(
    cache_dir=cache_dir,
).requires_grad_(False).to('mps', dtype=torch.float32)
if model_type == 'diff':
    pipe = GeometryCrafterDiffPipeline.from_pretrained(
        'stabilityai/stable-video-diffusion-img2vid-xt',
        unet=unet,
        torch_dtype=torch.float32,
        variant="fp16",
        cache_dir=cache_dir
    ).to("mps")
else:
    pipe = GeometryCrafterDetermPipeline.from_pretrained(
        'stabilityai/stable-video-diffusion-img2vid-xt',
        unet=unet,
        torch_dtype=torch.float32,
        variant="fp16",
        cache_dir=cache_dir
    ).to("mps")

try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception as e:
    print(e)
    print("Xformers is not enabled")
# bugs at https://github.com/continue-revolution/sd-webui-animatediff/issues/101
# pipe.enable_xformers_memory_efficient_attention()
pipe.enable_attention_slicing()

def read_video_frames(video_path, process_length, max_res):
    print("==> processing video: ", video_path)
    vid = VideoReader(video_path, ctx=cpu(0))
    fps = vid.get_avg_fps()
    print("==> original video shape: ", (len(vid), *vid.get_batch([0]).shape[1:]))
    original_height, original_width = vid.get_batch([0]).shape[1:3]
    if max(original_height, original_width) > max_res:
        scale = max_res / max(original_height, original_width)
        original_height, original_width = round(original_height * scale), round(original_width * scale)
    else:
        scale = 1.0
    height = round(original_height * scale / 64) * 64
    width = round(original_width * scale / 64) * 64
    vid = VideoReader(video_path, ctx=cpu(0), width=original_width, height=original_height)
    frames_idx = list(range(0, min(len(vid), process_length) if process_length != -1 else len(vid)))
    print(
        f"==> final processing shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}"
    )
    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0
    return frames, height, width, fps


def compute_edge_mask(depth: torch.Tensor, edge_dilation_radius: int):
    magnitude, edges = canny(depth[None, None, :, :], low_threshold=0.4, high_threshold=0.5)
    magnitude = magnitude[0, 0]
    edges = edges[0, 0]
    mask = (edges > 0).float()
    mask = dilation(mask[None, None, :, :], torch.ones((edge_dilation_radius,edge_dilation_radius), device=mask.device))
    return mask[0, 0] > 0.5

# Removed @spaces.GPU decorator - runs directly on available GPU
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
    downsample_ratio: float = 1.0, # downsample pcd for visualization
    remove_edge: bool = True, # remove edge for visualization
    save_folder: str = os.path.join('workspace', 'GeometryCrafterApp'),
):
    try:
        run_id = str(uuid.uuid4())
        set_seed(42)
        # Xformers is not compatible with MPS backend
        # pipe.enable_xformers_memory_efficient_attention()

        frames, height, width, fps = read_video_frames(video, process_length, max_res)
        aspect_ratio = width / height
        assert 0.5 <= aspect_ratio and aspect_ratio <= 2.0
        frames_tensor = torch.tensor(frames.astype("float32"), device='mps').float().permute(0, 3, 1, 2)
        window_size = min(window_size, len(frames))
        if window_size == len(frames): 
            overlap = 0

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
        )
        frames_tensor = frames_tensor.cpu()
        point_maps = point_maps.cpu()
        valid_masks = valid_masks.cpu()

        os.makedirs(save_folder, exist_ok=True)

        gc.collect()
        torch.mps.empty_cache()
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
            frames = F.interpolate(frames_tensor, (H, W)).permute(0,2,3,1)
            valid_masks = F.interpolate(valid_masks.float()[:, None], (H, W))[:, 0] > 0.5
        else:
            H, W = point_maps.shape[1:3]
            frames = frames_tensor.permute(0,2,3,1)

        
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
        
        
        gc.collect()
        torch.mps.empty_cache()
        
        return [
            gr.Model3D(value=mesh_seqs[idx], label=f"Frame: {frame_seqs[idx]}") for idx in range(len(frame_seqs))
        ] + [ 
            gr.Video(value=output_disp_path, label="Disparity", interactive=False), 
            gr.DownloadButton("Download Npz File", value=output_npz_path)
        ]
    except Exception as e:
        gc.collect()
        torch.mps.empty_cache()
        raise gr.Error(str(e))

def build_demo():
    with gr.Blocks(analytics_enabled=False) as gradio_demo:
        gr.HTML(
            """
            <div align='center'> 
                <h1> GeometryCrafter: Consistent Geometry Estimation for Open-world Videos with Diffusion Priors </h1> \
                <span style='font-size:18px'>\
                    <a href='https://scholar.google.com/citations?user=zHp0rMIAAAAJ'>Tian-Xing Xu</a>, \
                    <a href='https://scholar.google.com/citations?user=qgdesEcAAAAJ'>Xiangjun Gao</a>, \
                    <a href='https://wbhu.github.io'>Wenbo Hu</a>, \
                    <a href='https://xiaoyu258.github.io/'>Xiaoyu Li</a>, \
                    <a href='https://scholar.google.com/citations?user=AWtV-EQAAAAJ'>Song-Hai Zhang</a>,\
                    <a href='https://scholar.google.com/citations?user=4oXBp9UAAAAJ'>Ying Shan</a>\
                </span> \
                <br>
                <br>
                <span style='font-size:18px'>If you find GeometryCrafter useful, please help ⭐ the \
                    <a style='font-size:18px' href='https://github.com/TencentARC/GeometryCrafter/'>[Github Repo]</a>\
                    , which is important to Open-Source projects. Thanks!\
                    <a href='https://arxiv.org/abs/2504.01016'>[arXiv]</a> \
                    <a href='https://geometrycrafter.github.io'>[Project Page]</a> \
                </span> 
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
                    with gr.Accordion("Advanced Settings", open=False):
                        process_length = gr.Slider(
                            label="process length",
                            minimum=-1,
                            maximum=280,
                            value=110,
                            step=1,
                        )
                        max_res = gr.Slider(
                            label="max resolution",
                            minimum=512,
                            maximum=2048,
                            value=1024,
                            step=64,
                        )
                        num_denoising_steps = gr.Slider(
                            label="num denoising steps",
                            minimum=1,
                            maximum=25,
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
                            label="shift window size",
                            minimum=10,
                            maximum=110,
                            value=110,
                            step=10,
                        )
                        decode_chunk_size = gr.Slider(
                            label="decode chunk size",
                            minimum=1,
                            maximum=16,
                            value=6,
                            step=1,
                        )
                        overlap = gr.Slider(
                            label="overlap",
                            minimum=1,
                            maximum=50,
                            value=25,
                            step=1,
                        )    
                    generate_btn = gr.Button("Generate")

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
                    # display_mode="solid"
                    interactive=False
                )
            with gr.Column(scale=1):
                output_point_map1 = gr.Model3D(
                    label="Point Map 1",
                    clear_color=[1.0, 1.0, 1.0, 1.0],
                    # display_mode="solid"
                    interactive=False
                )
            with gr.Column(scale=1):
                output_point_map2 = gr.Model3D(
                    label="Point Map 2",
                    clear_color=[1.0, 1.0, 1.0, 1.0],
                    # display_mode="solid"
                    interactive=False
                )
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                output_point_map3 = gr.Model3D(
                    label="Point Map 3",
                    clear_color=[1.0, 1.0, 1.0, 1.0],
                    # display_mode="solid"
                    interactive=False
                )
            with gr.Column(scale=1):
                output_point_map4 = gr.Model3D(
                    label="Point Map 4",
                    clear_color=[1.0, 1.0, 1.0, 1.0],
                    # display_mode="solid"
                    interactive=False
                )
            with gr.Column(scale=1):
                output_point_map5 = gr.Model3D(
                    label="Point Map 5",
                    clear_color=[1.0, 1.0, 1.0, 1.0],
                    # display_mode="solid"
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
            # cache_examples="lazy",
        )
        gr.HTML(
            """
            <span style='font-size:18px'>Note: \
                Running locally without HuggingFace Spaces GPU allocation. \
                Ensure you have an Apple Silicon Mac with sufficient memory (20-40GB recommended). \
                You may adjust the parameters according to the \
                <a href='https://github.com/TencentARC/GeometryCrafter/'>[Github Repo]</a>\
                for better results. You can download the npz file and open it with Viser backend \
                in the repo for better visualization. \
            </span>
            """
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
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True, share=False)