# coding=utf-8
# Adapted from
# https://github.com/NVlabs/Sana/blob/main/inference_video_scripts/inference_sana_video.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025 - 2026. All rights reserved.
# Copyright 2024 NVIDIA CORPORATION & AFFILIATES

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import imageio
import pyrallis
import torch
import torch_npu
from accelerate import Accelerator
from termcolor import colored
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["DISABLE_XFORMERS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sana_npu_adaptation import install_compatibility_shims

install_compatibility_shims()

from diffusion import DPMS, FlowEuler, LongLiveFlowEuler, LTXFlowEuler
from diffusion.data.datasets.utils import *
from diffusion.data.transforms import read_image_from_path
from diffusion.guiders import AdaptiveProjectedGuidance
from diffusion.model.builder import (
    build_model,
    encode_image,
    get_tokenizer_and_text_encoder,
    get_vae,
    vae_decode,
    vae_encode,
)
from diffusion.model.utils import get_weight_dtype, prepare_prompt_ar
from diffusion.utils.config import SanaVideoConfig, model_video_init_config
from diffusion.utils.logger import get_root_logger
from tools.download import find_model


def set_env(seed=0, latent_size=256):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, latent_size, latent_size)


def get_dict_chunks(data, bs):
    keys = []
    for k in data:
        keys.append(k)
        if len(keys) == bs:
            yield keys
            keys = []
    if keys:
        yield keys


def sync_npu_if_needed():
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def empty_device_cache():
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_torch_profiler(args):
    if not args.enable_torch_profiler:
        return None

    os.makedirs(args.profiler_dir, exist_ok=True)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    )
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=args.profiler_wait,
            warmup=args.profiler_warmup,
            active=args.profiler_active,
            repeat=args.profiler_repeat,
            skip_first=0,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args.profiler_dir),
        record_shapes=args.profiler_record_shapes,
        profile_memory=False,
        with_stack=args.profiler_with_stack,
        experimental_config=experimental_config,
    )


def export_profiler_summary(profiler, args):
    if profiler is None:
        return None

    summary_path = Path(args.profiler_dir) / f"{args.metrics_tag}_key_averages.txt"
    summary = None
    for sort_key in ["self_npu_time_total", "self_cpu_time_total", None]:
        try:
            kwargs = {"row_limit": 40}
            if sort_key is not None:
                kwargs["sort_by"] = sort_key
            summary = profiler.key_averages().table(**kwargs)
            break
        except Exception:
            continue
    if summary is None:
        summary = "Failed to export profiler summary table."

    summary_path.write_text(summary, encoding="utf-8")
    return str(summary_path)


class DistributePromptsDataset(torch.utils.data.Dataset):
    def __init__(self, prompts, original_indices=None):
        if isinstance(prompts, dict):
            self.prompts = prompts
            self.keys_list = list(self.prompts.keys())
            self.original_indices = original_indices or list(range(len(prompts)))
        else:
            self.prompts = {
                prompt[:50].split("/")[0] + str(hashlib.sha256(prompt.encode()).hexdigest())[:10]: prompt
                for prompt in prompts
            }
            self.keys_list = list(self.prompts.keys())
            self.original_indices = original_indices or list(range(len(prompts)))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        key = self.keys_list[idx]
        prompt = self.prompts[key]
        txt_line_idx = self.original_indices[idx]
        return {
            "key": key,
            "prompt": prompt,
            "global_idx": txt_line_idx,
        }


@torch.inference_mode()
def visualize(config, args, model, items, bs, sample_steps, cfg_scale, profiler=None):
    cur_seed = args.seed + int(rank)
    generator = torch.Generator(device=device).manual_seed(cur_seed)
    tqdm_desc = f"{save_root.split('/')[-1]} Using GPU: {args.gpu_id}: {args.start_index}-{args.end_index}"
    batch_metrics = []
    for chunk in tqdm(prompts_dataloader, desc=tqdm_desc, unit="batch", position=rank, leave=True):
        prompts, hw = (
            [],
            torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1),
        )
        images = []
        if bs == 1:
            prompt = chunk["prompt"][0]
            prompt_clean, _, hw, _, _ = prepare_prompt_ar(prompt, base_ratios, device=device, show=False)
            if config.task == "ti2v" or config.task == "ltx":
                prompt_clean, image_path = prompt_clean.split(image_split_token)
                images.append(image_path)
            if args.prompt_split_token in prompt_clean:
                prompt_clean = prompt_clean.split(args.prompt_split_token)
                prompts.extend([_prompt.strip() + motion_prompt for _prompt in prompt_clean])
            else:
                prompts.append(prompt_clean.strip() + motion_prompt)
        else:
            for prompt in chunk["prompt"]:
                prompt_clean, _, hw, _, _ = prepare_prompt_ar(prompt, base_ratios, device=device, show=False)
                if config.task == "ti2v" or config.task == "ltx":
                    prompt_clean, image_path = prompt_clean.split(image_split_token)
                    images.append(image_path)
                prompts.append(prompt_clean.strip() + motion_prompt)
        latent_size_h, latent_size_w = (
            int(hw[0, 0] // config.vae.vae_downsample_rate),
            int(hw[0, 1] // config.vae.vae_downsample_rate),
        )

        exist = False
        for i in range(bs):
            save_file_name = f"{chunk['global_idx'][i]}. {chunk['key'][i]}.mp4"
            save_path = os.path.join(save_root, save_file_name)
            exist = os.path.exists(save_path)
            if not exist:
                break
        if exist:
            torch.randn(
                bs,
                config.vae.vae_latent_dim,
                latent_size_t,
                latent_size_h,
                latent_size_w,
                device=device,
                generator=generator,
            )
            if profiler is not None:
                profiler.step()
            continue

        if not config.text_encoder.chi_prompt:
            max_length_all = config.text_encoder.model_max_length
            prompts_all = prompts
        else:
            chi_prompt = "\n".join(config.text_encoder.chi_prompt)
            prompts_all = [chi_prompt + prompt for prompt in prompts]
            num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))
            max_length_all = num_chi_prompt_tokens + config.text_encoder.model_max_length - 2

        if "Qwen" in config.text_encoder.text_encoder_name:
            with torch.no_grad():
                caption_embs, emb_masks = text_handler.get_prompt_embeds(prompts_all, max_length=max_length_all)
                caption_embs = caption_embs[:, None]
                negative_embs_mask = negative_caption_embs_mask.repeat(bs, 1)
                negative_embs = negative_caption_embs.repeat(bs, 1, 1)[:, None]
        else:
            caption_token = tokenizer(
                prompts_all, max_length=max_length_all, padding="max_length", truncation=True, return_tensors="pt"
            ).to(device)
            select_index = [0] + list(range(-config.text_encoder.model_max_length + 1, 0))
            caption_embs = text_encoder(caption_token.input_ids, caption_token.attention_mask)[0][:, None][:, :, select_index]
            emb_masks = caption_token.attention_mask[:, select_index]
            negative_embs_mask = negative_caption_token.attention_mask.repeat(bs, 1)
            negative_embs = negative_caption_embs.repeat(bs, 1, 1)[:, None]

        if cfg_scale > 1.0:
            emb_masks = torch.cat([negative_embs_mask, emb_masks], dim=0)
        other_kwargs = dict(stg_applied_layers=args.stg_applied_layers, stg_scale=args.stg_scale)

        with torch.no_grad():
            n = bs
            z = torch.randn(
                n,
                config.vae.vae_latent_dim,
                latent_size_t,
                latent_size_h,
                latent_size_w,
                device=device,
                generator=generator,
            )
            model_kwargs = dict(data_info={"img_hw": hw}, mask=emb_masks)

            if config.task == "ltx":
                images = [
                    read_image_from_path(
                        imgp,
                        (
                            int(latent_size_h * config.vae.vae_downsample_rate),
                            int(latent_size_w * config.vae.vae_downsample_rate),
                        ),
                    )
                    for imgp in images
                ]

                image_vae_embeds = vae_encode(
                    config.vae.vae_type, vae, torch.stack(images, dim=0)[:, :, None].to(vae_dtype), device=device
                )
                condition_frame_info = {
                    0: (config.train.noise_multiplier if config.train.noise_multiplier is not None else 0.0),
                }
                for frame_idx in list(condition_frame_info.keys()):
                    z[:, :, frame_idx : frame_idx + 1] = image_vae_embeds
                model_kwargs["data_info"].update({"condition_frame_info": condition_frame_info})

                if image_encoder is not None:
                    image_embeds = encode_image(
                        name=config.image_encoder.image_encoder_name,
                        image_encoder=image_encoder,
                        image_processor=image_processor,
                        images=torch.stack(images, dim=0).to(device),
                        device=device,
                        dtype=weight_dtype,
                    )
                    if cfg_scale > 1.0:
                        image_embeds = torch.cat([image_embeds, image_embeds], dim=0)
                    model_kwargs["data_info"].update({"image_embeds": image_embeds})

            sampling_start = time.perf_counter()
            if args.sampling_algo == "flow_euler_ltx":
                flow_solver = LTXFlowEuler(
                    model,
                    condition=caption_embs,
                    uncondition=negative_embs,
                    cfg_scale=cfg_scale,
                    flow_shift=flow_shift,
                    model_kwargs=model_kwargs,
                )
                samples = flow_solver.sample(z, steps=sample_steps, generator=generator)
            elif args.sampling_algo == "flow_euler":
                flow_solver = FlowEuler(
                    model,
                    condition=caption_embs,
                    uncondition=negative_embs,
                    cfg_scale=cfg_scale,
                    flow_shift=flow_shift,
                    model_kwargs=model_kwargs,
                    apg=apg,
                )
                samples = flow_solver.sample(z, steps=sample_steps)
            elif args.sampling_algo == "flow_dpm-solver":
                dpm_solver = DPMS(
                    model,
                    condition=caption_embs,
                    uncondition=negative_embs,
                    cfg_scale=cfg_scale,
                    model_type="flow",
                    guidance_type=guidance_type,
                    model_kwargs=model_kwargs,
                    schedule="FLOW",
                    apg=apg,
                    **other_kwargs,
                )
                samples = dpm_solver.sample(
                    z,
                    steps=sample_steps,
                    order=2,
                    skip_type=args.skip_type,
                    method="multistep",
                    flow_shift=flow_shift,
                )
            elif args.sampling_algo == "longlive_flow_euler":
                base_chunk_frames = base_model_frames // config.vae.vae_stride[0]
                flow_solver = LongLiveFlowEuler(
                    model,
                    condition=caption_embs,
                    flow_shift=flow_shift,
                    model_kwargs=model_kwargs,
                    base_chunk_frames=base_chunk_frames,
                    num_cached_blocks=args.num_cached_blocks,
                )
                samples = flow_solver.sample(z, steps=sample_steps, generator=generator)
            else:
                raise ValueError(f"{args.sampling_algo} is not defined")
            sync_npu_if_needed()
            sampling_time = time.perf_counter() - sampling_start

        batch_metrics.append(
            {
                "global_indices": [int(v) for v in chunk["global_idx"]],
                "sampling_time_s": sampling_time,
                "sample_steps": int(sample_steps),
                "single_step_latency_s": sampling_time / max(sample_steps, 1),
            }
        )

        samples = samples.to(vae_dtype)
        samples = vae_decode(config.vae.vae_type, vae, samples)
        if isinstance(samples, list):
            samples = torch.stack(samples, dim=0)
        videos = torch.clamp(127.5 * samples + 127.5, 0, 255).permute(0, 2, 3, 4, 1).to("cpu", dtype=torch.uint8)
        empty_device_cache()

        if not args.skip_save:
            os.umask(0o000)
            for i, video in enumerate(videos):
                save_file_name = f"{chunk['global_idx'][i]:03d}. {chunk['key'][i]}.mp4"
                save_path = os.path.join(save_root, save_file_name)
                writer = imageio.get_writer(save_path, fps=args.fps, codec="libx264", quality=8)
                for frame in video.numpy():
                    writer.append_data(frame)
                writer.close()

        if profiler is not None:
            profiler.step()

    if not batch_metrics:
        return {
            "num_batches": 0,
            "sample_steps": int(sample_steps),
            "mean_sampling_time_s": 0.0,
            "mean_single_step_latency_s": 0.0,
            "batch_metrics": [],
        }

    mean_sampling_time = sum(item["sampling_time_s"] for item in batch_metrics) / len(batch_metrics)
    mean_single_step_latency = sum(item["single_step_latency_s"] for item in batch_metrics) / len(batch_metrics)
    return {
        "num_batches": len(batch_metrics),
        "sample_steps": int(sample_steps),
        "mean_sampling_time_s": mean_sampling_time,
        "mean_single_step_latency_s": mean_single_step_latency,
        "batch_metrics": batch_metrics,
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config")
    return parser.parse_known_args()[0]


@dataclass
class SanaInference(SanaVideoConfig):
    config: Optional[str] = "configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml"
    model_path: Optional[str] = "hf://Efficient-Large-Model/SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth"
    work_dir: Optional[str] = None
    txt_file: str = "asset/samples/video_prompts_samples.txt"
    json_file: Optional[str] = None
    fps: int = 16
    sample_nums: int = 100_000
    bs: int = 1
    num_frames: int = -1
    cfg_scale: float = 6.0
    flow_shift: Optional[float] = None
    sampling_algo: Optional[str] = None
    skip_type: str = "time_uniform_flow"
    guidance_type: str = "classifier-free"
    seed: int = 0
    dataset: str = ""
    step: int = -1
    add_label: str = ""
    tar_and_del: bool = False
    exist_time_prefix: str = ""
    gpu_id: int = 0
    custom_height_width: Optional[Tuple[int, int]] = None
    use_resolution_binning: bool = True
    start_index: int = 0
    end_index: int = 100_000
    interval_k: float = -1
    high_motion: bool = False
    motion_score: int = -1
    image_split_token: str = "<image>"
    prompt_split_token: str = "<split_prompt>"
    image_size: int = 480
    debug: bool = False
    if_save_dirname: bool = False
    negative_prompt: str = ""
    stg_scale: float = 1.0
    stg_applied_layers: List[int] = field(default_factory=lambda: [14])
    apg_mode: str = "hard"
    num_cached_blocks: int = 0
    skip_save: bool = False
    metrics_tag: str = "run"
    enable_torch_profiler: bool = False
    profiler_dir: str = "./prof"
    profiler_wait: int = 0
    profiler_warmup: int = 0
    profiler_active: int = 1
    profiler_repeat: int = 1
    profiler_record_shapes: bool = False
    profiler_with_stack: bool = False


if __name__ == "__main__":
    args = get_args()
    args = pyrallis.parse(config_class=SanaInference, config_path=args.config)
    config = args
    args.image_size = config.model.image_size
    set_env(args.seed, args.image_size // config.vae.vae_downsample_rate)

    accelerator = Accelerator(split_batches=True, mixed_precision=config.model.mixed_precision)
    device = accelerator.device
    rank = int(os.environ.get("RANK", accelerator.process_index))
    logger = get_root_logger()

    config.dataset = config.dataset or "sana_video"
    config.model.from_pretrained = args.model_path
    config.model.fp32_attention = getattr(config.model, "fp32_attention", False)
    if args.use_resolution_binning and args.custom_height_width is None:
        if config.vae.vae_downsample_rate in [16, 32]:
            base_ratios = eval(f"ASPECT_RATIO_VIDEO_{args.image_size}_TEST_DIV32")
        else:
            base_ratios = eval(f"ASPECT_RATIO_VIDEO_{args.image_size}_TEST")
        aspect_ratio_key = random.choice(list(base_ratios.keys()))
        video_height, video_width = map(int, base_ratios[aspect_ratio_key])
    elif args.custom_height_width is not None:
        video_height, video_width = args.custom_height_width
        base_ratios = {f"{video_height/video_width:.2f}": [float(video_height), float(video_width)]}
    else:
        logger.info("Using default height and width: 480, 832")
        video_height, video_width = 480, 832
        base_ratios = {f"{video_height/video_width:.2f}": [float(video_height), float(video_width)]}

    latent_size_w = int(video_width) // config.vae.vae_stride[2]
    latent_size_h = int(video_height) // config.vae.vae_stride[1]
    latent_size = args.image_size // config.vae.vae_downsample_rate
    base_model_frames = config.data.num_frames
    num_frames = config.data.num_frames if args.num_frames == -1 else args.num_frames
    config.num_frames = num_frames
    latent_size_t = int(num_frames - 1) // config.vae.vae_stride[0] + 1
    logger.info(f"Latent size: {latent_size_t}t, {latent_size_h}h, {latent_size_w}w")
    max_sequence_length = config.text_encoder.model_max_length
    if args.flow_shift is not None:
        flow_shift = args.flow_shift
    else:
        flow_shift = (
            config.scheduler.inference_flow_shift
            if config.scheduler.inference_flow_shift is not None
            else config.scheduler.flow_shift
        )

    if args.motion_score > 0:
        motion_prompt = f" motion score: {int(args.motion_score)}."
    else:
        motion_prompt = " high motion" if args.high_motion else " low motion"
    if config.negative_prompt is None or config.negative_prompt == "None":
        config.negative_prompt = ""
    elif config.negative_prompt.startswith("{") and config.negative_prompt.endswith("}"):
        negative_prompt_dict = eval(config.negative_prompt)
        negative_parts = []
        for key, value in negative_prompt_dict.items():
            negative_parts.append(f"{key}: {value}")
        config.negative_prompt = " ".join(negative_parts)
    logger.info(f"negative_prompt: {config.negative_prompt}")

    if args.sampling_algo == "longlive_flow_euler":
        assert args.cfg_scale == 1.0, "cfg_scale must be 1.0 for longlive_flow_euler"

    guidance_type = args.guidance_type
    apg = None
    if guidance_type == "adaptive_projected_guidance":
        apg = AdaptiveProjectedGuidance(
            guidance_scale=args.cfg_scale,
            adaptive_projected_guidance_momentum=-0.5,
            adaptive_projected_guidance_rescale=27,
            eta=1.0,
            mode=args.apg_mode,
        )

    sample_steps = args.step if args.step != -1 else 50
    image_split_token = args.image_split_token
    weight_dtype = get_weight_dtype(config.model.mixed_precision)
    logger.info(f"Inference with {weight_dtype}, default guidance_type: {guidance_type}, flow_shift: {flow_shift}")
    logger.info(f"motion_prompt: {motion_prompt}")

    vae_dtype = get_weight_dtype(config.vae.weight_dtype)
    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, device=device, dtype=vae_dtype, config=config.vae)
    tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder.text_encoder_name, device=device)

    if "Qwen" in config.text_encoder.text_encoder_name:
        text_handler = text_encoder
        text_encoder = text_handler.text_encoder
        with torch.no_grad():
            negative_caption_embs, negative_caption_embs_mask = text_handler.get_prompt_embeds(
                config.negative_prompt, max_length=max_sequence_length
            )
    else:
        negative_caption_token = tokenizer(
            config.negative_prompt,
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        negative_caption_embs = text_encoder(negative_caption_token.input_ids, negative_caption_token.attention_mask)[0]

    image_encoder, image_processor = None, None
    model_kwargs = model_video_init_config(config, latent_size=latent_size)
    model = build_model(
        config.model.model,
        use_fp32_attention=config.model.get("fp32_attention", False),
        **model_kwargs,
    ).to(device)
    logger.info(
        f"{model.__class__.__name__}:{config.model.model}, Model Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    logger.info(f"Generating sample from ckpt: {args.model_path}")
    state_dict = find_model(args.model_path)
    if "generator" in state_dict:
        state_dict = state_dict["generator"]
    if "state_dict" not in state_dict:
        new_state_dict = dict()
        for k, v in state_dict.items():
            if k.startswith("model."):
                k = k[len("model.") :]
            new_state_dict[k] = v
        state_dict = {"state_dict": new_state_dict}

    if args.model_path.endswith(".bin"):
        logger.info("Loading fsdp bin checkpoint....")
        old_state_dict = state_dict
        state_dict = {"state_dict": old_state_dict}

    if "pos_embed" in state_dict["state_dict"]:
        del state_dict["state_dict"]["pos_embed"]

    missing, unexpected = model.load_state_dict(state_dict["state_dict"], strict=False)
    logger.warning(f"Missing keys: {missing}")
    logger.warning(f"Unexpected keys: {unexpected}")
    model.eval().to(weight_dtype)

    args.sampling_algo = config.scheduler.vis_sampler if args.sampling_algo is None else args.sampling_algo
    if config.task in ["ltx", "ti2v"]:
        args.sampling_algo = "flow_euler_ltx"

    if args.work_dir is None:
        work_dir = (
            f"/{os.path.join(*args.model_path.split('/')[:-2])}"
            if args.model_path.startswith("/")
            else os.path.join(*args.model_path.split("/")[:-2])
        )
    else:
        work_dir = args.work_dir
    config.work_dir = work_dir
    img_save_dir = os.path.join(str(work_dir), "vis")

    logger.info(colored(f"Saving videos at {img_save_dir}", "green"))
    dict_prompt = args.json_file is not None
    if dict_prompt:
        data_dict = json.load(open(args.json_file))
        all_items = list(data_dict.keys())
        logger.info(f"Eval first {min(args.sample_nums, len(all_items))}/{len(all_items)} samples")
        total_items = all_items[: max(0, args.sample_nums)]
        start_idx = max(0, args.start_index)
        end_idx = min(len(total_items), args.end_index)
        items = total_items[start_idx:end_idx]
        original_indices = list(range(start_idx, end_idx))
    else:
        with open(args.txt_file) as f:
            all_items = [item.strip() for item in f.readlines()]
        logger.info(f"Eval first {min(args.sample_nums, len(all_items))}/{len(all_items)} samples")
        total_items = all_items[: max(0, args.sample_nums)]
        start_idx = max(0, args.start_index)
        end_idx = min(len(total_items), args.end_index)
        items = total_items[start_idx:end_idx]
        original_indices = list(range(start_idx, end_idx))

    match = re.search(r".*epoch_(\d+).*step_(\d+).*", args.model_path)
    epoch_name, step_name = match.groups() if match else ("unknown", "unknown")

    os.umask(0o000)
    os.makedirs(img_save_dir, exist_ok=True)
    logger.info(f"Sampler {args.sampling_algo}, t_type: {args.skip_type}")

    prompts_dataset = DistributePromptsDataset(items, original_indices)
    prompts_dataloader = torch.utils.data.DataLoader(prompts_dataset, batch_size=args.bs, shuffle=False)
    prompts_dataloader, model, text_encoder = accelerator.prepare(prompts_dataloader, model, text_encoder)
    if num_frames > base_model_frames:
        assert args.sampling_algo == "longlive_flow_euler"

    def create_save_root(args, dataset, epoch_name, step_name, sample_steps, num_frames):
        save_root = os.path.join(
            img_save_dir,
            f"{dataset}_step{step_name}_scale{args.cfg_scale}"
            f"_step{sample_steps}_size{args.image_size}_numframes{num_frames}_bs{args.bs}_samp{args.sampling_algo}"
            f"_seed{args.seed}_{str(weight_dtype).split('.')[-1]}",
        )
        if args.skip_type != "time_uniform_flow":
            save_root += f"_skip-{args.skip_type}"
        if args.skip_type == "time_uniform_flow" and flow_shift != 1.0:
            save_root += f"_flowshift{flow_shift}"
        if args.interval_k > 0:
            save_root += f"_interval_k{int(args.interval_k * 1000)}"
        if args.num_cached_blocks > 0:
            save_root += f"_numcb{args.num_cached_blocks}"
        if args.high_motion:
            save_root += "_highmotion"
        if args.motion_score > 0:
            save_root += f"_motion{args.motion_score}"
        if args.negative_prompt != "":
            save_root += f"_negp{args.negative_prompt[:5].replace(' ', '')}"
        save_root += f"_gd{''.join(word[0] for word in args.guidance_type.split('_'))}"
        if args.guidance_type == "adaptive_projected_guidance":
            save_root += f"_{args.apg_mode}"
        if args.guidance_type == "classifier-free_STG":
            save_root += f"_stg{args.stg_scale}_stgl{''.join(str(layer) for layer in args.stg_applied_layers)}"
        save_root += f"_imgnums{args.sample_nums}" + args.add_label
        return save_root

    dataset = args.dataset
    logger.info(f"Inference with {weight_dtype}, flow_shift: {flow_shift}")
    save_root = create_save_root(args, dataset, epoch_name, step_name, sample_steps, num_frames)
    os.makedirs(save_root, exist_ok=True)
    metrics_dir = Path(work_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        items = [
            "A fashionable woman in a black leather jacket, long red dress, and black boots confidently strolls down a wet, reflective Tokyo street.",
            "Several giant woolly mammoths lumber through a snowy meadow under warm afternoon sunlight.",
        ]

    profiler = build_torch_profiler(args)
    profiler_context = profiler if profiler is not None else nullcontext()
    with profiler_context:
        metrics = visualize(
            config=config,
            args=args,
            model=model,
            items=items,
            bs=args.bs,
            sample_steps=sample_steps,
            cfg_scale=args.cfg_scale,
            profiler=profiler,
        )

    profiler_summary_path = export_profiler_summary(profiler, args)
    metrics["metrics_tag"] = args.metrics_tag
    metrics["work_dir"] = work_dir
    metrics["save_root"] = save_root
    metrics["profiler_enabled"] = args.enable_torch_profiler
    metrics["profiler_dir"] = args.profiler_dir if args.enable_torch_profiler else None
    metrics["profiler_summary_path"] = profiler_summary_path
    metrics_path = metrics_dir / f"{args.metrics_tag}_summary.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(colored("Sana inference has finished. Results stored at ", "green"), colored(f"{img_save_dir}", attrs=["bold"]), ".")
    print(colored("Metrics summary stored at ", "green"), colored(str(metrics_path), attrs=["bold"]), ".")
