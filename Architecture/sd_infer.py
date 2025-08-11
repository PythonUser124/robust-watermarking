import argparse
import os
import re
from typing import List, Optional, Tuple

import torch


def detect_device(preferred: Optional[str] = None) -> torch.device:
    if preferred in {"cpu", "mps", "cuda"}:
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sanitize_filename(text: str, max_len: int = 60) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^a-zA-Z0-9_\-]", "", text)
    if len(text) == 0:
        text = "prompt"
    return text[:max_len]


MODEL_PRESETS = {
    # Baseline SD 1.5 (frozen weights)
    "sd15": {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "pipeline": "StableDiffusionPipeline",
        "default": {"steps": 30, "guidance": 7.5},
    },
    # Baseline SD 2.1
    "sd21": {
        "model_id": "stabilityai/stable-diffusion-2-1",
        "pipeline": "StableDiffusionPipeline",
        "default": {"steps": 30, "guidance": 7.0},
    },
    # SDXL base (heavier)
    "sdxl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline": "StableDiffusionXLPipeline",
        "default": {"steps": 30, "guidance": 5.0},
    },
    # Distilled / very-fast variants
    # SD Turbo (very distilled, 1-4 steps, guidance ~ 0)
    "sd-turbo": {
        "model_id": "stabilityai/sd-turbo",
        "pipeline": "AutoPipelineForText2Image",
        "default": {"steps": 2, "guidance": 0.0},
    },
    # SDXL Turbo
    "sdxl-turbo": {
        "model_id": "stabilityai/sdxl-turbo",
        "pipeline": "AutoPipelineForText2Image",
        "default": {"steps": 2, "guidance": 0.0},
    },
    # Segmind SSD-1B (distilled 1B SD 2.1-like)
    "ssd-1b": {
        "model_id": "segmind/SSD-1B",
        "pipeline": "StableDiffusionPipeline",
        "default": {"steps": 20, "guidance": 4.0},
    },
}


def load_pipeline(preset: str, custom_model: Optional[str], device: torch.device, dtype: torch.dtype):
    try:
        from diffusers import (
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
            AutoPipelineForText2Image,
            DPMSolverMultistepScheduler,
        )
    except Exception as e:
        raise RuntimeError(
            "Please install diffusers: pip install diffusers transformers accelerate safetensors"
        ) from e

    if preset == "custom":
        if not custom_model:
            raise ValueError("--model-id is required for preset=custom")
        # Best effort: try AutoPipelineForText2Image first
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(custom_model, torch_dtype=dtype)
        except Exception:
            # Fallback to SD 1.x pipeline
            pipe = StableDiffusionPipeline.from_pretrained(custom_model, torch_dtype=dtype)
        defaults = {"steps": 30, "guidance": 7.5}
    else:
        cfg = MODEL_PRESETS[preset]
        model_id = cfg["model_id"]
        pipeline_name = cfg["pipeline"]
        defaults = cfg["default"].copy()
        if pipeline_name == "StableDiffusionPipeline":
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
            # swap to DPM-Solver++ for speed/quality if available
            try:
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            except Exception:
                pass
        elif pipeline_name == "StableDiffusionXLPipeline":
            pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype)
        elif pipeline_name == "AutoPipelineForText2Image":
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_name}")

    # Frozen weights (inference only)
    for p in pipe.parameters():
        p.requires_grad_(False)

    # Optional safety checker disable switch is handled in main via args
    pipe.to(device)
    # Memory optimizations
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_sequential_cpu_offload() if device.type == "cuda" else None
    except Exception:
        pass
    return pipe, defaults


def generate_images(
    pipe,
    prompts: List[str],
    out_dir: str,
    steps: int,
    guidance: float,
    num_images: int,
    height: Optional[int],
    width: Optional[int],
    seed: Optional[int],
    negative_prompt: Optional[str],
    safety_off: bool,
):
    os.makedirs(out_dir, exist_ok=True)
    if safety_off:
        try:
            pipe.safety_checker = None
            if hasattr(pipe, "requires_safety_checker"):
                pipe.requires_safety_checker = False
        except Exception:
            pass

    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    for i, prompt in enumerate(prompts):
        prompt_clean = sanitize_filename(prompt) or f"prompt_{i}"
        for k in range(num_images):
            g = generator
            if generator is not None and num_images > 1:
                # derive a new seed per image for reproducibility
                g = torch.Generator(device=pipe.device).manual_seed(seed + k)
            kwargs = {
                "prompt": prompt,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "negative_prompt": negative_prompt,
                "generator": g,
            }
            if height is not None and width is not None:
                kwargs.update({"height": height, "width": width})
            with torch.inference_mode():
                image = pipe(**kwargs).images[0]
            out_path = os.path.join(out_dir, f"{i:03d}_{k:02d}_{prompt_clean}.png")
            image.save(out_path)
            print(f"Saved {out_path}")


def read_prompts(prompt: Optional[str], prompts_file: Optional[str]) -> List[str]:
    if prompt is not None:
        return [prompt]
    if prompts_file is not None:
        with open(prompts_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines()]
        return [ln for ln in lines if ln]
    raise ValueError("Provide --prompt or --prompts-file")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stable Diffusion inference (frozen weights) with distilled presets")
    p.add_argument("--preset", default="sd15", choices=["sd15", "sd21", "sdxl", "sd-turbo", "sdxl-turbo", "ssd-1b", "custom"], help="Model preset or 'custom'")
    p.add_argument("--model-id", default=None, help="Hugging Face model id when preset=custom")
    p.add_argument("--prompt", default=None, help="Text prompt")
    p.add_argument("--prompts-file", default=None, help="File with one prompt per line")
    p.add_argument("--negative-prompt", default=None, help="Negative prompt")
    p.add_argument("--out-dir", default="./sd_outputs")
    p.add_argument("--steps", type=int, default=None, help="Inference steps (None uses preset default)")
    p.add_argument("--guidance-scale", type=float, default=None, help="Classifier-free guidance scale (None uses preset default)")
    p.add_argument("--num-images", type=int, default=1, help="Images per prompt")
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--seed", type=int, default=-1, help="Seed (>=0 for reproducibility)")
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"], help="Override device")
    p.add_argument("--disable-safety", action="store_true", help="Disable safety checker")
    return p


def main():
    args = build_parser().parse_args()
    device = detect_device(args.device)

    # dtype selection
    if device.type == "cuda":
        dtype = torch.float16
    elif device.type == "mps":
        # fp16 often works; fall back to fp32 if issues
        dtype = torch.float16
    else:
        dtype = torch.float32

    pipe, defaults = load_pipeline(args.preset, args.model_id, device, dtype)

    steps = args.steps if args.steps is not None else defaults.get("steps", 30)
    guidance = args.guidance_scale if args.guidance_scale is not None else defaults.get("guidance", 7.5)

    # Turbo presets strongly prefer low steps and guidance 0
    if args.preset in {"sd-turbo", "sdxl-turbo"}:
        if args.steps is None:
            steps = 1 if device.type == "cuda" else 2
        if args.guidance_scale is None:
            guidance = 0.0

    prompts = read_prompts(args.prompt, args.prompts_file)
    seed = args.seed if args.seed is not None and args.seed >= 0 else None

    generate_images(
        pipe=pipe,
        prompts=prompts,
        out_dir=args.out_dir,
        steps=steps,
        guidance=guidance,
        num_images=args.num_images,
        height=args.height,
        width=args.width,
        seed=seed,
        negative_prompt=args.negative_prompt,
        safety_off=args.disable_safety,
    )


if __name__ == "__main__":
    main()


