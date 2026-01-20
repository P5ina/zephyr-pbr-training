"""
ComfyUI Custom Nodes for PBR Texture Generation

Generates basecolor, normal, roughness, and height maps from text prompts.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

import folder_paths
import comfy.model_management as mm


class PBRMapHeads(nn.Module):
    """Lightweight projection heads for each PBR map type."""

    def __init__(self, latent_channels: int = 4):
        super().__init__()
        self.heads = nn.ModuleDict({
            "basecolor": nn.Conv2d(latent_channels, latent_channels, 1),
            "normal": nn.Conv2d(latent_channels, latent_channels, 1),
            "roughness": nn.Conv2d(latent_channels, latent_channels, 1),
            "height": nn.Conv2d(latent_channels, latent_channels, 1),
        })

    def forward(self, x: torch.Tensor, map_type: str) -> torch.Tensor:
        return self.heads[map_type](x)


class LoadPBRModel:
    """Load trained PBR map heads."""

    @classmethod
    def INPUT_TYPES(cls):
        # Look for map_heads.pt files in models directory
        models_dir = Path(folder_paths.models_dir) / "pbr_heads"
        models_dir.mkdir(exist_ok=True)

        files = [f.name for f in models_dir.glob("*.pt")]
        if not files:
            files = ["none"]

        return {
            "required": {
                "map_heads_file": (files, {"default": files[0] if files else "none"}),
            }
        }

    RETURN_TYPES = ("PBR_HEADS",)
    RETURN_NAMES = ("pbr_heads",)
    FUNCTION = "load"
    CATEGORY = "PBR/loaders"

    def load(self, map_heads_file):
        if map_heads_file == "none":
            # Return untrained heads
            heads = PBRMapHeads()
            return (heads,)

        models_dir = Path(folder_paths.models_dir) / "pbr_heads"
        path = models_dir / map_heads_file

        heads = PBRMapHeads()
        state_dict = torch.load(path, map_location="cpu")

        # Handle different save formats
        if "basecolor.weight" in state_dict:
            # Saved as flat dict
            new_state_dict = {}
            for key, value in state_dict.items():
                new_state_dict[f"heads.{key}"] = value
            heads.load_state_dict(new_state_dict)
        else:
            heads.load_state_dict(state_dict)

        return (heads,)


class PBRGenerate:
    """Generate PBR maps from SDXL latents using trained heads."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "pbr_heads": ("PBR_HEADS",),
                "positive": ("STRING", {"default": "seamless tileable pbr texture, 4k, high detail", "multiline": True}),
                "negative": ("STRING", {"default": "blurry, low quality, watermark", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("basecolor", "normal", "roughness", "height")
    FUNCTION = "generate"
    CATEGORY = "PBR/generate"

    def generate(self, model, clip, vae, pbr_heads, positive, negative, seed, steps, cfg, width, height):
        device = mm.get_torch_device()
        dtype = torch.float16 if mm.should_use_fp16() else torch.float32

        # Move heads to device
        pbr_heads = pbr_heads.to(device).to(dtype)

        # Encode prompts
        tokens = clip.tokenize(positive)
        tokens_neg = clip.tokenize(negative)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        uncond, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)

        # Prepare conditioning
        positive_cond = [[cond, {"pooled_output": pooled}]]
        negative_cond = [[uncond, {"pooled_output": pooled_neg}]]

        results = {}
        generator = torch.Generator(device=device).manual_seed(seed)

        for map_type in ["basecolor", "normal", "roughness", "height"]:
            # Start from noise (same seed for consistency)
            generator.manual_seed(seed)
            latent = torch.randn(
                1, 4, height // 8, width // 8,
                device=device, dtype=dtype, generator=generator
            )

            # Sample using ComfyUI's sampler
            from comfy.samplers import KSampler

            # Create a wrapper model that applies the PBR head
            class PBRModelWrapper:
                def __init__(self, base_model, heads, map_type):
                    self.base_model = base_model
                    self.heads = heads
                    self.map_type = map_type

                def __call__(self, x, timestep, **kwargs):
                    # Get base model prediction
                    out = self.base_model(x, timestep, **kwargs)
                    # Apply PBR head
                    out = self.heads(out, self.map_type)
                    return out

            wrapped_model = PBRModelWrapper(model.model, pbr_heads, map_type)

            # Simple Euler sampling
            sigmas = model.model.model_sampling.get_sigmas(steps).to(device)

            for i in range(len(sigmas) - 1):
                sigma = sigmas[i]
                sigma_next = sigmas[i + 1]

                # Denoise
                with torch.no_grad():
                    # CFG
                    noise_pred_cond = wrapped_model(latent, sigma, cond=positive_cond[0])
                    noise_pred_uncond = wrapped_model(latent, sigma, cond=negative_cond[0])
                    noise_pred = noise_pred_uncond + cfg * (noise_pred_cond - noise_pred_uncond)

                    # Euler step
                    dt = sigma_next - sigma
                    latent = latent + noise_pred * dt

            # Decode
            with torch.no_grad():
                image = vae.decode(latent)

            # Convert to ComfyUI format [B, H, W, C]
            image = image.clamp(-1, 1)
            image = (image + 1) / 2
            image = image.permute(0, 2, 3, 1).cpu().float()

            results[map_type] = image

        return (results["basecolor"], results["normal"], results["roughness"], results["height"])


class PBRCombineGrid:
    """Combine 4 PBR maps into a 2x2 grid for preview."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "basecolor": ("IMAGE",),
                "normal": ("IMAGE",),
                "roughness": ("IMAGE",),
                "height": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grid",)
    FUNCTION = "combine"
    CATEGORY = "PBR/utils"

    def combine(self, basecolor, normal, roughness, height):
        # All images should be [B, H, W, C]
        top = torch.cat([basecolor, normal], dim=2)  # Concat width
        bottom = torch.cat([roughness, height], dim=2)
        grid = torch.cat([top, bottom], dim=1)  # Concat height
        return (grid,)


class PBRSeparateMaps:
    """Node to separate PBR maps for individual processing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "basecolor": ("IMAGE",),
                "normal": ("IMAGE",),
                "roughness": ("IMAGE",),
                "height": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("basecolor", "normal", "roughness", "height")
    FUNCTION = "separate"
    CATEGORY = "PBR/utils"

    def separate(self, basecolor, normal, roughness, height):
        return (basecolor, normal, roughness, height)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LoadPBRModel": LoadPBRModel,
    "PBRGenerate": PBRGenerate,
    "PBRCombineGrid": PBRCombineGrid,
    "PBRSeparateMaps": PBRSeparateMaps,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadPBRModel": "Load PBR Map Heads",
    "PBRGenerate": "Generate PBR Maps",
    "PBRCombineGrid": "Combine PBR Grid",
    "PBRSeparateMaps": "Separate PBR Maps",
}
