"""
StableMaterials ComfyUI Nodes

Generates tileable PBR material maps (basecolor, normal, height, roughness, metallic)
from text prompts using the StableMaterials diffusion model.
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import os

import folder_paths
import comfy.model_management as mm


# Global pipeline cache
_pipeline = None
_pipeline_type = None


def get_pipeline(use_lcm: bool = False):
    """Load or return cached StableMaterials pipeline."""
    global _pipeline, _pipeline_type

    target_type = "lcm" if use_lcm else "standard"

    if _pipeline is not None and _pipeline_type == target_type:
        return _pipeline

    # Clear old pipeline
    if _pipeline is not None:
        del _pipeline
        torch.cuda.empty_cache()

    from diffusers import DiffusionPipeline

    device = mm.get_torch_device()
    dtype = torch.float16 if mm.should_use_fp16() else torch.float32

    if use_lcm:
        from diffusers import LCMScheduler, UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained(
            "gvecchio/StableMaterials",
            subfolder="unet_lcm",
            torch_dtype=dtype,
        )
        _pipeline = DiffusionPipeline.from_pretrained(
            "gvecchio/StableMaterials",
            trust_remote_code=True,
            unet=unet,
            torch_dtype=dtype,
        )
        _pipeline.scheduler = LCMScheduler.from_config(_pipeline.scheduler.config)
    else:
        _pipeline = DiffusionPipeline.from_pretrained(
            "gvecchio/StableMaterials",
            trust_remote_code=True,
            torch_dtype=dtype,
        )

    _pipeline = _pipeline.to(device)
    _pipeline_type = target_type

    return _pipeline


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI tensor [B, H, W, C]."""
    arr = np.array(img).astype(np.float32) / 255.0
    if len(arr.shape) == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor


class LoadStableMaterials:
    """Load the StableMaterials pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_lcm": ("BOOLEAN", {
                    "default": True,
                    "label_on": "LCM (4 steps, fast)",
                    "label_off": "Standard (50 steps)"
                }),
            }
        }

    RETURN_TYPES = ("SM_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "StableMaterials"

    def load(self, use_lcm):
        pipe = get_pipeline(use_lcm)
        return (pipe,)


class StableMaterialsGenerate:
    """Generate PBR material maps from text prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("SM_PIPELINE",),
                "prompt": ("STRING", {
                    "default": "Rusty weathered metal surface with scratches",
                    "multiline": True
                }),
                "negative_prompt": ("STRING", {
                    "default": "blurry, low quality, distorted",
                    "multiline": True
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.5}),
                "tileable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("basecolor", "normal", "height", "roughness", "metallic")
    FUNCTION = "generate"
    CATEGORY = "StableMaterials"

    def generate(self, pipeline, prompt, negative_prompt, seed, steps, guidance_scale, tileable):
        device = mm.get_torch_device()

        generator = torch.Generator(device=device).manual_seed(seed)

        with torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                tileable=tileable,
                generator=generator,
            )

        material = result.images[0]

        # Convert each map to ComfyUI tensor format
        basecolor = pil_to_tensor(material.basecolor)
        normal = pil_to_tensor(material.normal)
        height = pil_to_tensor(material.height)
        roughness = pil_to_tensor(material.roughness)
        metallic = pil_to_tensor(material.metallic)

        return (basecolor, normal, height, roughness, metallic)


class StableMaterialsFromImage:
    """Generate PBR materials conditioned on an input image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("SM_PIPELINE",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.5}),
                "image_guidance_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 5.0, "step": 0.1}),
                "tileable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("basecolor", "normal", "height", "roughness", "metallic")
    FUNCTION = "generate"
    CATEGORY = "StableMaterials"

    def generate(self, pipeline, image, prompt, seed, steps, guidance_scale, image_guidance_scale, tileable):
        device = mm.get_torch_device()

        # Convert ComfyUI tensor to PIL
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        input_image = Image.fromarray(img_np)

        generator = torch.Generator(device=device).manual_seed(seed)

        with torch.inference_mode():
            result = pipeline(
                prompt=prompt if prompt else None,
                image=input_image,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                tileable=tileable,
                generator=generator,
            )

        material = result.images[0]

        basecolor = pil_to_tensor(material.basecolor)
        normal = pil_to_tensor(material.normal)
        height = pil_to_tensor(material.height)
        roughness = pil_to_tensor(material.roughness)
        metallic = pil_to_tensor(material.metallic)

        return (basecolor, normal, height, roughness, metallic)


class CombinePBRMaps:
    """Combine PBR maps into a 2x3 grid for preview."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "basecolor": ("IMAGE",),
                "normal": ("IMAGE",),
                "height": ("IMAGE",),
                "roughness": ("IMAGE",),
                "metallic": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grid",)
    FUNCTION = "combine"
    CATEGORY = "StableMaterials/utils"

    def combine(self, basecolor, normal, height, roughness, metallic):
        # Create 2x3 grid: [basecolor, normal, height] / [roughness, metallic, empty]
        h, w = basecolor.shape[1], basecolor.shape[2]

        # Create empty black image for padding
        empty = torch.zeros_like(basecolor)

        top = torch.cat([basecolor, normal, height], dim=2)
        bottom = torch.cat([roughness, metallic, empty], dim=2)
        grid = torch.cat([top, bottom], dim=1)

        return (grid,)


class ExtractPBRChannel:
    """Extract a single channel from roughness/metallic/height as grayscale."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_mode": (["grayscale", "rgb"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "extract"
    CATEGORY = "StableMaterials/utils"

    def extract(self, image, output_mode):
        if output_mode == "grayscale":
            # Take first channel and replicate to RGB
            gray = image[:, :, :, 0:1]
            return (gray.repeat(1, 1, 1, 3),)
        return (image,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LoadStableMaterials": LoadStableMaterials,
    "StableMaterialsGenerate": StableMaterialsGenerate,
    "StableMaterialsFromImage": StableMaterialsFromImage,
    "CombinePBRMaps": CombinePBRMaps,
    "ExtractPBRChannel": ExtractPBRChannel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadStableMaterials": "Load StableMaterials",
    "StableMaterialsGenerate": "Generate PBR (Text)",
    "StableMaterialsFromImage": "Generate PBR (Image)",
    "CombinePBRMaps": "Combine PBR Grid",
    "ExtractPBRChannel": "Extract PBR Channel",
}
