"""
PBR Multi-Output Model Training

Trains a model to generate 4 PBR maps (basecolor, normal, roughness, height)
from a text prompt. Uses WandB for monitoring with image logging.

Usage:
    python scripts/train.py --config config.yaml
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml
import wandb

from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed


PBR_MAPS = ["basecolor", "normal", "roughness", "height"]


class PBRDataset(Dataset):
    """Dataset for PBR materials with all maps."""

    def __init__(self, data_dir: str, resolution: int = 512):
        self.data_dir = Path(data_dir)
        self.resolution = resolution

        # Find all material directories
        self.materials = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and (d / "basecolor.png").exists()
        ])

        print(f"Found {len(self.materials)} materials")

    def __len__(self):
        return len(self.materials)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mat_dir = self.materials[idx]

        # Load all PBR maps
        maps = {}
        for map_name in PBR_MAPS:
            img_path = mat_dir / f"{map_name}.png"
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
                # Normalize to [-1, 1]
                arr = np.array(img).astype(np.float32) / 127.5 - 1.0
                maps[map_name] = torch.from_numpy(arr).permute(2, 0, 1)
            else:
                # Placeholder
                maps[map_name] = torch.zeros(3, self.resolution, self.resolution)

        # Load caption
        meta_path = mat_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            caption = meta.get("caption", "pbr texture")
        else:
            caption = "seamless tileable pbr texture"

        return {
            "basecolor": maps["basecolor"],
            "normal": maps["normal"],
            "roughness": maps["roughness"],
            "height": maps["height"],
            "caption": caption,
            "name": mat_dir.name,
        }


class PBRGenerator(nn.Module):
    """
    Multi-output PBR generator.
    Generates 4 PBR maps from a single forward pass.
    """

    def __init__(self, base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        super().__init__()

        # Load SDXL components
        print("Loading SDXL components...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )

        self.unet = pipe.unet
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

        # Freeze VAE
        self.vae.requires_grad_(False)

        # Output heads for each PBR map (lightweight projection)
        # These project from UNet output space to each map type
        latent_channels = self.unet.config.out_channels
        self.map_heads = nn.ModuleDict({
            map_name: nn.Conv2d(latent_channels, latent_channels, 1)
            for map_name in PBR_MAPS
        })

        del pipe
        torch.cuda.empty_cache()

    def encode_prompt(self, prompt: str, device: torch.device):
        """Encode text prompt using SDXL dual encoders."""
        # Tokenize
        tokens_1 = self.tokenizer(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(device)

        tokens_2 = self.tokenizer_2(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(device)

        # Encode
        with torch.no_grad():
            embeds_1 = self.text_encoder(tokens_1, output_hidden_states=True)
            embeds_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)

            pooled = embeds_2[0]
            prompt_embeds = torch.cat([
                embeds_1.hidden_states[-2],
                embeds_2.hidden_states[-2]
            ], dim=-1)

        return prompt_embeds, pooled

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space."""
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images."""
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            images = self.vae.decode(latents).sample
        return images

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        target_map: str = "basecolor",
    ) -> torch.Tensor:
        """Forward pass for a specific PBR map."""
        # SDXL needs added conditions
        add_time_ids = torch.tensor(
            [[1024, 1024, 0, 0, 1024, 1024]],
            device=latents.device, dtype=prompt_embeds.dtype
        ).repeat(latents.shape[0], 1)

        added_cond_kwargs = {
            "text_embeds": pooled_embeds,
            "time_ids": add_time_ids,
        }

        # UNet forward
        noise_pred = self.unet(
            latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # Apply map-specific head
        noise_pred = self.map_heads[target_map](noise_pred)

        return noise_pred


def train(config: dict):
    """Main training loop."""
    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision=config["training"]["mixed_precision"],
        log_with="wandb" if config["logging"]["use_wandb"] else None,
    )

    set_seed(42)

    # Initialize wandb
    if accelerator.is_main_process and config["logging"]["use_wandb"]:
        wandb.init(
            project=config["logging"]["project_name"],
            config=config,
        )

    # Load model
    print("Loading model...")
    model = PBRGenerator(config["model"]["base_model"])

    # Enable optimizations
    if config["training"]["gradient_checkpointing"]:
        model.unet.enable_gradient_checkpointing()

    if config["training"]["enable_xformers"]:
        try:
            model.unet.enable_xformers_memory_efficient_attention()
            print("xformers enabled")
        except:
            print("xformers not available")

    # Dataset
    print("Loading dataset...")
    dataset = PBRDataset(
        config["training"]["data_dir"],
        config["training"]["resolution"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer
    trainable_params = list(model.unet.parameters()) + list(model.map_heads.parameters())

    if config["training"]["use_8bit_adam"]:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                trainable_params,
                lr=config["training"]["learning_rate"],
                betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
                weight_decay=config["training"]["adam_weight_decay"],
            )
            print("Using 8-bit Adam")
        except ImportError:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config["training"]["learning_rate"],
            )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config["training"]["learning_rate"],
        )

    # Prepare with accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Training loop
    global_step = 0
    max_steps = config["training"]["max_train_steps"]

    print(f"\nStarting training for {max_steps} steps...")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Gradient accumulation: {config['training']['gradient_accumulation_steps']}")

    progress = tqdm(total=max_steps, desc="Training")

    model.train()
    while global_step < max_steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                # Get prompt embeddings
                prompt_embeds, pooled = model.module.encode_prompt(
                    batch["caption"][0],  # Use first caption in batch
                    accelerator.device,
                )
                prompt_embeds = prompt_embeds.repeat(len(batch["caption"]), 1, 1)
                pooled = pooled.repeat(len(batch["caption"]), 1)

                total_loss = 0

                # Train on each PBR map
                for map_name in PBR_MAPS:
                    target = batch[map_name].to(accelerator.device)

                    # Encode target
                    target_latent = model.module.encode_images(target.half())

                    # Sample noise and timesteps
                    noise = torch.randn_like(target_latent)
                    timesteps = torch.randint(
                        0, model.module.scheduler.config.num_train_timesteps,
                        (target_latent.shape[0],), device=accelerator.device
                    ).long()

                    # Add noise
                    noisy_latent = model.module.scheduler.add_noise(target_latent, noise, timesteps)

                    # Predict noise
                    noise_pred = model(
                        noisy_latent,
                        timesteps,
                        prompt_embeds,
                        pooled,
                        target_map=map_name,
                    )

                    # Loss
                    loss = F.mse_loss(noise_pred, noise)
                    total_loss += loss

                # Backward
                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, config["training"]["max_grad_norm"])

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)

                # Logging
                if global_step % config["logging"]["log_every_n_steps"] == 0:
                    log_dict = {"loss": total_loss.item(), "step": global_step}

                    if config["logging"]["use_wandb"] and accelerator.is_main_process:
                        wandb.log(log_dict)

                    progress.set_postfix(loss=f"{total_loss.item():.4f}")

                # Validation with image logging
                if global_step % config["training"]["validation_steps"] == 0:
                    if accelerator.is_main_process:
                        validate_and_log(model, config, global_step, accelerator.device)

                # Save checkpoint
                if global_step % config["checkpointing"]["save_steps"] == 0:
                    if accelerator.is_main_process:
                        save_checkpoint(model, config, global_step)

                if global_step >= max_steps:
                    break

    progress.close()

    # Final save
    if accelerator.is_main_process:
        save_checkpoint(model, config, global_step, final=True)

    if config["logging"]["use_wandb"]:
        wandb.finish()

    print("Training complete!")


@torch.no_grad()
def validate_and_log(model, config: dict, step: int, device: torch.device):
    """Generate validation images and log to wandb."""
    model.eval()

    prompt = config["training"]["validation_prompt"]
    print(f"\nValidation at step {step}...")
    print(f"Prompt: {prompt}")

    # Get model (unwrap if needed)
    m = model.module if hasattr(model, "module") else model

    # Encode prompt
    prompt_embeds, pooled = m.encode_prompt(prompt, device)

    # Generate each map
    images = {}
    for map_name in PBR_MAPS:
        # Start from noise
        latent = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)

        # Simple denoising (few steps for validation)
        m.scheduler.set_timesteps(20, device=device)
        for t in m.scheduler.timesteps:
            noise_pred = m(latent, t.unsqueeze(0), prompt_embeds, pooled, target_map=map_name)
            latent = m.scheduler.step(noise_pred, t, latent).prev_sample

        # Decode
        img = m.decode_latents(latent)
        img = (img.clamp(-1, 1) + 1) / 2  # [0, 1]
        img = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        images[map_name] = img

    # Log to wandb
    if config["logging"]["use_wandb"]:
        wandb_images = {
            f"val/{name}": wandb.Image(img, caption=f"{name} - step {step}")
            for name, img in images.items()
        }

        # Create combined grid
        grid = np.concatenate([
            np.concatenate([images["basecolor"], images["normal"]], axis=1),
            np.concatenate([images["roughness"], images["height"]], axis=1),
        ], axis=0)
        wandb_images["val/grid"] = wandb.Image(grid, caption=f"PBR Grid - step {step}")

        wandb.log(wandb_images, step=step)

    # Also save locally
    out_dir = Path(config["checkpointing"]["output_dir"]) / "validation" / f"step_{step}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, img in images.items():
        Image.fromarray(img).save(out_dir / f"{name}.png")

    print(f"Validation images saved to {out_dir}")

    model.train()


def save_checkpoint(model, config: dict, step: int, final: bool = False):
    """Save model checkpoint."""
    out_dir = Path(config["checkpointing"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    m = model.module if hasattr(model, "module") else model

    if final:
        save_path = out_dir / "final_model"
    else:
        save_path = out_dir / f"checkpoint-{step}"

    save_path.mkdir(exist_ok=True)

    # Save UNet
    m.unet.save_pretrained(save_path / "unet", safe_serialization=True)

    # Save map heads
    torch.save(m.map_heads.state_dict(), save_path / "map_heads.pt")

    # Save config
    with open(save_path / "config.yaml", "w") as f:
        yaml.dump(config, f)

    print(f"Checkpoint saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
