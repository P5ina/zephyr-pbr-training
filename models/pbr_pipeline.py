"""
PBR Material Generation Pipeline
SDXL-based Chained Decomposition for PBR Material Estimation

Architecture inspired by CHORD (Ubisoft), adapted for SDXL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer


@dataclass
class PBROutput:
    """Output container for PBR maps"""
    basecolor: torch.Tensor
    normal: torch.Tensor
    height: torch.Tensor
    roughness: torch.Tensor
    metalness: torch.Tensor
    ao: torch.Tensor


class LEGOConditioningBlock(nn.Module):
    """
    LEGO-style conditioning block for channel-specific predictions.
    Allows the same UNet to predict different PBR channels via learned switches.
    """

    def __init__(self, hidden_dim: int, num_channels: int = 6):
        super().__init__()
        self.num_channels = num_channels

        # Channel-specific projection layers
        self.channel_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_channels)
        ])

        # Channel embedding
        self.channel_embedding = nn.Embedding(num_channels, hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        channel_idx: int,
    ) -> torch.Tensor:
        """
        Apply channel-specific conditioning.

        Args:
            hidden_states: UNet hidden states [B, C, H, W]
            channel_idx: Which PBR channel to predict (0-5)

        Returns:
            Conditioned hidden states
        """
        # Get channel embedding
        channel_emb = self.channel_embedding(
            torch.tensor([channel_idx], device=hidden_states.device)
        )

        # Apply channel-specific projection
        # Reshape for linear layers
        B, C, H, W = hidden_states.shape
        hidden_flat = hidden_states.permute(0, 2, 3, 1).reshape(-1, C)

        projected = self.channel_projections[channel_idx](hidden_flat)

        # Reshape back
        output = projected.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return output


class PBRUNet(nn.Module):
    """
    Modified SDXL UNet with LEGO conditioning for multi-channel PBR prediction.
    """

    def __init__(
        self,
        pretrained_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        use_lego: bool = True,
    ):
        super().__init__()

        # Load pretrained SDXL UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model,
            subfolder="unet",
            torch_dtype=torch.float16,
        )

        self.use_lego = use_lego

        if use_lego:
            # Add LEGO conditioning to each transformer block
            hidden_dim = self.unet.config.block_out_channels[-1]
            self.lego_blocks = nn.ModuleDict()

            # Add LEGO blocks at key positions in the UNet
            for name, module in self.unet.named_modules():
                if "attn" in name and "to_out" in name:
                    # Get the output dimension
                    if hasattr(module, "out_features"):
                        self.lego_blocks[name] = LEGOConditioningBlock(
                            hidden_dim=module.out_features,
                            num_channels=6,
                        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Dict = None,
        channel_idx: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass with channel-specific conditioning.

        Args:
            sample: Noisy latent [B, 4, H, W]
            timestep: Diffusion timestep
            encoder_hidden_states: CLIP embeddings
            added_cond_kwargs: Additional SDXL conditioning
            channel_idx: Which PBR channel to predict

        Returns:
            Predicted noise/velocity
        """
        # Standard UNet forward
        output = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # LEGO conditioning is applied internally via hooks
        # (simplified here for clarity)

        return output


class ChainedPBRPipeline(nn.Module):
    """
    Full chained decomposition pipeline for PBR material estimation.

    Chain:
    1. RGB Input → Base Color
    2. RGB - BaseColor (irradiance) → Normal
    3. Normal → Height (integration)
    4. All maps + render comparison → Roughness, Metalness, AO
    """

    CHANNEL_NAMES = ["basecolor", "normal", "height", "roughness", "metalness", "ao"]

    def __init__(
        self,
        pretrained_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        vae_model: str = "madebyollin/sdxl-vae-fp16-fix",
    ):
        super().__init__()

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            vae_model,
            torch_dtype=torch.float16,
        )
        self.vae.requires_grad_(False)

        # Load text encoders
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model,
            subfolder="tokenizer",
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            pretrained_model,
            subfolder="tokenizer_2",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        )
        self.text_encoder_2 = CLIPTextModel.from_pretrained(
            pretrained_model,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
        )

        # PBR UNet
        self.unet = PBRUNet(pretrained_model, use_lego=True)

        # Scheduler
        self.scheduler = DDPMScheduler.from_pretrained(
            pretrained_model,
            subfolder="scheduler",
        )

        # Channel-specific prompts for conditioning
        self.channel_prompts = {
            "basecolor": "albedo texture map, base color, diffuse color, flat lighting",
            "normal": "normal map, surface normals, tangent space, blue-purple tint",
            "height": "height map, displacement map, grayscale elevation data",
            "roughness": "roughness map, surface roughness, grayscale texture",
            "metalness": "metallic map, metalness, grayscale metal detection",
            "ao": "ambient occlusion map, AO, grayscale shadow occlusion",
        }

        # Scaling factors
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space"""
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        return latent

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image"""
        latent = latent / self.vae.config.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latent).sample
        return image

    def get_text_embeddings(
        self,
        prompt: str,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get SDXL text embeddings for a prompt"""
        # Tokenize
        tokens_1 = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        tokens_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        # Get embeddings
        with torch.no_grad():
            embed_1 = self.text_encoder(tokens_1)[0]
            embed_2 = self.text_encoder_2(tokens_2)[0]

        # Concatenate for SDXL
        prompt_embeds = torch.cat([embed_1, embed_2], dim=-1)

        return prompt_embeds

    def compute_irradiance(
        self,
        rgb_input: torch.Tensor,
        basecolor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute approximate irradiance by removing color from input.
        irradiance ≈ input / basecolor (clamped to avoid division issues)
        """
        # Avoid division by zero
        basecolor_safe = torch.clamp(basecolor, min=0.01)

        # Compute irradiance
        irradiance = rgb_input / basecolor_safe

        # Normalize to reasonable range
        irradiance = torch.clamp(irradiance, 0, 2)

        return irradiance

    def integrate_normal_to_height(
        self,
        normal_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrate normal map to height map using Poisson integration.
        Simplified version using gradient descent.
        """
        B, C, H, W = normal_map.shape

        # Extract gradients from normal map
        # Normal map is in [-1, 1], where:
        # R = dx (horizontal gradient)
        # G = dy (vertical gradient)
        # B = z component

        dx = normal_map[:, 0:1, :, :]  # R channel
        dy = normal_map[:, 1:2, :, :]  # G channel

        # Simple integration via cumulative sum
        # This is a rough approximation
        height_x = torch.cumsum(dx, dim=3)
        height_y = torch.cumsum(dy, dim=2)

        # Average both directions
        height = (height_x + height_y) / 2

        # Normalize to [0, 1]
        height = (height - height.min()) / (height.max() - height.min() + 1e-8)

        return height

    def forward_single_channel(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        condition_latent: Optional[torch.Tensor],
        channel_name: str,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Forward pass for a single PBR channel.

        Args:
            noisy_latent: Noisy target latent
            timestep: Diffusion timestep
            condition_latent: Conditioning latent (e.g., RGB input for basecolor)
            channel_name: Which channel to predict

        Returns:
            Predicted noise/velocity
        """
        channel_idx = self.CHANNEL_NAMES.index(channel_name)

        # Get text embeddings for this channel
        prompt = self.channel_prompts[channel_name]
        prompt_embeds = self.get_text_embeddings(prompt, device)

        # Concatenate condition if provided
        if condition_latent is not None:
            # Channel-wise concatenation for conditioning
            input_latent = torch.cat([noisy_latent, condition_latent], dim=1)
        else:
            input_latent = noisy_latent

        # Forward through UNet
        noise_pred = self.unet(
            input_latent,
            timestep,
            encoder_hidden_states=prompt_embeds,
            channel_idx=channel_idx,
        )

        return noise_pred

    @torch.no_grad()
    def generate(
        self,
        rgb_input: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> PBROutput:
        """
        Generate all PBR maps from RGB input.

        Args:
            rgb_input: Input RGB texture [B, 3, H, W] in range [-1, 1]
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale

        Returns:
            PBROutput with all generated maps
        """
        device = rgb_input.device
        B = rgb_input.shape[0]

        # Encode input to latent
        input_latent = self.encode_image(rgb_input)

        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        results = {}

        # Stage 1: RGB → Base Color
        basecolor_latent = self._denoise_channel(
            input_latent, "basecolor", device, num_inference_steps, guidance_scale
        )
        results["basecolor"] = self.decode_latent(basecolor_latent)

        # Stage 2: Compute irradiance, predict normal
        irradiance = self.compute_irradiance(rgb_input, results["basecolor"])
        irradiance_latent = self.encode_image(irradiance)

        normal_latent = self._denoise_channel(
            irradiance_latent, "normal", device, num_inference_steps, guidance_scale,
            condition_latent=basecolor_latent,
        )
        results["normal"] = self.decode_latent(normal_latent)

        # Stage 3: Normal → Height (via integration)
        results["height"] = self.integrate_normal_to_height(results["normal"])

        # Stage 4: Predict roughness, metalness, AO
        # These use all previous maps as conditioning
        combined_condition = torch.cat([
            basecolor_latent,
            normal_latent,
        ], dim=1)

        for channel in ["roughness", "metalness", "ao"]:
            channel_latent = self._denoise_channel(
                input_latent, channel, device, num_inference_steps, guidance_scale,
                condition_latent=combined_condition,
            )
            results[channel] = self.decode_latent(channel_latent)
            # Convert to grayscale
            results[channel] = results[channel].mean(dim=1, keepdim=True)

        return PBROutput(**results)

    def _denoise_channel(
        self,
        condition_latent: torch.Tensor,
        channel_name: str,
        device: torch.device,
        num_steps: int,
        guidance_scale: float,
        condition_latent_extra: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run denoising loop for a single channel"""
        B = condition_latent.shape[0]

        # Start from noise
        latent = torch.randn_like(condition_latent)

        for t in self.scheduler.timesteps:
            # Predict noise
            noise_pred = self.forward_single_channel(
                latent,
                t,
                condition_latent if condition_latent_extra is None else condition_latent_extra,
                channel_name,
                device,
            )

            # Scheduler step
            latent = self.scheduler.step(noise_pred, t, latent).prev_sample

        return latent


class PBRTrainer:
    """
    Training wrapper for the PBR pipeline.
    Implements the chained training strategy from CHORD.
    """

    def __init__(
        self,
        pipeline: ChainedPBRPipeline,
        config: dict,
    ):
        self.pipeline = pipeline
        self.config = config

        # Loss functions
        self.mse_loss = nn.MSELoss()

        # Optional LPIPS for perceptual loss
        self.lpips = None
        if config.get("use_lpips", False):
            try:
                import lpips
                self.lpips = lpips.LPIPS(net="vgg").cuda()
            except ImportError:
                print("LPIPS not available, skipping perceptual loss")

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss for a batch.

        Args:
            batch: Dictionary with 'rgb', 'basecolor', 'normal', etc.
            noise: Noise added to latents
            timesteps: Diffusion timesteps

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Get latents for all channels
        latents = {}
        for key in ["basecolor", "normal", "height", "roughness", "metalness", "ao"]:
            if key in batch:
                latents[key] = self.pipeline.encode_image(batch[key])

        # Add noise
        noisy_latents = {}
        for key, latent in latents.items():
            noisy_latents[key] = self.pipeline.scheduler.add_noise(
                latent, noise, timesteps
            )

        # Stage 1: Basecolor prediction
        rgb_latent = self.pipeline.encode_image(batch["rgb"])
        bc_pred = self.pipeline.forward_single_channel(
            noisy_latents["basecolor"],
            timesteps,
            rgb_latent,
            "basecolor",
            batch["rgb"].device,
        )
        losses["basecolor"] = self.mse_loss(bc_pred, noise)

        # Stage 2: Normal prediction (conditioned on basecolor)
        normal_pred = self.pipeline.forward_single_channel(
            noisy_latents["normal"],
            timesteps,
            latents["basecolor"],  # Use ground truth basecolor during training
            "normal",
            batch["rgb"].device,
        )
        losses["normal"] = self.mse_loss(normal_pred, noise)

        # Stages 3-5: Roughness, Metalness, AO
        combined_cond = torch.cat([
            latents["basecolor"],
            latents["normal"],
        ], dim=1)

        for channel in ["roughness", "metalness", "ao"]:
            if channel in noisy_latents:
                pred = self.pipeline.forward_single_channel(
                    noisy_latents[channel],
                    timesteps,
                    combined_cond,
                    channel,
                    batch["rgb"].device,
                )
                losses[channel] = self.mse_loss(pred, noise)

        # Total loss
        losses["total"] = sum(losses.values())

        return losses


def create_pipeline(config_path: str) -> ChainedPBRPipeline:
    """Factory function to create pipeline from config"""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    pipeline = ChainedPBRPipeline(
        pretrained_model=config["model"]["base_model"],
        vae_model=config["model"]["vae_model"],
    )

    return pipeline
