"""
PBR Material Generation Inference Script

Run inference with trained model to generate PBR maps from RGB input.

Usage:
    python scripts/inference.py --model ./output/final_model --input texture.png --output ./results
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.pbr_pipeline import ChainedPBRPipeline, PBROutput


def load_model(model_path: str, device: str = "cuda") -> ChainedPBRPipeline:
    """Load trained model from checkpoint"""
    import yaml

    config_path = Path(model_path) / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create pipeline
    pipeline = ChainedPBRPipeline(
        pretrained_model=config["model"]["base_model"],
        vae_model=config["model"]["vae_model"],
    )

    # Load trained weights
    unet_path = Path(model_path) / "unet"
    if unet_path.exists():
        from diffusers import UNet2DConditionModel
        pipeline.unet.unet = UNet2DConditionModel.from_pretrained(
            str(unet_path),
            torch_dtype=torch.float16,
        )

    pipeline = pipeline.to(device)
    pipeline.eval()

    return pipeline


def preprocess_image(image_path: str, resolution: int = 1024) -> torch.Tensor:
    """Load and preprocess input image"""
    img = Image.open(image_path).convert("RGB")

    # Resize to target resolution
    img = img.resize((resolution, resolution), Image.LANCZOS)

    # Convert to tensor and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    tensor = transform(img).unsqueeze(0)
    return tensor


def save_pbr_output(output: PBROutput, output_dir: str, name: str):
    """Save all PBR maps to directory"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image"""
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2
        tensor = tensor.clamp(0, 1)

        # Handle grayscale
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)

        # Convert to numpy
        arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        arr = (arr * 255).astype(np.uint8)

        return Image.fromarray(arr)

    # Save each map
    maps = {
        "basecolor": output.basecolor,
        "normal": output.normal,
        "height": output.height,
        "roughness": output.roughness,
        "metalness": output.metalness,
        "ao": output.ao,
    }

    for map_name, tensor in maps.items():
        img = tensor_to_image(tensor)
        img.save(output_path / f"{name}_{map_name}.png")
        print(f"Saved {map_name} to {output_path / f'{name}_{map_name}.png'}")


def generate_pbr(
    model: ChainedPBRPipeline,
    image_path: str,
    output_dir: str,
    resolution: int = 1024,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    device: str = "cuda",
) -> PBROutput:
    """Generate PBR maps from input image"""
    # Preprocess input
    input_tensor = preprocess_image(image_path, resolution).to(device)

    # Generate
    with torch.no_grad():
        output = model.generate(
            input_tensor,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
        )

    # Save results
    name = Path(image_path).stem
    save_pbr_output(output, output_dir, name)

    return output


def main():
    parser = argparse.ArgumentParser(description="Generate PBR maps from RGB texture")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input RGB texture image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory for PBR maps",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Processing resolution",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda/cpu)",
    )

    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = load_model(args.model, args.device)

    print(f"Generating PBR maps for {args.input}...")
    generate_pbr(
        model=model,
        image_path=args.input,
        output_dir=args.output,
        resolution=args.resolution,
        num_steps=args.steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
    )

    print("Done!")


if __name__ == "__main__":
    main()
