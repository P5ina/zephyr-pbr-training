"""
Prepare MatSynth dataset for SimpleTuner training.

Downloads MatSynth and converts to SimpleTuner format:
- image.png + image.txt (caption file)

Usage:
    python prepare_dataset.py --output ./data/pbr_dataset
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List
import json
import random

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# Material category to descriptive keywords
CATEGORY_KEYWORDS = {
    "metal": "metallic, metal surface, industrial",
    "wood": "wooden, wood grain, natural wood",
    "stone": "stone, rock surface, natural stone",
    "concrete": "concrete, cement, urban surface",
    "brick": "brick, masonry, wall",
    "fabric": "fabric, textile, cloth, woven",
    "leather": "leather, hide, natural leather",
    "marble": "marble, polished stone, veined",
    "plastic": "plastic, synthetic, polymer",
    "ceramic": "ceramic, tile, glazed",
    "ground": "ground, earth, soil, dirt",
    "plaster": "plaster, stucco, wall surface",
    "terracotta": "terracotta, clay, earthenware",
}

# Caption templates for variety
CAPTION_TEMPLATES = [
    "seamless tileable pbr texture of {material}, {keywords}, high quality, 4k",
    "seamless {material} texture, {keywords}, tileable, detailed surface",
    "pbr material texture, {material}, {keywords}, seamless pattern",
    "{material} surface texture, {keywords}, tileable, photorealistic",
    "high resolution {material} texture, {keywords}, seamless, pbr ready",
]


def generate_caption(material_name: str, category: str) -> str:
    """Generate a descriptive caption for the material."""
    # Clean up material name
    material = material_name.replace("_", " ").replace("-", " ").lower()

    # Get category keywords
    keywords = CATEGORY_KEYWORDS.get(category.lower(), "surface, detailed")

    # Pick random template
    template = random.choice(CAPTION_TEMPLATES)

    caption = template.format(material=material, keywords=keywords)
    return caption


def download_and_prepare(output_dir: str, split: str = "train", max_samples: int = None):
    """Download MatSynth and prepare for SimpleTuner."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading MatSynth dataset ({split} split)...")

    # Load dataset with streaming to save memory
    dataset = load_dataset(
        "gvecchio/MatSynth",
        split=split,
        streaming=True,
    )

    processed = 0
    stats = {"categories": {}, "total": 0}

    print("Processing materials...")
    for idx, sample in enumerate(tqdm(dataset)):
        if max_samples and processed >= max_samples:
            break

        try:
            # Get material info
            material_name = sample.get("name", f"material_{idx:05d}")
            category = sample.get("category", "unknown")

            # Get the basecolor/diffuse image (this is what we want to generate)
            img = None
            for key in ["basecolor", "diffuse", "albedo"]:
                if key in sample and sample[key] is not None:
                    img = sample[key]
                    break

            if img is None:
                continue

            # Convert to PIL if needed
            if not isinstance(img, Image.Image):
                continue

            # Resize to 1024 if larger
            if img.width > 1024 or img.height > 1024:
                img = img.resize((1024, 1024), Image.LANCZOS)

            # Convert to RGB
            img = img.convert("RGB")

            # Generate filename
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in material_name)
            filename = f"{category}_{safe_name}_{idx:05d}"

            # Save image
            img_path = output_path / f"{filename}.png"
            img.save(img_path, "PNG")

            # Generate and save caption
            caption = generate_caption(material_name, category)
            caption_path = output_path / f"{filename}.txt"
            with open(caption_path, "w") as f:
                f.write(caption)

            # Update stats
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
            stats["total"] += 1
            processed += 1

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    # Save stats
    stats_path = output_path / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset prepared!")
    print(f"Total images: {stats['total']}")
    print(f"Categories: {json.dumps(stats['categories'], indent=2)}")
    print(f"Output directory: {output_path}")


def add_augmented_captions(data_dir: str):
    """Add caption variations for existing images."""
    data_path = Path(data_dir)

    for txt_file in data_path.glob("*.txt"):
        # Read existing caption
        with open(txt_file) as f:
            original = f.read().strip()

        # Parse category from filename
        filename = txt_file.stem
        parts = filename.split("_")
        category = parts[0] if parts else "unknown"

        # Generate new caption with different template
        keywords = CATEGORY_KEYWORDS.get(category.lower(), "surface, detailed")
        material = " ".join(parts[1:-1]) if len(parts) > 2 else "material"

        # Add some variations
        variations = [
            original,
            f"tileable {material} pbr texture, {keywords}, seamless, high detail",
            f"seamless {material}, {keywords}, 4k texture, photorealistic",
        ]

        # Pick one randomly (or could keep original)
        new_caption = random.choice(variations)

        with open(txt_file, "w") as f:
            f.write(new_caption)


def verify_dataset(data_dir: str):
    """Verify dataset integrity."""
    data_path = Path(data_dir)

    images = list(data_path.glob("*.png")) + list(data_path.glob("*.jpg"))
    captions = list(data_path.glob("*.txt"))

    print(f"Images found: {len(images)}")
    print(f"Caption files found: {len(captions)}")

    # Check for missing captions
    missing_captions = []
    for img in images:
        caption_path = img.with_suffix(".txt")
        if not caption_path.exists():
            missing_captions.append(img.name)

    if missing_captions:
        print(f"\nMissing captions for {len(missing_captions)} images:")
        for name in missing_captions[:10]:
            print(f"  - {name}")
        if len(missing_captions) > 10:
            print(f"  ... and {len(missing_captions) - 10} more")
    else:
        print("\nAll images have caption files!")

    # Show sample captions
    print("\nSample captions:")
    for txt in list(captions)[:5]:
        with open(txt) as f:
            print(f"  {txt.name}: {f.read().strip()[:80]}...")


def main():
    parser = argparse.ArgumentParser(description="Prepare MatSynth for SimpleTuner")
    parser.add_argument(
        "--output",
        type=str,
        default="./data/pbr_dataset",
        help="Output directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process (for testing)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing dataset",
    )

    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.output)
    else:
        download_and_prepare(args.output, args.split, args.max_samples)
        verify_dataset(args.output)


if __name__ == "__main__":
    main()
