"""
Prepare MatSynth dataset for PBR multi-output training.

Downloads all PBR maps: basecolor, normal, roughness, height.

Usage:
    python scripts/prepare_dataset.py --output ./data/pbr_dataset --max-samples 1000
"""

import os
import argparse
from pathlib import Path
import json

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# PBR maps we want to extract
PBR_MAPS = ["basecolor", "normal", "roughness", "height"]


def process_image(img, target_size: int = 1024) -> Image.Image:
    """Process and resize image."""
    if not isinstance(img, Image.Image):
        return None

    # Resize if needed
    if img.width != target_size or img.height != target_size:
        img = img.resize((target_size, target_size), Image.LANCZOS)

    # Convert to RGB (normal maps need this too for saving)
    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def download_and_prepare(output_dir: str, split: str = "train", max_samples: int = None, resolution: int = 1024):
    """Download MatSynth and prepare all PBR maps."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading MatSynth dataset ({split} split)...")
    print(f"Target resolution: {resolution}x{resolution}")
    print(f"PBR maps: {', '.join(PBR_MAPS)}")

    dataset = load_dataset(
        "gvecchio/MatSynth",
        split=split,
        streaming=True,
    )

    processed = 0
    skipped = 0
    stats = {"total": 0, "categories": {}}

    print("\nProcessing materials...")
    for idx, sample in enumerate(tqdm(dataset)):
        if max_samples and processed >= max_samples:
            break

        try:
            # Get material info
            material_name = str(sample.get("name", f"material_{idx:05d}"))
            category = str(sample.get("category", "unknown"))

            # Get metadata
            metadata = sample.get("metadata", {}) or {}
            tags = metadata.get("tags", []) if isinstance(metadata, dict) else []
            description = metadata.get("description", "") if isinstance(metadata, dict) else ""

            # Check if all required maps exist
            maps = {}
            all_maps_exist = True

            for map_name in PBR_MAPS:
                # Try different naming conventions
                img = None
                for key in [map_name, f"{map_name}_map", map_name.replace("basecolor", "diffuse")]:
                    if key in sample and sample[key] is not None:
                        img = process_image(sample[key], resolution)
                        break

                if img is None:
                    all_maps_exist = False
                    break

                maps[map_name] = img

            if not all_maps_exist:
                skipped += 1
                continue

            # Create material directory
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in material_name)
            material_dir = output_path / f"{idx:05d}_{safe_name}"
            material_dir.mkdir(exist_ok=True)

            # Save all maps
            for map_name, img in maps.items():
                img.save(material_dir / f"{map_name}.png", "PNG")

            # Save caption/metadata
            if description:
                caption = description
            elif tags:
                caption = ", ".join(tags) if isinstance(tags, list) else str(tags)
            else:
                caption = f"{material_name}, {category}"

            meta = {
                "name": material_name,
                "category": category,
                "caption": caption,
                "tags": tags if isinstance(tags, list) else [],
            }
            with open(material_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            # Update stats
            stats["total"] += 1
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
            processed += 1

            if processed % 100 == 0:
                print(f"\n  Processed: {processed}, Skipped: {skipped}")

        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            skipped += 1
            continue

    # Save stats
    stats["skipped"] = skipped
    with open(output_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Dataset prepared!")
    print(f"Total materials: {stats['total']}")
    print(f"Skipped (missing maps): {skipped}")
    print(f"Output: {output_path}")
    print(f"\nCategories:")
    for cat, count in sorted(stats["categories"].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


def verify_dataset(data_dir: str):
    """Verify dataset integrity."""
    data_path = Path(data_dir)

    materials = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"Materials found: {len(materials)}")

    complete = 0
    incomplete = []

    for mat_dir in materials:
        has_all = all((mat_dir / f"{m}.png").exists() for m in PBR_MAPS)
        if has_all:
            complete += 1
        else:
            missing = [m for m in PBR_MAPS if not (mat_dir / f"{m}.png").exists()]
            incomplete.append((mat_dir.name, missing))

    print(f"Complete materials: {complete}")
    print(f"Incomplete: {len(incomplete)}")

    if incomplete and len(incomplete) <= 10:
        for name, missing in incomplete:
            print(f"  {name}: missing {missing}")

    # Show sample
    if materials:
        sample = materials[0]
        print(f"\nSample material: {sample.name}")
        for f in sample.iterdir():
            print(f"  - {f.name}")


def main():
    parser = argparse.ArgumentParser(description="Prepare MatSynth for PBR training")
    parser.add_argument("--output", type=str, default="./data/pbr_dataset")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--verify", action="store_true")

    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.output)
    else:
        download_and_prepare(args.output, args.split, args.max_samples, args.resolution)
        print()
        verify_dataset(args.output)


if __name__ == "__main__":
    main()
