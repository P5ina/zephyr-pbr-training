# Zephyr PBR Training

Train a multi-output SDXL model for generating PBR texture maps (basecolor, normal, roughness, height) from text prompts.

## Architecture

Custom pipeline based on SDXL with map-specific output heads. Each forward pass generates noise predictions for a specific PBR map type.

## Quick Start (Vast.ai)

### 1. Rent GPU

- **Image**: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`
- **GPU**: RTX 4090 (recommended) or A100
- **Disk**: 100GB+ SSD

### 2. SSH & Setup

```bash
ssh -p <port> root@<ip>

cd /workspace
git clone https://github.com/P5ina/zephyr-pbr-training.git
cd zephyr-pbr-training

pip install -r requirements.txt
```

### 3. Login

```bash
# HF_TOKEN already set in Vast.ai env
wandb login  # For monitoring training with images
```

### 4. Prepare Dataset

Downloads MatSynth and saves all PBR maps per material.

```bash
# Full dataset (~4000 materials)
python scripts/prepare_dataset.py --output ./data/pbr_dataset

# Quick test (100 samples)
python scripts/prepare_dataset.py --output ./data/pbr_dataset --max-samples 100

# Verify
python scripts/prepare_dataset.py --output ./data/pbr_dataset --verify
```

Dataset structure:
```
data/pbr_dataset/
├── 00001_material_name/
│   ├── basecolor.png
│   ├── normal.png
│   ├── roughness.png
│   ├── height.png
│   └── meta.json
└── ...
```

### 5. Train

```bash
tmux new -s training
bash train.sh
# Detach: Ctrl+B, D
```

Or manually:
```bash
accelerate launch scripts/train.py --config config.yaml
```

### 6. Monitor

Open [wandb.ai](https://wandb.ai) to see:
- Training loss curves
- Validation images (all 4 PBR maps)
- Combined grid view

## Configuration

Edit `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 1 | Training batch size |
| `max_train_steps` | 20000 | Total training steps |
| `learning_rate` | 1e-5 | Learning rate |
| `validation_steps` | 500 | Generate samples every N steps |
| `resolution` | 512 | Training resolution |

### Memory Optimization

**RTX 4090 (24GB):**
```yaml
batch_size: 1
gradient_checkpointing: true
enable_xformers: true
```

**A100 (40GB+):**
```yaml
batch_size: 2
```

## Output

```
output/
├── checkpoint-2000/
│   ├── unet/
│   ├── map_heads.pt
│   └── config.yaml
├── validation/
│   └── step_500/
│       ├── basecolor.png
│       ├── normal.png
│       ├── roughness.png
│       └── height.png
└── final_model/
```

## Inference

```bash
python scripts/inference.py \
    --model ./output/final_model \
    --input texture.png \
    --output ./results
```

## PBR Maps

| Map | Description |
|-----|-------------|
| Basecolor | Albedo/diffuse color |
| Normal | Surface normals (tangent space) |
| Roughness | Surface roughness (0=smooth, 1=rough) |
| Height | Displacement/bump map |

## Dataset

[MatSynth](https://huggingface.co/datasets/gvecchio/MatSynth) - ~4000 PBR materials (CC-BY license).
