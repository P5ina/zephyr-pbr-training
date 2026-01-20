# Zephyr PBR Training

Train SDXL LoRA for generating seamless PBR textures using [SimpleTuner](https://github.com/bghira/SimpleTuner).

## Quick Start (Vast.ai)

### 1. Rent GPU

- **Image**: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`
- **GPU**: RTX 4090 (recommended) or A100
- **Disk**: 100GB+ SSD

### 2. SSH & Setup

```bash
ssh -p <port> root@<ip>

cd /workspace
git clone https://github.com/<your-username>/zephyr-pbr-training.git
cd zephyr-pbr-training

pip install 'simpletuner[cuda]'
pip install datasets pillow albumentations tqdm
```

### 3. Login

```bash
huggingface-cli login
wandb login  # optional, for monitoring
```

### 4. Prepare Dataset

```bash
# Full dataset (~4000 materials)
python prepare_dataset.py --output ./data/pbr_dataset

# Quick test (100 samples)
python prepare_dataset.py --output ./data/pbr_dataset --max-samples 100
```

### 5. Train

```bash
tmux new -s training
bash train.sh
# Detach: Ctrl+B, D
```

## Configuration

### `config.json`

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--model_family` | `sdxl` | Stable Diffusion XL |
| `--lora_rank` | `64` | LoRA capacity |
| `--learning_rate` | `1e-4` | Learning rate |
| `--max_train_steps` | `10000` | Total steps |
| `--train_batch_size` | `2` | Batch size |

### Memory Optimization

**RTX 4090 (24GB):**
```json
"--train_batch_size": 2,
"--gradient_checkpointing": "true"
```

**16GB VRAM:**
```json
"--train_batch_size": 1,
"--lora_rank": 32
```

## Output

```
output/pbr-texture-lora/
└── pytorch_lora_weights.safetensors
```

## Using the LoRA

### ComfyUI

1. Copy `.safetensors` to `ComfyUI/models/loras/`
2. Use "Load LoRA" node with SDXL

### Prompts

```
seamless tileable pbr texture of rusty metal, 4k, high detail
seamless wood texture, oak planks, tileable, photorealistic
```

## Dataset

[MatSynth](https://huggingface.co/datasets/gvecchio/MatSynth) - ~4000 PBR materials (CC-BY).

## Resources

- [SimpleTuner](https://github.com/bghira/SimpleTuner)
- [MatSynth](https://gvecchio.com/matsynth/)
