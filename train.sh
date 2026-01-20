#!/bin/bash
# PBR Multi-Output Training Script for Vast.ai

set -e

echo "=== PBR Multi-Output Model Training ==="

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Login to WandB (use WANDB_API_KEY env var)
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
fi

# Prepare dataset
echo "Preparing dataset..."
python scripts/prepare_dataset.py \
    --output ./data/pbr_dataset \
    --max-samples 1000 \
    --resolution 1024

# Start training
echo "Starting training..."
accelerate launch scripts/train.py --config config.yaml

echo "Training complete!"
