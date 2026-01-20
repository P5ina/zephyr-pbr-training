#!/bin/bash
# SimpleTuner training script for PBR textures

set -e

cd "$(dirname "$0")"

# Check if SimpleTuner is installed
if ! python -c "import simpletuner" 2>/dev/null; then
    echo "SimpleTuner not found. Installing..."
    pip install 'simpletuner[cuda]'
fi

# Check if dataset exists
if [ ! -d "./data/pbr_dataset" ]; then
    echo "Dataset not found. Preparing dataset..."
    python prepare_dataset.py --output ./data/pbr_dataset
fi

# Create directories
mkdir -p ./cache/vae/pbr ./cache/text ./output ./config

# Run training
echo "Starting training..."
simpletuner \
    --config_path ./config.json

echo "Training complete!"
