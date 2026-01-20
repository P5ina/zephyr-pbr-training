# Zephyr PBR - ComfyUI Nodes

ComfyUI custom nodes for generating PBR material maps using [StableMaterials](https://huggingface.co/gvecchio/StableMaterials).

## Features

- Generate 5 PBR maps from text: **basecolor, normal, height, roughness, metallic**
- Generate PBR from input image
- Built-in **tileability** (seamless textures)
- Fast inference with **LCM** (4 steps) or quality with standard (50 steps)
- OpenRAIL license (commercial use allowed)

## Installation

### Option 1: Clone to custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/P5ina/zephyr-pbr-training.git zephyr-pbr
cd zephyr-pbr/comfyui_nodes
pip install -r requirements.txt
```

### Option 2: Symlink

```bash
ln -s /path/to/zephyr-pbr-training/comfyui_nodes ComfyUI/custom_nodes/zephyr-pbr
pip install -r ComfyUI/custom_nodes/zephyr-pbr/requirements.txt
```

Restart ComfyUI after installation.

## Nodes

| Node | Description |
|------|-------------|
| **Load StableMaterials** | Load the pipeline (LCM or Standard) |
| **Generate PBR (Text)** | Generate materials from text prompt |
| **Generate PBR (Image)** | Generate materials from input image |
| **Combine PBR Grid** | Combine 5 maps into preview grid |
| **Extract PBR Channel** | Convert grayscale maps to RGB |

## Basic Workflow

```
┌─────────────────────┐     ┌─────────────────────┐     ┌──────────────┐
│ Load StableMaterials│────▶│ Generate PBR (Text) │────▶│ Save Image   │
│   [LCM: True]       │     │                     │     │ (basecolor)  │
└─────────────────────┘     │ prompt: "rusty      │     └──────────────┘
                            │   metal surface"    │     ┌──────────────┐
                            │ steps: 4            │────▶│ Save Image   │
                            │ tileable: True      │     │ (normal)     │
                            │ seed: 12345         │     └──────────────┘
                            └─────────────────────┘     ... (5 outputs)
```

## Parameters

### Generate PBR (Text)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | - | Material description |
| `negative_prompt` | "blurry, low quality" | What to avoid |
| `seed` | 0 | Random seed |
| `steps` | 4 | Inference steps (4 for LCM, 50 for standard) |
| `guidance_scale` | 7.5 | Prompt adherence |
| `tileable` | True | Generate seamless texture |

### Generate PBR (Image)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | - | Input image |
| `prompt` | "" | Optional text guidance |
| `image_guidance_scale` | 1.5 | How much to follow input image |

## Output Maps

| Map | Description | Channels |
|-----|-------------|----------|
| **Basecolor** | Diffuse/albedo color | RGB |
| **Normal** | Surface normals (tangent space) | RGB |
| **Height** | Displacement map | Grayscale |
| **Roughness** | Surface roughness | Grayscale |
| **Metallic** | Metalness | Grayscale |

## Example Prompts

```
Weathered rusty metal with scratches and peeling paint
Polished marble with gold veins
Old wooden planks with moss and dirt
Rough concrete with cracks
Brushed aluminum with fingerprints
```

## Performance

| Mode | Steps | Time (RTX 4090) |
|------|-------|-----------------|
| LCM | 4 | ~2 sec |
| Standard | 50 | ~15 sec |

## Credits

- [StableMaterials](https://huggingface.co/gvecchio/StableMaterials) by Giuseppe Vecchio
- [MatSynth Dataset](https://huggingface.co/datasets/gvecchio/MatSynth)
