"""
ComfyUI Custom Nodes for PBR Material Generation

Available nodes:
- StableMaterials: Generate PBR maps using pretrained StableMaterials model
"""

from .stable_materials.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
