"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from rfqa.blender_dataparser import BlenderDataParserConfig


rfqa_blender_dataparser = DataParserSpecification(config=BlenderDataParserConfig())