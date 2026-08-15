"""Video Editor package."""

from src.editor.export import run_export
from src.editor.export_plan import ExportPlan, build_export_plan
from src.editor.history import HistoryStack
from src.editor.models import (
    AspectRatio,
    AudioSettings,
    CaptionCue,
    CropSettings,
    ProjectState,
    TextOverlay,
    TimelineRange,
)
from src.editor.operations import (
    add_text_overlay,
    apply_split,
    apply_split_keep_left,
    apply_split_keep_right,
    apply_trim,
    delete_caption,
    frame_step,
    new_project_from_clip,
    remove_text,
    set_aspect,
    set_caption_style,
    set_crop,
    set_playhead,
)
from src.editor.project_io import load_project, save_project
from src.editor.timeline_html import render_timeline_html

__all__ = [
    "AspectRatio",
    "AudioSettings",
    "CaptionCue",
    "CropSettings",
    "ExportPlan",
    "HistoryStack",
    "ProjectState",
    "TextOverlay",
    "TimelineRange",
    "add_text_overlay",
    "apply_split",
    "apply_split_keep_left",
    "apply_split_keep_right",
    "apply_trim",
    "build_export_plan",
    "delete_caption",
    "frame_step",
    "load_project",
    "new_project_from_clip",
    "remove_text",
    "render_timeline_html",
    "run_export",
    "save_project",
    "set_aspect",
    "set_caption_style",
    "set_crop",
    "set_playhead",
]
