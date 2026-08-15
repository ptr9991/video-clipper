"""
Video Editor core (Phase 1).

Edit decisions are data — never re-render video on every drag.
FFmpeg export builds a command from ProjectState.
"""

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
    apply_split,
    apply_trim,
    new_project_from_clip,
    set_aspect,
    set_crop,
)
from src.editor.project_io import load_project, save_project
from src.editor.export_plan import ExportPlan, build_export_plan

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
    "apply_split",
    "apply_trim",
    "build_export_plan",
    "load_project",
    "new_project_from_clip",
    "save_project",
    "set_aspect",
    "set_crop",
]
