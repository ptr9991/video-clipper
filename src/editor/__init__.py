"""Video Editor — project state + FFmpeg export."""

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
    apply_split,
    apply_trim,
    frame_step,
    new_project_from_clip,
    set_aspect,
    set_crop,
)
from src.editor.project_io import load_project, save_project

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
    "frame_step",
    "load_project",
    "new_project_from_clip",
    "run_export",
    "save_project",
    "set_aspect",
    "set_crop",
]
