"""
Pure functions that transform ProjectState.
No disk / FFmpeg side effects — UI + export consume the new state.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from src.editor.models import (
    AspectRatio,
    CaptionCue,
    CropSettings,
    ProjectState,
    TimelineRange,
)
from src.transcription import Segment


def new_project_from_clip(
    source_path: Path | str,
    duration: float,
    fps: float = 30.0,
    name: str = "Clip",
    segments: Optional[list[Segment]] = None,
    clip_start_abs: float = 0.0,
) -> ProjectState:
    path = str(source_path)
    dur = max(0.0, float(duration))
    project = ProjectState(
        name=name,
        source_path=path,
        source_duration=dur,
        fps=fps if fps > 0 else 30.0,
        playable_range=TimelineRange(start=0.0, end=dur),
        aspect=AspectRatio.VERTICAL_9_16,
    )
    if segments:
        project.captions = captions_from_segments(segments, clip_start_abs, dur)
    project.validate()
    return project


def captions_from_segments(
    segments: list[Segment],
    clip_start_abs: float,
    clip_duration: float,
) -> list[CaptionCue]:
    cues: list[CaptionCue] = []
    for i, seg in enumerate(segments):
        if seg.end < clip_start_abs or seg.start > clip_start_abs + clip_duration:
            continue
        rel_s = max(0.0, seg.start - clip_start_abs)
        rel_e = min(clip_duration, max(0.05, seg.end - clip_start_abs))
        if rel_e <= rel_s:
            continue
        text = (seg.text or "").strip()
        if not text:
            continue
        cues.append(
            CaptionCue(
                id=f"cap_{i}_{uuid.uuid4().hex[:6]}",
                start=rel_s,
                end=rel_e,
                text=text,
            )
        )
    return cues


def apply_trim(state: ProjectState, start: float, end: float) -> ProjectState:
    """Set playable range within source."""
    s = state.clone()
    s.playable_range = TimelineRange(start=start, end=end).clamp(s.source_duration)
    s.playhead = min(s.playhead, s.timeline_duration)
    # Shift captions into new window relative to old range
    old = state.playable_range
    offset = s.playable_range.start - old.start
    # Captions are stored relative to playable timeline origin (0 = range.start)
    # When trimming, drop cues outside new window and shift remaining.
    new_caps: list[CaptionCue] = []
    new_start = s.playable_range.start
    new_end = s.playable_range.end
    for c in state.captions:
        # caption times are relative to previous playable start
        abs_s = old.start + c.start
        abs_e = old.start + c.end
        if abs_e < new_start or abs_s > new_end:
            continue
        rel_s = max(0.0, abs_s - new_start)
        rel_e = min(s.timeline_duration, abs_e - new_start)
        if rel_e > rel_s:
            new_caps.append(
                CaptionCue(
                    id=c.id,
                    start=rel_s,
                    end=rel_e,
                    text=c.text,
                    highlight_words=list(c.highlight_words),
                )
            )
    s.captions = new_caps
    s.validate()
    return s


def apply_split(state: ProjectState, at: float) -> tuple[ProjectState, ProjectState]:
    """
    Split playable range at `at` (seconds on timeline 0..duration).
    Returns (left, right) as two independent projects sharing the same source.
    """
    at = max(0.0, min(at, state.timeline_duration))
    abs_at = state.playable_range.start + at
    left = apply_trim(state, state.playable_range.start, abs_at)
    right = apply_trim(state, abs_at, state.playable_range.end)
    left.name = f"{state.name}_A"
    right.name = f"{state.name}_B"
    return left, right


def set_aspect(state: ProjectState, aspect: AspectRatio) -> ProjectState:
    s = state.clone()
    s.aspect = aspect
    return s


def set_crop(state: ProjectState, crop: CropSettings) -> ProjectState:
    s = state.clone()
    s.crop = crop
    s.validate()
    return s


def frame_step(state: ProjectState, direction: int) -> ProjectState:
    """Move playhead by one frame. direction: +1 or -1."""
    s = state.clone()
    step = 1.0 / max(s.fps, 1.0)
    s.playhead = max(0.0, min(s.timeline_duration, s.playhead + direction * step))
    return s
