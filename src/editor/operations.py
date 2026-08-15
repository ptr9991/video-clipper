"""Pure ProjectState transforms."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from src.editor.caption_styles import style_by_name
from src.editor.models import (
    AspectRatio,
    CaptionCue,
    CropSettings,
    ProjectState,
    TextOverlay,
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
    s = state.clone()
    s.playable_range = TimelineRange(start=start, end=end).clamp(s.source_duration)
    s.playhead = min(s.playhead, s.timeline_duration)
    old = state.playable_range
    new_start = s.playable_range.start
    new_end = s.playable_range.end
    new_caps: list[CaptionCue] = []
    for c in state.captions:
        abs_s = old.start + c.start
        abs_e = old.start + c.end
        if abs_e < new_start or abs_s > new_end:
            continue
        rel_s = max(0.0, abs_s - new_start)
        rel_e = min(s.timeline_duration, abs_e - new_start)
        if rel_e > rel_s:
            new_caps.append(
                CaptionCue(
                    id=c.id, start=rel_s, end=rel_e, text=c.text,
                    highlight_words=list(c.highlight_words),
                )
            )
    s.captions = new_caps
    # texts
    new_texts: list[TextOverlay] = []
    for t in state.texts:
        abs_s = old.start + t.start
        abs_e = old.start + t.end
        if abs_e < new_start or abs_s > new_end:
            continue
        rel_s = max(0.0, abs_s - new_start)
        rel_e = min(s.timeline_duration, abs_e - new_start)
        if rel_e > rel_s:
            nt = TextOverlay(
                id=t.id, text=t.text, start=rel_s, end=rel_e,
                x=t.x, y=t.y, font_size=t.font_size, color=t.color,
            )
            new_texts.append(nt)
    s.texts = new_texts
    s.validate()
    return s


def apply_split_keep_left(state: ProjectState, at: float) -> ProjectState:
    """Keep left part of timeline at playhead `at`."""
    at = max(0.05, min(at, state.timeline_duration - 0.05))
    abs_at = state.playable_range.start + at
    return apply_trim(state, state.playable_range.start, abs_at)


def apply_split_keep_right(state: ProjectState, at: float) -> ProjectState:
    at = max(0.05, min(at, state.timeline_duration - 0.05))
    abs_at = state.playable_range.start + at
    return apply_trim(state, abs_at, state.playable_range.end)


def apply_split(state: ProjectState, at: float) -> tuple[ProjectState, ProjectState]:
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


def set_caption_style(state: ProjectState, style_name: str) -> ProjectState:
    s = state.clone()
    s.caption_style = style_by_name(style_name)
    return s


def delete_caption(state: ProjectState, caption_id: str) -> ProjectState:
    s = state.clone()
    s.captions = [c for c in s.captions if c.id != caption_id]
    return s


def add_text_overlay(
    state: ProjectState,
    text: str,
    start: float = 0.0,
    end: Optional[float] = None,
    x: float = 0.5,
    y: float = 0.15,
    font_size: int = 48,
    color: str = "#FFFFFF",
) -> ProjectState:
    s = state.clone()
    if end is None:
        end = s.timeline_duration
    s.texts.append(
        TextOverlay(
            id=f"txt_{uuid.uuid4().hex[:8]}",
            text=text.strip() or "TEXT",
            start=max(0.0, start),
            end=min(s.timeline_duration, max(start + 0.1, end)),
            x=max(0.0, min(1.0, x)),
            y=max(0.0, min(1.0, y)),
            font_size=max(12, min(120, font_size)),
            color=color or "#FFFFFF",
        )
    )
    return s


def remove_text(state: ProjectState, text_id: str) -> ProjectState:
    s = state.clone()
    s.texts = [t for t in s.texts if t.id != text_id]
    return s


def set_playhead(state: ProjectState, t: float) -> ProjectState:
    s = state.clone()
    s.playhead = max(0.0, min(s.timeline_duration, t))
    return s


def frame_step(state: ProjectState, direction: int) -> ProjectState:
    s = state.clone()
    step = 1.0 / max(s.fps, 1.0)
    s.playhead = max(0.0, min(s.timeline_duration, s.playhead + direction * step))
    return s
