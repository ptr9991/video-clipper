"""
Build an FFmpeg argument list from ProjectState (no execution).
Phase 1: plan only — Phase 10 will run subprocess + progress.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.editor.models import AspectRatio, ProjectState


@dataclass
class ExportPlan:
    """Resolved FFmpeg invocation."""

    args: list[str]
    needs_reencode: bool
    output_path: str
    width: int
    height: int
    notes: list[str] = field(default_factory=list)


def _needs_reencode(state: ProjectState) -> bool:
    if state.aspect != AspectRatio.LANDSCAPE_16_9:
        return True
    if state.crop.zoom > 1.001:
        return True
    if state.crop.center_x != 0.5 or state.crop.center_y != 0.5:
        return True
    if state.audio.volume != 1.0 or state.audio.muted:
        return True
    if state.audio.fade_in > 0 or state.audio.fade_out > 0:
        return True
    if state.captions or state.texts:
        return True
    # trim alone can use -c copy
    return False


def build_export_plan(
    state: ProjectState,
    ffmpeg_path: str,
    output_path: Path | str,
) -> ExportPlan:
    state.validate()
    out = str(output_path)
    w, h = state.aspect.size
    reencode = _needs_reencode(state)
    notes: list[str] = []

    start = state.playable_range.start
    dur = state.playable_range.duration

    if not reencode:
        notes.append("Stream copy (-c copy) — trim only")
        args = [
            ffmpeg_path,
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            state.source_path,
            "-t",
            f"{dur:.3f}",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            out,
        ]
        return ExportPlan(
            args=args,
            needs_reencode=False,
            output_path=out,
            width=w,
            height=h,
            notes=notes,
        )

    notes.append("Re-encode (filters / audio / captions)")
    # Base: trim + scale/crop toward target aspect
    # Simplified crop: scale to cover then crop center with optional offset
    vf_parts: list[str] = []
    # scale to cover target
    vf_parts.append(f"scale={w}:{h}:force_original_aspect_ratio=increase")
    # crop to exact size; offset from center using center_x/y lightly
    # (full face-tracking comes in later phase)
    cx = state.crop.center_x
    cy = state.crop.center_y
    # crop=w:h:x:y — x/y as expressions
    vf_parts.append(
        f"crop={w}:{h}:(iw-{w})*{cx}:(ih-{h})*{cy}"
    )
    if state.crop.zoom > 1.001:
        # extra zoom before crop
        z = state.crop.zoom
        vf_parts.insert(0, f"scale=iw*{z}:ih*{z}")

    vf = ",".join(vf_parts)

    af_parts: list[str] = []
    if state.audio.muted:
        af_parts.append("volume=0")
    elif state.audio.volume != 1.0:
        af_parts.append(f"volume={state.audio.volume}")
    if state.audio.fade_in > 0:
        af_parts.append(f"afade=t=in:st=0:d={state.audio.fade_in:.3f}")
    if state.audio.fade_out > 0:
        st = max(0.0, dur - state.audio.fade_out)
        af_parts.append(f"afade=t=out:st={st:.3f}:d={state.audio.fade_out:.3f}")

    args = [
        ffmpeg_path,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        state.source_path,
        "-t",
        f"{dur:.3f}",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
    ]
    if af_parts:
        args += ["-af", ",".join(af_parts)]
    args += ["-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out]

    if state.captions:
        notes.append(f"{len(state.captions)} captions (burn-in in later phase)")
    if state.texts:
        notes.append(f"{len(state.texts)} text overlays (later phase)")

    return ExportPlan(
        args=args,
        needs_reencode=True,
        output_path=out,
        width=w,
        height=h,
        notes=notes,
    )
