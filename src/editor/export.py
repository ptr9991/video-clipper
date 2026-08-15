"""Execute ExportPlan with FFmpeg (subprocess, no shell)."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

from src.editor.export_plan import ExportPlan, build_export_plan
from src.editor.models import ProjectState
from src.config import get_ffmpeg_path, TEMP_DIR
from src.utils import safe_filename

log = logging.getLogger("video_clipper.export")

ProgressCb = Optional[Callable[[float, str], None]]


def _write_srt(state: ProjectState, path: Path) -> Path:
    lines: list[str] = []
    for i, c in enumerate(state.captions, 1):
        if c.end <= c.start or not c.text.strip():
            continue

        def ts(sec: float) -> str:
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = int(sec % 60)
            ms = int(round((sec - int(sec)) * 1000))
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        lines.append(str(i))
        lines.append(f"{ts(c.start)} --> {ts(c.end)}")
        lines.append(c.text.strip())
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) if lines else "1\n00:00:00,000 --> 00:00:01,000\n\n", encoding="utf-8")
    return path


def _escape_sub(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace(":", "\\:")


def build_full_export_plan(
    state: ProjectState,
    output_path: Path,
    burn_captions: bool = True,
) -> ExportPlan:
    """Extend basic plan with optional subtitle burn-in."""
    ffmpeg = get_ffmpeg_path()
    plan = build_export_plan(state, ffmpeg, output_path)

    if not burn_captions or not state.captions:
        return plan

    # Rebuild with subtitles filter (forces reencode)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    srt = TEMP_DIR / f"export_{safe_filename(state.name)}.srt"
    _write_srt(state, srt)
    sub = _escape_sub(srt)
    style = (
        f"FontName=Arial,FontSize={state.caption_style.font_size},"
        f"PrimaryColour=&H00FFFFFF,Outline={state.caption_style.outline},"
        f"Alignment=2,MarginV={state.caption_style.margin_v}"
    )

    w, h = state.aspect.size
    start = state.playable_range.start
    dur = state.playable_range.duration
    z = state.crop.zoom
    cx, cy = state.crop.center_x, state.crop.center_y

    vf = []
    if z > 1.001:
        vf.append(f"scale=iw*{z}:ih*{z}")
    vf.append(f"scale={w}:{h}:force_original_aspect_ratio=increase")
    vf.append(f"crop={w}:{h}:(iw-{w})*{cx}:(ih-{h})*{cy}")
    vf.append(f"subtitles='{sub}':force_style='{style}'")

    af = []
    if state.audio.muted:
        af.append("volume=0")
    elif state.audio.volume != 1.0:
        af.append(f"volume={state.audio.volume}")
    if state.audio.fade_in > 0:
        af.append(f"afade=t=in:st=0:d={state.audio.fade_in:.3f}")
    if state.audio.fade_out > 0:
        st = max(0.0, dur - state.audio.fade_out)
        af.append(f"afade=t=out:st={st:.3f}:d={state.audio.fade_out:.3f}")

    args = [
        ffmpeg, "-y",
        "-ss", f"{start:.3f}",
        "-i", state.source_path,
        "-t", f"{dur:.3f}",
        "-vf", ",".join(vf),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
    ]
    if af:
        args += ["-af", ",".join(af)]
    args += ["-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(output_path)]

    return ExportPlan(
        args=args,
        needs_reencode=True,
        output_path=str(output_path),
        width=w,
        height=h,
        notes=plan.notes + ["captions burned in"],
    )


def run_export(
    state: ProjectState,
    output_path: Path,
    burn_captions: bool = True,
    progress: ProgressCb = None,
) -> Path:
    if not Path(state.source_path).exists():
        raise FileNotFoundError(f"Source missing: {state.source_path}")

    plan = build_full_export_plan(state, output_path, burn_captions=burn_captions)
    log.info("Export: %s", " ".join(plan.args)[:400])

    if progress:
        progress(0.05, "Starting FFmpeg…")

    proc = subprocess.Popen(
        plan.args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stderr_data = []
    assert proc.stderr is not None
    time_re = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")
    dur = max(0.01, state.timeline_duration)

    for line in proc.stderr:
        stderr_data.append(line)
        m = time_re.search(line)
        if m and progress:
            h, mi, s = m.groups()
            t = int(h) * 3600 + int(mi) * 60 + float(s)
            progress(min(0.99, t / dur), f"Rendering {t:.1f}s / {dur:.1f}s")

    code = proc.wait(timeout=600)
    if code != 0:
        err = "".join(stderr_data)[-800:]
        raise RuntimeError(f"FFmpeg export failed: {err}")

    if not output_path.exists() or output_path.stat().st_size < 500:
        raise RuntimeError("Export file was not created.")

    if progress:
        progress(1.0, "Done")
    return output_path
