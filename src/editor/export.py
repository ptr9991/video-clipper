"""FFmpeg export — 9:16 + captions + CTA. NVENC on RTX when available."""

from __future__ import annotations

import logging
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from src.config import TEMP_DIR, get_ffmpeg_path
from src.editor.export_plan import ExportPlan
from src.editor.models import ProjectState, AspectRatio
from src.preset import CANVAS_H, CANVAS_W, DEFAULT, ass_force_style
from src.utils import safe_filename

log = logging.getLogger("video_clipper.export")
ProgressCb = Optional[Callable[[float, str], None]]


@lru_cache(maxsize=1)
def _has_nvenc() -> bool:
    try:
        ffmpeg = get_ffmpeg_path()
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=15,
        )
        if "h264_nvenc" not in ((r.stdout or "") + (r.stderr or "")):
            return False
        probe = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
                "-c:v", "h264_nvenc", "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=20,
        )
        return probe.returncode == 0
    except Exception as exc:
        log.warning("NVENC check: %s", exc)
        return False


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

        body = c.text.replace("\\N", "\n").strip()
        lines.extend([str(i), f"{ts(c.start)} --> {ts(c.end)}", body, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) if lines else "1\n00:00:00,000 --> 00:00:01,000\n\n", encoding="utf-8")
    return path


def _escape_sub(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace(":", "\\:")


def _escape_drawtext(text: str) -> str:
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace("%", "\\%")


def build_full_export_plan(
    state: ProjectState,
    output_path: Path,
    burn_captions: bool = True,
    force_cpu: bool = False,
) -> ExportPlan:
    ffmpeg = get_ffmpeg_path()
    w, h = CANVAS_W, CANVAS_H
    start = state.playable_range.start
    dur = state.playable_range.duration
    z = max(1.0, state.crop.zoom)
    cx, cy = state.crop.center_x, state.crop.center_y

    vf: list[str] = []
    if z > 1.001:
        vf.append(f"scale=iw*{z}:ih*{z}")
    vf.append(f"scale={w}:{h}:force_original_aspect_ratio=increase")
    vf.append(f"crop={w}:{h}:(iw-{w})*{cx}:(ih-{h})*{cy}")
    vf.append("setsar=1")

    for t in state.texts:
        if not t.text.strip():
            continue
        txt = _escape_drawtext(t.text)
        col = t.color if not t.color.startswith("#") else "0x" + t.color[1:]
        enable = f"between(t\,{t.start:.3f}\,{t.end:.3f})"
        vf.append(
            f"drawtext=text='{txt}':fontsize={min(t.font_size, 56)}:fontcolor={col}:"
            f"x=(w-text_w)*{t.x:.3f}:y=(h-text_h)*{t.y:.3f}:enable='{enable}'"
        )

    if burn_captions and state.captions:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        srt = TEMP_DIR / f"export_{safe_filename(state.name)}.srt"
        _write_srt(state, srt)
        vf.append(f"subtitles='{_escape_sub(srt)}':force_style='{ass_force_style(DEFAULT)}'")

    af: list[str] = []
    if state.audio.muted:
        af.append("volume=0")
    elif abs(state.audio.volume - 1.0) > 0.01:
        af.append(f"volume={state.audio.volume}")

    args = [
        ffmpeg, "-y",
        "-ss", f"{start:.3f}",
        "-i", state.source_path,
        "-t", f"{dur:.3f}",
        "-vf", ",".join(vf),
    ]

    use_nvenc = (not force_cpu) and _has_nvenc()
    if use_nvenc:
        args += [
            "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        ]
        notes = ["nvenc"]
    else:
        args += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-threads", "0",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        ]
        notes = ["x264-veryfast"]

    if af:
        args += ["-af", ",".join(af)]
    args += ["-c:a", "aac", "-b:a", "128k", str(output_path)]

    return ExportPlan(
        args=args, needs_reencode=True, output_path=str(output_path),
        width=w, height=h, notes=notes,
    )


def _run_ffmpeg(args: list[str], dur: float, progress: ProgressCb) -> str:
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr_data: list[str] = []
    assert proc.stderr is not None
    time_re = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")
    for line in proc.stderr:
        stderr_data.append(line)
        m = time_re.search(line)
        if m and progress:
            hh, mm, ss = m.groups()
            t = int(hh) * 3600 + int(mm) * 60 + float(ss)
            progress(min(0.99, t / max(dur, 0.01)), f"{t:.1f}s / {dur:.1f}s")
    code = proc.wait(timeout=600)
    tail = "".join(stderr_data)[-900:]
    if code != 0:
        raise RuntimeError(tail)
    return tail


def run_export(
    state: ProjectState,
    output_path: Path,
    burn_captions: bool = True,
    progress: ProgressCb = None,
) -> Path:
    if not Path(state.source_path).exists():
        raise FileNotFoundError(f"Source missing: {state.source_path}")

    state.aspect = AspectRatio.VERTICAL_9_16
    plan = build_full_export_plan(state, output_path, burn_captions=burn_captions)
    mode = "GPU NVENC" if "nvenc" in plan.notes else "CPU veryfast"
    log.info("Export %s", mode)
    if progress:
        progress(0.02, f"Export {mode}…")

    try:
        _run_ffmpeg(plan.args, state.timeline_duration, progress)
    except RuntimeError as first_err:
        if "nvenc" in plan.notes:
            log.warning("NVENC failed, CPU fallback")
            if progress:
                progress(0.05, "Fallback CPU…")
            plan = build_full_export_plan(
                state, output_path, burn_captions=burn_captions, force_cpu=True
            )
            try:
                _run_ffmpeg(plan.args, state.timeline_duration, progress)
            except RuntimeError as e2:
                raise RuntimeError(f"FFmpeg failed: {e2}") from e2
        else:
            raise RuntimeError(f"FFmpeg failed: {first_err}") from first_err

    if not output_path.exists() or output_path.stat().st_size < 500:
        raise RuntimeError("Export file missing.")
    if progress:
        progress(1.0, "Done")
    return output_path
