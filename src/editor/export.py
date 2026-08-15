"""
FFmpeg export for Dona 30K / VideoClipper.
Avoids Windows hang from subtitles= + force_style; uses ASS file + ass filter.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time as time_mod
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from src.config import TEMP_DIR, get_ffmpeg_path
from src.editor.export_plan import ExportPlan
from src.editor.models import AspectRatio, ProjectState
from src.preset import CANVAS_H, CANVAS_W, DEFAULT
from src.utils import safe_filename

log = logging.getLogger("video_clipper.export")
ProgressCb = Optional[Callable[[float, str], None]]


@lru_cache(maxsize=1)
def _has_nvenc() -> bool:
    try:
        ffmpeg = get_ffmpeg_path()
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=12,
        )
        blob = (r.stdout or "") + (r.stderr or "")
        if "h264_nvenc" not in blob:
            return False
        probe = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.05",
                "-c:v", "h264_nvenc", "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=15,
        )
        return probe.returncode == 0
    except Exception as exc:
        log.warning("NVENC: %s", exc)
        return False


def _ass_time(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _write_ass(state: ProjectState, path: Path) -> Path:
    """Self-contained ASS — no force_style needed (avoids Windows hang)."""
    fs = DEFAULT.font_size
    margin_v = DEFAULT.margin_v
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {CANVAS_W}
PlayResY: {CANVAS_H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{fs},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,0,2,70,70,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    for c in state.captions:
        if c.end <= c.start or not (c.text or "").strip():
            continue
        body = (c.text or "").replace("\n", "\\N").strip()
        # escape ASS special chars
        body = body.replace("{", "\\{").replace("}", "\\}")
        events.append(
            f"Dialogue: 0,{_ass_time(c.start)},{_ass_time(c.end)},Default,,0,0,0,,{body}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return path


def _escape_drawtext(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
    )


def _filter_path(path: Path) -> str:
    """Path for ass=/subtitles= filters on Windows."""
    # forward slashes; escape drive colon for ffmpeg filter syntax
    p = str(path.resolve()).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        p = p[0] + "\\:" + p[2:]
    return p


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
    # single scale+crop chain
    vf.append(f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=fast_bilinear")
    vf.append(f"crop={w}:{h}:(iw-{w})*{cx}:(ih-{h})*{cy}")
    vf.append("setsar=1")

    # CTA / text overlays (few — Twitch)
    for t in state.texts:
        if not t.text.strip():
            continue
        txt = _escape_drawtext(t.text)
        col = t.color if not t.color.startswith("#") else "0x" + t.color[1:]
        enable = f"between(t\,{t.start:.3f}\,{t.end:.3f})"
        vf.append(
            f"drawtext=text='{txt}':fontsize={min(max(t.font_size, 18), 36)}:"
            f"fontcolor={col}:borderw=2:bordercolor=0x000000:"
            f"x=(w-text_w)*{t.x:.3f}:y=(h-text_h)*{t.y:.3f}:enable='{enable}'"
        )

    if burn_captions and state.captions:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        ass_path = TEMP_DIR / f"export_{safe_filename(state.name)}.ass"
        _write_ass(state, ass_path)
        # ass= is more reliable than subtitles=+force_style on Windows
        vf.append(f"ass='{_filter_path(ass_path)}'")

    args = [
        ffmpeg, "-hide_banner", "-y",
        "-ss", f"{start:.3f}",
        "-i", state.source_path,
        "-t", f"{dur:.3f}",
        "-vf", ",".join(vf),
    ]

    use_nvenc = (not force_cpu) and _has_nvenc()
    if use_nvenc:
        args += [
            "-c:v", "h264_nvenc",
            "-preset", "p1",  # fastest NVENC preset
            "-rc", "vbr",
            "-cq", "26",
            "-b:v", "0",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
        notes = ["nvenc-fast"]
    else:
        args += [
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "24",
            "-threads", "0",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
        notes = ["x264-ultrafast"]

    args += ["-c:a", "aac", "-b:a", "128k", "-ac", "2", str(output_path)]

    return ExportPlan(
        args=args,
        needs_reencode=True,
        output_path=str(output_path),
        width=w,
        height=h,
        notes=notes,
    )


def _run_ffmpeg(args: list[str], dur: float, progress: ProgressCb, hard_timeout: float = 180.0) -> None:
    """Run FFmpeg; fail if no progress for too long or total timeout."""
    log.info("FFmpeg: %s", " ".join(args)[:500])
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stderr is not None
    stderr_data: list[str] = []
    time_re = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")
    t0 = time_mod.time()
    last_progress_t = t0
    last_media_t = 0.0

    while True:
        if proc.poll() is not None:
            # drain rest
            rest = proc.stderr.read() or ""
            stderr_data.append(rest)
            break

        line = proc.stderr.readline()
        if line:
            stderr_data.append(line)
            m = time_re.search(line)
            if m:
                hh, mm, ss = m.groups()
                media_t = int(hh) * 3600 + int(mm) * 60 + float(ss)
                last_media_t = media_t
                last_progress_t = time_mod.time()
                if progress:
                    progress(min(0.99, media_t / max(dur, 0.01)), f"{media_t:.1f}s / {dur:.1f}s")

        now = time_mod.time()
        if now - t0 > hard_timeout:
            proc.kill()
            raise RuntimeError(
                f"Export excedeu {hard_timeout:.0f}s (travou?). "
                f"Ultimo time={last_media_t:.1f}s. Tente Rascunho ou feche outros apps."
            )
        # no time= update for 90s after start → hung filter
        if now - last_progress_t > 90 and last_media_t < 1.0:
            proc.kill()
            raise RuntimeError(
                "FFmpeg travou no inicio (filtros/legendas). "
                "Use 'Rascunho rapido' ou tente de novo — versao nova usa ASS."
            )

        if not line and proc.poll() is None:
            time_mod.sleep(0.05)

    code = proc.wait(timeout=30)
    tail = "".join(stderr_data)[-1200:]
    if code != 0:
        raise RuntimeError(tail or f"exit {code}")


def run_export(
    state: ProjectState,
    output_path: Path,
    burn_captions: bool = True,
    progress: ProgressCb = None,
) -> Path:
    if not Path(state.source_path).exists():
        raise FileNotFoundError(f"Source missing: {state.source_path}")

    state.aspect = AspectRatio.VERTICAL_9_16
    # soft timeout: ~4s encode budget per second of media, min 120 max 300
    hard = min(300.0, max(120.0, state.timeline_duration * 4.0))

    plan = build_full_export_plan(state, output_path, burn_captions=burn_captions)
    mode = "GPU" if "nvenc" in plan.notes[0] else "CPU"
    if progress:
        progress(0.02, f"Export {mode}…")

    try:
        _run_ffmpeg(plan.args, state.timeline_duration, progress, hard_timeout=hard)
    except RuntimeError as e1:
        if "nvenc" in plan.notes[0]:
            log.warning("NVENC fail/hang → CPU ultrafast")
            if progress:
                progress(0.05, "Fallback CPU…")
            plan = build_full_export_plan(
                state, output_path, burn_captions=burn_captions, force_cpu=True
            )
            _run_ffmpeg(plan.args, state.timeline_duration, progress, hard_timeout=hard)
        else:
            # last resort: no captions
            if burn_captions:
                log.warning("Export with captions failed → retry without captions")
                if progress:
                    progress(0.08, "Sem legendas (fallback)…")
                plan = build_full_export_plan(
                    state, output_path, burn_captions=False, force_cpu=True
                )
                _run_ffmpeg(plan.args, state.timeline_duration, progress, hard_timeout=hard)
            else:
                raise RuntimeError(f"FFmpeg failed: {e1}") from e1

    if not output_path.exists() or output_path.stat().st_size < 500:
        raise RuntimeError("Arquivo de export nao foi criado.")
    if progress:
        progress(1.0, "Done")
    return output_path
