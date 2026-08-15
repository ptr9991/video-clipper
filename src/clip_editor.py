"""
Local clip editor (FFmpeg only — no heavy AI).

Features:
- Vertical 9:16 crop (Shorts / Reels / TikTok)
- Burn-in subtitles from transcription (SRT)
- Optional webcam / second video as PiP in a chosen corner
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config import OUTPUT_DIR, TEMP_DIR, get_ffmpeg_path
from src.transcription import Segment
from src.utils import generate_output_filename, safe_filename

log = logging.getLogger("video_clipper.editor")

PIP_POSITIONS = {
    "canto superior direito": "W-w-20:20",
    "canto superior esquerdo": "20:20",
    "canto inferior direito": "W-w-20:H-h-20",
    "canto inferior esquerdo": "20:H-h-20",
    "centro inferior": "(W-w)/2:H-h-20",
}


@dataclass
class EditOptions:
    vertical_9x16: bool = True
    add_subtitles: bool = True
    subtitle_font_size: int = 18
    webcam_path: Optional[Path] = None
    webcam_position: str = "canto superior direito"
    webcam_scale: float = 0.28  # fraction of main width


def _srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        ms = 0
        s += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt_for_clip(
    segments: list[Segment],
    clip_start: float,
    clip_end: float,
    out_path: Path,
) -> Path:
    """Build an SRT file with timestamps relative to the clip (start at 0)."""
    lines: list[str] = []
    idx = 1
    for seg in segments:
        if seg.end < clip_start or seg.start > clip_end:
            continue
        rel_s = max(0.0, seg.start - clip_start)
        rel_e = min(clip_end - clip_start, max(0.05, seg.end - clip_start))
        if rel_e <= rel_s:
            continue
        text = re.sub(r"\s+", " ", (seg.text or "").strip())
        if not text:
            continue
        lines.append(str(idx))
        lines.append(f"{_srt_timestamp(rel_s)} --> {_srt_timestamp(rel_e)}")
        lines.append(text)
        lines.append("")
        idx += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) if lines else "1\n00:00:00,000 --> 00:00:01,000\n\n", encoding="utf-8")
    return out_path


def _escape_sub_path(path: Path) -> str:
    # FFmpeg subtitles filter on Windows needs escaped path
    p = str(path).replace("\\", "/").replace(":", "\\:")
    return p


def render_edited_clip(
    clip_path: Path,
    options: EditOptions,
    segments: Optional[list[Segment]] = None,
    clip_start_abs: float = 0.0,
    clip_end_abs: float = 0.0,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Apply layout + optional subtitles + optional webcam PiP.
    Always re-encodes (needed for filters).
    """
    ffmpeg = get_ffmpeg_path()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = OUTPUT_DIR / generate_output_filename(prefix="edit")

    filters: list[str] = []
    inputs = ["-i", str(clip_path)]
    input_idx_webcam = None

    # Base video label
    # 1) optional vertical crop 9:16 centered
    if options.vertical_9x16:
        # crop to 9:16 then scale to 1080x1920
        filters.append(
            "[0:v]crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920:flags=lanczos[base]"
        )
    else:
        filters.append("[0:v]scale=trunc(iw/2)*2:trunc(ih/2)*2[base]")

    current = "base"

    # 2) webcam PiP
    if options.webcam_path and options.webcam_path.exists():
        inputs += ["-i", str(options.webcam_path)]
        input_idx_webcam = 1
        pos = PIP_POSITIONS.get(options.webcam_position, PIP_POSITIONS["canto superior direito"])
        scale = max(0.15, min(0.45, options.webcam_scale))
        # scale webcam relative to main width 1080 if vertical else keep relative
        filters.append(
            f"[{input_idx_webcam}:v]scale=iw*{scale}:-1[cam]"
        )
        filters.append(f"[{current}][cam]overlay={pos}[withcam]")
        current = "withcam"

    # 3) subtitles
    srt_path = None
    if options.add_subtitles and segments:
        srt_path = TEMP_DIR / f"subs_{safe_filename(clip_path.stem)}.srt"
        end = clip_end_abs if clip_end_abs > clip_start_abs else clip_start_abs + 60
        write_srt_for_clip(segments, clip_start_abs, end, srt_path)
        if srt_path.stat().st_size > 20:
            sub = _escape_sub_path(srt_path)
            style = (
                f"FontName=Arial,FontSize={options.subtitle_font_size},"
                "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                "BorderStyle=3,Outline=2,Shadow=0,Alignment=2,MarginV=40"
            )
            filters.append(f"[{current}]subtitles='{sub}':force_style='{style}'[final]")
            current = "final"

    if current != "final":
        filters.append(f"[{current}]copy[final]")  # may fail — use null
        # safer: just map current as final via rename
        filters[-1] = f"[{current}]null[final]" if current != "final" else filters[-1]

    # Rebuild filter graph more carefully without invalid copy
    filters = [f for f in filters if "copy[final]" not in f and "null[final]" not in f]
    if not any("[final]" in f for f in filters):
        # last label becomes final
        last = filters[-1] if filters else "[0:v]null[final]"
        if "[" in last and "]" in last:
            # replace output pad name
            filters[-1] = re.sub(r"\[[a-zA-Z0-9]+\]$", "[final]", last)
        else:
            filters.append(f"[{current}]format=yuv420p[final]")

    filter_complex = ";".join(filters)

    cmd = [
        ffmpeg,
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[final]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    log.info("Editing clip: %s", " ".join(cmd)[:500])
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
        if result.stderr:
            log.debug(result.stderr[-800:])
    except subprocess.CalledProcessError as exc:
        err = exc.stderr[:500] if exc.stderr else str(exc)
        log.error("Edit failed: %s", err)
        raise RuntimeError(f"Falha ao editar o clipe: {err}") from exc

    if not output_path.exists() or output_path.stat().st_size < 1000:
        raise RuntimeError("Arquivo editado não foi gerado corretamente.")

    return output_path
