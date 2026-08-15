"""
Local clip editor (FFmpeg only).

- Vertical 9:16 (Shorts / Reels / TikTok)
- Burn-in subtitles from transcription
- Optional second video (webcam) as PiP in a chosen corner
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
    "centro inferior": "(W-w)/2:H-h-40",
}


@dataclass
class EditOptions:
    vertical_9x16: bool = True
    add_subtitles: bool = True
    subtitle_font_size: int = 20
    webcam_path: Optional[Path] = None
    webcam_position: str = "canto superior direito"
    webcam_scale: float = 0.30


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
    content = "\n".join(lines) if lines else "1\n00:00:00,000 --> 00:00:02,000\n\n"
    out_path.write_text(content, encoding="utf-8")
    return out_path


def _escape_sub_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace(":", "\\:")


def render_edited_clip(
    clip_path: Path,
    options: EditOptions,
    segments: Optional[list[Segment]] = None,
    clip_start_abs: float = 0.0,
    clip_end_abs: float = 0.0,
    output_path: Optional[Path] = None,
) -> Path:
    ffmpeg = get_ffmpeg_path()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = OUTPUT_DIR / generate_output_filename(prefix="edit")

    has_cam = bool(options.webcam_path and options.webcam_path.exists())
    parts: list[str] = []

    # Main video → optional 9:16
    if options.vertical_9x16:
        parts.append(
            "[0:v]crop='min(iw\,ih*9/16)':'min(ih\,iw*16/9)':'(iw-ow)/2':'(ih-oh)/2',"
            "scale=1080:1920:flags=lanczos,setsar=1[main]"
        )
    else:
        parts.append("[0:v]scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1[main]")

    label = "main"

    if has_cam:
        scale = max(0.15, min(0.45, options.webcam_scale))
        pos = PIP_POSITIONS.get(
            options.webcam_position, PIP_POSITIONS["canto superior direito"]
        )
        # scale cam to fraction of 1080 if vertical else ~30% of source
        if options.vertical_9x16:
            cam_w = int(1080 * scale)
            parts.append(f"[1:v]scale={cam_w}:-1,setsar=1[cam]")
        else:
            parts.append(f"[1:v]scale=iw*{scale}:-1,setsar=1[cam]")
        parts.append(f"[{label}][cam]overlay={pos}[ov]")
        label = "ov"

    if options.add_subtitles and segments:
        end = clip_end_abs if clip_end_abs > clip_start_abs else clip_start_abs + 90
        srt_path = TEMP_DIR / f"subs_{safe_filename(clip_path.stem)}.srt"
        write_srt_for_clip(segments, clip_start_abs, end, srt_path)
        if srt_path.stat().st_size > 30:
            sub = _escape_sub_path(srt_path)
            style = (
                f"FontName=Arial,FontSize={options.subtitle_font_size},"
                "PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,"
                "BorderStyle=3,Outline=1,Shadow=0,Alignment=2,MarginV=60"
            )
            parts.append(
                f"[{label}]subtitles='{sub}':force_style='{style}'[outv]"
            )
            label = "outv"

    if label != "outv":
        parts.append(f"[{label}]format=yuv420p[outv]")

    filter_complex = ";".join(parts)

    cmd = [ffmpeg, "-y", "-i", str(clip_path)]
    if has_cam:
        cmd += ["-i", str(options.webcam_path)]

    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
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

    log.info("Edit cmd filter: %s", filter_complex[:300])
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=600
        )
        if result.stderr:
            log.debug(result.stderr[-600:])
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or str(exc))[:600]
        raise RuntimeError(f"Falha ao editar o clipe: {err}") from exc

    if not output_path.exists() or output_path.stat().st_size < 1000:
        raise RuntimeError("Arquivo editado não foi gerado.")

    return output_path
