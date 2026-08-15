"""Lightweight single-frame thumbnails via FFmpeg."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from src.cache import thumb_path
from src.config import get_ffmpeg_path

log = logging.getLogger("video_clipper.thumbs")


def extract_thumbnail(
    video_path: Path,
    at_sec: float,
    video_hash: str,
    width: int = 320,
) -> Path:
    out = thumb_path(video_hash, at_sec)
    if out.exists() and out.stat().st_size > 100:
        return out

    ffmpeg = get_ffmpeg_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{max(0.0, at_sec):.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", f"scale={width}:-1",
        "-q:v", "5",
        str(out),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
    except Exception as exc:
        log.warning("thumb failed at %.1fs: %s", at_sec, exc)
        # empty placeholder path still returned; UI can skip
    return out
