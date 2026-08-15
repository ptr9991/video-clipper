"""Adaptive frame sampling from short clips for visual analysis."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

from src.config import TEMP_DIR, get_ffmpeg_path
from src.utils import cleanup_file

log = logging.getLogger("video_clipper.frames")


def adaptive_timestamps(duration: float, max_frames: int = 12) -> list[float]:
    """
    Build a denser sample at the start (hook) and sparser later.
    Always includes 0 and near-end.
    """
    if duration <= 0:
        return [0.0]
    if duration <= 5:
        step = max(duration / max(max_frames - 1, 1), 0.5)
        return [round(min(i * step, duration - 0.05), 2) for i in range(max_frames) if i * step < duration]

    # Hook-heavy: first 5s denser
    times: list[float] = []
    # first 3 seconds every ~1s
    t = 0.0
    while t < min(5.0, duration) and len(times) < max_frames // 2:
        times.append(round(t, 2))
        t += 1.0
    # rest evenly
    remaining_slots = max(max_frames - len(times), 1)
    start = times[-1] + 1.0 if times else 0.0
    if start < duration:
        step = (duration - start) / remaining_slots
        for i in range(remaining_slots):
            ts = start + i * step
            if ts >= duration - 0.05:
                break
            times.append(round(ts, 2))
    # ensure last frame near end
    end_t = round(max(duration - 0.15, 0.0), 2)
    if not times or times[-1] < end_t - 0.3:
        times.append(end_t)
    # unique sorted
    return sorted(set(times))[:max_frames]


def extract_frames(
    video_path: Path,
    duration: float,
    max_frames: int = 12,
    width: int = 512,
    out_dir: Optional[Path] = None,
) -> list[Path]:
    """
    Extract JPEG frames at adaptive timestamps using FFmpeg.
    Caller should delete frames after analysis.
    """
    ffmpeg = get_ffmpeg_path()
    out_dir = out_dir or (TEMP_DIR / "frames")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous frames in dir
    for old in out_dir.glob("frame_*.jpg"):
        cleanup_file(old)

    timestamps = adaptive_timestamps(duration, max_frames=max_frames)
    paths: list[Path] = []

    for i, ts in enumerate(timestamps):
        out = out_dir / f"frame_{i:03d}_{ts:.2f}s.jpg"
        cmd = [
            ffmpeg,
            "-y",
            "-ss",
            f"{ts:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:-2",
            "-q:v",
            "5",
            str(out),
        ]
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            if out.exists() and out.stat().st_size > 0:
                paths.append(out)
        except Exception as exc:
            log.warning("Frame at %.2fs failed: %s", ts, exc)

    log.info("Extracted %d frames from %s", len(paths), video_path.name)
    return paths


def cleanup_frames(paths: list[Path]) -> None:
    for p in paths:
        cleanup_file(p)
