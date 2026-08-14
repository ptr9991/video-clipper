"""Video metadata extraction and FFmpeg-based cutting."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config import get_ffmpeg_path, logger
from src.utils import cleanup_file, create_temp_file, validate_timestamps

log = logging.getLogger("video_clipper.video")


@dataclass
class VideoInfo:
    """Metadata about a video file."""

    path: Path
    duration: float
    width: int
    height: int
    size_bytes: int
    codec: str = ""
    fps: float = 0.0

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


def get_video_info(video_path: Path) -> VideoInfo:
    """
    Extract basic metadata using ffprobe (ships with FFmpeg).
    """
    ffmpeg = get_ffmpeg_path()
    # ffprobe is usually next to ffmpeg
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    if not Path(ffprobe).exists():
        # Fallback: try system ffprobe
        import shutil

        ffprobe = shutil.which("ffprobe") or "ffprobe"

    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as exc:
        log.error("ffprobe failed: %s", exc)
        # Minimal fallback using ffmpeg itself
        return _fallback_info(video_path)

    duration = 0.0
    width = 0
    height = 0
    codec = ""
    fps = 0.0

    if "format" in data and "duration" in data["format"]:
        duration = float(data["format"]["duration"])

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = int(stream.get("width", 0))
            height = int(stream.get("height", 0))
            codec = stream.get("codec_name", "")
            # fps can be "30/1" or "29.97"
            avg_fps = stream.get("avg_frame_rate", "0/1")
            try:
                if "/" in avg_fps:
                    num, den = avg_fps.split("/")
                    fps = float(num) / float(den) if float(den) else 0.0
                else:
                    fps = float(avg_fps)
            except (ValueError, ZeroDivisionError):
                fps = 0.0
            break

    size_bytes = video_path.stat().st_size

    return VideoInfo(
        path=video_path,
        duration=duration,
        width=width,
        height=height,
        size_bytes=size_bytes,
        codec=codec,
        fps=fps,
    )


def _fallback_info(video_path: Path) -> VideoInfo:
    """Very basic fallback when ffprobe is unavailable."""
    size_bytes = video_path.stat().st_size
    # Try to get duration via ffmpeg -i (parse stderr)
    ffmpeg = get_ffmpeg_path()
    cmd = [ffmpeg, "-i", str(video_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        # Duration is in stderr: Duration: 00:01:23.45
        import re

        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
        duration = 0.0
        if match:
            h, m, s = match.groups()
            duration = int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        duration = 0.0

    return VideoInfo(
        path=video_path,
        duration=duration,
        width=0,
        height=0,
        size_bytes=size_bytes,
    )


def extract_audio(
    video_path: Path,
    output_path: Optional[Path] = None,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """
    Extract mono 16 kHz WAV audio optimised for Whisper transcription.

    Uses subprocess with a list of arguments (no shell=True).
    """
    ffmpeg = get_ffmpeg_path()
    if output_path is None:
        output_path = create_temp_file(suffix=".wav", prefix="audio_")

    cmd = [
        ffmpeg,
        "-y",  # overwrite
        "-i",
        str(video_path),
        "-vn",  # no video
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    log.info("Extracting audio: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        log.debug("FFmpeg stdout: %s", result.stdout)
        if result.stderr:
            log.debug("FFmpeg stderr: %s", result.stderr)
    except subprocess.CalledProcessError as exc:
        log.error("Audio extraction failed: %s", exc.stderr)
        raise RuntimeError(
            f"Falha ao extrair áudio com FFmpeg: {exc.stderr[:300] if exc.stderr else str(exc)}"
        ) from exc
    except subprocess.TimeoutExpired:
        raise RuntimeError("Timeout ao extrair áudio (vídeo muito longo?).") from None

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("Arquivo de áudio gerado está vazio ou não foi criado.")

    log.info("Audio extracted: %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path


def cut_video(
    input_path: Path,
    start: float,
    end: float,
    output_path: Path,
    mode: str = "fast",
) -> Path:
    """
    Cut a segment from the video.

    mode="fast"  → stream copy (-c copy) – default, very fast, keyframe aligned
    mode="precise" → re-encode (libx264) – slower, frame-accurate

    Arguments are always passed as a list to subprocess (never shell=True).
    """
    ffmpeg = get_ffmpeg_path()
    start, end = validate_timestamps(start, end, video_duration=1e9)  # safety
    duration = end - start

    if mode == "fast":
        # Place -ss before -i for fast seek (less accurate but instant)
        # -c copy avoids re-encoding
        cmd = [
            ffmpeg,
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(input_path),
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]
    else:
        # Accurate mode: -ss after -i + re-encode
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
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
            str(output_path),
        ]

    log.info("Cutting video (%s mode): %s", mode, " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
        if result.stderr:
            log.debug("FFmpeg cut stderr: %s", result.stderr[-500:])
    except subprocess.CalledProcessError as exc:
        log.error("Cut failed: %s", exc.stderr)
        raise RuntimeError(
            f"Falha ao cortar o vídeo: {exc.stderr[:400] if exc.stderr else str(exc)}"
        ) from exc
    except subprocess.TimeoutExpired:
        raise RuntimeError("Timeout ao cortar o vídeo.") from None

    if not output_path.exists() or output_path.stat().st_size < 1000:
        raise RuntimeError("Arquivo de saída do clipe não foi criado corretamente.")

    log.info("Clip created: %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path


def build_cut_command(
    ffmpeg_path: str,
    input_path: str,
    start: float,
    duration: float,
    output_path: str,
    mode: str = "fast",
) -> list[str]:
    """
    Pure function that builds the argument list (useful for unit tests).
    """
    if mode == "fast":
        return [
            ffmpeg_path,
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            input_path,
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            output_path,
        ]
    return [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
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
        output_path,
    ]
