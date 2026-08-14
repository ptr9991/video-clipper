"""Video metadata extraction and FFmpeg-based cutting."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config import get_ffmpeg_path, logger
from src.utils import create_temp_file, validate_timestamps

log = logging.getLogger("video_clipper.video")

# Groq Whisper practical upload limit (free tier can be tighter than 100 MB)
GROQ_SAFE_UPLOAD_MB = 15.0


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
    """Extract basic metadata using ffprobe (ships with FFmpeg)."""
    ffmpeg = get_ffmpeg_path()
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    if not Path(ffprobe).exists():
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
    ffmpeg = get_ffmpeg_path()
    cmd = [ffmpeg, "-i", str(video_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
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
    bitrate: str = "32k",
) -> Path:
    """
    Extract mono compressed MP3 audio optimised for Whisper.

    32 kbps mono ≈ 0.24 MB/min → 1 hour ≈ 14 MB (under Groq limits).
    Falls back to AAC if libmp3lame is unavailable.
    """
    ffmpeg = get_ffmpeg_path()
    if output_path is None:
        output_path = create_temp_file(suffix=".mp3", prefix="audio_")

    # Prefer MP3; fall back to AAC (.m4a) if lame is missing
    attempts = [
        (
            output_path if output_path.suffix.lower() == ".mp3" else output_path.with_suffix(".mp3"),
            ["-c:a", "libmp3lame", "-b:a", bitrate],
        ),
        (
            output_path.with_suffix(".m4a"),
            ["-c:a", "aac", "-b:a", bitrate],
        ),
    ]

    last_error = ""
    for out, codec_args in attempts:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            *codec_args,
            str(out),
        ]
        log.info("Extracting audio: %s", " ".join(cmd))
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300,
            )
            if out.exists() and out.stat().st_size > 0:
                size_mb = out.stat().st_size / (1024 * 1024)
                log.info("Audio extracted: %s (%.2f MB)", out, size_mb)
                return out
        except subprocess.CalledProcessError as exc:
            last_error = exc.stderr[:400] if exc.stderr else str(exc)
            log.warning("Audio extract attempt failed: %s", last_error)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Timeout ao extrair áudio (vídeo muito longo?).") from None

    raise RuntimeError(
        f"Falha ao extrair áudio com FFmpeg: {last_error or 'codec de áudio indisponível'}"
    )


def split_audio_chunks(
    audio_path: Path,
    chunk_duration_sec: float = 300.0,
) -> list[Path]:
    """
    Split a long audio file into fixed-duration chunks for API upload.
    Returns list of chunk file paths (caller must clean them up).
    """
    ffmpeg = get_ffmpeg_path()
    out_dir = audio_path.parent
    pattern = str(out_dir / f"{audio_path.stem}_chunk_%03d{audio_path.suffix}")

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_time",
        str(int(chunk_duration_sec)),
        "-c",
        "copy",
        "-reset_timestamps",
        "1",
        pattern,
    ]

    log.info("Splitting audio into ~%.0fs chunks", chunk_duration_sec)
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Falha ao dividir áudio: {exc.stderr[:300] if exc.stderr else str(exc)}"
        ) from exc

    chunks = sorted(out_dir.glob(f"{audio_path.stem}_chunk_*{audio_path.suffix}"))
    if not chunks:
        raise RuntimeError("Nenhum chunk de áudio foi gerado.")
    log.info("Created %d audio chunks", len(chunks))
    return chunks


def cut_video(
    input_path: Path,
    start: float,
    end: float,
    output_path: Path,
    mode: str = "fast",
) -> Path:
    """
    Cut a segment from the video.

    mode="fast"  → stream copy (-c copy)
    mode="precise" → re-encode (libx264)
    """
    ffmpeg = get_ffmpeg_path()
    start, end = validate_timestamps(start, end, video_duration=1e9)
    duration = end - start

    if mode == "fast":
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
    """Pure function that builds the argument list (useful for unit tests)."""
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
