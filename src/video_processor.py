"""Video metadata extraction and FFmpeg-based cutting."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config import get_ffmpeg_path
from src.utils import create_temp_file, validate_timestamps

log = logging.getLogger("video_clipper.video")

GROQ_SAFE_UPLOAD_MB = 15.0


@dataclass
class VideoInfo:
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


def _resolve_ffprobe(ffmpeg: str) -> str:
    candidate = ffmpeg.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
    if Path(candidate).exists():
        return candidate
    found = shutil.which("ffprobe")
    return found or candidate


def get_video_info(video_path: Path) -> VideoInfo:
    """Extract metadata with ffprobe. Never crashes on None stdout."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Arquivo de video nao encontrado: {video_path}")
    size_bytes = video_path.stat().st_size
    if size_bytes < 1000:
        raise RuntimeError(
            f"Arquivo invalido ou download incompleto ({size_bytes} bytes). "
            "Tente baixar de novo ou use a aba Arquivo."
        )

    ffmpeg = get_ffmpeg_path()
    ffprobe = _resolve_ffprobe(ffmpeg)

    cmd = [
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        stdout = result.stdout if result.stdout is not None else ""
        stderr = result.stderr if result.stderr is not None else ""

        if result.returncode != 0 or not stdout.strip():
            log.warning(
                "ffprobe failed code=%s stderr=%s",
                result.returncode,
                stderr[:300],
            )
            return _fallback_info(video_path)

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            log.warning("ffprobe JSON invalido: %s", stdout[:200])
            return _fallback_info(video_path)

    except FileNotFoundError:
        log.error("ffprobe nao encontrado")
        return _fallback_info(video_path)
    except subprocess.TimeoutExpired:
        log.error("ffprobe timeout")
        return _fallback_info(video_path)
    except Exception as exc:
        log.error("ffprobe unexpected: %s", exc)
        return _fallback_info(video_path)

    duration = 0.0
    width = 0
    height = 0
    codec = ""
    fps = 0.0

    if "format" in data and data["format"].get("duration"):
        try:
            duration = float(data["format"]["duration"])
        except (TypeError, ValueError):
            duration = 0.0

    for stream in data.get("streams") or []:
        if stream.get("codec_type") == "video":
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
            codec = str(stream.get("codec_name") or "")
            avg_fps = stream.get("avg_frame_rate") or "0/1"
            try:
                if isinstance(avg_fps, str) and "/" in avg_fps:
                    num, den = avg_fps.split("/")
                    fps = float(num) / float(den) if float(den) else 0.0
                else:
                    fps = float(avg_fps)
            except (ValueError, ZeroDivisionError):
                fps = 0.0
            break

    if duration <= 0:
        # still return what we have; UI can warn
        log.warning("duration unknown for %s", video_path.name)

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
    size_bytes = video_path.stat().st_size if video_path.exists() else 0
    duration = 0.0
    try:
        ffmpeg = get_ffmpeg_path()
        result = subprocess.run(
            [ffmpeg, "-i", str(video_path)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        import re

        err = result.stderr or ""
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", err)
        if match:
            h, m, s = match.groups()
            duration = int(h) * 3600 + int(m) * 60 + float(s)
    except Exception as exc:
        log.warning("fallback duration failed: %s", exc)

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
    ffmpeg = get_ffmpeg_path()
    if output_path is None:
        output_path = create_temp_file(suffix=".mp3", prefix="audio_")

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
            ffmpeg, "-y", "-i", str(video_path),
            "-vn", "-ac", str(channels), "-ar", str(sample_rate),
            *codec_args, str(out),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
            if out.exists() and out.stat().st_size > 0:
                return out
        except subprocess.CalledProcessError as exc:
            last_error = (exc.stderr or str(exc))[:400]
        except subprocess.TimeoutExpired:
            raise RuntimeError("Timeout ao extrair audio.") from None

    raise RuntimeError(f"Falha ao extrair audio: {last_error or 'codec indisponivel'}")


def split_audio_chunks(
    audio_path: Path,
    chunk_duration_sec: float = 300.0,
) -> list[Path]:
    ffmpeg = get_ffmpeg_path()
    out_dir = audio_path.parent
    pattern = str(out_dir / f"{audio_path.stem}_chunk_%03d{audio_path.suffix}")
    cmd = [
        ffmpeg, "-y", "-i", str(audio_path),
        "-f", "segment", "-segment_time", str(int(chunk_duration_sec)),
        "-c", "copy", "-reset_timestamps", "1", pattern,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Falha ao dividir audio: {(exc.stderr or str(exc))[:300]}"
        ) from exc

    chunks = sorted(out_dir.glob(f"{audio_path.stem}_chunk_*{audio_path.suffix}"))
    if not chunks:
        raise RuntimeError("Nenhum chunk de audio gerado.")
    return chunks


def cut_video(
    input_path: Path,
    start: float,
    end: float,
    output_path: Path,
    mode: str = "fast",
) -> Path:
    ffmpeg = get_ffmpeg_path()
    start, end = validate_timestamps(start, end, video_duration=1e9)
    duration = end - start

    if mode == "fast":
        cmd = [
            ffmpeg, "-y", "-ss", f"{start:.3f}", "-i", str(input_path),
            "-t", f"{duration:.3f}", "-c", "copy", "-avoid_negative_ts", "make_zero",
            str(output_path),
        ]
    else:
        cmd = [
            ffmpeg, "-y", "-i", str(input_path),
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", str(output_path),
        ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Falha ao cortar: {(exc.stderr or str(exc))[:400]}"
        ) from exc
    except subprocess.TimeoutExpired:
        raise RuntimeError("Timeout ao cortar o video.") from None

    if not output_path.exists() or output_path.stat().st_size < 1000:
        raise RuntimeError("Arquivo de saida do clipe nao foi criado.")
    return output_path


def build_cut_command(
    ffmpeg_path: str,
    input_path: str,
    start: float,
    duration: float,
    output_path: str,
    mode: str = "fast",
) -> list[str]:
    if mode == "fast":
        return [
            ffmpeg_path, "-y", "-ss", f"{start:.3f}", "-i", input_path,
            "-t", f"{duration:.3f}", "-c", "copy", "-avoid_negative_ts", "make_zero",
            output_path,
        ]
    return [
        ffmpeg_path, "-y", "-i", input_path,
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", output_path,
    ]
