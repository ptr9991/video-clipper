"""Configuration module for the Video Clipper application."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env if present (never commit real keys)
load_dotenv()

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
FFMPEG_PATH: Optional[str] = os.getenv("FFMPEG_PATH")
DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "output"

TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Models (current as of 2026)
# ---------------------------------------------------------------------------
TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
ANALYSIS_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Clip constraints
# ---------------------------------------------------------------------------
MIN_CLIP_DURATION = 30.0  # seconds
MAX_CLIP_DURATION = 50.0  # seconds
PREFERRED_MIN = 40.0
PREFERRED_MAX = 50.0

# ---------------------------------------------------------------------------
# Audio extraction settings (optimised for Whisper)
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = "wav"  # Whisper accepts wav well; flac is also fine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("video_clipper")


def get_ffmpeg_path() -> str:
    """
    Resolve the FFmpeg executable path.

    Priority:
    1. FFMPEG_PATH environment variable
    2. 'ffmpeg' found in system PATH

    Raises:
        RuntimeError: if FFmpeg cannot be located.
    """
    if FFMPEG_PATH:
        path = Path(FFMPEG_PATH)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        raise RuntimeError(
            f"FFMPEG_PATH is set to '{FFMPEG_PATH}' but the file is not executable."
        )

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    raise RuntimeError(
        "FFmpeg not found. Install FFmpeg and ensure it is in your PATH, "
        "or set the FFMPEG_PATH environment variable."
    )


def check_ffmpeg() -> tuple[bool, str]:
    """
    Check whether FFmpeg is available.

    Returns:
        (available: bool, message: str)
    """
    try:
        path = get_ffmpeg_path()
        return True, path
    except RuntimeError as exc:
        return False, str(exc)


def require_api_key() -> str:
    """Return GROQ_API_KEY or raise a clear error."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Create a .env file or export the variable before running the app."
        )
    return GROQ_API_KEY
