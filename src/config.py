"""Configuration module for the Video Clipper application."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env if present (never commit real keys)
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "output"

# User-writable settings (API key etc.) live in AppData on Windows
def _user_data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path.home() / ".config"
    d = base / "VideoClipper"
    d.mkdir(parents=True, exist_ok=True)
    return d


USER_DATA_DIR = _user_data_dir()
SETTINGS_FILE = USER_DATA_DIR / "settings.json"

TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Environment variables (env takes highest priority)
# ---------------------------------------------------------------------------
DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

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
AUDIO_FORMAT = "wav"

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


def load_settings() -> dict:
    """Load local settings (API key etc.). Never logs the key."""
    if not SETTINGS_FILE.exists():
        return {}
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(data: dict) -> None:
    """Persist settings to user data directory."""
    SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_api_key() -> Optional[str]:
    """
    Resolve Groq API key in order of priority:
    1. Environment variable GROQ_API_KEY
    2. Local settings file (set by the Windows launcher GUI)
    """
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key.strip()
    settings = load_settings()
    key = settings.get("groq_api_key")
    if key and isinstance(key, str) and key.strip():
        return key.strip()
    return None


def set_api_key(key: str) -> None:
    """Save API key to local settings (never to the repo)."""
    settings = load_settings()
    settings["groq_api_key"] = key.strip()
    save_settings(settings)


def get_ffmpeg_path() -> str:
    """
    Resolve the FFmpeg executable path.

    Priority:
    1. FFMPEG_PATH environment variable
    2. Bundled FFmpeg next to the application (portable install)
    3. 'ffmpeg' found in system PATH

    Raises:
        RuntimeError: if FFmpeg cannot be located.
    """
    # 1. Explicit env
    env_path = os.getenv("FFMPEG_PATH")
    if env_path:
        path = Path(env_path)
        if path.is_file():
            return str(path)

    # 2. Bundled with the portable app (installer layout)
    candidates = [
        BASE_DIR / "runtime" / "ffmpeg" / "bin" / "ffmpeg.exe",
        BASE_DIR / "runtime" / "ffmpeg" / "ffmpeg.exe",
        BASE_DIR / "ffmpeg" / "bin" / "ffmpeg.exe",
        BASE_DIR / "ffmpeg.exe",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    # 3. System PATH
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    raise RuntimeError(
        "FFmpeg não encontrado. "
        "Na versão instalada pelo VideoClipperSetup.exe o FFmpeg já vem embutido. "
        "Se você está rodando pelo código-fonte, instale o FFmpeg ou defina FFMPEG_PATH."
    )


def check_ffmpeg() -> tuple[bool, str]:
    """Check whether FFmpeg is available. Returns (ok, path_or_message)."""
    try:
        path = get_ffmpeg_path()
        return True, path
    except RuntimeError as exc:
        return False, str(exc)


def require_api_key() -> str:
    """Return the API key or raise a clear error."""
    key = get_api_key()
    if not key:
        raise RuntimeError(
            "Chave da API Groq não configurada. "
            "Abra o Video Clipper pelo atalho e informe a chave na tela inicial, "
            "ou defina a variável de ambiente GROQ_API_KEY."
        )
    return key
