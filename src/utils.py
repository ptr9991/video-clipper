"""Utility helpers for the Video Clipper application."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("video_clipper.utils")


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.ms or MM:SS.ms."""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes:02d}:{secs:06.3f}"


def seconds_to_hms(seconds: float) -> str:
    """Alias for format_timestamp for readability."""
    return format_timestamp(seconds)


def safe_filename(name: str) -> str:
    """Sanitize a filename keeping only safe characters."""
    name = re.sub(r"[^\w\-_. ]", "_", name)
    return name.strip()[:200] or "clip"


def file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hash of a file (used for simple caching)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def create_temp_file(suffix: str = "", prefix: str = "clip_") -> Path:
    """Create a temporary file path inside the project temp directory."""
    from src.config import TEMP_DIR

    fd, name = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=str(TEMP_DIR))
    import os

    os.close(fd)
    return Path(name)


def cleanup_file(path: Optional[Path]) -> None:
    """Safely delete a file if it exists."""
    if path is None:
        return
    try:
        if path.exists() and path.is_file():
            path.unlink()
            logger.debug("Deleted temporary file: %s", path)
    except OSError as exc:
        logger.warning("Could not delete %s: %s", path, exc)


def extract_json_from_text(text: str) -> Optional[dict[str, Any]]:
    """
    Try to extract a JSON object from an LLM response that may contain
    markdown fences, trailing commas, or extra prose.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Markdown code block ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # First balanced { ... } occurrence
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    # Fix common LLM mistakes: trailing commas
                    cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        obj = json.loads(cleaned)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break

    # Last resort: find start/end numbers with regex
    start_m = re.search(r'"start"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    end_m = re.search(r'"end"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if start_m and end_m:
        result: dict[str, Any] = {
            "start": float(start_m.group(1)),
            "end": float(end_m.group(1)),
        }
        reason_m = re.search(r'"reason"\s*:\s*"((?:\\.|[^"\\])*)"', text)
        hook_m = re.search(r'"hook"\s*:\s*"((?:\\.|[^"\\])*)"', text)
        score_m = re.search(r'"score"\s*:\s*([0-9]+)', text)
        if reason_m:
            result["reason"] = reason_m.group(1)
        if hook_m:
            result["hook"] = hook_m.group(1)
        if score_m:
            result["score"] = int(score_m.group(1))
        return result

    return None


def generate_output_filename(prefix: str = "clip") -> str:
    """Generate a timestamped filename."""
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{now}.mp4"


def validate_timestamps(
    start: float,
    end: float,
    video_duration: float,
    max_duration: float = 50.0,
    min_duration: float = 0.5,
) -> tuple[float, float]:
    """
    Clamp and validate start/end timestamps.

    Returns corrected (start, end).
    Raises ValueError on impossible values.
    """
    if video_duration <= 0:
        raise ValueError("Video duration must be positive.")

    start = max(0.0, float(start))
    end = min(float(end), video_duration)

    if end <= start:
        raise ValueError(f"end ({end}) must be greater than start ({start}).")

    duration = end - start
    if duration > max_duration:
        end = start + max_duration
        if end > video_duration:
            end = video_duration
            start = max(0.0, end - max_duration)

    if (end - start) < min_duration:
        raise ValueError(
            f"Clip duration too short: {end - start:.2f}s (minimum {min_duration}s)."
        )

    return start, end
