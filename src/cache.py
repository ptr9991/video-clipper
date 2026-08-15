"""Local disk cache keyed by file SHA-256 — avoid re-transcription / re-analysis."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from src.config import TEMP_DIR

log = logging.getLogger("video_clipper.cache")

CACHE_ROOT = Path.home() / "AppData" / "Local" / "VideoClipper" / "cache"
# Fallback for non-Windows
if not (Path.home() / "AppData").exists():
    CACHE_ROOT = TEMP_DIR / "cache"


def file_sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _dir(video_hash: str) -> Path:
    d = CACHE_ROOT / video_hash[:2] / video_hash
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_json(video_hash: str, name: str) -> Optional[dict[str, Any]]:
    p = _dir(video_hash) / f"{name}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("cache read fail %s: %s", p, exc)
        return None


def save_json(video_hash: str, name: str, data: dict[str, Any]) -> Path:
    p = _dir(video_hash) / f"{name}.json"
    data = {**data, "_cached_at": time.time()}
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("cache write %s", p)
    return p


def thumb_path(video_hash: str, start: float) -> Path:
    return _dir(video_hash) / f"thumb_{int(start * 1000)}.jpg"
