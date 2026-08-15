"""Download videos from URLs using yt-dlp (YouTube, etc.)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from src.config import TEMP_DIR, get_ffmpeg_path

log = logging.getLogger("video_clipper.downloader")

QUALITY_PRESETS: dict[str, str] = {
    "melhor": "bv*+ba/b",
    "1080p": "bv*[height<=1080]+ba/b[height<=1080]/b",
    "720p": "bv*[height<=720]+ba/b[height<=720]/b",
    "480p": "bv*[height<=480]+ba/b[height<=480]/b",
    "360p": "bv*[height<=360]+ba/b[height<=360]/b",
}


def _ffmpeg_location() -> Optional[str]:
    try:
        ff = Path(get_ffmpeg_path())
        if ff.exists():
            return str(ff.parent)
    except RuntimeError:
        pass
    return None


def _base_opts(out_dir: Path, fmt: str) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "format": fmt,
        "outtmpl": str(out_dir / "%(title).80B [%(id)s].%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 5,
        "fragment_retries": 5,
        "file_access_retries": 3,
        "concurrent_fragment_downloads": 1,
        # Reduce 403 from YouTube by preferring mobile clients
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "web"],
            }
        },
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
        "postprocessors": [
            {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"},
        ],
    }
    ff = _ffmpeg_location()
    if ff:
        opts["ffmpeg_location"] = ff
    return opts


def _resolve_path(ydl: Any, info: dict, out_dir: Path) -> Path:
    if "requested_downloads" in info and info["requested_downloads"]:
        filepath = info["requested_downloads"][0].get("filepath")
        if filepath and Path(filepath).exists():
            return Path(filepath)

    prepared = Path(ydl.prepare_filename(info))
    if prepared.exists():
        return prepared
    for ext in (".mp4", ".mkv", ".webm", ".mov"):
        cand = prepared.with_suffix(ext)
        if cand.exists():
            return cand

    files = sorted(
        [f for f in out_dir.glob("*") if f.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise RuntimeError("Download concluído, mas o arquivo não foi encontrado.")
    return files[0]


def download_video(
    url: str,
    quality: str = "720p",
    output_dir: Optional[Path] = None,
) -> Path:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp não está instalado. No PowerShell:\n"
            r'& "$env:LOCALAPPDATA\VideoClipper\runtime\python\python.exe" -m pip install -U yt-dlp'
        ) from exc

    url = (url or "").strip()
    if not url:
        raise ValueError("URL vazia.")
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL inválida. Cole um link completo (https://...).")

    out_dir = output_dir or TEMP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["720p"])

    # Attempt 1: preferred quality + mobile clients
    # Attempt 2: progressive single-file (often avoids 403 on fragments)
    attempts = [
        _base_opts(out_dir, fmt),
        {
            **_base_opts(out_dir, "18/22/best[ext=mp4]/best"),
            "extractor_args": {"youtube": {"player_client": ["android"]}},
        },
        {
            **_base_opts(out_dir, "best[ext=mp4]/best"),
            "extractor_args": {"youtube": {"player_client": ["ios", "android"]}},
        },
    ]

    last_err = ""
    for i, ydl_opts in enumerate(attempts, 1):
        log.info("Download attempt %d: %s", i, url)
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    raise RuntimeError("Não foi possível obter informações do vídeo.")
                path = _resolve_path(ydl, info, out_dir)
                log.info("Downloaded: %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
                return path
        except yt_dlp.utils.DownloadError as exc:
            last_err = str(exc)
            log.warning("Attempt %d failed: %s", i, last_err[:200])
            # try next strategy
            continue
        except Exception as exc:
            last_err = str(exc)
            log.warning("Attempt %d error: %s", i, last_err[:200])
            continue

    msg = last_err or "erro desconhecido"
    low = msg.lower()
    if "403" in msg or "forbidden" in low:
        raise RuntimeError(
            "YouTube bloqueou o download (HTTP 403).\n\n"
            "Tente isto:\n"
            "1) Atualizar yt-dlp:\n"
            r'   & "$env:LOCALAPPDATA\VideoClipper\runtime\python\python.exe" -m pip install -U yt-dlp'\n"
            "2) Usar qualidade 360p ou 480p\n"
            "3) Baixar o MP4 no navegador e usar a aba Arquivo\n"
            f"\nDetalhe: {msg[:250]}"
        )
    if "private" in low:
        raise RuntimeError("Vídeo privado — não pode ser baixado.")
    if "age" in low:
        raise RuntimeError("Restrição de idade — não foi possível baixar.")
    raise RuntimeError(f"Falha no download: {msg[:350]}")
