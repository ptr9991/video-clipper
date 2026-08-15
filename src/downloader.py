"""Download videos from URLs using yt-dlp (YouTube, etc.)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.config import TEMP_DIR, get_ffmpeg_path
from src.utils import safe_filename

log = logging.getLogger("video_clipper.downloader")

# Quality presets mapped to yt-dlp format selectors
QUALITY_PRESETS: dict[str, str] = {
    "melhor": "bv*+ba/b",  # best video+audio
    "1080p": "bv*[height<=1080]+ba/b[height<=1080]",
    "720p": "bv*[height<=720]+ba/b[height<=720]",
    "480p": "bv*[height<=480]+ba/b[height<=480]",
    "360p": "bv*[height<=360]+ba/b[height<=360]",
}


def download_video(
    url: str,
    quality: str = "720p",
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Download a video from a URL using yt-dlp.

    Returns the path to the downloaded .mp4 file.
    Raises RuntimeError with a clear message on failure.
    """
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp não está instalado. Execute no PowerShell:\n"
            r'& "$env:LOCALAPPDATA\VideoClipper\runtime\python\python.exe" -m pip install yt-dlp'
        ) from exc

    url = (url or "").strip()
    if not url:
        raise ValueError("URL vazia.")
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL inválida. Cole um link completo (https://...).")

    out_dir = output_dir or TEMP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["720p"])

    # Point yt-dlp at the bundled FFmpeg when available
    ffmpeg_location = None
    try:
        ff = Path(get_ffmpeg_path())
        if ff.exists():
            ffmpeg_location = str(ff.parent)
    except RuntimeError:
        pass

    outtmpl = str(out_dir / "%(title).80B [%(id)s].%(ext)s")

    ydl_opts: dict = {
        "format": fmt,
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "fragment_retries": 3,
        # Prefer mp4 containers when possible
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
    }
    if ffmpeg_location:
        ydl_opts["ffmpeg_location"] = ffmpeg_location

    log.info("Downloading %s (quality=%s)", url, quality)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise RuntimeError("Não foi possível obter informações do vídeo.")

            # Resolve final path
            if "requested_downloads" in info and info["requested_downloads"]:
                filepath = info["requested_downloads"][0].get("filepath")
                if filepath and Path(filepath).exists():
                    path = Path(filepath)
                    log.info("Downloaded: %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
                    return path

            # Fallback: prepare_filename
            prepared = ydl.prepare_filename(info)
            path = Path(prepared)
            # Extension may have changed after merge
            if not path.exists():
                for ext in (".mp4", ".mkv", ".webm", ".mov"):
                    candidate = path.with_suffix(ext)
                    if candidate.exists():
                        path = candidate
                        break
            if not path.exists():
                # Last resort: newest file in out_dir
                files = sorted(out_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
                files = [f for f in files if f.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
                if not files:
                    raise RuntimeError("Download concluído, mas o arquivo não foi encontrado.")
                path = files[0]

            log.info("Downloaded: %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
            return path

    except yt_dlp.utils.DownloadError as exc:
        msg = str(exc)
        if "Private video" in msg or "private" in msg.lower():
            raise RuntimeError("Este vídeo é privado e não pode ser baixado.") from exc
        if "age" in msg.lower():
            raise RuntimeError("Vídeo com restrição de idade. Não foi possível baixar.") from exc
        raise RuntimeError(f"Falha no download: {msg[:300]}") from exc
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Erro ao baixar vídeo: {exc}") from exc
