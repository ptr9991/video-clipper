"""Audio transcription via Groq Whisper API."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from groq import Groq
from groq import APIError, APIConnectionError, RateLimitError, AuthenticationError

from src.config import TRANSCRIPTION_MODEL, require_api_key, logger
from src.video_processor import extract_audio

log = logging.getLogger("video_clipper.transcription")


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    segments: list[Segment] = field(default_factory=list)
    language: Optional[str] = None
    duration: Optional[float] = None
    raw: Any = None


def get_client() -> Groq:
    """Create a Groq client using the environment API key."""
    api_key = require_api_key()
    return Groq(api_key=api_key)


def transcribe_audio(
    audio_path: Path,
    language: Optional[str] = None,
    model: str = TRANSCRIPTION_MODEL,
) -> TranscriptionResult:
    """
    Transcribe an audio file using Groq's Whisper endpoint.

    Uses response_format="verbose_json" to obtain segment-level timestamps.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")

    size_mb = audio_path.stat().st_size / (1024 * 1024)
    if size_mb > 95:
        raise RuntimeError(
            f"Arquivo de áudio muito grande ({size_mb:.1f} MB). "
            "O limite da API Groq é aproximadamente 100 MB. "
            "Considere um vídeo mais curto ou comprima o áudio."
        )

    client = get_client()
    log.info("Starting transcription with model=%s, file=%.1f MB", model, size_mb)

    try:
        with audio_path.open("rb") as audio_file:
            kwargs: dict[str, Any] = {
                "file": audio_file,
                "model": model,
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment"],
            }
            if language:
                kwargs["language"] = language

            transcription = client.audio.transcriptions.create(**kwargs)

    except AuthenticationError as exc:
        log.error("Authentication error: %s", exc)
        raise RuntimeError(
            "Chave da API Groq inválida ou ausente. Verifique GROQ_API_KEY."
        ) from exc
    except RateLimitError as exc:
        log.error("Rate limit: %s", exc)
        raise RuntimeError(
            "Limite de requisições da API Groq atingido. Aguarde alguns minutos e tente novamente."
        ) from exc
    except APIConnectionError as exc:
        log.error("Connection error: %s", exc)
        raise RuntimeError(
            "Falha de conexão com a API Groq. Verifique sua internet."
        ) from exc
    except APIError as exc:
        log.error("API error: %s", exc)
        raise RuntimeError(f"Erro da API Groq: {exc}") from exc
    except Exception as exc:
        log.exception("Unexpected transcription error")
        raise RuntimeError(f"Erro inesperado na transcrição: {exc}") from exc

    # Parse response
    text = getattr(transcription, "text", "") or ""
    language = getattr(transcription, "language", None)
    duration = getattr(transcription, "duration", None)

    segments: list[Segment] = []
    raw_segments = getattr(transcription, "segments", None) or []
    for seg in raw_segments:
        # seg can be a dict or an object depending on SDK version
        if isinstance(seg, dict):
            s = float(seg.get("start", 0))
            e = float(seg.get("end", 0))
            t = str(seg.get("text", "")).strip()
        else:
            s = float(getattr(seg, "start", 0))
            e = float(getattr(seg, "end", 0))
            t = str(getattr(seg, "text", "")).strip()
        if t:
            segments.append(Segment(start=s, end=e, text=t))

    log.info(
        "Transcription finished: %d chars, %d segments, duration=%.1fs",
        len(text),
        len(segments),
        duration or 0,
    )

    return TranscriptionResult(
        text=text,
        segments=segments,
        language=language,
        duration=duration,
        raw=transcription,
    )


def transcribe_video(
    video_path: Path,
    language: Optional[str] = None,
) -> tuple[TranscriptionResult, Path]:
    """
    Convenience: extract audio then transcribe.
    Returns (result, audio_path) so the caller can clean up the audio file.
    """
    audio_path = extract_audio(video_path)
    try:
        result = transcribe_audio(audio_path, language=language)
        return result, audio_path
    except Exception:
        # Leave cleanup to the caller on success; on error we still clean
        from src.utils import cleanup_file

        cleanup_file(audio_path)
        raise
