"""Audio transcription via Groq Whisper API."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from groq import Groq
from groq import APIError, APIConnectionError, RateLimitError, AuthenticationError

from src.config import TRANSCRIPTION_MODEL, require_api_key, logger
from src.utils import cleanup_file
from src.video_processor import GROQ_SAFE_UPLOAD_MB, extract_audio, split_audio_chunks

log = logging.getLogger("video_clipper.transcription")

# Always split into small pieces for reliability on free-tier limits
CHUNK_DURATION_SEC = 300.0  # 5 minutes
# Force chunking above this size (MB)
FORCE_CHUNK_MB = 8.0


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


def _parse_response(transcription: Any, time_offset: float = 0.0) -> TranscriptionResult:
    """Parse a Groq verbose_json response into TranscriptionResult."""
    text = getattr(transcription, "text", "") or ""
    language = getattr(transcription, "language", None)
    duration = getattr(transcription, "duration", None)

    segments: list[Segment] = []
    raw_segments = getattr(transcription, "segments", None) or []
    for seg in raw_segments:
        if isinstance(seg, dict):
            s = float(seg.get("start", 0))
            e = float(seg.get("end", 0))
            t = str(seg.get("text", "")).strip()
        else:
            s = float(getattr(seg, "start", 0))
            e = float(getattr(seg, "end", 0))
            t = str(getattr(seg, "text", "")).strip()
        if t:
            segments.append(
                Segment(start=s + time_offset, end=e + time_offset, text=t)
            )

    return TranscriptionResult(
        text=text,
        segments=segments,
        language=language,
        duration=duration,
        raw=transcription,
    )


def _transcribe_single_file(
    client: Groq,
    audio_path: Path,
    model: str,
    language: Optional[str],
    time_offset: float = 0.0,
) -> TranscriptionResult:
    """Send one audio file to Groq Whisper."""
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    log.info(
        "Transcribing chunk: %s (%.2f MB, offset=%.1fs)",
        audio_path.name,
        size_mb,
        time_offset,
    )

    if size_mb > GROQ_SAFE_UPLOAD_MB:
        raise RuntimeError(
            f"Chunk ainda grande demais ({size_mb:.1f} MB). "
            f"Limite seguro: {GROQ_SAFE_UPLOAD_MB:.0f} MB."
        )

    try:
        with audio_path.open("rb") as audio_file:
            kwargs: dict[str, Any] = {
                "file": (audio_path.name, audio_file.read()),
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

    return _parse_response(transcription, time_offset=time_offset)


def transcribe_audio(
    audio_path: Path,
    language: Optional[str] = None,
    model: str = TRANSCRIPTION_MODEL,
) -> TranscriptionResult:
    """
    Transcribe an audio file using Groq's Whisper endpoint.

    Files above FORCE_CHUNK_MB are split into 5-minute pieces, transcribed
    separately, and timestamps are stitched back together.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")

    size_mb = audio_path.stat().st_size / (1024 * 1024)
    client = get_client()
    log.info("Starting transcription with model=%s, file=%.2f MB", model, size_mb)

    # Fast path for short/small audio
    if size_mb <= FORCE_CHUNK_MB:
        result = _transcribe_single_file(client, audio_path, model, language)
        log.info(
            "Transcription finished: %d chars, %d segments, duration=%.1fs",
            len(result.text),
            len(result.segments),
            result.duration or 0,
        )
        return result

    # Chunk path — used for longer videos
    log.info(
        "Audio is %.2f MB (> %.0f MB). Splitting into %.0fs chunks…",
        size_mb,
        FORCE_CHUNK_MB,
        CHUNK_DURATION_SEC,
    )
    chunks = split_audio_chunks(audio_path, chunk_duration_sec=CHUNK_DURATION_SEC)
    all_segments: list[Segment] = []
    all_text_parts: list[str] = []
    language_detected: Optional[str] = None
    total_duration = 0.0

    try:
        for i, chunk_path in enumerate(chunks):
            offset = i * CHUNK_DURATION_SEC
            part = _transcribe_single_file(
                client, chunk_path, model, language, time_offset=offset
            )
            all_segments.extend(part.segments)
            if part.text.strip():
                all_text_parts.append(part.text.strip())
            if part.language and not language_detected:
                language_detected = part.language
            if part.duration:
                total_duration = max(total_duration, offset + part.duration)
    finally:
        for c in chunks:
            cleanup_file(c)

    result = TranscriptionResult(
        text=" ".join(all_text_parts),
        segments=all_segments,
        language=language_detected,
        duration=total_duration or None,
        raw=None,
    )
    log.info(
        "Chunked transcription finished: %d chars, %d segments",
        len(result.text),
        len(result.segments),
    )
    return result


def transcribe_video(
    video_path: Path,
    language: Optional[str] = None,
) -> tuple[TranscriptionResult, Path]:
    """
    Extract compressed audio then transcribe.
    Returns (result, audio_path) so the caller can clean up the audio file.
    """
    audio_path = extract_audio(video_path)
    try:
        result = transcribe_audio(audio_path, language=language)
        return result, audio_path
    except Exception:
        cleanup_file(audio_path)
        raise
