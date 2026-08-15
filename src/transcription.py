"""Audio transcription via Groq Whisper, with safe CPU-only local fallback."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from groq import Groq
from groq import APIError, APIConnectionError, RateLimitError, AuthenticationError

from src.config import TRANSCRIPTION_MODEL, require_api_key
from src.utils import cleanup_file
from src.video_processor import GROQ_SAFE_UPLOAD_MB, extract_audio, split_audio_chunks

log = logging.getLogger("video_clipper.transcription")

CHUNK_DURATION_SEC = 300.0
FORCE_CHUNK_MB = 8.0

# Prevent accidental CUDA use in this process (safer on RTX 2070 8GB + 16GB RAM)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")


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
    source: str = "groq"


def get_client() -> Groq:
    return Groq(api_key=require_api_key())


def _parse_response(transcription: Any, time_offset: float = 0.0) -> TranscriptionResult:
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
            segments.append(Segment(start=s + time_offset, end=e + time_offset, text=t))
    return TranscriptionResult(
        text=text, segments=segments, language=language, duration=duration,
        raw=transcription, source="groq",
    )


def transcribe_local_faster_whisper(
    audio_path: Path,
    language: Optional[str] = None,
) -> TranscriptionResult:
    """
    CPU-only local transcription — NEVER uses GPU.
    Model: tiny/base int8 to protect 16GB RAM systems.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Transcricao local indisponivel. Instale: python -m pip install faster-whisper"
        ) from exc

    # Force CPU even if CUDA is installed
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    log.info("Local CPU transcription: %s", audio_path.name)
    # tiny = lightest; base = better quality still OK on CPU
    model_name = os.environ.get("VIDEOCLIPPER_WHISPER_MODEL", "tiny")
    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1,
    )
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=1,
        best_of=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    segments: list[Segment] = []
    parts: list[str] = []
    for seg in segments_iter:
        t = (seg.text or "").strip()
        if not t:
            continue
        segments.append(Segment(start=float(seg.start), end=float(seg.end), text=t))
        parts.append(t)

    return TranscriptionResult(
        text=" ".join(parts),
        segments=segments,
        language=getattr(info, "language", language),
        duration=segments[-1].end if segments else None,
        source="local",
    )


def _transcribe_single_file(
    client: Groq,
    audio_path: Path,
    model: str,
    language: Optional[str],
    time_offset: float = 0.0,
) -> TranscriptionResult:
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    log.info("Groq chunk: %s (%.2f MB)", audio_path.name, size_mb)
    if size_mb > GROQ_SAFE_UPLOAD_MB:
        raise RuntimeError(
            f"Chunk grande demais ({size_mb:.1f} MB). Limite: {GROQ_SAFE_UPLOAD_MB:.0f} MB."
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
        raise RuntimeError("Chave da API Groq invalida.") from exc
    except RateLimitError:
        raise
    except APIConnectionError as exc:
        raise RuntimeError("Falha de conexao com a API Groq.") from exc
    except APIError as exc:
        raise RuntimeError(f"Erro da API Groq: {exc}") from exp if False else RuntimeError(f"Erro da API Groq: {exc}") from exc

    return _parse_response(transcription, time_offset=time_offset)


def transcribe_audio(
    audio_path: Path,
    language: Optional[str] = None,
    model: str = TRANSCRIPTION_MODEL,
    prefer_local: bool = False,
) -> TranscriptionResult:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio nao encontrado: {audio_path}")

    if prefer_local:
        return transcribe_local_faster_whisper(audio_path, language=language)

    size_mb = audio_path.stat().st_size / (1024 * 1024)

    try:
        client = get_client()
        log.info("Groq transcription model=%s file=%.2f MB", model, size_mb)

        if size_mb <= FORCE_CHUNK_MB:
            return _transcribe_single_file(client, audio_path, model, language)

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

        return TranscriptionResult(
            text=" ".join(all_text_parts),
            segments=all_segments,
            language=language_detected,
            duration=total_duration or None,
            source="groq",
        )

    except RateLimitError:
        log.warning("Groq rate limit — CPU local fallback")
        return transcribe_local_faster_whisper(audio_path, language=language)
    except RuntimeError as exc:
        if "rate" in str(exc).lower() or "limite" in str(exc).lower():
            return transcribe_local_faster_whisper(audio_path, language=language)
        raise


def transcribe_video(
    video_path: Path,
    language: Optional[str] = None,
    prefer_local: bool = False,
) -> tuple[TranscriptionResult, Path]:
    audio_path = extract_audio(video_path)
    try:
        result = transcribe_audio(audio_path, language=language, prefer_local=prefer_local)
        return result, audio_path
    except Exception:
        cleanup_file(audio_path)
        raise
