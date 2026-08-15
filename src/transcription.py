"""Transcription with segment + word-level timestamps (Groq or CPU local)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from groq import Groq
from groq import APIError, APIConnectionError, RateLimitError, AuthenticationError

from src.captions import WordStamp
from src.config import TRANSCRIPTION_MODEL, require_api_key
from src.utils import cleanup_file
from src.video_processor import GROQ_SAFE_UPLOAD_MB, extract_audio, split_audio_chunks

log = logging.getLogger("video_clipper.transcription")

CHUNK_DURATION_SEC = 300.0
FORCE_CHUNK_MB = 8.0

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
    words: list[WordStamp] = field(default_factory=list)
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
    for seg in getattr(transcription, "segments", None) or []:
        if isinstance(seg, dict):
            s, e = float(seg.get("start", 0)), float(seg.get("end", 0))
            t = str(seg.get("text", "")).strip()
        else:
            s = float(getattr(seg, "start", 0))
            e = float(getattr(seg, "end", 0))
            t = str(getattr(seg, "text", "")).strip()
        if t:
            segments.append(Segment(start=s + time_offset, end=e + time_offset, text=t))

    words: list[WordStamp] = []
    raw_words = getattr(transcription, "words", None) or []
    for w in raw_words:
        if isinstance(w, dict):
            ww = str(w.get("word", "")).strip()
            ws = float(w.get("start", 0))
            we = float(w.get("end", 0))
            conf = float(w.get("probability", w.get("confidence", 1.0)) or 1.0)
        else:
            ww = str(getattr(w, "word", "")).strip()
            ws = float(getattr(w, "start", 0))
            we = float(getattr(w, "end", 0))
            conf = float(getattr(w, "probability", getattr(w, "confidence", 1.0)) or 1.0)
        if ww:
            words.append(
                WordStamp(word=ww, start=ws + time_offset, end=we + time_offset, confidence=conf)
            )

    return TranscriptionResult(
        text=text,
        segments=segments,
        words=words,
        language=language,
        duration=duration,
        raw=transcription,
        source="groq",
    )


def transcribe_local_faster_whisper(
    audio_path: Path,
    language: Optional[str] = None,
) -> TranscriptionResult:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Transcricao local indisponivel. pip install faster-whisper"
        ) from exc

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    model_name = os.environ.get("VIDEOCLIPPER_WHISPER_MODEL", "tiny")
    model = WhisperModel(
        model_name, device="cpu", compute_type="int8", cpu_threads=4, num_workers=1
    )
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=1,
        best_of=1,
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    segments: list[Segment] = []
    words: list[WordStamp] = []
    parts: list[str] = []
    for seg in segments_iter:
        t = (seg.text or "").strip()
        if t:
            segments.append(Segment(float(seg.start), float(seg.end), t))
            parts.append(t)
        for w in getattr(seg, "words", None) or []:
            ww = (getattr(w, "word", "") or "").strip()
            if not ww:
                continue
            words.append(
                WordStamp(
                    word=ww,
                    start=float(w.start),
                    end=float(w.end),
                    confidence=float(getattr(w, "probability", 1.0) or 1.0),
                )
            )

    return TranscriptionResult(
        text=" ".join(parts),
        segments=segments,
        words=words,
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
    if size_mb > GROQ_SAFE_UPLOAD_MB:
        raise RuntimeError(f"Chunk grande demais ({size_mb:.1f} MB).")

    try:
        with audio_path.open("rb") as audio_file:
            kwargs: dict[str, Any] = {
                "file": (audio_path.name, audio_file.read()),
                "model": model,
                "response_format": "verbose_json",
                # word + segment for professional short-form captions
                "timestamp_granularities": ["word", "segment"],
            }
            if language:
                kwargs["language"] = language
            transcription = client.audio.transcriptions.create(**kwargs)
    except AuthenticationError as exc:
        raise RuntimeError("Chave Groq invalida.") from exp if False else RuntimeError("Chave Groq invalida.") from exp if False else None

    except AuthenticationError as exc:
        raise RuntimeError("Chave Groq invalida.") from exp if False else RuntimeError("Chave Groq invalida.") from exc
    except RateLimitError:
        raise
    except APIConnectionError as exc:
        raise RuntimeError("Falha de conexao Groq.") from exc
    except APIError as exc:
        # fallback without word granularity
        log.warning("word timestamps failed, retry segment-only: %s", exc)
        with audio_path.open("rb") as audio_file:
            kwargs = {
                "file": (audio_path.name, audio_file.read()),
                "model": model,
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment"],
            }
            if language:
                kwargs["language"] = language
            transcription = client.audio.transcriptions.create(**kwargs)

    return _parse_response(transcription, time_offset=time_offset)


def transcribe_audio(
    audio_path: Path,
    language: Optional[str] = None,
    model: str = TRANSCRIPTION_MODEL,
    prefer_local: bool = False,
) -> TranscriptionResult:
    if not audio_path.exists():
        raise FileNotFoundError(str(audio_path))

    if prefer_local:
        return transcribe_local_faster_whisper(audio_path, language=language)

    size_mb = audio_path.stat().st_size / (1024 * 1024)
    try:
        client = get_client()
        if size_mb <= FORCE_CHUNK_MB:
            return _transcribe_single_file(client, audio_path, model, language)

        chunks = split_audio_chunks(audio_path, chunk_duration_sec=CHUNK_DURATION_SEC)
        all_segments: list[Segment] = []
        all_words: list[WordStamp] = []
        all_text: list[str] = []
        lang = None
        total_duration = 0.0
        try:
            for i, chunk_path in enumerate(chunks):
                offset = i * CHUNK_DURATION_SEC
                part = _transcribe_single_file(
                    client, chunk_path, model, language, time_offset=offset
                )
                all_segments.extend(part.segments)
                all_words.extend(part.words)
                if part.text.strip():
                    all_text.append(part.text.strip())
                if part.language and not lang:
                    lang = part.language
                if part.duration:
                    total_duration = max(total_duration, offset + part.duration)
        finally:
            for c in chunks:
                cleanup_file(c)

        return TranscriptionResult(
            text=" ".join(all_text),
            segments=all_segments,
            words=all_words,
            language=lang,
            duration=total_duration or None,
            source="groq",
        )
    except RateLimitError:
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
        return transcribe_audio(audio_path, language=language, prefer_local=prefer_local), audio_path
    except Exception:
        cleanup_file(audio_path)
        raise
