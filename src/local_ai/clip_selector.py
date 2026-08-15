"""
Local clip segment selection via Ollama (no Groq).
Uses transcription text only — does not replace FFmpeg cutting.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.clip_analyzer import ClipCandidate, _build_segments_text, _density_fallback, _validate_candidate
from src.config import ANALYSIS_MODEL, MAX_CLIP_DURATION, MIN_CLIP_DURATION
from src.ollama_manager import (
    DEFAULT_VISION_MODEL,
    ensure_optimized_model,
    is_ollama_running,
    model_is_installed,
    BASE_VISION_MODEL,
)
from src.transcription import TranscriptionResult
from src.utils import extract_json_from_text

log = logging.getLogger("video_clipper.local_clip")

LOCAL_SYSTEM = """Você é um editor de vídeos curtos.
Escolha o MELHOR trecho contínuo de 30 a 50 segundos na transcrição com timestamps.
Responda SOMENTE JSON:
{"start": 12.5, "end": 55.0, "duration": 42.5, "reason": "...", "hook": "...", "score": 85}
Regras: end-start entre 30 e 50 quando possível; nunca > 50; start/end em segundos da transcrição.
"""


def _pick_ollama_model() -> str:
    """Prefer lightweight text-capable local model already on machine."""
    # Optimized VL can do text; also accept base
    if model_is_installed(DEFAULT_VISION_MODEL):
        return DEFAULT_VISION_MODEL
    if model_is_installed(BASE_VISION_MODEL):
        return BASE_VISION_MODEL
    # common small text models if user has any
    for name in ("llama3.2:3b", "llama3.2", "qwen2.5:3b", "qwen2.5:7b", "mistral"):
        if model_is_installed(name):
            return name
    return ensure_optimized_model()


def analyze_best_clip_local(
    transcription: TranscriptionResult,
    video_duration: float,
) -> ClipCandidate:
    """
    Select best 30–50s window using local Ollama.
    Falls back to density heuristic if model fails.
    """
    if not transcription.segments and not transcription.text:
        raise ValueError("Transcrição vazia.")
    if video_duration <= 0:
        raise ValueError("Duração inválida.")

    if not is_ollama_running():
        log.warning("Ollama offline — density fallback")
        return _density_fallback(transcription, video_duration)

    if transcription.segments:
        content = _build_segments_text(transcription.segments)
    else:
        content = transcription.text

    max_chars = 20000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n...[truncado]..."

    user = (
        f"Duração do vídeo: {video_duration:.1f}s\n\n"
        f"Transcrição:\n{content}\n\n"
        "JSON apenas."
    )

    model = _pick_ollama_model()
    log.info("Local clip selection with model=%s", model)

    try:
        import ollama

        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": LOCAL_SYSTEM},
                {"role": "user", "content": user},
            ],
            options={
                "temperature": 0.1,
                "num_predict": 400,
                "num_ctx": 4096,
                "num_batch": 256,
            },
            format="json",
        )
        if isinstance(response, dict):
            raw = response.get("message", {}).get("content", "") or ""
        else:
            raw = getattr(getattr(response, "message", None), "content", "") or ""

        data = extract_json_from_text(raw)
        if data is None:
            log.warning("Local JSON parse failed — density fallback")
            cand = _density_fallback(transcription, video_duration)
            cand.raw_response = raw
            return cand

        cand = _validate_candidate(data, video_duration)
        cand.raw_response = raw
        cand.reason = (cand.reason or "") + " [seleção local Ollama]"
        return cand

    except Exception as exc:
        log.error("Local analysis failed: %s", exc)
        cand = _density_fallback(transcription, video_duration)
        cand.reason += f" (fallback local: {exc})"
        return cand
