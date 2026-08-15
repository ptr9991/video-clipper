"""Local Ollama multi-clip selection."""

from __future__ import annotations

import logging

from src.clip_analyzer import (
    ClipCandidate,
    _build_segments_text,
    _density_multi,
    _snip,
    _validate_one,
    diversify,
)
from src.ollama_manager import (
    BASE_VISION_MODEL,
    DEFAULT_VISION_MODEL,
    ensure_optimized_model,
    is_ollama_running,
    model_is_installed,
)
from src.transcription import TranscriptionResult
from src.utils import extract_json_from_text

log = logging.getLogger("video_clipper.local_clip")

LOCAL_SYSTEM = """Editor de shorts. Escolha vários trechos 30–50s distintos.
JSON only: {"candidates":[{"start":0,"end":40,"duration":40,"reason":"","hook":"","score":80,"title_hint":""}]}
"""


def _pick_model() -> str:
    if model_is_installed(DEFAULT_VISION_MODEL):
        return DEFAULT_VISION_MODEL
    if model_is_installed(BASE_VISION_MODEL):
        return BASE_VISION_MODEL
    for name in ("llama3.2:3b", "qwen2.5:3b", "qwen2.5:7b"):
        if model_is_installed(name):
            return name
    return ensure_optimized_model()


def analyze_best_clips_local(
    transcription: TranscriptionResult,
    video_duration: float,
    n: int = 5,
) -> list[ClipCandidate]:
    n = max(1, min(15, n))
    if not is_ollama_running():
        return _density_multi(transcription, video_duration, n)

    content = (
        _build_segments_text(transcription.segments)
        if transcription.segments
        else transcription.text
    )
    if len(content) > 20000:
        content = content[:20000] + "\n..."

    user = f"Duração {video_duration:.1f}s. Até {n} candidatos diversos.\n{content}\nJSON."
    model = _pick_model()
    try:
        import ollama

        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": LOCAL_SYSTEM},
                {"role": "user", "content": user},
            ],
            options={"temperature": 0.2, "num_predict": 1500, "num_ctx": 4096, "num_batch": 256},
            format="json",
        )
        raw = (
            response.get("message", {}).get("content", "")
            if isinstance(response, dict)
            else getattr(getattr(response, "message", None), "content", "") or ""
        )
        data = extract_json_from_text(raw) or {}
        items = data.get("candidates") if isinstance(data, dict) else None
        out: list[ClipCandidate] = []
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict):
                    c = _validate_one(it, video_duration)
                    if c:
                        c.transcript_snip = _snip(transcription.segments, c.start, c.end)
                        out.append(c)
        if not out:
            return _density_multi(transcription, video_duration, n)
        return diversify(out, n)
    except Exception as exc:
        log.error("local multi-clip fail: %s", exc)
        return _density_multi(transcription, video_duration, n)


def analyze_best_clip_local(transcription, video_duration):
    return analyze_best_clips_local(transcription, video_duration, n=1)[0]
