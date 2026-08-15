"""Local visual analysis of clips via Ollama Qwen2.5-VL (RTX 2070 tuned)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.frame_extractor import cleanup_frames, extract_frames
from src.ollama_manager import (
    DEFAULT_VISION_MODEL,
    ensure_optimized_model,
    is_ollama_running,
)
from src.scoring import FinalScore, combine_scores
from src.transcription import Segment
from src.utils import extract_json_from_text

log = logging.getLogger("video_clipper.visual")

# RTX 2070 8GB: few small frames keep vision tokens under num_ctx 4096
DEFAULT_MAX_FRAMES = 3
DEFAULT_FRAME_WIDTH = 320

SYSTEM_PROMPT = (
    "Editor de shorts. Com frames em ordem + fala, responda SÓ JSON:\n"
    '{"overall_score":80,"visual_hook_score":80,"retention_score":80,'
    '"composition_score":80,"emotion_score":80,"visual_quality_score":80,'
    '"context_match_score":80,"short_form_score":80,"verdict":"APPROVE",'
    '"confidence":0.8,"problems":[],"strengths":[],"suggestions":[],'
    '"suggested_start":null,"suggested_end":null}\n'
    "verdict: APPROVE|REVIEW|REJECT. suggested_* segundos relativos ou null."
)


@dataclass
class VisualAnalysis:
    overall_score: int = 0
    visual_hook_score: int = 0
    retention_score: int = 0
    composition_score: int = 0
    emotion_score: int = 0
    visual_quality_score: int = 0
    context_match_score: int = 0
    short_form_score: int = 0
    verdict: str = "REVIEW"
    confidence: float = 0.0
    problems: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    suggested_start: Optional[float] = None
    suggested_end: Optional[float] = None
    final: Optional[FinalScore] = None
    raw: str = ""
    frames_used: int = 0
    inference_ms: int = 0


def _segments_for_clip(
    segments: list[Segment],
    clip_start: float,
    clip_end: float,
    max_chars: int = 600,
) -> str:
    lines = []
    for seg in segments:
        if seg.end < clip_start or seg.start > clip_end:
            continue
        rel_s = max(0.0, seg.start - clip_start)
        rel_e = max(0.0, seg.end - clip_start)
        lines.append(f"[{rel_s:.0f}-{rel_e:.0f}s] {seg.text}")
    text = " | ".join(lines) if lines else "(sem fala)"
    return text[:max_chars]


def _parse_visual_json(raw: str) -> dict[str, Any]:
    data = extract_json_from_text(raw)
    if not data:
        raise ValueError("Resposta da IA visual não é JSON válido.")
    return data


def _to_analysis(data: dict[str, Any], raw: str, frames: int, ms: int) -> VisualAnalysis:
    def gi(key: str, default: int = 70) -> int:
        try:
            return int(max(0, min(100, float(data.get(key, default)))))
        except (TypeError, ValueError):
            return default

    def gf(key: str, default: float = 0.5) -> float:
        try:
            return float(data.get(key, default))
        except (TypeError, ValueError):
            return default

    verdict = str(data.get("verdict", "REVIEW")).upper()
    if verdict not in {"APPROVE", "REVIEW", "REJECT", "EXCELLENT"}:
        verdict = "REVIEW"

    ss = data.get("suggested_start")
    se = data.get("suggested_end")
    try:
        ss_f = float(ss) if ss is not None else None
    except (TypeError, ValueError):
        ss_f = None
    try:
        se_f = float(se) if se is not None else None
    except (TypeError, ValueError):
        se_f = None

    problems = data.get("problems") or []
    strengths = data.get("strengths") or []
    suggestions = data.get("suggestions") or []
    if not isinstance(problems, list):
        problems = [str(problems)]
    if not isinstance(strengths, list):
        strengths = [str(strengths)]
    if not isinstance(suggestions, list):
        suggestions = [str(suggestions)]

    return VisualAnalysis(
        overall_score=gi("overall_score"),
        visual_hook_score=gi("visual_hook_score"),
        retention_score=gi("retention_score"),
        composition_score=gi("composition_score"),
        emotion_score=gi("emotion_score"),
        visual_quality_score=gi("visual_quality_score"),
        context_match_score=gi("context_match_score"),
        short_form_score=gi("short_form_score"),
        verdict=verdict,
        confidence=max(0.0, min(1.0, gf("confidence", 0.7))),
        problems=[str(p) for p in problems][:5],
        strengths=[str(s) for s in strengths][:5],
        suggestions=[str(s) for s in suggestions][:5],
        suggested_start=ss_f,
        suggested_end=se_f,
        raw=raw,
        frames_used=frames,
        inference_ms=ms,
    )


def analyze_clip_visual(
    clip_path: Path,
    clip_duration: float,
    clip_start_abs: float,
    clip_end_abs: float,
    segments: list[Segment],
    speech_score: float = 70.0,
    model: Optional[str] = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> VisualAnalysis:
    """
    Visual analysis tuned for RTX 2070 8 GB:
    - derived model with num_ctx 4096 (avoids 128k KV allocation)
    - 3 frames @ 320px
    - short transcript + JSON-only prompt
    """
    if not is_ollama_running():
        raise RuntimeError("Ollama não está em execução.")

    model_name = model or ensure_optimized_model()

    frames = extract_frames(
        clip_path,
        duration=clip_duration,
        max_frames=max_frames,
        width=DEFAULT_FRAME_WIDTH,
    )
    if not frames:
        raise RuntimeError("Não foi possível extrair frames do clipe.")

    transcript_block = _segments_for_clip(segments, clip_start_abs, clip_end_abs)
    user_text = (
        f"{clip_duration:.0f}s, {len(frames)} frames.\n"
        f"Fala: {transcript_block}\nJSON only."
    )

    t0 = time.time()
    raw = ""
    try:
        import ollama

        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_text,
                    "images": [str(p) for p in frames],
                },
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
    except Exception as exc:
        cleanup_frames(frames)
        raise RuntimeError(f"Falha na IA visual local: {exc}") from exc

    ms = int((time.time() - t0) * 1000)
    try:
        data = _parse_visual_json(raw)
        analysis = _to_analysis(data, raw, len(frames), ms)
        analysis.final = combine_scores(speech_score, data)
        if analysis.final:
            analysis.overall_score = analysis.final.overall
            analysis.verdict = analysis.final.verdict
    finally:
        cleanup_frames(frames)

    log.info(
        "Visual analysis: score=%d verdict=%s frames=%d ms=%d model=%s",
        analysis.overall_score,
        analysis.verdict,
        analysis.frames_used,
        analysis.inference_ms,
        model_name,
    )
    return analysis
