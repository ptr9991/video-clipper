"""Local visual analysis of clips via Ollama Qwen2.5-VL."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.frame_extractor import cleanup_frames, extract_frames
from src.ollama_manager import DEFAULT_VISION_MODEL, is_ollama_running
from src.scoring import FinalScore, combine_scores
from src.transcription import Segment
from src.utils import extract_json_from_text

log = logging.getLogger("video_clipper.visual")

SYSTEM_PROMPT = """Você é um editor especialista em vídeos curtos (TikTok, Reels, Shorts).
Analise as imagens (frames em ordem temporal) e a transcrição do clipe.
Responda SOMENTE com JSON válido, sem markdown:
{
  "overall_score": 0-100,
  "visual_hook_score": 0-100,
  "retention_score": 0-100,
  "composition_score": 0-100,
  "emotion_score": 0-100,
  "visual_quality_score": 0-100,
  "context_match_score": 0-100,
  "short_form_score": 0-100,
  "verdict": "APPROVE",
  "confidence": 0.0-1.0,
  "problems": ["..."],
  "strengths": ["..."],
  "suggestions": ["..."],
  "suggested_start": null,
  "suggested_end": null
}
verdict deve ser APPROVE, REVIEW ou REJECT.
suggested_start/end são offsets em segundos relativos ao clipe (ou null).
"""


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
) -> str:
    lines = []
    for seg in segments:
        if seg.end < clip_start or seg.start > clip_end:
            continue
        rel_s = max(0.0, seg.start - clip_start)
        rel_e = max(0.0, seg.end - clip_start)
        lines.append(f"[{rel_s:.1f}s–{rel_e:.1f}s] {seg.text}")
    return "\n".join(lines) if lines else "(sem transcrição neste trecho)"


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
        problems=[str(p) for p in problems][:8],
        strengths=[str(s) for s in strengths][:8],
        suggestions=[str(s) for s in suggestions][:8],
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
    model: str = DEFAULT_VISION_MODEL,
    max_frames: int = 10,
) -> VisualAnalysis:
    """
    Extract frames + call Ollama vision model.
    Does not send video to any external server — only local Ollama.
    """
    if not is_ollama_running():
        raise RuntimeError("Ollama não está em execução.")

    frames = extract_frames(clip_path, duration=clip_duration, max_frames=max_frames, width=512)
    if not frames:
        raise RuntimeError("Não foi possível extrair frames do clipe.")

    transcript_block = _segments_for_clip(segments, clip_start_abs, clip_end_abs)
    user_text = (
        f"Clipe de {clip_duration:.1f} segundos. Frames em ordem temporal.\n\n"
        f"Transcrição:\n{transcript_block}\n\n"
        "Avalie hook visual, retenção, composição, emoção, qualidade e match com a fala. "
        "Responda só JSON."
    )

    t0 = time.time()
    try:
        import ollama

        # Limit images to avoid VRAM blow-up on 8GB cards
        image_paths = [str(p) for p in frames[:max_frames]]
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_text,
                    "images": image_paths,
                },
            ],
            options={
                "temperature": 0.1,
                "num_predict": 800,
            },
            format="json",
        )
        raw = response["message"]["content"] if isinstance(response, dict) else response.message.content
    except Exception as exc:
        cleanup_frames(frames)
        raise RuntimeError(f"Falha na IA visual local: {exc}") from exc
    finally:
        # Always free disk
        pass

    ms = int((time.time() - t0) * 1000)
    try:
        data = _parse_visual_json(raw or "")
        analysis = _to_analysis(data, raw or "", len(frames), ms)
        analysis.final = combine_scores(speech_score, data)
        # Prefer combined overall if available
        if analysis.final:
            analysis.overall_score = analysis.final.overall
            analysis.verdict = analysis.final.verdict
    finally:
        cleanup_frames(frames)

    log.info(
        "Visual analysis done: score=%d verdict=%s frames=%d ms=%d",
        analysis.overall_score,
        analysis.verdict,
        analysis.frames_used,
        analysis.inference_ms,
    )
    return analysis
