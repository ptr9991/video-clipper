"""
Visual review of clips.

Default path is LIGHTWEIGHT (no neural net / no Ollama GPU load):
  - samples a few frames from the CLIP only (never the full video)
  - brightness, contrast, blur proxy, motion between frames via FFmpeg/PIL-free stats

Heavy Qwen2.5-VL path is OPTIONAL and OFF by default (can BSOD weak/unstable drivers).
"""

from __future__ import annotations

import logging
import struct
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.config import get_ffmpeg_path
from src.frame_extractor import cleanup_frames, extract_frames
from src.scoring import FinalScore, combine_scores
from src.transcription import Segment
from src.utils import extract_json_from_text

log = logging.getLogger("video_clipper.visual")

# Light path only — never loads 6GB VLM unless explicitly requested
DEFAULT_MAX_FRAMES = 3
DEFAULT_FRAME_WIDTH = 320


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
    mode: str = "light"  # light | vlm


def _read_jpeg_luma_stats(path: Path) -> tuple[float, float]:
    """
    Approximate mean/variance of luminance without heavy deps.
    Uses FFmpeg to raw gray 64x64 then Python stats.
    """
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(path),
        "-vf",
        "scale=64:64,format=gray",
        "-f",
        "rawvideo",
        "-",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=15)
        data = r.stdout
        if not data:
            return 128.0, 0.0
        n = len(data)
        total = sum(data)
        mean = total / n
        var = sum((b - mean) ** 2 for b in data) / n
        return mean, var ** 0.5
    except Exception:
        return 128.0, 0.0


def _frame_diff_score(path_a: Path, path_b: Path) -> float:
    """0–100 motion-ish score between two frames."""
    ffmpeg = get_ffmpeg_path()
    # blend difference → mean signal
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(path_a),
        "-i",
        str(path_b),
        "-filter_complex",
        "[0:v][1:v]blend=all_mode=difference,scale=32:32,format=gray",
        "-f",
        "rawvideo",
        "-",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=15)
        data = r.stdout
        if not data:
            return 30.0
        mean = sum(data) / len(data)
        # map mean diff 0–40 → score-ish
        return max(0.0, min(100.0, mean * 3.5))
    except Exception:
        return 40.0


def analyze_clip_light(
    clip_path: Path,
    clip_duration: float,
    speech_score: float = 70.0,
) -> VisualAnalysis:
    """
    Lightweight analysis of the CLIP file only (3 small frames).
    No Ollama, no GPU neural net — safe for unstable systems.
    """
    t0 = time.time()
    frames = extract_frames(
        clip_path,
        duration=clip_duration,
        max_frames=DEFAULT_MAX_FRAMES,
        width=DEFAULT_FRAME_WIDTH,
    )
    problems: list[str] = []
    strengths: list[str] = []
    suggestions: list[str] = []

    if not frames:
        return VisualAnalysis(
            overall_score=int(speech_score),
            verdict="REVIEW",
            problems=["Não foi possível extrair frames do clipe."],
            confidence=0.3,
            mode="light",
            inference_ms=int((time.time() - t0) * 1000),
        )

    means = []
    stds = []
    for f in frames:
        m, s = _read_jpeg_luma_stats(f)
        means.append(m)
        stds.append(s)

    avg_bright = sum(means) / len(means)
    avg_std = sum(stds) / len(stds)

    # Quality heuristics
    quality = 75
    if avg_bright < 40:
        quality -= 20
        problems.append("Clipe escuro demais")
        suggestions.append("Preferir trecho mais iluminado")
    elif avg_bright > 220:
        quality -= 15
        problems.append("Clipe muito estourado/claro")
    else:
        strengths.append("Iluminação aceitável")

    if avg_std < 18:
        quality -= 15
        problems.append("Imagem com pouco detalhe (possível blur/plano limpo demais)")
    else:
        strengths.append("Contraste/detalhe razoável")

    # Motion between consecutive frames
    motion_scores = []
    for i in range(len(frames) - 1):
        motion_scores.append(_frame_diff_score(frames[i], frames[i + 1]))
    motion = sum(motion_scores) / len(motion_scores) if motion_scores else 40.0

    hook = 55
    if motion > 25:
        hook = min(90, 55 + motion * 0.4)
        strengths.append("Há variação visual entre os frames")
    else:
        hook = 45
        problems.append("Pouca movimentação visual nos primeiros frames")
        suggestions.append("Começar em um momento com mais ação ou expressão")

    retention = min(90, (hook + quality) / 2)
    composition = quality
    emotion = 60  # unknown without face model
    context = 70  # light mode doesn't match speech deeply
    short_form = min(90, (hook + retention + quality) / 3)

    visual = {
        "retention_score": retention,
        "visual_hook_score": hook,
        "visual_quality_score": quality,
        "composition_score": composition,
        "context_match_score": context,
    }
    final = combine_scores(speech_score, visual)

    cleanup_frames(frames)
    ms = int((time.time() - t0) * 1000)

    return VisualAnalysis(
        overall_score=final.overall,
        visual_hook_score=int(hook),
        retention_score=int(retention),
        composition_score=int(composition),
        emotion_score=emotion,
        visual_quality_score=int(quality),
        context_match_score=int(context),
        short_form_score=int(short_form),
        verdict=final.verdict,
        confidence=0.55,
        problems=problems[:5],
        strengths=strengths[:5],
        suggestions=suggestions[:5],
        final=final,
        frames_used=len(frames),
        inference_ms=ms,
        mode="light",
        raw="light-heuristic",
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
    use_vlm: bool = False,
) -> VisualAnalysis:
    """
    Always analyzes the CLIP only (clip_path), never the full source video.

    use_vlm=False (default): safe lightweight heuristics.
    use_vlm=True: optional Qwen — can stress GPU/drivers; not recommended if system is unstable.
    """
    if not use_vlm:
        return analyze_clip_light(clip_path, clip_duration, speech_score=speech_score)

    # ---- Optional heavy path (explicit opt-in only) ----
    from src.ollama_manager import DEFAULT_VISION_MODEL, ensure_optimized_model, is_ollama_running

    if not is_ollama_running():
        raise RuntimeError("Ollama não está em execução.")

    model_name = model or ensure_optimized_model()
    frames = extract_frames(
        clip_path,
        duration=clip_duration,
        max_frames=min(max_frames, 2),  # hard cap 2 frames for VLM
        width=256,
    )
    if not frames:
        raise RuntimeError("Não foi possível extrair frames do clipe.")

    # Minimal text
    user_text = f"Clipe {clip_duration:.0f}s. Responda só JSON com scores 0-100 e verdict APPROVE|REVIEW|REJECT."

    t0 = time.time()
    raw = ""
    try:
        import ollama

        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": user_text,
                    "images": [str(p) for p in frames],
                }
            ],
            options={
                "temperature": 0.1,
                "num_predict": 300,
                "num_ctx": 2048,
                "num_batch": 128,
            },
            format="json",
        )
        if isinstance(response, dict):
            raw = response.get("message", {}).get("content", "") or ""
        else:
            raw = getattr(getattr(response, "message", None), "content", "") or ""
    except Exception as exc:
        cleanup_frames(frames)
        raise RuntimeError(f"Falha na IA visual (VLM): {exc}") from exc

    ms = int((time.time() - t0) * 1000)
    data = extract_json_from_text(raw) or {}
    final = combine_scores(speech_score, data)
    cleanup_frames(frames)

    def gi(k: str, d: int = 70) -> int:
        try:
            return int(max(0, min(100, float(data.get(k, d)))))
        except (TypeError, ValueError):
            return d

    return VisualAnalysis(
        overall_score=final.overall,
        visual_hook_score=gi("visual_hook_score"),
        retention_score=gi("retention_score"),
        composition_score=gi("composition_score"),
        emotion_score=gi("emotion_score"),
        visual_quality_score=gi("visual_quality_score"),
        context_match_score=gi("context_match_score"),
        short_form_score=gi("short_form_score"),
        verdict=final.verdict,
        confidence=0.6,
        problems=list(data.get("problems") or [])[:5],
        strengths=list(data.get("strengths") or [])[:5],
        suggestions=list(data.get("suggestions") or [])[:5],
        final=final,
        raw=raw,
        frames_used=len(frames),
        inference_ms=ms,
        mode="vlm",
    )
