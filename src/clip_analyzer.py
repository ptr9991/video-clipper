"""Clip selection — single analysis pass returning TOP-N diverse candidates."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Optional

from groq import Groq
from groq import APIError, APIConnectionError, RateLimitError, AuthenticationError

from src.config import ANALYSIS_MODEL, MAX_CLIP_DURATION, MIN_CLIP_DURATION, require_api_key
from src.transcription import Segment, TranscriptionResult
from src.utils import extract_json_from_text, validate_timestamps

log = logging.getLogger("video_clipper.analyzer")

SYSTEM_MULTI = """Você é um editor de vídeos curtos (TikTok/Reels/Shorts).
Analise a transcrição com timestamps e escolha os MELHORES trechos CONTINUOS de 30 a 50 segundos.

Regras:
- Retorne entre 5 e 15 candidatos (conforme pedido).
- Cada trecho: 30–50s (ideal 35–45).
- Diversifique: NÃO escolha trechos que se sobreponham fortemente.
- Priorize: hook forte, clareza, payoff, emoção, autonomia do trecho.
- score 0–100.

Responda SOMENTE JSON:
{"candidates":[{"start":12.5,"end":52.0,"duration":39.5,"reason":"...","hook":"...","score":90,"title_hint":"..."}]}
"""


@dataclass
class ClipCandidate:
    start: float
    end: float
    duration: float
    reason: str
    hook: str
    score: int
    title_hint: str = ""
    transcript_snip: str = ""
    raw_response: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "reason": self.reason,
            "hook": self.hook,
            "score": self.score,
            "title_hint": self.title_hint,
            "transcript_snip": self.transcript_snip,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ClipCandidate":
        return cls(
            start=float(d["start"]),
            end=float(d["end"]),
            duration=float(d.get("duration", float(d["end"]) - float(d["start"]))),
            reason=str(d.get("reason", "")),
            hook=str(d.get("hook", "")),
            score=int(d.get("score", 70)),
            title_hint=str(d.get("title_hint", "")),
            transcript_snip=str(d.get("transcript_snip", "")),
        )


def _build_segments_text(segments: list[Segment]) -> str:
    return "\n".join(f"[{seg.start:.2f}s - {seg.end:.2f}s] {seg.text}" for seg in segments)


def _snip(segments: list[Segment], start: float, end: float, max_chars: int = 180) -> str:
    parts = []
    for seg in segments:
        if seg.end < start or seg.start > end:
            continue
        parts.append(seg.text.strip())
    t = " ".join(parts)
    return (t[:max_chars] + "…") if len(t) > max_chars else t


def _validate_one(data: dict[str, Any], video_duration: float) -> Optional[ClipCandidate]:
    try:
        start = float(data["start"])
        end = float(data["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if end <= start:
        return None
    start, end = validate_timestamps(
        start, end, video_duration=video_duration,
        max_duration=MAX_CLIP_DURATION, min_duration=1.0,
    )
    duration = end - start
    try:
        score = int(float(data.get("score", 70)))
    except (TypeError, ValueError):
        score = 70
    score = max(0, min(100, score))
    return ClipCandidate(
        start=start,
        end=end,
        duration=duration,
        reason=str(data.get("reason", "")),
        hook=str(data.get("hook", "")),
        score=score,
        title_hint=str(data.get("title_hint", "")),
    )


def _overlap_ratio(a: ClipCandidate, b: ClipCandidate) -> float:
    left = max(a.start, b.start)
    right = min(a.end, b.end)
    inter = max(0.0, right - left)
    union = max(a.duration + b.duration - inter, 0.01)
    return inter / union


def diversify(candidates: list[ClipCandidate], max_n: int, max_overlap: float = 0.35) -> list[ClipCandidate]:
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    picked: list[ClipCandidate] = []
    for c in ranked:
        if any(_overlap_ratio(c, p) > max_overlap for p in picked):
            continue
        picked.append(c)
        if len(picked) >= max_n:
            break
    return picked


def _density_multi(
    transcription: TranscriptionResult,
    video_duration: float,
    n: int,
    target: float = 40.0,
) -> list[ClipCandidate]:
    segments = transcription.segments
    if not segments:
        mid = max(0.0, video_duration / 2 - target / 2)
        end = min(video_duration, mid + target)
        return [ClipCandidate(mid, end, end - mid, "Janela central", "", 50)]

    windows: list[ClipCandidate] = []
    step = max(8.0, target * 0.4)
    t = 0.0
    while t + MIN_CLIP_DURATION <= video_duration:
        w_end = min(video_duration, t + target)
        words = 0
        for seg in segments:
            if seg.end < t or seg.start > w_end:
                continue
            words += len(seg.text.split())
        dens = words / max(w_end - t, 1.0)
        score = int(min(95, 40 + dens * 8))
        windows.append(
            ClipCandidate(
                start=t, end=w_end, duration=w_end - t,
                reason="Densidade de fala", hook="", score=score,
                transcript_snip=_snip(segments, t, w_end),
            )
        )
        t += step

    return diversify(windows, n)


def analyze_best_clips(
    transcription: TranscriptionResult,
    video_duration: float,
    n: int = 5,
    model: str = ANALYSIS_MODEL,
    prefer_local: bool = False,
) -> list[ClipCandidate]:
    """One analysis pass → up to N diverse candidates."""
    n = max(1, min(15, int(n)))
    if not transcription.segments and not transcription.text:
        raise ValueError("Transcrição vazia.")
    if video_duration <= 0:
        raise ValueError("Duração inválida.")

    if prefer_local:
        from src.local_ai.clip_selector import analyze_best_clips_local

        return analyze_best_clips_local(transcription, video_duration, n=n)

    content = _build_segments_text(transcription.segments) if transcription.segments else transcription.text
    if len(content) > 60000:
        content = content[:60000] + "\n...[truncado]..."

    user = (
        f"Duração do vídeo: {video_duration:.1f}s\n"
        f"Retorne até {n} candidatos diversos (30–50s).\n\n"
        f"{content}\n\nJSON only."
    )

    def _parse(raw: str) -> list[ClipCandidate]:
        data = extract_json_from_text(raw)
        if not data:
            return []
        items = data.get("candidates") if isinstance(data, dict) else None
        if not isinstance(items, list):
            # single candidate fallback
            one = _validate_one(data if isinstance(data, dict) else {}, video_duration)
            return [one] if one else []
        out: list[ClipCandidate] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            c = _validate_one(it, video_duration)
            if c:
                c.transcript_snip = _snip(transcription.segments, c.start, c.end)
                out.append(c)
        return diversify(out, n)

    try:
        client = Groq(api_key=require_api_key())
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_MULTI},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
        }
        try:
            raw = client.chat.completions.create(**kwargs).choices[0].message.content or ""
        except APIError:
            kwargs.pop("response_format", None)
            raw = client.chat.completions.create(**kwargs).choices[0].message.content or ""

        cands = _parse(raw)
        if not cands:
            log.warning("Empty LLM candidates — density fallback")
            return _density_multi(transcription, video_duration, n)
        return cands

    except (RateLimitError, APIConnectionError, APIError) as exc:
        log.warning("Groq failed (%s) — local/density", exc)
        try:
            from src.local_ai.clip_selector import analyze_best_clips_local

            return analyze_best_clips_local(transcription, video_duration, n=n)
        except Exception:
            return _density_multi(transcription, video_duration, n)
    except AuthenticationError as exc:
        raise RuntimeError("Chave Groq inválida.") from exc


def analyze_best_clip(
    transcription: TranscriptionResult,
    video_duration: float,
    model: str = ANALYSIS_MODEL,
    prefer_local: bool = False,
) -> ClipCandidate:
    """Backward compatible: return best single candidate."""
    cands = analyze_best_clips(
        transcription, video_duration, n=1, model=model, prefer_local=prefer_local
    )
    return cands[0]
