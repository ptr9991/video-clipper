"""Configurable scoring weights and verdict thresholds for hybrid clip evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Weights for FINAL score (speech from Groq + visual from local AI)
CLIP_SCORE_WEIGHTS = {
    "speech": 0.45,
    "visual_retention": 0.25,
    "visual_quality": 0.15,
    "context_match": 0.10,
    "technical_quality": 0.05,
}

# Verdict bands
THRESHOLDS = {
    "excellent": 90,
    "approve": 80,
    "review": 70,
}


@dataclass
class FinalScore:
    overall: int
    verdict: str  # EXCELLENT | APPROVE | REVIEW | REJECT
    breakdown: dict[str, float]


def verdict_from_score(score: int) -> str:
    if score >= THRESHOLDS["excellent"]:
        return "EXCELLENT"
    if score >= THRESHOLDS["approve"]:
        return "APPROVE"
    if score >= THRESHOLDS["review"]:
        return "REVIEW"
    return "REJECT"


def combine_scores(
    speech_score: float,
    visual: dict[str, Any],
) -> FinalScore:
    """
    Combine Groq textual score with local visual metrics.
    `visual` is the parsed JSON from the vision model.
    """
    def g(key: str, default: float = 70.0) -> float:
        v = visual.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    retention = g("retention_score", g("visual_hook_score", 70))
    quality = g("visual_quality_score", g("composition_score", 70))
    context = g("context_match_score", 70)
    technical = g("visual_quality_score", 70)

    breakdown = {
        "speech": float(speech_score),
        "visual_retention": retention,
        "visual_quality": quality,
        "context_match": context,
        "technical_quality": technical,
    }

    overall = 0.0
    for k, w in CLIP_SCORE_WEIGHTS.items():
        overall += breakdown.get(k, 70.0) * w
    overall_i = int(round(max(0, min(100, overall))))

    return FinalScore(
        overall=overall_i,
        verdict=verdict_from_score(overall_i),
        breakdown=breakdown,
    )
