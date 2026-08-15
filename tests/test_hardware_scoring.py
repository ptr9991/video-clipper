"""Unit tests for hardware detection helpers and scoring (no real GPU/Ollama)."""

from src.frame_extractor import adaptive_timestamps
from src.scoring import combine_scores, verdict_from_score
from src.utils import extract_json_from_text


def test_verdict_bands():
    assert verdict_from_score(95) == "EXCELLENT"
    assert verdict_from_score(85) == "APPROVE"
    assert verdict_from_score(75) == "REVIEW"
    assert verdict_from_score(50) == "REJECT"


def test_combine_scores():
    visual = {
        "retention_score": 80,
        "visual_hook_score": 90,
        "visual_quality_score": 70,
        "composition_score": 70,
        "context_match_score": 85,
    }
    result = combine_scores(speech_score=90, visual=visual)
    assert 0 <= result.overall <= 100
    assert result.verdict in {"EXCELLENT", "APPROVE", "REVIEW", "REJECT"}
    assert "speech" in result.breakdown


def test_adaptive_timestamps_short():
    ts = adaptive_timestamps(4.0, max_frames=8)
    assert ts[0] == 0.0
    assert len(ts) <= 8
    assert all(t >= 0 for t in ts)


def test_adaptive_timestamps_long():
    ts = adaptive_timestamps(45.0, max_frames=12)
    assert ts[0] == 0.0
    assert len(ts) <= 12
    assert ts[-1] < 45.0


def test_json_extract_visual():
    raw = '```json\n{"overall_score": 87, "verdict": "APPROVE"}\n```'
    data = extract_json_from_text(raw)
    assert data is not None
    assert data["overall_score"] == 87
