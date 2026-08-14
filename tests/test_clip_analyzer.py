"""Unit tests for clip_analyzer (no real API calls)."""

import pytest

from src.clip_analyzer import parse_and_validate_json


def test_parse_valid_json():
    raw = '{"start": 12.5, "end": 52.0, "duration": 39.5, "reason": "hook forte", "hook": "afirmação", "score": 88}'
    cand = parse_and_validate_json(raw, video_duration=180.0)
    assert cand.start == 12.5
    assert cand.end == 52.0
    assert cand.duration == 39.5
    assert cand.score == 88


def test_parse_markdown_wrapped():
    raw = """Aqui está o melhor trecho:
```json
{
  "start": 100.0,
  "end": 145.0,
  "reason": "revelação",
  "hook": "surpresa",
  "score": 95
}
```
"""
    cand = parse_and_validate_json(raw, video_duration=300.0)
    assert cand.start == 100.0
    assert abs(cand.duration - 45.0) < 0.01


def test_duration_clamped_to_50():
    raw = '{"start": 0, "end": 80, "reason": "longo", "score": 70}'
    cand = parse_and_validate_json(raw, video_duration=200.0)
    assert cand.duration <= 50.0


def test_end_beyond_video():
    raw = '{"start": 90, "end": 140, "reason": "fim", "score": 60}'
    cand = parse_and_validate_json(raw, video_duration=100.0)
    assert cand.end <= 100.0
    assert cand.start < cand.end


def test_invalid_json_raises():
    with pytest.raises(ValueError):
        parse_and_validate_json("isto não é json", video_duration=100.0)


def test_end_before_start_raises():
    with pytest.raises(ValueError):
        parse_and_validate_json(
            '{"start": 50, "end": 30, "reason": "inv", "score": 10}',
            video_duration=100.0,
        )
