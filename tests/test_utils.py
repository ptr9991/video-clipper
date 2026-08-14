"""Unit tests for utils."""

import pytest

from src.utils import (
    extract_json_from_text,
    format_timestamp,
    validate_timestamps,
)


def test_format_timestamp():
    assert format_timestamp(0) == "00:00.000"
    assert format_timestamp(65.5) == "01:05.500"
    assert format_timestamp(3661.25).startswith("01:01:01")


def test_validate_timestamps_ok():
    s, e = validate_timestamps(10.0, 45.0, video_duration=120.0)
    assert s == 10.0
    assert e == 45.0


def test_validate_timestamps_clamp_max_duration():
    s, e = validate_timestamps(10.0, 80.0, video_duration=200.0, max_duration=50.0)
    assert e - s == 50.0
    assert s == 10.0


def test_validate_timestamps_end_beyond_video():
    s, e = validate_timestamps(90.0, 120.0, video_duration=100.0, max_duration=50.0)
    assert e == 100.0
    assert s >= 50.0  # adjusted


def test_validate_timestamps_invalid():
    with pytest.raises(ValueError):
        validate_timestamps(50.0, 40.0, video_duration=100.0)


def test_extract_json_plain():
    data = extract_json_from_text('{"start": 1.0, "end": 40.0}')
    assert data["start"] == 1.0


def test_extract_json_markdown():
    text = 'Here is the result:\n```json\n{"start": 5, "end": 45, "score": 90}\n```\nDone.'
    data = extract_json_from_text(text)
    assert data["score"] == 90


def test_extract_json_fail():
    assert extract_json_from_text("no json here") is None
