"""Critical sync tests: absolute → relative word timestamps."""

from src.captions import WordStamp, captions_from_words, chunk_words, shift_words_to_clip


def test_word_relative_offset_critical():
    """clip starts 42.350, word at 43.120 → relative 0.770"""
    words = [WordStamp(word="exemplo", start=43.120, end=43.500)]
    rel = shift_words_to_clip(words, clip_start_abs=42.350, clip_duration=40.0)
    assert len(rel) == 1
    assert abs(rel[0].start - 0.770) < 1e-6
    assert abs(rel[0].end - 1.150) < 1e-6


def test_word_outside_clip_dropped():
    words = [
        WordStamp("a", 10.0, 10.5),
        WordStamp("b", 42.5, 43.0),
        WordStamp("c", 100.0, 101.0),
    ]
    rel = shift_words_to_clip(words, 42.0, 5.0)
    assert len(rel) == 1
    assert rel[0].word == "b"
    assert abs(rel[0].start - 0.5) < 1e-6


def test_chunk_readable():
    words = [
        WordStamp(w, i * 0.3, i * 0.3 + 0.25)
        for i, w in enumerate("eu fui pra sao paulo ontem de manha cedo".split())
    ]
    cues = chunk_words(words, max_words=4)
    assert len(cues) >= 2
    assert all(c.end > c.start for c in cues)
    assert all(c.text == c.text.upper() for c in cues)


def test_captions_from_words_pipeline():
    words = [
        WordStamp("mano", 42.35, 42.55),
        WordStamp("isso", 42.60, 42.90),
        WordStamp("e", 43.00, 43.10),
        WordStamp("absurdo", 43.15, 43.60),
    ]
    cues = captions_from_words(words, clip_start_abs=42.35, clip_duration=30.0)
    assert cues
    assert cues[0].start >= 0.0
    assert cues[0].start < 0.5
