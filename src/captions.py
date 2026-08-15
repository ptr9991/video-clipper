"""
Short-form caption pipeline: word-level timing, chunking, relative offset, safe zone.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.editor.models import CaptionCue, CaptionStyle


@dataclass
class WordStamp:
    word: str
    start: float
    end: float
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WordStamp":
        return cls(
            word=str(d.get("word", "")),
            start=float(d.get("start", 0)),
            end=float(d.get("end", 0)),
            confidence=float(d.get("confidence", 1.0)),
        )


# 1080x1920 safe margins (do not burn into export)
SAFE_ZONE = {
    "tiktok": {"top": 0.12, "bottom": 0.22, "left": 0.06, "right": 0.18},
    "instagram_reels": {"top": 0.10, "bottom": 0.20, "left": 0.06, "right": 0.14},
    "youtube_shorts": {"top": 0.10, "bottom": 0.18, "left": 0.06, "right": 0.12},
}

DEFAULT_SHORTS = CaptionStyle(
    name="default_shorts",
    font_size=64,
    primary_color="#FFFFFF",
    highlight_color="#C8F542",
    outline=4,
    margin_v=280,  # ASS bottom margin — above platform UI (~15% of 1920)
)


def shift_words_to_clip(
    words: list[WordStamp],
    clip_start_abs: float,
    clip_duration: float,
) -> list[WordStamp]:
    """
    Convert absolute video times → relative to clip [0, duration].

    Example: clip_start=42.35, word at 43.12 → 0.77
    """
    out: list[WordStamp] = []
    for w in words:
        if w.end < clip_start_abs or w.start > clip_start_abs + clip_duration:
            continue
        rs = max(0.0, w.start - clip_start_abs)
        re_ = min(clip_duration, max(rs + 0.02, w.end - clip_start_abs))
        if re_ <= rs:
            continue
        text = (w.word or "").strip()
        if not text:
            continue
        out.append(WordStamp(word=text, start=rs, end=re_, confidence=w.confidence))
    return out


def _should_break(prev: str, nxt: str) -> bool:
    if not prev:
        return False
    if prev[-1] in ".!?…":
        return True
    if prev[-1] in ",;:" and len(prev) > 12:
        return True
    return False


def chunk_words(
    words: list[WordStamp],
    min_words: int = 2,
    max_words: int = 6,
    max_gap: float = 0.55,
    max_chars: int = 36,
) -> list[CaptionCue]:
    """Group words into readable 1–2 line caption blocks."""
    if not words:
        return []

    cues: list[CaptionCue] = []
    buf: list[WordStamp] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        text = " ".join(w.word for w in buf).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            buf = []
            return
        start = buf[0].start
        end = max(buf[-1].end, start + 0.35)
        # min on-screen time for readability
        if end - start < 0.4:
            end = start + 0.4
        cues.append(
            CaptionCue(
                id=f"cap_{uuid.uuid4().hex[:8]}",
                start=start,
                end=end,
                text=text.upper(),
                highlight_words=[],
            )
        )
        buf = []

    for w in words:
        if not buf:
            buf = [w]
            continue
        gap = w.start - buf[-1].end
        joined = " ".join(x.word for x in buf) + " " + w.word
        if (
            gap > max_gap
            or len(buf) >= max_words
            or len(joined) > max_chars
            or _should_break(buf[-1].word, w.word)
        ):
            if len(buf) >= min_words or gap > max_gap:
                flush()
                buf = [w]
            else:
                buf.append(w)
        else:
            buf.append(w)
    flush()
    return _fix_overlaps(cues)


def _fix_overlaps(cues: list[CaptionCue]) -> list[CaptionCue]:
    if not cues:
        return []
    fixed = [cues[0]]
    for c in cues[1:]:
        prev = fixed[-1]
        if c.start < prev.end:
            # small gap between blocks
            mid = (prev.end + c.start) / 2 if c.start < prev.end else c.start
            prev.end = max(prev.start + 0.2, min(prev.end, mid))
            c.start = max(c.start, prev.end + 0.02)
            if c.end <= c.start:
                c.end = c.start + 0.35
        if c.end <= c.start:
            continue
        fixed.append(c)
    return fixed


def captions_from_words(
    words: list[WordStamp],
    clip_start_abs: float,
    clip_duration: float,
) -> list[CaptionCue]:
    rel = shift_words_to_clip(words, clip_start_abs, clip_duration)
    return chunk_words(rel)


def captions_from_segments_fallback(
    segments: list,
    clip_start_abs: float,
    clip_duration: float,
) -> list[CaptionCue]:
    """When word timestamps unavailable, split segment text evenly in time."""
    words: list[WordStamp] = []
    for seg in segments:
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        parts = text.split()
        if not parts:
            continue
        s = float(getattr(seg, "start", 0))
        e = float(getattr(seg, "end", s + 1))
        dur = max(e - s, 0.05)
        step = dur / len(parts)
        for i, p in enumerate(parts):
            words.append(WordStamp(word=p, start=s + i * step, end=s + (i + 1) * step))
    return captions_from_words(words, clip_start_abs, clip_duration)
