"""Caption pipeline — VideoClipper Default only. Word-level, relative, max 2 lines."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

from src.preset import DEFAULT, balance_two_lines


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


def get_default_shorts() -> Any:
    from src.editor.models import CaptionStyle

    return CaptionStyle(
        name="videoclipper_default",
        font_size=DEFAULT.font_size,
        primary_color=DEFAULT.primary_color,
        highlight_color="#C8F542",
        outline=DEFAULT.outline,
        margin_v=DEFAULT.margin_v,
    )


def shift_words_to_clip(
    words: list[WordStamp],
    clip_start_abs: float,
    clip_duration: float,
) -> list[WordStamp]:
    out: list[WordStamp] = []
    for w in words:
        if w.end < clip_start_abs or w.start > clip_start_abs + clip_duration:
            continue
        rs = max(0.0, w.start - clip_start_abs)
        re_ = min(clip_duration, max(rs + 0.02, w.end - clip_start_abs))
        if re_ <= rs:
            continue
        text = re.sub(r"\s+", "", (w.word or "")).strip() or (w.word or "").strip()
        # keep internal spaces for multi-token rare cases
        text = (w.word or "").strip()
        if not text:
            continue
        out.append(WordStamp(word=text, start=rs, end=re_, confidence=w.confidence))
    return out


def _clean_word(w: str) -> str:
    return re.sub(r"\s+", " ", (w or "")).strip()


def chunk_words(
    words: list[WordStamp],
    min_words: int | None = None,
    max_words: int | None = None,
    max_gap: float | None = None,
    max_chars: int | None = None,
) -> list:
    from src.editor.models import CaptionCue

    min_words = min_words if min_words is not None else DEFAULT.min_words
    max_words = max_words if max_words is not None else DEFAULT.max_words
    max_gap = max_gap if max_gap is not None else 0.50
    max_chars = max_chars if max_chars is not None else DEFAULT.max_chars_line * 2

    if not words:
        return []

    cues: list = []
    buf: list[WordStamp] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        toks = [_clean_word(w.word) for w in buf if _clean_word(w.word)]
        if not toks:
            buf = []
            return
        text = balance_two_lines(toks, max_chars=DEFAULT.max_chars_line)
        start = buf[0].start
        end = max(buf[-1].end, start + DEFAULT.min_words * 0.15)
        # min readable duration
        if end - start < 0.45:
            end = start + 0.45
        cues.append(
            CaptionCue(
                id=f"cap_{uuid.uuid4().hex[:8]}",
                start=start,
                end=end,
                text=text,
                highlight_words=[],
            )
        )
        buf = []

    for w in words:
        if not buf:
            buf = [w]
            continue
        gap = w.start - buf[-1].end
        joined_len = sum(len(_clean_word(x.word)) + 1 for x in buf) + len(_clean_word(w.word))
        punct_break = _clean_word(buf[-1].word)[-1:] in ".!?…"
        if (
            gap > max_gap
            or len(buf) >= max_words
            or joined_len > max_chars
            or punct_break
        ):
            if len(buf) >= min_words or gap > max_gap or punct_break:
                flush()
                buf = [w]
            else:
                buf.append(w)
        else:
            buf.append(w)
    flush()
    return _fix_overlaps(cues)


def _fix_overlaps(cues: list) -> list:
    if not cues:
        return []
    fixed = [cues[0]]
    for c in cues[1:]:
        prev = fixed[-1]
        if c.start < prev.end:
            mid = (prev.end + c.start) / 2
            prev.end = max(prev.start + 0.25, min(prev.end, mid))
            c.start = max(c.start, prev.end + 0.02)
            if c.end <= c.start:
                c.end = c.start + 0.45
        if c.end <= c.start:
            continue
        fixed.append(c)
    return fixed


def captions_from_words(
    words: list[WordStamp],
    clip_start_abs: float,
    clip_duration: float,
) -> list:
    rel = shift_words_to_clip(words, clip_start_abs, clip_duration)
    return chunk_words(rel)


def captions_from_segments_fallback(
    segments: list,
    clip_start_abs: float,
    clip_duration: float,
) -> list:
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
