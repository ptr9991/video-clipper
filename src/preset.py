"""
VIDEOCLIPPER DEFAULT — single standard for all shorts.
Captions: moderate size, bottom-center, max 2 lines.
"""

from __future__ import annotations

from dataclasses import dataclass

CANVAS_W = 1080
CANVAS_H = 1920
ASPECT = "9:16"

# ASS FontSize is large on 1080x1920 — keep SMALL
# 24–28 reads well on phone without covering the speaker
CAPTION_FONT_SIZE = 26
CAPTION_FONT_NAME = "Arial"
CAPTION_PRIMARY = "#FFFFFF"
CAPTION_OUTLINE = 2
# Bottom-center, above platform UI (~14% from bottom)
CAPTION_MARGIN_V = 200

MIN_WORDS = 2
MAX_WORDS = 4
MAX_CHARS_LINE = 22
MAX_LINES = 2
MIN_CUE_DURATION = 0.50


def calculate_caption_size(canvas_w: int = CANVAS_W, canvas_h: int = CANVAS_H) -> int:
    """~2.4% of width → 26 at 1080. Never huge."""
    size = int(round(canvas_w * 0.024))
    return max(22, min(28, size))


@dataclass(frozen=True)
class VideoClipperDefault:
    canvas_w: int = CANVAS_W
    canvas_h: int = CANVAS_H
    font_name: str = CAPTION_FONT_NAME
    font_size: int = CAPTION_FONT_SIZE
    primary_color: str = CAPTION_PRIMARY
    outline: int = CAPTION_OUTLINE
    margin_v: int = CAPTION_MARGIN_V
    max_words: int = MAX_WORDS
    min_words: int = MIN_WORDS
    max_chars_line: int = MAX_CHARS_LINE
    max_lines: int = MAX_LINES

    @classmethod
    def create(cls) -> "VideoClipperDefault":
        return cls(font_size=calculate_caption_size())


DEFAULT = VideoClipperDefault.create()


def ass_force_style(p: VideoClipperDefault | None = None) -> str:
    """Bottom-center, compact, high contrast — same every export."""
    p = p or DEFAULT
    return (
        f"FontName={p.font_name},"
        f"FontSize={p.font_size},"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"BackColour=&H80000000,"
        f"BorderStyle=1,"
        f"Outline={p.outline},"
        f"Shadow=0,"
        f"Bold=1,"
        f"Alignment=2,"  # bottom-center
        f"MarginV={p.margin_v},"
        f"MarginL=80,"
        f"MarginR=80,"
        f"ScaleX=100,"
        f"ScaleY=100"
    )


def balance_two_lines(words: list[str], max_chars: int = MAX_CHARS_LINE) -> str:
    if not words:
        return ""
    text = " ".join(words)
    if len(text) <= max_chars:
        return text.upper()

    best_i = max(1, len(words) // 2)
    best_score = 10**9
    for i in range(1, len(words)):
        left = " ".join(words[:i])
        right = " ".join(words[i:])
        if len(left) > max_chars + 4 or len(right) > max_chars + 4:
            continue
        score = abs(len(left) - len(right))
        if score < best_score:
            best_score = score
            best_i = i

    left = " ".join(words[:best_i]).upper()
    right = " ".join(words[best_i:]).upper()
    if not right:
        return left
    return f"{left}\\N{right}"
