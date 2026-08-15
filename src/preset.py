"""
VIDEOCLIPPER DEFAULT — single official visual standard for all exports.
Deterministic. No style picker. No random sizing.
"""

from __future__ import annotations

from dataclasses import dataclass

# Canvas
CANVAS_W = 1080
CANVAS_H = 1920
ASPECT = "9:16"

# Caption geometry (percent of canvas)
CAPTION_MAX_WIDTH_RATIO = 0.82  # never touch edges
CAPTION_BOTTOM_RATIO = 0.18     # center of text block from bottom (~safe)
# ASS MarginV ≈ pixels from bottom
CAPTION_MARGIN_V = int(CANVAS_H * CAPTION_BOTTOM_RATIO)  # ~346 → clamp usable
# Practical ASS margin so text sits above TikTok/IG UI (~18–22% from bottom)
CAPTION_MARGIN_V = 260

# Font — ASS FontSize is NOT CSS px; keep moderate for 1080x1920
CAPTION_FONT_SIZE = 42
CAPTION_FONT_NAME = "Arial"
CAPTION_PRIMARY = "#FFFFFF"
CAPTION_OUTLINE = 3
CAPTION_BOLD = True

# Chunking
MIN_WORDS = 2
MAX_WORDS = 5
MAX_CHARS_LINE = 28
MAX_LINES = 2
MIN_CUE_DURATION = 0.45
MAX_GAP_BREAK = 0.50


def calculate_caption_size(canvas_w: int = CANVAS_W, canvas_h: int = CANVAS_H) -> int:
    """
    Stable ASS font size from canvas width.
    Tuned so captions are readable on phone without dominating the frame.
    """
    # ~3.9% of width → ~42 at 1080
    size = int(round(canvas_w * 0.039))
    return max(32, min(48, size))


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
    """FFmpeg subtitles force_style string — identical for every export."""
    p = p or DEFAULT
    # ASS colour: &HAABBGGRR white opaque
    return (
        f"FontName={p.font_name},"
        f"FontSize={p.font_size},"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"BorderStyle=1,"
        f"Outline={p.outline},"
        f"Shadow=1,"
        f"Bold=1,"
        f"Alignment=2,"  # bottom center
        f"MarginV={p.margin_v},"
        f"MarginL=60,"
        f"MarginR=60"
    )


def balance_two_lines(words: list[str], max_chars: int = MAX_CHARS_LINE) -> str:
    """
    Join words into at most 2 lines with ASS \\N, balanced lengths.
    """
    if not words:
        return ""
    text = " ".join(words)
    if len(text) <= max_chars:
        return text.upper()

    # find split near midpoint by word count
    best_i = max(1, len(words) // 2)
    best_score = 10**9
    for i in range(1, len(words)):
        left = " ".join(words[:i])
        right = " ".join(words[i:])
        if len(left) > max_chars * 1.15 or len(right) > max_chars * 1.15:
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
