"""Built-in caption style presets (original, not copied from CapCut/Cut.Pro)."""

from __future__ import annotations

from src.editor.models import CaptionStyle

STYLES: dict[str, CaptionStyle] = {
    "clean": CaptionStyle(
        name="clean", font_size=42, primary_color="#FFFFFF",
        highlight_color="#C8F542", outline=2, margin_v=90,
    ),
    "bold": CaptionStyle(
        name="bold", font_size=52, primary_color="#FFFFFF",
        highlight_color="#FFDD00", outline=4, margin_v=100,
    ),
    "highlight": CaptionStyle(
        name="highlight", font_size=46, primary_color="#FFFFFF",
        highlight_color="#C8F542", outline=3, margin_v=95,
    ),
    "karaoke": CaptionStyle(
        name="karaoke", font_size=48, primary_color="#AAAAAA",
        highlight_color="#FFFFFF", outline=3, margin_v=100,
    ),
    "minimal": CaptionStyle(
        name="minimal", font_size=36, primary_color="#F0F0F0",
        highlight_color="#FFFFFF", outline=1, margin_v=70,
    ),
}


def get_style(name: str) -> CaptionStyle:
    return STYLES.get(name, STYLES["clean"]).to_dict() and CaptionStyle(**STYLES.get(name, STYLES["clean"]).to_dict())  # type: ignore


def style_by_name(name: str) -> CaptionStyle:
    base = STYLES.get((name or "clean").lower(), STYLES["clean"])
    return CaptionStyle(
        name=base.name,
        font_size=base.font_size,
        primary_color=base.primary_color,
        highlight_color=base.highlight_color,
        outline=base.outline,
        margin_v=base.margin_v,
    )
