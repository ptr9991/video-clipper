"""Campaign data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CampaignCTA:
    text: str = "Twitch.tv/dona"
    position: str = "top_left"
    x: float = 0.08
    y: float = 0.06
    font_size: int = 28
    color: str = "#FFFFFF"
    full_duration: bool = True


@dataclass
class CampaignProfile:
    campaign_id: str
    name: str
    hashtag: str  # preserve exact casing e.g. #dona30K
    official_handle: str = ""
    twitch_url: str = "https://www.twitch.tv/dona"
    twitch_display: str = "Twitch.tv/dona"
    start_date: str = ""
    end_date: str = ""
    ranking_date: str = ""
    platforms: list[str] = field(default_factory=list)
    required_in_video: list[str] = field(default_factory=list)
    required_in_title: list[str] = field(default_factory=list)
    required_in_caption: list[str] = field(default_factory=list)
    canvas_w: int = 1080
    canvas_h: int = 1920
    cta: CampaignCTA = field(default_factory=CampaignCTA)
    prizes: list[dict] = field(default_factory=list)
    rules_required: list[str] = field(default_factory=list)
    rules_forbidden: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def handle_ok(self) -> bool:
        h = (self.official_handle or "").strip()
        return bool(h) and h.startswith("@") and len(h) > 1

    def with_handle(self, handle: str) -> "CampaignProfile":
        h = (handle or "").strip()
        if h and not h.startswith("@"):
            h = "@" + h
        self.official_handle = h
        return self


@dataclass
class ValidationResult:
    ok: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def failures(self) -> list[str]:
        return [msg for name, passed, msg in self.checks if not passed]
