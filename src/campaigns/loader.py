"""Load campaign JSON profiles from campaigns/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.campaigns.models import CampaignCTA, CampaignProfile

# Resolve campaigns dir relative to repo root or install root
def _campaigns_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parents[2], here.parents[1], Path.cwd()]:
        cand = parent / "campaigns"
        if cand.is_dir():
            return cand
    # fallback next to package
    return here.parents[1].parent / "campaigns"


def list_campaigns() -> list[str]:
    d = _campaigns_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def load_campaign(campaign_id: str = "dona30k") -> CampaignProfile:
    path = _campaigns_dir() / f"{campaign_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Campanha nao encontrada: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    cta_raw = data.get("cta") or {}
    cta = CampaignCTA(
        text=str(cta_raw.get("text") or data.get("twitch_display") or "Twitch.tv/dona"),
        position=str(cta_raw.get("position") or "top_left"),
        x=float(cta_raw.get("x", 0.08)),
        y=float(cta_raw.get("y", 0.06)),
        font_size=int(cta_raw.get("font_size", 28)),
        color=str(cta_raw.get("color") or "#FFFFFF"),
        full_duration=bool(cta_raw.get("full_duration", True)),
    )
    canvas = data.get("canvas") or {}
    return CampaignProfile(
        campaign_id=str(data.get("campaign_id") or campaign_id),
        name=str(data.get("name") or campaign_id),
        hashtag=str(data.get("hashtag") or ""),  # exact casing
        official_handle=str(data.get("official_handle") or ""),
        twitch_url=str(data.get("twitch_url") or ""),
        twitch_display=str(data.get("twitch_display") or cta.text),
        start_date=str(data.get("start_date") or ""),
        end_date=str(data.get("end_date") or ""),
        ranking_date=str(data.get("ranking_date") or ""),
        platforms=list(data.get("platforms") or []),
        required_in_video=list(data.get("required_in_video") or []),
        required_in_title=list(data.get("required_in_title") or []),
        required_in_caption=list(data.get("required_in_caption") or []),
        canvas_w=int(canvas.get("width") or 1080),
        canvas_h=int(canvas.get("height") or 1920),
        cta=cta,
        prizes=list(data.get("prizes") or []),
        rules_required=list(data.get("rules_required") or []),
        rules_forbidden=list(data.get("rules_forbidden") or []),
        warnings=list(data.get("warnings") or []),
        raw=data,
    )
