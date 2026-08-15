"""Apply campaign template to ProjectState (CTA overlay, 9:16)."""

from __future__ import annotations

import uuid

from src.campaigns.models import CampaignProfile
from src.editor.models import AspectRatio, ProjectState, TextOverlay
from src.preset import get_default_shorts if False else None  # type: ignore


def apply_campaign_to_project(state: ProjectState, campaign: CampaignProfile) -> ProjectState:
    """
    Clone project, force 9:16, ensure Twitch CTA text overlay for full duration.
    Does not invent @handle. Does not burn hashtag into video frames.
    """
    s = state.clone()
    s.aspect = AspectRatio.VERTICAL_9_16

    # Remove previous campaign CTA markers
    s.texts = [t for t in s.texts if not (t.id or "").startswith("cta_campaign_")]

    cta = campaign.cta
    end = s.timeline_duration if cta.full_duration else min(5.0, s.timeline_duration)
    s.texts.append(
        TextOverlay(
            id=f"cta_campaign_{uuid.uuid4().hex[:6]}",
            text=cta.text,
            start=0.0,
            end=max(0.5, end),
            x=float(cta.x),
            y=float(cta.y),
            font_size=int(cta.font_size),
            color=cta.color or "#FFFFFF",
        )
    )
    s.validate()
    return s


def project_has_twitch_cta(state: ProjectState, campaign: CampaignProfile) -> bool:
    needle = (campaign.cta.text or "twitch").lower()
    for t in state.texts:
        if needle in (t.text or "").lower() or "twitch" in (t.text or "").lower():
            if t.end > t.start and (t.end - t.start) >= min(1.0, state.timeline_duration * 0.5):
                return True
    return False
