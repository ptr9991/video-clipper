"""Platform publication copy — hashtag exact casing, no invented @."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.campaigns.models import CampaignProfile


@dataclass
class PlatformCopy:
    platform: str
    title: str
    description: str


def build_platform_copy(
    campaign: CampaignProfile,
    context: str = "",
    clip_hook: str = "",
) -> list[PlatformCopy]:
    tag = campaign.hashtag  # exact #dona30K
    handle = campaign.official_handle.strip() if campaign.handle_ok else ""
    twitch = campaign.twitch_url
    hook = (clip_hook or context or "Melhor momento").strip()[:120]

    mention = f" {handle}" if handle else ""
    base_body = f"{hook}\n\n{tag}{mention}\n{twitch}".strip()

    out: list[PlatformCopy] = []

    # TikTok — hashtag in title area / description
    out.append(
        PlatformCopy(
            platform="TikTok",
            title=f"{hook} {tag}".strip()[:150],
            description=f"{tag}{mention}\nAo vivo: {twitch}",
        )
    )

    # Instagram — no collab suggestion
    out.append(
        PlatformCopy(
            platform="Instagram Reels",
            title="",
            description=f"{hook}\n\n{tag}{mention}\n{twitch}",
        )
    )

    # YouTube Shorts
    out.append(
        PlatformCopy(
            platform="YouTube Shorts",
            title=f"{hook} {tag}".strip()[:100],
            description=f"{hook}\n\n{tag}{mention}\nTwitch: {twitch}\n",
        )
    )

    # Facebook
    out.append(
        PlatformCopy(
            platform="Facebook",
            title="",
            description=f"{hook}\n\n{tag}{mention}\n{twitch}",
        )
    )

    return out
