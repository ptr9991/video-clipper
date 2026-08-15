"""Deterministic campaign export validation."""

from __future__ import annotations

from pathlib import Path

from src.campaigns.apply import project_has_twitch_cta
from src.campaigns.models import CampaignProfile, ValidationResult
from src.editor.models import AspectRatio, ProjectState
from src.preset import CANVAS_H, CANVAS_W


def validate_campaign_export(
    state: ProjectState,
    campaign: CampaignProfile,
    output_path: Path | None = None,
) -> ValidationResult:
    checks: list[tuple[str, bool, str]] = []

    # 9:16
    is_vertical = state.aspect == AspectRatio.VERTICAL_9_16
    checks.append((
        "aspect",
        is_vertical,
        "9:16 OK" if is_vertical else "Formato deve ser 9:16",
    ))

    checks.append((
        "canvas",
        True,
        f"Export padrao {CANVAS_W}x{CANVAS_H}",
    ))

    # Twitch CTA in project timeline
    has_cta = project_has_twitch_cta(state, campaign)
    checks.append((
        "twitch_cta",
        has_cta,
        "CTA Twitch no video OK" if has_cta else "CTA Twitch.tv/dona ausente no video",
    ))

    # Twitch URL configured
    tw = bool(campaign.twitch_url and "twitch.tv" in campaign.twitch_url.lower())
    checks.append((
        "twitch_url",
        tw,
        "Twitch URL OK" if tw else "URL Twitch nao configurada",
    ))

    # Hashtag exact
    tag_ok = bool(campaign.hashtag) and campaign.hashtag.startswith("#")
    checks.append((
        "hashtag",
        tag_ok,
        f"Hashtag {campaign.hashtag}" if tag_ok else "Hashtag ausente",
    ))

    # Official handle — must be set by user, never invented
    handle_ok = campaign.handle_ok
    checks.append((
        "official_handle",
        handle_ok,
        f"@ oficial: {campaign.official_handle}" if handle_ok else "@ oficial do Dona NAO configurado",
    ))

    # Captions recommended
    has_caps = len(state.captions) > 0
    checks.append((
        "captions",
        has_caps,
        f"{len(state.captions)} legendas" if has_caps else "Sem legendas (opcional, recomendado)",
    ))
    # captions not blocking
    if not has_caps:
        checks[-1] = ("captions", True, "Sem legendas (permitido)")

    duration_ok = state.timeline_duration >= 5.0
    checks.append((
        "duration",
        duration_ok,
        f"Duracao {state.timeline_duration:.1f}s" if duration_ok else "Corte muito curto (<5s)",
    ))

    if output_path is not None and output_path.exists():
        size_ok = output_path.stat().st_size > 10_000
        checks.append((
            "file",
            size_ok,
            "Arquivo exportado OK" if size_ok else "Arquivo exportado invalido",
        ))

    # Blocking checks: aspect, cta, twitch_url, hashtag, handle, duration
    blocking = {"aspect", "twitch_cta", "twitch_url", "hashtag", "official_handle", "duration"}
    ok = all(passed for name, passed, _ in checks if name in blocking)
    return ValidationResult(ok=ok, checks=checks)
