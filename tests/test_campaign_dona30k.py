"""Tests for Dona 30K campaign profile."""

from pathlib import Path

from src.campaigns.apply import apply_campaign_to_project, project_has_twitch_cta
from src.campaigns.copygen import build_platform_copy
from src.campaigns.loader import load_campaign
from src.campaigns.validator import validate_campaign_export
from src.editor.models import AspectRatio, ProjectState, TimelineRange


def test_load_dona30k_hashtag_casing():
    c = load_campaign("dona30k")
    assert c.hashtag == "#dona30K"  # exact
    assert "twitch.tv/dona" in c.twitch_url


def test_handle_not_invented():
    c = load_campaign("dona30k")
    assert c.official_handle == ""
    assert not c.handle_ok
    c.with_handle("dona_oficial")
    assert c.official_handle == "@dona_oficial"
    assert c.handle_ok


def test_apply_cta_and_validate():
    c = load_campaign("dona30k")
    c.with_handle("@test_dona")
    proj = ProjectState(
        name="t",
        source_path="x.mp4",
        source_duration=20.0,
        playable_range=TimelineRange(0, 20),
        aspect=AspectRatio.VERTICAL_9_16,
    )
    proj = apply_campaign_to_project(proj, c)
    assert project_has_twitch_cta(proj, c)
    v = validate_campaign_export(proj, c)
    assert v.ok


def test_validate_fails_without_handle():
    c = load_campaign("dona30k")
    proj = ProjectState(
        name="t",
        source_path="x.mp4",
        source_duration=20.0,
        playable_range=TimelineRange(0, 20),
        aspect=AspectRatio.VERTICAL_9_16,
    )
    proj = apply_campaign_to_project(proj, c)
    v = validate_campaign_export(proj, c)
    assert not v.ok
    assert any("oficial" in f.lower() or "@" in f for f in v.failures)


def test_copy_preserves_hashtag():
    c = load_campaign("dona30k")
    c.with_handle("@foo")
    copies = build_platform_copy(c, clip_hook="Momento epico")
    assert len(copies) == 4
    for pc in copies:
        blob = pc.title + pc.description
        assert "#dona30K" in blob
        assert "#DONA30K" not in blob.replace("#dona30K", "")
