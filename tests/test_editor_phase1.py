"""Phase 1 editor architecture tests — no real video render."""

import json
from pathlib import Path

from src.editor.export_plan import build_export_plan
from src.editor.history import HistoryStack
from src.editor.models import AspectRatio, CropSettings, ProjectState, TimelineRange
from src.editor.operations import apply_split, apply_trim, frame_step, new_project_from_clip, set_aspect
from src.editor.project_io import load_project, save_project
from src.transcription import Segment


def test_timeline_range_rejects_inverted():
    try:
        TimelineRange(start=10, end=5)
        assert False, "should raise"
    except ValueError:
        pass


def test_new_project_and_trim():
    p = new_project_from_clip("/tmp/clip.mp4", duration=40.0, fps=24.0, name="t")
    assert p.timeline_duration == 40.0
    assert p.fps == 24.0
    trimmed = apply_trim(p, 5.0, 35.0)
    assert abs(trimmed.timeline_duration - 30.0) < 1e-6
    assert trimmed.playable_range.start == 5.0


def test_captions_from_segments():
    segs = [
        Segment(start=100.0, end=105.0, text="hello"),
        Segment(start=106.0, end=110.0, text="world"),
        Segment(start=200.0, end=205.0, text="out"),
    ]
    p = new_project_from_clip(
        "c.mp4", duration=20.0, segments=segs, clip_start_abs=100.0
    )
    assert len(p.captions) == 2
    assert p.captions[0].text == "hello"
    assert abs(p.captions[0].start - 0.0) < 1e-6


def test_split():
    p = new_project_from_clip("c.mp4", duration=40.0)
    left, right = apply_split(p, 15.0)
    assert abs(left.timeline_duration - 15.0) < 1e-6
    assert abs(right.timeline_duration - 25.0) < 1e-6


def test_history_undo_redo():
    p = new_project_from_clip("c.mp4", duration=40.0)
    h = HistoryStack(p)
    h.push(apply_trim(h.current, 0, 20))
    assert abs(h.current.timeline_duration - 20.0) < 1e-6
    h.undo()
    assert abs(h.current.timeline_duration - 40.0) < 1e-6
    h.redo()
    assert abs(h.current.timeline_duration - 20.0) < 1e-6


def test_project_io(tmp_path: Path):
    p = new_project_from_clip("c.mp4", duration=12.5, fps=29.97)
    p = set_aspect(p, AspectRatio.SQUARE_1_1)
    path = tmp_path / "project.json"
    save_project(p, path)
    loaded = load_project(path)
    assert loaded.aspect == AspectRatio.SQUARE_1_1
    assert abs(loaded.source_duration - 12.5) < 1e-6
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "source_path" in data
    assert "video" not in data  # never embed binary


def test_export_plan_copy_vs_reencode():
    p = new_project_from_clip("/media/c.mp4", duration=30.0)
    p.aspect = AspectRatio.LANDSCAPE_16_9
    plan = build_export_plan(p, "ffmpeg", "/out/a.mp4")
    assert plan.needs_reencode is False
    assert "copy" in plan.args

    p2 = set_aspect(p, AspectRatio.VERTICAL_9_16)
    plan2 = build_export_plan(p2, "ffmpeg", "/out/b.mp4")
    assert plan2.needs_reencode is True
    assert "libx264" in plan2.args
    assert plan2.width == 1080 and plan2.height == 1920


def test_frame_step():
    p = new_project_from_clip("c.mp4", duration=10.0, fps=25.0)
    p.playhead = 1.0
    n = frame_step(p, +1)
    assert abs(n.playhead - 1.04) < 1e-6


def test_crop_clamp():
    c = CropSettings(zoom=99, center_x=-1, center_y=2)
    assert c.zoom == 4.0
    assert c.center_x == 0.0
    assert c.center_y == 1.0
