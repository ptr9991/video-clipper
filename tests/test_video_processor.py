"""Unit tests for video_processor helpers (no real FFmpeg calls required for these)."""

from src.video_processor import build_cut_command


def test_build_cut_command_fast():
    cmd = build_cut_command(
        ffmpeg_path="/usr/bin/ffmpeg",
        input_path="/tmp/in.mp4",
        start=12.345,
        duration=40.0,
        output_path="/tmp/out.mp4",
        mode="fast",
    )
    assert cmd[0] == "/usr/bin/ffmpeg"
    assert "-ss" in cmd
    assert "12.345" in cmd
    assert "-c" in cmd
    assert "copy" in cmd
    assert "-t" in cmd
    assert "40.000" in cmd
    assert cmd[-1] == "/tmp/out.mp4"
    # -ss should appear before -i in fast mode
    ss_idx = cmd.index("-ss")
    i_idx = cmd.index("-i")
    assert ss_idx < i_idx


def test_build_cut_command_precise():
    cmd = build_cut_command(
        ffmpeg_path="ffmpeg",
        input_path="in.mp4",
        start=5.0,
        duration=35.5,
        output_path="out.mp4",
        mode="precise",
    )
    assert "libx264" in cmd
    assert "aac" in cmd
    # -ss after -i for accuracy
    i_idx = cmd.index("-i")
    ss_idx = cmd.index("-ss")
    assert ss_idx > i_idx
