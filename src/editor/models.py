"""
Editor domain models — pure data, no FFmpeg side effects.

ProjectState is the single source of truth for the timeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class AspectRatio(str, Enum):
    VERTICAL_9_16 = "9:16"
    SQUARE_1_1 = "1:1"
    LANDSCAPE_16_9 = "16:9"

    @property
    def size(self) -> tuple[int, int]:
        return {
            AspectRatio.VERTICAL_9_16: (1080, 1920),
            AspectRatio.SQUARE_1_1: (1080, 1080),
            AspectRatio.LANDSCAPE_16_9: (1920, 1080),
        }[self]


@dataclass
class TimelineRange:
    """Inclusive start, exclusive end in seconds relative to source media."""

    start: float = 0.0
    end: float = 0.0

    def __post_init__(self) -> None:
        if self.start < 0:
            self.start = 0.0
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) < start ({self.start})")

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def clamp(self, media_duration: float) -> "TimelineRange":
        s = max(0.0, min(self.start, media_duration))
        e = max(s, min(self.end, media_duration))
        return TimelineRange(start=s, end=e)

    def to_dict(self) -> dict[str, float]:
        return {"start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TimelineRange":
        return cls(start=float(d.get("start", 0)), end=float(d.get("end", 0)))


@dataclass
class CropSettings:
    """Normalized crop center (0–1) and zoom (>=1). Applied at export."""

    zoom: float = 1.0
    center_x: float = 0.5  # 0 left … 1 right
    center_y: float = 0.5

    def __post_init__(self) -> None:
        self.zoom = max(1.0, min(4.0, float(self.zoom)))
        self.center_x = max(0.0, min(1.0, float(self.center_x)))
        self.center_y = max(0.0, min(1.0, float(self.center_y)))

    def to_dict(self) -> dict[str, float]:
        return {"zoom": self.zoom, "center_x": self.center_x, "center_y": self.center_y}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CropSettings":
        return cls(
            zoom=float(d.get("zoom", 1.0)),
            center_x=float(d.get("center_x", 0.5)),
            center_y=float(d.get("center_y", 0.5)),
        )


@dataclass
class AudioSettings:
    volume: float = 1.0  # 0–2
    muted: bool = False
    fade_in: float = 0.0
    fade_out: float = 0.0

    def __post_init__(self) -> None:
        self.volume = max(0.0, min(2.0, float(self.volume)))
        self.fade_in = max(0.0, float(self.fade_in))
        self.fade_out = max(0.0, float(self.fade_out))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AudioSettings":
        return cls(
            volume=float(d.get("volume", 1.0)),
            muted=bool(d.get("muted", False)),
            fade_in=float(d.get("fade_in", 0.0)),
            fade_out=float(d.get("fade_out", 0.0)),
        )


@dataclass
class CaptionCue:
    id: str
    start: float
    end: float
    text: str
    highlight_words: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "highlight_words": list(self.highlight_words),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CaptionCue":
        return cls(
            id=str(d.get("id", "")),
            start=float(d.get("start", 0)),
            end=float(d.get("end", 0)),
            text=str(d.get("text", "")),
            highlight_words=list(d.get("highlight_words") or []),
        )


@dataclass
class TextOverlay:
    id: str
    text: str
    start: float
    end: float
    x: float = 0.5  # normalized
    y: float = 0.2
    font_size: int = 48
    color: str = "#FFFFFF"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TextOverlay":
        return cls(
            id=str(d.get("id", "")),
            text=str(d.get("text", "")),
            start=float(d.get("start", 0)),
            end=float(d.get("end", 0)),
            x=float(d.get("x", 0.5)),
            y=float(d.get("y", 0.2)),
            font_size=int(d.get("font_size", 48)),
            color=str(d.get("color", "#FFFFFF")),
        )


@dataclass
class CaptionStyle:
    name: str = "clean"
    font_size: int = 42
    primary_color: str = "#FFFFFF"
    highlight_color: str = "#C8F542"
    outline: int = 2
    margin_v: int = 80

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CaptionStyle":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})  # type: ignore


@dataclass
class ProjectState:
    """
    Full editor project (serializable to project.json).

    source_path: absolute or project-relative path to the clip media.
    playable_range: portion of source used on the timeline (trim).
    """

    version: int = 1
    name: str = "Untitled"
    source_path: str = ""
    source_duration: float = 0.0
    fps: float = 30.0
    playable_range: TimelineRange = field(default_factory=TimelineRange)
    aspect: AspectRatio = AspectRatio.VERTICAL_9_16
    crop: CropSettings = field(default_factory=CropSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    captions: list[CaptionCue] = field(default_factory=list)
    caption_style: CaptionStyle = field(default_factory=CaptionStyle)
    texts: list[TextOverlay] = field(default_factory=list)
    playhead: float = 0.0  # seconds within playable timeline (0..duration)

    @property
    def timeline_duration(self) -> float:
        return self.playable_range.duration

    def validate(self) -> None:
        if self.source_duration < 0:
            raise ValueError("source_duration < 0")
        self.playable_range = self.playable_range.clamp(self.source_duration)
        if self.fps <= 0:
            self.fps = 30.0
        self.playhead = max(0.0, min(self.playhead, self.timeline_duration))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "source_path": self.source_path,
            "source_duration": self.source_duration,
            "fps": self.fps,
            "playable_range": self.playable_range.to_dict(),
            "aspect": self.aspect.value,
            "crop": self.crop.to_dict(),
            "audio": self.audio.to_dict(),
            "captions": [c.to_dict() for c in self.captions],
            "caption_style": self.caption_style.to_dict(),
            "texts": [t.to_dict() for t in self.texts],
            "playhead": self.playhead,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProjectState":
        aspect_raw = d.get("aspect", "9:16")
        try:
            aspect = AspectRatio(aspect_raw)
        except ValueError:
            aspect = AspectRatio.VERTICAL_9_16
        proj = cls(
            version=int(d.get("version", 1)),
            name=str(d.get("name", "Untitled")),
            source_path=str(d.get("source_path", "")),
            source_duration=float(d.get("source_duration", 0)),
            fps=float(d.get("fps", 30)),
            playable_range=TimelineRange.from_dict(d.get("playable_range") or {}),
            aspect=aspect,
            crop=CropSettings.from_dict(d.get("crop") or {}),
            audio=AudioSettings.from_dict(d.get("audio") or {}),
            captions=[CaptionCue.from_dict(c) for c in (d.get("captions") or [])],
            caption_style=CaptionStyle.from_dict(d.get("caption_style") or {}),
            texts=[TextOverlay.from_dict(t) for t in (d.get("texts") or [])],
            playhead=float(d.get("playhead", 0)),
        )
        proj.validate()
        return proj

    def clone(self) -> "ProjectState":
        return ProjectState.from_dict(self.to_dict())
