"""
AI Content Advisor — optional post-clip step.
Does NOT choose cuts or alter timestamps.
Produces publication package (context, titles, platform copy).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from src.frame_extractor import cleanup_frames, extract_frames
from src.local_ai.prompts import SYSTEM_PROMPT, build_user_prompt
from src.ollama_manager import (
    BASE_VISION_MODEL,
    DEFAULT_VISION_MODEL,
    ensure_optimized_model,
    get_status,
    is_ollama_running,
)
from src.transcription import Segment
from src.utils import extract_json_from_text

log = logging.getLogger("video_clipper.content_advisor")


@dataclass
class ContextBlock:
    summary: str = ""
    topic: str = ""
    tone: str = ""
    hook: str = ""
    key_points: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    artists: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)


@dataclass
class TitleBlock:
    primary: str = ""
    alternatives: list[str] = field(default_factory=list)


@dataclass
class TikTokBlock:
    hook: str = ""
    caption: str = ""
    hashtags: list[str] = field(default_factory=list)
    cta: str = ""
    cover_text: str = ""
    strategy: str = ""


@dataclass
class YouTubeBlock:
    title: str = ""
    description: str = ""
    hashtags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    cta: str = ""


@dataclass
class InstagramBlock:
    caption: str = ""
    hashtags: list[str] = field(default_factory=list)
    cta: str = ""
    cover_text: str = ""
    strategy: str = ""


@dataclass
class ContentPackage:
    context: ContextBlock = field(default_factory=ContextBlock)
    title: TitleBlock = field(default_factory=TitleBlock)
    tiktok: TikTokBlock = field(default_factory=TikTokBlock)
    youtube_shorts: YouTubeBlock = field(default_factory=YouTubeBlock)
    instagram_reels: InstagramBlock = field(default_factory=InstagramBlock)
    raw: str = ""
    inference_ms: int = 0
    model: str = ""
    used_frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": asdict(self.context),
            "title": asdict(self.title),
            "platforms": {
                "tiktok": asdict(self.tiktok),
                "youtube_shorts": asdict(self.youtube_shorts),
                "instagram_reels": asdict(self.instagram_reels),
            },
        }

    def copy_all_text(self) -> str:
        lines = [
            "=== TÍTULO ===",
            self.title.primary,
            *self.title.alternatives,
            "",
            "=== TIKTOK ===",
            self.tiktok.hook,
            self.tiktok.caption,
            " ".join(self.tiktok.hashtags),
            self.tiktok.cta,
            "",
            "=== YOUTUBE SHORTS ===",
            self.youtube_shorts.title,
            self.youtube_shorts.description,
            " ".join(self.youtube_shorts.hashtags),
            "",
            "=== INSTAGRAM REELS ===",
            self.instagram_reels.caption,
            " ".join(self.instagram_reels.hashtags),
            self.instagram_reels.cover_text,
        ]
        return "\n".join(lines)


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v).strip()] if str(v).strip() else []


def _s(v: Any, default: str = "Não identificado.") -> str:
    if v is None:
        return default
    t = str(v).strip()
    return t if t else default


def parse_content_json(raw: str) -> ContentPackage:
    data = extract_json_from_text(raw)
    if not data:
        raise ValueError("JSON inválido do Content Advisor.")

    ctx = data.get("context") or {}
    title = data.get("title") or {}
    plats = data.get("platforms") or {}
    tt = plats.get("tiktok") or {}
    yt = plats.get("youtube_shorts") or {}
    ig = plats.get("instagram_reels") or {}

    return ContentPackage(
        context=ContextBlock(
            summary=_s(ctx.get("summary"), ""),
            topic=_s(ctx.get("topic")),
            tone=_s(ctx.get("tone")),
            hook=_s(ctx.get("hook"), ""),
            key_points=_as_list(ctx.get("key_points")),
            people=_as_list(ctx.get("people")),
            artists=_as_list(ctx.get("artists")),
            references=_as_list(ctx.get("references")),
        ),
        title=TitleBlock(
            primary=_s(title.get("primary"), ""),
            alternatives=_as_list(title.get("alternatives"))[:5],
        ),
        tiktok=TikTokBlock(
            hook=_s(tt.get("hook"), ""),
            caption=_s(tt.get("caption"), ""),
            hashtags=_as_list(tt.get("hashtags")),
            cta=_s(tt.get("cta"), ""),
            cover_text=_s(tt.get("cover_text"), ""),
            strategy=_s(tt.get("strategy"), ""),
        ),
        youtube_shorts=YouTubeBlock(
            title=_s(yt.get("title"), ""),
            description=_s(yt.get("description"), ""),
            hashtags=_as_list(yt.get("hashtags")),
            keywords=_as_list(yt.get("keywords")),
            cta=_s(yt.get("cta"), ""),
        ),
        instagram_reels=InstagramBlock(
            caption=_s(ig.get("caption"), ""),
            hashtags=_as_list(ig.get("hashtags")),
            cta=_s(ig.get("cta"), ""),
            cover_text=_s(ig.get("cover_text"), ""),
            strategy=_s(ig.get("strategy"), ""),
        ),
        raw=raw,
    )


def _transcript_text(segments: list[Segment], full_text: str = "") -> str:
    if segments:
        return " ".join(s.text.strip() for s in segments if s.text.strip())
    return full_text or ""


def analyze_content(
    clip_path: Path,
    duration_sec: float,
    segments: Optional[list[Segment]] = None,
    full_text: str = "",
    use_vision: bool = True,
    max_frames: int = 2,
) -> ContentPackage:
    """
    Run local Content Advisor on an already-generated clip.
    Never modifies timestamps or re-cuts video.
    """
    if not is_ollama_running():
        raise RuntimeError(
            "Ollama offline. Inicie o Ollama para usar o AI Content Advisor."
        )

    status = get_status()
    if not status.model_installed:
        raise RuntimeError(
            "Modelo local não encontrado. Rode: ollama pull qwen2.5vl:7b"
        )

    model = ensure_optimized_model()
    transcript = _transcript_text(segments or [], full_text)
    user_prompt = build_user_prompt(transcript, duration_sec)

    frames: list[Path] = []
    if use_vision and clip_path.exists():
        try:
            frames = extract_frames(
                clip_path, duration=duration_sec, max_frames=max_frames, width=320
            )
        except Exception as exc:
            log.warning("Frame extract failed, text-only: %s", exc)
            frames = []

    t0 = time.time()
    raw = ""
    try:
        import ollama

        msg: dict[str, Any] = {"role": "user", "content": user_prompt}
        if frames:
            msg["images"] = [str(p) for p in frames]

        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                msg,
            ],
            options={
                "temperature": 0.2,
                "num_predict": 1200,
                "num_ctx": 4096,
                "num_batch": 256,
            },
            format="json",
        )
        if isinstance(response, dict):
            raw = response.get("message", {}).get("content", "") or ""
        else:
            raw = getattr(getattr(response, "message", None), "content", "") or ""
    except Exception as exc:
        cleanup_frames(frames)
        raise RuntimeError(f"Content Advisor falhou: {exc}") from exc
    finally:
        cleanup_frames(frames)

    ms = int((time.time() - t0) * 1000)
    pkg = parse_content_json(raw)
    pkg.inference_ms = ms
    pkg.model = model
    pkg.used_frames = len(frames)
    log.info("Content Advisor done model=%s ms=%d frames=%d", model, ms, len(frames))
    return pkg
