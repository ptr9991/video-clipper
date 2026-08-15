"""
Video Clipper — professional local tool.
Clip Generator (Groq + FFmpeg) + optional AI Content Advisor (Ollama).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import streamlit as st

from src.clip_analyzer import ClipCandidate, analyze_best_clip
from src.clip_editor import PIP_POSITIONS, EditOptions, render_edited_clip
from src.config import MAX_CLIP_DURATION, OUTPUT_DIR, TEMP_DIR, check_ffmpeg, require_api_key
from src.downloader import QUALITY_PRESETS, download_video
from src.local_ai.content_advisor import ContentPackage, analyze_content
from src.ollama_manager import DEFAULT_VISION_MODEL, get_status
from src.transcription import Segment, TranscriptionResult, transcribe_video
from src.utils import cleanup_file, format_timestamp, generate_output_filename, safe_filename
from src.video_processor import VideoInfo, cut_video, get_video_info

logger = logging.getLogger("video_clipper.app")

st.set_page_config(
    page_title="Video Clipper",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design system (dark / editorial / media tool)
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
  font-family: 'Inter', system-ui, sans-serif;
}

.stApp {
  background: #0a0a0b;
  color: #e8e8ea;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; height: 0; }
[data-testid="stToolbar"] { display: none; }
.block-container {
  padding-top: 1.5rem !important;
  padding-bottom: 3rem !important;
  max-width: 1100px;
}

/* Typography */
h1, h2, h3 { letter-spacing: -0.02em; font-weight: 600 !important; color: #f4f4f5 !important; }

/* Accent */
:root {
  --accent: #c8f542;
  --surface: #141416;
  --border: #2a2a2e;
  --muted: #8b8b93;
}

/* Cards */
.vc-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
}
.vc-label {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin-bottom: 0.35rem;
}
.vc-value {
  font-size: 1.05rem;
  font-weight: 500;
  color: #f4f4f5;
}
.vc-muted { color: var(--muted); font-size: 0.875rem; }
.vc-accent { color: var(--accent); }

.vc-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: 1rem;
}
.vc-logo {
  font-size: 1.15rem;
  font-weight: 700;
  letter-spacing: -0.03em;
}
.vc-logo span { color: var(--accent); }

.vc-section-title {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  margin: 1.75rem 0 0.75rem;
}

/* Buttons */
.stButton > button {
  border-radius: 8px !important;
  font-weight: 600 !important;
  border: 1px solid var(--border) !important;
  background: #1a1a1d !important;
  color: #f4f4f5 !important;
  transition: all 0.15s ease;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}
.stButton > button[kind="primary"] {
  background: var(--accent) !important;
  color: #0a0a0b !important;
  border-color: var(--accent) !important;
}
.stButton > button[kind="primary"]:hover {
  filter: brightness(1.08);
  color: #0a0a0b !important;
}

/* Inputs */
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
  background: #0e0e10 !important;
  border-radius: 8px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.5rem;
  border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  color: var(--muted);
  border-radius: 0;
  font-weight: 500;
}
.stTabs [aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent);
}

/* Metrics strip */
.vc-meta {
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
  margin: 0.75rem 0 1rem;
}
.vc-meta-item .k { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
.vc-meta-item .v { font-size: 0.95rem; font-weight: 500; }

/* Timeline bar */
.vc-timeline {
  height: 6px;
  background: #1e1e22;
  border-radius: 3px;
  position: relative;
  margin: 0.75rem 0 0.25rem;
}
.vc-timeline-fill {
  position: absolute;
  height: 100%;
  background: var(--accent);
  border-radius: 3px;
  opacity: 0.85;
}

div[data-testid="stStatusWidget"] { display: none; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def init_state() -> None:
    defaults = {
        "video_path": None,
        "video_name": None,
        "video_info": None,
        "transcription": None,
        "candidate": None,
        "clip_path": None,
        "edited_path": None,
        "manual_start": 0.0,
        "manual_end": 40.0,
        "cut_mode": "fast",
        "source_url": "",
        "webcam_path": None,
        "content_pkg": None,
        "advisor_enabled": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def show_error(msg: str) -> None:
    st.error(msg)
    logger.error(msg)


def save_upload(uploaded, prefix: str = "upload") -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.name).suffix.lower() or ".mp4"
    dest = TEMP_DIR / f"{prefix}_{safe_filename(Path(uploaded.name).stem) or prefix}{suffix}"
    with dest.open("wb") as f:
        f.write(uploaded.getbuffer())
    return dest


def reset_pipeline() -> None:
    st.session_state.video_info = None
    st.session_state.transcription = None
    st.session_state.candidate = None
    st.session_state.clip_path = None
    st.session_state.edited_path = None
    st.session_state.content_pkg = None


def copy_block(label: str, text: str, key: str) -> None:
    st.code(text or "—", language=None)
    st.download_button(
        f"Copiar / baixar · {label}",
        data=(text or "").encode("utf-8"),
        file_name=f"{key}.txt",
        mime="text/plain",
        key=f"dl_{key}",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="vc-header"><div class="vc-logo">VIDEO <span>CLIPPER</span></div>'
    '<div class="vc-muted">Local · Groq speech · FFmpeg · Optional local advisor</div></div>',
    unsafe_allow_html=True,
)

ok_ffmpeg, _ = check_ffmpeg()
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.markdown(
        f'<div class="vc-card"><div class="vc-label">FFmpeg</div>'
        f'<div class="vc-value">{"Online" if ok_ffmpeg else "Offline"}</div></div>',
        unsafe_allow_html=True,
    )
with col_s2:
    try:
        require_api_key()
        groq_ok = True
    except RuntimeError:
        groq_ok = False
    st.markdown(
        f'<div class="vc-card"><div class="vc-label">Groq</div>'
        f'<div class="vc-value">{"Ready" if groq_ok else "API key missing"}</div></div>',
        unsafe_allow_html=True,
    )
with col_s3:
    ost = get_status(DEFAULT_VISION_MODEL)
    st.markdown(
        f'<div class="vc-card"><div class="vc-label">Content Advisor</div>'
        f'<div class="vc-value">{"Connected" if ost.ready else ost.message or "Offline"}</div></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------
st.markdown('<div class="vc-section-title">Source</div>', unsafe_allow_html=True)
tab_up, tab_url = st.tabs(["File", "URL"])
with tab_up:
    uploaded = st.file_uploader("Video file", type=["mp4", "mov", "mkv", "webm"], label_visibility="collapsed")
    if uploaded is not None:
        name = safe_filename(uploaded.name)
        if st.session_state.video_name != name or not (
            st.session_state.video_path and Path(st.session_state.video_path).exists()
        ):
            if st.session_state.video_path:
                cleanup_file(Path(st.session_state.video_path))
            st.session_state.video_path = str(save_upload(uploaded))
            st.session_state.video_name = name
            reset_pipeline()

with tab_url:
    url = st.text_input("URL", value=st.session_state.source_url, placeholder="https://youtube.com/...")
    q = st.selectbox("Quality", list(QUALITY_PRESETS.keys()), index=2)
    if st.button("Download", use_container_width=True):
        if url.strip():
            with st.spinner("Downloading…"):
                try:
                    path = download_video(url.strip(), quality=q)
                    st.session_state.video_path = str(path)
                    st.session_state.video_name = path.name
                    st.session_state.source_url = url.strip()
                    reset_pipeline()
                    st.rerun()
                except Exception as e:
                    show_error(str(e))

video_path = Path(st.session_state.video_path) if st.session_state.video_path else None
if video_path and not video_path.exists():
    video_path = None
    st.session_state.video_path = None

# ---------------------------------------------------------------------------
# Clip Generator
# ---------------------------------------------------------------------------
if video_path:
    if st.session_state.video_info is None:
        with st.spinner("Reading metadata…"):
            st.session_state.video_info = get_video_info(video_path)
    info: VideoInfo = st.session_state.video_info

    st.markdown('<div class="vc-section-title">Clip Generator</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="vc-meta">'
        f'<div class="vc-meta-item"><div class="k">File</div><div class="v">{(st.session_state.video_name or "")[:40]}</div></div>'
        f'<div class="vc-meta-item"><div class="k">Size</div><div class="v">{info.size_mb:.1f} MB</div></div>'
        f'<div class="vc-meta-item"><div class="k">Duration</div><div class="v">{format_timestamp(info.duration)}</div></div>'
        f'<div class="vc-meta-item"><div class="k">Resolution</div><div class="v">{info.resolution if info.width else "—"}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.video(str(video_path))

    if not ok_ffmpeg:
        st.warning("FFmpeg required.")
        st.stop()
    if not groq_ok:
        st.warning("Configure GROQ_API_KEY to find clips.")
        st.stop()

    if st.button("Find best clip", type="primary", use_container_width=True):
        st.session_state.candidate = None
        st.session_state.clip_path = None
        st.session_state.edited_path = None
        st.session_state.content_pkg = None
        status = st.status("Working…", expanded=True)
        try:
            status.write("Transcribing audio (Groq Whisper)…")
            transcription, audio_path = transcribe_video(video_path)
            st.session_state.transcription = transcription
            cleanup_file(audio_path)
            status.write("Selecting segment (Groq LLM)…")
            cand = analyze_best_clip(transcription, video_duration=info.duration)
            st.session_state.candidate = cand
            st.session_state.manual_start = cand.start
            st.session_state.manual_end = cand.end
            status.update(label="Segment ready", state="complete")
        except Exception as e:
            status.update(label="Error", state="error")
            show_error(str(e))

if st.session_state.candidate and st.session_state.video_info:
    cand: ClipCandidate = st.session_state.candidate
    info = st.session_state.video_info
    max_dur = float(info.duration) or 3600.0
    start_pct = (cand.start / max_dur) * 100 if max_dur else 0
    width_pct = (cand.duration / max_dur) * 100 if max_dur else 5

    st.markdown(
        f'<div class="vc-card">'
        f'<div class="vc-label">Selected segment</div>'
        f'<div class="vc-timeline"><div class="vc-timeline-fill" style="left:{start_pct:.2f}%;width:{width_pct:.2f}%"></div></div>'
        f'<div class="vc-meta">'
        f'<div class="vc-meta-item"><div class="k">Start</div><div class="v">{format_timestamp(cand.start)}</div></div>'
        f'<div class="vc-meta-item"><div class="k">End</div><div class="v">{format_timestamp(cand.end)}</div></div>'
        f'<div class="vc-meta-item"><div class="k">Duration</div><div class="v">{cand.duration:.1f}s</div></div>'
        f'<div class="vc-meta-item"><div class="k">Speech score</div><div class="v">{cand.score}/100</div></div>'
        f'</div>'
        f'<p class="vc-muted">{cand.reason}</p></div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        new_start = st.slider("Start (s)", 0.0, max(0.1, max_dur - 1), float(st.session_state.manual_start), 0.1)
    with c2:
        new_end = st.slider("End (s)", 0.1, max_dur, float(st.session_state.manual_end), 0.1)
    if new_end <= new_start:
        new_end = min(new_start + 30, max_dur)
    if new_end - new_start > MAX_CLIP_DURATION:
        new_end = new_start + MAX_CLIP_DURATION
    st.session_state.manual_start = new_start
    st.session_state.manual_end = new_end

    mode = st.radio("Cut mode", ["fast", "precise"], horizontal=True, format_func=lambda x: "Fast copy" if x == "fast" else "Precise re-encode")
    if st.button("Export clip", type="primary", use_container_width=True):
        OUTPUT_DIR.mkdir(exist_ok=True)
        out = OUTPUT_DIR / generate_output_filename()
        with st.spinner("Cutting with FFmpeg…"):
            try:
                cut_video(Path(st.session_state.video_path), new_start, new_end, out, mode=mode)
                st.session_state.clip_path = str(out)
                st.session_state.edited_path = None
                st.session_state.content_pkg = None
                st.success("Clip exported")
            except Exception as e:
                show_error(str(e))

# ---------------------------------------------------------------------------
# Clip preview + editor + Content Advisor
# ---------------------------------------------------------------------------
if st.session_state.clip_path and Path(st.session_state.clip_path).exists():
    clip_p = Path(st.session_state.clip_path)
    st.markdown('<div class="vc-section-title">Clip</div>', unsafe_allow_html=True)
    st.video(str(clip_p))
    with clip_p.open("rb") as f:
        st.download_button("Download clip", f.read(), file_name=clip_p.name, mime="video/mp4", use_container_width=True)

    with st.expander("Editor (9:16 · subtitles · webcam PiP)"):
        vertical = st.checkbox("Vertical 9:16", True)
        add_subs = st.checkbox("Burn-in subtitles", True)
        cam = st.file_uploader("Webcam video (optional)", type=["mp4", "mov", "webm"], key="cam")
        if cam:
            st.session_state.webcam_path = str(save_upload(cam, "webcam"))
        cam_pos = st.selectbox("Webcam position", list(PIP_POSITIONS.keys()))
        if st.button("Render edit", use_container_width=True):
            tr: Optional[TranscriptionResult] = st.session_state.transcription
            segs = tr.segments if tr else []
            opts = EditOptions(
                vertical_9x16=vertical,
                add_subtitles=add_subs,
                webcam_path=Path(st.session_state.webcam_path) if st.session_state.webcam_path else None,
                webcam_position=cam_pos,
            )
            with st.spinner("Rendering…"):
                try:
                    edited = render_edited_clip(
                        clip_p,
                        opts,
                        segments=segs,
                        clip_start_abs=float(st.session_state.manual_start),
                        clip_end_abs=float(st.session_state.manual_end),
                    )
                    st.session_state.edited_path = str(edited)
                except Exception as e:
                    show_error(str(e))
        if st.session_state.edited_path and Path(st.session_state.edited_path).exists():
            ep = Path(st.session_state.edited_path)
            st.video(str(ep))
            with ep.open("rb") as f:
                st.download_button("Download edited", f.read(), file_name=ep.name, mime="video/mp4", use_container_width=True)

    # ---- AI Content Advisor (optional, independent) ----
    st.markdown('<div class="vc-section-title">AI Content Advisor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="vc-card"><p class="vc-muted">'
        "Optional. Runs <b>locally</b> via Ollama. Does not change the cut. "
        "Produces context, titles and platform-ready copy from the finished clip."
        "</p></div>",
        unsafe_allow_html=True,
    )

    if not ost.ready:
        st.caption(f"Advisor offline: {ost.message}. Clip download still works.")
    else:
        if st.button("Analyze with local AI", use_container_width=True):
            tr = st.session_state.transcription
            segs = tr.segments if tr else []
            text = tr.text if tr else ""
            dur = max(1.0, float(st.session_state.manual_end - st.session_state.manual_start))
            with st.spinner("Local model analyzing clip (may take 1–3 min)…"):
                try:
                    pkg = analyze_content(
                        clip_path=clip_p,
                        duration_sec=dur,
                        segments=segs,
                        full_text=text,
                        use_vision=True,
                        max_frames=2,
                    )
                    st.session_state.content_pkg = pkg
                except Exception as e:
                    show_error(str(e))

    pkg: Optional[ContentPackage] = st.session_state.content_pkg
    if pkg:
        st.markdown("#### Context")
        st.write(pkg.context.summary or "—")
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"**Topic**  \n{pkg.context.topic}")
        m2.markdown(f"**Tone**  \n{pkg.context.tone}")
        m3.markdown(f"**Hook**  \n{pkg.context.hook}")
        if pkg.context.key_points:
            st.markdown("**Key points**")
            for p in pkg.context.key_points:
                st.markdown(f"- {p}")
        if pkg.context.people or pkg.context.artists:
            st.caption(
                "People: " + ", ".join(pkg.context.people or ["Não identificado."])
                + " · Artists: " + ", ".join(pkg.context.artists or ["Não identificado."])
            )

        st.markdown("#### Title")
        copy_block("title", pkg.title.primary, "title_primary")
        if pkg.title.alternatives:
            st.caption("Alternatives")
            for i, alt in enumerate(pkg.title.alternatives):
                st.text(alt)

        t_tt, t_yt, t_ig = st.tabs(["TikTok", "YouTube Shorts", "Instagram Reels"])
        with t_tt:
            body = (
                f"Hook: {pkg.tiktok.hook}\n\n{pkg.tiktok.caption}\n\n"
                f"{' '.join(pkg.tiktok.hashtags)}\n\nCTA: {pkg.tiktok.cta}\n"
                f"Cover: {pkg.tiktok.cover_text}\n\nStrategy: {pkg.tiktok.strategy}"
            )
            copy_block("TikTok", body, "tiktok")
        with t_yt:
            body = (
                f"{pkg.youtube_shorts.title}\n\n{pkg.youtube_shorts.description}\n\n"
                f"{' '.join(pkg.youtube_shorts.hashtags)}\n"
                f"Keywords: {', '.join(pkg.youtube_shorts.keywords)}\n"
                f"CTA: {pkg.youtube_shorts.cta}"
            )
            copy_block("YouTube", body, "youtube")
        with t_ig:
            body = (
                f"{pkg.instagram_reels.caption}\n\n{' '.join(pkg.instagram_reels.hashtags)}\n\n"
                f"CTA: {pkg.instagram_reels.cta}\nCover: {pkg.instagram_reels.cover_text}\n"
                f"Strategy: {pkg.instagram_reels.strategy}"
            )
            copy_block("Instagram", body, "instagram")

        st.download_button(
            "Download all copy",
            data=pkg.copy_all_text().encode("utf-8"),
            file_name="content_package.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.caption(f"Model {pkg.model} · {pkg.inference_ms} ms · frames {pkg.used_frames}")

st.markdown(
    '<p class="vc-muted" style="margin-top:2rem">'
    "Clip selection uses Groq. Content Advisor is local and optional. Video files stay on your machine for FFmpeg and Ollama."
    "</p>",
    unsafe_allow_html=True,
)
