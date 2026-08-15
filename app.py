"""Video Clipper — generator + professional local editor + advisor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import streamlit as st

from src.clip_analyzer import ClipCandidate, analyze_best_clip
from src.config import MAX_CLIP_DURATION, OUTPUT_DIR, TEMP_DIR, check_ffmpeg, require_api_key
from src.downloader import QUALITY_PRESETS, download_video
from src.editor import (
    AspectRatio,
    AudioSettings,
    CropSettings,
    HistoryStack,
    add_text_overlay,
    apply_split_keep_left,
    apply_split_keep_right,
    apply_trim,
    frame_step,
    new_project_from_clip,
    render_timeline_html,
    run_export,
    set_aspect,
    set_caption_style,
    set_crop,
    set_playhead,
)
from src.editor.caption_styles import STYLES
from src.local_ai.content_advisor import ContentPackage, analyze_content
from src.ollama_manager import DEFAULT_VISION_MODEL, get_status
from src.transcription import TranscriptionResult, transcribe_video
from src.utils import cleanup_file, format_timestamp, generate_output_filename, safe_filename
from src.video_processor import VideoInfo, cut_video, get_video_info

logger = logging.getLogger("video_clipper.app")

st.set_page_config(page_title="Video Clipper", page_icon="▶", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: Inter, system-ui, sans-serif; }
.stApp { background: #080808; color: #f5f5f5; }
#MainMenu, footer, header { visibility: hidden; height: 0; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding-top: 1rem !important; max-width: 1280px; }
:root { --accent: #c8f542; --surface: #111; --border: #222; --muted: #8a8a8a; }
.vc-header { display:flex; justify-content:space-between; border-bottom:1px solid #222; padding-bottom:0.75rem; margin-bottom:1rem; }
.vc-logo { font-weight:700; letter-spacing:-0.03em; } .vc-logo span { color: var(--accent); }
.vc-section { font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; color:var(--muted); margin:1.25rem 0 0.5rem; }
.vc-card { background:#111; border:1px solid #222; border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.75rem; }
.vc-label { font-size:0.65rem; text-transform:uppercase; color:var(--muted); }
.vc-value { font-weight:500; }
.vc-muted { color:var(--muted); font-size:0.85rem; }
.vc-meta { display:flex; gap:1.2rem; flex-wrap:wrap; margin:0.4rem 0; }
.vc-meta .k { font-size:0.6rem; text-transform:uppercase; color:var(--muted); }
.stButton > button { border-radius:8px !important; font-weight:600 !important; background:#161616 !important; border:1px solid #2a2a2a !important; color:#f5f5f5 !important; }
.stButton > button[kind="primary"] { background:var(--accent) !important; color:#080808 !important; border-color:var(--accent) !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def init_state() -> None:
    d = {
        "video_path": None, "video_name": None, "video_info": None,
        "transcription": None, "candidate": None, "clip_path": None,
        "source_url": "", "content_pkg": None, "prefer_local": True,
        "editor_open": False, "editor_history": None, "export_path": None,
        "manual_start": 0.0, "manual_end": 40.0, "editor_tool": "Trim",
        "burn_captions": True,
    }
    for k, v in d.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def err(msg: str) -> None:
    st.error(msg)
    logger.error(msg)


def save_upload(uploaded, prefix: str = "upload") -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.name).suffix.lower() or ".mp4"
    dest = TEMP_DIR / f"{prefix}_{safe_filename(Path(uploaded.name).stem) or prefix}{suffix}"
    dest.write_bytes(uploaded.getbuffer())
    return dest


def reset_pipeline() -> None:
    for k in ("video_info", "transcription", "candidate", "clip_path", "content_pkg", "export_path"):
        st.session_state[k] = None
    st.session_state.editor_open = False
    st.session_state.editor_history = None


st.markdown(
    '<div class="vc-header"><div class="vc-logo">VIDEO <span>CLIPPER</span></div>'
    '<div class="vc-muted">Generate · Edit · Export</div></div>',
    unsafe_allow_html=True,
)

ok_ffmpeg, _ = check_ffmpeg()
try:
    require_api_key()
    groq_ok = True
except RuntimeError:
    groq_ok = False
ost = get_status(DEFAULT_VISION_MODEL)

a, b, c = st.columns(3)
a.markdown(f'<div class="vc-card"><div class="vc-label">FFmpeg</div><div class="vc-value">{"OK" if ok_ffmpeg else "—"}</div></div>', unsafe_allow_html=True)
b.markdown(f'<div class="vc-card"><div class="vc-label">Groq</div><div class="vc-value">{"Ready" if groq_ok else "Optional"}</div></div>', unsafe_allow_html=True)
c.markdown(f'<div class="vc-card"><div class="vc-label">Ollama</div><div class="vc-value">{"On" if ost.ready else "Off"}</div></div>', unsafe_allow_html=True)

st.session_state.prefer_local = st.checkbox("IA local (sem limite Groq)", value=st.session_state.prefer_local)

st.markdown('<div class="vc-section">1 · Source</div>', unsafe_allow_html=True)
t1, t2 = st.tabs(["Arquivo", "URL"])
with t1:
    up = st.file_uploader("Vídeo", type=["mp4", "mov", "mkv", "webm"], label_visibility="collapsed")
    if up is not None:
        name = safe_filename(up.name)
        if st.session_state.video_name != name or not (st.session_state.video_path and Path(st.session_state.video_path).exists()):
            if st.session_state.video_path:
                cleanup_file(Path(st.session_state.video_path))
            st.session_state.video_path = str(save_upload(up))
            st.session_state.video_name = name
            reset_pipeline()
with t2:
    url = st.text_input("URL", value=st.session_state.source_url)
    q = st.selectbox("Qualidade", list(QUALITY_PRESETS.keys()), index=2)
    if st.button("Baixar", use_container_width=True) and url.strip():
        with st.spinner("Baixando…"):
            try:
                path = download_video(url.strip(), quality=q)
                st.session_state.video_path = str(path)
                st.session_state.video_name = path.name
                st.session_state.source_url = url.strip()
                reset_pipeline()
                st.rerun()
            except Exception as e:
                err(str(e))

vpath = Path(st.session_state.video_path) if st.session_state.video_path else None
if vpath and not vpath.exists():
    vpath = None

if vpath:
    if st.session_state.video_info is None:
        st.session_state.video_info = get_video_info(vpath)
    info: VideoInfo = st.session_state.video_info
    st.markdown('<div class="vc-section">2 · Clip Generator</div>', unsafe_allow_html=True)
    st.video(str(vpath))
    if not ok_ffmpeg:
        st.stop()
    if st.button("Encontrar melhor clipe", type="primary", use_container_width=True):
        st.session_state.candidate = None
        st.session_state.clip_path = None
        st.session_state.editor_open = False
        status = st.status("…", expanded=True)
        try:
            prefer = st.session_state.prefer_local
            status.write("Transcrevendo…")
            tr, audio = transcribe_video(vpath, prefer_local=prefer)
            st.session_state.transcription = tr
            cleanup_file(audio)
            status.write("Selecionando trecho…")
            cand = analyze_best_clip(tr, info.duration, prefer_local=prefer)
            st.session_state.candidate = cand
            st.session_state.manual_start, st.session_state.manual_end = cand.start, cand.end
            status.update(label="OK", state="complete")
        except Exception as e:
            status.update(label="Erro", state="error")
            err(str(e))

if st.session_state.candidate and st.session_state.video_info:
    cand: ClipCandidate = st.session_state.candidate
    info = st.session_state.video_info
    max_dur = float(info.duration) or 3600.0
    st.markdown(f'<div class="vc-card"><div class="vc-label">Trecho · score {cand.score}</div>'
                f'<p class="vc-muted">{cand.reason}</p></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    ns = c1.slider("Início", 0.0, max(0.1, max_dur - 1), float(st.session_state.manual_start), 0.1)
    ne = c2.slider("Fim", 0.1, max_dur, float(st.session_state.manual_end), 0.1)
    if ne <= ns:
        ne = min(ns + 30, max_dur)
    if ne - ns > MAX_CLIP_DURATION:
        ne = ns + MAX_CLIP_DURATION
    st.session_state.manual_start, st.session_state.manual_end = ns, ne
    if st.button("Gerar clipe", type="primary", use_container_width=True):
        OUTPUT_DIR.mkdir(exist_ok=True)
        out = OUTPUT_DIR / generate_output_filename()
        with st.spinner("Cortando…"):
            try:
                cut_video(Path(st.session_state.video_path), ns, ne, out, mode="fast")
                st.session_state.clip_path = str(out)
                st.session_state.editor_open = False
                st.session_state.editor_history = None
                st.session_state.export_path = None
                st.success("Clipe OK")
            except Exception as e:
                err(str(e))

if st.session_state.clip_path and Path(st.session_state.clip_path).exists():
    clip_p = Path(st.session_state.clip_path)
    st.markdown('<div class="vc-section">3 · Clipe</div>', unsafe_allow_html=True)
    st.video(str(clip_p))
    with clip_p.open("rb") as f:
        st.download_button("Baixar bruto", f.read(), file_name=clip_p.name, mime="video/mp4", use_container_width=True)

    if st.button("Abrir editor", type="primary", use_container_width=True):
        meta = get_video_info(clip_p)
        tr = st.session_state.transcription
        proj = new_project_from_clip(
            clip_p,
            duration=meta.duration or max(1.0, float(st.session_state.manual_end - st.session_state.manual_start)),
            fps=meta.fps or 30.0,
            name=clip_p.stem,
            segments=tr.segments if tr else [],
            clip_start_abs=float(st.session_state.manual_start),
        )
        st.session_state.editor_history = HistoryStack(proj)
        st.session_state.editor_open = True
        st.session_state.export_path = None
        st.rerun()

    if st.session_state.editor_open and st.session_state.editor_history is not None:
        hist: HistoryStack = st.session_state.editor_history
        state = hist.current

        st.markdown('<div class="vc-section">4 · Editor</div>', unsafe_allow_html=True)

        # Top bar
        tb1, tb2, tb3, tb4, tb5 = st.columns([1, 1, 1, 1, 2])
        with tb1:
            if st.button("Undo", disabled=not hist.can_undo(), use_container_width=True):
                hist.undo()
                st.rerun()
        with tb2:
            if st.button("Redo", disabled=not hist.can_redo(), use_container_width=True):
                hist.redo()
                st.rerun()
        with tb3:
            if st.button("−1f", use_container_width=True):
                hist.push(frame_step(state, -1))
                st.rerun()
        with tb4:
            if st.button("+1f", use_container_width=True):
                hist.push(frame_step(state, +1))
                st.rerun()
        with tb5:
            st.caption(f"{state.timeline_duration:.2f}s · {state.aspect.value} · style={state.caption_style.name}")

        # Timeline visualization
        st.markdown(render_timeline_html(state), unsafe_allow_html=True)

        ph = st.slider(
            "Playhead (s)",
            0.0,
            max(0.01, state.timeline_duration),
            float(state.playhead),
            1.0 / max(state.fps, 1),
            key="playhead_slider",
        )
        if abs(ph - state.playhead) > 1e-4:
            hist.push(set_playhead(state, ph))
            st.rerun()

        # Tools + inspector
        tools = ["Trim", "Split", "Crop", "Audio", "Captions", "Text", "Export"]
        st.session_state.editor_tool = st.radio(
            "Tool", tools, horizontal=True,
            index=tools.index(st.session_state.editor_tool) if st.session_state.editor_tool in tools else 0,
            label_visibility="collapsed",
        )
        tool = st.session_state.editor_tool

        left, right = st.columns([1.2, 1])
        with left:
            st.video(str(clip_p))  # preview source; final after export

        with right:
            if tool == "Trim":
                tr_s = st.number_input("Início fonte (s)", 0.0, state.source_duration, float(state.playable_range.start), 0.05)
                tr_e = st.number_input("Fim fonte (s)", 0.0, state.source_duration, float(state.playable_range.end), 0.05)
                if st.button("Aplicar trim", use_container_width=True) and tr_e > tr_s:
                    hist.push(apply_trim(state, tr_s, tr_e))
                    st.rerun()

            elif tool == "Split":
                st.caption(f"Split no playhead ({state.playhead:.2f}s)")
                s1, s2 = st.columns(2)
                if s1.button("Manter esquerda", use_container_width=True):
                    hist.push(apply_split_keep_left(state, state.playhead))
                    st.rerun()
                if s2.button("Manter direita", use_container_width=True):
                    hist.push(apply_split_keep_right(state, state.playhead))
                    st.rerun()

            elif tool == "Crop":
                labels = {AspectRatio.VERTICAL_9_16: "9:16", AspectRatio.SQUARE_1_1: "1:1", AspectRatio.LANDSCAPE_16_9: "16:9"}
                choice = st.radio("Formato", list(AspectRatio), format_func=lambda x: labels[x], horizontal=True,
                                 index=list(AspectRatio).index(state.aspect))
                if choice != state.aspect:
                    hist.push(set_aspect(state, choice))
                    st.rerun()
                z = st.slider("Zoom", 1.0, 3.0, float(state.crop.zoom), 0.05)
                cx = st.slider("X", 0.0, 1.0, float(state.crop.center_x), 0.01)
                cy = st.slider("Y", 0.0, 1.0, float(state.crop.center_y), 0.01)
                if st.button("Aplicar enquadramento", use_container_width=True):
                    hist.push(set_crop(state, CropSettings(zoom=z, center_x=cx, center_y=cy)))
                    st.rerun()

            elif tool == "Audio":
                vol = st.slider("Volume", 0.0, 2.0, float(state.audio.volume), 0.05)
                muted = st.checkbox("Mudo", state.audio.muted)
                fi = st.number_input("Fade in", 0.0, 5.0, float(state.audio.fade_in), 0.1)
                fo = st.number_input("Fade out", 0.0, 5.0, float(state.audio.fade_out), 0.1)
                if st.button("Aplicar áudio", use_container_width=True):
                    s = state.clone()
                    s.audio = AudioSettings(volume=vol, muted=muted, fade_in=fi, fade_out=fo)
                    hist.push(s)
                    st.rerun()

            elif tool == "Captions":
                style_names = list(STYLES.keys())
                cur = state.caption_style.name if state.caption_style.name in style_names else "clean"
                style = st.selectbox("Estilo", style_names, index=style_names.index(cur))
                if style != state.caption_style.name:
                    hist.push(set_caption_style(state, style))
                    st.rerun()
                st.session_state.burn_captions = st.checkbox("Queimar no export", value=st.session_state.burn_captions)
                st.caption(f"{len(state.captions)} legendas")
                for i, cap in enumerate(state.captions[:25]):
                    nt = st.text_input(f"{cap.start:.1f}–{cap.end:.1f}", cap.text, key=f"c_{cap.id}")
                    if nt != cap.text:
                        s = state.clone()
                        s.captions[i].text = nt
                        hist.push(s)
                        st.rerun()

            elif tool == "Text":
                tx = st.text_input("Texto", "TÍTULO")
                ty = st.slider("Posição Y", 0.0, 1.0, 0.12)
                ts = st.slider("Tamanho", 24, 96, 48)
                if st.button("Adicionar texto", use_container_width=True):
                    hist.push(add_text_overlay(state, tx, start=0, end=state.timeline_duration, y=ty, font_size=ts))
                    st.rerun()
                for t in state.texts:
                    st.caption(f"{t.start:.1f}–{t.end:.1f}s · {t.text}")

            elif tool == "Export":
                st.caption("Render FFmpeg com todas as decisões do projeto.")
                if st.button("EXPORTAR MP4", type="primary", use_container_width=True):
                    OUTPUT_DIR.mkdir(exist_ok=True)
                    out = OUTPUT_DIR / generate_output_filename(prefix="final")
                    bar = st.progress(0.0)

                    def cb(p, msg):
                        bar.progress(min(1.0, p), text=msg)

                    try:
                        run_export(hist.current, out, burn_captions=st.session_state.burn_captions, progress=cb)
                        st.session_state.export_path = str(out)
                        st.success("Export OK")
                    except Exception as e:
                        err(str(e))

        if st.session_state.export_path and Path(st.session_state.export_path).exists():
            ep = Path(st.session_state.export_path)
            st.markdown('<div class="vc-section">Export</div>', unsafe_allow_html=True)
            st.video(str(ep))
            with ep.open("rb") as f:
                st.download_button("Baixar final", f.read(), file_name=ep.name, mime="video/mp4", use_container_width=True)

            st.markdown('<div class="vc-section">AI Content Advisor</div>', unsafe_allow_html=True)
            if ost.ready and st.button("Analisar (local)", use_container_width=True):
                tr = st.session_state.transcription
                with st.spinner("Ollama…"):
                    try:
                        st.session_state.content_pkg = analyze_content(
                            ep, hist.current.timeline_duration,
                            segments=tr.segments if tr else [],
                            full_text=tr.text if tr else "",
                            use_vision=True, max_frames=2,
                        )
                    except Exception as e:
                        err(str(e))
            pkg: Optional[ContentPackage] = st.session_state.content_pkg
            if pkg:
                st.write(pkg.context.summary)
                st.markdown(f"**{pkg.title.primary}**")
                st.download_button("Baixar textos", pkg.copy_all_text().encode(), "content.txt", "text/plain", use_container_width=True)

st.caption("Editor local · decisões em ProjectState · FFmpeg só no export.")
