"""Video Clipper — 9:16 editor + word-synced short-form captions."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from src.cache import file_sha256, load_json, save_json
from src.captions import WordStamp
from src.clip_analyzer import ClipCandidate, analyze_best_clips
from src.config import OUTPUT_DIR, TEMP_DIR, check_ffmpeg, require_api_key
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
)
from src.editor.caption_styles import STYLES
from src.local_ai.content_advisor import ContentPackage, analyze_content
from src.ollama_manager import DEFAULT_VISION_MODEL, get_status
from src.thumbnails import extract_thumbnail
from src.transcription import Segment, TranscriptionResult, transcribe_video
from src.utils import cleanup_file, format_timestamp, generate_output_filename, safe_filename
from src.video_processor import VideoInfo, cut_video, get_video_info

logger = logging.getLogger("video_clipper.app")

st.set_page_config(page_title="Video Clipper", page_icon="▶", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: Inter, system-ui, sans-serif; }
.stApp { background:#080808; color:#f5f5f5; }
#MainMenu, footer, header { visibility:hidden; height:0; }
[data-testid="stToolbar"]{display:none;}
.block-container{padding-top:1rem!important;max-width:1280px;}
.vc-header{display:flex;justify-content:space-between;border-bottom:1px solid #222;padding-bottom:.7rem;margin-bottom:1rem;}
.vc-logo{font-weight:700;} .vc-logo span{color:#c8f542;}
.vc-section{font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:#8a8a8a;margin:1.2rem 0 .5rem;}
.vc-card{background:#111;border:1px solid #222;border-radius:10px;padding:1rem;margin-bottom:.75rem;}
.vc-muted{color:#8a8a8a;font-size:.85rem;}
.stButton>button{border-radius:8px!important;font-weight:600!important;background:#161616!important;border:1px solid #2a2a2a!important;color:#f5f5f5!important;}
.stButton>button[kind="primary"]{background:#c8f542!important;color:#080808!important;border-color:#c8f542!important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def init_state() -> None:
    defaults: dict[str, Any] = {
        "video_path": None, "video_name": None, "video_info": None,
        "video_hash": None, "transcription": None,
        "candidates": [], "selected_idx": None,
        "clip_path": None, "source_url": "",
        "content_pkg": None, "prefer_local": False,
        "top_n": 5, "editor_open": False, "editor_history": None,
        "export_path": None, "perf_log": [], "burn_captions": True,
        "editor_tool": "Captions",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def err(msg: str) -> None:
    st.error(msg)
    logger.error(msg)


def perf(label: str, seconds: float) -> None:
    st.session_state.perf_log.append(f"{label}: {seconds:.2f}s")
    logger.info("PERF %s %.2fs", label, seconds)


def save_upload(uploaded, prefix: str = "upload") -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.name).suffix.lower() or ".mp4"
    dest = TEMP_DIR / f"{prefix}_{safe_filename(Path(uploaded.name).stem) or prefix}{suffix}"
    dest.write_bytes(uploaded.getbuffer())
    return dest


def reset_analysis() -> None:
    st.session_state.transcription = None
    st.session_state.candidates = []
    st.session_state.selected_idx = None
    st.session_state.clip_path = None
    st.session_state.editor_open = False
    st.session_state.editor_history = None
    st.session_state.export_path = None
    st.session_state.content_pkg = None
    st.session_state.perf_log = []


def transcription_from_cache(data: dict) -> TranscriptionResult:
    segs = [
        Segment(float(s["start"]), float(s["end"]), str(s["text"]))
        for s in data.get("segments") or []
    ]
    words = [WordStamp.from_dict(w) for w in data.get("words") or []]
    return TranscriptionResult(
        text=str(data.get("text", "")),
        segments=segs,
        words=words,
        language=data.get("language"),
        duration=data.get("duration"),
        source=str(data.get("source", "cache")),
    )


def transcription_to_cache(tr: TranscriptionResult) -> dict:
    return {
        "text": tr.text,
        "language": tr.language,
        "duration": tr.duration,
        "source": tr.source,
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in tr.segments],
        "words": [w.to_dict() for w in (tr.words or [])],
    }


def open_editor_for_candidate(cand: ClipCandidate, idx: int) -> None:
    """Cut once + open 9:16 project with word-synced captions."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / generate_output_filename(prefix=f"clip{idx+1}")
    with st.spinner("Preparando editor 9:16 + legendas…"):
        cut_video(Path(st.session_state.video_path), cand.start, cand.end, out, mode="fast")
    st.session_state.clip_path = str(out)
    meta = get_video_info(out)
    tr: Optional[TranscriptionResult] = st.session_state.transcription
    proj = new_project_from_clip(
        out,
        duration=meta.duration or cand.duration,
        fps=meta.fps or 30.0,
        name=out.stem,
        segments=tr.segments if tr else [],
        clip_start_abs=cand.start,
        transcription=tr,
    )
    # force 9:16 + captions on
    assert proj.aspect == AspectRatio.VERTICAL_9_16
    st.session_state.editor_history = HistoryStack(proj)
    st.session_state.editor_open = True
    st.session_state.export_path = None
    st.session_state.burn_captions = True
    st.session_state.editor_tool = "Captions"


st.markdown(
    '<div class="vc-header"><div class="vc-logo">VIDEO <span>CLIPPER</span></div>'
    '<div class="vc-muted">9:16 · Word captions · Safe zone</div></div>',
    unsafe_allow_html=True,
)

ok_ffmpeg, _ = check_ffmpeg()
ost = get_status(DEFAULT_VISION_MODEL)

st.session_state.prefer_local = st.checkbox(
    "IA local (CPU) — so se Groq no limite", value=st.session_state.prefer_local
)
st.session_state.top_n = st.select_slider("Melhores momentos", options=[5, 10, 15], value=st.session_state.top_n)

st.markdown('<div class="vc-section">Source</div>', unsafe_allow_html=True)
tab_f, tab_u = st.tabs(["Arquivo", "URL"])
with tab_f:
    up = st.file_uploader("Video", type=["mp4", "mov", "mkv", "webm"], label_visibility="collapsed")
    if up is not None:
        name = safe_filename(up.name)
        if st.session_state.video_name != name or not (st.session_state.video_path and Path(st.session_state.video_path).exists()):
            if st.session_state.video_path:
                cleanup_file(Path(st.session_state.video_path))
            st.session_state.video_path = str(save_upload(up))
            st.session_state.video_name = name
            st.session_state.video_info = None
            st.session_state.video_hash = None
            reset_analysis()
with tab_u:
    url = st.text_input("URL", value=st.session_state.source_url)
    q = st.selectbox("Qualidade", list(QUALITY_PRESETS.keys()), index=2)
    if st.button("Baixar", use_container_width=True) and url.strip():
        with st.spinner("Download…"):
            try:
                path = download_video(url.strip(), quality=q)
                st.session_state.video_path = str(path)
                st.session_state.video_name = path.name
                st.session_state.source_url = url.strip()
                st.session_state.video_info = None
                st.session_state.video_hash = None
                reset_analysis()
                st.rerun()
            except Exception as e:
                err(str(e))

vpath = Path(st.session_state.video_path) if st.session_state.video_path else None
if vpath and not vpath.exists():
    vpath = None

if vpath:
    t0 = time.time()
    if st.session_state.video_info is None:
        st.session_state.video_info = get_video_info(vpath)
        perf("Metadata", time.time() - t0)
    info: VideoInfo = st.session_state.video_info

    st.markdown(
        f'<div class="vc-card"><span class="vc-muted">{st.session_state.video_name}</span> · '
        f'{format_timestamp(info.duration)} · {info.resolution if info.width else "—"}</div>',
        unsafe_allow_html=True,
    )
    st.video(str(vpath))

    if not ok_ffmpeg:
        st.warning("FFmpeg necessario")
        st.stop()

    try:
        require_api_key()
        has_groq = True
    except RuntimeError:
        has_groq = False

    if not has_groq and not st.session_state.prefer_local:
        st.info("Sem chave Groq: marque IA local ou configure a API key.")

    if st.button("Encontrar melhores momentos", type="primary", use_container_width=True):
        reset_analysis()
        status = st.status("Analisando…", expanded=True)
        prefer = st.session_state.prefer_local
        n = int(st.session_state.top_n)
        try:
            if not st.session_state.video_hash:
                status.write("Hash…")
                t0 = time.time()
                st.session_state.video_hash = file_sha256(vpath)
                perf("Hash", time.time() - t0)
            vhash = st.session_state.video_hash

            cached_tr = load_json(vhash, "transcription")
            # Invalidate old cache without words
            if cached_tr and not cached_tr.get("words"):
                cached_tr = None

            if cached_tr:
                status.write("Transcricao (cache com words)")
                tr = transcription_from_cache(cached_tr)
            else:
                status.write("Etapa 1/3 · Transcricao word-level")
                t0 = time.time()
                tr, audio = transcribe_video(vpath, prefer_local=prefer)
                cleanup_file(audio)
                perf("Transcription", time.time() - t0)
                save_json(vhash, "transcription", transcription_to_cache(tr))
            st.session_state.transcription = tr
            status.write(f"Words: {len(tr.words)} · Segments: {len(tr.segments)}")

            cache_key = f"analysis_n{n}_{'local' if prefer else 'groq'}"
            cached_an = load_json(vhash, cache_key)
            if cached_an and cached_an.get("candidates"):
                status.write("Analise (cache)")
                cands = [ClipCandidate.from_dict(c) for c in cached_an["candidates"]]
            else:
                status.write(f"Etapa 2/3 · TOP {n}")
                t0 = time.time()
                cands = analyze_best_clips(tr, info.duration, n=n, prefer_local=prefer)
                perf("Analysis", time.time() - t0)
                save_json(vhash, cache_key, {"candidates": [c.to_dict() for c in cands]})

            status.write("Etapa 3/3 · Thumbs")
            t0 = time.time()
            for c in cands:
                extract_thumbnail(vpath, c.start + min(2.0, c.duration / 3), vhash)
            perf("Thumbnails", time.time() - t0)

            st.session_state.candidates = cands
            status.update(label=f"{len(cands)} momentos", state="complete")
        except Exception as e:
            status.update(label="Erro", state="error")
            err(str(e))

if st.session_state.candidates:
    st.markdown('<div class="vc-section">Melhores clipes</div>', unsafe_allow_html=True)
    if st.session_state.perf_log:
        st.caption(" · ".join(st.session_state.perf_log))

    vhash = st.session_state.video_hash or "x"
    cols = st.columns(min(3, len(st.session_state.candidates)))
    for i, cand in enumerate(st.session_state.candidates):
        with cols[i % len(cols)]:
            thumb = extract_thumbnail(
                Path(st.session_state.video_path),
                cand.start + min(2.0, cand.duration / 3),
                vhash,
            )
            if thumb.exists() and thumb.stat().st_size > 100:
                st.image(str(thumb), use_container_width=True)
            st.markdown(
                f"**#{i+1}** · `{format_timestamp(cand.start)} → {format_timestamp(cand.end)}`  \n"
                f"Score **{cand.score}** · {cand.duration:.0f}s"
            )
            if st.button("Abrir editor 9:16", key=f"ed_{i}", type="primary", use_container_width=True):
                open_editor_for_candidate(cand, i)
                st.rerun()

if st.session_state.editor_open and st.session_state.editor_history and st.session_state.clip_path:
    hist: HistoryStack = st.session_state.editor_history
    state = hist.current
    clip_p = Path(st.session_state.clip_path)

    st.markdown(
        f'<div class="vc-section">Editor · {state.aspect.value} · '
        f'{len(state.captions)} legendas · style={state.caption_style.name}</div>',
        unsafe_allow_html=True,
    )

    u1, u2, u3, u4 = st.columns(4)
    if u1.button("Undo", disabled=not hist.can_undo()):
        hist.undo()
        st.rerun()
    if u2.button("Redo", disabled=not hist.can_redo()):
        hist.redo()
        st.rerun()
    if u3.button("-1f"):
        hist.push(frame_step(state, -1))
        st.rerun()
    if u4.button("+1f"):
        hist.push(frame_step(state, +1))
        st.rerun()

    st.markdown(render_timeline_html(state), unsafe_allow_html=True)

    tools = ["Captions", "Trim", "Split", "Crop", "Audio", "Text", "Export"]
    st.session_state.editor_tool = st.radio(
        "Tool", tools, horizontal=True, label_visibility="collapsed",
        index=tools.index(st.session_state.editor_tool) if st.session_state.editor_tool in tools else 0,
    )
    tool = st.session_state.editor_tool

    left, right = st.columns([1.2, 1])
    with left:
        st.caption("Preview do clipe (export aplica 9:16 + legendas)")
        st.video(str(clip_p))
    with right:
        if tool == "Captions":
            st.caption("Legendas word-level convertidas para o tempo do clipe (0s = inicio do corte).")
            st.session_state.burn_captions = st.checkbox("Queimar no export", True)
            style_opts = ["default_shorts"] + list(STYLES.keys())
            cur = state.caption_style.name
            if cur not in style_opts:
                cur = "default_shorts"
            style = st.selectbox("Estilo", style_opts, index=style_opts.index(cur))
            if style != state.caption_style.name:
                hist.push(set_caption_style(state, style))
                st.rerun()
            for i, cap in enumerate(state.captions[:30]):
                nt = st.text_input(
                    f"{cap.start:.2f}–{cap.end:.2f}s",
                    cap.text,
                    key=f"cap{cap.id}",
                )
                if nt != cap.text:
                    s = state.clone()
                    s.captions[i].text = nt
                    hist.push(s)
                    st.rerun()
        elif tool == "Trim":
            a = st.number_input("Start", 0.0, state.source_duration, float(state.playable_range.start), 0.05)
            b = st.number_input("End", 0.0, state.source_duration, float(state.playable_range.end), 0.05)
            if st.button("Aplicar trim") and b > a:
                hist.push(apply_trim(state, a, b))
                st.rerun()
        elif tool == "Split":
            if st.button("Manter esquerda"):
                hist.push(apply_split_keep_left(state, state.playhead))
                st.rerun()
            if st.button("Manter direita"):
                hist.push(apply_split_keep_right(state, state.playhead))
                st.rerun()
        elif tool == "Crop":
            labels = {AspectRatio.VERTICAL_9_16: "9:16", AspectRatio.SQUARE_1_1: "1:1", AspectRatio.LANDSCAPE_16_9: "16:9"}
            ch = st.radio("Formato", list(AspectRatio), format_func=lambda x: labels[x], horizontal=True,
                          index=list(AspectRatio).index(state.aspect))
            if ch != state.aspect:
                hist.push(set_aspect(state, ch))
                st.rerun()
            z = st.slider("Zoom", 1.0, 3.0, float(state.crop.zoom), 0.05)
            cx = st.slider("X", 0.0, 1.0, float(state.crop.center_x), 0.01)
            cy = st.slider("Y", 0.0, 1.0, float(state.crop.center_y), 0.01)
            if st.button("Aplicar crop"):
                hist.push(set_crop(state, CropSettings(z, cx, cy)))
                st.rerun()
        elif tool == "Audio":
            vol = st.slider("Volume", 0.0, 2.0, float(state.audio.volume), 0.05)
            muted = st.checkbox("Mudo", state.audio.muted)
            if st.button("Aplicar audio"):
                s = state.clone()
                s.audio = AudioSettings(vol, muted, 0.0, 0.0)
                hist.push(s)
                st.rerun()
        elif tool == "Text":
            tx = st.text_input("Texto", "TITULO")
            if st.button("Add texto"):
                hist.push(add_text_overlay(state, tx, end=state.timeline_duration))
                st.rerun()
        elif tool == "Export":
            st.caption("Export: 1080x1920 · H.264 · legendas queimadas na safe zone")
            if st.button("EXPORTAR MP4", type="primary", use_container_width=True):
                OUTPUT_DIR.mkdir(exist_ok=True)
                out = OUTPUT_DIR / generate_output_filename(prefix="final")
                bar = st.progress(0.0)

                def cb(p, msg):
                    bar.progress(min(1.0, p), text=msg)

                try:
                    run_export(hist.current, out, burn_captions=True, progress=cb)
                    st.session_state.export_path = str(out)
                    st.success("Export OK · 9:16")
                except Exception as e:
                    err(str(e))

    if st.session_state.export_path and Path(st.session_state.export_path).exists():
        ep = Path(st.session_state.export_path)
        st.video(str(ep))
        with ep.open("rb") as f:
            st.download_button("Baixar final", f.read(), file_name=ep.name, mime="video/mp4", use_container_width=True)

st.caption("9:16 padrao · legendas word-level relativas ao clipe · cache invalida sem words")
