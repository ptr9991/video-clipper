"""Video Clipper — simple flow: analyze → pick → 9:16 editor → export."""

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
    HistoryStack,
    apply_trim,
    new_project_from_clip,
    render_timeline_html,
    run_export,
)
from src.preset import CANVAS_H, CANVAS_W, DEFAULT
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
.block-container{padding-top:1rem!important;max-width:1100px;}
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
    d: dict[str, Any] = {
        "video_path": None, "video_name": None, "video_info": None,
        "video_hash": None, "transcription": None,
        "candidates": [], "clip_path": None, "source_url": "",
        "prefer_local": False, "top_n": 5,
        "editor_open": False, "editor_history": None,
        "export_path": None, "perf_log": [],
    }
    for k, v in d.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def err(msg: str) -> None:
    st.error(msg)
    logger.error(msg)


def perf(label: str, seconds: float) -> None:
    st.session_state.perf_log.append(f"{label}: {seconds:.2f}s")


def save_upload(uploaded) -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.name).suffix.lower() or ".mp4"
    dest = TEMP_DIR / f"upload_{safe_filename(Path(uploaded.name).stem)}{suffix}"
    dest.write_bytes(uploaded.getbuffer())
    return dest


def reset_analysis() -> None:
    for k in ("transcription", "candidates", "clip_path", "export_path"):
        st.session_state[k] = None if k != "candidates" else []
    st.session_state.editor_open = False
    st.session_state.editor_history = None
    st.session_state.perf_log = []


def transcription_from_cache(data: dict) -> TranscriptionResult:
    segs = [Segment(float(s["start"]), float(s["end"]), str(s["text"])) for s in data.get("segments") or []]
    words = [WordStamp.from_dict(w) for w in data.get("words") or []]
    return TranscriptionResult(
        text=str(data.get("text", "")), segments=segs, words=words,
        language=data.get("language"), duration=data.get("duration"),
        source=str(data.get("source", "cache")),
    )


def transcription_to_cache(tr: TranscriptionResult) -> dict:
    return {
        "text": tr.text, "language": tr.language, "duration": tr.duration, "source": tr.source,
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in tr.segments],
        "words": [w.to_dict() for w in (tr.words or [])],
    }


def open_editor(cand: ClipCandidate, idx: int) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / generate_output_filename(prefix=f"clip{idx+1}")
    with st.spinner("Preparando 9:16 + legendas padrao…"):
        cut_video(Path(st.session_state.video_path), cand.start, cand.end, out, mode="fast")
    st.session_state.clip_path = str(out)
    meta = get_video_info(out)
    tr = st.session_state.transcription
    proj = new_project_from_clip(
        out,
        duration=meta.duration or cand.duration,
        fps=meta.fps or 30.0,
        name=out.stem,
        segments=tr.segments if tr else [],
        clip_start_abs=cand.start,
        transcription=tr,
    )
    # lock standard
    proj.aspect = AspectRatio.VERTICAL_9_16
    st.session_state.editor_history = HistoryStack(proj)
    st.session_state.editor_open = True
    st.session_state.export_path = None


st.markdown(
    '<div class="vc-header"><div class="vc-logo">VIDEO <span>CLIPPER</span></div>'
    f'<div class="vc-muted">Default · {CANVAS_W}x{CANVAS_H} · font {DEFAULT.font_size}</div></div>',
    unsafe_allow_html=True,
)

ok_ffmpeg, _ = check_ffmpeg()
st.session_state.prefer_local = st.checkbox("IA local (CPU)", value=False)
st.session_state.top_n = st.select_slider("Cortes", options=[5, 10, 15], value=5)

st.markdown('<div class="vc-section">1 · Video</div>', unsafe_allow_html=True)
t1, t2 = st.tabs(["Arquivo", "URL"])
with t1:
    up = st.file_uploader("Video", type=["mp4", "mov", "mkv", "webm"], label_visibility="collapsed")
    if up is not None:
        name = safe_filename(up.name)
        if st.session_state.video_name != name or not (st.session_state.video_path and Path(str(st.session_state.video_path)).exists()):
            st.session_state.video_path = str(save_upload(up))
            st.session_state.video_name = name
            st.session_state.video_info = None
            st.session_state.video_hash = None
            reset_analysis()
with t2:
    url = st.text_input("URL", value=st.session_state.source_url or "")
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
    if st.session_state.video_info is None:
        t0 = time.time()
        try:
            st.session_state.video_info = get_video_info(vpath)
            perf("Metadata", time.time() - t0)
        except Exception as e:
            err(str(e))
            st.stop()
    info: VideoInfo = st.session_state.video_info
    st.caption(f"{st.session_state.video_name} · {format_timestamp(info.duration)} · {info.resolution}")
    st.video(str(vpath))

    if not ok_ffmpeg:
        st.warning("FFmpeg necessario")
        st.stop()

    if st.button("Analisar melhores cortes", type="primary", use_container_width=True):
        reset_analysis()
        status = st.status("…", expanded=True)
        prefer = st.session_state.prefer_local
        n = int(st.session_state.top_n)
        try:
            if not st.session_state.video_hash:
                status.write("Hash…")
                t0 = time.time()
                st.session_state.video_hash = file_sha256(vpath)
                perf("Hash", time.time() - t0)
            vhash = st.session_state.video_hash

            cached = load_json(vhash, "transcription")
            if cached and not cached.get("words"):
                cached = None
            if cached:
                status.write("Transcricao (cache)")
                tr = transcription_from_cache(cached)
            else:
                status.write("Transcrevendo (1x)…")
                t0 = time.time()
                tr, audio = transcribe_video(vpath, prefer_local=prefer)
                cleanup_file(audio)
                perf("Transcription", time.time() - t0)
                save_json(vhash, "transcription", transcription_to_cache(tr))
            st.session_state.transcription = tr

            ck = f"analysis_n{n}_{'local' if prefer else 'groq'}"
            ca = load_json(vhash, ck)
            if ca and ca.get("candidates"):
                status.write("Analise (cache)")
                cands = [ClipCandidate.from_dict(c) for c in ca["candidates"]]
            else:
                status.write(f"TOP {n}…")
                t0 = time.time()
                cands = analyze_best_clips(tr, info.duration, n=n, prefer_local=prefer)
                perf("Analysis", time.time() - t0)
                save_json(vhash, ck, {"candidates": [c.to_dict() for c in cands]})

            for c in cands:
                extract_thumbnail(vpath, c.start + min(2.0, c.duration / 3), vhash)
            st.session_state.candidates = cands
            status.update(label=f"{len(cands)} cortes", state="complete")
        except Exception as e:
            status.update(label="Erro", state="error")
            err(str(e))

if st.session_state.candidates:
    st.markdown('<div class="vc-section">2 · Escolha um corte</div>', unsafe_allow_html=True)
    if st.session_state.perf_log:
        st.caption(" · ".join(st.session_state.perf_log))
    vhash = st.session_state.video_hash or "x"
    cols = st.columns(min(3, len(st.session_state.candidates)))
    for i, cand in enumerate(st.session_state.candidates):
        with cols[i % len(cols)]:
            th = extract_thumbnail(Path(st.session_state.video_path), cand.start + 1.0, vhash)
            if th.exists() and th.stat().st_size > 100:
                st.image(str(th), use_container_width=True)
            st.markdown(f"**#{i+1}** `{format_timestamp(cand.start)}` · score {cand.score}")
            if st.button("Editar", key=f"e{i}", type="primary", use_container_width=True):
                open_editor(cand, i)
                st.rerun()

# ── Editor (minimal) ─────────────────────────────────────
if st.session_state.editor_open and st.session_state.editor_history and st.session_state.clip_path:
    hist: HistoryStack = st.session_state.editor_history
    state = hist.current
    clip_p = Path(st.session_state.clip_path)

    st.markdown(
        f'<div class="vc-section">3 · Editor · 9:16 · {len(state.captions)} legendas · '
        f'padrao font={DEFAULT.font_size}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(render_timeline_html(state), unsafe_allow_html=True)

    left, right = st.columns([1.3, 1])
    with left:
        st.video(str(clip_p))
    with right:
        st.caption("Padrao VideoClipper — sem seletor de estilo.")
        st.write(f"Formato: **{CANVAS_W}×{CANVAS_H}** · Legendas: **ON**")
        # optional fine trim only
        with st.expander("Ajuste fino (opcional)"):
            a = st.number_input("Início", 0.0, state.source_duration, float(state.playable_range.start), 0.05)
            b = st.number_input("Fim", 0.0, state.source_duration, float(state.playable_range.end), 0.05)
            if st.button("Aplicar trim") and b > a:
                hist.push(apply_trim(state, a, b))
                st.rerun()
            for i, cap in enumerate(state.captions[:15]):
                nt = st.text_input(f"{cap.start:.1f}s", cap.text.replace("\\N", " / "), key=f"c{cap.id}")
                # store without breaking ASS newline marker if user edits simply
                cleaned = nt.replace(" / ", "\\N")
                if cleaned != cap.text:
                    s = state.clone()
                    s.captions[i].text = cleaned
                    hist.push(s)
                    st.rerun()

        if st.button("EXPORTAR 1080×1920", type="primary", use_container_width=True):
            OUTPUT_DIR.mkdir(exist_ok=True)
            out = OUTPUT_DIR / generate_output_filename(prefix="final")
            bar = st.progress(0.0)

            def cb(p, msg):
                bar.progress(min(1.0, p), text=msg)

            try:
                run_export(hist.current, out, burn_captions=True, progress=cb)
                st.session_state.export_path = str(out)
                st.success("Export OK")
            except Exception as e:
                err(str(e))

    if st.session_state.export_path and Path(st.session_state.export_path).exists():
        ep = Path(st.session_state.export_path)
        st.video(str(ep))
        with ep.open("rb") as f:
            st.download_button("Baixar MP4", f.read(), file_name=ep.name, mime="video/mp4", use_container_width=True)

st.caption("Um padrao · 9:16 · legendas consistentes · sem estilos extras")
