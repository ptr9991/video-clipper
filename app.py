"""
AI Video Clipper Local
Groq (speech) + FFmpeg (cut + edit: vertical, subs, webcam PiP).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import streamlit as st

from src.clip_editor import PIP_POSITIONS, EditOptions, render_edited_clip
from src.clip_analyzer import ClipCandidate, analyze_best_clip
from src.config import MAX_CLIP_DURATION, OUTPUT_DIR, TEMP_DIR, check_ffmpeg, require_api_key
from src.downloader import QUALITY_PRESETS, download_video
from src.hardware import detect_hardware
from src.transcription import Segment, TranscriptionResult, transcribe_video
from src.utils import cleanup_file, format_timestamp, generate_output_filename, safe_filename
from src.video_processor import VideoInfo, cut_video, get_video_info
from src.visual_analyzer import VisualAnalysis, analyze_clip_visual

logger = logging.getLogger("video_clipper.app")

st.set_page_config(
    page_title="AI Video Clipper Local",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
        "last_cut_error": None,
        "source_url": "",
        "visual_result": None,
        "enable_visual": False,
        "hw_info": None,
        "webcam_path": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def show_error(msg: str) -> None:
    st.error(msg)
    logger.error(msg)


def save_uploaded_file(uploaded, prefix: str = "upload") -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.name).suffix.lower() or ".mp4"
    safe = safe_filename(Path(uploaded.name).stem) or prefix
    dest = TEMP_DIR / f"{prefix}_{safe}{suffix}"
    with dest.open("wb") as f:
        f.write(uploaded.getbuffer())
    return dest


def reset_video_state() -> None:
    st.session_state.video_info = None
    st.session_state.transcription = None
    st.session_state.candidate = None
    st.session_state.clip_path = None
    st.session_state.edited_path = None
    st.session_state.last_cut_error = None
    st.session_state.visual_result = None


with st.sidebar:
    st.header("⚙️ Status")
    ok_ffmpeg, _ = check_ffmpeg()
    st.success("FFmpeg OK") if ok_ffmpeg else st.error("FFmpeg não encontrado")
    try:
        require_api_key()
        st.success("API Key Groq OK")
    except RuntimeError:
        st.warning("API Key não configurada")

    st.divider()
    if st.session_state.hw_info is None:
        try:
            st.session_state.hw_info = detect_hardware()
        except Exception:
            pass
    hw = st.session_state.hw_info
    if hw:
        st.caption(f"GPU: {hw.gpu_name} · {hw.vram_gb} GB")

    st.session_state.enable_visual = st.checkbox(
        "Revisão visual leve (opcional)", value=st.session_state.enable_visual
    )

    if st.button("🗑️ Limpar sessão"):
        for p in (st.session_state.video_path, st.session_state.webcam_path):
            if p:
                cleanup_file(Path(p))
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


st.title("🎬 AI Video Clipper Local")
st.markdown("Encontre o trecho → corte → **edite** (vertical, legendas, webcam).")

tab_upload, tab_url = st.tabs(["📁 Upload", "🔗 Link"])
with tab_upload:
    uploaded = st.file_uploader("Vídeo principal", type=["mp4", "mov", "mkv", "webm"])
    if uploaded is not None:
        name = safe_filename(uploaded.name)
        if st.session_state.video_name != name or not (
            st.session_state.video_path and Path(st.session_state.video_path).exists()
        ):
            if st.session_state.video_path:
                cleanup_file(Path(st.session_state.video_path))
            path = save_uploaded_file(uploaded)
            st.session_state.video_path = str(path)
            st.session_state.video_name = name
            reset_video_state()

with tab_url:
    url = st.text_input("URL", value=st.session_state.source_url)
    quality = st.selectbox("Qualidade", list(QUALITY_PRESETS.keys()), index=2)
    if st.button("⬇️ Baixar", type="primary"):
        if url.strip():
            with st.spinner("Baixando…"):
                try:
                    path = download_video(url.strip(), quality=quality)
                    st.session_state.video_path = str(path)
                    st.session_state.video_name = path.name
                    st.session_state.source_url = url.strip()
                    reset_video_state()
                    st.rerun()
                except Exception as e:
                    show_error(str(e))

video_path = Path(st.session_state.video_path) if st.session_state.video_path else None
if video_path and not video_path.exists():
    st.warning("Vídeo perdido. Envie de novo.")
    video_path = None

if video_path:
    if st.session_state.video_info is None:
        with st.spinner("Metadados…"):
            st.session_state.video_info = get_video_info(video_path)
    info: VideoInfo = st.session_state.video_info
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Arquivo", (st.session_state.video_name or "")[:24])
    c2.metric("MB", f"{info.size_mb:.1f}")
    c3.metric("Duração", format_timestamp(info.duration))
    c4.metric("Res", info.resolution if info.width else "—")
    st.video(str(video_path))

    if not ok_ffmpeg:
        st.stop()
    try:
        require_api_key()
    except RuntimeError as e:
        st.warning(str(e))
        st.stop()

    if st.button("🔍 Encontrar melhor clipe com IA", type="primary", use_container_width=True):
        st.session_state.candidate = None
        st.session_state.clip_path = None
        st.session_state.edited_path = None
        status = st.status("Processando…", expanded=True)
        try:
            status.write("Transcrevendo…")
            transcription, audio_path = transcribe_video(video_path)
            st.session_state.transcription = transcription
            cleanup_file(audio_path)
            status.write("Analisando…")
            cand = analyze_best_clip(transcription, video_duration=info.duration)
            st.session_state.candidate = cand
            st.session_state.manual_start = cand.start
            st.session_state.manual_end = cand.end
            status.update(label="OK", state="complete")
        except Exception as e:
            status.update(label="Erro", state="error")
            show_error(str(e))

if st.session_state.candidate and st.session_state.video_info:
    cand: ClipCandidate = st.session_state.candidate
    info = st.session_state.video_info
    st.divider()
    st.subheader("🎯 Trecho escolhido")
    a, b, c, d = st.columns(4)
    a.metric("Início", format_timestamp(cand.start))
    b.metric("Fim", format_timestamp(cand.end))
    c.metric("Duração", f"{cand.duration:.1f}s")
    d.metric("Score", f"{cand.score}/100")
    st.caption(cand.reason)

    max_dur = float(info.duration) or 3600.0
    col_a, col_b = st.columns(2)
    with col_a:
        new_start = st.slider("Início", 0.0, max(0.1, max_dur - 1), float(st.session_state.manual_start), 0.1)
    with col_b:
        new_end = st.slider("Fim", 0.1, max_dur, float(st.session_state.manual_end), 0.1)
    if new_end <= new_start:
        new_end = min(new_start + 30, max_dur)
    if new_end - new_start > MAX_CLIP_DURATION:
        new_end = new_start + MAX_CLIP_DURATION
    st.session_state.manual_start = new_start
    st.session_state.manual_end = new_end
    st.info(f"Duração: **{new_end - new_start:.1f}s**")

    mode = st.radio("Corte", ["fast", "precise"], format_func=lambda x: "Rápido" if x == "fast" else "Preciso", horizontal=True)
    if st.button("✂️ Gerar clipe", type="primary", use_container_width=True):
        out = OUTPUT_DIR / generate_output_filename()
        OUTPUT_DIR.mkdir(exist_ok=True)
        with st.spinner("Cortando…"):
            try:
                cut_video(Path(st.session_state.video_path), new_start, new_end, out, mode=mode)
                st.session_state.clip_path = str(out)
                st.session_state.edited_path = None
                st.success("Clipe gerado")
            except Exception as e:
                show_error(str(e))

if st.session_state.clip_path and Path(st.session_state.clip_path).exists():
    clip_p = Path(st.session_state.clip_path)
    st.divider()
    st.subheader("📺 Clipe")
    st.video(str(clip_p))

    # -------- EDITOR --------
    st.divider()
    st.subheader("✂️ Editor do clipe")
    st.caption("Formato vertical, legendas automáticas e posição da webcam (PiP).")

    e1, e2 = st.columns(2)
    with e1:
        vertical = st.checkbox("Formato vertical 9:16 (Shorts/Reels)", value=True)
        add_subs = st.checkbox("Legendas automáticas (da transcrição)", value=True)
        font_size = st.slider("Tamanho da legenda", 14, 36, 20)
    with e2:
        st.markdown("**Webcam / segunda câmera (opcional)**")
        cam_file = st.file_uploader("Vídeo da webcam", type=["mp4", "mov", "webm"], key="cam_up")
        if cam_file is not None:
            cam_path = save_uploaded_file(cam_file, prefix="webcam")
            st.session_state.webcam_path = str(cam_path)
        cam_pos = st.selectbox("Posição da webcam", list(PIP_POSITIONS.keys()))
        cam_scale = st.slider("Tamanho da webcam", 0.15, 0.45, 0.30, 0.01)

    if st.button("🎬 Aplicar edição", type="primary", use_container_width=True):
        tr: Optional[TranscriptionResult] = st.session_state.transcription
        segs: list[Segment] = tr.segments if tr else []
        opts = EditOptions(
            vertical_9x16=vertical,
            add_subtitles=add_subs,
            subtitle_font_size=font_size,
            webcam_path=Path(st.session_state.webcam_path) if st.session_state.webcam_path else None,
            webcam_position=cam_pos,
            webcam_scale=cam_scale,
        )
        with st.spinner("Renderizando edição (FFmpeg)…"):
            try:
                edited = render_edited_clip(
                    clip_path=clip_p,
                    options=opts,
                    segments=segs,
                    clip_start_abs=float(st.session_state.manual_start),
                    clip_end_abs=float(st.session_state.manual_end),
                )
                st.session_state.edited_path = str(edited)
                st.success("Edição concluída")
            except Exception as e:
                show_error(str(e))

    if st.session_state.edited_path and Path(st.session_state.edited_path).exists():
        ep = Path(st.session_state.edited_path)
        st.subheader("✅ Clipe editado")
        st.video(str(ep))
        with ep.open("rb") as f:
            st.download_button("⬇️ Baixar clipe editado", f.read(), file_name=ep.name, mime="video/mp4", use_container_width=True)
    else:
        with clip_p.open("rb") as f:
            st.download_button("⬇️ Baixar clipe (sem edição)", f.read(), file_name=clip_p.name, mime="video/mp4", use_container_width=True)

    if st.session_state.enable_visual:
        if st.button("Revisão visual leve"):
            try:
                st.session_state.visual_result = analyze_clip_visual(
                    clip_p,
                    max(1.0, float(st.session_state.manual_end - st.session_state.manual_start)),
                    float(st.session_state.manual_start),
                    float(st.session_state.manual_end),
                    [],
                    speech_score=float(st.session_state.candidate.score) if st.session_state.candidate else 70,
                    use_vlm=False,
                )
            except Exception as e:
                show_error(str(e))
        vr: Optional[VisualAnalysis] = st.session_state.visual_result
        if vr:
            st.caption(f"Score visual leve: {vr.overall_score}/100 ({vr.verdict})")

st.caption("Edição 100% local com FFmpeg · Groq só na fala.")
