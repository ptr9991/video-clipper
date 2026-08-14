"""
AI Video Clipper Local
Streamlit application that finds the best 30-50s clip from a long video
using Groq for transcription + analysis and FFmpeg for local cutting.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

from src.config import (
    DEBUG,
    MAX_CLIP_DURATION,
    MIN_CLIP_DURATION,
    OUTPUT_DIR,
    check_ffmpeg,
    require_api_key,
)
from src.clip_analyzer import ClipCandidate, analyze_best_clip
from src.transcription import TranscriptionResult, transcribe_video
from src.utils import (
    cleanup_file,
    format_timestamp,
    generate_output_filename,
    safe_filename,
)
from src.video_processor import VideoInfo, cut_video, get_video_info

logger = logging.getLogger("video_clipper.app")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Video Clipper Local",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "video_path": None,
        "video_info": None,
        "transcription": None,
        "candidate": None,
        "clip_path": None,
        "manual_start": 0.0,
        "manual_end": 40.0,
        "processing": False,
        "error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def show_error(msg: str) -> None:
    st.error(msg)
    logger.error(msg)


def save_uploaded_file(uploaded) -> Path:
    """Persist the uploaded file to a temporary location."""
    suffix = Path(uploaded.name).suffix.lower() or ".mp4"
    tmp = Path(tempfile.mkstemp(suffix=suffix, prefix="upload_")[1])
    with tmp.open("wb") as f:
        f.write(uploaded.getbuffer())
    return tmp


# ---------------------------------------------------------------------------
# Sidebar – status & help
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Status")
    ok_ffmpeg, ffmpeg_msg = check_ffmpeg()
    if ok_ffmpeg:
        st.success("FFmpeg OK")
        st.caption(ffmpeg_msg)
    else:
        st.error("FFmpeg não encontrado")
        st.markdown(
            "Na versão instalada pelo **VideoClipperSetup.exe** o FFmpeg já vem embutido.\n\n"
            "Se você está rodando pelo código-fonte, instale o FFmpeg ou defina FFMPEG_PATH."
        )

    try:
        require_api_key()
        st.success("API Key configurada")
    except RuntimeError:
        st.warning("API Key não configurada")
        st.markdown(
            "Feche e abra o **Video Clipper** pelo atalho para informar a chave, "
            "ou defina a variável de ambiente `GROQ_API_KEY`."
        )

    st.divider()
    st.caption("Modo padrão de corte: **rápido** (`-c copy`)")
    if DEBUG:
        st.caption("DEBUG=true")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("🎬 AI Video Clipper Local")
st.markdown(
    "Faça upload de um vídeo longo → a IA encontra o melhor trecho de **30–50 segundos** → "
    "você ajusta e baixa o clipe. **Todo o processamento de vídeo é local.**"
)

# ---- Upload ----
uploaded = st.file_uploader(
    "Selecione um vídeo",
    type=["mp4", "mov", "mkv", "webm"],
    help="Formatos suportados: MP4, MOV, MKV, WEBM",
)

if uploaded is not None:
    # Save only once
    if (
        st.session_state.video_path is None
        or Path(st.session_state.video_path).name != safe_filename(uploaded.name)
    ):
        # Clean previous
        if st.session_state.video_path:
            cleanup_file(Path(st.session_state.video_path))
        path = save_uploaded_file(uploaded)
        st.session_state.video_path = str(path)
        st.session_state.video_info = None
        st.session_state.transcription = None
        st.session_state.candidate = None
        st.session_state.clip_path = None

    video_path = Path(st.session_state.video_path)

    # Metadata
    if st.session_state.video_info is None:
        with st.spinner("Lendo metadados do vídeo..."):
            try:
                info = get_video_info(video_path)
                st.session_state.video_info = info
            except Exception as exc:
                show_error(f"Não foi possível ler o vídeo: {exc}")
                st.stop()

    info: VideoInfo = st.session_state.video_info

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Arquivo", uploaded.name[:30] + ("…" if len(uploaded.name) > 30 else ""))
    col2.metric("Tamanho", f"{info.size_mb:.1f} MB")
    col3.metric("Duração", format_timestamp(info.duration))
    col4.metric("Resolução", info.resolution if info.width else "—")

    st.video(str(video_path))

    # ---- Find best clip button ----
    if not ok_ffmpeg:
        st.warning("FFmpeg não encontrado. Na versão instalada ele já vem embutido.")
        st.stop()

    try:
        require_api_key()
    except RuntimeError as exc:
        st.warning(str(exc))
        st.stop()

    if st.button("🔍 Encontrar melhor clipe com IA", type="primary", use_container_width=True):
        st.session_state.candidate = None
        st.session_state.clip_path = None
        st.session_state.error = None

        progress = st.progress(0, text="Iniciando…")
        status = st.status("Processando…", expanded=True)

        try:
            # 1. Extract audio + transcribe
            status.write("1️⃣ Extraindo áudio localmente…")
            progress.progress(10, text="Extraindo áudio…")
            status.write("2️⃣ Enviando áudio para transcrição (Groq Whisper)…")
            progress.progress(25, text="Transcrevendo…")

            transcription, audio_path = transcribe_video(video_path)
            st.session_state.transcription = transcription
            cleanup_file(audio_path)  # free disk

            status.write(f"✅ Transcrição concluída ({len(transcription.segments)} segmentos)")
            progress.progress(55, text="Analisando com IA…")

            # 2. Analyze
            status.write("3️⃣ Analisando transcrição para encontrar o melhor trecho…")
            candidate = analyze_best_clip(transcription, video_duration=info.duration)
            st.session_state.candidate = candidate
            st.session_state.manual_start = candidate.start
            st.session_state.manual_end = candidate.end

            status.write("✅ Melhor clipe identificado!")
            progress.progress(100, text="Concluído")
            status.update(label="Análise concluída", state="complete")

        except Exception as exc:
            status.update(label="Erro", state="error")
            show_error(str(exc))
            st.session_state.error = str(exc)

# ---- Result & manual adjustment ----
if st.session_state.candidate is not None:
    cand: ClipCandidate = st.session_state.candidate
    info: VideoInfo = st.session_state.video_info

    st.divider()
    st.subheader("🎯 Resultado da IA")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Início", format_timestamp(cand.start))
    c2.metric("Fim", format_timestamp(cand.end))
    c3.metric("Duração", f"{cand.duration:.1f}s")
    c4.metric("Score", f"{cand.score}/100")

    st.markdown(f"**Motivo:** {cand.reason}")
    if cand.hook:
        st.markdown(f"**Hook:** {cand.hook}")

    st.markdown("#### Ajuste manual (opcional)")
    st.caption(
        f"Limite máximo recomendado: {MAX_CLIP_DURATION}s. "
        "O modo rápido (`-c copy`) pode ter pequena imprecisão no início por causa de keyframes."
    )

    # Sliders – keep start < end
    max_dur = info.duration
    col_a, col_b = st.columns(2)
    with col_a:
        new_start = st.slider(
            "Início (segundos)",
            min_value=0.0,
            max_value=max(0.1, max_dur - 1),
            value=float(st.session_state.manual_start),
            step=0.1,
            key="slider_start",
        )
    with col_b:
        new_end = st.slider(
            "Fim (segundos)",
            min_value=0.1,
            max_value=max_dur,
            value=float(st.session_state.manual_end),
            step=0.1,
            key="slider_end",
        )

    # Numeric inputs for precision
    n1, n2 = st.columns(2)
    with n1:
        new_start = st.number_input(
            "Início exato",
            min_value=0.0,
            max_value=max_dur,
            value=new_start,
            step=0.05,
            format="%.2f",
            key="num_start",
        )
    with n2:
        new_end = st.number_input(
            "Fim exato",
            min_value=0.0,
            max_value=max_dur,
            value=new_end,
            step=0.05,
            format="%.2f",
            key="num_end",
        )

    # Enforce constraints
    if new_end <= new_start:
        st.warning("O fim deve ser maior que o início.")
        new_end = min(new_start + 30.0, max_dur)

    duration = new_end - new_start
    if duration > MAX_CLIP_DURATION:
        st.warning(f"Duração {duration:.1f}s > {MAX_CLIP_DURATION}s. Será limitado no corte.")
        new_end = new_start + MAX_CLIP_DURATION
        duration = MAX_CLIP_DURATION

    st.info(f"**Duração do clipe:** {duration:.1f} segundos")

    st.session_state.manual_start = new_start
    st.session_state.manual_end = new_end

    # Mode selector
    mode = st.radio(
        "Modo de corte",
        options=["fast", "precise"],
        format_func=lambda x: "⚡ Rápido (-c copy)" if x == "fast" else "🎯 Preciso (re-encode)",
        horizontal=True,
        index=0,
    )

    if st.button("✂️ Gerar clipe", type="primary", use_container_width=True):
        out_name = generate_output_filename()
        out_path = OUTPUT_DIR / out_name

        with st.spinner(f"Cortando vídeo ({mode})…"):
            try:
                cut_video(
                    input_path=Path(st.session_state.video_path),
                    start=new_start,
                    end=new_end,
                    output_path=out_path,
                    mode=mode,
                )
                st.session_state.clip_path = str(out_path)
                st.success(f"Clipe gerado: `{out_name}`")
            except Exception as exc:
                show_error(str(exc))

# ---- Preview & download ----
if st.session_state.clip_path:
    clip_p = Path(st.session_state.clip_path)
    if clip_p.exists():
        st.divider()
        st.subheader("📺 Pré-visualização do clipe")
        st.video(str(clip_p))

        with clip_p.open("rb") as f:
            st.download_button(
                label="⬇️ Baixar MP4",
                data=f,
                file_name=clip_p.name,
                mime="video/mp4",
                use_container_width=True,
            )

st.divider()
st.caption(
    "Processamento de vídeo 100% local com FFmpeg. "
    "Apenas o áudio é enviado à API Groq para transcrição e análise."
)
