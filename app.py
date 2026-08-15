"""
AI Video Clipper Local
Streamlit application that finds the best 30-50s clip from a long video
using Groq for transcription + analysis and FFmpeg for local cutting.
Supports local upload and URL download via yt-dlp.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import streamlit as st

from src.config import (
    DEBUG,
    MAX_CLIP_DURATION,
    OUTPUT_DIR,
    TEMP_DIR,
    check_ffmpeg,
    require_api_key,
)
from src.clip_analyzer import ClipCandidate, analyze_best_clip
from src.downloader import QUALITY_PRESETS, download_video
from src.transcription import transcribe_video
from src.utils import (
    cleanup_file,
    format_timestamp,
    generate_output_filename,
    safe_filename,
)
from src.video_processor import VideoInfo, cut_video, get_video_info

logger = logging.getLogger("video_clipper.app")

st.set_page_config(
    page_title="AI Video Clipper Local",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def init_state() -> None:
    defaults = {
        "video_path": None,
        "video_name": None,
        "video_info": None,
        "transcription": None,
        "candidate": None,
        "clip_path": None,
        "manual_start": 0.0,
        "manual_end": 40.0,
        "cut_mode": "fast",
        "error": None,
        "last_cut_error": None,
        "source_url": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def show_error(msg: str) -> None:
    st.error(msg)
    logger.error(msg)


def save_uploaded_file(uploaded) -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.name).suffix.lower() or ".mp4"
    safe = safe_filename(Path(uploaded.name).stem) or "upload"
    dest = TEMP_DIR / f"upload_{safe}{suffix}"
    with dest.open("wb") as f:
        f.write(uploaded.getbuffer())
    return dest


def reset_video_state() -> None:
    st.session_state.video_info = None
    st.session_state.transcription = None
    st.session_state.candidate = None
    st.session_state.clip_path = None
    st.session_state.last_cut_error = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Status")
    ok_ffmpeg, ffmpeg_msg = check_ffmpeg()
    if ok_ffmpeg:
        st.success("FFmpeg OK")
        st.caption(ffmpeg_msg)
    else:
        st.error("FFmpeg não encontrado")

    try:
        require_api_key()
        st.success("API Key configurada")
    except RuntimeError:
        st.warning("API Key não configurada")

    st.divider()
    st.caption("Fontes: upload local ou URL (yt-dlp)")
    if DEBUG:
        st.caption("DEBUG=true")

    if st.button("🗑️ Limpar sessão"):
        if st.session_state.video_path:
            cleanup_file(Path(st.session_state.video_path))
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🎬 AI Video Clipper Local")
st.markdown(
    "Envie um vídeo **ou cole um link** (YouTube, etc.) → a IA encontra o melhor trecho de "
    "**30–50 segundos** → você ajusta e baixa o clipe."
)

tab_upload, tab_url = st.tabs(["📁 Upload de arquivo", "🔗 Link (YouTube / URL)"])

with tab_upload:
    uploaded = st.file_uploader(
        "Selecione um vídeo",
        type=["mp4", "mov", "mkv", "webm"],
        help="Formatos: MP4, MOV, MKV, WEBM",
    )
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
            st.session_state.source_url = ""
            reset_video_state()

with tab_url:
    st.caption("Funciona com YouTube e dezenas de outros sites suportados pelo yt-dlp.")
    url = st.text_input(
        "Cole a URL do vídeo",
        value=st.session_state.source_url,
        placeholder="https://www.youtube.com/watch?v=...",
        key="url_input",
    )
    quality = st.selectbox(
        "Qualidade do download",
        options=list(QUALITY_PRESETS.keys()),
        index=2,  # 720p default
        help="Qualidades menores = download mais rápido e arquivo menor.",
    )
    if st.button("⬇️ Baixar vídeo", type="primary", use_container_width=True, key="btn_download_url"):
        if not url.strip():
            st.warning("Cole uma URL válida.")
        else:
            with st.spinner(f"Baixando vídeo ({quality})… isso pode levar alguns minutos"):
                try:
                    if st.session_state.video_path:
                        cleanup_file(Path(st.session_state.video_path))
                    path = download_video(url.strip(), quality=quality)
                    st.session_state.video_path = str(path)
                    st.session_state.video_name = path.name
                    st.session_state.source_url = url.strip()
                    reset_video_state()
                    st.success(f"Download concluído: {path.name}")
                    st.rerun()
                except Exception as exc:
                    show_error(str(exc))

# ---- Active video from session ----
video_path: Optional[Path] = None
if st.session_state.video_path:
    video_path = Path(st.session_state.video_path)
    if not video_path.exists():
        st.warning("O arquivo de vídeo temporário foi perdido. Envie ou baixe novamente.")
        st.session_state.video_path = None
        st.session_state.video_name = None
        video_path = None

if video_path is not None:
    if st.session_state.video_info is None:
        with st.spinner("Lendo metadados do vídeo..."):
            try:
                st.session_state.video_info = get_video_info(video_path)
            except Exception as exc:
                show_error(f"Não foi possível ler o vídeo: {exc}")
                st.stop()

    info: VideoInfo = st.session_state.video_info

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Arquivo", (st.session_state.video_name or video_path.name)[:30])
    col2.metric("Tamanho", f"{info.size_mb:.1f} MB")
    col3.metric("Duração", format_timestamp(info.duration))
    col4.metric("Resolução", info.resolution if info.width else "—")

    st.video(str(video_path))

    if not ok_ffmpeg:
        st.warning("FFmpeg não encontrado.")
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
        st.session_state.last_cut_error = None

        progress = st.progress(0, text="Iniciando…")
        status = st.status("Processando…", expanded=True)

        try:
            status.write("1️⃣ Extraindo áudio localmente…")
            progress.progress(10, text="Extraindo áudio…")
            status.write("2️⃣ Enviando áudio para transcrição (Groq Whisper)…")
            progress.progress(25, text="Transcrevendo…")

            transcription, audio_path = transcribe_video(video_path)
            st.session_state.transcription = transcription
            cleanup_file(audio_path)

            status.write(f"✅ Transcrição concluída ({len(transcription.segments)} segmentos)")
            progress.progress(55, text="Analisando com IA…")

            status.write("3️⃣ Analisando transcrição…")
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

# ---- Result ----
if st.session_state.candidate is not None and st.session_state.video_info is not None:
    cand: ClipCandidate = st.session_state.candidate
    info = st.session_state.video_info

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
    st.caption(f"Limite máximo recomendado: {MAX_CLIP_DURATION}s.")

    max_dur = float(info.duration) if info.duration > 0 else 3600.0
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

    n1, n2 = st.columns(2)
    with n1:
        new_start = st.number_input(
            "Início exato",
            min_value=0.0,
            max_value=max_dur,
            value=float(new_start),
            step=0.05,
            format="%.2f",
            key="num_start",
        )
    with n2:
        new_end = st.number_input(
            "Fim exato",
            min_value=0.0,
            max_value=max_dur,
            value=float(new_end),
            step=0.05,
            format="%.2f",
            key="num_end",
        )

    if new_end <= new_start:
        st.warning("O fim deve ser maior que o início.")
        new_end = min(new_start + 30.0, max_dur)

    duration = new_end - new_start
    if duration > MAX_CLIP_DURATION:
        st.warning(f"Duração {duration:.1f}s > {MAX_CLIP_DURATION}s. Será limitado.")
        new_end = new_start + MAX_CLIP_DURATION
        duration = MAX_CLIP_DURATION

    st.info(f"**Duração do clipe:** {duration:.1f} segundos")
    st.session_state.manual_start = new_start
    st.session_state.manual_end = new_end

    mode = st.radio(
        "Modo de corte",
        options=["fast", "precise"],
        format_func=lambda x: "⚡ Rápido (-c copy)" if x == "fast" else "🎯 Preciso (re-encode)",
        horizontal=True,
        index=0 if st.session_state.cut_mode == "fast" else 1,
        key="cut_mode_radio",
    )
    st.session_state.cut_mode = mode

    if st.session_state.last_cut_error:
        st.error(st.session_state.last_cut_error)

    if st.button("✂️ Gerar clipe", type="primary", use_container_width=True, key="btn_cut"):
        if not st.session_state.video_path or not Path(st.session_state.video_path).exists():
            st.session_state.last_cut_error = "Arquivo de vídeo não encontrado."
            st.rerun()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_name = generate_output_filename()
        out_path = OUTPUT_DIR / out_name

        with st.spinner(f"Cortando vídeo ({mode})…"):
            try:
                cut_video(
                    input_path=Path(st.session_state.video_path),
                    start=float(new_start),
                    end=float(new_end),
                    output_path=out_path,
                    mode=mode,
                )
                st.session_state.clip_path = str(out_path)
                st.session_state.last_cut_error = None
                st.success(f"Clipe gerado: `{out_name}`")
            except Exception as exc:
                st.session_state.last_cut_error = str(exc)
                st.session_state.clip_path = None
                show_error(str(exc))

if st.session_state.clip_path:
    clip_p = Path(st.session_state.clip_path)
    if clip_p.exists():
        st.divider()
        st.subheader("📺 Pré-visualização do clipe")
        st.video(str(clip_p))
        with clip_p.open("rb") as f:
            data = f.read()
        st.download_button(
            label="⬇️ Baixar MP4",
            data=data,
            file_name=clip_p.name,
            mime="video/mp4",
            use_container_width=True,
            key="btn_download_clip",
        )
        st.caption(f"Arquivo salvo em: `{clip_p}`")
    else:
        st.warning("O arquivo do clipe não foi encontrado. Gere novamente.")
        st.session_state.clip_path = None

st.divider()
st.caption(
    "Processamento de vídeo 100% local com FFmpeg. "
    "Apenas o áudio é enviado à API Groq. Downloads via yt-dlp."
)
