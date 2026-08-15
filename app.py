"""
AI Video Clipper Local
Groq (speech) + FFmpeg (cut) + optional local Qwen2.5-VL (visual review).
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
from src.hardware import detect_hardware
from src.ollama_manager import (
    DEFAULT_VISION_MODEL,
    download_ollama_installer,
    get_status,
    install_ollama_windows,
    pull_model,
    start_ollama,
)
from src.transcription import Segment, TranscriptionResult, transcribe_video
from src.utils import (
    cleanup_file,
    format_timestamp,
    generate_output_filename,
    safe_filename,
)
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
        "manual_start": 0.0,
        "manual_end": 40.0,
        "cut_mode": "fast",
        "error": None,
        "last_cut_error": None,
        "source_url": "",
        "visual_result": None,
        "enable_visual": True,
        "hw_info": None,
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
    st.session_state.visual_result = None


# ---------------------------------------------------------------------------
# Sidebar: status + hardware + Ollama
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Status")
    ok_ffmpeg, ffmpeg_msg = check_ffmpeg()
    if ok_ffmpeg:
        st.success("FFmpeg OK")
    else:
        st.error("FFmpeg não encontrado")

    try:
        require_api_key()
        st.success("API Key Groq OK")
    except RuntimeError:
        st.warning("API Key não configurada")

    st.divider()
    st.subheader("🖥️ Hardware")
    if st.session_state.hw_info is None:
        try:
            st.session_state.hw_info = detect_hardware()
        except Exception:
            st.session_state.hw_info = None
    hw = st.session_state.hw_info
    if hw:
        st.caption(f"GPU: **{hw.gpu_name}**")
        st.caption(f"VRAM: **{hw.vram_gb} GB**")
        st.caption(f"RAM: **{hw.ram_gb} GB**")
        st.caption(f"CPU: {hw.cpu[:40]}")

    st.divider()
    st.subheader("👁️ IA Visual Local")
    st.caption("Análise **100% local** — o vídeo não sai do PC.")
    ostatus = get_status(DEFAULT_VISION_MODEL)
    st.caption(f"Modelo: `{DEFAULT_VISION_MODEL}`")
    if ostatus.ready:
        st.success("✓ Pronto")
    else:
        st.warning(ostatus.message or "Indisponível")

    if not ostatus.installed:
        if st.button("⬇️ Instalar Ollama (oficial)"):
            with st.spinner("Baixando instalador oficial…"):
                try:
                    setup = TEMP_DIR / "OllamaSetup.exe"
                    download_ollama_installer(setup)
                    st.info("Abrindo instalador. Conclua a instalação e reinicie o app.")
                    install_ollama_windows(setup)
                except Exception as exc:
                    show_error(str(exc))
    elif not ostatus.running:
        if st.button("▶️ Iniciar Ollama"):
            if start_ollama():
                st.success("Ollama iniciado")
                st.rerun()
            else:
                st.error("Não foi possível iniciar. Abra o Ollama manualmente.")
    elif not ostatus.model_installed:
        if st.button("📦 Instalar IA Visual (baixar modelo)"):
            prog = st.empty()

            def _cb(msg: str) -> None:
                prog.caption(msg)

            try:
                ok = pull_model(DEFAULT_VISION_MODEL, progress=_cb)
                if ok:
                    st.success("Modelo instalado!")
                    st.rerun()
                else:
                    st.error("Falha no download do modelo.")
            except Exception as exc:
                show_error(str(exc))

    st.session_state.enable_visual = st.checkbox(
        "Analisar clipe com IA local",
        value=st.session_state.enable_visual,
    )

    st.divider()
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
    "Upload ou link → **Groq** (fala) → **FFmpeg** (corte) → **Qwen local** (visão, opcional)."
)

tab_upload, tab_url = st.tabs(["📁 Upload de arquivo", "🔗 Link (YouTube / URL)"])

with tab_upload:
    uploaded = st.file_uploader("Selecione um vídeo", type=["mp4", "mov", "mkv", "webm"])
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
    st.caption("YouTube e sites suportados pelo yt-dlp.")
    url = st.text_input("Cole a URL do vídeo", value=st.session_state.source_url, key="url_input")
    quality = st.selectbox("Qualidade", list(QUALITY_PRESETS.keys()), index=2)
    if st.button("⬇️ Baixar vídeo", type="primary", use_container_width=True, key="btn_dl"):
        if not url.strip():
            st.warning("Cole uma URL válida.")
        else:
            with st.spinner(f"Baixando ({quality})…"):
                try:
                    if st.session_state.video_path:
                        cleanup_file(Path(st.session_state.video_path))
                    path = download_video(url.strip(), quality=quality)
                    st.session_state.video_path = str(path)
                    st.session_state.video_name = path.name
                    st.session_state.source_url = url.strip()
                    reset_video_state()
                    st.success(f"Download: {path.name}")
                    st.rerun()
                except Exception as exc:
                    show_error(str(exc))

video_path: Optional[Path] = None
if st.session_state.video_path:
    video_path = Path(st.session_state.video_path)
    if not video_path.exists():
        st.warning("Arquivo de vídeo perdido. Envie novamente.")
        st.session_state.video_path = None
        video_path = None

if video_path is not None:
    if st.session_state.video_info is None:
        with st.spinner("Lendo metadados…"):
            try:
                st.session_state.video_info = get_video_info(video_path)
            except Exception as exc:
                show_error(str(exc))
                st.stop()

    info: VideoInfo = st.session_state.video_info
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Arquivo", (st.session_state.video_name or "")[:28])
    c2.metric("Tamanho", f"{info.size_mb:.1f} MB")
    c3.metric("Duração", format_timestamp(info.duration))
    c4.metric("Resolução", info.resolution if info.width else "—")
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
        st.session_state.visual_result = None
        progress = st.progress(0, text="Iniciando…")
        status = st.status("Processando…", expanded=True)
        try:
            status.write("Extraindo áudio…")
            progress.progress(15)
            transcription, audio_path = transcribe_video(video_path)
            st.session_state.transcription = transcription
            cleanup_file(audio_path)
            status.write(f"Transcrição OK ({len(transcription.segments)} segmentos)")
            progress.progress(55)
            status.write("Analisando com Groq LLM…")
            candidate = analyze_best_clip(transcription, video_duration=info.duration)
            st.session_state.candidate = candidate
            st.session_state.manual_start = candidate.start
            st.session_state.manual_end = candidate.end
            progress.progress(100)
            status.update(label="Análise concluída", state="complete")
        except Exception as exc:
            status.update(label="Erro", state="error")
            show_error(str(exc))

if st.session_state.candidate is not None and st.session_state.video_info is not None:
    cand: ClipCandidate = st.session_state.candidate
    info = st.session_state.video_info
    st.divider()
    st.subheader("🎯 Resultado da IA (Groq)")
    a, b, c, d = st.columns(4)
    a.metric("Início", format_timestamp(cand.start))
    b.metric("Fim", format_timestamp(cand.end))
    c.metric("Duração", f"{cand.duration:.1f}s")
    d.metric("Score fala", f"{cand.score}/100")
    st.markdown(f"**Motivo:** {cand.reason}")
    if cand.hook:
        st.markdown(f"**Hook:** {cand.hook}")

    max_dur = float(info.duration) if info.duration > 0 else 3600.0
    col_a, col_b = st.columns(2)
    with col_a:
        new_start = st.slider("Início (s)", 0.0, max(0.1, max_dur - 1), float(st.session_state.manual_start), 0.1, key="slider_start")
    with col_b:
        new_end = st.slider("Fim (s)", 0.1, max_dur, float(st.session_state.manual_end), 0.1, key="slider_end")
    n1, n2 = st.columns(2)
    with n1:
        new_start = st.number_input("Início exato", 0.0, max_dur, float(new_start), 0.05, format="%.2f", key="num_start")
    with n2:
        new_end = st.number_input("Fim exato", 0.0, max_dur, float(new_end), 0.05, format="%.2f", key="num_end")
    if new_end <= new_start:
        new_end = min(new_start + 30.0, max_dur)
    duration = new_end - new_start
    if duration > MAX_CLIP_DURATION:
        new_end = new_start + MAX_CLIP_DURATION
        duration = MAX_CLIP_DURATION
    st.info(f"Duração do clipe: **{duration:.1f}s**")
    st.session_state.manual_start = new_start
    st.session_state.manual_end = new_end

    mode = st.radio(
        "Modo de corte",
        ["fast", "precise"],
        format_func=lambda x: "⚡ Rápido (-c copy)" if x == "fast" else "🎯 Preciso",
        horizontal=True,
        key="cut_mode_radio",
    )
    st.session_state.cut_mode = mode
    if st.session_state.last_cut_error:
        st.error(st.session_state.last_cut_error)

    if st.button("✂️ Gerar clipe", type="primary", use_container_width=True, key="btn_cut"):
        if not st.session_state.video_path or not Path(st.session_state.video_path).exists():
            st.session_state.last_cut_error = "Vídeo não encontrado."
            st.rerun()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / generate_output_filename()
        with st.spinner(f"Cortando ({mode})…"):
            try:
                cut_video(Path(st.session_state.video_path), float(new_start), float(new_end), out_path, mode=mode)
                st.session_state.clip_path = str(out_path)
                st.session_state.last_cut_error = None
                st.session_state.visual_result = None
                st.success(f"Clipe: `{out_path.name}`")
            except Exception as exc:
                st.session_state.last_cut_error = str(exc)
                show_error(str(exc))

if st.session_state.clip_path:
    clip_p = Path(st.session_state.clip_path)
    if clip_p.exists():
        st.divider()
        st.subheader("📺 Pré-visualização do clipe")
        st.video(str(clip_p))
        with clip_p.open("rb") as f:
            data = f.read()
        st.download_button("⬇️ Baixar MP4", data, file_name=clip_p.name, mime="video/mp4", use_container_width=True)

        # ---- Local visual AI ----
        st.divider()
        st.subheader("👁️ IA Visual Local (Qwen2.5-VL)")
        st.caption("Frames analisados **no seu PC**. Nada de vídeo é enviado à nuvem nesta etapa.")

        ostatus = get_status(DEFAULT_VISION_MODEL)
        if not st.session_state.enable_visual:
            st.info("Análise visual desativada na barra lateral.")
        elif not ostatus.ready:
            st.warning(
                f"IA visual indisponível: {ostatus.message}. "
                "Você pode continuar só com o clipe gerado."
            )
        else:
            if st.button("🔍 Analisar clipe com IA local", type="secondary", use_container_width=True):
                with st.spinner("Extraindo frames e analisando (pode levar 1–3 min na RTX 2070)…"):
                    try:
                        tr: Optional[TranscriptionResult] = st.session_state.transcription
                        segs: list[Segment] = tr.segments if tr else []
                        speech = float(st.session_state.candidate.score) if st.session_state.candidate else 70.0
                        dur = float(st.session_state.manual_end - st.session_state.manual_start)
                        result = analyze_clip_visual(
                            clip_path=clip_p,
                            clip_duration=max(dur, 1.0),
                            clip_start_abs=float(st.session_state.manual_start),
                            clip_end_abs=float(st.session_state.manual_end),
                            segments=segs,
                            speech_score=speech,
                            max_frames=10,
                        )
                        st.session_state.visual_result = result
                    except Exception as exc:
                        show_error(str(exc))

            vr: Optional[VisualAnalysis] = st.session_state.visual_result
            if vr is not None:
                st.markdown(f"### Score final: **{vr.overall_score}/100** — `{vr.verdict}`")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Hook visual", vr.visual_hook_score)
                m2.metric("Retenção", vr.retention_score)
                m3.metric("Composição", vr.composition_score)
                m4.metric("Contexto", vr.context_match_score)
                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Emoção", vr.emotion_score)
                m6.metric("Qualidade", vr.visual_quality_score)
                m7.metric("Short-form", vr.short_form_score)
                m8.metric("Confiança", f"{vr.confidence:.0%}")
                if vr.strengths:
                    st.markdown("**Pontos fortes:** " + "; ".join(vr.strengths))
                if vr.problems:
                    st.markdown("**Problemas:** " + "; ".join(vr.problems))
                if vr.suggestions:
                    st.markdown("**Sugestões:** " + "; ".join(vr.suggestions))
                st.caption(f"Frames: {vr.frames_used} · Inferência: {vr.inference_ms} ms")

                if vr.suggested_start is not None or vr.suggested_end is not None:
                    abs_start = float(st.session_state.manual_start) + float(vr.suggested_start or 0)
                    abs_end = float(st.session_state.manual_start) + float(
                        vr.suggested_end if vr.suggested_end is not None else (st.session_state.manual_end - st.session_state.manual_start)
                    )
                    st.info(f"Sugestão de recorte: {abs_start:.1f}s → {abs_end:.1f}s")
                    colx, coly = st.columns(2)
                    with colx:
                        if st.button("✅ Aplicar sugestão"):
                            st.session_state.manual_start = abs_start
                            st.session_state.manual_end = abs_end
                            st.session_state.clip_path = None
                            st.session_state.visual_result = None
                            st.success("Sugestão aplicada. Gere o clipe novamente.")
                            st.rerun()
                    with coly:
                        if st.button("↩️ Manter clipe"):
                            st.info("Mantido o clipe atual.")
    else:
        st.warning("Arquivo do clipe não encontrado.")
        st.session_state.clip_path = None

st.divider()
st.caption(
    "Groq = fala/transcrição · FFmpeg = corte local · Qwen2.5-VL via Ollama = visão local (RTX 2070 8GB)."
)
