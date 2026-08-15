"""
Video Clipper — clip generator + local editor + content advisor.
"""

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
    apply_trim,
    new_project_from_clip,
    run_export,
    set_aspect,
    set_crop,
)
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
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: #0a0a0b; color: #e8e8ea; }
#MainMenu, footer, header { visibility: hidden; height: 0; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding-top: 1.25rem !important; max-width: 1200px; }
h1,h2,h3 { letter-spacing: -0.02em; font-weight: 600 !important; color: #f4f4f5 !important; }
:root { --accent: #c8f542; --surface: #141416; --border: #2a2a2e; --muted: #8b8b93; }
.vc-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.1rem 1.35rem; margin-bottom: 0.85rem; }
.vc-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
.vc-value { font-size: 1rem; font-weight: 500; }
.vc-muted { color: var(--muted); font-size: 0.85rem; }
.vc-header { display:flex; justify-content:space-between; align-items:baseline; border-bottom:1px solid var(--border); padding-bottom:0.85rem; margin-bottom:1.25rem; }
.vc-logo { font-size:1.15rem; font-weight:700; } .vc-logo span { color: var(--accent); }
.vc-section { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em; color:var(--muted); margin:1.5rem 0 0.6rem; }
.stButton > button { border-radius:8px !important; font-weight:600 !important; border:1px solid var(--border) !important; background:#1a1a1d !important; color:#f4f4f5 !important; }
.stButton > button[kind="primary"] { background:var(--accent) !important; color:#0a0a0b !important; border-color:var(--accent) !important; }
.vc-meta { display:flex; gap:1.25rem; flex-wrap:wrap; margin:0.5rem 0 0.75rem; }
.vc-meta .k { font-size:0.62rem; text-transform:uppercase; color:var(--muted); }
.vc-meta .v { font-size:0.92rem; font-weight:500; }
.vc-timeline { height:6px; background:#1e1e22; border-radius:3px; position:relative; margin:0.5rem 0; }
.vc-timeline-fill { position:absolute; height:100%; background:var(--accent); border-radius:3px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def init_state() -> None:
    defaults = {
        "video_path": None, "video_name": None, "video_info": None,
        "transcription": None, "candidate": None, "clip_path": None,
        "source_url": "", "content_pkg": None, "prefer_local": True,
        "editor_open": False, "editor_history": None, "export_path": None,
        "manual_start": 0.0, "manual_end": 40.0,
    }
    for k, v in defaults.items():
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
    '<div class="vc-muted">Generate · Edit · Export · Local</div></div>',
    unsafe_allow_html=True,
)

ok_ffmpeg, _ = check_ffmpeg()
try:
    require_api_key()
    groq_ok = True
except RuntimeError:
    groq_ok = False
ost = get_status(DEFAULT_VISION_MODEL)

c1, c2, c3 = st.columns(3)
c1.markdown(f'<div class="vc-card"><div class="vc-label">FFmpeg</div><div class="vc-value">{"OK" if ok_ffmpeg else "Missing"}</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="vc-card"><div class="vc-label">Groq</div><div class="vc-value">{"Ready" if groq_ok else "Optional"}</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="vc-card"><div class="vc-label">Ollama</div><div class="vc-value">{"On" if ost.ready else "Off"}</div></div>', unsafe_allow_html=True)

st.session_state.prefer_local = st.checkbox(
    "IA local (faster-whisper + Ollama) — evita limite da Groq",
    value=st.session_state.prefer_local,
)

# ── Source ──────────────────────────────────────────────
st.markdown('<div class="vc-section">1 · Source</div>', unsafe_allow_html=True)
t_up, t_url = st.tabs(["Arquivo", "URL"])
with t_up:
    up = st.file_uploader("Vídeo", type=["mp4", "mov", "mkv", "webm"], label_visibility="collapsed")
    if up is not None:
        name = safe_filename(up.name)
        if st.session_state.video_name != name or not (st.session_state.video_path and Path(st.session_state.video_path).exists()):
            if st.session_state.video_path:
                cleanup_file(Path(st.session_state.video_path))
            st.session_state.video_path = str(save_upload(up))
            st.session_state.video_name = name
            reset_pipeline()
with t_url:
    url = st.text_input("URL", value=st.session_state.source_url, placeholder="YouTube…")
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
    st.session_state.video_path = None

# ── Generator ───────────────────────────────────────────
if vpath:
    if st.session_state.video_info is None:
        with st.spinner("Metadados…"):
            st.session_state.video_info = get_video_info(vpath)
    info: VideoInfo = st.session_state.video_info

    st.markdown('<div class="vc-section">2 · Clip Generator</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="vc-meta">'
        f'<div><div class="k">Arquivo</div><div class="v">{(st.session_state.video_name or "")[:36]}</div></div>'
        f'<div><div class="k">Duração</div><div class="v">{format_timestamp(info.duration)}</div></div>'
        f'<div><div class="k">Res</div><div class="v">{info.resolution if info.width else "—"}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.video(str(vpath))

    if not ok_ffmpeg:
        st.warning("FFmpeg necessário.")
        st.stop()

    if st.button("Encontrar melhor clipe", type="primary", use_container_width=True):
        st.session_state.candidate = None
        st.session_state.clip_path = None
        st.session_state.editor_open = False
        st.session_state.editor_history = None
        st.session_state.export_path = None
        status = st.status("Processando…", expanded=True)
        try:
            prefer = st.session_state.prefer_local
            status.write("Transcrevendo…")
            tr, audio = transcribe_video(vpath, prefer_local=prefer)
            st.session_state.transcription = tr
            cleanup_file(audio)
            status.write(f"OK ({tr.source}) · {len(tr.segments)} segmentos")
            status.write("Escolhendo trecho…")
            cand = analyze_best_clip(tr, video_duration=info.duration, prefer_local=prefer)
            st.session_state.candidate = cand
            st.session_state.manual_start = cand.start
            st.session_state.manual_end = cand.end
            status.update(label="Pronto", state="complete")
        except Exception as e:
            status.update(label="Erro", state="error")
            err(str(e))

if st.session_state.candidate and st.session_state.video_info:
    cand: ClipCandidate = st.session_state.candidate
    info = st.session_state.video_info
    max_dur = float(info.duration) or 3600.0
    sp = (cand.start / max_dur) * 100 if max_dur else 0
    wp = (cand.duration / max_dur) * 100 if max_dur else 5

    st.markdown(
        f'<div class="vc-card"><div class="vc-label">Trecho</div>'
        f'<div class="vc-timeline"><div class="vc-timeline-fill" style="left:{sp:.2f}%;width:{wp:.2f}%"></div></div>'
        f'<div class="vc-meta">'
        f'<div><div class="k">Início</div><div class="v">{format_timestamp(cand.start)}</div></div>'
        f'<div><div class="k">Fim</div><div class="v">{format_timestamp(cand.end)}</div></div>'
        f'<div><div class="k">Duração</div><div class="v">{cand.duration:.1f}s</div></div>'
        f'<div><div class="k">Score</div><div class="v">{cand.score}</div></div>'
        f'</div><p class="vc-muted">{cand.reason}</p></div>',
        unsafe_allow_html=True,
    )

    a, b = st.columns(2)
    with a:
        ns = st.slider("Início", 0.0, max(0.1, max_dur - 1), float(st.session_state.manual_start), 0.1)
    with b:
        ne = st.slider("Fim", 0.1, max_dur, float(st.session_state.manual_end), 0.1)
    if ne <= ns:
        ne = min(ns + 30, max_dur)
    if ne - ns > MAX_CLIP_DURATION:
        ne = ns + MAX_CLIP_DURATION
    st.session_state.manual_start, st.session_state.manual_end = ns, ne

    if st.button("Gerar clipe (corte)", type="primary", use_container_width=True):
        OUTPUT_DIR.mkdir(exist_ok=True)
        out = OUTPUT_DIR / generate_output_filename()
        with st.spinner("FFmpeg cortando…"):
            try:
                cut_video(Path(st.session_state.video_path), ns, ne, out, mode="fast")
                st.session_state.clip_path = str(out)
                st.session_state.editor_open = False
                st.session_state.editor_history = None
                st.session_state.export_path = None
                st.success("Clipe gerado")
            except Exception as e:
                err(str(e))

# ── Clip + Editor ───────────────────────────────────────
if st.session_state.clip_path and Path(st.session_state.clip_path).exists():
    clip_p = Path(st.session_state.clip_path)
    st.markdown('<div class="vc-section">3 · Clipe</div>', unsafe_allow_html=True)
    st.video(str(clip_p))
    with clip_p.open("rb") as f:
        st.download_button("Baixar clipe bruto", f.read(), file_name=clip_p.name, mime="video/mp4", use_container_width=True)

    if st.button("Abrir editor", type="primary", use_container_width=True):
        info = get_video_info(clip_p)
        tr: Optional[TranscriptionResult] = st.session_state.transcription
        segs = tr.segments if tr else []
        proj = new_project_from_clip(
            clip_p,
            duration=info.duration or max(1.0, float(st.session_state.manual_end - st.session_state.manual_start)),
            fps=info.fps or 30.0,
            name=clip_p.stem,
            segments=segs,
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
        st.caption("Alterações ficam no projeto até você exportar. Undo/Redo disponíveis.")

        u1, u2, u3 = st.columns(3)
        with u1:
            if st.button("Desfazer", disabled=not hist.can_undo(), use_container_width=True):
                hist.undo()
                st.rerun()
        with u2:
            if st.button("Refazer", disabled=not hist.can_redo(), use_container_width=True):
                hist.redo()
                st.rerun()
        with u3:
            st.caption(f"Timeline: **{state.timeline_duration:.1f}s** · FPS {state.fps:.2f}")

        # Trim
        st.markdown("**Trim**")
        t1, t2 = st.columns(2)
        with t1:
            tr_s = st.number_input("Início no arquivo (s)", 0.0, state.source_duration, float(state.playable_range.start), 0.05)
        with t2:
            tr_e = st.number_input("Fim no arquivo (s)", 0.0, state.source_duration, float(state.playable_range.end), 0.05)
        if st.button("Aplicar trim", use_container_width=True):
            if tr_e > tr_s:
                hist.push(apply_trim(state, tr_s, tr_e))
                st.rerun()

        # Aspect
        st.markdown("**Formato**")
        aspect_labels = {
            AspectRatio.VERTICAL_9_16: "9:16 Vertical",
            AspectRatio.SQUARE_1_1: "1:1 Quadrado",
            AspectRatio.LANDSCAPE_16_9: "16:9 Horizontal",
        }
        choice = st.radio(
            "Aspect",
            list(AspectRatio),
            format_func=lambda a: aspect_labels[a],
            index=list(AspectRatio).index(state.aspect),
            horizontal=True,
            label_visibility="collapsed",
        )
        if choice != state.aspect:
            hist.push(set_aspect(state, choice))
            st.rerun()

        # Crop
        st.markdown("**Enquadramento**")
        z = st.slider("Zoom", 1.0, 3.0, float(state.crop.zoom), 0.05)
        cx = st.slider("Centro X", 0.0, 1.0, float(state.crop.center_x), 0.01)
        cy = st.slider("Centro Y", 0.0, 1.0, float(state.crop.center_y), 0.01)
        pc1, pc2, pc3 = st.columns(3)
        if pc1.button("Centralizar", use_container_width=True):
            hist.push(set_crop(state, CropSettings(zoom=z, center_x=0.5, center_y=0.5)))
            st.rerun()
        if pc2.button("Esquerda", use_container_width=True):
            hist.push(set_crop(state, CropSettings(zoom=z, center_x=0.25, center_y=cy)))
            st.rerun()
        if pc3.button("Direita", use_container_width=True):
            hist.push(set_crop(state, CropSettings(zoom=z, center_x=0.75, center_y=cy)))
            st.rerun()
        if st.button("Aplicar crop/zoom", use_container_width=True):
            hist.push(set_crop(state, CropSettings(zoom=z, center_x=cx, center_y=cy)))
            st.rerun()

        # Audio
        st.markdown("**Áudio**")
        vol = st.slider("Volume", 0.0, 2.0, float(state.audio.volume), 0.05)
        muted = st.checkbox("Mudo", value=state.audio.muted)
        fi, fo = st.columns(2)
        with fi:
            fade_in = st.number_input("Fade in (s)", 0.0, 5.0, float(state.audio.fade_in), 0.1)
        with fo:
            fade_out = st.number_input("Fade out (s)", 0.0, 5.0, float(state.audio.fade_out), 0.1)
        if st.button("Aplicar áudio", use_container_width=True):
            s = state.clone()
            s.audio = AudioSettings(volume=vol, muted=muted, fade_in=fade_in, fade_out=fade_out)
            hist.push(s)
            st.rerun()

        # Captions
        st.markdown("**Legendas**")
        st.caption(f"{len(state.captions)} cues da transcrição")
        burn = st.checkbox("Queimar legendas no export", value=True)
        if state.captions:
            with st.expander("Editar textos das legendas"):
                for i, cap in enumerate(state.captions[:40]):
                    new_t = st.text_input(
                        f"{cap.start:.1f}–{cap.end:.1f}s",
                        value=cap.text,
                        key=f"cap_{cap.id}",
                    )
                    if new_t != cap.text:
                        s = state.clone()
                        s.captions[i].text = new_t
                        hist.push(s)
                        st.rerun()

        # Export
        st.markdown("**Exportar**")
        if st.button("Exportar MP4 final", type="primary", use_container_width=True):
            OUTPUT_DIR.mkdir(exist_ok=True)
            out = OUTPUT_DIR / generate_output_filename(prefix="final")
            prog = st.progress(0.0, text="Render…")

            def _cb(p: float, msg: str) -> None:
                prog.progress(min(1.0, p), text=msg)

            try:
                run_export(hist.current, out, burn_captions=burn, progress=_cb)
                st.session_state.export_path = str(out)
                st.success("Export concluído")
            except Exception as e:
                err(str(e))

        if st.session_state.export_path and Path(st.session_state.export_path).exists():
            ep = Path(st.session_state.export_path)
            st.video(str(ep))
            with ep.open("rb") as f:
                st.download_button("Baixar MP4 final", f.read(), file_name=ep.name, mime="video/mp4", use_container_width=True)

            # Content advisor
            st.markdown('<div class="vc-section">5 · AI Content Advisor</div>', unsafe_allow_html=True)
            if ost.ready and st.button("Analisar publicação (local)", use_container_width=True):
                tr = st.session_state.transcription
                with st.spinner("Ollama…"):
                    try:
                        st.session_state.content_pkg = analyze_content(
                            clip_path=ep,
                            duration_sec=hist.current.timeline_duration,
                            segments=tr.segments if tr else [],
                            full_text=tr.text if tr else "",
                            use_vision=True,
                            max_frames=2,
                        )
                    except Exception as e:
                        err(str(e))

            pkg: Optional[ContentPackage] = st.session_state.content_pkg
            if pkg:
                st.write(pkg.context.summary)
                st.markdown(f"**Título:** {pkg.title.primary}")
                st.download_button(
                    "Baixar pacote de texto",
                    pkg.copy_all_text().encode("utf-8"),
                    file_name="content_package.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

st.caption("Editor local · FFmpeg · IA local opcional · Groq só se desmarcar modo local.")
