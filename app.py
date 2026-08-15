"""
Video Clipper — focado no Campeonato Dona 30K.
Fluxo: video → cortes → editar → export campeonato (CTA + legendas + copy).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from src.cache import file_sha256, load_json, save_json
from src.campaigns.apply import apply_campaign_to_project
from src.campaigns.copygen import build_platform_copy
from src.campaigns.loader import load_campaign
from src.campaigns.validator import validate_campaign_export
from src.captions import WordStamp
from src.clip_analyzer import ClipCandidate, analyze_best_clips
from src.config import OUTPUT_DIR, TEMP_DIR, check_ffmpeg, require_api_key
from src.downloader import QUALITY_PRESETS, download_video
from src.editor import AspectRatio, HistoryStack, apply_trim, new_project_from_clip, render_timeline_html, run_export
from src.preset import CANVAS_H, CANVAS_W, DEFAULT
from src.thumbnails import extract_thumbnail
from src.transcription import Segment, TranscriptionResult, transcribe_video
from src.utils import cleanup_file, format_timestamp, generate_output_filename, safe_filename
from src.video_processor import VideoInfo, cut_video, get_video_info

logger = logging.getLogger("video_clipper.app")

st.set_page_config(
    page_title="Dona 30K Clipper",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
        "campaign_handle": "",
        "platform_copies": None,
        "selected_cand": None,
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
    st.session_state.transcription = None
    st.session_state.candidates = []
    st.session_state.clip_path = None
    st.session_state.export_path = None
    st.session_state.editor_open = False
    st.session_state.editor_history = None
    st.session_state.perf_log = []
    st.session_state.platform_copies = None
    st.session_state.selected_cand = None


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


def get_camp():
    c = load_campaign("dona30k")
    if st.session_state.campaign_handle:
        c.with_handle(st.session_state.campaign_handle)
    return c


def open_editor(cand: ClipCandidate, idx: int) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / generate_output_filename(prefix=f"clip{idx+1}")
    with st.spinner("Cortando + legendas + CTA Twitch…"):
        cut_video(Path(st.session_state.video_path), cand.start, cand.end, out, mode="fast")
    st.session_state.clip_path = str(out)
    st.session_state.selected_cand = cand
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
    proj.aspect = AspectRatio.VERTICAL_9_16
    try:
        proj = apply_campaign_to_project(proj, get_camp())
    except Exception as e:
        logger.warning("campaign: %s", e)
    st.session_state.editor_history = HistoryStack(proj)
    st.session_state.editor_open = True
    st.session_state.export_path = None
    st.session_state.platform_copies = None


# ── Header ───────────────────────────────────────────────
st.markdown(
    '<div class="vc-header"><div class="vc-logo">DONA <span>30K</span> CLIPPER</div>'
    '<div class="vc-muted">16/08 → 15/09 · #dona30K · twitch.tv/dona</div></div>',
    unsafe_allow_html=True,
)

ok_ffmpeg, _ = check_ffmpeg()

# ── Setup minimo da campanha ─────────────────────────────
st.markdown('<div class="vc-section">Configuracao (1x)</div>', unsafe_allow_html=True)
try:
    camp0 = load_campaign("dona30k")
except Exception as e:
    err(f"Perfil dona30k.json nao encontrado: {e}")
    st.stop()

handle = st.text_input(
    "@ oficial do Dona (obrigatorio — coloque o correto do regulamento)",
    value=st.session_state.campaign_handle,
    placeholder="@…",
)
st.session_state.campaign_handle = (handle or "").strip()

st.markdown(
    f'<div class="vc-card">'
    f'<b>O que o app faz por voce</b><br>'
    f'• Corta em <b>9:16 1080×1920</b><br>'
    f'• Legendas padrao sincronizadas<br>'
    f'• CTA <code>{camp0.twitch_display}</code> <b>dentro</b> do video<br>'
    f'• Prepara titulo/descricao com <code>{camp0.hashtag}</code> e o @<br>'
    f'• Valida antes de exportar<br>'
    f'<span class="vc-muted">Nao publica sozinho · nao compra view · nao faz collab</span>'
    f'</div>',
    unsafe_allow_html=True,
)

with st.expander("Regras resumidas"):
    st.markdown("**Obrigatorio:** " + " · ".join(camp0.rules_required[:4]))
    st.caption("Proibido: compra de engajamento, fake react, collab, afiliados, repost de outros, Opus Clip exclusivo, fora de contexto.")
    st.caption("WhatsApp oficial: entre manualmente no grupo do campeonato.")

st.session_state.prefer_local = st.checkbox(
    "Usar IA local se Groq estiver no limite",
    value=False,
    help="Padrao = Groq (mais leve no PC)",
)
st.session_state.top_n = st.select_slider("Quantos cortes sugerir", options=[5, 10, 15], value=5)

# ── Video ────────────────────────────────────────────────
st.markdown('<div class="vc-section">1 · Video da live / VOD</div>', unsafe_allow_html=True)
t1, t2 = st.tabs(["Arquivo MP4", "Link YouTube"])
with t1:
    up = st.file_uploader("Video", type=["mp4", "mov", "mkv", "webm"], label_visibility="collapsed")
    if up is not None:
        name = safe_filename(up.name)
        if st.session_state.video_name != name or not (
            st.session_state.video_path and Path(str(st.session_state.video_path)).exists()
        ):
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
        try:
            t0 = time.time()
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

    if st.button("Encontrar melhores cortes", type="primary", use_container_width=True):
        reset_analysis()
        status = st.status("Analisando…", expanded=True)
        prefer = st.session_state.prefer_local
        n = int(st.session_state.top_n)
        try:
            if not st.session_state.video_hash:
                status.write("Hash (cache)…")
                t0 = time.time()
                st.session_state.video_hash = file_sha256(vpath)
                perf("Hash", time.time() - t0)
            vhash = st.session_state.video_hash

            cached = load_json(vhash, "transcription")
            if cached and not cached.get("words"):
                cached = None
            if cached:
                status.write("Transcricao em cache")
                tr = transcription_from_cache(cached)
            else:
                status.write("Transcrevendo (1 vez por video)…")
                t0 = time.time()
                tr, audio = transcribe_video(vpath, prefer_local=prefer)
                cleanup_file(audio)
                perf("Transcription", time.time() - t0)
                save_json(vhash, "transcription", transcription_to_cache(tr))
            st.session_state.transcription = tr

            ck = f"analysis_n{n}_{'local' if prefer else 'groq'}"
            ca = load_json(vhash, ck)
            if ca and ca.get("candidates"):
                status.write("Analise em cache")
                cands = [ClipCandidate.from_dict(c) for c in ca["candidates"]]
            else:
                status.write(f"TOP {n} momentos…")
                t0 = time.time()
                cands = analyze_best_clips(tr, info.duration, n=n, prefer_local=prefer)
                perf("Analysis", time.time() - t0)
                save_json(vhash, ck, {"candidates": [c.to_dict() for c in cands]})

            for c in cands:
                extract_thumbnail(vpath, c.start + min(2.0, c.duration / 3), vhash)
            st.session_state.candidates = cands
            status.update(label=f"{len(cands)} cortes prontos", state="complete")
        except Exception as e:
            status.update(label="Erro", state="error")
            err(str(e))

if st.session_state.candidates:
    st.markdown('<div class="vc-section">2 · Escolha o corte</div>', unsafe_allow_html=True)
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
            if st.button("Usar este corte", key=f"e{i}", type="primary", use_container_width=True):
                open_editor(cand, i)
                st.rerun()

# ── Editor + export campeonato ───────────────────────────
if st.session_state.editor_open and st.session_state.editor_history and st.session_state.clip_path:
    hist: HistoryStack = st.session_state.editor_history
    state = hist.current
    clip_p = Path(st.session_state.clip_path)
    camp = get_camp()

    st.markdown(
        f'<div class="vc-section">3 · Revisar e exportar · {len(state.captions)} legendas · CTA Twitch</div>',
        unsafe_allow_html=True,
    )
    st.markdown(render_timeline_html(state), unsafe_allow_html=True)

    left, right = st.columns([1.3, 1])
    with left:
        st.video(str(clip_p))
        st.caption("Preview sem overlay. No MP4 final entram legenda + Twitch.tv/dona.")
    with right:
        # ensure CTA
        if not any("twitch" in (t.text or "").lower() for t in hist.current.texts):
            hist.push(apply_campaign_to_project(hist.current, camp))
            st.rerun()

        result = validate_campaign_export(hist.current, camp)
        st.markdown("**Checklist Dona 30K**")
        for _n, passed, msg in result.checks:
            st.write(("✓ " if passed else "✗ ") + msg)

        if not camp.handle_ok:
            st.error("Coloque o @ oficial em cima da pagina antes de exportar.")

        with st.expander("Trim opcional"):
            a = st.number_input("Inicio", 0.0, state.source_duration, float(state.playable_range.start), 0.05)
            b = st.number_input("Fim", 0.0, state.source_duration, float(state.playable_range.end), 0.05)
            if st.button("Aplicar") and b > a:
                hist.push(apply_campaign_to_project(apply_trim(state, a, b), camp))
                st.rerun()

        st.markdown("---")
        # Rascunho rapido (sem legenda) so para ver crop/CTA
        if st.button("Rascunho rapido (sem legenda)", use_container_width=True):
            OUTPUT_DIR.mkdir(exist_ok=True)
            out = OUTPUT_DIR / generate_output_filename(prefix="draft")
            bar = st.progress(0.0)

            def cb_d(p, msg):
                bar.progress(min(1.0, p), text=msg)

            try:
                fs = apply_campaign_to_project(hist.current, camp)
                run_export(fs, out, burn_captions=False, progress=cb_d)
                st.session_state.export_path = str(out)
                st.info("Rascunho: so 9:16 + CTA. Use o botao verde para o arquivo final.")
            except Exception as e:
                err(str(e))

        can_final = result.ok and camp.handle_ok
        if st.button(
            "EXPORTAR FINAL PARA O CAMPEONATO",
            type="primary",
            use_container_width=True,
            disabled=not can_final,
        ):
            OUTPUT_DIR.mkdir(exist_ok=True)
            out = OUTPUT_DIR / generate_output_filename(prefix="dona30k")
            bar = st.progress(0.0)

            def cb(p, msg):
                bar.progress(min(1.0, p), text=msg)

            try:
                final_state = apply_campaign_to_project(hist.current, camp)
                run_export(final_state, out, burn_captions=True, progress=cb)
                st.session_state.export_path = str(out)
                hook = ""
                if st.session_state.selected_cand:
                    hook = (st.session_state.selected_cand.hook or st.session_state.selected_cand.reason or "")[:100]
                st.session_state.platform_copies = build_platform_copy(camp, clip_hook=hook)
                st.success("Pronto. Baixe o MP4 e copie o texto de cada rede.")
            except Exception as e:
                err(str(e))

    if st.session_state.export_path and Path(st.session_state.export_path).exists():
        ep = Path(st.session_state.export_path)
        st.video(str(ep))
        with ep.open("rb") as f:
            st.download_button(
                "Baixar MP4",
                f.read(),
                file_name=ep.name,
                mime="video/mp4",
                use_container_width=True,
            )

        if st.session_state.platform_copies:
            st.markdown('<div class="vc-section">4 · Copiar e postar</div>', unsafe_allow_html=True)
            st.caption("Cole no TikTok / Reels / Shorts / Facebook. A hashtag esta exatamente #dona30K.")
            for pc in st.session_state.platform_copies:
                with st.expander(pc.platform, expanded=(pc.platform == "TikTok")):
                    if pc.title:
                        st.text_area("Titulo", pc.title, key=f"t_{pc.platform}", height=68)
                    st.text_area("Legenda / descricao", pc.description, key=f"d_{pc.platform}", height=120)

st.caption("Ferramenta de edicao para o Campeonato Dona 30K · voce publica e cumpre o regulamento")
