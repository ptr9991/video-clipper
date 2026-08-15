"""Render a visual multi-track timeline from ProjectState (HTML only)."""

from __future__ import annotations

from src.editor.models import ProjectState


def render_timeline_html(state: ProjectState) -> str:
    dur = max(state.timeline_duration, 0.01)
    ph = max(0.0, min(100.0, (state.playhead / dur) * 100))

    def blocks(items: list, start_attr: str, end_attr: str, label_attr: str, color: str) -> str:
        html = []
        for it in items:
            s = getattr(it, start_attr)
            e = getattr(it, end_attr)
            left = (s / dur) * 100
            width = max(0.4, ((e - s) / dur) * 100)
            label = getattr(it, label_attr, "")[:18]
            html.append(
                f'<div class="tl-block" style="left:{left:.2f}%;width:{width:.2f}%;'
                f'background:{color}" title="{label}">{label}</div>'
            )
        return "".join(html)

    video_w = 100.0
    caps = blocks(state.captions, "start", "end", "text", "#3d5a80")
    texts = blocks(state.texts, "start", "end", "text", "#6a994e")

    return f"""
<div class="tl-wrap">
  <div class="tl-row"><span class="tl-lab">VIDEO</span>
    <div class="tl-track"><div class="tl-block" style="left:0;width:{video_w}%;background:#c8f54233;border:1px solid #c8f54288">clip</div>
    <div class="tl-playhead" style="left:{ph:.2f}%"></div></div></div>
  <div class="tl-row"><span class="tl-lab">AUDIO</span>
    <div class="tl-track"><div class="tl-block" style="left:0;width:{video_w}%;background:#4a556840;border:1px solid #718096">∿ audio</div>
    <div class="tl-playhead" style="left:{ph:.2f}%"></div></div></div>
  <div class="tl-row"><span class="tl-lab">TEXT</span>
    <div class="tl-track">{texts or '&nbsp;'}<div class="tl-playhead" style="left:{ph:.2f}%"></div></div></div>
  <div class="tl-row"><span class="tl-lab">CAPS</span>
    <div class="tl-track">{caps or '&nbsp;'}<div class="tl-playhead" style="left:{ph:.2f}%"></div></div></div>
  <div class="tl-meta">{state.playhead:.2f}s / {dur:.2f}s · {state.aspect.value} · {state.fps:.0f}fps</div>
</div>
<style>
.tl-wrap {{ background:#111; border:1px solid #2a2a2e; border-radius:10px; padding:10px 12px; margin:8px 0 14px; }}
.tl-row {{ display:flex; align-items:center; gap:8px; margin:4px 0; }}
.tl-lab {{ width:48px; font-size:10px; color:#8a8a8a; letter-spacing:.06em; }}
.tl-track {{ position:relative; flex:1; height:22px; background:#1a1a1d; border-radius:4px; overflow:hidden; }}
.tl-block {{ position:absolute; top:2px; height:18px; border-radius:3px; font-size:9px; color:#eee;
  white-space:nowrap; overflow:hidden; padding:0 4px; line-height:18px; }}
.tl-playhead {{ position:absolute; top:0; bottom:0; width:2px; background:#c8f542; z-index:5; }}
.tl-meta {{ font-size:11px; color:#8a8a8a; margin-top:6px; }}
</style>
"""
