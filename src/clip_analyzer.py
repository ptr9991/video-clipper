"""Intelligent clip selection — Groq LLM with local Ollama fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from groq import Groq
from groq import APIError, APIConnectionError, RateLimitError, AuthenticationError

from src.config import (
    ANALYSIS_MODEL,
    MAX_CLIP_DURATION,
    MIN_CLIP_DURATION,
    require_api_key,
)
from src.transcription import Segment, TranscriptionResult
from src.utils import extract_json_from_text, validate_timestamps

log = logging.getLogger("video_clipper.analyzer")


SYSTEM_PROMPT = """Você é um editor profissional de vídeos curtos.
Analise a transcrição com timestamps e escolha o MELHOR trecho contínuo de 30 a 50 segundos.

Critérios:
1. Hook forte nos primeiros 3-5 segundos
2. Desenvolvimento claro
3. Payoff / conclusão dentro do trecho
4. Emoção, surpresa ou opinião forte
5. O trecho deve fazer sentido isolado
6. Preferir 40-50 segundos; nunca passar de 50

Responda SOMENTE com JSON válido, sem markdown e sem texto extra:
{"start": 125.4, "end": 170.2, "duration": 44.8, "reason": "motivo curto", "hook": "gancho", "score": 85}

start e end são floats em segundos. score é 0-100.
"""


@dataclass
class ClipCandidate:
    start: float
    end: float
    duration: float
    reason: str
    hook: str
    score: int
    raw_response: Optional[str] = None


def _build_segments_text(segments: list[Segment]) -> str:
    lines = []
    for seg in segments:
        lines.append(f"[{seg.start:.2f}s - {seg.end:.2f}s] {seg.text}")
    return "\n".join(lines)


def _validate_candidate(
    data: dict[str, Any],
    video_duration: float,
) -> ClipCandidate:
    required = ("start", "end")
    for key in required:
        if key not in data:
            raise ValueError(f"Campo obrigatório ausente no JSON: {key}")

    start = float(data["start"])
    end = float(data["end"])

    if end <= start:
        raise ValueError(f"end ({end}) deve ser maior que start ({start})")

    start, end = validate_timestamps(
        start,
        end,
        video_duration=video_duration,
        max_duration=MAX_CLIP_DURATION,
        min_duration=1.0,
    )
    duration = end - start

    reason = str(data.get("reason", "Trecho selecionado pela IA."))
    hook = str(data.get("hook", ""))
    try:
        score = int(float(data.get("score", 70)))
    except (TypeError, ValueError):
        score = 70
    score = max(0, min(100, score))

    if duration < MIN_CLIP_DURATION:
        reason += (
            f" (Atenção: duração {duration:.1f}s abaixo do ideal de 30s – "
            "foi o melhor trecho contínuo encontrado.)"
        )

    return ClipCandidate(
        start=start,
        end=end,
        duration=duration,
        reason=reason,
        hook=hook,
        score=score,
    )


def _density_fallback(
    transcription: TranscriptionResult,
    video_duration: float,
    target: float = 40.0,
) -> ClipCandidate:
    segments = transcription.segments
    if not segments:
        start = max(0.0, video_duration / 2 - target / 2)
        end = min(video_duration, start + target)
        return ClipCandidate(
            start=start,
            end=end,
            duration=end - start,
            reason="Fallback automático (janela central).",
            hook="",
            score=50,
        )

    best_start = segments[0].start
    best_end = min(segments[0].start + target, video_duration)
    best_score = -1.0

    for i, seg in enumerate(segments):
        window_start = seg.start
        window_end = window_start
        word_count = 0
        for j in range(i, len(segments)):
            s = segments[j]
            if s.end - window_start > MAX_CLIP_DURATION:
                break
            window_end = s.end
            word_count += len(s.text.split())
            duration = window_end - window_start
            if duration < MIN_CLIP_DURATION:
                continue
            density = word_count / max(duration, 1.0)
            length_bonus = 1.0 - abs(duration - target) / target
            score = density * (0.7 + 0.3 * max(0.0, length_bonus))
            if score > best_score:
                best_score = score
                best_start = window_start
                best_end = window_end

    best_start, best_end = validate_timestamps(
        best_start,
        best_end,
        video_duration=video_duration,
        max_duration=MAX_CLIP_DURATION,
        min_duration=1.0,
    )
    return ClipCandidate(
        start=best_start,
        end=best_end,
        duration=best_end - best_start,
        reason="Seleção por densidade de fala (fallback).",
        hook="",
        score=55,
    )


def _call_llm(
    client: Groq,
    model: str,
    user_prompt: str,
    *,
    use_json_mode: bool = True,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 600,
    }
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content or ""


def analyze_best_clip(
    transcription: TranscriptionResult,
    video_duration: float,
    model: str = ANALYSIS_MODEL,
    prefer_local: bool = False,
) -> ClipCandidate:
    """
    Prefer Groq; on rate-limit / connection failure use local Ollama.
    Set prefer_local=True to skip Groq analysis entirely.
    """
    if not transcription.segments and not transcription.text:
        raise ValueError("Transcrição vazia – não é possível analisar.")

    if video_duration <= 0:
        raise ValueError("Duração do vídeo inválida.")

    if prefer_local:
        from src.local_ai.clip_selector import analyze_best_clip_local

        return analyze_best_clip_local(transcription, video_duration)

    if transcription.segments:
        content = _build_segments_text(transcription.segments)
    else:
        content = transcription.text

    max_chars = 60000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n...[transcrição truncada]..."

    user_prompt = (
        f"Duração total do vídeo: {video_duration:.1f} segundos.\n\n"
        f"Transcrição com timestamps:\n\n{content}\n\n"
        "Escolha o melhor trecho de 30 a 50 segundos. Responda só com JSON."
    )

    try:
        client = Groq(api_key=require_api_key())
        log.info("Sending analysis request to Groq model=%s", model)

        try:
            raw = _call_llm(client, model, user_prompt, use_json_mode=True)
        except APIError:
            raw = _call_llm(client, model, user_prompt, use_json_mode=False)

        data = extract_json_from_text(raw)
        if data is None:
            raw = _call_llm(
                client,
                model,
                user_prompt + "\n\nAPENAS JSON.",
                use_json_mode=False,
            )
            data = extract_json_from_text(raw)

        if data is None:
            log.warning("Groq JSON failed — trying local Ollama")
            from src.local_ai.clip_selector import analyze_best_clip_local

            return analyze_best_clip_local(transcription, video_duration)

        candidate = _validate_candidate(data, video_duration)
        candidate.raw_response = raw
        return candidate

    except RateLimitError:
        log.warning("Groq rate limit — local Ollama fallback")
        from src.local_ai.clip_selector import analyze_best_clip_local

        cand = analyze_best_clip_local(transcription, video_duration)
        cand.reason = (cand.reason or "") + " [Groq limite → Ollama local]"
        return cand

    except APIConnectionError:
        log.warning("Groq connection failed — local fallback")
        from src.local_ai.clip_selector import analyze_best_clip_local

        return analyze_best_clip_local(transcription, video_duration)

    except AuthenticationError as exc:
        raise RuntimeError("Chave da API Groq inválida.") from exc

    except APIError as exc:
        # Any other API error: try local before giving up
        log.warning("Groq API error %s — local fallback", exc)
        try:
            from src.local_ai.clip_selector import analyze_best_clip_local

            return analyze_best_clip_local(transcription, video_duration)
        except Exception:
            raise RuntimeError(f"Erro da API Groq: {exc}") from exc


def parse_and_validate_json(
    text: str,
    video_duration: float,
) -> ClipCandidate:
    data = extract_json_from_text(text)
    if data is None:
        raise ValueError("JSON inválido")
    return _validate_candidate(data, video_duration)
