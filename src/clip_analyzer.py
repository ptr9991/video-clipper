"""Intelligent clip selection using Groq LLM."""

from __future__ import annotations

import json
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


SYSTEM_PROMPT = """Você é um editor profissional de vídeos curtos especializado em identificar momentos de alto potencial de retenção e viralização em vídeos longos.

Sua tarefa é analisar a transcrição (com timestamps) de um vídeo e escolher o MELHOR trecho contínuo que funcione como um clipe independente de 30 a 50 segundos.

Critérios prioritários (em ordem de importância):
1. Hook forte nos primeiros 3-5 segundos (afirmação inesperada, pergunta provocativa, revelação, tensão).
2. Desenvolvimento claro e ritmo.
3. Payoff / conclusão satisfatória dentro do trecho.
4. Densidade de informação, emoção, surpresa ou opinião forte.
5. O trecho deve fazer sentido isolado, sem depender de contexto anterior.
6. Preferir 40-50 segundos quando houver conteúdo suficiente; 30-40s se o momento for mais conciso e impactante.
7. Nunca ultrapassar 50 segundos.
8. Evitar introduções longas, saudações, silêncios, frases incompletas ou trechos que precisem de mais de 50s para fazer sentido.

Você receberá a lista de segmentos com start/end em segundos e o texto de cada um.

Responda EXCLUSIVAMENTE com um JSON válido no formato:

{
  "start": 125.4,
  "end": 170.2,
  "duration": 44.8,
  "reason": "Explicação curta em português do porquê este trecho é o mais engajante.",
  "hook": "Descrição breve do gancho inicial.",
  "score": 92
}

Regras do JSON:
- start e end devem ser números (float) em segundos, baseados nos timestamps fornecidos.
- duration = end - start (deve estar entre 30 e 50 quando possível).
- score de 0 a 100 (indicativo, não científico).
- Não inclua markdown, comentários ou texto fora do JSON.
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
    """Format segments for the prompt."""
    lines = []
    for seg in segments:
        lines.append(f"[{seg.start:.2f}s - {seg.end:.2f}s] {seg.text}")
    return "\n".join(lines)


def _validate_candidate(
    data: dict[str, Any],
    video_duration: float,
) -> ClipCandidate:
    """Validate and normalise the JSON returned by the LLM."""
    required = ("start", "end")
    for key in required:
        if key not in data:
            raise ValueError(f"Campo obrigatório ausente no JSON: {key}")

    start = float(data["start"])
    end = float(data["end"])

    if end <= start:
        raise ValueError(f"end ({end}) deve ser maior que start ({start})")

    # Clamp to video bounds and max duration
    start, end = validate_timestamps(
        start,
        end,
        video_duration=video_duration,
        max_duration=MAX_CLIP_DURATION,
        min_duration=1.0,
    )
    duration = end - start

    # Soft preference for >= 30s – if shorter we still accept but note it
    reason = str(data.get("reason", "Trecho selecionado pela IA."))
    hook = str(data.get("hook", ""))
    score = int(data.get("score", 70))
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


def analyze_best_clip(
    transcription: TranscriptionResult,
    video_duration: float,
    model: str = ANALYSIS_MODEL,
) -> ClipCandidate:
    """
    Send the transcription to a Groq LLM and obtain the best 30-50s clip.
    """
    if not transcription.segments and not transcription.text:
        raise ValueError("Transcrição vazia – não é possível analisar.")

    if video_duration <= 0:
        raise ValueError("Duração do vídeo inválida.")

    # Prefer segment-level data; fall back to full text
    if transcription.segments:
        content = _build_segments_text(transcription.segments)
    else:
        content = transcription.text

    # Truncate extremely long transcripts to stay within context
    max_chars = 60000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n...[transcrição truncada]..."

    user_prompt = (
        f"Duração total do vídeo: {video_duration:.1f} segundos.\n\n"
        f"Transcrição com timestamps:\n\n{content}\n\n"
        "Escolha o melhor trecho de 30 a 50 segundos e responda apenas com o JSON."
    )

    client = Groq(api_key=require_api_key())
    log.info("Sending analysis request to model=%s", model)

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        raw = completion.choices[0].message.content or ""
        log.debug("LLM raw response: %s", raw[:500])

    except AuthenticationError as exc:
        raise RuntimeError("Chave da API Groq inválida.") from exc
    except RateLimitError as exc:
        raise RuntimeError(
            "Limite de requisições da API Groq atingido. Tente novamente mais tarde."
        ) from exc
    except APIConnectionError as exc:
        raise RuntimeError("Falha de conexão com a API Groq.") from exc
    except APIError as exc:
        raise RuntimeError(f"Erro da API Groq: {exc}") from exc

    data = extract_json_from_text(raw)
    if data is None:
        log.error("Could not parse JSON from LLM: %s", raw[:300])
        raise RuntimeError(
            "A IA retornou uma resposta que não pôde ser interpretada como JSON. "
            "Tente novamente."
        )

    candidate = _validate_candidate(data, video_duration)
    candidate.raw_response = raw
    log.info(
        "Best clip selected: %.2f – %.2f (%.1fs) score=%d",
        candidate.start,
        candidate.end,
        candidate.duration,
        candidate.score,
    )
    return candidate


def parse_and_validate_json(
    text: str,
    video_duration: float,
) -> ClipCandidate:
    """Public helper used by unit tests."""
    data = extract_json_from_text(text)
    if data is None:
        raise ValueError("JSON inválido")
    return _validate_candidate(data, video_duration)
