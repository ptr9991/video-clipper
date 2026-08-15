"""Unit tests for AI Content Advisor parsing (no live Ollama)."""

from src.local_ai.content_advisor import parse_content_json

SAMPLE = """
{
  "context": {
    "summary": "O artista comenta o lançamento.",
    "topic": "Lançamento",
    "tone": "Direto",
    "hook": "Você não vai acreditar",
    "key_points": ["Lançamento", "Bastidores"],
    "people": [],
    "artists": ["Não identificado."],
    "references": []
  },
  "title": {
    "primary": "Bastidores do lançamento",
    "alternatives": ["O que rolou no estúdio", "Comentário sobre o drop"]
  },
  "platforms": {
    "tiktok": {
      "hook": "Olha isso",
      "caption": "Clip do momento",
      "hashtags": ["#hiphop", "#rap"],
      "cta": "Segue",
      "cover_text": "BASTIDORES",
      "strategy": "Postar à noite"
    },
    "youtube_shorts": {
      "title": "Bastidores do lançamento",
      "description": "Trecho do comentário",
      "hashtags": ["#shorts"],
      "keywords": ["rap"],
      "cta": "Inscreva-se"
    },
    "instagram_reels": {
      "caption": "Momento",
      "hashtags": ["#reels"],
      "cta": "Salva",
      "cover_text": "DROP",
      "strategy": "Stories + feed"
    }
  }
}
"""


def test_parse_ok():
    pkg = parse_content_json(SAMPLE)
    assert pkg.title.primary
    assert pkg.tiktok.caption
    assert pkg.youtube_shorts.title
    assert "#hiphop" in pkg.tiktok.hashtags


def test_parse_markdown_fenced():
    raw = "```json\n" + SAMPLE + "\n```"
    pkg = parse_content_json(raw)
    assert pkg.context.topic == "Lançamento"


def test_copy_all():
    pkg = parse_content_json(SAMPLE)
    text = pkg.copy_all_text()
    assert "TIKTOK" in text
    assert "YOUTUBE" in text
