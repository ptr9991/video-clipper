"""Tests for Ollama manager helpers without requiring a live server."""

from src.ollama_manager import OllamaStatus, DEFAULT_VISION_MODEL


def test_default_model_name():
    assert DEFAULT_VISION_MODEL == "qwen2.5vl:7b"


def test_status_not_ready_by_default():
    s = OllamaStatus(
        installed=False,
        running=False,
        model_installed=False,
        model_name=DEFAULT_VISION_MODEL,
        message="Ollama não instalado",
    )
    assert s.ready is False


def test_status_ready():
    s = OllamaStatus(
        installed=True,
        running=True,
        model_installed=True,
        model_name=DEFAULT_VISION_MODEL,
    )
    assert s.ready is True
