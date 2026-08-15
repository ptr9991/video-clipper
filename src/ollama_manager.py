"""Manage Ollama install, service, and vision model for local visual AI."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("video_clipper.ollama")

# Base model from ollama.com/library/qwen2.5vl
BASE_VISION_MODEL = "qwen2.5vl:7b"
# Derived model tuned for 8 GB VRAM (RTX 2070): small context = small KV cache
DEFAULT_VISION_MODEL = "qwen2.5vl-2070"
OLLAMA_API = "http://127.0.0.1:11434"
OLLAMA_WINDOWS_URL = "https://ollama.com/download/OllamaSetup.exe"

# Modelfile baked for 8 GB cards — prevents 128k default context allocation
OPTIMIZED_MODELFILE = """FROM qwen2.5vl:7b
PARAMETER num_ctx 4096
PARAMETER temperature 0.1
PARAMETER num_predict 400
PARAMETER num_batch 256
"""


@dataclass
class OllamaStatus:
    installed: bool
    running: bool
    model_installed: bool
    model_name: str
    version: str = ""
    message: str = ""

    @property
    def ready(self) -> bool:
        return self.installed and self.running and self.model_installed


def _no_window_flags() -> int:
    if platform.system() == "Windows":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


def find_ollama_binary() -> Optional[str]:
    path = shutil.which("ollama")
    if path:
        return path
    if platform.system() == "Windows":
        candidates = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
            Path(os.environ.get("PROGRAMFILES", "C:\\Program Files")) / "Ollama" / "ollama.exe",
        ]
        for c in candidates:
            if c.is_file():
                return str(c)
    return None


def is_ollama_running(timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_API}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def list_models() -> list[str]:
    try:
        import json

        with urllib.request.urlopen(f"{OLLAMA_API}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode())
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception as exc:
        log.debug("list_models failed: %s", exc)
        return []


def model_is_installed(model: str) -> bool:
    names = list_models()
    for n in names:
        if n == model or n.startswith(model + ":") or n.split(":")[0] == model.split(":")[0] and model in n:
            if n == model or n.startswith(model):
                return True
    return model in names or any(n.split(":")[0] == model for n in names)


def get_status(model: str = DEFAULT_VISION_MODEL) -> OllamaStatus:
    binary = find_ollama_binary()
    installed = binary is not None
    running = is_ollama_running() if installed else False
    # Ready if optimized model OR base model is present
    has_model = False
    if running:
        has_model = model_is_installed(model) or model_is_installed(BASE_VISION_MODEL)
    version = ""
    if binary:
        try:
            r = subprocess.run(
                [binary, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=_no_window_flags(),
            )
            version = (r.stdout or r.stderr or "").strip()[:80]
        except Exception:
            pass
    msg = "Pronto" if (installed and running and has_model) else ""
    if not installed:
        msg = "Ollama não instalado"
    elif not running:
        msg = "Ollama instalado, mas não está em execução"
    elif not has_model:
        msg = f"Modelo {BASE_VISION_MODEL} não baixado"
    return OllamaStatus(
        installed=installed,
        running=running,
        model_installed=has_model,
        model_name=model,
        version=version,
        message=msg,
    )


def start_ollama() -> bool:
    if is_ollama_running():
        return True
    binary = find_ollama_binary()
    if not binary:
        return False
    try:
        if platform.system() == "Windows":
            app = Path(binary).with_name("ollama app.exe")
            target = str(app) if app.is_file() else binary
            subprocess.Popen(
                [target],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=_no_window_flags(),
            )
        else:
            subprocess.Popen(
                [binary, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        for _ in range(20):
            time.sleep(0.5)
            if is_ollama_running():
                return True
    except Exception as exc:
        log.error("Failed to start Ollama: %s", exc)
    return is_ollama_running()


def download_ollama_installer(dest: Path, progress: Optional[Callable[[float], None]] = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _report(block_num: int, block_size: int, total: int) -> None:
        if progress and total > 0:
            progress(min(1.0, (block_num * block_size) / total))

    urllib.request.urlretrieve(OLLAMA_WINDOWS_URL, str(dest), reporthook=_report)
    return dest


def install_ollama_windows(installer_path: Path) -> bool:
    try:
        subprocess.run(
            [str(installer_path), "/SILENT"],
            timeout=600,
            creationflags=_no_window_flags(),
        )
        time.sleep(3)
        return find_ollama_binary() is not None
    except Exception as exc:
        log.error("Ollama install failed: %s", exc)
        try:
            os.startfile(str(installer_path))  # type: ignore[attr-defined]
        except Exception:
            pass
        return False


def pull_model(
    model: str = BASE_VISION_MODEL,
    progress: Optional[Callable[[str], None]] = None,
) -> bool:
    binary = find_ollama_binary()
    if not binary:
        raise RuntimeError("Ollama não encontrado.")
    if not is_ollama_running():
        if not start_ollama():
            raise RuntimeError("Não foi possível iniciar o Ollama.")

    if progress:
        progress(f"Baixando {model}…")

    proc = subprocess.Popen(
        [binary, "pull", model],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=_no_window_flags(),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if line and progress:
            progress(line[:120])
    code = proc.wait(timeout=3600)
    ok = code == 0 and model_is_installed(model)
    if progress:
        progress("Base OK." if ok else "Falha no download.")
    if ok:
        # Create optimized variant for 8GB
        ensure_optimized_model(progress=progress)
    return ok


def ensure_optimized_model(
    progress: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Create derived model with num_ctx=4096 so KV cache fits in 8 GB VRAM.
    Returns the model name to use for inference.
    """
    binary = find_ollama_binary()
    if not binary:
        raise RuntimeError("Ollama não encontrado.")
    if not is_ollama_running():
        if not start_ollama():
            raise RuntimeError("Ollama não está em execução.")

    if model_is_installed(DEFAULT_VISION_MODEL):
        if progress:
            progress(f"Modelo otimizado já existe: {DEFAULT_VISION_MODEL}")
        return DEFAULT_VISION_MODEL

    if not model_is_installed(BASE_VISION_MODEL):
        if progress:
            progress(f"Baixando base {BASE_VISION_MODEL}…")
        if not pull_model(BASE_VISION_MODEL, progress=progress):
            raise RuntimeError(f"Não foi possível baixar {BASE_VISION_MODEL}")

    if progress:
        progress("Criando variante otimizada para RTX 2070 (num_ctx 4096)…")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".Modelfile", delete=False, encoding="utf-8"
    ) as f:
        f.write(OPTIMIZED_MODELFILE)
        mf_path = f.name

    try:
        r = subprocess.run(
            [binary, "create", DEFAULT_VISION_MODEL, "-f", mf_path],
            capture_output=True,
            text=True,
            timeout=300,
            creationflags=_no_window_flags(),
        )
        if r.returncode != 0:
            log.error("create failed: %s %s", r.stdout, r.stderr)
            # Fallback to base model
            if progress:
                progress("Fallback para modelo base.")
            return BASE_VISION_MODEL
    finally:
        try:
            os.unlink(mf_path)
        except OSError:
            pass

    if progress:
        progress(f"Pronto: {DEFAULT_VISION_MODEL}")
    return DEFAULT_VISION_MODEL if model_is_installed(DEFAULT_VISION_MODEL) else BASE_VISION_MODEL
