"""Manage Ollama install, service, and vision model for local visual AI."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("video_clipper.ollama")

# Official model tag (verified on ollama.com/library/qwen2.5vl)
DEFAULT_VISION_MODEL = "qwen2.5vl:7b"
OLLAMA_API = "http://127.0.0.1:11434"
# Official Windows installer (ollama.com/download)
OLLAMA_WINDOWS_URL = "https://ollama.com/download/OllamaSetup.exe"


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
        import urllib.request

        req = urllib.request.Request(f"{OLLAMA_API}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def list_models() -> list[str]:
    try:
        import json
        import urllib.request

        with urllib.request.urlopen(f"{OLLAMA_API}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode())
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception as exc:
        log.debug("list_models failed: %s", exc)
        return []


def model_is_installed(model: str = DEFAULT_VISION_MODEL) -> bool:
    names = list_models()
    # Match exact or prefix (qwen2.5vl:7b vs qwen2.5vl:7b-...)
    base = model.split(":")[0]
    for n in names:
        if n == model or n.startswith(model) or n.startswith(base + ":"):
            if model in n or n.startswith(model):
                return True
            # also accept bare tag family with :7b
            if ":7b" in model and ":7b" in n and base in n:
                return True
    return model in names


def get_status(model: str = DEFAULT_VISION_MODEL) -> OllamaStatus:
    binary = find_ollama_binary()
    installed = binary is not None
    running = is_ollama_running() if installed else False
    has_model = model_is_installed(model) if running else False
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
        msg = f"Modelo {model} não baixado"
    return OllamaStatus(
        installed=installed,
        running=running,
        model_installed=has_model,
        model_name=model,
        version=version,
        message=msg,
    )


def start_ollama() -> bool:
    """Try to start Ollama service (Windows app / ollama serve)."""
    if is_ollama_running():
        return True
    binary = find_ollama_binary()
    if not binary:
        return False
    try:
        # On Windows, launching ollama app usually starts the tray service
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
    """Download official Windows OllamaSetup.exe."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = OLLAMA_WINDOWS_URL
    log.info("Downloading Ollama from %s", url)

    def _report(block_num: int, block_size: int, total: int) -> None:
        if progress and total > 0:
            progress(min(1.0, (block_num * block_size) / total))

    urllib.request.urlretrieve(url, str(dest), reporthook=_report)
    return dest


def install_ollama_windows(installer_path: Path) -> bool:
    """Run OllamaSetup.exe silently if possible."""
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
        # Fallback: open installer for user
        try:
            os.startfile(str(installer_path))  # type: ignore[attr-defined]
        except Exception:
            pass
        return False


def pull_model(
    model: str = DEFAULT_VISION_MODEL,
    progress: Optional[Callable[[str], None]] = None,
) -> bool:
    """Pull vision model via `ollama pull`."""
    binary = find_ollama_binary()
    if not binary:
        raise RuntimeError("Ollama não encontrado.")
    if not is_ollama_running():
        if not start_ollama():
            raise RuntimeError("Não foi possível iniciar o Ollama.")

    if progress:
        progress(f"Baixando {model}… (pode levar vários minutos)")

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
        log.debug("pull: %s", line)
    code = proc.wait(timeout=3600)
    ok = code == 0 and model_is_installed(model)
    if progress:
        progress("Modelo pronto." if ok else "Falha ao baixar o modelo.")
    return ok
