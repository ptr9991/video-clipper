"""
Video Clipper — Windows launcher

- Shows a simple GUI to configure the Groq API Key (first run / when missing)
- Locates bundled FFmpeg
- Starts Streamlit headless
- Opens the default browser when ready
- No console window when launched via pythonw.exe
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(ROOT)

# Force UTF-8 on Windows consoles when present
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _port_open(host: str = "127.0.0.1", port: int = 8501, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def show_api_key_dialog() -> bool:
    """
    Tkinter dialog to enter / test the Groq API key.
    Returns True if a key is now available, False if the user cancelled.
    """
    import tkinter as tk
    from tkinter import messagebox, ttk

    from src.config import get_api_key, set_api_key

    existing = get_api_key() or ""

    root = tk.Tk()
    root.title("Video Clipper — Configuração")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    # Center window
    w, h = 460, 260
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    frame = ttk.Frame(root, padding=20)
    frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(
        frame,
        text="Video Clipper",
        font=("Segoe UI", 16, "bold"),
    ).pack(anchor=tk.W)

    ttk.Label(
        frame,
        text="Informe sua chave da API Groq para continuar.\n"
        "A chave fica salva apenas neste computador.",
        justify=tk.LEFT,
    ).pack(anchor=tk.W, pady=(8, 12))

    ttk.Label(frame, text="Groq API Key:").pack(anchor=tk.W)

    key_var = tk.StringVar(value=existing)
    entry = ttk.Entry(frame, textvariable=key_var, width=52, show="•")
    entry.pack(fill=tk.X, pady=(4, 8))
    entry.focus()

    status_var = tk.StringVar(value="")
    status_lbl = ttk.Label(frame, textvariable=status_var, foreground="#555")
    status_lbl.pack(anchor=tk.W, pady=(0, 8))

    result = {"ok": False}

    def test_connection() -> None:
        key = key_var.get().strip()
        if not key:
            status_var.set("Digite a chave antes de testar.")
            return
        status_var.set("Testando conexão…")
        root.update_idletasks()
        try:
            from groq import Groq

            client = Groq(api_key=key)
            # Lightweight call — list models
            client.models.list()
            status_var.set("✓ Conexão funcionando")
            status_lbl.configure(foreground="#0a7")
        except Exception as exc:
            msg = str(exc)
            if "401" in msg or "authentication" in msg.lower() or "invalid" in msg.lower():
                status_var.set("Chave inválida. Verifique e tente novamente.")
            else:
                status_var.set(f"Erro: {msg[:80]}")
            status_lbl.configure(foreground="#c00")

    def on_continue() -> None:
        key = key_var.get().strip()
        if not key:
            messagebox.showwarning("Chave obrigatória", "Informe a Groq API Key para continuar.")
            return
        set_api_key(key)
        # Also set in current process so Streamlit child can inherit if needed
        os.environ["GROQ_API_KEY"] = key
        result["ok"] = True
        root.destroy()

    def on_cancel() -> None:
        result["ok"] = False
        root.destroy()

    btn_row = ttk.Frame(frame)
    btn_row.pack(fill=tk.X, pady=(8, 0))

    ttk.Button(btn_row, text="Testar conexão", command=test_connection).pack(side=tk.LEFT)
    ttk.Button(btn_row, text="Continuar", command=on_continue).pack(side=tk.RIGHT, padx=(8, 0))
    ttk.Button(btn_row, text="Cancelar", command=on_cancel).pack(side=tk.RIGHT)

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()
    return result["ok"]


def ensure_api_key() -> bool:
    from src.config import get_api_key

    if get_api_key():
        # Still set env so child processes see it
        os.environ["GROQ_API_KEY"] = get_api_key()
        return True
    return show_api_key_dialog()


def ensure_ffmpeg_env() -> None:
    """Point FFMPEG_PATH at the bundled binary when present."""
    from src.config import get_ffmpeg_path

    try:
        path = get_ffmpeg_path()
        os.environ["FFMPEG_PATH"] = path
    except RuntimeError:
        pass  # Streamlit UI will show a friendly message


def start_streamlit() -> subprocess.Popen:
    """Start Streamlit in the background."""
    python = sys.executable
    # Prefer pythonw on Windows to avoid console flash if possible
    if os.name == "nt":
        pythonw = Path(python).with_name("pythonw.exe")
        if pythonw.is_file():
            python = str(pythonw)

    cmd = [
        python,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--server.port",
        "8501",
        "--server.address",
        "127.0.0.1",
    ]

    # Hide console window on Windows
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    return proc


def wait_and_open_browser(timeout: float = 60.0) -> bool:
    """Wait until Streamlit answers, then open the browser."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_open():
            webbrowser.open("http://127.0.0.1:8501")
            return True
        time.sleep(0.4)
    return False


def main() -> int:
    if not ensure_api_key():
        return 1

    ensure_ffmpeg_env()

    proc = start_streamlit()

    # Open browser in a background thread so we can still monitor the process
    opened = {"ok": False}

    def _open() -> None:
        opened["ok"] = wait_and_open_browser()

    t = threading.Thread(target=_open, daemon=True)
    t.start()
    t.join(timeout=65)

    if not opened["ok"]:
        # Fallback: try opening anyway
        webbrowser.open("http://127.0.0.1:8501")

    # Keep launcher alive while Streamlit runs (so the process tree stays up)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
