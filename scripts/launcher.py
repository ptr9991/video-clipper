"""
Video Clipper — Windows launcher

- Shows a native Windows dialog to configure the Groq API Key (first run)
- Locates bundled FFmpeg
- Starts Streamlit headless
- Opens the default browser when ready
- No dependency on tkinter (Python embeddable does not ship it)
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
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


def _powershell_input_dialog(existing: str = "") -> str | None:
    """
    Native Windows Forms dialog via PowerShell.
    Returns the entered key, or None if cancelled / empty.
    Works without tkinter (required for Python embeddable).
    """
    # Escape for embedding inside a single-quoted PowerShell string
    safe_existing = (existing or "").replace("'", "''")

    ps_script = f"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$form = New-Object System.Windows.Forms.Form
$form.Text = 'Video Clipper - Configuracao'
$form.Size = New-Object System.Drawing.Size(480, 240)
$form.StartPosition = 'CenterScreen'
$form.FormBorderStyle = 'FixedDialog'
$form.MaximizeBox = $false
$form.MinimizeBox = $false
$form.TopMost = $true

$label = New-Object System.Windows.Forms.Label
$label.Location = New-Object System.Drawing.Point(20, 20)
$label.Size = New-Object System.Drawing.Size(420, 40)
$label.Text = 'Informe sua chave da API Groq para continuar.`nA chave fica salva apenas neste computador.'
$form.Controls.Add($label)

$keyLabel = New-Object System.Windows.Forms.Label
$keyLabel.Location = New-Object System.Drawing.Point(20, 70)
$keyLabel.Size = New-Object System.Drawing.Size(100, 20)
$keyLabel.Text = 'Groq API Key:'
$form.Controls.Add($keyLabel)

$textBox = New-Object System.Windows.Forms.TextBox
$textBox.Location = New-Object System.Drawing.Point(20, 95)
$textBox.Size = New-Object System.Drawing.Size(420, 25)
$textBox.UseSystemPasswordChar = $true
$textBox.Text = '{safe_existing}'
$form.Controls.Add($textBox)

$okButton = New-Object System.Windows.Forms.Button
$okButton.Location = New-Object System.Drawing.Point(250, 140)
$okButton.Size = New-Object System.Drawing.Size(90, 30)
$okButton.Text = 'Continuar'
$okButton.DialogResult = [System.Windows.Forms.DialogResult]::OK
$form.AcceptButton = $okButton
$form.Controls.Add($okButton)

$cancelButton = New-Object System.Windows.Forms.Button
$cancelButton.Location = New-Object System.Drawing.Point(350, 140)
$cancelButton.Size = New-Object System.Drawing.Size(90, 30)
$cancelButton.Text = 'Cancelar'
$cancelButton.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
$form.CancelButton = $cancelButton
$form.Controls.Add($cancelButton)

$result = $form.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {{
    Write-Output $textBox.Text
}} else {{
    Write-Output ''
}}
"""

    # Write script to a temp file to avoid quoting hell on the command line
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ps1", delete=False, encoding="utf-8-sig"
    ) as f:
        f.write(ps_script)
        script_path = f.name

    try:
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

        proc = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                script_path,
            ],
            capture_output=True,
            text=True,
            timeout=300,
            creationflags=creationflags,
        )
        key = (proc.stdout or "").strip()
        return key if key else None
    except Exception:
        return None
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def _console_input_dialog(existing: str = "") -> str | None:
    """Fallback when PowerShell dialog is unavailable."""
    print()
    print("=" * 50)
    print("  Video Clipper — Configuração")
    print("=" * 50)
    print()
    print("Informe sua chave da API Groq para continuar.")
    print("A chave fica salva apenas neste computador.")
    print()
    if existing:
        print(f"(Já existe uma chave salva. Pressione Enter para manter.)")
    try:
        key = input("Groq API Key: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not key and existing:
        return existing
    return key if key else None


def show_api_key_dialog() -> bool:
    """
    Prompt for Groq API key using a native Windows dialog (or console fallback).
    Returns True if a key is now available, False if the user cancelled.
    """
    from src.config import get_api_key, set_api_key

    existing = get_api_key() or ""

    key: str | None = None
    if os.name == "nt":
        key = _powershell_input_dialog(existing)

    if key is None and sys.stdin and sys.stdin.isatty():
        key = _console_input_dialog(existing)

    if not key:
        return False

    set_api_key(key)
    os.environ["GROQ_API_KEY"] = key
    return True


def ensure_api_key() -> bool:
    from src.config import get_api_key

    key = get_api_key()
    if key:
        os.environ["GROQ_API_KEY"] = key
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
