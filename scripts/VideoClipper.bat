@echo off
REM Development convenience launcher (for source tree, not the portable one)
cd /d "%~dp0.."
if exist ".venv\Scripts\pythonw.exe" (
    ".venv\Scripts\pythonw.exe" scripts\launcher.py
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\launcher.py
) else (
    python scripts\launcher.py
)
