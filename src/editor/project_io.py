"""Load / save project.json (paths only — never embeds video bytes)."""

from __future__ import annotations

import json
from pathlib import Path

from src.editor.models import ProjectState


def save_project(state: ProjectState, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state.validate()
    path.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_project(path: Path) -> ProjectState:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Project not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Invalid project.json")
    return ProjectState.from_dict(data)
