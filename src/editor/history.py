"""Undo / redo stack over ProjectState snapshots (data only)."""

from __future__ import annotations

from typing import Optional

from src.editor.models import ProjectState


class HistoryStack:
    def __init__(self, initial: ProjectState, limit: int = 50) -> None:
        self._limit = max(1, limit)
        self._undo: list[ProjectState] = [initial.clone()]
        self._redo: list[ProjectState] = []

    @property
    def current(self) -> ProjectState:
        return self._undo[-1]

    def push(self, state: ProjectState) -> ProjectState:
        self._undo.append(state.clone())
        if len(self._undo) > self._limit:
            self._undo.pop(0)
        self._redo.clear()
        return self.current

    def can_undo(self) -> bool:
        return len(self._undo) > 1

    def can_redo(self) -> bool:
        return len(self._redo) > 0

    def undo(self) -> Optional[ProjectState]:
        if not self.can_undo():
            return None
        cur = self._undo.pop()
        self._redo.append(cur)
        return self.current.clone()

    def redo(self) -> Optional[ProjectState]:
        if not self.can_redo():
            return None
        state = self._redo.pop()
        self._undo.append(state)
        return self.current.clone()
