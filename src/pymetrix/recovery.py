"""Autosaved copies of open sessions, for recovery after a crash.

The desktop app writes a copy of a session with unsaved changes every few minutes to ``recovery`` in
``~/.pymetrix`` (or ``$PYMETRIX_HOME``), as an ordinary project file, and deletes it when the session is saved,
closed or replaced. Each running app also keeps a heartbeat file fresh; a copy whose heartbeat has gone stale was
left by an app that stopped without closing (a crash, or the machine turned off), and the next app offers it back.

    write(session, key)       # the app's autosave: a copy, the session itself unchanged
    orphans()                 # copies left by apps that are no longer running, newest first
    discard(key)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .commands import Session

__all__ = ["Recovery", "discard", "heartbeat", "orphans", "recovery_dir", "write"]

STALE_SECONDS = 600  # a heartbeat older than this: its app is not running


def recovery_dir() -> Path:
    base = os.environ.get("PYMETRIX_HOME") or os.path.join(Path.home(), ".pymetrix")
    return Path(base) / "recovery"


@dataclass(frozen=True)
class Recovery:
    """A copy an app left behind: where it is, the project it came from (None: never saved), and when."""

    key: str
    path: Path
    original: str | None
    title: str
    saved: datetime

    def describe(self) -> str:
        where = Path(self.original).name if self.original else "an unsaved session"
        return f"{self.title or where}, autosaved {self.saved:%d %b %Y at %H:%M}"


def _files(key: str) -> tuple[Path, Path, Path]:
    d = recovery_dir()
    return d / f"{key}.pymx", d / f"{key}.json", d / f"{key}.alive"


def heartbeat(key: str) -> None:
    """Mark the app that owns ``key`` as running (call it every minute or so)."""
    _, _, alive = _files(key)
    alive.parent.mkdir(parents=True, exist_ok=True)
    alive.write_text(str(os.getpid()), encoding="utf-8")


def write(session: Session, key: str, title: str = "") -> Path:
    """Write the autosaved copy of ``session`` (the session's path and unsaved-changes flag are left alone)."""
    from .commands.project import save_project

    project, meta, _ = _files(key)
    project.parent.mkdir(parents=True, exist_ok=True)
    temporary = project.with_name(f"{key}.writing.pymx")
    save_project(session, temporary, as_copy=True)
    os.replace(temporary, project)  # never half a file under the real name
    meta.write_text(json.dumps({"original": str(session.path) if session.path else None, "title": title,
                                "saved": datetime.now().isoformat(timespec="seconds")}), encoding="utf-8")
    heartbeat(key)
    return project


def discard(key: str) -> None:
    """Delete the copy and heartbeat of ``key``."""
    for f in (*_files(key), recovery_dir() / f"{key}.writing.pymx"):
        try:
            f.unlink()
        except FileNotFoundError:
            pass


def orphans(exclude: str | None = None, now: float | None = None) -> list[Recovery]:
    """The copies whose apps are no longer running (a stale or missing heartbeat), newest first."""
    d = recovery_dir()
    if not d.is_dir():
        return []
    now = time.time() if now is None else now
    out = []
    for project in d.glob("*.pymx"):
        key = project.stem
        if key == exclude or key.endswith(".writing"):
            continue
        _, meta, alive = _files(key)
        if alive.exists() and now - alive.stat().st_mtime < STALE_SECONDS:
            continue  # another app is running with it
        info: dict[str, Any] = {}
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        saved = datetime.fromisoformat(info["saved"]) if info.get("saved") else datetime.fromtimestamp(project.stat().st_mtime)
        out.append(Recovery(key, project, info.get("original"), info.get("title", ""), saved))
    return sorted(out, key=lambda r: r.saved, reverse=True)
