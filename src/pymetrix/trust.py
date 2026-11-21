"""Which projects may run their typed code when opened.

Opening a project replays its steps. Dialog steps only run pymetrix's own commands,
but steps typed in the Code or Immediate windows are Python, which can do anything a
program can. So a project's typed code runs on opening only when it is trusted:
projects you saved yourself (with all their code run), or ones you chose to trust
after reading the code. Trust is remembered by a fingerprint of the code itself, so
re-saving a project keeps it, and any change to the code asks again.

The fingerprints are kept in ``trusted_projects.json`` in ``~/.pymetrix`` (or the
folder named by the ``PYMETRIX_HOME`` environment variable).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

__all__ = ["code_fingerprint", "forget", "is_trusted", "remember", "trust_file"]


def trust_file() -> Path:
    base = os.environ.get("PYMETRIX_HOME") or os.path.join(Path.home(), ".pymetrix")
    return Path(base) / "trusted_projects.json"


def code_fingerprint(steps: Sequence[str]) -> str:
    """A fingerprint of a project's typed code, step by step."""
    return hashlib.sha256(json.dumps(list(steps), ensure_ascii=False).encode("utf-8")).hexdigest()


def _read() -> dict[str, dict[str, str]]:
    try:
        data = json.loads(trust_file().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(entries: dict[str, dict[str, str]]) -> None:
    path = trust_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(entries, indent=1)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(5):  # sync tools such as OneDrive briefly lock files
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    path.write_text(text, encoding="utf-8")
    tmp.unlink(missing_ok=True)


def is_trusted(fingerprint: str) -> bool:
    return fingerprint in _read()


def remember(fingerprint: str, label: str = "") -> None:
    """Trust code with this fingerprint from now on."""
    entries = _read()
    entries[fingerprint] = {"project": label, "trusted": datetime.now().isoformat(timespec="seconds")}
    try:
        _write(entries)
    except OSError:  # a read-only home: trust lasts for this decision only
        pass


def forget(fingerprint: str) -> None:
    entries = _read()
    if entries.pop(fingerprint, None) is not None:
        _write(entries)
