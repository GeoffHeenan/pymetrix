"""Which pymetrix made a result: its version and, when it runs from a git checkout, the commit and whether the code
differed from it.

    build_info()
    # {"commit": "628beba5...", "dirty": False}                  from a checkout
    # {"commit": None, "note": "installed package (no git checkout): the version identifies the code"}

Provenance (``Provenance.build``), each save in a project's ``save_log`` and a saved result's manifest record it beside
the version. A version alone names a release, but a development checkout runs code between releases: the commit says
which, and ``dirty`` that files under the package differed from it (uncommitted changes), so the commit alone does not
reproduce it.

It is read once per process, cheaply and safely: only when the package's own folder is tracked by a git checkout (an
installed wheel never is), by ``git`` with a time limit, with optional locks off so it never waits on or blocks
another git process. Anything that goes wrong (no git, a slow disk, a broken repository) gives ``"commit": None`` and
a note saying why; it never raises.
"""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path
from typing import Any

__all__ = ["build_info"]

TIMEOUT = 2.0  # seconds for each git command
INSTALLED = "installed package (no git checkout): the version identifies the code"


def _git(folder: Path, *args: str) -> str | None:
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}
    try:
        out = subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=True, timeout=TIMEOUT, env=env,
                             stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _in_checkout(folder: Path) -> bool:
    """Whether a folder above the package holds a ``.git`` (a directory, or a worktree's file): no git run otherwise."""
    return any((p / ".git").exists() for p in (folder, *folder.parents))


@functools.lru_cache(maxsize=1)
def _read(folder: str) -> tuple[tuple[str, Any], ...]:
    here = Path(folder)
    if not _in_checkout(here):
        return (("commit", None), ("note", INSTALLED))
    if _git(here, "ls-files", "--error-unmatch", "--", "__init__.py") is None:  # a checkout, but not of this package
        return (("commit", None), ("note", INSTALLED))
    commit = (_git(here, "rev-parse", "--verify", "HEAD") or "").strip()
    if len(commit) < 40:
        return (("commit", None), ("note", "git checkout, but its commit could not be read"))
    status = _git(here, "status", "--porcelain", "--untracked-files=no", "--", ".")
    if status is None:
        return (("commit", commit), ("dirty", None), ("note", "whether the code differed from the commit could not be read"))
    return (("commit", commit), ("dirty", bool(status.strip())))


def build_info() -> dict[str, Any]:
    """The git commit pymetrix runs from and whether its files differ from it (``{"commit", "dirty"}``), or
    ``{"commit": None, "note": ...}`` for an installed package (or when it cannot be read). Read once per process."""
    return dict(_read(str(Path(__file__).resolve().parent)))
