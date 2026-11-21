"""Progress and cancelling for long steps.

A long computation (MCMC chains, bootstrap draws, backtest origins, sweeps, rolling windows) calls ``report`` as it
goes: how far it is (0 to 1) and what it is doing. When the desktop app runs a step in the background it opens a
``task``; ``report`` then updates that task's progress bar and, if Cancel was pressed, stops the step by raising
``Cancelled``. Outside a task (scripts, notebooks, tests) ``report`` does nothing and costs almost nothing.

Cancelling is cooperative: a step stops where it reports. A computation that never reports runs to its end.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from .errors import PyMetrixError

__all__ = ["Cancelled", "Task", "check", "report", "task"]


class Cancelled(PyMetrixError):
    """A step stopped because Cancel was pressed."""


@dataclass
class Task:
    """A running step: how far it is, what it is doing, and whether it should stop."""

    fraction: float | None = None  # 0 to 1; None: not known
    message: str = ""
    cancelled: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def cancel(self) -> None:
        self.cancelled = True


_current: Task | None = None


def report(fraction: float | None = None, message: str = "") -> None:
    """How far the running step is (0 to 1) and, optionally, what it is doing; stops it if Cancel was pressed."""
    t = _current
    if t is None:
        return
    if t.cancelled:
        raise Cancelled("cancelled")
    with t.lock:
        if fraction is not None:
            t.fraction = max(0.0, min(1.0, float(fraction)))
        if message:
            t.message = message


def check() -> None:
    """Stop the running step here if Cancel was pressed."""
    t = _current
    if t is not None and t.cancelled:
        raise Cancelled("cancelled")


@contextmanager
def task() -> Iterator[Task]:
    """The step run inside reports to this task (one at a time)."""
    global _current
    previous, _current = _current, Task()
    try:
        yield _current
    finally:
        _current = previous
