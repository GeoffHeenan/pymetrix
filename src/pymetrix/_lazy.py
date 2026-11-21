"""Modules imported on first use, to keep ``import pymetrix`` quick.

``scipy.stats`` takes about 0.6 s to import, more than the rest of pymetrix's own modules together, and most sessions
use it only once an estimate is made. Modules write ``from pymetrix._lazy import stats as sps`` and use ``sps`` as
they would ``scipy.stats`` (and ``signal`` as ``scipy.signal``, which imports ``scipy.stats``); the real module is imported on the first attribute read.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["signal", "stats"]


class _LazyModule:
    """Stands in for a module until an attribute is read, then imports it and hands the attribute over."""

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_name", name)

    def __getattr__(self, attr: str) -> Any:
        return getattr(importlib.import_module(object.__getattribute__(self, "_name")), attr)

    def __setattr__(self, attr: str, value: Any) -> None:  # e.g. a test's monkeypatch: set it on the module itself
        setattr(importlib.import_module(object.__getattribute__(self, "_name")), attr, value)

    def __delattr__(self, attr: str) -> None:
        delattr(importlib.import_module(object.__getattribute__(self, "_name")), attr)

    def __dir__(self) -> list[str]:
        return dir(importlib.import_module(object.__getattribute__(self, "_name")))

    def __repr__(self) -> str:
        return f"<lazy module {object.__getattribute__(self, '_name')!r}>"


stats: Any = _LazyModule("scipy.stats")
signal: Any = _LazyModule("scipy.signal")  # imports scipy.stats itself
