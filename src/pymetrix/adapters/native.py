"""pymetrix's own estimators, described like the wrapped ones: their backend is pymetrix itself.

Provenance from the core sample builders (``estimation.design.provenance``) names this backend unless a library
adapter replaces it (``base.with_backend``), so every result says which code did the numbers.
"""

from __future__ import annotations

from typing import Any

from .base import Adapter

ADAPTER = Adapter("pymetrix")


def backend_record(what: str = "") -> dict[str, Any]:
    """``{"name": "pymetrix", "version": ..., "wraps": what}``, for provenance."""
    from .._version import __version__

    return {"name": "pymetrix", "version": __version__, "wraps": what or "pymetrix (native)"}
