"""sequence-jacobian, imported on first use, and what every result records about it.

The library's quirks handled here: it declares no dependencies (the plugin does); it prints its solvers' iterations
(``quiet`` keeps them instead); and its defaults are passed explicitly from ``HAOptions`` so that a new release cannot
change a result silently.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
from typing import Any

LIBRARY = "sequence-jacobian"
INSTALL = 'pip install pymetrix-ha (or pip install "pymetrix[ha]"; from the repository, pip install -e packages/pymetrix-ha)'


def library() -> Any:
    """``sequence_jacobian``, or an error that says how to install it."""
    try:
        import sequence_jacobian
    except ImportError as exc:
        raise ImportError(f"heterogeneous-agent models need sequence-jacobian and numba: {INSTALL}") from exc
    return sequence_jacobian


def versions() -> dict[str, str]:
    """The versions of the library and the packages it runs on, for provenance ("" when one is not installed)."""
    out = {}
    for name in (LIBRARY, "numba", "numpy", "scipy"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = ""
    return out


def backend_record() -> dict[str, Any]:
    """``{"name": "sequence-jacobian", "version": ..., "wraps": "sequence_jacobian"}``, with numba's and NumPy's
    versions beside it."""
    v = versions()
    return {"name": LIBRARY, "version": v[LIBRARY], "wraps": "sequence_jacobian",
            "numba": v["numba"], "numpy": v["numpy"], "scipy": v["scipy"]}


def label() -> str:
    """``"sequence-jacobian 1.0.0"``: ``provenance.method["backend"]``."""
    return f"{LIBRARY} {versions()[LIBRARY]}".strip()


@contextlib.contextmanager
def quiet() -> Iterator[list[str]]:
    """What the library prints inside the block, kept as lines (for provenance) instead of printed."""
    lines: list[str] = []
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            yield lines
    finally:
        lines.extend(line.rstrip() for line in buffer.getvalue().splitlines() if line.strip())


def source_fingerprint(*functions: Any) -> str:
    """A short hash of the functions' source text (blocks are code, so the code is part of the model); "" when a
    function's source cannot be read."""
    digest = hashlib.sha256()
    for f in functions:
        try:
            digest.update(inspect.getsource(f).encode())
        except (OSError, TypeError):
            return ""
    return digest.hexdigest()[:16]
