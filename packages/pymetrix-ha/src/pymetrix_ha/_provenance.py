"""Provenance for the plugin's results: the model, the grids, every option passed, and what the library said."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

import numpy as np

from ._library import backend_record, label


def grids(raw_ss: Any, het_names: list[str]) -> dict[str, Any]:
    """Each het block's grids and Markov matrices at the steady state (``*_grid``, ``Pi``): their shape, range and a
    hash of their values, so a result says which grid it was solved on (``grids.agrid`` and ``grids.asset_grid``
    give different points for the same arguments)."""
    out: dict[str, Any] = {}
    internals = getattr(raw_ss, "internals", {}) or {}
    for name in het_names:
        found = {}
        for key, value in (internals.get(name) or {}).items():
            if not (str(key).endswith("_grid") or str(key) == "Pi"):
                continue
            arr = np.ascontiguousarray(np.asarray(value, dtype=float))
            found[str(key)] = {"shape": list(arr.shape), "min": float(arr.min()), "max": float(arr.max()),
                               "sha256": hashlib.sha256(arr.tobytes()).hexdigest()[:16]}
        if found:
            out[name] = found
    return out


def make(model: Any, what: str, calls: dict[str, Any], caught: tuple[str, ...], printed: tuple[str, ...],
         resolved: dict[str, Any], steady: Any = None, converged: bool | None = True, spec: dict[str, Any] | None = None) -> Any:
    """A ``Provenance`` for one of the plugin's results. ``steady``: the ``HASteadyState`` it was computed at, whose
    grids and solve are recorded with it."""
    from pymetrix import __version__
    from pymetrix.metadata import Provenance

    from .blocks import het_blocks

    full_spec = model.spec()
    raw = steady.raw if steady is not None else None
    if raw is not None:
        full_spec["grids"] = grids(raw, [str(b.name) for b in het_blocks(model.blocks)])
    full_spec.update(spec or {})
    if steady is not None and what != "steady state":
        resolved = {"steady_state": {"evaluations": steady.evaluations, "max_target_residual": steady.max_residual},
                    **resolved}
    return Provenance(data_vintages={}, spec=full_spec,
                      method={"type": what, "backend": label(), "plugin": "pymetrix-ha"},
                      sample=None, nobs=0, dropped=0, seed=None, package_version=__version__, warnings=tuple(caught),
                      converged=converged, resolved={"options": asdict(model.options), **resolved},
                      backend=backend_record(), options={"stages": calls, "backend_options": {}},
                      messages=tuple(printed))
