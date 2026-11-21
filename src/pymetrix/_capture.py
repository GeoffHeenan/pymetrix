"""What a result keeps about how it was made: the warnings raised while making it, and a fingerprint of its data.

    with capture() as caught:          # warnings raised inside are recorded, and still shown as usual
        result = estimate(...)
    caught                             # ["ConvergenceWarning: Maximum Likelihood optimization failed to converge", ...]
    fingerprint(frame)                 # "3f9a...": the same data give the same fingerprint, on any machine

Warnings about the libraries themselves (deprecations, future changes) are not recorded: they say nothing about
the result. Everything is re-issued after recording, so filters set by the user still apply.
"""

from __future__ import annotations

import contextlib
import hashlib
import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["capture", "data_fingerprint", "effective_sample", "fingerprint", "provenance_for", "records_warnings", "with_effective_sample"]

_ABOUT_LIBRARIES = (DeprecationWarning, PendingDeprecationWarning, FutureWarning)


@contextlib.contextmanager
def capture() -> Iterator[list[str]]:
    """Record the warnings raised inside the block (as ``"Category: message"``), then re-issue them."""
    caught: list[str] = []
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        yield caught
    seen: set[str] = set()
    for w in records:
        text = f"{w.category.__name__}: {w.message}"
        if not issubclass(w.category, _ABOUT_LIBRARIES) and text not in seen:
            caught.append(text)
            seen.add(text)
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source)


def fingerprint(frame: pd.DataFrame | pd.Series) -> str:
    """A short hash of the values, labels and order of the data a result used."""
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    digest = hashlib.sha256()
    digest.update("|".join(map(str, frame.columns)).encode())
    digest.update("|".join(map(str, frame.index)).encode())
    digest.update(np.ascontiguousarray(frame.to_numpy(dtype=float, na_value=np.nan)).tobytes())
    return digest.hexdigest()[:16]


def data_fingerprint(data: Any, names: Any = None, index: Any = None, before: int = 0) -> str | None:
    """The fingerprint of the columns ``names`` of a dataset (all, without names) over the rows ``index`` (all, without
    it) and the ``before`` rows preceding them (the periods lags reached back to): the data a result used."""
    if data is None:
        return None
    frame = data.frame
    cols = [c for c in (frame.columns if names is None else names) if c in frame.columns]
    if index is not None and len(index):
        pos = frame.index.get_indexer(index)
        pos = pos[pos >= 0]
        if len(pos):
            frame = frame.iloc[max(0, int(pos.min()) - before):int(pos.max()) + 1]
    return fingerprint(frame[cols])


def provenance_for(spec: Any, data: Any, names: Any, index: Any, used: Any, caught: list[str], method: Any,
                converged: bool | None = None, seed: int | None = None, span: Any = None, resolved: Any = None,
                presample: Any = (), where: str | None = None) -> Any:
    """Provenance for results estimated outside ``Equation.fit``: data vintages, sample (and how it was chosen, from
    ``span``), the rows actually used (``index``; ``presample``: rows the library used only for lags or initial
    values), method, fingerprint, warnings, and any choices made automatically (``resolved``)."""
    from ._serial import to_jsonable
    from ._version import __version__
    from .adapters.native import backend_record
    from .metadata import Provenance

    prov = Provenance(data_vintages={n: data.variables[n].vintage for n in names if n in data.variables}, spec=to_jsonable(spec),
                      method=method, sample=None, nobs=len(index), dropped=0,
                      seed=seed, package_version=__version__, data_fingerprint=fingerprint(used), warnings=tuple(caught),
                      converged=converged, resolved=dict(resolved) if resolved else None, backend=backend_record(type(spec).__name__),
                      **_sample_fields(data, span))
    return with_effective_sample(prov, data, index, span=span, where=where, variables=names, presample=presample)


# -- the rows actually used ---------------------------------------------------------------------------------------------

_LISTED = 50  # dropped rows listed one by one; beyond that, only counted


def _runs(periods: Any) -> list[list[str]]:
    """Consecutive periods as [first, last] runs."""
    out: list[list[str]] = []
    prev = None
    for p in sorted(periods):
        if prev is not None and isinstance(p, pd.Period) and p.freq == prev.freq and (p - prev).n == 1:
            out[-1][1] = str(p)
        else:
            out.append([str(p), str(p)])
        prev = p
    return out


def _row_runs(data: Any, used: pd.Index) -> Any:
    t = data.time_axis
    if t is None or not len(used):
        return None
    if not isinstance(used, pd.MultiIndex):
        return _runs(used) if used.name in (t.name, None) and isinstance(used, pd.PeriodIndex) else None
    names = list(used.names)
    if t.name not in names:
        return None
    others = [n for n in names if n != t.name]
    frame = used.to_frame(index=False)
    out = {}
    for key, part in frame.groupby(others, sort=True):
        label = key[0] if isinstance(key, tuple) and len(key) == 1 else key
        out[str(label)] = _runs(pd.PeriodIndex(part[t.name]))
    return out


def effective_sample(data: Any, used: Any, *, span: Any = None, where: str | None = None, variables: Any = (),
                     presample: Any = ()) -> dict[str, Any]:
    """The provenance fields that describe the rows a result used: ``sample`` (first and last period), ``nobs``,
    ``dropped`` (rows of the requested sample not used), ``rows`` (runs of consecutive periods), ``presample`` and
    ``dropped_rows`` with a reason for each: ``"presample"`` (taken by the library for lags or initial values),
    ``"missing"`` (a variable of the result is missing there) or ``"lag"`` (its variables are observed, but a lag,
    lead or difference reaches outside the data)."""
    used = used if isinstance(used, pd.Index) else pd.Index(used)  # (pd.Index of a MultiIndex flattens it to tuples)
    out: dict[str, Any] = {"nobs": len(used)}
    t = data.time_axis
    if t is not None and len(used):
        level = used.get_level_values(t.name) if isinstance(used, pd.MultiIndex) and t.name in used.names else used
        if not isinstance(used, pd.MultiIndex) or t.name in used.names:
            out["sample"] = (str(level.min()), str(level.max()))
    out["rows"] = _row_runs(data, used)
    frame = data.frame
    same_shape = used.nlevels == frame.index.nlevels and (used.nlevels == 1 or list(used.names) == list(frame.index.names))
    if not same_shape:  # results by entity (between estimators): rows are not comparable with the data's
        out["dropped"], out["dropped_rows"] = 0, None
        return out
    requested = frame.index[np.asarray(data.sample_mask(span, where))]
    unused = requested.difference(used, sort=False) if len(requested) else requested
    pre = presample if isinstance(presample, pd.Index) else (pd.Index(presample) if len(presample) else pd.Index([]))
    names = [n for n in dict.fromkeys(variables) if n in frame.columns]
    observed = frame.loc[unused, names].notna().all(axis=1) if names and len(unused) else pd.Series(True, index=unused)
    reasons: dict[str, int] = {}
    listed = []
    for row in unused:
        reason = "presample" if row in pre else ("lag" if bool(observed.get(row, True)) else "missing")
        reasons[reason] = reasons.get(reason, 0) + 1
        if len(listed) < _LISTED:
            listed.append([_label(row), reason])
    out["dropped"] = len(unused)
    out["presample"] = tuple(_label(r) for r in pre if r in requested)
    out["dropped_rows"] = {"by_reason": reasons, "rows": listed, "more": max(0, len(unused) - _LISTED)} if len(unused) else None
    return out


def _label(row: Any) -> str:
    return "/".join(map(str, row)) if isinstance(row, tuple) else str(row)


def with_effective_sample(provenance: Any, data: Any, used: Any, **kw: Any) -> Any:
    """``provenance`` with the fields of ``effective_sample``."""
    from dataclasses import replace

    if provenance is None or data is None or not hasattr(data, "sample_mask"):
        return provenance
    request = getattr(getattr(data, "meta", None), "vintage", None)
    return replace(provenance, **effective_sample(data, used, **kw), vintage_request=dict(request) if request else {"policy": "unknown"})


def records_warnings(fit: Any) -> Any:
    """For an estimator's ``fit``: the warnings raised while it runs are kept on the result's provenance (and still
    shown), as ``Equation.fit`` keeps them."""
    import functools
    from dataclasses import replace

    @functools.wraps(fit)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with capture() as caught:
            result = fit(*args, **kwargs)
        prov = getattr(result, "provenance", None)
        if caught and prov is not None:
            new = tuple(w for w in caught if w not in prov.warnings)
            object.__setattr__(result, "provenance", replace(prov, warnings=(*prov.warnings, *new)))
        return result

    return wrapper


def _sample_fields(data: Any, span: Any) -> dict[str, Any]:
    if not hasattr(data, "sample_record"):
        return {}
    source, rule = data.sample_record(span)
    return {"sample_source": source, "sample_definition": rule}
