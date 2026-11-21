"""What every result shares: its own fields are the record, and a library's result object is optional.

A result keeps the numbers it reports in pymetrix fields (parameters, residuals, statistics, method-specific
tables), read from the library once, when it is made. The library's own result object, when there is one, is
``result.backend_result``: there for power users, compared by nothing, and never saved. Pickling drops it (so a
project's saved results never contain it), and ``codec`` refuses results altogether (they are referred to by name).

Some results need the live library object for a verb (a VECM's ``forecast`` calls statsmodels). Those classes
set ``needs_backend = True``: they keep the object when pickled for another process, and projects recompute them
on opening instead of saving them.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as _np
import pandas as _pd

__all__ = ["BackendResult", "attach_verbs", "coefficients", "deprecated_alias", "observations", "statistics", "to_record"]


class deprecated_alias:
    """A class attribute that reads another attribute, warning that the old name is deprecated::

        result = deprecated_alias("backend_result", "result")
    """

    def __init__(self, new: str, old: str, since: str = "0.1.0a9",
                 why: str = "the library's own object, which is not saved with projects") -> None:
        self.new, self.old, self.since, self.why = new, old, since, why

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        if obj is None:
            return self
        warnings.warn(f"{type(obj).__name__}.{self.old} is deprecated since pymetrix {self.since}; use .{self.new} "
                      f"({self.why})", DeprecationWarning, stacklevel=2)
        return getattr(obj, self.new)


class BackendResult:
    """Mixin for results with an optional library object in ``backend_result`` (see the module docstring)."""

    needs_backend: ClassVar[bool] = False
    _backend_fields: ClassVar[tuple[str, ...]] = ("backend_result",)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        if not self.needs_backend:
            for name in self._backend_fields:
                if name in state:
                    state[name] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def _require_backend(self, what: str) -> Any:
        value = getattr(self, self._backend_fields[0], None)
        if value is None:
            from .errors import PyMetrixError

            raise PyMetrixError(f"{what} needs the library's own result object, which is not kept when a result is saved; "
                                "estimate it again (or reopen its project) to use it")
        return value


# -- the generic verbs ------------------------------------------------------------------------------------------------
#
# Every estimate has summary(), params, resid, fitted, predict(), plot(), and three tidy tables:
#
#     fit.coefficients()     term, estimate, std_error, statistic, p_value, ci_low, ci_high (+ equation, for systems)
#     fit.statistics()       statistic, value, definition (estimation.criteria)
#     fit.observations()     actual, fitted, resid on the data's axes (+ equation)
#
# A verb a kind of result does not have raises NotSupportedError, saying what to use instead.

VERBS = ("summary", "params", "resid", "fitted", "predict", "plot")
TABLES = ("coefficients", "statistics", "observations")
_INSTEAD = {"predict": "forecast", "resid": "innovations", "fitted": "smoothed"}


def _not_supported(result: Any, verb: str) -> Exception:
    from .errors import NotSupportedError

    instead = _INSTEAD.get(verb)
    hint = f"; use {verb_or(result, instead)}" if instead and hasattr(type(result), instead) else ""
    return NotSupportedError(f"a {type(result).__name__} has no {verb}{hint}")


def verb_or(result: Any, name: str) -> str:
    value = getattr(type(result), name, None)
    return f".{name}()" if callable(value) else f".{name}"


def _split_term(result: Any, name: str) -> tuple[str | None, str]:
    """(equation, term) for multi-equation results: VARs name parameters "y: c[-1]", systems "y.const"."""
    if ": " in name and type(result).__name__ in ("FittedVAR", "FittedBVAR"):
        eq, _, term = name.partition(": ")
        return eq, term
    equations = getattr(getattr(result, "system", None), "names", None)
    if equations:
        for eq in sorted(equations, key=len, reverse=True):
            if name.startswith(f"{eq}."):
                return eq, name[len(eq) + 1:]
    return None, name


def coefficients(result: Any, alpha: float = 0.05) -> _pd.DataFrame:
    """The estimates as a tidy table, one row per parameter: ``term``, ``estimate``, ``std_error``, ``statistic``,
    ``p_value``, ``ci_low``, ``ci_high`` (a 1 - alpha interval), and ``equation`` for systems and VARs.
    ``statistic`` is t when the result has residual degrees of freedom, z otherwise."""
    from scipy import stats as sps

    params = getattr(result, "params", None)
    columns = ["term", "estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high"]
    if hasattr(params, "frame") and isinstance(params.frame, _pd.DataFrame):  # rolling fits: one set per window, long
        long = params.frame.stack().rename("estimate").reset_index(level=-1).rename(columns={params.frame.columns.name or "level_1": "term"})
        long.columns = ["term", "estimate"]
        return long
    if params is None or not (hasattr(params, "values") and hasattr(params, "se") and hasattr(params, "names")):
        return _pd.DataFrame(columns=columns)
    names = list(params.names)
    est = _np.asarray(params.values, float)
    se = _np.asarray(params.se, float)
    inference = getattr(result, "inference", None)  # the library's own inference, where kept (ARDL, Markov switching)
    if isinstance(inference, _pd.DataFrame) and list(inference.index) == names and inference.shape[1] >= 4:
        stat, p = inference.iloc[:, 2].to_numpy(float), inference.iloc[:, 3].to_numpy(float)
        df = None
    else:
        df = (getattr(result, "stats", None) or {}).get("df_resid")
        with _np.errstate(divide="ignore", invalid="ignore"):
            stat = est / se
        dist = sps.t(df) if df else sps.norm
        p = 2 * dist.sf(_np.abs(stat))
    if type(params).__name__ == "Draws":  # posterior draws: quantiles, no test statistic
        interval = params.interval(alpha)
        low, high = interval["lower"].to_numpy(float), interval["upper"].to_numpy(float)
        stat = p = _np.full(len(names), _np.nan)
    else:
        q = (sps.t(df) if df else sps.norm).ppf(1 - alpha / 2)
        low, high = est - q * se, est + q * se
    split = [_split_term(result, n) for n in names]
    out = _pd.DataFrame({"term": [t for _, t in split], "estimate": est, "std_error": se, "statistic": stat, "p_value": p,
                         "ci_low": low, "ci_high": high})
    if any(eq is not None for eq, _ in split):
        out.insert(0, "equation", [eq for eq, _ in split])
    return out


def statistics(result: Any) -> _pd.DataFrame:
    """The fit statistics as a tidy table: ``statistic``, ``value`` and ``definition`` (an id of
    ``estimation.criteria.DEFINITIONS``, where pymetrix defines one)."""
    from .estimation.criteria import definitions

    stats = getattr(result, "stats", None)
    if callable(stats):
        stats = None
    if not isinstance(stats, Mapping):
        return _pd.DataFrame(columns=["statistic", "value", "definition"])
    labels = definitions(result)
    rows = [(str(k), float(v) if isinstance(v, (int, float, _np.number)) else _np.nan, labels.get(k)) for k, v in stats.items()
            if isinstance(v, (int, float, _np.number))]
    return _pd.DataFrame(rows, columns=["statistic", "value", "definition"])


def _frame(value: Any) -> _pd.DataFrame | None:
    if value is None:
        return None
    if hasattr(value, "frame") and isinstance(value.frame, _pd.DataFrame):  # a Dataset
        return value.frame
    if isinstance(value, _pd.Series):
        return value.to_frame()
    return value if isinstance(value, _pd.DataFrame) else None


def observations(result: Any) -> _pd.DataFrame:
    """Actual, fitted and residual values on the data's axes: columns ``actual``, ``fitted``, ``resid``; for
    systems and VARs, long, with an ``equation`` column."""
    resid = getattr(result, "resid", None)
    fitted = getattr(result, "fitted", None)
    if isinstance(resid, _pd.Series):
        fitted_s = fitted if isinstance(fitted, _pd.Series) else None
        out = _pd.DataFrame({"actual": (fitted_s + resid) if fitted_s is not None else _np.nan,
                             "fitted": fitted_s if fitted_s is not None else _np.nan, "resid": resid})
        return out
    r, f = _frame(resid), _frame(fitted)
    if r is None:
        raise _not_supported(result, "observations")
    parts = []
    for eq in r.columns:
        fe = f[eq] if f is not None and eq in f.columns else _pd.Series(_np.nan, index=r.index)
        parts.append(_pd.DataFrame({"equation": str(eq), "actual": fe + r[eq], "fitted": fe, "resid": r[eq]}, index=r.index))
    return _pd.concat(parts)


def default_plot(result: Any) -> Any:
    """The result's first figure, as its window shows it."""
    from .figures import Figure
    from .views import view_sections

    for _, value in view_sections(result):
        if isinstance(value, Figure):
            return value
    try:  # no figure of its own: actual and fitted, one panel per equation
        obs = observations(result)
    except Exception:  # noqa: BLE001 - no observations either
        raise _not_supported(result, "plot") from None
    from .figures import Line, Panel

    groups = obs.groupby("equation", sort=False) if "equation" in obs.columns else [("", obs)]
    panels = tuple(Panel((Line(part["actual"], "actual", color="#222222"), Line(part["fitted"], "fitted", style="--")), title=str(eq))
                   for eq, part in groups)
    return Figure(panels)


def default_summary(result: Any) -> Any:
    from .output import Table
    from .views import view_sections

    for _, value in view_sections(result):
        if isinstance(value, Table):
            return value
    raise _not_supported(result, "summary")


def to_record(result: Any) -> dict[str, Any]:
    """The result as pymetrix records it: its kind, statistics with their definitions, the tidy coefficients and
    observations, and provenance; JSON-ready (no library objects). ``saved.save_result`` writes it, and
    ``load_result`` gives it back with the same tables."""
    from ._serial import to_jsonable

    record: dict[str, Any] = {"kind": type(result).__name__, "record_version": 1}
    for name, make in (("coefficients", coefficients), ("statistics", statistics), ("observations", observations)):
        try:
            record[name] = frame_to_json(make(result))
        except Exception:  # noqa: BLE001 - NotSupportedError, or a result without the parts
            continue
    prov = getattr(result, "provenance", None)
    if prov is not None:
        record["provenance"] = to_jsonable(prov)
    return record


def frame_to_json(frame: _pd.DataFrame) -> dict[str, Any]:
    """A table as JSON: columns, rows and row labels (periods and entities as text, with the frequency)."""
    index = frame.index
    levels = [index.get_level_values(i) for i in range(index.nlevels)]
    freq = next((lv.freqstr for lv in levels if isinstance(lv, _pd.PeriodIndex)), None)
    rows = [[None if (isinstance(v, float) and _np.isnan(v)) else (v.item() if isinstance(v, _np.generic) else v) for v in row]
            for row in frame.itertuples(index=False, name=None)]
    return {"columns": [str(c) for c in frame.columns], "index_names": [str(n) if n is not None else None for n in index.names],
            "index": [[str(lv[i]) for lv in levels] for i in range(len(frame))], "freq": freq, "rows": rows}


def frame_from_json(value: Mapping[str, Any]) -> _pd.DataFrame:
    names = value.get("index_names") or [None]
    columns = list(zip(*value["index"])) if value["index"] else [[] for _ in names]
    levels = []
    for name, labels in zip(names, columns):
        labels = list(labels)
        if value.get("freq"):
            try:
                levels.append(_pd.PeriodIndex(labels, freq=value["freq"], name=name))
                continue
            except (ValueError, TypeError):
                pass
        levels.append(_pd.Index(labels, name=name))
    index = levels[0] if len(levels) == 1 else _pd.MultiIndex.from_arrays(levels)
    frame = _pd.DataFrame(value["rows"], columns=value["columns"], index=index if len(value["rows"]) else None)
    for column in frame.columns:  # missing numbers were saved as null: a column of numbers and nulls is numbers again
        values = frame[column]
        if values.dtype == object and all(v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)) for v in values):
            frame[column] = values.astype(float)
    return frame


def _is_estimate(cls: type) -> bool:
    return cls.__name__.startswith("Fitted") or cls.__name__ in ("RollingFit", "QuantileProcess")


def attach_verbs(classes: Any) -> None:
    """Give every estimate class the generic verbs and tidy tables it does not define itself."""
    import dataclasses

    for cls in classes:
        if not isinstance(cls, type) or not _is_estimate(cls):
            continue
        fields = {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set()
        for name, fn in (("coefficients", coefficients), ("statistics", statistics), ("observations", observations),
                         ("to_record", to_record)):
            if not hasattr(cls, name) and name not in fields:
                setattr(cls, name, fn)
        if not hasattr(cls, "summary") and "summary" not in fields:
            cls.summary = default_summary  # type: ignore[attr-defined]
        if not hasattr(cls, "plot") and "plot" not in fields:
            cls.plot = default_plot  # type: ignore[attr-defined]
        for name in ("params", "resid", "fitted"):
            if not hasattr(cls, name) and name not in fields:
                setattr(cls, name, property(lambda self, _n=name: (_ for _ in ()).throw(_not_supported(self, _n))))
        if not hasattr(cls, "predict") and "predict" not in fields:
            def predict(self: Any, *args: Any, **kwargs: Any) -> Any:
                raise _not_supported(self, "predict")
            cls.predict = predict  # type: ignore[attr-defined]
