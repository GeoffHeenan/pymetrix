"""pymetrix with NumPy, pandas, SciPy and statsmodels: out to ordinary Python objects, and back in.

Out: ``Dataset.to_frame`` and ``to_numpy`` give the series of a dataset on its sample (the same rule as every pymetrix
command: an explicit span, else the active sample), with an explicit missing-value policy; ``Dataset.design(y, x)``
gives a labelled ``y`` and ``X`` for statsmodels, SciPy or scikit-learn, built by the same code as pymetrix's own
estimators (formula expressions, lags within entity, sample, missing values, a constant). Each frame says in
``.attrs["pymetrix"]`` how its rows were chosen.

In: ``Dataset.add_series`` adds an array or a pandas series (refusing an existing name unless ``overwrite=True``, and
a series of another frequency); ``apply_np``, ``apply_pd`` and ``transform_each`` add series computed by your own
functions, recording that they came from user code (and ``source_expr`` when given); ``Table.from_frame`` wraps any
DataFrame; ``FittedEquation.from_statsmodels`` turns a statsmodels regression into an ordinary pymetrix result, with
its views, provenance and generated code like any other.

Nothing here imports Qt: it is for scripts, notebooks and headless runs.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .errors import DataError, SpecificationError

if TYPE_CHECKING:
    from .data import Dataset
    from .estimation import FittedEquation

__all__ = ["Statsmodels", "add_series", "apply_np", "apply_pd", "design", "from_statsmodels", "to_frame", "transform_each"]

_MISSING = ("keep", "drop", "raise")


def _names(data: Dataset, names: Any) -> list[str]:
    if names is None:
        return list(data.names)
    if isinstance(names, str):
        return [n.strip() for n in names.split(",") if n.strip()]
    return [str(n) for n in names]  # a list, or a Selection (iterable of names)


def _rows(data: Dataset, sample: Any) -> tuple[np.ndarray, str, Any]:
    """The rows for ``sample``, with how they were chosen: ``None`` follows the active sample (as commands do),
    ``"all"`` takes every row, a ``(start, end)`` pair or a ``Sample`` is explicit."""
    if isinstance(sample, str) and sample == "all":
        return np.ones(len(data.frame), dtype=bool), "all observations", None
    source, rule = data.sample_record(sample)
    return np.asarray(data.sample_mask(sample)), source, rule


def to_frame(data: Dataset, names: Any = None, sample: Any = None, missing: str = "keep") -> pd.DataFrame:
    """The series ``names`` (default all; a list, a comma-separated string or a ``Selection``) on the sample's rows, as a
    DataFrame with the dataset's index (periods, entities). ``missing``: keep the rows, drop rows with any missing value,
    or raise."""
    if missing not in _MISSING:
        raise SpecificationError(f"missing is one of {', '.join(_MISSING)}, not {missing!r}")
    cols = _names(data, names)
    unknown = [c for c in cols if c not in data.frame.columns]
    if unknown:
        raise DataError(f"no series {', '.join(unknown)} in the dataset; it has {', '.join(data.names[:12])}")
    mask, source, rule = _rows(data, sample)
    frame = data.frame.loc[mask, cols].copy()
    incomplete = frame.isna().any(axis=1)
    if missing == "raise" and incomplete.any():
        raise DataError(f"{int(incomplete.sum())} rows have missing values (first: {frame.index[incomplete][0]}); "
                        "use missing='drop' or 'keep'")
    dropped = int(incomplete.sum()) if missing == "drop" else 0
    if missing == "drop":
        frame = frame[~incomplete]
    frame.attrs["pymetrix"] = {"sample_source": source, "sample_definition": rule, "dropped": dropped}
    return frame


def design(data: Dataset, y: str, x: Any = (), sample: Any = None, add_constant: bool = True, missing: str = "drop",
           where: str | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """``(y, X)`` for an outside estimator: ``y`` a labelled Series, ``X`` a DataFrame of regressors (``const`` first
    when ``add_constant``), on the rows pymetrix's own estimators would use. ``y`` and each ``x`` may be names or
    expressions in pymetrix's formula language (``log(c)``, ``c[-1]``, ``d(y)``). ``missing``: drop rows with missing
    values, or raise. A series from outside the dataset is added first with ``add_series`` (which checks its
    frequency)."""
    from .estimation import Equation
    from .toolkit import design as _design

    if missing not in ("drop", "raise"):
        raise SpecificationError("missing is 'drop' or 'raise' for a design")
    terms = _names(data, x) if x else []
    rhs = " + ".join(terms) if terms else "1"
    text = f"{y} ~ {rhs}" + ("" if add_constant else " - 1")
    if not terms and not add_constant:
        raise SpecificationError("no regressors: give x, or add_constant=True")
    span = None if (isinstance(sample, str) and sample == "all") else sample
    d = _design(Equation(text), data, span=span, where=where, missing=missing)
    yv = pd.Series(d.y, index=d.index, name=str(d.equation.lhs))
    X = pd.DataFrame(d.X, index=d.index, columns=d.names)
    if isinstance(sample, str) and sample == "all":
        source, rule = "all observations", None
    else:
        source, rule = data.sample_record(span, where)
    X.attrs["pymetrix"] = {"sample_source": source, "sample_definition": rule, "dropped": int(d.dropped), "equation": text}
    yv.attrs["pymetrix"] = dict(X.attrs["pymetrix"])
    return yv, X


def add_series(data: Dataset, name: str, values: Any, source: str | None = None, overwrite: bool = False,
               label: str = "", units: str = "", derived_from: str | None = None) -> Dataset:
    """``data`` with the series ``name``: a pandas Series (aligned on the dataset's periods; another frequency is
    refused) or an array (one value per row, in order). An existing name is refused unless ``overwrite``. The series
    records where it came from: ``source`` (default ``user:pandas`` or ``user:numpy``) and when it was added."""
    from .metadata import Variable

    if not name.isidentifier():
        raise SpecificationError(f"{name!r} is not a valid series name")
    if name in data.frame.columns and not overwrite:
        raise DataError(f"the dataset already has a series {name}: choose another name, or overwrite=True")
    if isinstance(values, pd.Series):
        series = values
        source = source or "user:pandas"
    else:
        array = np.asarray(values, dtype=float).ravel()
        if len(array) != len(data.frame):
            raise DataError(f"{name}: {len(array)} values for {len(data.frame)} rows; give one value per row, or a pandas "
                            "Series with the dataset's periods")
        series = pd.Series(array, index=data.frame.index)
        source = source or "user:numpy"
    out = data.with_series(**{name: series})  # aligns, and refuses another frequency
    old = data.variables.get(name)
    variable = Variable(name, label=label or (old.label if old else ""), units=units or (old.units if old else ""),
                        source=source, retrieved=datetime.now().isoformat(timespec="seconds"), derived_from=derived_from)
    return type(out)(out.frame, out.axes, {**out.variables, name: variable}, out.flags, out.meta)


def _describe(func: Callable[..., Any], inputs: Sequence[str]) -> str:
    name = getattr(func, "__qualname__", None) or getattr(func, "__name__", None) or type(func).__name__
    module = getattr(func, "__module__", "") or ""
    full = f"{module}.{name}" if module and module not in ("__main__", "builtins") else name
    return f"user code: {full}({', '.join(inputs)})"


def apply_np(data: Dataset, name: str, func: Callable[..., Any], series: Any, *, source_expr: str | None = None,
             overwrite: bool = False) -> Dataset:
    """Add ``name = func(arrays...)``: ``func`` gets each of ``series`` as a NumPy array (all rows, in order) and
    returns one value per row. Recorded as user code; ``source_expr`` (e.g. ``"np.log(c)"``) says what it computed."""
    inputs = _names(data, series)
    result = func(*[data.frame[s].to_numpy(dtype=float) for s in inputs])
    return add_series(data, name, np.asarray(result, dtype=float), source="user:numpy", overwrite=overwrite,
                      derived_from=source_expr or _describe(func, inputs))


def apply_pd(data: Dataset, name: str, func: Callable[..., Any], series: Any, *, source_expr: str | None = None,
             overwrite: bool = False) -> Dataset:
    """Add ``name = func(series...)``: ``func`` gets each of ``series`` as a pandas Series with the dataset's index
    and returns a Series on it (or one value per row). Recorded as user code, with ``source_expr`` when given."""
    inputs = _names(data, series)
    result = func(*[data.frame[s] for s in inputs])
    if not isinstance(result, pd.Series):
        result = np.asarray(result, dtype=float)
    return add_series(data, name, result, source="user:pandas", overwrite=overwrite, derived_from=source_expr or _describe(func, inputs))


def transform_each(data: Dataset, selection: Any, func: Callable[[pd.Series], Any], prefix: str = "", suffix: str = "",
                   source_expr: str | None = None, overwrite: bool = False) -> Dataset:
    """Add ``prefix + s + suffix = func(s)`` for each series ``s`` of ``selection`` (names, a pattern such as
    ``"g*"`` or a ``Selection``), as ``apply_pd`` does. ``source_expr`` may use ``{s}``: ``"np.log({s})"``."""
    names = list(data.series(selection)) if isinstance(selection, str) and any(ch in selection for ch in "*?[") else _names(data, selection)
    if not prefix and not suffix and not overwrite:
        raise SpecificationError("give a prefix or suffix for the new series, or overwrite=True to replace each one")
    out = data
    for s in names:
        out = apply_pd(out, f"{prefix}{s}{suffix}", func, [s], overwrite=overwrite,
                       source_expr=source_expr.format(s=s) if source_expr else None)
    return out


# --- statsmodels results as pymetrix results ---------------------------------------------------------------------

_COV = {"nonrobust": "classic", "hc0": "hc0", "hc1": "white", "hc2": "hc2", "hc3": "hc3", "hac": "hac"}


@dataclass(frozen=True)
class Statsmodels:
    """The method of a result estimated by statsmodels: its model class and covariance type."""

    model: str = ""
    cov_type: str = "nonrobust"
    hac_lags: int | None = None

    @property
    def cov(self) -> str:
        """pymetrix's name for the covariance type, where there is one (for the summary's note)."""
        return _COV.get(self.cov_type.lower(), self.cov_type)


def _param_name(name: str) -> str:
    return "const" if name in ("Intercept", "const") else name


def _equation(res: Any, endog: str, exog: list[str]) -> Any:
    """A pymetrix ``Equation`` for the fit: its formula when pymetrix reads it, else one written from its names."""
    from .estimation import Equation

    formula = getattr(res.model, "formula", None)
    if formula:
        try:
            return Equation(str(formula))
        except Exception:  # noqa: BLE001 - patsy's language is not pymetrix's: fall back to the names
            pass
    ident = lambda n: re.sub(r"\W", "_", n) if not n.isidentifier() else n  # noqa: E731
    regressors = [ident(n) for n in exog if _param_name(n) != "const"]
    text = f"{ident(endog)} ~ {' + '.join(regressors) if regressors else '1'}" + ("" if "const" in map(_param_name, exog) else " - 1")
    return Equation(text)


def from_statsmodels(res: Any, data: Dataset | None = None, command: str | None = None, index: Any = None) -> FittedEquation:
    """A pymetrix ``FittedEquation`` from a statsmodels regression result (OLS, WLS, GLS, GLM, Logit/Probit and other
    models with ``params`` and ``cov_params``): coefficients with their covariance (so standard errors, t or z and p
    values), fitted values and residuals, fit statistics, the covariance type, convergence where reported, and
    provenance (``data``, when given, supplies vintages and a fingerprint; ``command`` the code that made it).

    Residuals and fitted values are placed on pymetrix's periods: statsmodels' dates are converted; results made
    from plain arrays need ``index=`` (the rows estimated on), and without it keep integer positions, with a warning."""
    import statsmodels

    from ._capture import fingerprint
    from ._version import __version__
    from .estimation import FittedEquation
    from .estimation.params import point
    from .metadata import Provenance

    endog = str(res.model.endog_names)
    exog = [str(n) for n in res.model.exog_names]
    params = np.asarray(res.params, dtype=float)
    cov = np.asarray(res.cov_params(), dtype=float)
    names = [_param_name(n) for n in exog]
    resid = pd.Series(np.asarray(res.resid, dtype=float), index=getattr(res.resid, "index", None), name="resid")
    if index is not None:
        from .adapters.base import to_axes

        resid = to_axes(resid.to_numpy(), index, name="resid")
    elif isinstance(resid.index, pd.DatetimeIndex):  # statsmodels' dates: pymetrix's periods
        freq = data.time_axis.freq if data is not None and data.time_axis is not None else pd.infer_freq(resid.index)
        if freq:
            resid.index = resid.index.to_period(freq)
    elif isinstance(resid.index, pd.RangeIndex):
        import warnings

        warnings.warn("the statsmodels result was made from arrays, so its residuals are on integer positions; pass index= "
                      "(the periods estimated on) to put them on pymetrix's periods", UserWarning, stacklevel=2)
    fitted = pd.Series(np.asarray(res.fittedvalues, dtype=float), index=resid.index, name="fitted")
    cov_type = str(getattr(res, "cov_type", "nonrobust"))
    hac = (getattr(res, "cov_kwds", None) or {}).get("maxlags") if cov_type.lower() == "hac" else None
    method = Statsmodels(type(res.model).__name__, cov_type, hac)
    stats: dict[str, float] = {"nobs": float(res.nobs), "k": float(len(params))}
    if getattr(res, "use_t", True):  # t statistics on these degrees of freedom; z otherwise (as statsmodels reports)
        stats["df_resid"] = float(res.df_resid)
    for key, attr in (("r2", "rsquared"), ("r2_adj", "rsquared_adj"), ("pseudo_r2", "prsquared"), ("loglike", "llf"),
                      ("aic", "aic"), ("bic", "bic"), ("f_statistic", "fvalue"), ("f_pvalue", "f_pvalue")):
        try:
            value = getattr(res, attr)
        except Exception:  # noqa: BLE001 - not every model has every statistic
            continue
        if value is not None and np.isscalar(value) and np.isfinite(value):
            stats[key] = float(value)
    if hac is not None:
        stats["hac_lags"] = float(hac)
    retvals = getattr(res, "mle_retvals", None) or {}
    converged = retvals.get("converged") if isinstance(retvals, dict) else None
    index = resid.index
    periods = index.get_level_values(-1) if isinstance(index, pd.MultiIndex) else index
    sample = (str(periods.min()), str(periods.max())) if isinstance(periods, pd.PeriodIndex) and len(periods) else None
    variables = [endog, *[n for n in exog if _param_name(n) != "const"]]
    if data is not None:
        vintages = {v: data.variables[v].vintage for v in variables if v in data.variables}
        used = data.frame.reindex(index)[[v for v in variables if v in data.frame.columns]] if len(index) else pd.DataFrame()
        data_print = fingerprint(used) if len(used.columns) else ""
    else:
        vintages, data_print = {}, ""
    equation = _equation(res, endog, exog)
    spec: dict[str, Any] = {"equation": equation.text, "endog": endog, "exog": exog}
    if command:
        spec["command"] = command
    prov = Provenance(data_vintages=vintages, spec=spec,
                      method={"type": "statsmodels", "model": method.model, "cov_type": cov_type,
                              "backend": f"statsmodels {statsmodels.__version__}"},
                      sample=sample, nobs=int(res.nobs), dropped=0, package_version=__version__, data_fingerprint=data_print,
                      converged=None if converged is None else bool(converged),
                      resolved={"hac_lags": int(hac)} if hac is not None else None)
    notes = [f"Estimated by statsmodels ({method.model}, cov_type {cov_type})"]
    extra = {"notes": notes, "statsmodels": {"tvalues": np.asarray(res.tvalues, dtype=float), "pvalues": np.asarray(res.pvalues, dtype=float)}}
    try:
        ci = np.asarray(res.conf_int(), dtype=float)
        extra["conf_int"] = pd.DataFrame(ci, index=names, columns=["lower", "upper"])
    except Exception:  # noqa: BLE001
        pass
    design_frame = None
    if getattr(res.model, "exog", None) is not None:
        design_frame = pd.DataFrame(np.asarray(res.model.exog, dtype=float), index=index, columns=names)
    return FittedEquation(equation=equation, params=point(names, params, cov), method=method, resid=resid, fitted=fitted,
                          stats=stats, provenance=prov, design=design_frame, extra=extra)
