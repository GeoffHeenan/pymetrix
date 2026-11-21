"""Temporal disaggregation: quarterly or monthly series consistent with annual (or quarterly) totals.

    td = TemporalDisaggregation("gdp ~ ip", method="chow-lin")     # quarterly GDP from annual GDP and an indicator
    fit = td.fit(annual, quarterly)                                # the indicator's data may run past the last year
    quarterly = quarterly.with_series(gdp=fit.series)
    TemporalDisaggregation("gdp ~ 1", method="denton").fit(annual, quarterly)  # smooth, no indicator

Regression methods (Chow and Lin 1971; Fernández 1981; Litterman 1983) model the high-frequency series as
indicators times coefficients plus an autocorrelated residual, estimate them from the low-frequency data by GLS,
and distribute the low-frequency residual over the high-frequency periods. Chow-Lin's residual is AR(1), with rho
chosen by maximum likelihood (``chow-lin``, the default) or by minimising the residual sum of squares
(``chow-lin-minrss``), searched in ``rho_bounds`` (by default [0, 0.999]); Fernández's is a random walk;
Litterman's a random walk whose increments are AR(1) (``litterman``, ``litterman-minrss``; rho = 0 is Fernández).
Denton-Cholette moves a preliminary series (the indicator, or a constant) as little as possible, in proportional
(``denton``) or additive (``denton-additive``) first differences, or in seasonal differences
(``differences="seasonal"``), to meet the totals.

Every method meets the low-frequency data exactly unless ``soft=`` says how far it may miss them: sums for flows
(``conversion="sum"``), averages for rates and indices, first or last values for stocks, or any weighted sums
given as an aggregation matrix (``fit(low, high, aggregation=C)``: calendar- and fiscal-year totals together,
overlapping or not). Where the indicators run past the last complete low-frequency period, the regression methods
extrapolate, which is how a current-year estimate is built from monthly indicators. ``reconcile`` (in
``pymetrix.reconciliation``) makes several disaggregated components add up to a high-frequency total.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import linalg, optimize

from .errors import DataError, NotFoundError, SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset
    from .matrix import Matrix

__all__ = ["FittedDisaggregation", "TemporalDisaggregation", "aggregation_matrix"]

Method = Literal["chow-lin", "chow-lin-minrss", "fernandez", "litterman", "litterman-minrss", "denton", "denton-additive"]
Conversion = Literal["sum", "average", "first", "last"]
Differences = Literal["first", "seasonal"]
METHODS = ("chow-lin", "chow-lin-minrss", "fernandez", "litterman", "litterman-minrss", "denton", "denton-additive")
DEFAULT_LEVEL_PENALTY = 1e-3  # seasonal-difference Denton: the weight on first differences when level_penalty=None
_AT_BOUND = 1e-4  # an estimated rho this close to an end of its search range is reported as at the bound


def conversion_matrix(low: pd.PeriodIndex, high: pd.PeriodIndex, conversion: str) -> np.ndarray:
    """C with C @ y_high = y_low: each low-frequency period from its high-frequency periods (all must be present)."""
    owner = pd.PeriodIndex([p.asfreq(low.freq, how="end") for p in high])  # the low-frequency period each belongs to
    C = np.zeros((len(low), len(high)))
    for i, period in enumerate(low):
        cols = np.flatnonzero(owner == period)
        expected = len(pd.period_range(period.start_time, period.end_time, freq=high.freq))
        if len(cols) != expected:
            raise SpecificationError(f"{period} is not fully covered by the high-frequency periods ({len(cols)} of {expected})")
        if conversion == "sum":
            C[i, cols] = 1.0
        elif conversion == "average":
            C[i, cols] = 1.0 / len(cols)
        elif conversion == "first":
            C[i, cols[0]] = 1.0
        elif conversion == "last":
            C[i, cols[-1]] = 1.0
        else:
            raise SpecificationError("conversion must be 'sum', 'average', 'first' or 'last'")
    return C


def _period_label(p: pd.Period) -> str:
    """A period's name, with its frequency where the name alone would be ambiguous (a July-June year: '2020 Y-JUN')."""
    plain = {"Y-DEC", "YE-DEC", "A-DEC", "Q-DEC", "QE-DEC", "M", "ME", "D", "B"}
    return str(p) if p.freqstr in plain or p.freqstr.startswith("W") else f"{p} {p.freqstr}"


def aggregation_matrix(high: Any, periods: Sequence[Any], conversion: Conversion = "sum") -> Matrix:
    """The aggregation matrix of low-frequency ``periods`` of any frequencies over the high-frequency periods of
    ``high`` (a dataset or a PeriodIndex): one row per period, named by it ('2019', '2020 Y-JUN'), one column per
    high-frequency period. Calendar years, fiscal years and quarters can be mixed and can overlap::

        C = aggregation_matrix(quarterly, [pd.Period("2018", "Y"), pd.Period("2019", "Y-JUN"), pd.Period("2020", "Y-JUN")])
    """
    from .matrix import Matrix

    index = pd.PeriodIndex(high.frame.index if hasattr(high, "frame") else high)
    rows, labels = [], []
    for p in periods:
        p = p if isinstance(p, pd.Period) else pd.Period(p)
        rows.append(conversion_matrix(pd.PeriodIndex([p]), index, conversion)[0])
        labels.append(_period_label(p))
    if not rows:
        raise SpecificationError("give at least one low-frequency period")
    return Matrix(np.vstack(rows), tuple(labels), tuple(str(p) for p in index))


def _read_table(path: str) -> pd.DataFrame:
    from pathlib import Path

    suffix = Path(path).suffix.lower()
    if not Path(path).exists():
        raise NotFoundError(f"no aggregation file {path!r}")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, index_col=0)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, index_col=0)


def _aggregation_rows(aggregation: Any, high: Dataset) -> tuple[np.ndarray, list[str] | None]:
    """The aggregation matrix over every period of ``high`` (columns in its order), and its rows' names (None: rows
    are matched to the low-frequency data by position).

    Accepted: a Matrix or DataFrame (rows = constraints, columns = high-frequency periods, matched by name), a dataset
    at the high frequency (each series one constraint's weights, the transpose), a path to a CSV, Excel or Parquet
    table laid out like the Matrix, or a NumPy array (by position: one row per low-frequency observation, one column
    per high-frequency period)."""
    from .data import Dataset
    from .matrix import Matrix, _numbered

    hi_labels = [str(p) for p in high.frame.index]
    if isinstance(aggregation, str):
        aggregation = _read_table(aggregation)
    if isinstance(aggregation, Dataset):
        if aggregation.kind != "time_series" or aggregation.frame.index.freq != high.frame.index.freq:
            raise SpecificationError("an aggregation dataset has the high frequency: one series per constraint, its weights by period")
        frame = aggregation.frame.select_dtypes("number").reindex(high.frame.index)
        return np.nan_to_num(frame.to_numpy(float).T), [str(c) for c in frame.columns]
    if isinstance(aggregation, pd.DataFrame):
        aggregation = Matrix.from_frame(aggregation)
    if isinstance(aggregation, Matrix):
        values, rows, cols = np.asarray(aggregation.values, float), list(aggregation.rows), list(aggregation.cols)
        if all(c in hi_labels for c in cols):
            C = np.zeros((values.shape[0], len(hi_labels)))
            position = {c: j for j, c in enumerate(hi_labels)}
            for k, c in enumerate(cols):
                C[:, position[c]] = values[:, k]
        elif tuple(cols) == _numbered(len(cols)) and len(cols) == len(hi_labels):
            C = values
        else:
            unknown = [c for c in cols if c not in hi_labels]
            raise SpecificationError(f"the aggregation matrix's columns are the high-frequency periods ({hi_labels[0]}...); "
                                     f"{unknown[0]!r} is not one of them")
        named = tuple(rows) != _numbered(len(rows))
        return np.nan_to_num(C), (rows if named else None)
    values = np.asarray(aggregation, float)
    if values.ndim != 2 or values.shape[1] != len(hi_labels):
        raise SpecificationError(f"an aggregation array has one column per high-frequency period ({len(hi_labels)}), not shape {values.shape}")
    return np.nan_to_num(values), None


def _ar1_cov(n: int, rho: float) -> np.ndarray:
    i = np.arange(n)
    return rho ** np.abs(i[:, None] - i[None, :]) / (1 - rho**2)


def _rw_cov(n: int) -> np.ndarray:
    """Fernández: (D'D)^-1 with D the first-difference matrix whose first row is the first value."""
    D = np.eye(n) - np.eye(n, k=-1)
    return np.linalg.inv(D.T @ D)


def _litterman_cov(n: int, rho: float) -> np.ndarray:
    """Litterman: (D'H'HD)^-1, a random walk whose increments are AR(1) (H = I - rho L), as tempdisagg's CalcQ_Lit."""
    D = np.eye(n) - np.eye(n, k=-1)
    H = np.eye(n) - rho * np.eye(n, k=-1)
    HD = H @ D
    return np.linalg.inv(HD.T @ HD)


@dataclass(frozen=True)
class _GLS:
    beta: np.ndarray
    cov_beta: np.ndarray
    y: np.ndarray
    rho: float
    loglike: float
    rss: float
    sigma2: float


def _gls(y_low: np.ndarray, X: np.ndarray, C: np.ndarray, Sigma: np.ndarray, rho: float) -> _GLS:
    Xl = C @ X
    V = C @ Sigma @ C.T
    cV = linalg.cho_factor(V)
    Vi_Xl = linalg.cho_solve(cV, Xl)
    A = Xl.T @ Vi_Xl
    beta = np.linalg.solve(A, Vi_Xl.T @ y_low)
    u = y_low - Xl @ beta
    Vi_u = linalg.cho_solve(cV, u)
    rss = float(u @ Vi_u)
    n = len(y_low)
    sigma2 = rss / n
    logdet = 2 * np.sum(np.log(np.diag(cV[0])))
    loglike = -n / 2 * (np.log(2 * np.pi * sigma2) + 1) - logdet / 2
    y = X @ beta + Sigma @ C.T @ Vi_u
    k = X.shape[1]
    cov_beta = rss / max(n - k, 1) * np.linalg.inv(A)
    return _GLS(beta, cov_beta, y, rho, float(loglike), float(u @ u), sigma2)


def _gls_soft(y_low: np.ndarray, X: np.ndarray, C: np.ndarray, Sigma: np.ndarray, rho: float, var: np.ndarray) -> _GLS:
    """GLS with measurement error on the low-frequency values: Y = C X b + C u + e, u ~ (0, s2 Sigma), e ~ (0, diag(var)).

    ``var`` is in the data's units, so s2 cannot be concentrated out: it is chosen by maximum likelihood (a grid in
    log s2 around the exact-constraint estimate, then a bounded search). The series is X b + s2 Sigma C' V^-1 u; where
    var is zero the constraint still holds exactly. The coefficients' covariance is (X'C'V^-1 C X)^-1."""
    Xl, M, n = C @ X, C @ Sigma @ C.T, len(y_low)

    def at(log_s2: float) -> tuple[float, Any]:
        s2 = float(np.exp(log_s2))
        V = s2 * M + np.diag(var)
        try:
            cV = linalg.cho_factor(V)
        except linalg.LinAlgError:
            return -np.inf, None
        Vi_Xl = linalg.cho_solve(cV, Xl)
        A = Xl.T @ Vi_Xl
        beta = np.linalg.solve(A, Vi_Xl.T @ y_low)
        u = y_low - Xl @ beta
        Vi_u = linalg.cho_solve(cV, u)
        logdet = 2 * np.sum(np.log(np.diag(cV[0])))
        ll = -0.5 * (n * np.log(2 * np.pi) + logdet + float(u @ Vi_u))
        return float(ll), (s2, beta, u, Vi_u, A)

    center = float(np.log(max(_gls(y_low, X, C, Sigma, rho).sigma2, 1e-300)))
    grid = center + np.linspace(-25.0, 5.0, 31)
    best = float(grid[int(np.argmax([at(g)[0] for g in grid]))])
    found = optimize.minimize_scalar(lambda g: -at(g)[0], bounds=(best - 1.0, best + 1.0), method="bounded", options={"xatol": 1e-8})
    ll, parts = at(float(found.x))
    if parts is None or ll < at(best)[0]:
        ll, parts = at(best)
    s2, beta, u, Vi_u, A = parts
    y = X @ beta + s2 * (Sigma @ C.T @ Vi_u)
    return _GLS(beta, np.linalg.inv(A), y, rho, ll, float(u @ u), s2)


def _difference(n: int, lag: int) -> np.ndarray:
    return (np.eye(n) - np.eye(n, k=-lag))[lag:]


def _denton(y_low: np.ndarray, x: np.ndarray, C: np.ndarray, proportional: bool, seasonal: int = 0,
            level_penalty: float = 0.0, sd: np.ndarray | None = None) -> np.ndarray:
    """Denton-Cholette: the series closest to ``x`` in (proportional) first differences that meets C y = y_low.

    ``seasonal`` (s, e.g. 4 or 12): differences at lag s instead, plus ``level_penalty`` times the first-difference
    criterion, which pins the within-year pattern seasonal differences leave free. ``sd``: soft constraints, where
    ((C y - y_low) / sd)^2 is added to the criterion (sd = 0: that constraint holds exactly)."""
    n = len(x)
    if seasonal and n <= seasonal:
        raise SpecificationError(f"seasonal differences need more than {seasonal} high-frequency periods")
    D = (np.eye(n) - np.eye(n, k=-1))[1:] if not seasonal else _difference(n, seasonal)  # no condition on the first periods (Cholette)
    if proportional:
        if np.any(x == 0):
            raise SpecificationError("proportional Denton needs an indicator without zeros; use denton-additive")
        W = D @ np.diag(1.0 / x)  # first differences of the ratio y / x
        A, rhs = W.T @ W, np.zeros(n)
        if seasonal and level_penalty:
            W1 = _difference(n, 1) @ np.diag(1.0 / x)
            A = A + level_penalty * (W1.T @ W1)
    elif seasonal and level_penalty:
        D1 = _difference(n, 1)
        A = D.T @ D + level_penalty * (D1.T @ D1)
        rhs = A @ x
    else:
        A, rhs = D.T @ D, D.T @ D @ x
    m = len(y_low)
    if sd is None:
        K = np.block([[A, C.T], [C, np.zeros((m, m))]])
        solution = np.linalg.lstsq(K, np.r_[rhs, y_low], rcond=None)[0]
        return solution[:n]
    # soft constraints: C y + e = y_low with the error e costing (e / sd)^2. With the multipliers nu = sd * lambda the
    # KKT system is [[A, C'F], [F C, -I]] for soft rows (F = diag(1/sd)) and the exact system's rows where sd = 0:
    # scaled so that it stays well conditioned however small or large sd is
    soft = sd > 0
    f = np.where(soft, 1.0 / np.where(soft, sd, 1.0), 1.0)
    FC = f[:, None] * C
    K = np.block([[A, FC.T], [FC, -np.diag(soft.astype(float))]])
    b = np.r_[rhs, f * y_low]
    try:
        solution = np.linalg.solve(K, b)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(K, b, rcond=None)[0]
    return solution[:n]


def _season_length(index: pd.PeriodIndex) -> int:
    base = index.freqstr.split("-")[0].rstrip("E")
    if base == "Q":
        return 4
    if base == "M":
        return 12
    raise SpecificationError(f"seasonal differences need quarterly or monthly data, not {index.freqstr}")


def _normalise_soft(soft: Any) -> Any:
    if soft is None:
        return None
    if isinstance(soft, Mapping):
        soft = tuple((str(k), float(v)) for k, v in soft.items())
    elif isinstance(soft, (list, tuple)):
        try:
            soft = tuple((str(k), float(v)) for k, v in soft)
        except (TypeError, ValueError):
            raise SpecificationError("soft is a standard deviation, or {low-frequency period: standard deviation}") from None
    else:
        try:
            soft = float(soft)
        except (TypeError, ValueError):
            raise SpecificationError(f"soft is a standard deviation, or {{period: standard deviation}}, not {soft!r}") from None
    values = [soft] if isinstance(soft, float) else [v for _, v in soft]
    if any(not np.isfinite(v) or v < 0 for v in values):
        raise SpecificationError("soft constraints' standard deviations are zero (exact) or positive")
    return soft


@dataclass(frozen=True)
class TemporalDisaggregation:
    """``formula``: the low-frequency series on high-frequency indicators (``gdp ~ ip + emp``; ``gdp ~ 1`` for none).

    ``method``: ``chow-lin`` (AR(1) residual, rho by maximum likelihood), ``chow-lin-minrss`` (rho by minimum residual
    sum of squares), ``fernandez`` (random walk), ``litterman`` and ``litterman-minrss`` (random walk with AR(1)
    increments, Litterman 1983), ``denton`` (proportional) or ``denton-additive``.
    ``conversion``: how the low-frequency data aggregate the high-frequency series (sum, average, first, last);
    ``fit(..., aggregation=C)`` gives any aggregation matrix instead.
    ``rho``: Chow-Lin and Litterman: fix rho instead of estimating it.
    ``rho_bounds``: the range rho is estimated in, by default [0, 0.999] (tempdisagg's: rho is kept non-negative);
    ``(-0.999, 0.999)`` allows negative rho. An estimate at an end of the range is reported in the summary's notes.
    ``soft``: soft constraints: the standard deviation of the error on the low-frequency values (in their units), one
    for all, or ``{period: sd}`` (periods not named stay exact; 0 = exact). Regression methods treat it as measurement
    error on the aggregates (GLS); Denton adds ((C y - Y) / sd)^2 to its criterion. ``None`` (the default): exact.
    ``differences``: Denton: ``first`` (the default) or ``seasonal`` (lag 4 for quarterly, 12 for monthly data), which
    keeps the indicator's seasonal pattern; ``level_penalty``: with seasonal differences, the weight on first
    differences that pins the pattern seasonal differences leave free (None: 0.001; 0: none, the minimum-norm solution).
    """

    formula: str
    method: Method = "chow-lin"
    conversion: Conversion = "sum"
    rho: float | None = None  # Chow-Lin and Litterman: fix rho instead of estimating it
    rho_bounds: tuple[float, float] = (0.0, 0.999)  # tempdisagg's default keeps rho non-negative
    soft: float | tuple[tuple[str, float], ...] | None = None  # exact constraints unless given
    differences: Differences = "first"  # Denton: first or seasonal differences
    level_penalty: float | None = None  # seasonal Denton: weight on first differences (None: 0.001)

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise SpecificationError(f"unknown method {self.method!r}; one of {', '.join(METHODS)}")
        if self.differences not in ("first", "seasonal"):
            raise SpecificationError("differences is 'first' or 'seasonal'")
        if self.differences == "seasonal" and not self.method.startswith("denton"):
            raise SpecificationError("seasonal differences are for Denton; the regression methods model the residual instead")
        if self.level_penalty is not None and (not np.isfinite(self.level_penalty) or self.level_penalty < 0):
            raise SpecificationError("level_penalty is zero or positive")
        object.__setattr__(self, "soft", _normalise_soft(self.soft))
        bounds = tuple(float(b) for b in self.rho_bounds)
        if len(bounds) != 2 or not -1 < bounds[0] < bounds[1] < 1:
            raise SpecificationError(f"rho_bounds is two numbers between -1 and 1, low first, not {self.rho_bounds!r}")
        object.__setattr__(self, "rho_bounds", bounds)

    def _soft_sd(self, labels: list[str], every: list[str]) -> np.ndarray | None:
        """The standard deviation of each constraint's error (0 = exact), or None when every constraint is exact."""
        if self.soft is None:
            return None
        if isinstance(self.soft, float):
            sd = np.full(len(labels), self.soft)
        else:
            given = dict(self.soft)
            unknown = [k for k in given if k not in every]
            if unknown:
                raise NotFoundError(f"soft names {unknown[0]!r}, which is not a low-frequency period or constraint ({every[0]}...)")
            sd = np.array([given.get(k, 0.0) for k in labels])
        return sd if np.any(sd > 0) else None

    def fit(self, low: Dataset, high: Dataset, aggregation: Any = None) -> FittedDisaggregation:
        """Disaggregate ``low``'s series with ``high``'s indicators.

        ``aggregation``: an aggregation matrix C (low × high) in place of ``conversion``, so that C y = Y: a Matrix or
        DataFrame with one row per low-frequency value (named as ``low``'s rows: periods, or any labels in a
        cross-section dataset) and one column per high-frequency period (named as ``high``'s), a dataset at the high
        frequency with one series of weights per constraint, a CSV, Excel or Parquet file laid out like the Matrix, or
        an array matched by position (see ``aggregation_matrix``). Rows whose low-frequency value is missing are
        dropped; the rest must be independent and lie within the indicators' sample."""
        from .estimation.design import as_series
        from .estimation.equation import Equation
        from .expr import evaluate

        eq = Equation(self.formula)
        target = eq.require_dependent()
        if target not in low.names:
            raise NotFoundError(f"no series {target!r} in the low-frequency data")
        if high.time_axis is None or high.kind != "time_series":
            raise SpecificationError("temporal disaggregation needs two time series datasets")
        if aggregation is None and (low.time_axis is None or low.kind != "time_series"):
            raise SpecificationError("temporal disaggregation needs two time series datasets (or an aggregation matrix)")
        if aggregation is not None and self.conversion != "sum":
            raise SpecificationError("an aggregation matrix replaces conversion: put the averages' weights in the matrix")
        hi_index = high.frame.index
        y_low_all = low[target].astype(float)
        cols = []
        names = []
        if eq.intercept:
            cols.append(np.ones(len(hi_index)))
            names.append("const")
        for term in eq.terms:
            cols.append(as_series(evaluate(term.expr, high.lagged), hi_index).to_numpy(float))
            names.append(term.label)
        X_all = np.column_stack(cols) if cols else np.zeros((len(hi_index), 0))
        ok = np.all(np.isfinite(X_all), axis=1) if X_all.shape[1] else np.ones(len(hi_index), bool)
        if not ok.any():
            raise SpecificationError("the indicators have no complete observations")
        first, last = np.flatnonzero(ok)[[0, -1]]
        if not ok[first : last + 1].all():
            raise SpecificationError("the indicators have gaps; disaggregation needs an unbroken high-frequency sample")
        hi = pd.PeriodIndex(hi_index[first : last + 1])
        X = X_all[first : last + 1]
        info: dict[str, Any] = {}
        dropped = 0
        if aggregation is None:
            covered = [p for p in y_low_all.dropna().index if p.start_time >= hi[0].start_time and p.end_time <= hi[-1].end_time]
            if len(covered) < 3:
                raise SpecificationError("fewer than three low-frequency periods are covered by the indicators")
            lo = pd.PeriodIndex(covered)
            used_rows: Any = lo
            y_low = y_low_all.loc[lo].to_numpy(float)
            C = conversion_matrix(lo, hi, self.conversion)
            labels = [str(p) for p in lo]
            every = [str(p) for p in y_low_all.index]
        else:
            C, lo, used_rows, y_low, labels, every, dropped = self._matrix_rows(aggregation, high, y_low_all, int(first), int(last))
            info["aggregation"] = f"an aggregation matrix of {len(labels)} constraints" + (f" ({dropped} dropped: no value)" if dropped else "")
        sd = self._soft_sd(labels, every)
        if sd is not None:
            info["soft"] = sd
        if self.method.startswith("denton"):
            preliminary = X @ np.ones(X.shape[1]) if names != ["const"] else np.ones(len(hi))
            if len(names) > 2 or (len(names) == 2 and names[0] != "const"):
                raise SpecificationError("Denton takes one indicator (gdp ~ ip) or none (gdp ~ 1)")
            if len(names) == 2:
                preliminary = X[:, 1]
            seasonal = _season_length(hi) if self.differences == "seasonal" else 0
            penalty = (DEFAULT_LEVEL_PENALTY if self.level_penalty is None else self.level_penalty) if seasonal else 0.0
            if seasonal:
                info["differences"] = f"seasonal differences (lag {seasonal}), first-difference weight {penalty:g}"
            y = _denton(y_low, preliminary, C, proportional=self.method == "denton", seasonal=seasonal, level_penalty=penalty, sd=sd)
            params, cov, rho = np.zeros(0), np.zeros((0, 0)), np.nan
            names = []
        else:
            g = self._regression(y_low, X, C, len(hi), sd, info)
            y, params, cov, rho = g.y, g.beta, g.cov_beta, g.rho
            info.update(loglike=g.loglike, sigma2=g.sigma2)
        series = pd.Series(y, index=hi, name=target)
        from ._capture import data_fingerprint
        from ._version import __version__
        from .metadata import Provenance

        indicators = sorted(set().union(*(t.expr.variables() for t in eq.terms))) if eq.terms else []
        spec: dict[str, Any] = {"formula": self.formula, "method": self.method}
        if aggregation is not None:
            spec["aggregation"] = f"matrix ({len(labels)} constraints)"
        for key, default in (("soft", None), ("differences", "first"), ("level_penalty", None)):
            if getattr(self, key) != default:
                spec[key] = getattr(self, key)
        prov = Provenance(
            data_vintages={**{n: high.variables[n].vintage for n in indicators if n in high.variables},
                           **({target: low.variables[target].vintage} if target in low.variables else {})},
            spec=spec, method={"type": "temporal disaggregation", "method": self.method},
            sample=(str(hi[0]), str(hi[-1])), nobs=len(lo), dropped=dropped, package_version=__version__,
            data_fingerprint=f"{data_fingerprint(low, [target], used_rows)}:{data_fingerprint(high, indicators, hi)}",
            messages=(info["rho_note"],) if "rho_note" in info else ())
        return FittedDisaggregation(self, series, names, params, cov, rho, lo, y_low, C, info, prov)

    def _matrix_rows(self, aggregation: Any, high: Dataset, y_low_all: pd.Series, first: int, last: int) -> tuple[Any, ...]:
        """The aggregation matrix restricted to the indicators' sample and the low-frequency values it meets."""
        C_all, rows = _aggregation_rows(aggregation, high)
        low_labels = [str(i) for i in y_low_all.index]
        if rows is None:
            if C_all.shape[0] != len(y_low_all):
                raise SpecificationError(f"the aggregation matrix has {C_all.shape[0]} rows for {len(y_low_all)} low-frequency values; "
                                         "name its rows as the low-frequency data's to match them by name")
            labels, positions = low_labels, list(range(len(low_labels)))
        else:
            if len(set(rows)) != len(rows):
                raise SpecificationError("the aggregation matrix's rows need different names")
            where = {lab: i for i, lab in enumerate(low_labels)}
            labels, positions = list(rows), [where.get(r, -1) for r in rows]
        values = np.array([y_low_all.iloc[p] if p >= 0 else np.nan for p in positions], float)
        keep = np.isfinite(values)
        if not keep.any():
            raise DataError("no row of the aggregation matrix has a low-frequency value (its rows are named as the low-frequency data's)")
        C_keep = C_all[keep]
        outside = np.any(C_keep[:, :first] != 0, axis=1) | np.any(C_keep[:, last + 1 :] != 0, axis=1)
        kept = [lab for lab, k in zip(labels, keep) if k]
        if outside.any():
            raise SpecificationError(f"constraint {kept[int(np.flatnonzero(outside)[0])]!r} uses high-frequency periods outside the "
                                     "indicators' sample; drop its value or shorten the matrix")
        C = C_keep[:, first : last + 1]
        if np.any(np.all(C == 0, axis=1)):
            raise SpecificationError(f"constraint {kept[int(np.flatnonzero(np.all(C == 0, axis=1))[0])]!r} has no weights")
        if len(kept) < 3:
            raise SpecificationError("fewer than three low-frequency values are covered by the indicators")
        rank = np.linalg.matrix_rank(C)
        if rank < C.shape[0]:
            raise SpecificationError(f"the aggregation constraints are not independent (rank {rank} of {C.shape[0]}): a row repeats or "
                                     "is a combination of others")
        used = [y_low_all.index[p] for p, k in zip(positions, keep) if k]
        return C, pd.Index(kept), pd.Index(used), values[keep], kept, [*low_labels, *labels], int((~keep).sum())

    def _regression(self, y_low: np.ndarray, X: np.ndarray, C: np.ndarray, n: int, sd: np.ndarray | None, info: dict[str, Any]) -> _GLS:
        def fit_at(Sigma: np.ndarray, r: float) -> _GLS:
            return _gls(y_low, X, C, Sigma, r) if sd is None else _gls_soft(y_low, X, C, Sigma, r, sd**2)

        if self.method == "fernandez":
            return fit_at(_rw_cov(n), 1.0)
        cov_of = _ar1_cov if self.method.startswith("chow-lin") else _litterman_cov
        if self.rho is not None:
            return fit_at(cov_of(n, self.rho), self.rho)
        ml = not self.method.endswith("minrss")

        def objective(r: float) -> float:
            res = fit_at(cov_of(n, r), r)
            return -res.loglike if ml else res.rss

        lo_b, hi_b = self.rho_bounds
        grid = np.linspace(lo_b, hi_b, 41)
        best = grid[int(np.argmin([objective(r) for r in grid]))]
        step = (hi_b - lo_b) / 40
        found = optimize.minimize_scalar(objective, bounds=(max(lo_b, best - step), min(hi_b, best + step)), method="bounded",
                                         options={"xatol": 1e-6})
        r = float(found.x)
        info["rho_criterion"] = "maximum likelihood" if ml else "minimum residual sum of squares"
        for end, bound in (("lower", lo_b), ("upper", hi_b)):
            if abs(r - bound) < _AT_BOUND:
                info["rho_note"] = (f"rho is at the {end} end of its search range [{lo_b:g}, {hi_b:g}]"
                                    + ("; rho_bounds=(-0.999, 0.999) allows negative values" if end == "lower" and lo_b >= 0 else ""))
        return fit_at(cov_of(n, r), r)


@dataclass(frozen=True, eq=False)
class FittedDisaggregation:
    spec: TemporalDisaggregation
    series: pd.Series  # the high-frequency series
    names: list[str]
    params: np.ndarray
    cov: np.ndarray
    rho: float
    low_index: pd.Index  # the low-frequency periods (or an aggregation matrix's row names) used
    low_values: np.ndarray
    C: np.ndarray = field(repr=False)
    info: dict[str, Any] = field(default_factory=dict)
    provenance: Any = None  # data vintages, sample, method, fingerprint

    @property
    def check(self) -> float:
        """Largest gap between the low-frequency data and the disaggregated series aggregated (zero, unless soft)."""
        return float(np.max(np.abs(self.C @ self.series.to_numpy()[: self.C.shape[1]] - self.low_values)))

    @property
    def gaps(self) -> pd.Series:
        """The low-frequency data minus the disaggregated series aggregated, by period (zero, unless soft)."""
        return pd.Series(self.low_values - self.C @ self.series.to_numpy()[: self.C.shape[1]], index=self.low_index, name="gap")

    def summary(self) -> Table:
        from scipy import stats as sps

        se = np.sqrt(np.diag(self.cov)) if self.cov.size else np.zeros(0)
        rows = [[b, s, b / s, 2 * sps.t.sf(abs(b / s), max(len(self.low_values) - len(self.params), 1))]
                for b, s in zip(self.params, se)]
        s = self.spec
        conversion = self.info.get("aggregation", f"conversion: {s.conversion}")
        notes = [f"Method: {s.method}; {conversion}; {len(self.low_values)} low-frequency periods "
                 f"({self.low_index[0]} to {self.low_index[-1]}), {len(self.series)} high-frequency ({self.series.index[0]} to {self.series.index[-1]})"]
        if "differences" in self.info:
            notes.append(f"Denton in {self.info['differences']}")
        if np.isfinite(self.rho):
            notes.append(f"rho = {self.rho:.4f}" + (f" ({self.info['rho_criterion']})" if "rho_criterion" in self.info else ""))
        if "rho_note" in self.info:
            notes.append(f"Note: {self.info['rho_note']}")
        if "soft" in self.info:
            sd = self.info["soft"]
            z = np.abs(self.gaps.to_numpy()[sd > 0] / sd[sd > 0])
            notes.append(f"Soft constraints ({int((sd > 0).sum())} of {len(sd)}): the series misses the low-frequency data by at most "
                         f"{self.check:.3g}, {z.max():.2f} standard deviations")
        else:
            notes.append(f"The series meets the low-frequency data to within {self.check:.2e}")
        return Table(["coef", "std err", "t", "p>|t|"], rows, index=self.names, title=f"Temporal disaggregation: {s.formula}", notes=notes)

    def plot(self) -> Any:
        """The disaggregated series, with the low-frequency data spread evenly over each period for comparison."""
        from .figures import Figure, Line, Panel

        layers = [Line(self.series, "disaggregated")]
        if self.spec.conversion in ("sum", "average") and "aggregation" not in self.info:
            per_period = self.C.astype(bool).sum(axis=1)
            level = self.low_values / per_period if self.spec.conversion == "sum" else self.low_values
            even = pd.Series(np.nan, index=self.series.index)
            for i, row in enumerate(self.C.astype(bool)):
                even.iloc[np.flatnonzero(row)] = level[i]
            layers.append(Line(even.dropna(), "low-frequency, spread evenly", style="--"))
        return Figure((Panel(tuple(layers), title=str(self.series.name or "")),))

    def __repr__(self) -> str:
        return f"<FittedDisaggregation {self.spec.formula}, {self.spec.method}: {len(self.series)} periods>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()
