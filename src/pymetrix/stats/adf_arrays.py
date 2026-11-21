"""The ADF and Engle-Granger statistics on NumPy arrays, many series at once, and their null distributions by simulation.

One computation serves the data and the simulation: ``adf`` and ``engle_granger`` take a stack of series (one row per
draw) and compute, for each, exactly the statistic statsmodels' ``adfuller`` and ``coint`` compute (and that
``ADF`` and ``EngleGranger`` report by default):

- **ADF regression:** the first difference on the lagged level, the deterministic terms (``c`` a constant, ``ct`` and
  a trend 1, 2, ..., ``ctt`` and its square, ``n`` none) and ``L`` lagged differences; the statistic is the t-ratio on
  the lagged level.
- **Lag rule:** a fixed ``lags``, or a search over 0 to ``max_lags`` (default: see ``default_max_lag``; a lag given
  must leave the regression at least one residual degree of freedom) on the common sample of the largest lag, by
  ``aic`` or ``bic`` (smallest wins, ties to the
  fewer lags; the criteria count the regressors, from the profile Gaussian log-likelihood) or ``t-stat`` (from the
  largest lag down, the first whose last lag has |t| >= 1.645); the chosen lag is then refitted on its own, longer
  sample.
- **Engle-Granger:** the first series on the others, the deterministic terms and any fixed ``exog`` columns by least
  squares; the ADF regression, with no deterministic terms, on its residuals; a combination fitting to within
  ``100 sqrt(eps)`` of R² = 1 gives minus infinity, as in statsmodels. The default largest lag also counts the levels
  regression's columns as used degrees of freedom.

statsmodels' own default largest lag, ``min(ceil(12 (n/100)^(1/4)), n//2 - terms - 1)``, leaves the largest search
regression ``n - 2(n//2) + terms`` residual degrees of freedom: none at all for an even ``n`` with no deterministic
terms, which is the Engle-Granger residual regression. The sum of squares there is zero, the information criteria
pick that lag, and the statistic is a t-ratio with an infinite standard error, 0.0. ``default_max_lag`` caps the
rule so that at least ``MIN_DF`` residual degrees of freedom remain, and ``degenerate_statistic`` refuses a zero or
undefined statistic computed from data.

The regressions are solved by batched QR decompositions: one decomposition of the largest lag search regression
gives every smaller one's sum of squares (the columns are nested), so a search costs one factorisation per draw.

``simulate_adf`` and ``simulate_engle_granger`` draw the null (independent Gaussian random walks of the sample's
length, ``cumsum`` of standard normals starting from the first draw) from ``pymetrix._random.generator(seed)`` as
``standard_normal((draws, n))`` or ``standard_normal((draws, n, k))``, in blocks of draws that continue the same
stream, and return each draw's statistic. ``simulated_summary`` turns them into critical values (the 1, 5 and 10%
quantiles, NumPy's default linear interpolation) and the p-value (the share of draws at or below the statistic) with
its Monte Carlo standard error ``sqrt(p (1 - p) / draws)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._random import generator as make_rng
from ..errors import SpecificationError

__all__ = ["ADFArrays", "MIN_DF", "adf", "check_lag", "default_max_lag", "degenerate_statistic", "engle_granger",
           "simulate_adf", "simulate_engle_granger", "simulated_summary", "trend_terms"]

TRENDS = ("c", "ct", "ctt", "n")
AUTOLAGS = ("aic", "bic", "t-stat")
T_STOP = 1.6448536269514722  # the one-sided 5% normal quantile statsmodels' t-stat rule stops at
SQRTEPS = float(np.sqrt(np.finfo(np.double).eps))
MIN_DF = 4  # residual degrees of freedom the default largest lag leaves in the largest search regression
_BLOCK_ELEMENTS = 8_000_000  # numbers in one block of draws' largest regression: bounds the memory of a simulation


@dataclass(frozen=True)
class ADFArrays:
    """Each series' ADF statistic, the lagged differences used and the observations of its final regression."""

    statistic: np.ndarray
    lags: np.ndarray
    nobs: np.ndarray


def _ntrend(trend: str) -> int:
    if trend not in TRENDS:
        raise SpecificationError(f"trend is one of {', '.join(TRENDS)}, not {trend!r}")
    return 0 if trend == "n" else len(trend)


def trend_terms(n: int, trend: str) -> np.ndarray:
    """The deterministic columns of ``trend`` over ``n`` rows, as statsmodels' ``add_trend`` makes them: a constant,
    then t = 1, ..., n, then t squared."""
    t = np.arange(1, n + 1, dtype=float)
    return np.column_stack([t ** p for p in range(_ntrend(trend))]) if trend != "n" else np.empty((n, 0))


def _residual_df(n: int, ntrend: int, lag: int, used: int = 0) -> int:
    """Residual degrees of freedom of the ADF regression with ``lag`` lagged differences on ``n`` levels: n - 1 - lag
    rows, ntrend + 1 + lag regressors, less ``used`` parameters already estimated (the Engle-Granger levels regression)."""
    return n - 2 - 2 * lag - ntrend - used


def default_max_lag(n: int, ntrend: int, used: int = 0) -> int:
    """The default largest lag for ``n`` observations: Schwert's ``ceil(12 (n/100)^(1/4))``, as statsmodels and arch
    use, capped at statsmodels' ``n//2 - ntrend - 1`` and so that the largest search regression keeps ``MIN_DF``
    residual degrees of freedom after ``used`` parameters estimated beforehand (the Engle-Granger levels regression's
    columns): ``lag <= (n - 2 - ntrend - used - MIN_DF) // 2``, and at least 0. The caps bind only in short samples
    (with a constant, below about 25 observations); above them the rule is statsmodels'."""
    schwert = int(np.ceil(12.0 * np.power(n / 100.0, 1 / 4.0)))
    lag = max(0, min(schwert, n // 2 - ntrend - 1, (n - 2 - ntrend - used - MIN_DF) // 2))
    if _residual_df(n, ntrend, lag, used) < 1:
        raise SpecificationError(f"{n} observations are too few for this test with these deterministic terms "
                                 "(its regression would have no residual degrees of freedom)")
    return lag


def check_lag(lag: int, n: int, ntrend: int) -> int:
    """A lag given by the user: zero or more, at most statsmodels' ``n//2 - ntrend - 1``, and leaving the regression at
    least one residual degree of freedom (with none the statistic is 0.0 whatever the data)."""
    lag = int(lag)
    if lag < 0:
        raise SpecificationError(f"the lag must be zero or more, not {lag}")
    top = min(n // 2 - ntrend - 1, (n - 3 - ntrend) // 2)
    if lag > top:
        if top < 0:
            raise SpecificationError(f"{n} observations are too few for this test with these deterministic terms")
        raise SpecificationError(f"with {n} observations the largest lag is {top}, not {lag}: more would leave the "
                                 "test regression no residual degrees of freedom, and its statistic would be 0.0")
    return lag


def degenerate_statistic(statistic: float, what: str) -> None:
    """Refuse a statistic computed from data that is undefined or exactly zero: a regression with no residual
    variation (a perfect fit or no degrees of freedom), never evidence about a unit root."""
    if np.isnan(statistic) or statistic == 0.0:
        raise SpecificationError(f"{what}: the test regression is degenerate (it fits exactly, so the statistic is "
                                 f"{'undefined' if np.isnan(statistic) else '0.0'}); give a smaller maximum lag or "
                                 "use a longer sample")


def _upper(A: np.ndarray) -> np.ndarray:
    """Each matrix's upper Cholesky factor of its Gram matrix, A'A = R'R."""
    return np.swapaxes(np.linalg.cholesky(np.matmul(np.swapaxes(A, 1, 2), A)), 1, 2)


def _qr(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched thin QR of X (D, T, K). CholeskyQR2 on unit-norm columns (two Cholesky passes, as accurate as
    Householder QR for the well-conditioned ADF regressions and many times faster on a stack of them); Householder QR
    when a Gram matrix is not numerically positive definite."""
    scale = np.sqrt(np.einsum("dtk,dtk->dk", X, X))
    scale[scale == 0] = 1.0
    Xs = X / scale[:, None, :]
    try:
        with np.errstate(all="raise"):
            R1 = _upper(Xs)
            Q1 = np.matmul(Xs, np.linalg.inv(R1))
            R2 = _upper(Q1)
            Q, R = np.matmul(Q1, np.linalg.inv(R2)), np.matmul(R2, R1)
        if not (np.isfinite(R).all() and np.isfinite(Q).all()):
            raise np.linalg.LinAlgError
    except (np.linalg.LinAlgError, FloatingPointError):
        Q, R = np.linalg.qr(Xs)
    return Q, R * scale[:, None, :]


def _qr_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched least squares by QR: X (D, T, K), y (D, T) -> Q'y (D, K), R (D, K, K), the residual sum of squares (D,)."""
    Q, R = _qr(X)
    c = np.matmul(np.swapaxes(Q, 1, 2), y[:, :, None])[:, :, 0]
    resid = y - np.matmul(Q, c[:, :, None])[:, :, 0]
    return c, R, np.einsum("dt,dt->d", resid, resid)


def _t_last(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The OLS t-ratio on the last column of each regression (X (D, T, K), y (D, T)); with no residual degrees of
    freedom the standard error is infinite and the ratio a signed zero, as statsmodels gives."""
    c, R, ssr = _qr_fit(X, y)
    df = X.shape[1] - X.shape[2]
    rkk = R[:, -1, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.sqrt(ssr / df) if df > 0 else np.full(len(y), np.inf)
        return (c[:, -1] / rkk) / (s / np.abs(rkk))


def _lag_columns(x: np.ndarray, dx: np.ndarray, lag: int, rows: int) -> list[np.ndarray]:
    """The lagged level and ``lag`` lagged differences over the last ``rows`` differences (each (D, rows))."""
    m = dx.shape[1]
    return [x[:, m - rows:m]] + [dx[:, m - rows - j:m - j] for j in range(1, lag + 1)]


def _search(x: np.ndarray, dx: np.ndarray, trend: str, max_lag: int, method: str) -> np.ndarray:
    """The lag each series' search chooses, on the common sample of ``max_lag``: regressors [terms, level, lags],
    nested, so one QR gives every candidate's sum of squares and last coefficient."""
    D, m = dx.shape
    ntrend, rows = _ntrend(trend), m - max_lag
    X = np.concatenate([np.broadcast_to(trend_terms(rows, trend), (D, rows, ntrend)),
                        np.stack(_lag_columns(x, dx, max_lag, rows), axis=2)], axis=2)
    c, _, ssr_full = _qr_fit(X, dx[:, max_lag:])
    tail = np.concatenate([np.cumsum((c ** 2)[:, ::-1], axis=1)[:, ::-1], np.zeros((D, 1))], axis=1)  # tail[:, j] = sum_{i >= j} c_i^2
    sizes = ntrend + 1 + np.arange(max_lag + 1)  # regressors with 0, 1, ..., max_lag lags
    ssr = ssr_full[:, None] + tail[:, sizes]  # (D, max_lag + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        if method in ("aic", "bic"):
            half = rows / 2.0
            llf = -np.log(ssr) * half - (1 + np.log(np.pi / half)) * half
            penalty = 2.0 if method == "aic" else np.log(rows)
            return np.argmin(-2 * llf + penalty * sizes[None, :], axis=1)  # the first minimum: fewer lags on a tie
        s = np.sqrt(ssr / (rows - sizes))  # no residual degrees of freedom: an infinite s, so |t| = 0
        t = np.abs(c[:, sizes - 1]) / s  # |t| on the last regressor of each nested regression
        chosen = np.zeros(D, dtype=int)
        decided = np.zeros(D, dtype=bool)
        for lag in range(max_lag, 0, -1):  # from the largest lag down, the first significant one
            hit = ~decided & (t[:, lag] >= T_STOP)
            chosen[hit], decided = lag, decided | hit
        return chosen


def adf(x: Any, trend: str = "c", lags: int | None = None, autolag: str | None = "aic", max_lags: int | None = None) -> ADFArrays:
    """ADF statistics of each row of ``x`` ((n,) or (draws, n)), as ``statsmodels.tsa.stattools.adfuller`` computes
    them: ``lags`` fixed with ``autolag=None``, else searched by ``autolag`` up to ``max_lags``."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    D, n = x.shape
    ntrend = _ntrend(trend)
    if autolag is not None and autolag not in AUTOLAGS:
        raise SpecificationError(f"autolag is one of {', '.join(AUTOLAGS)} or None, not {autolag!r}")
    if autolag is None:
        if lags is None:
            raise SpecificationError("give lags, or a rule to choose them (autolag)")
        top = check_lag(lags, n, ntrend)
    else:
        top = default_max_lag(n, ntrend) if max_lags is None else check_lag(max_lags, n, ntrend)
    dx = np.diff(x, axis=1)
    chosen = np.full(D, top) if autolag is None else _search(x, dx, trend, top, autolag)
    stat = np.empty(D)
    for lag in np.unique(chosen):  # each chosen lag refitted on its own sample: [terms, lags, level], the level last
        rows_of = np.flatnonzero(chosen == lag)
        rows = n - 1 - int(lag)
        cols = _lag_columns(x[rows_of], dx[rows_of], int(lag), rows)
        X = np.concatenate([np.broadcast_to(trend_terms(rows, trend), (len(rows_of), rows, ntrend)),
                            np.stack(cols[1:] + cols[:1], axis=2)], axis=2)
        stat[rows_of] = _t_last(X, dx[rows_of, int(lag):])
    return ADFArrays(stat, chosen.astype(int), n - 1 - chosen.astype(int))


def _levels_design(X: np.ndarray, trend: str, exog: np.ndarray | None) -> np.ndarray:
    D, n, _ = X.shape
    parts = [X, np.broadcast_to(trend_terms(n, trend), (D, n, _ntrend(trend)))]
    if exog is not None and exog.size:
        parts.append(np.broadcast_to(exog, (D, n, exog.shape[1])))
    return np.concatenate(parts, axis=2)


def engle_granger(y: Any, X: Any, trend: str = "c", autolag: str | None = "aic", max_lags: int | None = None,
                  exog: Any = None) -> ADFArrays:
    """Engle-Granger statistics: each row of ``y`` ((n,) or (draws, n)) on the matching ``X`` ((n, k) or
    (draws, n, k)), the deterministic terms of ``trend`` and the fixed columns ``exog`` ((n, m)); then the ADF
    regression with no deterministic terms on the residuals, its lags chosen as in ``adf``. As
    ``statsmodels.tsa.stattools.coint`` computes it (which has no ``exog``)."""
    y = np.atleast_2d(np.asarray(y, dtype=float))
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim == 2:
        X = X[None]
    D, n = y.shape
    if X.shape[0] != D or X.shape[1] != n:
        raise SpecificationError("y and X need the same draws and observations")
    E = None if exog is None else np.asarray(exog, dtype=float).reshape(n, -1)
    Z = _levels_design(X, trend, E)
    Q, _ = np.linalg.qr(Z)
    resid = y - np.matmul(Q, np.matmul(np.swapaxes(Q, 1, 2), y[:, :, None]))[:, :, 0]
    ssr = np.einsum("dt,dt->d", resid, resid)
    centred = y - y.mean(axis=1, keepdims=True) if trend != "n" else y
    with np.errstate(divide="ignore", invalid="ignore"):
        rsquared = 1 - ssr / np.einsum("dt,dt->d", centred, centred)
    ok = rsquared < 1 - 100 * SQRTEPS
    stat, lags, nobs = np.full(D, -np.inf), np.zeros(D, dtype=int), np.full(D, n - 1)
    top = default_max_lag(n, 0, Z.shape[2]) if max_lags is None else max_lags
    if ok.any():
        fit = adf(resid[ok], "n", lags=top, autolag=autolag, max_lags=top)
        stat[ok], lags[ok], nobs[ok] = fit.statistic, fit.lags, fit.nobs
    return ADFArrays(stat, lags, nobs)


def _blocks(draws: int, per_draw: int) -> list[int]:
    size = max(1, _BLOCK_ELEMENTS // max(per_draw, 1))
    return [min(size, draws - start) for start in range(0, draws, size)]


def _checked_draws(draws: Any) -> int:
    if isinstance(draws, bool) or int(draws) != draws or int(draws) < 10:
        raise SpecificationError(f"draws is a whole number of at least 10, not {draws!r}")
    return int(draws)


def simulate_adf(n: int, trend: str = "c", lags: int | None = None, autolag: str | None = "aic", max_lags: int | None = None,
                 draws: int = 2000, seed: Any = 0) -> np.ndarray:
    """The ADF statistic of ``draws`` Gaussian random walks of length ``n``, computed as ``adf`` computes it."""
    draws, rng = _checked_draws(draws), make_rng(seed)
    out = [adf(np.cumsum(rng.standard_normal((size, n)), axis=1), trend, lags, autolag, max_lags).statistic
           for size in _blocks(draws, n * (_columns(n, _ntrend(trend), lags, autolag, max_lags) + 1))]
    return np.concatenate(out)


def _columns(n: int, ntrend: int, lags: int | None, autolag: str | None, max_lags: int | None) -> int:
    """Regressors in the largest ADF regression, to size the blocks of draws."""
    top = lags if autolag is None else max_lags
    return ntrend + 2 + (default_max_lag(n, ntrend) if top is None else int(top))


def simulate_engle_granger(n: int, k: int, trend: str = "c", autolag: str | None = "aic", max_lags: int | None = None,
                           exog: Any = None, draws: int = 2000, seed: Any = 0) -> np.ndarray:
    """The Engle-Granger statistic of ``k`` independent Gaussian random walks of length ``n`` (the first on the
    others), with the fixed columns ``exog`` in every levels regression, computed as ``engle_granger`` computes it."""
    if k < 2:
        raise SpecificationError("Engle-Granger needs two or more variables")
    draws, rng = _checked_draws(draws), make_rng(seed)
    out = []
    used = k - 1 + _ntrend(trend) + (0 if exog is None else np.asarray(exog, dtype=float).reshape(n, -1).shape[1])
    top = default_max_lag(n, 0, used) if max_lags is None else max_lags
    for size in _blocks(draws, n * (_columns(n, 0, top, autolag, top) + k + 3)):
        walks = np.cumsum(rng.standard_normal((size, n, k)), axis=1)
        out.append(engle_granger(walks[:, :, 0], walks[:, :, 1:], trend, autolag, max_lags, exog).statistic)
    return np.concatenate(out)


def simulated_summary(statistic: float, sims: np.ndarray) -> dict[str, Any]:
    """Critical values (1, 5 and 10% quantiles of the simulated statistics), the p-value (the share at or below the
    statistic) and its Monte Carlo standard error."""
    sims = np.asarray(sims, dtype=float)
    p = float(np.mean(sims <= statistic))
    return {"critical_values": {f"{q:.0%}": float(np.quantile(sims, q)) for q in (0.01, 0.05, 0.10)},
            "p_value": p, "mcse": float(np.sqrt(p * (1 - p) / len(sims))), "draws": len(sims)}
