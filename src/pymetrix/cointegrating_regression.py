"""Cointegrating regressions: fully modified OLS (Phillips and Hansen 1990), canonical cointegrating regression (Park
1992) and dynamic OLS (Saikkonen 1991; Stock and Watson 1993); and the Gregory-Hansen (1996) test for cointegration
with a break.

    cointegrating_regression(data, "lc", ["ly"], method="fmols")       # or "ccr", "dols"
    cointegrating_regression(data, "lc", ["ly"], method="dols", leads=2, lags=2)
    gregory_hansen(data, "lc", ["ly"], model="regime")                   # "level", "trend" or "regime" shifts

Least squares on cointegrated series is consistent but its distribution depends on nuisance parameters; these
estimators remove the endogeneity and serial-correlation bias so that t statistics are standard normal. FMOLS and CCR
use the long-run covariance Omega and the one-sided covariance Lambda of (u_t, Delta x_t), u the least-squares residual,
by the Bartlett kernel with Newey and West's (1994) automatic bandwidth; their standard errors use Omega_11.2 = Omega_11
- Omega_12 Omega_22^-1 Omega_21. DOLS adds leads and lags of Delta x (chosen by BIC unless given) and uses the
residuals' long-run variance.

The automatic bandwidth (``bandwidth_rule="newey-west"``, the default) is that of Newey and West (1994) and Andrews
(1991), as in arch 9.0 or later and R's sandwich (``bwNeweyWest`` with ``kernHAC``): the pilot lag is the integer
part of 4 (T/100)^(2/9), S = 1.1447 (alpha T)^(1/3) is the bandwidth of the Bartlett kernel k(j/S), and lag j has
the weight 1 - j/S for every j < S, that is 1 - j/(b+1) with b = max(S - 1, 0); FMOLS and CCR round up first, b =
ceil(S) - 1. ``bandwidth_rule="legacy"`` gives pymetrix 0.1.0a18's numbers (and arch's before 9.0): the pilot lag
rounded up, b = S (FMOLS and CCR: ceil(S)), and only the lags j <= floor(b). That rule leaves out lag floor(b) + 1 of
a fractional b, so its long-run variance can be negative; it is kept for one release to reproduce earlier results.
The estimators equal ``arch.unitroot.cointegration``'s: arch 9's under the default rule, and arch 7's and 8's under
"legacy", with the bandwidth chosen automatically or given.

Gregory-Hansen: the Engle-Granger ADF test on the residuals of a regression with a break, at every date in the middle
70% of the sample; the smallest ADF statistic is reported. Critical values are simulated for the sample size, number
of regressors and model (seeded, so they repeat); Gregory and Hansen's Table 1 gives the asymptotic ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from ._api import api
from ._random import generator as make_rng
from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["CointegratingRegression", "cointegrating_regression", "gregory_hansen", "long_run_covariance"]


def _trend(n: int, trend: str) -> np.ndarray:
    t = np.arange(1, n + 1, dtype=float)
    cols = {"n": [], "c": [np.ones(n)], "ct": [np.ones(n), t], "ctt": [np.ones(n), t, t**2]}[trend]
    return np.column_stack(cols) if cols else np.zeros((n, 0))


BANDWIDTH_RULES = ("newey-west", "legacy")


def _check_rule(bandwidth_rule: str) -> bool:
    """True for the legacy rule; refuses an unknown rule."""
    if bandwidth_rule not in BANDWIDTH_RULES:
        raise SpecificationError("bandwidth_rule is 'newey-west' or 'legacy' (pymetrix 0.1.0a18's rule)")
    return bandwidth_rule == "legacy"


def newey_west_bandwidth(x: np.ndarray, bandwidth_rule: str = "newey-west") -> float:
    """Newey and West's (1994) automatic bandwidth S = 1.1447 (alpha T)^(1/3) of the Bartlett kernel k(j/S), with
    alpha = (s1/s0)^2 from the autocovariances sigma_j (divided by T) of the summed columns of x up to the pilot lag n:
    s0 = sigma_0 + 2 sum_{j=1..n} sigma_j, s1 = 2 sum_{j=1..n} j sigma_j.

    "newey-west": n is the integer part of 4 (T/100)^(2/9), as Newey and West, arch 9.0 or later and R's sandwich
    take it, and S is at most T - 1. "legacy" (pymetrix 0.1.0a18, arch before 9.0): n is rounded up, and S is not
    capped. S is returned unrounded; ``long_run_covariance`` turns it into the bandwidth b of the weights."""
    legacy = _check_rule(bandwidth_rule)
    v = x.sum(axis=1)
    T = len(v)
    pilot = 4 * (T / 100) ** (2 / 9)
    n = int(np.ceil(pilot)) if legacy else int(pilot)
    f0 = fq = 0.0
    for j in range(n + 1):
        s = float(v[j:] @ v[:T - j]) / T
        scale = 1 + (j != 0)
        f0 += scale * s
        fq += scale * j * s
    S = float(1.1447 * ((fq / f0) ** 2 * T) ** (1 / 3))
    return S if legacy else min(S, T - 1.0)


def long_run_covariance(x: np.ndarray, bandwidth: float | None = None, force_int: bool = True,
                        bandwidth_rule: str = "newey-west") -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(Omega, Lambda, Sigma, b) of x (not centred): Sigma = Gamma_0, Lambda = Sigma + sum_j w_j Gamma_j, Omega =
    Lambda + Lambda' - Sigma, Gamma_j = sum_t x_t x_{t-j}' / T, with the Bartlett weights w_j = 1 - j/(b+1).

    b is ``bandwidth`` when given; otherwise, from Newey and West's S (``newey_west_bandwidth``), b = max(S - 1, 0)
    under "newey-west", so that the weights 1 - j/S are those of the kernel k(j/S) (as arch 9.0 or later and R's
    sandwich ``kernHAC``), and b = S under "legacy". ``force_int`` rounds b up to a whole number: under "newey-west"
    the automatic b is then ceil(S) - 1 (arch's ``force_int``, and sandwich's ``NeweyWest`` lag floor(S) when S is
    fractional); under "legacy" ceil(S). A given bandwidth is rounded up too.

    Lags used: "newey-west" uses every lag j < b + 1, each with a positive weight, so a fractional b uses lag
    floor(b) + 1 and the estimate is positive semi-definite for any b; "legacy" stops at floor(b), as pymetrix
    0.1.0a18 and arch before 9.0 did, which is not a Bartlett kernel for a fractional b and can give a negative long-run
    variance. A whole-number b uses the lags 1..b under either rule."""
    legacy = _check_rule(bandwidth_rule)
    x = np.atleast_2d(np.asarray(x, float).T).T
    T = len(x)
    if bandwidth is not None:
        b = float(bandwidth)
    elif legacy:
        b = newey_west_bandwidth(x, bandwidth_rule)
    else:
        S = newey_west_bandwidth(x, bandwidth_rule)
        b = max((float(np.ceil(S)) if force_int else S) - 1.0, 0.0)
    if force_int:
        b = float(np.ceil(b))
    sigma = x.T @ x / T
    oss = np.zeros_like(sigma)
    for j, w in enumerate(bartlett_weights(b, bandwidth_rule)[:T - 1], start=1):
        oss += w * (x[j:].T @ x[:-j]) / T
    return sigma + oss + oss.T, sigma + oss, sigma, b


def bartlett_weights(b: float, bandwidth_rule: str = "newey-west") -> np.ndarray:
    """The weights w_1, w_2, ... of lags 1, 2, ... for the bandwidth b: w_j = 1 - j/(b+1) for every j < b + 1
    ("newey-west"), or only for j <= floor(b) ("legacy", pymetrix 0.1.0a18)."""
    legacy = _check_rule(bandwidth_rule)
    top = int(b + 1) if legacy else int(np.ceil(b + 1))
    j = np.arange(1, top, dtype=float)
    return (b + 1 - j) / (b + 1)


@dataclass(frozen=True, eq=False)
class CointegratingRegression:
    """A cointegrating regression estimated by FMOLS, CCR or DOLS."""

    method: str
    y_name: str
    names: tuple[str, ...]
    params: np.ndarray
    cov: np.ndarray = field(repr=False)
    resid: pd.Series = field(repr=False)
    long_run_variance: float = 0.0
    bandwidth: float = 0.0
    leads: int | None = None
    lags: int | None = None
    r2: float = 0.0
    bandwidth_rule: str = "newey-west"

    stage = "estimate"

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def summary(self) -> Table:
        z = self.params / self.se
        rows = [[float(b), float(s), float(t), float(2 * sps.norm.sf(abs(t)))] for b, s, t in zip(self.params, self.se, z)]
        label = {"fmols": "Fully modified OLS", "ccr": "Canonical cointegrating regression", "dols": "Dynamic OLS"}[self.method]
        notes = [f"{label}; {len(self.resid)} observations; R^2 {self.r2:.4f}; long-run variance {self.long_run_variance:.6g}; "
                 f"Bartlett bandwidth {self.bandwidth:g}"
                 + (" (legacy rule of pymetrix 0.1.0a18)." if self.bandwidth_rule == "legacy" else ".")]
        if self.method == "dols":
            notes.append(f"{self.leads} leads and {self.lags} lags of the differenced regressors (their coefficients not shown).")
        notes.append("Standard errors are asymptotically normal: the cointegrating vector can be tested with z statistics.")
        return Table(["estimate", "std err", "z", "p-value"], rows, index=list(self.names), index_name="parameter",
                     title=f"Cointegrating regression: {self.y_name}", notes=tuple(notes))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return pd.DataFrame({"estimate": self.params, "se": self.se}, index=list(self.names))

    def __repr__(self) -> str:
        return f"<CointegratingRegression {self.method} {self.y_name}: {dict(zip(self.names, np.round(self.params, 4)))}>"


def _frame(data: Dataset, y: str, x: Sequence[str], span: Any, where: str | None = None) -> pd.DataFrame:
    """The rows of ``span`` and ``where`` with ``y`` and ``x`` observed; the rows kept are taken as consecutive periods (as
    rows with missing values dropped are)."""
    if data.kind != "time_series":
        raise SpecificationError("cointegrating regressions need a single time series")
    missing = [v for v in [y, *x] if v not in data]
    if missing:
        raise SpecificationError(f"{', '.join(missing)} are not series of the data")
    frame = data.frame[[y, *x]].astype(float)
    if span is not None or where:
        frame = frame[data.sample_mask(span, where or None)]
    frame = frame.dropna()
    if len(frame) < 20:
        raise SpecificationError("too few observations for a cointegrating regression")
    return frame


def _r2(y: np.ndarray, resid: np.ndarray, trend: str) -> float:
    centre = y.mean() if "c" in trend else 0.0
    return float(1 - resid @ resid / np.sum((y - centre) ** 2))


@api("pymetrix:cointegrating_regression")
def cointegrating_regression(data: Dataset, y: str, x: Sequence[str], method: Literal["fmols", "ccr", "dols"] = "fmols",
                             trend: Literal["n", "c", "ct", "ctt"] = "c", bandwidth: float | None = None,
                             leads: int | None = None, lags: int | None = None, span: Any = None,
                             where: str | None = None, bandwidth_rule: Literal["newey-west", "legacy"] = "newey-west") -> CointegratingRegression:
    """The cointegrating regression of ``y`` on ``x`` (and ``trend``) by FMOLS, CCR or DOLS, over the rows of ``span`` (first,
    last) and ``where`` (a condition on the rows, e.g. ``"year >= 1990"``). ``bandwidth_rule`` is "newey-west" (Newey and
    West 1994, as arch 9.0 and R's sandwich) or "legacy" (pymetrix 0.1.0a18's automatic bandwidth and lags, kept for one
    release to reproduce earlier results); see ``long_run_covariance``."""
    x = [x] if isinstance(x, str) else list(x)
    _check_rule(bandwidth_rule)
    if method not in ("fmols", "ccr", "dols"):
        raise SpecificationError("method is 'fmols', 'ccr' or 'dols'")
    if trend not in ("n", "c", "ct", "ctt"):
        raise SpecificationError("trend is 'n', 'c', 'ct' or 'ctt'")
    frame = _frame(data, y, x, span, where)
    Y, X = frame[y].to_numpy(), frame[x].to_numpy()
    T, k = X.shape
    D = _trend(T, trend)
    names = [*x, *{"n": [], "c": ["const"], "ct": ["const", "trend"], "ctt": ["const", "trend", "trend^2"]}[trend]]
    if method == "dols":
        return _dols(frame, y, x, Y, X, D, trend, names, bandwidth, leads, lags, bandwidth_rule)
    Z = np.column_stack([X, D])
    b_ols = np.linalg.lstsq(Z, Y, rcond=None)[0]
    u = Y - Z @ b_ols
    eps = X - D @ np.linalg.lstsq(D, X, rcond=None)[0] if D.shape[1] else X
    eta = np.column_stack([u[1:], np.diff(eps, axis=0)])
    omega, lam, sigma, b = long_run_covariance(eta, bandwidth, bandwidth_rule=bandwidth_rule)
    o12, o22i = omega[:1, 1:], np.linalg.inv(omega[1:, 1:])
    omega_112 = float(omega[0, 0] - (o12 @ o22i @ o12.T)[0, 0])
    Zs = Z[1:]
    if method == "fmols":
        y_dot = Y[1:] - eta[:, 1:] @ o22i @ o12.T[:, 0]
        bias = np.zeros(Z.shape[1])
        bias[:k] = (lam[:1, 1:] - o12 @ o22i @ lam[1:, 1:])[0]
        params = np.linalg.solve(Zs.T @ Zs, Zs.T @ y_dot - (T - 1) * bias)
        cov = omega_112 * np.linalg.inv(Zs.T @ Zs)
    else:  # CCR: both sides transformed
        lam2 = lam[:, 1:]
        sinv = np.linalg.inv(sigma)
        x_star = X[1:] - eta @ (sinv @ lam2)
        shift = sinv @ lam2 @ b_ols[:k] + np.r_[0.0, (o22i @ o12.T)[:, 0]]
        y_star = Y[1:] - eta @ shift
        Zs = np.column_stack([x_star, _trend(T - 1, trend)])  # the trend counts from 1 on the transformed rows (arch's convention)
        params = np.linalg.lstsq(Zs, y_star, rcond=None)[0]
        cov = omega_112 * np.linalg.inv(Zs.T @ Zs)
    resid = Y - Z @ params
    return CointegratingRegression(method, y, tuple(names), params, cov, pd.Series(resid, index=frame.index, name="resid"),
                                   omega_112, b, None, None, _r2(Y, resid, trend), bandwidth_rule)


def _dols(frame: pd.DataFrame, y: str, x: list[str], Y: np.ndarray, X: np.ndarray, D: np.ndarray, trend: str, names: list[str],
          bandwidth: float | None, leads: int | None, lags: int | None, bandwidth_rule: str = "newey-west") -> CointegratingRegression:
    T, k = X.shape
    dX = np.vstack([np.full((1, k), np.nan), np.diff(X, axis=0)])

    def design(nlead: int, nlag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rows where every lead and lag exists: y, [x, trend, dx_{t+lead}..dx_{t-lag}], their row numbers."""
        cols = [X]
        for s in range(-nlag, nlead + 1):  # dx shifted: s > 0 a lead
            shifted = np.full((T, k), np.nan)
            if s >= 0:
                shifted[:T - s] = dX[s:]
            else:
                shifted[-s:] = dX[:T + s]
            cols.append(shifted)
        M = np.column_stack(cols)
        ok = np.all(np.isfinite(M), axis=1)
        M = M[ok]
        trend_cols = _trend(int(ok.sum()), trend)  # the trend counts from 1 on the rows used, as arch's does
        return Y[ok], np.column_stack([M[:, :k], trend_cols, M[:, k:]]), np.flatnonzero(ok)

    if leads is None or lags is None:
        top = int(np.ceil(12 * (T / 100) ** 0.25))
        lead_range = range(0, top + 1) if leads is None else [leads]
        lag_range = range(0, top + 1) if lags is None else [lags]
        ymax, Mmax, rows = design(max(lead_range), max(lag_range))
        best, choice = np.inf, (0, 0)
        for lg in lag_range:
            for ld in lead_range:
                keep = list(range(k + D.shape[1]))
                first = k + D.shape[1]
                for s_index, s in enumerate(range(-max(lag_range), max(lead_range) + 1)):
                    if -lg <= s <= ld:
                        keep += list(range(first + s_index * k, first + (s_index + 1) * k))
                Msub = Mmax[:, keep]
                b = np.linalg.lstsq(Msub, ymax, rcond=None)[0]
                e = ymax - Msub @ b
                n = len(e)
                ic = np.log(e @ e / n) + Msub.shape[1] * np.log(n) / n
                if ic < best:
                    best, choice = ic, (ld, lg)
        leads, lags = choice
    yv, M, rows = design(leads, lags)
    b = np.linalg.lstsq(M, yv, rcond=None)[0]
    e = yv - M @ b
    n = len(e)
    s2, _, _, bw = long_run_covariance(e[:, None], bandwidth, force_int=False, bandwidth_rule=bandwidth_rule)
    cov_all = float(s2[0, 0]) * np.linalg.inv(M.T @ M / n) / n
    m = k + D.shape[1]
    return CointegratingRegression("dols", y, tuple(names), b[:m], cov_all[:m, :m], pd.Series(e, index=frame.index[rows], name="resid"),
                                   float(s2[0, 0]), bw, int(leads), int(lags), _r2(yv, e, trend), bandwidth_rule)


# Gregory-Hansen

# Gregory and Hansen (1996), Table 1, ADF* and Zt*: 1%, 5%, 10%, by number of regressors m
GH_CRITICAL = {
    "level": {1: (-5.13, -4.61, -4.34), 2: (-5.44, -4.92, -4.69), 3: (-5.77, -5.28, -5.02), 4: (-6.05, -5.56, -5.31)},
    "trend": {1: (-5.45, -4.99, -4.72), 2: (-5.80, -5.29, -5.03), 3: (-6.05, -5.57, -5.33), 4: (-6.36, -5.83, -5.59)},
    "regime": {1: (-5.47, -4.95, -4.68), 2: (-5.97, -5.50, -5.23), 3: (-6.51, -6.00, -5.75), 4: (-6.92, -6.41, -6.17)},
}


def _adf_t(u: np.ndarray, lags: int) -> float:
    du = np.diff(u)
    X = np.column_stack([u[lags:-1]] + [du[lags - j:len(du) - j] for j in range(1, lags + 1)]) if lags else u[:-1, None]
    yv = du[lags:]
    XtX = np.linalg.inv(X.T @ X)
    b = XtX @ X.T @ yv
    e = yv - X @ b
    return float(b[0] / np.sqrt(e @ e / (len(e) - X.shape[1]) * XtX[0, 0]))


def _gh_statistic(Y: np.ndarray, X: np.ndarray, model: str, lags: int, trim: float) -> tuple[float, int]:
    T = len(Y)
    t = np.arange(1, T + 1, dtype=float)
    best, at = np.inf, -1
    for tb in range(int(np.floor(trim * T)), int(np.ceil((1 - trim) * T))):
        phi = (np.arange(T) > tb).astype(float)
        cols = [np.ones(T), phi]
        if model == "trend":
            cols.append(t)
        cols.append(X)
        if model == "regime":
            cols.append(X * phi[:, None])
        Z = np.column_stack(cols)
        u = Y - Z @ np.linalg.lstsq(Z, Y, rcond=None)[0]
        stat = _adf_t(u, lags)
        if stat < best:
            best, at = stat, tb
    return best, at


@api("pymetrix:gregory_hansen")
def gregory_hansen(data: Dataset, y: str, x: Sequence[str], model: Literal["level", "trend", "regime"] = "level", lags: int = 1,
                   trim: float = 0.15, reps: int = 500, seed: int = 0, span: Any = None, where: str | None = None) -> Any:
    """Gregory and Hansen's test of no cointegration against cointegration with one break at an unknown date: a level
    shift, a level shift with a trend, or a regime shift (level and slopes)."""
    from .stats.base import TestResult

    x = [x] if isinstance(x, str) else list(x)
    if model not in GH_CRITICAL:
        raise SpecificationError("model is 'level', 'trend' or 'regime'")
    frame = _frame(data, y, x, span, where)
    Y, X = frame[y].to_numpy(), frame[x].to_numpy()
    T, m = X.shape
    stat, tb = _gh_statistic(Y, X, model, lags, trim)
    rng = make_rng(seed)
    sims = np.array([_gh_statistic(np.cumsum(rng.standard_normal(T)), np.cumsum(rng.standard_normal((T, m)), axis=0), model, lags, trim)[0]
                     for _ in range(int(reps))])
    p = float((np.sum(sims <= stat) + 1) / (len(sims) + 1))
    crit = {f"{q:.0%}": float(np.quantile(sims, q)) for q in (0.01, 0.05, 0.10)}
    details: dict[str, Any] = {"break": str(frame.index[tb + 1]), "lags": lags, "replications": int(reps)}
    if m in GH_CRITICAL[model]:
        details.update({f"asymptotic {lv}": v for lv, v in zip(("1%", "5%", "10%"), GH_CRITICAL[model][m])})
    return TestResult(f"Gregory-Hansen ({model} shift)", "no cointegration (against cointegration with one break)", stat, None, p,
                      "simulated", critical_values=crit, details=details)
