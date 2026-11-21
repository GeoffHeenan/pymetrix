"""Panel cointegration tests: Pedroni's seven residual-based statistics and Kao's pooled tests.

    data.test("pedroni", "y, x1, x2")                 # the null: no cointegration in any entity
    data.test("kao", "y, x", lags=1)

Pedroni (1999, 2004) regresses each entity's first variable on the others (with its own constant, and trend if
asked) and tests the residuals for a unit root in seven ways: four pool along the within dimension (panel v, rho, PP
t and ADF t: a common autoregressive coefficient under the alternative) and three average along the between
dimension (group rho, PP t and ADF t: a coefficient of each entity's own). The statistics are standardised by the
mean and variance of their parts under the null. Pedroni tabulates the asymptotic moments; here they are simulated
for the panel's own T, number of regressors, trend and lags (independent random walks; seeded and cached), so short
panels are standardised by their own moments; at large T the simulation reproduces Pedroni's Table 2. Panel v is
large under the alternative; the other six are large and negative.

Kao (1999) pools: one regression with entity constants (the within regression), its residuals tested by Dickey-Fuller
and augmented Dickey-Fuller regressions, adjusted with the long-run covariance of the residuals and the regressors'
differences. Kao derives the statistics for one regressor; the adjustments use the conditional (long-run) variance of
the residuals given the differences, which is how they extend to several, as EViews and Stata compute them.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, ClassVar, Literal

import numpy as np

from pymetrix._lazy import stats as sps

from .._random import generator as make_rng
from . import adf_arrays
from ..data import Dataset
from ..errors import SpecificationError
from ..forms import Widget, advanced
from .base import Test, TestResult, register
from .data_tests import _int_or

STATISTICS = ("panel v", "panel rho", "panel PP t", "panel ADF t", "group rho", "group PP t", "group ADF t")


def _variables(spec: str, test: str) -> list[str]:
    names = [s.strip() for s in spec.split(",") if s.strip()]
    if len(names) < 2:
        raise SpecificationError(f"{test} needs two or more variables; the first is the dependent variable")
    return names


def _entities(data: Dataset, names: list[str], test: str) -> list[tuple[str, np.ndarray]]:
    """Each entity's (T, k) block of the variables, the rows where all are observed."""
    if data.kind != "panel" or len(data.entity_axes) != 1:
        raise SpecificationError(f"{test} needs a panel with one entity axis; for one series use Engle-Granger")
    entity = data.entity_axes[0].name
    out = [(str(key), g.dropna().to_numpy(float)) for key, g in data.frame[names].groupby(level=entity)]
    out = [(key, v) for key, v in out if len(v)]
    if len(out) < 2:
        raise SpecificationError(f"{test} needs at least two entities with data")
    return out


def _bartlett(u: np.ndarray, k: int) -> np.ndarray:
    """Newey-West weighted autocovariances sum_{s=1..k} (1 - s/(k+1)) sum_t u_t u_t-s / T, over the last axis."""
    T = u.shape[-1]
    return sum((1 - s / (k + 1)) * np.einsum("...t,...t->...", u[..., s:], u[..., :-s]) for s in range(1, k + 1)) / T if k else np.zeros(u.shape[:-1])


def _detrend_resid(y: np.ndarray, X: np.ndarray, trend: str) -> np.ndarray:
    """Residuals of y (R, T) on X (R, T, m) and the deterministic terms, row by row."""
    R, T = y.shape
    det = [np.ones((R, T))] if trend != "n" else []
    if trend == "ct":
        det.append(np.broadcast_to(np.arange(1, T + 1.0), (R, T)))
    Z = np.concatenate([X] + [d[..., None] for d in det], axis=2) if det else X
    b = np.linalg.solve(np.einsum("rti,rtj->rij", Z, Z), np.einsum("rti,rt->ri", Z, y)[..., None])[..., 0]
    return y - np.einsum("rti,ri->rt", Z, b)


def _ols_resid(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    b = np.linalg.solve(np.einsum("rti,rtj->rij", Z, Z), np.einsum("rti,rt->ri", Z, y)[..., None])[..., 0]
    return y - np.einsum("rti,ri->rt", Z, b)


def _pedroni_parts(Y: np.ndarray, X: np.ndarray, trend: str, p: int, k: int) -> dict[str, np.ndarray]:
    """Each row's ingredients of Pedroni's statistics (Pedroni 1999, Table 1), for rows of equal length T.

    Y (R, T) and X (R, T, m). Returns arrays of length R: A = sum L^-2 (e_t-1 de_t - lambda) / T, B = sum L^-2 e_t-1^2 /
    T^2, C = L^-2 sigma^2, the same unweighted (a, b, sigma2), and the ADF parts (A*, B*, s*^2 and a*, b*)."""
    R, T = Y.shape
    e = _detrend_resid(Y, X, trend)  # step 1
    eta = _ols_resid(np.diff(Y, axis=1), np.diff(X, axis=1))  # step 2: dy on dx, for L11^2
    L2 = np.einsum("rt,rt->r", eta, eta) / (T - 1) + 2 * _bartlett(eta, k)
    el, de = e[:, :-1], np.diff(e, axis=1)
    gamma = np.einsum("rt,rt->r", el, e[:, 1:]) / np.einsum("rt,rt->r", el, el)  # step 3: e_t on e_t-1
    u = e[:, 1:] - gamma[:, None] * el
    n = T - 1
    s2 = np.einsum("rt,rt->r", u, u) / n
    lam = _bartlett(u, k)
    sigma2 = s2 + 2 * lam
    a = (np.einsum("rt,rt->r", el, de) / n - lam)
    b = np.einsum("rt,rt->r", el, el) / n**2
    if p:  # step 4: the ADF regression, its lagged differences partialled out
        lags = np.stack([de[:, p - j:n - j] for j in range(1, p + 1)], axis=2)
        el_s, de_s = _ols_resid(el[:, p:], lags), _ols_resid(de[:, p:], lags)
    else:
        el_s, de_s = el, de
    gs = np.einsum("rt,rt->r", el_s, de_s) / np.einsum("rt,rt->r", el_s, el_s)
    us = de_s - gs[:, None] * el_s
    ns = n - p
    s2_star = np.einsum("rt,rt->r", us, us) / ns
    a_star = np.einsum("rt,rt->r", el_s, de_s) / ns
    b_star = np.einsum("rt,rt->r", el_s, el_s) / ns**2
    return {"A": a / L2, "B": b / L2, "C": sigma2 / L2, "a": a, "b": b, "sigma2": sigma2,
            "As": a_star / L2, "Bs": b_star / L2, "s2s": s2_star, "as": a_star, "bs": b_star}


def _statistics(parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The seven statistics before standardisation, as functions of the parts' means (panel) or of each entity's parts
    (group, then averaged), each scaled so that it is sqrt(N) times a function of averages."""
    return {
        "panel v": 1 / parts["B"],
        "panel rho": parts["A"] / parts["B"],
        "panel PP t": parts["A"] / np.sqrt(parts["C"] * parts["B"]),
        "panel ADF t": parts["As"] / np.sqrt(parts["s2s"] * parts["Bs"]),
        "group rho": parts["a"] / parts["b"],
        "group PP t": parts["a"] / np.sqrt(parts["sigma2"] * parts["b"]),
        "group ADF t": parts["as"] / np.sqrt(parts["s2s"] * parts["bs"]),
    }


_PANEL_KEYS = {"panel v": ("B",), "panel rho": ("A", "B"), "panel PP t": ("A", "C", "B"), "panel ADF t": ("As", "s2s", "Bs")}


@lru_cache(maxsize=128)
def _pedroni_null(T: int, m: int, trend: str, p: int, k: int, replications: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[float, float]]]:
    """Under the null (m + 1 independent Gaussian random walks of length T): the means and covariance of the parts
    the panel statistics are built from, and the mean and variance of each group statistic's term."""
    rng = make_rng(seed)
    keys = sorted({key for ks in _PANEL_KEYS.values() for key in ks})
    chunk = max(200, 2_000_000 // (T * (m + 2)))
    rows, group = [], {name: [] for name in STATISTICS[4:]}
    for start in range(0, replications, chunk):
        r = min(chunk, replications - start)
        W = np.cumsum(rng.normal(size=(r, T, m + 1)), axis=1)
        parts = _pedroni_parts(W[:, :, 0], W[:, :, 1:], trend, p, k)
        rows.append(np.column_stack([parts[key] for key in keys]))
        stats = _statistics(parts)
        for name in group:
            group[name].append(stats[name])
    M = np.concatenate(rows)
    return M.mean(axis=0), np.cov(M, rowvar=False), {name: (float(np.mean(np.concatenate(v))), float(np.var(np.concatenate(v), ddof=1)))
                                                     for name, v in group.items()}


def _panel_function(name: str, means: dict[str, float]) -> float:
    return float(_statistics({key: np.asarray(means.get(key, np.nan)) for key in ("A", "B", "C", "a", "b", "sigma2", "As", "Bs", "s2s", "as", "bs")})[name])


def _delta_moments(name: str, mean: dict[str, float], cov: dict[tuple[str, str], float]) -> tuple[float, float]:
    """The mean and variance of sqrt(N) (g(averages) - g(means)) by the delta method, g the panel statistic's function."""
    keys = _PANEL_KEYS[name]
    mu = _panel_function(name, mean)
    grad = []
    for key in keys:
        h = 1e-6 * max(abs(mean[key]), 1e-8)
        up, dn = dict(mean), dict(mean)
        up[key] += h
        dn[key] -= h
        grad.append((_panel_function(name, up) - _panel_function(name, dn)) / (2 * h))
    S = np.array([[cov[(i, j)] for j in keys] for i in keys])
    g = np.array(grad)
    return mu, float(g @ S @ g)


def _default_bandwidth(T: int) -> int:
    return int(4 * (T / 100) ** (2 / 9))


def _adf_lags(e: np.ndarray, max_lags: int, criterion: str) -> int:
    """The lag order of the residuals' ADF regression (no constant) by AIC or BIC, on a common sample."""
    de = np.diff(e)
    n = len(de) - max_lags
    best, best_p = np.inf, 0
    for p in range(max_lags + 1):
        cols = [e[max_lags:-1]] + [de[max_lags - j:len(de) - j] for j in range(1, p + 1)]
        Z = np.column_stack(cols)
        u = de[max_lags:] - Z @ np.linalg.lstsq(Z, de[max_lags:], rcond=None)[0]
        ic = n * np.log(u @ u / n) + (2 if criterion == "aic" else np.log(n)) * Z.shape[1]
        if ic < best - 1e-12:
            best, best_p = ic, p
    return best_p


@register
@dataclass(frozen=True)
class Pedroni(Test):
    """Pedroni's panel cointegration tests: four within-dimension (panel) and three between-dimension (group)
    statistics of the residuals of each entity's cointegrating regression.

    The null is no cointegration in any entity; each entity has its own cointegrating coefficients. The four panel
    statistics pool the residuals' autoregressive coefficient, so their alternative is cointegration in every entity
    with a common coefficient; the three group statistics average entity by entity and let it differ. The statistic
    shown is group ADF t; all seven, standardised to N(0, 1), are in the details. Panel v rejects when large, the
    other six when large and negative, and they can disagree in short panels. The entities are assumed independent:
    a common shock makes the tests reject too often. Up to seven regressors.
    """

    maturity: ClassVar[str] = "beta"
    variables: Annotated[str, Widget("", help="the series, comma-separated; in each entity the first is regressed on the others")]
    trend: Annotated[Literal["c", "ct", "n"], Widget("", help="each entity's terms in its cointegrating regression: c constant, ct constant and trend, n none")] = "c"
    lags: Annotated[str | int, advanced(help="lagged differences in the ADF statistics' regressions, or aic / bic to choose entity by entity")] = "aic"
    bandwidth: Annotated[str | int, advanced(help="Bartlett lags for the long-run variances; auto = 4 (T / 100)^(2/9)")] = "auto"
    replications: Annotated[int, advanced(help="simulated entities for the moments under the null")] = 20_000
    seed: Annotated[int, advanced()] = 0

    name = "pedroni"
    topics = ("cointegration", "panel-data")
    references = (
        ("Pedroni, P. (1999). Critical values for cointegration tests in heterogeneous panels with multiple regressors. Oxford Bulletin of Economics and Statistics 61(s1), 653-670. doi:10.1111/1468-0084.61.s1.14"),
        ("Pedroni, P. (2004). Panel cointegration: asymptotic and finite sample properties of pooled time series tests with an application to the PPP hypothesis. Econometric Theory 20(3), 597-625. doi:10.1017/s0266466604203073"),
    )
    null = "no cointegration in any entity"
    applies_to = (Dataset,)

    def run(self, data: Dataset) -> TestResult:
        names = _variables(self.variables, "Pedroni's test")
        m = len(names) - 1
        if m > 7:
            raise SpecificationError("Pedroni's test takes up to seven regressors")
        entities = _entities(data, names, "Pedroni's test")
        chosen = _int_or(self.lags, ("aic", "bic"))
        reps, seed = int(self.replications), int(self.seed)
        per_entity, keys_used = [], []
        for key, v in entities:
            T = len(v)
            k = _default_bandwidth(T) if str(self.bandwidth).strip().lower() in ("auto", "") else int(self.bandwidth)
            if isinstance(chosen, int):
                p = chosen
            else:
                levels = m + (self.trend != "n") + (self.trend == "ct")  # the cointegrating regression's columns
                max_lags = min(int(12 * (T / 100) ** 0.25), adf_arrays.default_max_lag(T, 0, levels))
                e = _detrend_resid(v[None, :, 0], v[None, :, 1:], self.trend)[0]
                p = _adf_lags(e, max_lags, chosen or "aic")
            if T - p - 2 - m - (self.trend != "n") - (self.trend == "ct") < 5:
                raise SpecificationError(f"entity {key} has too few periods ({T}) for {m} regressors and {p} lags")
            parts = _pedroni_parts(v[None, :, 0], v[None, :, 1:], self.trend, p, k)
            per_entity.append({name: float(val[0]) for name, val in parts.items()})
            keys_used.append((T, p, k))
        N = len(entities)
        nulls = [_pedroni_null(T, m, self.trend, p, k, reps, seed) for T, p, k in keys_used]
        sorted_keys = sorted({key for ks in _PANEL_KEYS.values() for key in ks})
        mean = dict(zip(sorted_keys, np.mean([nl[0] for nl in nulls], axis=0)))
        C = np.mean([nl[1] for nl in nulls], axis=0)
        cov = {(i, j): float(C[a, b]) for a, i in enumerate(sorted_keys) for b, j in enumerate(sorted_keys)}
        rows, values = [], {}
        averages = {key: float(np.mean([e[key] for e in per_entity])) for key in per_entity[0]}
        for name in STATISTICS:
            if name in _PANEL_KEYS:
                raw = _panel_function(name, averages)
                mu, var = _delta_moments(name, mean, cov)
            else:
                raw = float(np.mean([float(_statistics({k2: np.asarray(v2) for k2, v2 in e.items()})[name]) for e in per_entity]))
                mu = float(np.mean([nl[2][name][0] for nl in nulls]))
                var = float(np.mean([nl[2][name][1] for nl in nulls]))
            z = float(np.sqrt(N) * (raw - mu) / np.sqrt(var))
            p_value = float(sps.norm.sf(z) if name == "panel v" else sps.norm.cdf(z))
            values[name] = (z, p_value)
            rows.append(f"{z:.3f} (p = {p_value:.4f})")
        lags = sorted({p for _, p, _ in keys_used})
        details = {name: row for name, row in zip(STATISTICS, rows)}
        details.update({"entities": N, "regressors": m,
                        "lags": lags[0] if len(lags) == 1 else f"{lags[0]} to {lags[-1]}",
                        "Bartlett lags": sorted({k for *_, k in keys_used})[0],
                        "simulated entities for the moments": reps})
        z, p_value = values["group ADF t"]
        return TestResult(f"Pedroni panel cointegration ({', '.join(names)})", self.null, z, None, p_value, "normal",
                          details=details)


@register
@dataclass(frozen=True)
class Kao(Test):
    """Kao's panel cointegration tests: Dickey-Fuller and ADF tests on the residuals of the pooled within regression.

    The null is no cointegration. The within regression gives each entity its own constant but the same slopes, so
    the alternative is cointegration with a common cointegrating vector; when the slopes may differ by entity,
    Pedroni's or Westerlund's tests suit better. There is no trend option. The statistic shown is the ADF one; the
    four Dickey-Fuller statistics are in the details, all standard normal under the null and rejecting when large and
    negative. DF rho*, DF t* and ADF use the long-run covariances to allow endogenous regressors and serially
    correlated errors; DF rho and DF t assume neither. It needs a balanced panel and assumes independent entities.
    """

    maturity: ClassVar[str] = "beta"
    variables: Annotated[str, Widget("", help="the series, comma-separated; the first is regressed on the others, pooled with entity constants")]
    lags: Annotated[int, advanced(help="lagged differences in the pooled ADF regression of the residuals")] = 1
    bandwidth: Annotated[str | int, advanced(help="Bartlett lags for the long-run covariance; auto = 4 (T / 100)^(2/9)")] = "auto"

    name = "kao"
    topics = ("cointegration", "panel-data")
    references = (
        ("Kao, C. (1999). Spurious regression and residual-based tests for cointegration in panel data. Journal of Econometrics 90(1), 1-44. doi:10.1016/s0304-4076(98)00023-2"),
    )
    null = "no cointegration"
    applies_to = (Dataset,)

    def run(self, data: Dataset) -> TestResult:
        names = _variables(self.variables, "Kao's test")
        entities = _entities(data, names, "Kao's test")
        lengths = {len(v) for _, v in entities}
        if len(lengths) != 1:
            raise SpecificationError("Kao's test needs a balanced panel (every entity with the same periods); use balance()")
        T, N, p = lengths.pop(), len(entities), int(self.lags)
        k = _default_bandwidth(T) if str(self.bandwidth).strip().lower() in ("auto", "") else int(self.bandwidth)
        V = np.stack([v for _, v in entities])  # (N, T, 1 + m)
        D = V - V.mean(axis=1, keepdims=True)  # the within transformation
        y, X = D[:, :, 0].reshape(-1), D[:, :, 1:].reshape(N * T, -1)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        e = (y - X @ beta).reshape(N, T)
        # the long-run and contemporaneous covariances of (u, dx): under the null e_t = e_t-1 + u_t, so u = de
        u = np.diff(e, axis=1)
        dx = np.diff(V[:, :, 1:], axis=1)
        W = np.concatenate([u[..., None], dx], axis=2)  # (N, T - 1, 1 + m)
        W = W - W.mean(axis=1, keepdims=True)
        n = T - 1
        Sigma = np.einsum("nti,ntj->ij", W, W) / (N * n)
        Omega = Sigma.copy()
        for s in range(1, k + 1):
            G = np.einsum("nti,ntj->ij", W[:, s:], W[:, :-s]) / (N * n)
            Omega += (1 - s / (k + 1)) * (G + G.T)
        s2_v = float(Sigma[0, 0] - Sigma[0, 1:] @ np.linalg.solve(Sigma[1:, 1:], Sigma[1:, 0]))
        s2_0v = float(Omega[0, 0] - Omega[0, 1:] @ np.linalg.solve(Omega[1:, 1:], Omega[1:, 0]))
        sv, s0v = np.sqrt(s2_v), np.sqrt(s2_0v)
        # Dickey-Fuller: e_t on e_t-1, pooled
        el, ec = e[:, :-1].reshape(-1), e[:, 1:].reshape(-1)
        rho = float(el @ ec / (el @ el))
        res = ec - rho * el
        se2 = float(res @ res / (len(res) - 1))
        t_rho = (rho - 1) / np.sqrt(se2 / (el @ el))
        sqN = np.sqrt(N)
        df_rho = (sqN * T * (rho - 1) + 3 * sqN) / np.sqrt(10.2)
        df_t = np.sqrt(1.25) * t_rho + np.sqrt(1.875 * N)
        df_rho_star = (sqN * T * (rho - 1) + 3 * sqN * s2_v / s2_0v) / np.sqrt(3 + 36 * s2_v**2 / (5 * s2_0v**2))
        df_t_star = (t_rho + np.sqrt(6 * N) * sv / (2 * s0v)) / np.sqrt(s2_0v / (2 * s2_v) + 3 * s2_v / (10 * s2_0v))
        # ADF: de_t on e_t-1 and p lagged differences, pooled
        de = np.diff(e, axis=1)
        Z = [e[:, p:-1]] + [de[:, p - j:n - j] for j in range(1, p + 1)]
        Zm = np.column_stack([z.reshape(-1) for z in Z])
        d = de[:, p:].reshape(-1)
        b = np.linalg.lstsq(Zm, d, rcond=None)[0]
        r = d - Zm @ b
        s2 = float(r @ r / (len(d) - Zm.shape[1]))
        t_adf = float(b[0] / np.sqrt(s2 * np.linalg.inv(Zm.T @ Zm)[0, 0]))
        adf = (t_adf + np.sqrt(6 * N) * sv / (2 * s0v)) / np.sqrt(s2_0v / (2 * s2_v) + 3 * s2_v / (10 * s2_0v))
        values = {"ADF": adf, "DF rho*": df_rho_star, "DF t*": df_t_star, "DF rho": df_rho, "DF t": df_t}
        details = {name: f"{float(v):.3f} (p = {float(sps.norm.cdf(v)):.4f})" for name, v in values.items()}
        details.update({"entities": N, "periods": T, "lags": p, "Bartlett lags": k, "pooled rho": rho,
                        "residual variance": s2_v, "long-run residual variance": s2_0v})
        return TestResult(f"Kao panel cointegration ({', '.join(names)})", self.null, float(adf), None,
                          float(sps.norm.cdf(adf)), "normal", details=details)


def _lr_sd(u: np.ndarray, k: int) -> np.ndarray:
    """Bartlett long-run standard deviation over the last axis (u demeaned first)."""
    u = u - u.mean(axis=-1, keepdims=True)
    return np.sqrt(np.einsum("...t,...t->...", u, u) / u.shape[-1] + 2 * _bartlett(u, k))


def _westerlund_parts(Y: np.ndarray, X: np.ndarray, trend: str, p: int, q: int, k: int) -> dict[str, np.ndarray]:
    """Each row's error-correction regression (Westerlund 2007, equation 2), for rows of equal length T:
    dy_t on the deterministic terms, y_t-1, x_t-1, dy_t-1..t-p and dx_t+q..t-p. Returns alpha, its t-statistic, alpha(1)
    (the ratio of the long-run standard deviations of the residuals and of dy), the regression's standard error, and
    the panel statistics' parts, with y_t-1 and dy_t residualised on the other regressors."""
    R, T, m = X.shape
    dy, dx = np.diff(Y, axis=1), np.diff(X, axis=1)  # dy_t at index t-1
    start, end = p + 1, T - q  # usable t (levels index): needs dy_t-p (t-1-p >= 0) and dx_t+q (t+q <= T-1)
    n = end - start
    t_idx = np.arange(start, end)
    others = [X[:, t_idx - 1, :]]
    others += [dy[:, t_idx - 1 - j, None] for j in range(1, p + 1)]
    others += [dx[:, t_idx - 1 - j, :] for j in range(-q, p + 1)]
    if trend != "n":
        others.append(np.ones((R, n, 1)))
    if trend == "ct":
        others.append(np.broadcast_to(t_idx.astype(float)[None, :, None], (R, n, 1)))
    W = np.concatenate(others, axis=2)
    ylag, d = Y[:, t_idx - 1], dy[:, t_idx - 1]
    y_res, d_res = _ols_resid(ylag, W), _ols_resid(d, W)  # Frisch-Waugh: alpha from the residualised pair
    alpha = np.einsum("rt,rt->r", y_res, d_res) / np.einsum("rt,rt->r", y_res, y_res)
    e = d_res - alpha[:, None] * y_res
    dof = n - W.shape[2] - 1
    sigma = np.sqrt(np.einsum("rt,rt->r", e, e) / dof)
    t_alpha = alpha / (sigma / np.sqrt(np.einsum("rt,rt->r", y_res, y_res)))
    # alpha(1) = 1 - sum alpha_j as the ratio of the long-run standard deviations of the errors and of the part of dy
    # the regressors' differences do not explain (dy on dx_t+q..t-p and the deterministic terms)
    Wx = np.concatenate([dx[:, t_idx - 1 - j, :] for j in range(-q, p + 1)] + others[1 + p + (p + q + 1):], axis=2)
    alpha1 = _lr_sd(e, k) / _lr_sd(_ols_resid(d, Wx), k)
    return {"t": t_alpha, "a": T * alpha / alpha1,
            "num": np.einsum("rt,rt->r", y_res, d_res) / alpha1 / T, "den": np.einsum("rt,rt->r", y_res, y_res) / T**2,
            "S": sigma / alpha1}


def _westerlund_panel(name: str, means: dict[str, float]) -> float:
    return means["num"] / (means["S"] * np.sqrt(means["den"])) if name == "Pt" else means["num"] / means["den"]


@lru_cache(maxsize=128)
def _westerlund_null(T: int, m: int, trend: str, p: int, q: int, k: int, replications: int, seed: int):
    rng = make_rng(seed)
    chunk = max(200, 2_000_000 // (T * (m + 2) * (p + q + 3)))
    rows = []
    for start in range(0, replications, chunk):
        r = min(chunk, replications - start)
        Wk = np.cumsum(rng.normal(size=(r, T, m + 1)), axis=1)
        parts = _westerlund_parts(Wk[:, :, 0], Wk[:, :, 1:], trend, p, q, k)
        rows.append(np.column_stack([parts[key] for key in ("t", "a", "num", "den", "S")]))
    M = np.concatenate(rows)
    return M.mean(axis=0), np.cov(M, rowvar=False)


@register
@dataclass(frozen=True)
class Westerlund(Test):
    """Westerlund's error-correction panel cointegration tests: the group mean (Gt, Ga) and pooled (Pt, Pa) tests that
    the error-correction coefficient is zero.

    Each entity's error-correction regression explains the change in the first variable by its own lagged level, the
    others' lagged levels, lags of its change and leads and lags of the others' changes. The null is that the
    coefficient on the lagged level is zero in every entity: no error correction, so no cointegration. The group-mean
    statistics Gt and Ga let each entity have its own coefficient and reject when some entities error-correct; the
    pooled Pt and Pa reject when the panel as a whole does. All four are standardised to N(0, 1) and reject when large
    and negative; Gt is shown, the others are in the details. The regressors are assumed weakly exogenous. Westerlund's
    bootstrap for cross-sectionally dependent panels is not computed, so the p-values assume independent entities.
    """

    maturity: ClassVar[str] = "beta"
    variables: Annotated[str, Widget("", help="the series, comma-separated; the first is the one whose error correction is tested")]
    trend: Annotated[Literal["c", "ct", "n"], Widget("", help="each entity's terms in its error-correction regression: c constant, ct constant and trend, n none")] = "c"
    lags: Annotated[int, advanced(help="lagged differences of every variable in each entity's error-correction regression")] = 1
    leads: Annotated[int, advanced(help="leads of the regressors' differences, absorbing their correlation with the errors; 0 = none")] = 0
    bandwidth: Annotated[str | int, advanced(help="Bartlett lags for the long-run variances; auto = 4 (T / 100)^(2/9)")] = "auto"
    replications: Annotated[int, advanced(help="simulated entities for the moments under the null")] = 20_000
    seed: Annotated[int, advanced()] = 0

    name = "westerlund"
    topics = ("cointegration", "panel-data")
    references = (
        ("Westerlund, J. (2007). Testing for error correction in panel data. Oxford Bulletin of Economics and Statistics 69(6), 709-748. doi:10.1111/j.1468-0084.2007.00477.x"),
    )
    null = "no cointegration (no error correction in any entity)"
    applies_to = (Dataset,)

    def run(self, data: Dataset) -> TestResult:
        names = _variables(self.variables, "Westerlund's test")
        m = len(names) - 1
        entities = _entities(data, names, "Westerlund's test")
        p, q = int(self.lags), int(self.leads)
        reps, seed = int(self.replications), int(self.seed)
        parts, nulls = [], []
        for key, v in entities:
            T = len(v)
            k = _default_bandwidth(T) if str(self.bandwidth).strip().lower() in ("auto", "") else int(self.bandwidth)
            width = 1 + m + p + (p + q + 1) * m + (self.trend != "n") + (self.trend == "ct")
            if T - p - q - 1 - width < 5:
                raise SpecificationError(f"entity {key} has too few periods ({T}) for {m} regressors, {p} lags and {q} leads")
            parts.append({name: float(val[0]) for name, val in _westerlund_parts(v[None, :, 0], v[None, :, 1:], self.trend, p, q, k).items()})
            nulls.append(_westerlund_null(T, m, self.trend, p, q, k, reps, seed))
        N = len(entities)
        keys = ("t", "a", "num", "den", "S")
        mu = dict(zip(keys, np.mean([nl[0] for nl in nulls], axis=0)))
        C = np.mean([nl[1] for nl in nulls], axis=0)
        avg = {key: float(np.mean([e[key] for e in parts])) for key in keys}
        values = {"Gt": (avg["t"], mu["t"], C[0, 0]), "Ga": (avg["a"], mu["a"], C[1, 1])}
        for name in ("Pt", "Pa"):
            used = ("num", "S", "den") if name == "Pt" else ("num", "den")
            grad = []
            for key in used:
                h = 1e-6 * max(abs(mu[key]), 1e-8)
                up, dn = dict(mu), dict(mu)
                up[key] += h
                dn[key] -= h
                grad.append((_westerlund_panel(name, up) - _westerlund_panel(name, dn)) / (2 * h))
            idx = [keys.index(key) for key in used]
            g = np.array(grad)
            values[name] = (_westerlund_panel(name, avg), _westerlund_panel(name, mu), float(g @ C[np.ix_(idx, idx)] @ g))
        z = {name: float(np.sqrt(N) * (raw - m_) / np.sqrt(var)) for name, (raw, m_, var) in values.items()}
        details = {name: f"{z[name]:.3f} (p = {float(sps.norm.cdf(z[name])):.4f})" for name in ("Gt", "Ga", "Pt", "Pa")}
        details.update({"Gt unstandardised": avg["t"], "Ga unstandardised": avg["a"], "entities": N, "regressors": m,
                        "lags": p, "leads": q, "simulated entities for the moments": reps})
        return TestResult(f"Westerlund panel cointegration ({', '.join(names)})", self.null, z["Gt"], None,
                          float(sps.norm.cdf(z["Gt"])), "normal", details=details)
