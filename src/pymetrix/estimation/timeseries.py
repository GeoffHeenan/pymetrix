"""Error processes for single equations: ARMA errors and GARCH variance.

Both are part of the specification::

    Equation("y ~ x", errors=ARMA(ar=1, ma=1))
    Equation("r ~ 1", variance=GARCH(arch=1, garch=1, asym=1, dist="t"))

and are estimated by ``ML()``.

* ARMA errors: exact Gaussian likelihood from the Kalman filter, with the
  error process started at its stationary distribution. With consecutive lags
  (``ar=2``), stationarity and invertibility are enforced by the same
  transformation statsmodels uses; subset lags (``ar=[1, 4]``) are only
  checked.
* GARCH: conditional likelihood with the pre-sample variance set by
  exponential backcasting, as in the ``arch`` package.
* Both (``errors=ARMA(...)`` with ``variance=GARCH(...)``): conditional likelihood,
  as fGarch and rugarch: the first p periods are conditioned on, errors before the
  sample are zero, and the innovations of the ARMA filter have the GARCH variance
  (normal or t). Starting values are conditional least squares; the pre-sample
  variance is backcast from their innovations. With AR terms alone this is the
  model ``arch``'s AR mean estimates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import minimize
from scipy.special import gammaln

from pymetrix._lazy import signal
from pymetrix._lazy import stats as sps

from ..errors import SpecificationError
from ..forms import Widget
from ..plugins import register
from .design import LinearDesign, linear_design, provenance
from .fitted import FittedEquation
from .numdiff import ml_covariance
from .params import point

if TYPE_CHECKING:
    from ..data import Dataset
    from .equation import Equation

_LOG2PI = np.log(2 * np.pi)


def _lags(spec: int | Sequence[int], what: str) -> tuple[int, ...]:
    lags = tuple(range(1, spec + 1)) if isinstance(spec, int) else tuple(sorted(set(spec)))
    if any(lag < 1 for lag in lags):
        raise SpecificationError(f"{what} lags must be positive")
    return lags


@dataclass(frozen=True)
class ARMA:
    """ARMA errors: ``ar`` and ``ma`` are orders, or lists of lags such as ``[1, 4]``."""

    ar: Annotated[int | Sequence[int], Widget("text", help="AR order, or a list of lags such as [1, 4]")] = 0
    ma: Annotated[int | Sequence[int], Widget("text", help="MA order, or a list of lags")] = 0

    @property
    def ar_lags(self) -> tuple[int, ...]:
        return _lags(self.ar, "AR")

    @property
    def ma_lags(self) -> tuple[int, ...]:
        return _lags(self.ma, "MA")


Dist = Literal["normal", "t", "skewt", "ged"]


@dataclass(frozen=True)
class GARCH:
    """Conditional variance: ``arch`` lags of squared errors, ``garch`` lags of the variance,
    ``asym`` GJR terms (extra weight on negative errors).

    Normal and t errors are estimated natively (cross-checked against the arch
    package); skewed-t and GED errors, or ``backend="arch"``, use arch."""

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        import numpy as np
        import pandas as pd
        from pymetrix import GARCH, Dataset, Equation

        rng = np.random.default_rng(1)                           # simulated returns from a known GARCH(1, 1)
        r, h = np.zeros(1500), np.ones(1500)
        for t in range(1, 1500):                                 # omega 0.1, alpha 0.1, beta 0.85
            h[t] = 0.1 + 0.1 * r[t - 1] ** 2 + 0.85 * h[t - 1]
            r[t] = np.sqrt(h[t]) * rng.standard_normal()
        data = Dataset.from_frame(pd.DataFrame({"r": r}, index=pd.period_range("2020-01-01", periods=1500, freq="D", name="time")))
        fit = Equation("r ~ 1", variance=GARCH(1, 1)).fit(data)
        print(fit.summary())                                     # omega, alpha[1], beta[1] near the true values; their sum near 1: persistent volatility
        gjr = Equation("r ~ 1", variance=GARCH(1, 1, asym=1, dist="t")).fit(data)
        print(gjr.summary())                                     # gamma[1], the extra response to negative shocks, near 0: these shocks are symmetric
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("omega, alpha[i], beta[j]", ("the variance equation: a constant, the response to past squared shocks and the "
                                      "weight on past variance; const and others are the mean equation")),
        ("gamma[i], nu", ("gamma: the extra response to negative shocks (GJR); nu: the degrees of freedom of t "
                          "errors, small for fat tails")),
        ("std err", ("from the inverse Hessian; with method=ML(cov='white') the sandwich, robust to a wrong error "
                     "distribution")),
        ("fit.stats['persistence']", ("alpha + beta + gamma / 2: near 1, volatility shocks die out slowly; the fit "
                                      "keeps it at most 1")),
        ("fit.extra['volatility'], fit.extra['std_resid']", ("the conditional standard deviation and the standardised "
                                                             "residuals, on which the arch and ljung_box tests are "
                                                             "run")),
    )

    label: ClassVar[str] = "GARCH"
    topics: ClassVar[tuple[str, ...]] = ('garch',)
    references: ClassVar[tuple[str, ...]] = (
        ("Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. Econometrica 50(4), 987-1007. doi:10.2307/1912773"),
        "Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics 31(3), 307-327. doi:10.1016/0304-4076(86)90063-1",
        ("Glosten, L. R., Jagannathan, R. and Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. Journal of Finance 48(5), 1779-1801. doi:10.1111/j.1540-6261.1993.tb05128.x"),
    )

    arch: Annotated[int, Widget("number", min=0, max=10, help="lags of the squared shocks (ARCH terms)")] = 1
    garch: Annotated[int, Widget("number", min=0, max=10, help="lags of the conditional variance (GARCH terms)")] = 1
    asym: Annotated[int, Widget("number", min=0, max=10, help="GJR terms: extra response to negative shocks")] = 0
    dist: Dist = "normal"
    backend: Annotated[Literal["native", "arch"], Widget("choice", advanced=True,
                       help="normal or t errors: our implementation or the arch package")] = "native"

    @property
    def native(self) -> bool:
        return self.backend == "native" and self.dist in ("normal", "t")

    @property
    def libraries(self) -> tuple[str, ...]:
        """Libraries this configuration needs (skewed-t and GED errors use arch)."""
        return () if self.native else ("arch",)

    def fit(self, method: Any, eq: Equation, data: Dataset, design: LinearDesign) -> FittedEquation:
        if self.native:
            return _fit_garch(method, eq, data, design)
        from ..ext.arch import fit_volatility

        return fit_volatility(method, eq, data, design)

    def forecast(self, fit: FittedEquation, data: Dataset, span: tuple[Any, Any], dynamic: bool, alpha: float) -> Any:
        backend = (getattr(fit.provenance, "backend", None) or {}).get("name")
        if backend != "arch":  # estimated natively: the native recursion below (the record says which, also once saved)
            return NotImplemented
        from ..ext.arch import forecast

        return forecast(fit, data, span, dynamic, alpha)




def estimate(method: Any, eq: Equation, data: Dataset, span: Any, where: Any, weights: Any, missing: str) -> FittedEquation:
    if weights:
        raise NotImplementedError("weights with ARMA errors or GARCH variance")
    if data.kind != "time_series":
        raise SpecificationError("ARMA errors and GARCH variance need a single time series")
    d = linear_design(eq, data, span, where, None, missing, allow_empty=True)
    periods = d.index
    if len(pd.period_range(periods.min(), periods.max(), freq=periods.freq)) != len(periods):
        raise SpecificationError("ARMA and GARCH models need consecutive periods; the sample has gaps (missing values inside the span)")
    if eq.errors is not None and eq.variance is not None:
        return _fit_arma_garch(method, eq, data, d)
    if eq.errors is not None:
        return _fit_arma(method, eq, data, d)
    return eq.variance.fit(method, eq, data, d)


def _stats(n: int, k: int, loglike: float, **more: float) -> dict[str, float]:
    from . import criteria

    return {"nobs": float(n), "k": float(k), "loglike": float(loglike), "aic": criteria.aic(loglike, k), "bic": criteria.bic(loglike, k, n),
            "hq": criteria.hq(loglike, k, n), **more}


# ARMA errors


def constrain_stationary(x: np.ndarray) -> np.ndarray:
    """Map unconstrained values to the coefficients of a stationary AR polynomial
    (Monahan's transformation, as in statsmodels)."""
    r = x / np.sqrt(1 + x**2)
    n = len(r)
    y = np.zeros((n, n))
    for k in range(n):
        for i in range(k):
            y[k, i] = y[k - 1, i] + r[k] * y[k - 1, k - i - 1]
        y[k, k] = r[k]
    return -y[n - 1, :] if n else r


def _stationary(phi: np.ndarray) -> bool:
    if not len(phi) or not np.any(phi):
        return True
    return bool(np.all(np.abs(np.roots(np.r_[1.0, -phi])) < 1))


def arma_predict(u: np.ndarray, known: np.ndarray, phi: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kalman one-step predictions of the ARMA errors ``u`` and their variances (in units of sigma2).

    Where ``known`` is False the observation is skipped, so predictions run on
    as multi-step forecasts from the last known value.
    """
    p, q = len(phi), len(theta)
    r = max(p, q + 1)
    T = np.zeros((r, r))
    T[:p, 0] = phi
    T[:-1, 1:] += np.eye(r - 1)
    R = np.zeros(r)
    R[0] = 1.0
    R[1 : q + 1] = theta
    Q = np.outer(R, R)
    P = solve_discrete_lyapunov(T, Q)
    a = np.zeros(r)
    n = len(u)
    pred, F = np.empty(n), np.empty(n)
    for t in range(n):
        pred[t], F[t] = a[0], P[0, 0]
        if known[t]:
            K = (T @ P[:, 0]) / F[t]
            a = T @ a + K * (u[t] - a[0])
            P = T @ P @ T.T + Q - np.outer(K, K) * F[t]
        else:
            a = T @ a
            P = T @ P @ T.T + Q
    return pred, F


def arma_filter(u: np.ndarray, phi: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-step prediction errors ``v`` and their variances ``F`` (in units of sigma2)."""
    pred, F = arma_predict(u, np.ones(len(u), dtype=bool), phi, theta)
    return u - pred, F


def _fit_arma(method: Any, eq: Equation, data: Dataset, d: LinearDesign) -> FittedEquation:
    spec: ARMA = eq.errors  # type: ignore[assignment]
    ar_lags, ma_lags = spec.ar_lags, spec.ma_lags
    if not ar_lags and not ma_lags:
        raise SpecificationError("ARMA errors need at least one AR or MA lag")
    p, q = max(ar_lags, default=0), max(ma_lags, default=0)
    contiguous_ar, contiguous_ma = ar_lags == tuple(range(1, p + 1)), ma_lags == tuple(range(1, q + 1))
    y, X = d.y, d.X
    n, k = X.shape
    na, nm = len(ar_lags), len(ma_lags)

    def place(coefs: np.ndarray, lags: tuple[int, ...], size: int) -> np.ndarray:
        out = np.zeros(size)
        out[[lag - 1 for lag in lags]] = coefs
        return out

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, m = x[k : k + na], x[k + na :]
        ar = constrain_stationary(a) if contiguous_ar else a
        ma = -constrain_stationary(m) if contiguous_ma else m
        return x[:k], ar, ma

    def concentrated(x: np.ndarray) -> float:
        beta, ar, ma = unpack(x)
        phi, theta = place(ar, ar_lags, p), place(ma, ma_lags, q)
        if not (_stationary(phi) and _stationary(-theta)):
            return -1e10
        v, F = arma_filter(y - X @ beta, phi, theta)
        s2 = np.mean(v**2 / F)
        return -n / 2 * (_LOG2PI + np.log(s2) + 1) - 0.5 * np.sum(np.log(F))

    beta0 = np.linalg.lstsq(X, y, rcond=None)[0] if k else np.zeros(0)
    result = minimize(lambda x: -concentrated(x) / n, np.r_[beta0, np.zeros(na + nm)], method="BFGS", options={"gtol": 1e-9, "maxiter": 5000})
    beta, ar, ma = unpack(result.x)
    phi, theta = place(ar, ar_lags, p), place(ma, ma_lags, q)
    v, F = arma_filter(y - X @ beta, phi, theta)
    s2 = float(np.mean(v**2 / F))
    estimate_ = np.r_[beta, ar, ma, s2]

    def loglike(psi: np.ndarray) -> np.ndarray:
        b, a_, m_, sig2 = psi[:k], psi[k : k + na], psi[k + na : k + na + nm], psi[-1]
        vv, FF = arma_filter(y - X @ b, place(a_, ar_lags, p), place(m_, ma_lags, q))
        return -0.5 * (_LOG2PI + np.log(sig2 * FF) + vv**2 / (sig2 * FF))

    cov = ml_covariance(method.cov, loglike, estimate_)
    names = d.names + [f"ar[{lag}]" for lag in ar_lags] + [f"ma[{lag}]" for lag in ma_lags] + ["sigma2"]
    ll = float(np.sum(loglike(estimate_)))
    resid = pd.Series(v, index=d.index, name="resid")
    tss = float(np.sum((y - y.mean()) ** 2))
    return FittedEquation(
        equation=eq,
        params=point(names, estimate_, cov),
        method=method,
        resid=resid,
        fitted=pd.Series(y - v, index=d.index, name="fitted"),
        stats=_stats(n, len(names), ll, se_regression=float(np.sqrt(s2)), r2=float(1 - np.sum(v**2) / tss),
                     durbin_watson=float(np.sum(np.diff(v) ** 2) / np.sum(v**2))),
        provenance=provenance(eq, method, data, d.index, d.dropped),
        design=pd.DataFrame(X, index=d.index, columns=d.names),
        extra={
            "converged": _converged(result),
            "model_df": na + nm,
            "std_resid": pd.Series(v / np.sqrt(s2 * F), index=d.index, name="std_resid"),
            "structural_resid": pd.Series(y - X @ beta, index=d.index, name="u"),
        },
    )


# GARCH


def backcast(e: np.ndarray) -> float:
    """Pre-sample variance: exponentially weighted mean of early squared errors (as in ``arch``)."""
    tau = min(75, len(e))
    w = 0.94 ** np.arange(tau)
    return float(np.sum(e[:tau] ** 2 * w / w.sum()))


def garch_variance(e: np.ndarray, omega: float, alpha: np.ndarray, gamma: np.ndarray, beta: np.ndarray, bc: float) -> np.ndarray:
    n = len(e)
    m = max(len(alpha), len(gamma), 1)
    E2 = np.r_[np.full(m, bc), e**2]
    N = np.r_[np.full(m, 0.5 * bc), e**2 * (e < 0)]
    c = np.full(n, omega)
    for i, a in enumerate(alpha, start=1):
        c += a * E2[m - i : m - i + n]
    for i, g in enumerate(gamma, start=1):
        c += g * N[m - i : m - i + n]
    if not len(beta):
        return c
    den = np.r_[1.0, -beta]
    return signal.lfilter([1.0], den, c, zi=signal.lfiltic([1.0], den, np.full(len(beta), bc)))[0]


def _garch_loglike(e: np.ndarray, s2: np.ndarray, nu: float | None) -> np.ndarray:
    if nu is None:
        return -0.5 * (_LOG2PI + np.log(s2) + e**2 / s2)
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    return c - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log1p(e**2 / (s2 * (nu - 2)))


def _fit_garch(method: Any, eq: Equation, data: Dataset, d: LinearDesign) -> FittedEquation:
    spec: GARCH = eq.variance  # type: ignore[assignment]
    if spec.arch < 1 and spec.asym < 1:
        raise SpecificationError("GARCH needs at least one arch or asym term")
    y, X = d.y, d.X
    n, k = X.shape
    qa, qg, qb = spec.arch, spec.asym, spec.garch
    t_dist = spec.dist == "t"
    beta0 = np.linalg.lstsq(X, y, rcond=None)[0] if k else np.zeros(0)
    e0 = y - X @ beta0
    v0 = float(np.var(e0))
    bc = backcast(e0)

    def unpack(x: np.ndarray):
        i = k
        omega = x[i]
        alpha = x[i + 1 : i + 1 + qa]
        gamma = x[i + 1 + qa : i + 1 + qa + qg]
        beta = x[i + 1 + qa + qg : i + 1 + qa + qg + qb]
        nu = x[-1] if t_dist else None
        return x[:k], omega, alpha, gamma, beta, nu

    def loglike(x: np.ndarray) -> np.ndarray:
        b, omega, alpha, gamma, beta, nu = unpack(x)
        e = y - X @ b
        s2 = garch_variance(e, omega, alpha, gamma, beta, bc)
        if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
            return np.full(n, -1e10)
        return _garch_loglike(e, s2, nu)

    persistence0 = 0.9 if qb else 0.3
    x0 = np.r_[
        beta0,
        v0 * (1 - persistence0),
        np.full(qa, (0.1 if qb else 0.3) / max(qa, 1)),
        np.zeros(qg),
        np.full(qb, 0.8 / max(qb, 1)),
        [8.0] if t_dist else [],
    ]
    bounds = ([(None, None)] * k + [(1e-12 * v0, 10 * v0)] + [(0.0, 1.0)] * qa + [(-1.0, 2.0)] * qg
              + [(0.0, 1.0)] * qb + ([(2.05, 500.0)] if t_dist else []))

    def persistence(x: np.ndarray) -> float:
        _, _, alpha, gamma, beta, _ = unpack(x)
        return float(np.sum(alpha) + 0.5 * np.sum(gamma) + np.sum(beta))

    constraints = [{"type": "ineq", "fun": lambda x: 1.0 - persistence(x)}]
    for i in range(min(qa, qg)):
        constraints.append({"type": "ineq", "fun": lambda x, i=i: unpack(x)[2][i] + unpack(x)[3][i]})
    result = minimize(lambda x: -np.sum(loglike(x)) / n, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-12, "maxiter": 2000})
    if not result.success:
        raise RuntimeError(f"GARCH estimation did not converge: {result.message}")
    x = result.x
    cov = ml_covariance(method.cov, loglike, x)
    b, omega, alpha, gamma, beta, nu = unpack(x)
    e = y - X @ b
    s2 = garch_variance(e, omega, alpha, gamma, beta, bc)
    names = (d.names + ["omega"] + [f"alpha[{i}]" for i in range(1, qa + 1)] + [f"gamma[{i}]" for i in range(1, qg + 1)]
             + [f"beta[{i}]" for i in range(1, qb + 1)] + (["nu"] if t_dist else []))
    ll = float(np.sum(loglike(x)))
    return FittedEquation(
        equation=eq,
        params=point(names, x, cov),
        method=method,
        resid=pd.Series(e, index=d.index, name="resid"),
        fitted=pd.Series(y - e, index=d.index, name="fitted"),
        stats=_stats(n, len(names), ll, persistence=persistence(x)),
        provenance=provenance(eq, method, data, d.index, d.dropped),
        design=pd.DataFrame(X, index=d.index, columns=d.names),
        extra={
            "converged": _converged(result),
            "volatility": pd.Series(np.sqrt(s2), index=d.index, name="volatility"),
            "std_resid": pd.Series(e / np.sqrt(s2), index=d.index, name="std_resid"),
            "backcast": bc,
        },
    )


# ARMA errors with GARCH variance


def _arma_polynomials(spec: ARMA, ar: np.ndarray, ma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phi, theta = np.zeros(max(spec.ar_lags, default=0)), np.zeros(max(spec.ma_lags, default=0))
    phi[[lag - 1 for lag in spec.ar_lags]] = ar
    theta[[lag - 1 for lag in spec.ma_lags]] = ma
    return phi, theta


def arma_innovations(u: np.ndarray, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """The innovations of ARMA errors ``u``, conditional on the first p values, with errors before them zero:
    ``e(t) = u(t) - sum phi(i) u(t-i) - sum theta(j) e(t-j)``, for t = p, ..., n - 1."""
    p = len(phi)
    w = u[p:] - sum(phi[i] * u[p - 1 - i : len(u) - 1 - i] for i in range(p)) if p else u.copy()
    return signal.lfilter([1.0], np.r_[1.0, theta], w) if len(theta) else np.asarray(w, dtype=float)


def _fit_arma_garch(method: Any, eq: Equation, data: Dataset, d: LinearDesign) -> FittedEquation:
    from scipy.optimize import least_squares

    arma: ARMA = eq.errors  # type: ignore[assignment]
    spec: GARCH = eq.variance  # type: ignore[assignment]
    if not spec.native:
        raise SpecificationError("ARMA errors with GARCH variance are estimated here with normal or t errors; "
                                 "skewed-t and GED errors, and the arch backend, are for GARCH alone")
    if spec.arch < 1 and spec.asym < 1:
        raise SpecificationError("GARCH needs at least one arch or asym term")
    ar_lags, ma_lags = arma.ar_lags, arma.ma_lags
    if not ar_lags and not ma_lags:
        raise SpecificationError("ARMA errors need at least one AR or MA lag")
    y, X = d.y, d.X
    n, k = X.shape
    na, nm = len(ar_lags), len(ma_lags)
    p = max(ar_lags, default=0)
    if n - p < 30:
        raise SpecificationError(f"{n - p} observations after the first {p}: too few for ARMA errors with GARCH variance")
    qa, qg, qb = spec.arch, spec.asym, spec.garch
    t_dist = spec.dist == "t"

    def innovations(z: np.ndarray) -> np.ndarray:
        phi, theta = _arma_polynomials(arma, z[k : k + na], z[k + na : k + na + nm])
        return arma_innovations(y - X @ z[:k], phi, theta)

    beta0 = np.linalg.lstsq(X, y, rcond=None)[0] if k else np.zeros(0)
    css = least_squares(innovations, np.r_[beta0, np.zeros(na + nm)], method="lm", xtol=1e-12, ftol=1e-12)
    e0 = innovations(css.x)
    bc = backcast(e0)
    v0 = float(np.mean(e0 ** 2))
    m = k + na + nm

    def unpack(x: np.ndarray):
        i = m
        return (x[:m], x[i], x[i + 1 : i + 1 + qa], x[i + 1 + qa : i + 1 + qa + qg],
                x[i + 1 + qa + qg : i + 1 + qa + qg + qb], x[-1] if t_dist else None)

    def admissible(z: np.ndarray) -> bool:
        phi, theta = _arma_polynomials(arma, z[k : k + na], z[k + na : k + na + nm])
        return _stationary(phi) and _stationary(-theta)

    def loglike(x: np.ndarray) -> np.ndarray:
        z, omega, alpha, gamma, beta, nu = unpack(x)
        if not admissible(z):
            return np.full(n - p, -1e10)
        e = innovations(z)
        s2 = garch_variance(e, omega, alpha, gamma, beta, bc)
        if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
            return np.full(n - p, -1e10)
        return _garch_loglike(e, s2, nu)

    persistence0 = 0.9 if qb else 0.3
    x0 = np.r_[css.x if admissible(css.x) else np.r_[beta0, np.zeros(na + nm)], v0 * (1 - persistence0),
               np.full(qa, (0.1 if qb else 0.3) / max(qa, 1)), np.zeros(qg), np.full(qb, 0.8 / max(qb, 1)), [8.0] if t_dist else []]
    bounds = ([(None, None)] * m + [(1e-12 * v0, 10 * v0)] + [(0.0, 1.0)] * qa + [(-1.0, 2.0)] * qg
              + [(0.0, 1.0)] * qb + ([(2.05, 500.0)] if t_dist else []))

    def persistence(x: np.ndarray) -> float:
        _, _, alpha, gamma, beta, _ = unpack(x)
        return float(np.sum(alpha) + 0.5 * np.sum(gamma) + np.sum(beta))

    constraints = [{"type": "ineq", "fun": lambda x: 1.0 - persistence(x)}]
    for i in range(min(qa, qg)):
        constraints.append({"type": "ineq", "fun": lambda x, i=i: unpack(x)[2][i] + unpack(x)[3][i]})
    result = minimize(lambda x: -np.sum(loglike(x)) / (n - p), x0, method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-12, "maxiter": 3000})
    if not result.success:
        raise RuntimeError(f"ARMA-GARCH estimation did not converge: {result.message}")
    x = result.x
    cov = ml_covariance(method.cov, loglike, x)
    z, omega, alpha, gamma, beta, nu = unpack(x)
    e = innovations(z)
    s2 = garch_variance(e, omega, alpha, gamma, beta, bc)
    names = (d.names + [f"ar[{lag}]" for lag in ar_lags] + [f"ma[{lag}]" for lag in ma_lags] + ["omega"]
             + [f"alpha[{i}]" for i in range(1, qa + 1)] + [f"gamma[{i}]" for i in range(1, qg + 1)]
             + [f"beta[{i}]" for i in range(1, qb + 1)] + (["nu"] if t_dist else []))
    ll = float(np.sum(loglike(x)))
    index = d.index[p:]
    u = y - X @ z[:k]
    tss = float(np.sum((y[p:] - y[p:].mean()) ** 2))
    return FittedEquation(
        equation=eq,
        params=point(names, x, cov),
        method=method,
        resid=pd.Series(e, index=index, name="resid"),
        fitted=pd.Series(y[p:] - e, index=index, name="fitted"),
        stats=_stats(n - p, len(names), ll, persistence=persistence(x), r2=float(1 - np.sum(e ** 2) / tss) if tss else float("nan")),
        provenance=provenance(eq, method, data, index, d.dropped),
        design=pd.DataFrame(X, index=d.index, columns=d.names),
        extra={
            "converged": _converged(result),
            "model_df": na + nm,
            "volatility": pd.Series(np.sqrt(s2), index=index, name="volatility"),
            "std_resid": pd.Series(e / np.sqrt(s2), index=index, name="std_resid"),
            "structural_resid": pd.Series(u, index=d.index, name="u"),
            "backcast": bc,
            "conditioning": p,  # the first p periods are conditioned on
        },
    )


def _forecast_arma_garch(fit: FittedEquation, u: np.ndarray, known: np.ndarray, values: dict[str, float], dynamic: bool,
                         own: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Forecasts of ARMA errors with GARCH variance: the predicted errors and their variances, from the first
    estimation period (the first p periods conditioned on, as in estimation). ``own`` are the coefficients on lags
    of the left-hand side among the regressors, whose dynamics the dynamic variances include."""
    arma: ARMA = fit.equation.errors  # type: ignore[assignment]
    phi, theta = _arma_polynomials(arma, np.array([values[f"ar[{lag}]"] for lag in arma.ar_lags]),
                                   np.array([values[f"ma[{lag}]"] for lag in arma.ma_lags]))
    p, q, n = len(phi), len(theta), len(u)
    u = u.copy()
    e, pred = np.zeros(n), np.zeros(n)
    for t in range(p, n):
        pred[t] = sum(phi[i] * u[t - 1 - i] for i in range(p)) + sum(theta[j] * e[t - 1 - j] for j in range(q) if t - 1 - j >= p)
        if known[t]:
            e[t] = u[t] - pred[t]
        else:
            u[t] = pred[t]
    h = np.full(n, np.nan)
    h[p:] = _garch_path(fit, e[p:], known[p:], values)
    var = h.copy()
    if dynamic and not known[p:].all():
        psi = _psi_weights(_ar_product(phi, own), theta, n)
        origin = p + int(np.argmin(known[p:]))  # the first unknown period: steps are counted from it
        for t in range(origin, n):
            j = np.arange(t - origin + 1)
            var[t] = float(np.sum(psi[j] ** 2 * h[t - j]))
    return pred, var


def _psi_weights(phi: np.ndarray, theta: np.ndarray, n: int) -> np.ndarray:
    """The MA(infinity) weights of an ARMA process: psi(0) = 1."""
    psi = np.zeros(n)
    psi[0] = 1.0
    for j in range(1, n):
        psi[j] = (theta[j - 1] if j - 1 < len(theta) else 0.0) + sum(phi[i] * psi[j - 1 - i] for i in range(min(len(phi), j)))
    return psi


def _ar_product(phi: np.ndarray, own: np.ndarray | None) -> np.ndarray:
    """The AR coefficients of (1 - phi(L)) (1 - own(L)): ARMA errors under an equation with lags of its left-hand side."""
    if own is None or not len(own):
        return phi
    return -np.convolve(np.r_[1.0, -phi], np.r_[1.0, -own])[1:]


def _variance_ahead(psi: np.ndarray, h: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Dynamic forecast variances, sum of psi(j)^2 h(t - j) over the steps from the first unknown period."""
    var = h.copy()
    if not known.all():
        origin = int(np.argmin(known))
        for t in range(origin, len(h)):
            j = np.arange(t - origin + 1)
            var[t] = float(np.sum(psi[j] ** 2 * h[t - j]))
    return var


def _lhs_lags(eq: Any, dependent: str) -> dict[str, int] | None:
    """The regressors that are lags of the left-hand side, {term label: lag}; None when the dependent variable
    enters a regressor in any other way."""
    found = {}
    for term in eq.terms:
        if dependent not in term.expr.variables():
            continue
        lags = [n.lag for n in term.expr.refs()]
        k = min(n.lag for n in eq.lhs.refs()) - min(lags)
        if k < 1 or term.expr != eq.lhs.shift(-k):
            return None
        found[term.label] = k
    return found


def _forecast_through_model(fit: FittedEquation, data: Dataset, span: tuple[Any, Any], index: pd.PeriodIndex) -> pd.DataFrame:
    """A dynamic forecast solved as a one-equation model, for regressors that are functions of the dependent
    variable other than lags of the left-hand side; without standard errors (stochastic simulation of the model
    gives its distribution)."""
    from ..expr import evaluate
    from ..models import Model

    eq = fit.equation
    dependent = eq.require_dependent()
    solved = Model([fit]).solve(data, span).solved().frame[dependent]
    frame = data.frame.copy()
    frame.loc[solved.index, dependent] = solved.to_numpy()
    full = type(data)(frame, data.axes, data.variables)
    lhs = pd.Series(np.broadcast_to(np.asarray(evaluate(eq.lhs, full.lagged), dtype=float), (len(frame),)), index=frame.index)
    actual = pd.Series(np.broadcast_to(np.asarray(evaluate(eq.lhs, data.lagged), dtype=float), (len(frame),)), index=frame.index)
    point = lhs.reindex(index).to_numpy()
    nan = np.full(len(index), np.nan)
    return pd.DataFrame({"forecast": point, "se": nan, "lower": nan, "upper": nan, "actual": actual.reindex(index).to_numpy()}, index=index)


# forecasting


def forecast(fit: FittedEquation, data: Dataset, span: tuple[Any, Any], dynamic: bool, alpha: float) -> Dataset:
    """Forecasts of the left-hand side for an equation with ARMA errors or GARCH variance.

    The recursions start at the first estimation period, so forecasts condition
    on all data before the span (static: one step ahead from actual data;
    dynamic: multi-step from the start of the span). Standard errors reflect
    shock uncertainty only, not parameter uncertainty.
    """
    from ..data import Dataset
    from ..expr import evaluate

    eq = fit.equation
    if data.kind != "time_series":
        raise SpecificationError("forecasting needs a single time series")
    hook = getattr(eq.variance, "forecast", None)
    if hook is not None:
        out = hook(fit, data, span, dynamic, alpha)
        if out is not NotImplemented:
            return out
    freq = data.frame.index.freq
    from ..data.sample import as_span
    span = as_span(span)  # '1990Q1:2019Q4' or a pair
    start, end = pd.Period(str(span[0]), freq=freq), pd.Period(str(span[1]), freq=freq)
    first = pd.Period(fit.provenance.sample[0], freq=freq)  # type: ignore[union-attr]
    if start <= first:
        raise SpecificationError(f"the forecast span must start after the first estimation period, {first}")
    dependent = eq.require_dependent()
    own_lags = _lhs_lags(eq, dependent) if dynamic else {}
    periods = pd.period_range(first, end, freq=freq, name=data.time_axis.name)  # type: ignore[union-attr]
    if own_lags is None:
        frame = _forecast_through_model(fit, data, span, periods)
        note = " Other regressors depend on the dependent variable than lags of the left-hand side: no standard errors."
        return _forecast_dataset(frame[np.asarray(periods >= start)], data, eq, dynamic, note)
    # evaluated with the data's earlier periods, so lags at the first estimation period are known
    lo = min(data.frame.index[0], first)
    ds = Dataset(data.frame.reindex(pd.period_range(lo, end, freq=freq, name=data.time_axis.name)), data.axes, data.variables)
    values = fit.params.as_dict()
    coefs = {k: values[k] for k in fit.design.columns}  # type: ignore[union-attr]

    def path(expr: Any, params: Any = None, whole: bool = False) -> np.ndarray:
        out = np.broadcast_to(np.asarray(evaluate(expr, ds.lagged, params), dtype=float), (len(ds.frame),)).copy()
        return out if whole else out[-len(periods):]

    y = path(eq.lhs)
    in_span = np.asarray(periods >= start)
    # regressors that are lags of the left-hand side are carried forward from its forecasts; the rest, mu, are given
    own = np.zeros(max(own_lags.values(), default=0))
    for label, k in own_lags.items():
        own[k - 1] += values[label]
    mu = path(eq.linear_rhs(), {**coefs, **{label: 0.0 for label in own_lags}})
    if not np.all(np.isfinite(mu[in_span])):
        raise SpecificationError("regressors are missing for some forecast periods")
    lagged = sum((own[k - 1] * path(eq.lhs.shift(-k)) for k in range(1, len(own) + 1)), np.zeros(len(periods)))
    known = np.isfinite(y) & (~in_span if dynamic else True)
    u = np.where(known, y - mu - lagged, 0.0)

    if eq.errors is not None and eq.variance is not None:
        conditioning = int(fit.extra.get("conditioning", 0))
        if start < first + conditioning:
            raise SpecificationError(f"the forecast span must start after the first {conditioning} estimation periods, which are conditioned on")
        pred, var = _forecast_arma_garch(fit, u, known, values, dynamic, own)
        nu = values.get("nu")
        quantile = sps.t.ppf(1 - alpha / 2, nu) * np.sqrt((nu - 2) / nu) if nu else sps.norm.ppf(1 - alpha / 2)
    elif eq.errors is not None:
        spec: ARMA = eq.errors
        phi = np.zeros(max(spec.ar_lags, default=0))
        theta = np.zeros(max(spec.ma_lags, default=0))
        for lag in spec.ar_lags:
            phi[lag - 1] = values[f"ar[{lag}]"]
        for lag in spec.ma_lags:
            theta[lag - 1] = values[f"ma[{lag}]"]
        pred, F = arma_predict(u, known, phi, theta)
        var = values["sigma2"] * F
        if own_lags:  # the forecast errors of the left-hand side, through its own lags too
            var = _variance_ahead(_psi_weights(_ar_product(phi, own), theta, len(u)), np.full(len(u), values["sigma2"]), known)
        quantile = sps.norm.ppf(1 - alpha / 2)
    else:
        var = _garch_path(fit, u, known, values)
        pred = np.zeros(len(u))
        if own_lags:
            var = _variance_ahead(_psi_weights(own, np.zeros(0), len(u)), var, known)
        nu = values.get("nu")
        quantile = sps.t.ppf(1 - alpha / 2, nu) * np.sqrt((nu - 2) / nu) if nu else sps.norm.ppf(1 - alpha / 2)
    mean = mu + lagged + pred
    if own_lags and dynamic:  # each forecast of the left-hand side feeds the next period's lags
        z = path(eq.lhs, whole=True)
        offset = len(z) - len(periods)
        for t in np.flatnonzero(in_span):
            mean[t] = mu[t] + sum(own[k - 1] * z[offset + t - k] for k in range(1, len(own) + 1)) + pred[t]
            z[offset + t] = mean[t]

    se = np.sqrt(var)
    frame = pd.DataFrame(
        {"forecast": mean, "se": se, "lower": mean - quantile * se, "upper": mean + quantile * se,
         "actual": np.where(np.isfinite(y), y, np.nan)},
        index=periods,
    )
    if eq.variance is not None:
        frame.insert(2, "variance", var)
    return _forecast_dataset(frame[in_span], data, eq, dynamic)


def _forecast_dataset(frame: pd.DataFrame, data: Dataset, eq: Any, dynamic: bool, note: str = "") -> Dataset:
    from ..data import Dataset
    from ..metadata import DatasetMeta

    kind = "Dynamic" if dynamic else "Static (one-step-ahead)"
    meta = DatasetMeta(title=f"{kind} forecast: {eq.text}",
                       description="Intervals reflect shock uncertainty only, not parameter uncertainty." + note)
    return Dataset(frame, data.axes, meta=meta)


def _garch_path(fit: FittedEquation, e: np.ndarray, known: np.ndarray, values: dict[str, float]) -> np.ndarray:
    """Conditional variances; where an error is unknown its square is replaced by its expectation."""
    spec: GARCH = fit.equation.variance
    omega = values["omega"]
    alpha = [values[f"alpha[{i}]"] for i in range(1, spec.arch + 1)]
    gamma = [values[f"gamma[{i}]"] for i in range(1, spec.asym + 1)]
    beta = [values[f"beta[{i}]"] for i in range(1, spec.garch + 1)]
    bc = fit.extra["backcast"]
    m = max(len(alpha), len(gamma), len(beta), 1)
    e2, neg, s2 = [bc] * m, [0.5 * bc] * m, [bc] * m
    out = np.empty(len(e))
    for t in range(len(e)):
        value = omega + sum(a * e2[-i] for i, a in enumerate(alpha, 1)) + sum(g * neg[-i] for i, g in enumerate(gamma, 1)) + sum(b * s2[-j] for j, b in enumerate(beta, 1))
        out[t] = value
        if known[t]:
            e2.append(e[t] ** 2)
            neg.append(e[t] ** 2 if e[t] < 0 else 0.0)
        else:  # symmetric shocks: E[e^2] = sigma2, E[e^2 1(e<0)] = sigma2 / 2
            e2.append(value)
            neg.append(0.5 * value)
        s2.append(value)
    return out


register("variances", GARCH)
register("variances", GARCH, name="gjr_garch", label="GJR-GARCH", preset={"asym": 1})


def _converged(result: Any, tol: float = 1e-5) -> bool:
    """Whether a SciPy optimisation reached an optimum: it says so, or it stopped for lack of precision (BFGS's
    "precision loss" at a strict tolerance) with the gradient essentially zero."""
    if result.success:
        return True
    jac = getattr(result, "jac", None)
    return jac is not None and bool(np.all(np.isfinite(jac))) and float(np.max(np.abs(jac))) < tol
