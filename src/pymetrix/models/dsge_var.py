"""DSGE-VAR(lambda): a VAR whose prior comes from a linearised model (Del Negro and Schorfheide 2004; Del Negro,
Schorfheide, Smets and Wouters 2007).

    dv = model.dsge_var(data, measurement, lags=4, lambda_=1.0, estimate=[...], priors={...})   # lambda fixed
    dv = model.dsge_var(data, measurement, lags=4, lambda_="grid")       # lambda-hat: the best marginal data density on a grid
    dv = model.dsge_var(data, measurement, lags=4, lambda_=Gamma(1, 0.5))  # lambda estimated with theta, under its prior
    dv.summary(); dv.lambdas(); dv.lambda_plot()     # the mode of theta; the marginal data density over the lambda grid
    dv = dv.mcmc(draws=5000, seed=1)                 # the posterior of theta by random-walk Metropolis-Hastings
    dv.irf(20); dv.fevd(20)                          # responses identified by the model's rotation, with posterior bands
    dv.forecast(("2020Q1", "2021Q4"))                # forecasts from the VAR's predictive distribution
    dv.var_draws(500)                                # draws of the VAR's coefficients and covariance given theta draws

The VAR(p) in the observables with a constant, ``y(t)' = x(t)' Phi + u(t)', u(t) ~ N(0, Sigma)``, ``x(t) = (1, y(t-1)',
..., y(t-p)')'``, gets a prior from the model at its parameters theta: its population moments, uncentred,
``G_yy = E[y y']``, ``G_xy = E[x y']`` and ``G_xx = E[x x']`` (from the first-order solution, the measurement equations
and their errors, ``bayes.moments``' autocovariances with the means), weighted as ``lambda T`` observations:

    Sigma | theta ~ IW(lambda T Sigma*(theta), lambda T - k),   Phi | Sigma, theta ~ N(Phi*(theta), Sigma (x) (lambda T G_xx)^-1)

with ``Phi* = G_xx^-1 G_xy`` and ``Sigma* = G_yy - G_yx G_xx^-1 G_xy``, the model's own VAR(p) approximation (the
population projection), and ``k = 1 + n p``. The prior is proper when ``lambda T >= k + n``: lambda is at least
``(k + n) / T``. The posterior given theta is normal-inverse-Wishart, ``Phi~ = (lambda T G_xx + X'X)^-1 (lambda T G_xy + X'Y)``,
``Sigma | Y ~ IW((1 + lambda) T Sigma~, (1 + lambda) T - k)``, and the likelihood of theta, ``p(Y | theta, lambda)`` with the
VAR's coefficients integrated out, is in closed form (DS 2004, eq. 21): the ratio of the posterior's to the prior's
normalising constants. With the parameters' priors it gives theta's posterior, whose mode and Metropolis-Hastings draws
are found as ``Model.estimate`` finds them (``pymetrix.models.estimation``, ``pymetrix.models.bayes``). The first ``p``
observations of the sample are the VAR's initial lags.

``lambda_``: a number (fixed); ``"grid"``, each value of ``lambdas`` estimated in turn and the one with the highest marginal
data density kept (``lambda-hat``; the density by the Laplace approximation at each mode, or the highest likelihood
without priors); or a prior (``Gamma(1, 0.5)``, ...), lambda then estimated with theta as ``lambda``. Large lambda
holds the VAR to the model's restrictions; small lambda leaves it close to the unrestricted VAR (least squares).

Impulse responses use the model's rotation (DS 2004, section 4.3): the model's impact of its shocks on the
observables, ``A0(theta)``, is factored by QR into ``Sigma*_tr(theta) Omega(theta)``, lower triangular times orthonormal,
and the VAR's impact is ``chol(Sigma) Omega(theta)``: one response per draw of (theta, Phi, Sigma). It needs as many
shocks as observables.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg, special

from .._api import api
from ..errors import DataError, SpecificationError
from ..output import Table
from .._random import generator as make_rng
from .estimation import _PENALTY, FittedModel, _Likelihood, _optimise, _scale, _setup
from .solvers import ModelError

__all__ = ["FittedDSGEVAR", "LAMBDA", "dsge_var", "log_marginal_likelihood", "population_moments", "var_approximation",
           "var_posterior"]

LAMBDA = "lambda"  # the name of the prior weight when it is estimated
GRID = (0.25, 0.33, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0)  # lambda-hat's default grid (as DSSW 2007's), those above the minimum


# --- the algebra --------------------------------------------------------------------------------------------------

def population_moments(system: Any, lags: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The uncentred population moments of a VAR(``lags``) with a constant in the observables of a state-space form
    (``y = d + Z a + e``, ``a(t) = T a(t-1) + R eta(t)``): ``G_yy = E[y y']`` (n x n), ``G_xy = E[x y']`` (k x n) and
    ``G_xx = E[x x']`` (k x k), with ``x = (1, y(t-1)', ..., y(t-p)')'``. Measurement errors add to the variance only."""
    T, R, Q, Z, H = system.T, system.R, system.Q, system.Z, system.H
    if T.size and np.max(np.abs(np.linalg.eigvals(T))) >= 1 - 1e-8:
        raise ModelError("the model has a unit root: its observables have no population moments for a DSGE-VAR prior "
                         "(observe growth rates or deviations from trend)")
    mu = np.asarray(system.d, dtype=float).ravel()
    n, p = len(mu), int(lags)
    S = linalg.solve_discrete_lyapunov(T, R @ Q @ R.T)
    S = (S + S.T) / 2
    gamma = []  # gamma[h] = E[(y(t) - mu)(y(t-h) - mu)']
    TS = S
    for h in range(p + 1):
        gamma.append(Z @ TS @ Z.T + (H if h == 0 else 0.0))
        TS = T @ TS
    mm = np.outer(mu, mu)
    G_yy = gamma[0] + mm
    k = 1 + n * p
    G_xy = np.empty((k, n))
    G_xy[0] = mu
    G_xx = np.empty((k, k))
    G_xx[0, 0] = 1.0
    for i in range(1, p + 1):
        rows = slice(1 + (i - 1) * n, 1 + i * n)
        G_xy[rows] = gamma[i].T + mm  # E[y(t-i) y(t)']
        G_xx[rows, 0] = mu
        G_xx[0, rows] = mu
        for j in range(1, p + 1):
            cols = slice(1 + (j - 1) * n, 1 + j * n)
            G_xx[rows, cols] = (gamma[j - i] if j >= i else gamma[i - j].T) + mm  # E[y(t-i) y(t-j)']
    return G_yy, G_xy, (G_xx + G_xx.T) / 2


def var_approximation(G_yy: np.ndarray, G_xy: np.ndarray, G_xx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The model's VAR approximation, the population projection: ``Phi* = G_xx^-1 G_xy`` and ``Sigma* = G_yy - G_yx Phi*``."""
    phi = linalg.solve(G_xx, G_xy, assume_a="pos")
    sigma = G_yy - G_xy.T @ phi
    return phi, (sigma + sigma.T) / 2


def _lag_design(Y_all: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """``Y`` (T x n) and ``X`` (T x k: a constant and ``lags`` lags), the first ``lags`` rows the initial lags."""
    p = int(lags)
    T = len(Y_all) - p
    X = np.column_stack([np.ones(T), *[Y_all[p - i:p - i + T] for i in range(1, p + 1)]])
    return Y_all[p:], X


@dataclass(frozen=True, eq=False)
class _NIW:
    """The DSGE-VAR posterior of (Phi, Sigma) given theta and lambda, with the prior it came from."""

    phi_star: np.ndarray
    sigma_star: np.ndarray
    phi: np.ndarray  # posterior mean of Phi (k x n)
    sigma: np.ndarray  # Sigma~: the posterior scale matrix divided by (1 + lambda) T
    precision: np.ndarray  # lambda T G_xx + X'X: Phi | Sigma ~ N(phi, Sigma (x) precision^-1)
    df: float  # (1 + lambda) T - k
    scale: np.ndarray  # (1 + lambda) T Sigma~: Sigma | Y ~ IW(scale, df)
    logml: float  # log p(Y | theta, lambda)


def var_posterior(G_yy: np.ndarray, G_xy: np.ndarray, G_xx: np.ndarray, Y: np.ndarray, X: np.ndarray, lam: float) -> _NIW:
    """The posterior of the VAR's coefficients and covariance given the model's moments and ``lambda``, and the
    marginal likelihood ``p(Y | theta, lambda)`` (DS 2004, eq. 21)."""
    T, n = Y.shape
    k = X.shape[1]
    lam = float(lam)
    if not lam * T >= k + n - 1e-9:
        raise SpecificationError(f"lambda must be at least (k + n) / T = {(k + n) / T:.4g} for a proper prior ({k} regressors, "
                                 f"{n} observables, {T} observations), not {lam:g}")
    phi_star, sigma_star = var_approximation(G_yy, G_xy, G_xx)
    lT = lam * T
    prior_xx, prior_xy, prior_yy = lT * G_xx, lT * G_xy, lT * G_yy
    A = prior_xx + X.T @ X
    cA = linalg.cho_factor((A + A.T) / 2)
    phi = linalg.cho_solve(cA, prior_xy + X.T @ Y)
    S = prior_yy + Y.T @ Y - (prior_xy + X.T @ Y).T @ phi
    S = (S + S.T) / 2  # (1 + lambda) T Sigma~
    df_post, df_prior = (1 + lam) * T - k, lT - k
    sign_s, logdet_s = np.linalg.slogdet(S)
    sign_p, logdet_p = np.linalg.slogdet(lT * sigma_star)
    sign_x, logdet_x = np.linalg.slogdet(prior_xx)
    if min(sign_s, sign_p, sign_x) <= 0:
        raise ModelError("the DSGE-VAR's moments are singular: the observables are collinear in the model (more observables "
                         "than shocks and measurement errors?)")
    logdet_A = 2 * float(np.sum(np.log(np.diag(cA[0]))))
    logml = (-n / 2 * logdet_A - df_post / 2 * logdet_s + n / 2 * logdet_x + df_prior / 2 * logdet_p
             - n * T / 2 * np.log(2 * np.pi) + n * (df_post - df_prior) / 2 * np.log(2)
             + special.multigammaln(df_post / 2, n) - special.multigammaln(df_prior / 2, n))
    return _NIW(phi_star, sigma_star, phi, S / ((1 + lam) * T), A, df_post, S, float(logml))


def log_marginal_likelihood(system: Any, Y_all: np.ndarray, lags: int, lam: float) -> float:
    """``log p(Y | theta, lambda)`` of the observed matrix ``Y_all`` (the first ``lags`` rows the initial lags) under the
    model in state-space form ``system``."""
    Y, X = _lag_design(np.asarray(Y_all, dtype=float), lags)
    return var_posterior(*population_moments(system, lags), Y, X, lam).logml


def rotation(impact: np.ndarray) -> np.ndarray:
    """``Omega`` of the QR factorisation ``A0 = Sigma_tr Omega``: ``Sigma_tr`` lower triangular with a positive diagonal,
    ``Omega`` orthonormal (DS 2004)."""
    q, r = np.linalg.qr(impact.T)
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return (q * signs).T


def _impact(system: Any) -> np.ndarray:
    """The model's impact of one-standard-deviation (orthogonalised) shocks on the observables: ``Z R chol(Q)``."""
    return system.Z @ system.R @ np.linalg.cholesky(system.Q + 1e-300 * np.eye(len(system.Q)))


def _ma(phi: np.ndarray, n: int, lags: int, periods: int) -> np.ndarray:
    """Moving-average coefficients (periods + 1, n, n) of the VAR with coefficients ``phi`` (k x n)."""
    from ..var import _lag_matrices, _ma as ma

    return ma(_lag_matrices(phi, 1, n, lags), periods)


# --- the likelihood ----------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class _DSGEVARLikelihood(_Likelihood):
    """``log p(Y | theta, lambda)`` in place of the Kalman filter's likelihood: the rest (building the model at trial
    values, priors, bounds, the objective) as ``Model.estimate``'s."""

    lags: int = 4
    lam: float = 1.0
    Y_all: Any = None  # the observed matrix, the first ``lags`` rows the initial lags

    def loglike(self, values: Mapping[str, float]) -> float:
        _, _, system = self.build(values)
        return log_marginal_likelihood(system, self.Y_all, self.lags, float(values.get(LAMBDA, self.lam)))

    def at(self, lam: float) -> _DSGEVARLikelihood:
        out = replace(self, lam=float(lam))
        return out


def _observed(measurement: Any, data: Any, span: Any) -> tuple[np.ndarray, pd.Index]:
    """The observed series over the sample, without the periods at either end where any is missing (a growth rate's
    first period); a gap inside is an error."""
    frame = measurement.observed(data, span)
    complete = np.isfinite(frame.to_numpy(float)).all(axis=1)
    if not complete.any():
        raise DataError("no period of the sample has every observed series")
    first, last = int(np.argmax(complete)), len(complete) - int(np.argmax(complete[::-1]))
    frame = frame.iloc[first:last]
    values = frame.to_numpy(float)
    if not np.all(np.isfinite(values)):
        bad = frame.index[~np.isfinite(values).all(axis=1)]
        raise DataError(f"the observed series are missing in {len(bad)} periods of the sample (first {bad[0]}): a DSGE-VAR "
                        "needs complete data; choose the span")
    return values, frame.index


# --- estimation --------------------------------------------------------------------------------------------------

def _mode(lik: _DSGEVARLikelihood, free: list[str], x0: np.ndarray, scale: np.ndarray, max_iter: int) -> tuple[np.ndarray, float, bool, int, np.ndarray]:
    """The posterior mode (or maximum likelihood) from ``x0``: the point, the objective, convergence, evaluations and
    the inverse Hessian (NaN where a parameter is at a bound or the Hessian is not negative definite)."""
    from ..estimation.numdiff import hessian
    from ..statespace import _at_optimum

    bounds = [lik.limits[n] for n in free]
    best = _optimise(lik, x0, scale, bounds, max_iter)
    if not np.isfinite(best["fun"]) or best["fun"] >= _PENALTY:
        raise ModelError("the DSGE-VAR's likelihood cannot be evaluated near the starting values")
    x = np.asarray(best["x"], dtype=float)
    converged = bool(best["success"]) or _at_optimum(lik.objective, x, bounds)
    k = len(free)
    cov = np.full((k, k), np.nan)
    interior = [i for i, n in enumerate(free) if not (np.isclose(x[i], bounds[i][0], atol=1e-5) or np.isclose(x[i], bounds[i][1], atol=1e-5))]
    if interior:
        def f(z: np.ndarray) -> float:
            full = x.copy()
            full[interior] = z
            return -lik.objective(full)

        try:
            sub = np.linalg.inv(-hessian(f, x[interior]))
            if np.all(np.diag(sub) > 0):
                cov[np.ix_(interior, interior)] = sub
        except (np.linalg.LinAlgError, ValueError):
            pass
    return x, float(best["fun"]), converged, int(best["nfev"]), cov


def _laplace(logpost: float, cov: np.ndarray) -> float:
    if not np.all(np.isfinite(cov)):
        return float("nan")
    sign, logdet = np.linalg.slogdet(cov)
    return float(logpost + 0.5 * len(cov) * np.log(2 * np.pi) + 0.5 * logdet) if sign > 0 else float("nan")


def dsge_var(model: Any, data: Any, measurement: Any, lags: int = 4, lambda_: Any = 1.0, lambdas: Sequence[float] | None = None,
             shocks: Sequence[str] | None = None, shock_sd: Mapping[str, float] | None = None, estimate: Sequence[str] | None = None,
             priors: Mapping[str, Any] | None = None, bounds: Mapping[str, tuple[float, float]] | None = None, span: Any = None,
             guess: Mapping[str, float] | None = None, max_iter: int = 2000, shock_corr: Mapping[Any, float] | None = None
             ) -> FittedDSGEVAR:
    """The work behind ``Model.dsge_var``: see the module docstring."""
    from .._capture import capture
    from ..threads import blas_threads

    with capture() as caught, blas_threads():
        out = _dsge_var(model, data, measurement, lags, lambda_, lambdas, shocks, shock_sd, estimate, priors, bounds, span,
                        guess, max_iter, shock_corr)
    if caught and out.provenance is not None:
        out = replace(out, provenance=replace(out.provenance, warnings=(*out.provenance.warnings, *caught)))
    return out


def _dsge_var(model: Any, data: Any, measurement: Any, lags: int, lambda_: Any, lambdas: Sequence[float] | None,
              shocks: Any, shock_sd: Any, estimate: Any, priors: Any, bounds: Any, span: Any, guess: Any, max_iter: int,
              shock_corr: Any) -> FittedDSGEVAR:
    from .._capture import fingerprint
    from .._version import __version__
    from ..estimation.params import point
    from ..metadata import Provenance

    if isinstance(lags, bool) or not isinstance(lags, (int, np.integer)) or lags < 1:
        raise SpecificationError(f"lags is a number of lags, at least 1, not {lags!r}")
    lags = int(lags)
    priors = dict(priors or {})
    lambda_prior = None
    if isinstance(lambda_, str):
        if lambda_ != "grid":
            raise SpecificationError(f"lambda_ is a number, 'grid' (lambda-hat) or a prior for lambda, not {lambda_!r}")
        how = "grid"
    elif hasattr(lambda_, "logpdf"):
        how, lambda_prior = "estimated", lambda_
    else:
        try:
            lambda_ = float(lambda_)
        except (TypeError, ValueError):
            raise SpecificationError(f"lambda_ is a number, 'grid' (lambda-hat) or a prior for lambda, not {lambda_!r}") from None
        how = "fixed"
    if LAMBDA in priors:
        if how == "fixed" and estimate is not None and LAMBDA in estimate:
            how, lambda_prior = "estimated", priors[LAMBDA]
        elif lambda_prior is None:
            raise SpecificationError("a prior for lambda is given in priors=: estimate it with lambda_=<prior> (or name it in estimate=)")
    rest = [n for n in estimate if n != LAMBDA] if estimate is not None else None
    if estimate is not None and LAMBDA in estimate and how != "estimated":
        raise SpecificationError("lambda is estimated with a prior: give it as lambda_=Gamma(...) (or in priors=)")
    measurement, shocks, params, structural, sd, errors, candidates, free, priors_all, limits, specs, (corr, pairs, given) = _setup(
        model, measurement, shocks, shock_sd, rest, {n: p for n, p in priors.items() if n != LAMBDA}, bounds, (), shock_corr)
    if LAMBDA in candidates:
        raise SpecificationError(f"the model has a parameter named {LAMBDA!r}: rename it, as the DSGE-VAR's prior weight takes the name")
    Y_all, index = _observed(measurement, data, span)
    n = Y_all.shape[1]
    T = len(Y_all) - lags
    k = 1 + n * lags
    if T <= k:
        raise DataError(f"{len(Y_all)} observations leave {T} after {lags} initial lags, too few for {k} regressors in each equation")
    lam_min = (k + n) / T
    lik = _DSGEVARLikelihood(model, params, structural, shocks, sd, errors, measurement, tuple(free), priors_all, limits, data, span,
                             dict(guess or {}), False, "auto", 0, False, 40, (), {n_: v for n_, v in corr.items() if n_ in free or v != 0},
                             pairs, lags=lags, lam=float(lambda_) if how == "fixed" else 1.0, Y_all=Y_all)
    initial = {**candidates, **{n_: float(s.start) for n_, s in specs.items() if s is not None and s.start is not None}}
    x0 = np.array([min(max(initial[n_], lo + 1e-6 if np.isfinite(lo) else -np.inf), hi - 1e-6 if np.isfinite(hi) else np.inf)
                   for n_, (lo, hi) in ((n_, limits[n_]) for n_ in free)])
    if how == "fixed" and lambda_ < lam_min:
        raise SpecificationError(f"lambda must be at least (k + n) / T = {lam_min:.4g} ({k} regressors, {n} observables, {T} "
                                 f"observations), not {lambda_:g}")
    if how == "estimated":
        support = getattr(lambda_prior, "support", (0.0, np.inf))
        low = max(lam_min, support[0])
        free = [*free, LAMBDA]
        limits = {**limits, LAMBDA: (low, support[1])}
        priors_all = {**priors_all, LAMBDA: lambda_prior}
        start = float(getattr(lambda_prior, "mean", max(1.0, 2 * low)))
        x0 = np.append(x0, min(max(start, low * 1.05 + 1e-6), support[1] - 1e-6))
        lik = replace(lik, free=tuple(free), limits=limits, priors=priors_all)
    if lik.objective(x0) >= _PENALTY:
        raise ModelError("the model cannot be solved, or has no population moments (a unit root?), at the starting values; "
                         "check the calibration, the measurement equations and the data")
    scale = np.array([_scale(priors_all.get(n_), x0[i]) for i, n_ in enumerate(free)])

    grid = None
    evaluations = 0
    if how == "grid":
        values = sorted(float(v) for v in (lambdas if lambdas is not None else GRID))
        if lambdas is None:
            values = [v for v in values if v >= lam_min]
        low_ones = [v for v in values if v < lam_min]
        if low_ones:
            raise SpecificationError(f"lambda must be at least (k + n) / T = {lam_min:.4g}: {', '.join(f'{v:g}' for v in low_ones)} "
                                     "in lambdas are below it")
        if not values:
            raise SpecificationError(f"no value of the lambda grid is at least (k + n) / T = {lam_min:.4g}: give lambdas=")
        rows, results, x_start = [], [], x0
        from ..progress import report

        for i, lam in enumerate(values):
            report(i / len(values), f"DSGE-VAR: lambda {lam:g} ({i + 1} of {len(values)})")
            at = lik.at(lam)
            x, fun, converged, nfev, cov = _mode(at, free, x_start, scale, max_iter)
            evaluations += nfev
            loglike = at.loglike(dict(zip(free, map(float, x))))
            mdd = _laplace(-fun, cov) if priors_all else loglike
            rows.append({"lambda": lam, "log marginal density": mdd, "log likelihood": loglike, "log posterior": -fun,
                         **dict(zip(free, map(float, x)))})
            results.append((lam, x, fun, converged, cov))
            x_start = x
        grid = pd.DataFrame(rows).set_index("lambda")
        scores = grid["log marginal density"].to_numpy()
        best = int(np.nanargmax(np.where(np.isfinite(scores), scores, -np.inf))) if np.any(np.isfinite(scores)) else 0
        lam_used, x, fun, converged, cov = results[best]
        lik = lik.at(lam_used)
    else:
        x, fun, converged, nfev, cov = _mode(lik, free, x0, scale, max_iter)
        evaluations += nfev
        lam_used = float(x[free.index(LAMBDA)]) if how == "estimated" else float(lambda_)

    values = dict(zip(free, map(float, x)))
    loglike = lik.loglike(values)
    m, solution, system = lik.build(values)
    se = dict(zip(free, np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))))
    updated = params.with_values({n_: values[n_] for n_ in free if n_ in structural}, status="estimated")
    updated = type(updated)(tuple(replace(p, se=float(se[p.name])) if p.name in se else p for p in updated))
    nk = len(free)
    stats = {"loglike": float(loglike), "nobs": float(T), "k": float(nk), "aic": -2 * loglike + 2 * nk, "bic": -2 * loglike + nk * np.log(T),
             "lambda": float(lam_used), "lambda_min": float(lam_min)}
    if priors_all:
        stats["logpost"] = float(-fun)
    shock_sizes = {s: values.get(f"sd_{s}", sd[f"sd_{s}"]) for s in shocks}
    used_measurement = measurement.with_errors({o: values.get(f"sd_{o}", errors[f"sd_{o}"]) for o in measurement.observables})
    sources = list(measurement.data_series())
    kind = ("posterior mode" if priors_all else "maximum likelihood") + " (DSGE-VAR)"
    rows_used = index[lags:]
    method = {"type": kind, "solution": "first-order rational expectations", "likelihood": "DSGE-VAR p(Y | theta, lambda), closed form",
              "lags": lags, "lambda": float(lam_used), "lambda choice": how, "lambda minimum": float(lam_min),
              "optimizer": "BFGS on transformed parameters, then L-BFGS-B", "evaluations": int(evaluations), "estimated": list(free)}
    if grid is not None:
        method["lambda grid"] = [float(v) for v in grid.index]
    if lambda_prior is not None:
        from .system import prior_text

        method["lambda prior"] = prior_text(lambda_prior)
    provenance = Provenance(
        data_vintages={s: data.variables[s].vintage for s in sources if s in data.variables},
        spec={"equations": [e.name for e in model.compile().equations], "measurement": list(measurement.equations), "shocks": list(shocks),
              "lags": lags},
        method=method, sample=(str(rows_used[0]), str(rows_used[-1])), nobs=int(T), dropped=0, package_version=__version__,
        sample_source=data.sample_record(span)[0], sample_definition=data.sample_record(span)[1],
        data_fingerprint=fingerprint(data.frame.loc[index, [s for s in sources if s in data.frame.columns]]), converged=converged,
        presample=tuple(str(i) for i in index[:lags]),
    )
    try:
        smoothed = system.filter(data, span)
    except Exception:  # noqa: BLE001 - the Kalman smoother is extra here: the DSGE-VAR does not need it
        smoothed = None
    shock_correlations = {p: r for p, r in lik.shock_corr(values).items() if r != 0}
    fit = FittedModel(m, point(free, x, cov), updated, shock_sizes, used_measurement, solution, system, smoothed, stats, priors_all,
                      {n_: lik.limits[n_] for n_ in free}, kind, provenance, lik.evaluator(solution), lik.log_posterior, data, span, (),
                      shock_correlations)
    return FittedDSGEVAR(fit, lags, float(lam_used), how, tuple(measurement.observables), Y_all, index, grid, None, lambda_prior)


# --- the result --------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FittedDSGEVAR:
    """A DSGE-VAR(lambda): the model's parameters at their posterior mode (``fit``, a ``FittedModel`` whose likelihood
    is the DSGE-VAR's), and after ``mcmc`` their posterior draws (``posterior``); the VAR given them. See
    ``pymetrix.models.dsge_var``."""

    fit: Any  # FittedModel at the mode of theta (and lambda when estimated)
    lags: int
    lambda_: float  # the lambda used: fixed, lambda-hat from the grid, or the mode of its posterior
    lambda_choice: str  # "fixed", "grid" or "estimated"
    observables: tuple[str, ...]
    Y_all: np.ndarray = field(repr=False)  # the observed series, the first ``lags`` rows the initial lags
    index: Any = field(repr=False)
    grid: pd.DataFrame | None = None  # lambda-hat: each lambda's log marginal density, likelihood and mode
    posterior: Any = None  # a bayes.Posterior of theta (and lambda when estimated), after mcmc
    lambda_prior: Any = None
    _cache: dict = field(default_factory=dict, repr=False)

    # basic facts

    @property
    def nobs(self) -> int:
        return len(self.Y_all) - self.lags

    @property
    def lambda_min(self) -> float:
        """The smallest lambda with a proper prior: ``(k + n) / T``."""
        n = len(self.observables)
        return (1 + n * self.lags + n) / self.nobs

    @property
    def params(self) -> Any:
        return self.fit.params

    @property
    def stats(self) -> dict[str, float]:
        out = dict(self.fit.stats)
        if self.posterior is not None:
            out["draws"] = float(self.posterior.draws.shape[0] * self.posterior.draws.shape[1])
        return out

    @property
    def provenance(self) -> Any:
        """The mode's provenance, with the sampler's after ``mcmc``."""
        if self.posterior is not None:
            return self.posterior.provenance
        return self.fit.provenance

    @property
    def regressors(self) -> list[str]:
        return ["const"] + [f"{v}[-{lag}]" for lag in range(1, self.lags + 1) for v in self.observables]

    def _design(self) -> tuple[np.ndarray, np.ndarray]:
        return _lag_design(self.Y_all, self.lags)

    def _values(self, at: Mapping[str, float] | None = None) -> dict[str, float]:
        values = self.fit.params.as_dict()
        unknown = sorted(set(at or {}) - set(values))
        if unknown:
            raise SpecificationError(f"at= gives {', '.join(unknown)}, which are not estimated ({', '.join(values)})")
        return {**values, **{n: float(v) for n, v in (at or {}).items()}}

    def _niw(self, values: Mapping[str, float], lam: float | None = None) -> tuple[_NIW, Any]:
        _, _, system = self.fit.evaluate(dict(values))
        Y, X = self._design()
        lam = float(values.get(LAMBDA, self.lambda_)) if lam is None else float(lam)
        return var_posterior(*population_moments(system, self.lags), Y, X, lam), system

    def _frame(self, values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=self.regressors, columns=list(self.observables))

    def _square(self, values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=list(self.observables), columns=list(self.observables))

    # the VAR given theta

    def moments(self, at: Mapping[str, float] | None = None) -> dict[str, pd.DataFrame]:
        """The model's population moments behind the prior at the mode (or ``at``): ``G_yy``, ``G_xy`` and ``G_xx``
        (uncentred, the constant first)."""
        _, _, system = self.fit.evaluate(self._values(at))
        G_yy, G_xy, G_xx = population_moments(system, self.lags)
        return {"G_yy": self._square(G_yy), "G_xy": self._frame(G_xy),
                "G_xx": pd.DataFrame(G_xx, index=self.regressors, columns=self.regressors)}

    def var_prior(self, at: Mapping[str, float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The prior's centre at the mode (or ``at``): the model's VAR approximation ``Phi*`` (regressors x equations) and
        ``Sigma*``."""
        post, _ = self._niw(self._values(at))
        return self._frame(post.phi_star), self._square(post.sigma_star)

    def var_posterior(self, at: Mapping[str, float] | None = None, lambda_: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The VAR's posterior given theta at the mode (or ``at``) and lambda (or ``lambda_``): the posterior mean of
        ``Phi`` and ``Sigma~`` (the posterior's scale divided by ``(1 + lambda) T``)."""
        post, _ = self._niw(self._values(at), lambda_)
        return self._frame(post.phi), self._square(post.sigma)

    def var_draws(self, draws: int = 500, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Draws of the VAR's coefficients ``Phi`` (draws x k x n, regressors as ``regressors``) and covariance ``Sigma``
        (draws x n x n): one from the normal-inverse-Wishart posterior given each of ``draws`` draws of theta spread
        over the chains (after ``mcmc``), or given theta at the mode."""
        _, B, S, _ = self._draws(draws, seed)
        return B, S

    def _thetas(self, draws: int) -> list[dict[str, float]]:
        if self.posterior is not None:
            return self.posterior._subsample(draws)
        return [self._values()] * int(draws)

    def _draws(self, draws: int, seed: int | None) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray, list[np.ndarray]]:
        from scipy import stats

        if int(draws) < 1:
            raise SpecificationError("draws is a number of draws, at least 1")
        key = ("draws", int(draws), seed)
        if key in self._cache:
            return self._cache[key]
        rng = make_rng(seed)
        thetas = self._thetas(int(draws))
        n, k = len(self.observables), 1 + len(self.observables) * self.lags
        B, S, impacts = np.empty((len(thetas), k, n)), np.empty((len(thetas), n, n)), []
        last: tuple[Any, Any, Any] | None = None
        for i, values in enumerate(thetas):
            if last is None or last[0] != values:
                post, system = self._niw(values)
                chol_inv = np.linalg.cholesky(np.linalg.inv(post.precision))
                last = (values, post, (system, chol_inv))
            post, (system, chol_inv) = last[1], last[2]
            sigma = np.atleast_2d(stats.invwishart(df=post.df, scale=post.scale).rvs(random_state=rng))
            B[i] = post.phi + chol_inv @ rng.standard_normal((k, n)) @ np.linalg.cholesky(sigma).T
            S[i] = sigma
            impacts.append(_impact(system))
        out = (thetas, B, S, impacts)
        self._cache[key] = out
        return out

    # responses, variance decompositions and forecasts

    def _shocks(self) -> list[str]:
        shocks = list(self.fit.system.shocks)
        if len(shocks) != len(self.observables):
            raise SpecificationError(f"the model's rotation needs as many shocks as observables: {len(shocks)} shocks "
                                     f"({', '.join(shocks)}) for {len(self.observables)} observables")
        return shocks

    @api()
    def irf(self, periods: int = 20, draws: int = 500, coverage: float = 0.9, seed: int | None = None) -> Any:
        """Impulse responses of the observables to the model's shocks (one standard deviation), identified by the
        model's rotation: for each draw of theta (after ``mcmc``; else at the mode) and of the VAR given it, the impact
        ``chol(Sigma) Omega(theta)`` (DS 2004). The median with the central ``coverage`` band (an ``IRF``); ``draws=0``:
        at the posterior mean given theta at the mode, without bands."""
        from ..var import IRF

        shocks = self._shocks()
        n = len(self.observables)
        label = "DSGE rotation (Del Negro-Schorfheide)"
        if draws == 0:
            post, system = self._niw(self._values())
            values = _ma(post.phi, n, self.lags, periods) @ (np.linalg.cholesky(post.sigma) @ rotation(_impact(system)))
            return IRF(values, self.observables, shocks, identification=label + ", posterior mean given the mode", coverage=coverage)
        _, B, S, impacts = self._draws(draws, seed)
        sims = np.stack([_ma(B[i], n, self.lags, periods) @ (np.linalg.cholesky(S[i]) @ rotation(impacts[i])) for i in range(len(B))])
        tail = (1 - coverage) / 2
        lower, upper = np.quantile(sims, [tail, 1 - tail], axis=0)
        return IRF(np.median(sims, axis=0), self.observables, shocks, lower, upper,
                   identification=label + (", posterior median" if self.posterior is not None else ", posterior median given the mode"),
                   coverage=coverage)

    @api()
    def dsge_irf(self, periods: int = 20, at: Mapping[str, float] | None = None) -> Any:
        """The model's own responses of the observables to its shocks at the mode (or ``at``), to set beside ``irf``."""
        from ..var import IRF

        _, _, system = self.fit.evaluate(self._values(at))
        x = system.R @ np.linalg.cholesky(system.Q + 1e-300 * np.eye(len(system.Q)))
        out = []
        for _ in range(periods + 1):
            out.append(system.Z @ x)
            x = system.T @ x
        return IRF(np.stack(out), self.observables, list(system.shocks), identification="the model's own (DSGE), at the mode")

    @api()
    def fevd(self, periods: int = 20) -> Any:
        """Forecast-error variance decomposition of the observables by the model's shocks, at the posterior mean of the
        VAR given theta at the mode, with the model's rotation (an ``FEVD``)."""
        from ..var import FEVD

        shocks = self._shocks()
        post, system = self._niw(self._values())
        theta = _ma(post.phi, len(self.observables), self.lags, periods - 1) @ (np.linalg.cholesky(post.sigma) @ rotation(_impact(system)))
        contributions = np.cumsum(theta**2, axis=0)
        return FEVD(contributions / contributions.sum(axis=2, keepdims=True), self.observables, shocks,
                    "DSGE rotation (Del Negro-Schorfheide)")

    def _bvar(self, draws: int, seed: int | None) -> Any:
        """The VAR as a ``FittedBVAR`` (posterior mean given the mode; the draws given theta's draws), for its forecasts."""
        from ..bvar import FittedBVAR
        from ..var import VAR, Cholesky, _Arrays, _Estimate

        n, p = len(self.observables), self.lags
        Y, X = self._design()
        post, _ = self._niw(self._values())
        U = Y - X @ post.phi
        T = len(Y)
        arrays = _Arrays(Y, X[:, 1:], X[:, :1], ["const"], self.index[p:], self.observables)
        sigma_ml = U.T @ U / T
        est = _Estimate(post.phi, U, post.scale / max(post.df - n - 1, 1.0), sigma_ml, np.linalg.inv(post.precision), {},
                        float(-T / 2 * (n * np.log(2 * np.pi) + np.linalg.slogdet(sigma_ml)[1] + n)))
        _, B, S, _ = self._draws(draws, seed)
        return FittedBVAR(VAR(list(self.observables), p), p, est, arrays, Cholesky(), self.provenance, hyperparameters={"lambda": self.lambda_},
                          log_ml=float(self.fit.stats["loglike"]), draws_B=B, draws_sigma=S)

    @api()
    def forecast(self, span: tuple[Any, Any], data: Any = None, draws: int = 500, alpha: float = 0.1, seed: int | None = None) -> Any:
        """Forecasts of the observables over ``span`` from the VAR's predictive distribution: each draw of (theta, Phi,
        Sigma) simulates a path with its own shocks, so the bands include parameter uncertainty. From the end of the
        estimation data, or of ``data`` (made into the observables by the measurement equations). Columns: mean,
        ``_se``, ``_median`` and the ``1 - alpha`` interval, ``_lower`` and ``_upper`` (as ``FittedBVAR.forecast``)."""
        from ..data import Dataset
        from ..metadata import DatasetMeta

        source = self.fit.data if data is None else data
        observed = Dataset.from_frame(self.fit.measurement.observed(source))
        model = self._bvar(draws, seed)
        periods, paths = model.predictive_draws(observed, span, draws, 0 if seed is None else seed)
        return model._summarised(observed, periods, paths, alpha, DatasetMeta(
            title=f"DSGE-VAR({self.lags}) forecast, lambda {self.lambda_:.3g}",
            description=f"Predictive mean and {1 - alpha:.0%} interval from {len(paths)} draws of the VAR"
                        + (" given posterior draws of the model's parameters." if self.posterior is not None else " given the mode.")))

    # theta's posterior

    @api()
    def mcmc(self, draws: int = 10000, chains: int = 2, burn: int | None = None, scale: float | None = None,
             seed: int | None = None, progress: Any = None, workers: int | None = None, adapt: bool = True,
             modes: Any = None, jump: float = 0.05) -> FittedDSGEVAR:
        """The posterior of theta (and lambda, when estimated) by random-walk Metropolis-Hastings from the mode, as
        ``FittedModel.mcmc`` (every parameter needs a prior; ``modes`` and ``jump`` as there): this DSGE-VAR with its
        ``posterior``, so that ``irf``, ``forecast`` and ``var_draws`` integrate over theta."""
        from .bayes import mcmc

        post = mcmc(self.fit, draws=draws, chains=chains, burn=burn, scale=scale, seed=seed, progress=progress, workers=workers,
                    adapt=adapt, modes=modes, jump=jump)
        return replace(self, posterior=post, _cache={})

    def laplace(self) -> float:
        """Log marginal data density ``p(Y | lambda)`` by the Laplace approximation at the mode (needs priors)."""
        from .bayes import laplace

        return laplace(self.fit)

    def marginal_likelihood(self) -> float:
        """Log marginal data density: the modified harmonic mean of the draws after ``mcmc``, else Laplace's."""
        return self.posterior.marginal_likelihood() if self.posterior is not None else self.laplace()

    def lambdas(self) -> Table:
        """The log marginal data density at each lambda of the grid (``lambda_="grid"``), with the likelihood and the
        mode of theta at each: the hump whose top is lambda-hat."""
        if self.grid is None:
            raise SpecificationError("lambda was not chosen on a grid: estimate with lambda_='grid' (lambdas=[...])")
        frame = self.grid.copy()
        best = frame.index[np.nanargmax(frame["log marginal density"].to_numpy())]
        notes = [f"lambda-hat = {best:g}: the highest log marginal data density "
                 + ("(Laplace approximation at each mode)" if "logpost" in self.fit.stats else "(highest likelihood, without priors)"),
                 f"lambda is at least (k + n) / T = {self.lambda_min:.4g}; large lambda holds the VAR to the model, small lambda leaves it free"]
        frame.index = pd.Index([f"{v:g}" for v in frame.index], name="lambda")
        return Table.from_frame(frame, title=f"DSGE-VAR({self.lags}): marginal data density by lambda", notes=notes)

    def lambda_plot(self) -> Any:
        """The log marginal data density against lambda (the hump), lambda-hat marked."""
        from ..figures import Figure, Line, Panel, Rule

        if self.grid is None:
            raise SpecificationError("lambda was not chosen on a grid: estimate with lambda_='grid' (lambdas=[...])")
        series = self.grid["log marginal density"].copy()
        series.index = pd.Index(series.index.to_numpy(float), name="lambda")
        best = float(series.index[np.nanargmax(series.to_numpy())])
        panel = Panel((Line(series, "log marginal data density"), Rule(best, style="--", vertical=True)),
                      title=f"DSGE-VAR({self.lags}): lambda-hat {best:g}", xlabel="lambda", ylabel="log marginal data density")
        return Figure((panel,), title="Marginal data density by lambda")

    def summary(self) -> Table:
        from .system import prior_text
        from .bayes import hpd

        p = self.fit.params
        rows = []
        post = self.posterior
        for j, (name, value, se) in enumerate(zip(p.names, p.values, p.se)):
            row = [prior_text(self.fit.priors.get(name)), float(value), float(se)]
            if post is not None:
                x = post.draws[:, :, list(post.names).index(name)]
                low, high = hpd(x, post.coverage)
                row += [float(x.mean()), low, high]
            rows.append(row)
        columns = ["prior", "mode", "std err"] + (["mean", f"HPD {post.coverage:.0%} low", f"HPD {post.coverage:.0%} high"] if post is not None else [])
        s = self.fit.stats
        prov = self.fit.provenance
        choice = {"fixed": "fixed", "grid": "lambda-hat, the best marginal data density on the grid",
                  "estimated": f"estimated, prior {prior_text(self.lambda_prior)}"}[self.lambda_choice]
        notes = [f"DSGE-VAR({self.lags}) in {', '.join(self.observables)} with a constant; lambda {self.lambda_:.4g} ({choice}); "
                 f"lambda T = {self.lambda_ * self.nobs:.1f} prior observations",
                 f"Sample: {prov.sample[0]} to {prov.sample[1]}, {self.nobs} observations after {self.lags} initial lags; lambda is at least "
                 f"(k + n) / T = {self.lambda_min:.4g}",
                 f"log p(Y | theta, lambda) at the mode: {s['loglike']:.4f}" + (f"; log posterior {s['logpost']:.4f}" if "logpost" in s else "")]
        try:
            notes.append(f"Log marginal data density: {self.marginal_likelihood():.3f} "
                         f"({'modified harmonic mean' if post is not None else 'Laplace'})")
        except (ModelError, ValueError, np.linalg.LinAlgError):
            pass
        if post is not None:
            chains, n = post.draws.shape[:2]
            notes.append(f"Random-walk Metropolis-Hastings: {chains} chain(s) of {n} draws; acceptance {', '.join(f'{a:.2f}' for a in post.acceptance)}")
        if prov is not None and not prov.converged:
            notes.append("The optimiser did not converge: check the estimates and starting values")
        if prov is not None and prov.warnings:
            notes.append("Warnings: " + "; ".join(prov.warnings))
        return Table(columns, rows, index=list(p.names), title=f"DSGE-VAR({self.lags}), lambda {self.lambda_:.3g}", notes=notes)

    def to_frame(self, view: str | None = None) -> Any:
        if view is not None:  # one of its views by name, as for every result (pymetrix.views)
            from ..views import _views_to_frame

            return _views_to_frame(self, view)
        return self.fit.params.to_frame() if hasattr(self.fit.params, "to_frame") else pd.DataFrame(
            {"estimate": self.fit.params.values}, index=list(self.fit.params.names))

    def __repr__(self) -> str:
        drawn = f", {self.posterior.draws.shape[0]}x{self.posterior.draws.shape[1]} draws" if self.posterior is not None else ""
        return (f"<FittedDSGEVAR VAR({self.lags}) in {', '.join(self.observables)}, lambda {self.lambda_:.3g} ({self.lambda_choice}); "
                f"{len(self.fit.params.names)} estimated{drawn}>")

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()
