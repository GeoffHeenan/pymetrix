"""Bayesian VARs: the Minnesota prior in its conjugate normal-inverse-Wishart form.

    VAR(["y", "pi", "r"], lags=4).fit(data, method=Bayes(prior=Minnesota(tightness=0.2)))
    VAR(...).fit(data, method=Bayes(prior=Minnesota(sum_of_coefficients=1, initial_observation=1, optimize=True)))

The prior shrinks each equation towards a random walk (``own_mean=1``; 0 for growth
rates): the coefficient on lag l of variable j in equation i has prior variance
``(tightness / l**decay)**2 * psi_i / psi_j``, where psi are residual variances of
univariate autoregressions. Deterministic terms get a loose prior. Optional dummy
observations add the sum-of-coefficients prior (Doan, Litterman and Sims) and the
single-unit-root (initial observation) prior (Sims): small values of their
hyperparameters pull towards unit roots and cointegration.

The prior is conjugate, so the posterior and the marginal likelihood are exact:
``B | Sigma ~ N(B~, Sigma (x) Omega~)``, ``Sigma ~ IW(Psi~, d + T)``. With
``optimize=True`` the hyperparameters maximise the marginal likelihood, as in
Giannone, Lenza and Primiceri (2015). Impulse responses and forecasts use posterior
draws, so their bands include parameter uncertainty.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

import numpy as np
import pandas as pd
from scipy import linalg, optimize, special

from pymetrix._lazy import stats

from ._api import api
from ._capture import _sample_fields, data_fingerprint
from ._random import generator as make_rng
from ._serial import to_jsonable
from ._version import __version__
from .errors import SpecificationError
from .forms import Widget
from .metadata import Provenance
from .output import Table
from .var import IRF, FittedVAR, _Arrays, _Estimate, _lag_matrices, _ma, _scheme_label

if TYPE_CHECKING:
    from .data import Dataset
    from .var import VAR

__all__ = ["FittedBVAR", "Minnesota", "NormalWishart"]


@dataclass(frozen=True)
class Minnesota:
    """The Minnesota prior for a VAR, with optional dummy-observation priors."""

    label: ClassVar[str] = "Minnesota prior"
    topics: ClassVar[tuple[str, ...]] = ("bayesian-var",)

    tightness: Annotated[float, Widget("number", min=1e-4, max=10, step=0.05, help="overall tightness (lambda); smaller shrinks more")] = 0.2
    decay: Annotated[float, Widget("number", min=0, max=4, help="longer lags shrink as 1/lag**decay")] = 2.0
    own_mean: Annotated[float | Mapping[str, float], Widget("number", help="prior mean of each variable's own first lag: 1 for levels (random walk), 0 for growth rates")] = 1.0
    sum_of_coefficients: Annotated[float | None, Widget("number", help="sum-of-coefficients prior (mu); smaller pulls harder towards unit roots; empty = none")] = None
    initial_observation: Annotated[float | None, Widget("number", help="single-unit-root prior (delta); empty = none")] = None
    optimize: Annotated[bool, Widget("checkbox", help="choose the hyperparameters by maximising the marginal likelihood")] = False
    deterministic: Annotated[float, Widget("number", advanced=True, help="prior variance of the constant, trend and exogenous terms (loose)")] = 1e6
    scale_lags: Annotated[int, Widget("number", min=1, max=12, advanced=True, help="lags of the univariate autoregressions that scale the prior")] = 1

    def __post_init__(self) -> None:
        for name in ("tightness", "sum_of_coefficients", "initial_observation"):
            value = getattr(self, name)
            if value is not None and not value > 0:
                raise SpecificationError(f"{name} must be positive")


@dataclass(frozen=True)
class NormalWishart:
    """The Minnesota prior in its independent normal-Wishart form, sampled by Gibbs: the coefficients B ~ N(b, V) apart
    from Sigma ~ IW. Unlike the conjugate prior, other variables' lags can be shrunk harder than a variable's own
    (``cross_tightness`` below 1), and each equation's prior does not depend on the others' error variances.

    The coefficient on lag l of variable j in equation i has prior standard deviation tightness / l^decay, times
    cross_tightness sqrt(psi_i / psi_j) when j is not i (psi from univariate autoregressions); Sigma ~ IW with mean
    diag(psi), as in the conjugate prior. There is no closed-form marginal likelihood: Chib's (1995) estimate from the
    Gibbs draws stands in for it (``pymetrix.bvar_hierarchical``), and with ``optimize=True`` the hyperparameters
    maximise it."""

    label: ClassVar[str] = "Independent normal-Wishart (Minnesota) prior"
    topics: ClassVar[tuple[str, ...]] = ("bayesian-var",)

    tightness: Annotated[float, Widget("number", min=1e-4, max=10, step=0.05, help="overall tightness (lambda); smaller shrinks more")] = 0.2
    cross_tightness: Annotated[float, Widget("number", min=1e-4, max=1, step=0.05, help="relative tightness of other variables' lags (theta); 1 = as own lags")] = 0.5
    decay: Annotated[float, Widget("number", min=0, max=4, help="longer lags shrink as 1/lag**decay")] = 2.0
    own_mean: Annotated[float | Mapping[str, float], Widget("number", help="prior mean of each variable's own first lag: 1 for levels (random walk), 0 for growth rates")] = 1.0
    sum_of_coefficients: Annotated[float | None, Widget("number", help="sum-of-coefficients prior (mu), as dummy observations; empty = none")] = None
    initial_observation: Annotated[float | None, Widget("number", help="single-unit-root prior (delta), as dummy observations; empty = none")] = None
    deterministic: Annotated[float, Widget("number", advanced=True, help="prior variance of the constant, trend and exogenous terms (loose)")] = 1e6
    scale_lags: Annotated[int, Widget("number", min=1, max=12, advanced=True, help="lags of the univariate autoregressions that scale the prior")] = 1
    burn: Annotated[int, Widget("number", min=0, max=100_000, advanced=True, help="Gibbs draws discarded before the posterior draws are kept")] = 1000
    optimize: Annotated[bool, Widget("checkbox", help="choose the hyperparameters by Chib's estimate of the marginal likelihood")] = False

    def __post_init__(self) -> None:
        for name in ("tightness", "cross_tightness", "sum_of_coefficients", "initial_observation"):
            value = getattr(self, name)
            if value is not None and not value > 0:
                raise SpecificationError(f"{name} must be positive")
        if self.burn < 0:
            raise SpecificationError("burn must be zero or more")


def _gibbs(Y: np.ndarray, X: np.ndarray, b: np.ndarray, V: np.ndarray, Psi: np.ndarray, d: float, draws: int, burn: int,
           rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Gibbs sampling of ``Y = X B + E``, rows of E ~ N(0, Sigma), with vec(B) ~ N(vec(b), diag(V)) (V (k, K), column by
    column: equation by equation) and Sigma ~ IW(Psi, d): vec(B) | Sigma is normal with precision diag(1/V) +
    Sigma^-1 (x) X'X, and Sigma | B is IW(Psi + E'E, d + T). Starts from least squares."""
    T, K = Y.shape
    k = X.shape[1]
    XX, XY = X.T @ X, X.T @ Y
    prior_prec = 1.0 / V.reshape(-1, order="F")
    prior_term = prior_prec * b.reshape(-1, order="F")
    B = np.linalg.lstsq(X, Y, rcond=None)[0]
    Bs, Ss = np.empty((draws, k, K)), np.empty((draws, K, K))
    for i in range(burn + draws):
        B, Sigma = _gibbs_step(Y, X, b, V, Psi, d, B, rng, (XX, XY, prior_prec, prior_term))
        if i >= burn:
            Bs[i - burn], Ss[i - burn] = B, Sigma
    return Bs, Ss


def _gibbs_step(Y: np.ndarray, X: np.ndarray, b: np.ndarray, V: np.ndarray, Psi: np.ndarray, d: float, B: np.ndarray,
                rng: np.random.Generator, cached: tuple[Any, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One sweep of ``_gibbs``: Sigma given B, then B given Sigma."""
    T, K = Y.shape
    k = X.shape[1]
    if cached is None:
        prior_prec = 1.0 / V.reshape(-1, order="F")
        cached = (X.T @ X, X.T @ Y, prior_prec, prior_prec * b.reshape(-1, order="F"))
    XX, XY, prior_prec, prior_term = cached
    E = Y - X @ B
    S = Psi + E.T @ E
    Sigma = stats.invwishart.rvs(df=d + T, scale=(S + S.T) / 2, random_state=rng).reshape(K, K)
    Si = np.linalg.inv(Sigma)
    P = np.kron(Si, XX)
    P[np.diag_indices_from(P)] += prior_prec
    c = linalg.cho_factor(P)
    mean = linalg.cho_solve(c, prior_term + (XY @ Si).reshape(-1, order="F"))
    z = linalg.solve_triangular(c[0], rng.standard_normal(k * K), lower=c[1])  # P = U'U: U^-1 z has covariance P^-1
    return (mean + z).reshape(k, K, order="F"), Sigma


# ---------------------------------------------------------------------------------------------
# the conjugate posterior


@dataclass(frozen=True, eq=False)
class _Posterior:
    B: np.ndarray  # posterior mean (k, K)
    Omega: np.ndarray  # (k, k): B | Sigma ~ N(B, Sigma (x) Omega)
    Psi: np.ndarray  # (K, K): Sigma ~ IW(Psi, df)
    df: float
    logml: float


def _niw(Y: np.ndarray, X: np.ndarray, b: np.ndarray, omega: np.ndarray, Psi: np.ndarray, d: float) -> _Posterior:
    """Posterior and log marginal likelihood for ``Y = X B + E``, ``B | Sigma ~ N(b, Sigma (x) diag(omega))``,
    ``Sigma ~ IW(Psi, d)`` (Giannone, Lenza and Primiceri 2015, appendix)."""
    T, K = Y.shape
    Oinv = np.diag(1.0 / omega)
    A = X.T @ X + Oinv
    cA = linalg.cho_factor(A)
    B = linalg.cho_solve(cA, X.T @ Y + Oinv @ b)
    E = Y - X @ B
    D = B - b
    Psi_post = Psi + E.T @ E + D.T @ Oinv @ D
    Psi_post = (Psi_post + Psi_post.T) / 2
    logdet_A = 2 * np.sum(np.log(np.diag(cA[0])))
    logml = (-T * K / 2 * np.log(np.pi) + special.multigammaln((T + d) / 2, K) - special.multigammaln(d / 2, K)
             - K / 2 * np.sum(np.log(omega)) - K / 2 * logdet_A
             + d / 2 * np.linalg.slogdet(Psi)[1] - (T + d) / 2 * np.linalg.slogdet(Psi_post)[1])
    return _Posterior(B, linalg.cho_solve(cA, np.eye(len(A))), Psi_post, T + d, float(logml))


def _scales(a: _Arrays, lags: int) -> np.ndarray:
    """Residual variances of univariate AR(lags) regressions with a constant, one per variable."""
    T, K = a.Y.shape
    out = np.empty(K)
    for j in range(K):
        X = np.column_stack([np.ones(T)] + [a.L[:, l * K + j] for l in range(lags)])
        u = a.Y[:, j] - X @ np.linalg.lstsq(X, a.Y[:, j], rcond=None)[0]
        out[j] = u @ u / (T - X.shape[1])
    return out


@dataclass(frozen=True, eq=False)
class _Setup:
    Y: np.ndarray
    X: np.ndarray
    psi: np.ndarray
    ybar: np.ndarray  # mean of the presample observations
    mean: np.ndarray  # prior mean of each own first lag
    d_names: list[str]
    p: int


def _setup(spec: VAR, a: _Arrays, p: int, prior: Minnesota) -> _Setup:
    K = len(spec.variables)
    if prior.scale_lags > a.L.shape[1] // K:
        raise SpecificationError("scale_lags cannot exceed the VAR's lags")
    own = prior.own_mean
    if isinstance(own, Mapping):
        unknown = sorted(set(own) - set(spec.variables))
        if unknown:
            raise SpecificationError(f"own_mean names {unknown}, which are not variables of the VAR")
        mean = np.array([float(own.get(v, 1.0)) for v in spec.variables])
    else:
        mean = np.full(K, float(own))
    X = np.column_stack([a.D, a.L[:, : K * p]])
    ybar = a.L[0, : K * p].reshape(p, K).mean(axis=0)
    return _Setup(a.Y, X, _scales(a, prior.scale_lags), ybar, mean, list(a.d_names), p)


def _prior(s: _Setup, prior: Minnesota, tightness: float, soc: float | None, dio: float | None):
    """Prior mean b, variances omega, Psi, d, and dummy observations (Yd, Xd)."""
    K = len(s.psi)
    nd = len(s.d_names)
    k = nd + K * s.p
    b = np.zeros((k, K))
    b[nd : nd + K, :] = np.diag(s.mean)
    omega = np.empty(k)
    omega[:nd] = prior.deterministic
    for l in range(1, s.p + 1):
        omega[nd + (l - 1) * K : nd + l * K] = (tightness / l**prior.decay) ** 2 / s.psi
    d = K + 2.0
    Psi = np.diag(s.psi) * (d - K - 1)  # so that E[Sigma] = diag(psi)
    Yd, Xd = [], []
    if soc is not None:
        Yd.append(np.diag(s.ybar * s.mean) / soc)
        Xd.append(np.hstack([np.zeros((K, nd)), np.tile(np.diag(s.ybar * s.mean) / soc, s.p)]))
    if dio is not None:
        Yd.append((s.ybar * s.mean)[None, :] / dio)
        det = np.zeros((1, nd))
        if "const" in s.d_names:
            det[0, s.d_names.index("const")] = 1 / dio
        Xd.append(np.hstack([det, np.tile(s.ybar * s.mean, s.p)[None, :] / dio]))
    dummies = (np.vstack(Yd), np.vstack(Xd)) if Yd else None
    return b, omega, Psi, d, dummies


def _posterior(s: _Setup, prior: Minnesota, tightness: float, soc: float | None, dio: float | None) -> _Posterior:
    b, omega, Psi, d, dummies = _prior(s, prior, tightness, soc, dio)
    if dummies is None:
        return _niw(s.Y, s.X, b, omega, Psi, d)
    Yd, Xd = dummies
    post = _niw(np.vstack([Yd, s.Y]), np.vstack([Xd, s.X]), b, omega, Psi, d)
    only = _niw(Yd, Xd, b, omega, Psi, d)  # the dummies are prior, not data: p(Y) = p(Y, Yd) / p(Yd)
    return replace(post, logml=post.logml - only.logml)


def _choose(s: _Setup, prior: Minnesota) -> tuple[dict[str, float], _Posterior]:
    names = ["tightness"] + [n for n in ("sum_of_coefficients", "initial_observation") if getattr(prior, n) is not None]
    start = np.log([getattr(prior, n) for n in names])
    bounds = [(np.log(1e-4), np.log(5.0))] + [(np.log(1e-4), np.log(50.0))] * (len(names) - 1)

    def unpack(x: np.ndarray) -> dict[str, float]:
        return dict(zip(names, np.exp(x)))

    def negative(x: np.ndarray) -> float:
        h = unpack(x)
        try:
            return -_posterior(s, prior, h["tightness"], h.get("sum_of_coefficients"), h.get("initial_observation")).logml
        except (np.linalg.LinAlgError, linalg.LinAlgError, ValueError):
            return np.inf

    found = optimize.minimize(negative, start, method="L-BFGS-B", bounds=bounds,
                              options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1000})
    h = unpack(found.x)
    return h, _posterior(s, prior, h["tightness"], h.get("sum_of_coefficients"), h.get("initial_observation"))


def _independent_prior(s: _Setup, prior: NormalWishart) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, Any]:
    """The independent normal-Wishart prior: mean b, variances V (k, K), Psi, d and the dummy observations."""
    b, omega, Psi, d, dummies = _prior(s, prior, prior.tightness, prior.sum_of_coefficients, prior.initial_observation)
    K = len(s.psi)
    nd = len(s.d_names)
    V = np.repeat(omega[:, None], K, axis=1)  # the conjugate omega_j = (lambda / l^decay)^2 / psi_j, times psi_i:
    V[nd:] *= s.psi[None, :]  # (lambda / l^decay)^2 psi_i / psi_j, and theta^2 off the diagonal
    off = np.ones((K, K)) * prior.cross_tightness**2
    np.fill_diagonal(off, 1.0)
    V[nd:] *= np.tile(off, (s.p, 1))
    V[:nd] = prior.deterministic
    return b, V, Psi, d, dummies


_SEARCH_DRAWS, _SEARCH_BURN = 500, 200  # each evaluation of the hyperparameter search for the independent prior


def _fit_independent(spec: VAR, prior: NormalWishart, method: Any, s: _Setup) -> tuple[_Posterior, np.ndarray, np.ndarray]:
    from .bvar_hierarchical import chib_log_ml

    b, V, Psi, d, dummies = _independent_prior(s, prior)
    Y, X = (np.vstack([dummies[0], s.Y]), np.vstack([dummies[1], s.X])) if dummies is not None else (s.Y, s.X)
    rng = make_rng(method.seed)
    Bs, Ss = _gibbs(Y, X, b, V, Psi, d, int(method.draws), int(prior.burn), rng)
    B = Bs.mean(axis=0)
    Omega = np.cov(Bs.reshape(len(Bs), -1, order="F"), rowvar=False)  # the posterior covariance of vec(B), as drawn
    logml = chib_log_ml(s.Y, s.X, b, V, Psi, d, int(method.draws), int(prior.burn), rng, dummies, Bs, Ss)
    return _Posterior(B, Omega, Ss.mean(axis=0), np.nan, logml), Bs, Ss


def fit_bayes(spec: VAR, data: Dataset, span: Any, method: Any, p: int, a: _Arrays) -> FittedBVAR:
    prior = method.prior if method.prior is not None else Minnesota()
    from .bvar_hierarchical import GLP

    if not isinstance(prior, (Minnesota, NormalWishart, GLP)):
        raise TypeError(f"a Bayesian VAR takes a Minnesota, NormalWishart or GLP prior, not {type(prior).__name__}")
    s = _setup(spec, a, p, prior)
    if isinstance(prior, NormalWishart):
        return _independent_result(spec, data, span, method, p, a, s, prior)
    glp: dict[str, Any] | None = None
    if isinstance(prior, GLP):
        from .bvar_hierarchical import fit_glp

        glp = fit_glp(s, prior, int(method.draws), method.seed)
        names = [n if not n.startswith("psi[") else f"psi {spec.variables[int(n[4:-1])]}" for n in glp["space"].names]
        hyper = dict(zip(names, map(float, glp["mode"])))
        post = glp["at_mode"]
    elif prior.optimize:
        hyper, post = _choose(s, prior)
    else:
        hyper = {"tightness": prior.tightness}
        hyper.update({n: getattr(prior, n) for n in ("sum_of_coefficients", "initial_observation") if getattr(prior, n) is not None})
        post = _posterior(s, prior, prior.tightness, prior.sum_of_coefficients, prior.initial_observation)
    K = s.Y.shape[1]
    T = len(s.Y)
    if glp is not None:
        Bs, sigmas = glp["B"], glp["sigma"]
    else:
        rng = make_rng(method.seed)
        n = int(method.draws)
        sigmas = stats.invwishart(df=post.df, scale=post.Psi).rvs(size=n, random_state=rng).reshape(n, K, K)
        chol_O = np.linalg.cholesky(post.Omega)
        Bs = post.B[None] + np.einsum("ij,njk,nlk->nil", chol_O, rng.standard_normal((n, *post.B.shape)), np.linalg.cholesky(sigmas))
    hyper_draws = None
    if glp is not None and "hyper_draws" in glp:  # integrated over the hyperparameters: the point estimates from the draws
        post = _Posterior(Bs.mean(axis=0), np.cov(Bs.reshape(len(Bs), -1, order="F"), rowvar=False), post.Psi, post.df, post.logml)
        hyper_draws = pd.DataFrame(glp["hyper_draws"], columns=list(hyper))
        hyper_draws["log posterior"] = glp["hyper_log_posterior"]
    sigma_mean = post.Psi / (post.df - K - 1) if hyper_draws is None else sigmas.mean(axis=0)
    U = s.Y - s.X @ post.B
    kz = s.X.shape[1]
    sigma_ml = U.T @ U / T
    logdet = np.linalg.slogdet(sigma_ml)[1]
    loglike = -T / 2 * (K * np.log(2 * np.pi) + logdet + K)
    free = K * kz
    criteria = {"aic": logdet + 2 * free / T, "bic": logdet + np.log(T) * free / T,
                "hq": logdet + 2 * np.log(np.log(T)) * free / T, "fpe": ((T + kz) / max(T - kz, 1)) ** K * np.exp(logdet)}
    k = s.X.shape[1]
    est = _Estimate(post.B, U, sigma_mean, sigma_ml, post.Omega if hyper_draws is None else np.full((k, k), np.nan), criteria,
                    float(loglike))
    sampled: dict[str, Any] = {}
    if glp is not None:
        sampled = {"hyperparameter log posterior at the mode": round(float(glp["log_posterior"]), 6)}
        if hyper_draws is not None:
            sampled.update({"hyperparameter sampler": "random-walk Metropolis-Hastings on the logs", "burn": int(prior.burn),
                            "acceptance": round(float(glp["acceptance"]), 4), "scale": round(float(glp["scale"]), 4)})
    prov = Provenance(
        data_vintages={v: data.variables[v].vintage for v in spec.variables if v in data.variables},
        spec={**to_jsonable(spec), "lags": p},
        method={**to_jsonable(method), "hyperparameters": hyper, **sampled},
        sample=(str(a.index[0]), str(a.index[-1])),
        nobs=len(a.index), dropped=0, package_version=__version__, seed=method.seed,
        data_fingerprint=data_fingerprint(data, [*spec.variables, *getattr(spec, "exog", ())], a.index, p),
        resolved={"lags": p, "chosen by": spec.lags} if isinstance(spec.lags, str) else None,
        **_sample_fields(data, span),
    )
    from .var import Cholesky

    return FittedBVAR(spec, p, est, a, Cholesky(), prov, method=method, hyperparameters=hyper, log_ml=post.logml,
                      posterior=post, draws_B=Bs, draws_sigma=sigmas, hyperparameter_draws=hyper_draws)


def _independent_result(spec: VAR, data: Dataset, span: Any, method: Any, p: int, a: _Arrays, s: _Setup,
                        prior: NormalWishart) -> FittedBVAR:
    searched: dict[str, Any] = {}
    if prior.optimize:
        from .bvar_hierarchical import independent_search

        found, value = independent_search(s, prior, method.seed, min(int(method.draws), _SEARCH_DRAWS), min(int(prior.burn), _SEARCH_BURN))
        prior = replace(prior, **found)
        searched = {"search": f"Nelder-Mead on Chib's log marginal likelihood ({min(int(method.draws), _SEARCH_DRAWS)} draws an "
                              f"evaluation, common random numbers); best {value:.4f}"}
    post, Bs, Ss = _fit_independent(spec, prior, method, s)
    K, T = s.Y.shape[1], len(s.Y)
    U = s.Y - s.X @ post.B
    kz = s.X.shape[1]
    sigma_ml = U.T @ U / T
    logdet = np.linalg.slogdet(sigma_ml)[1]
    free = K * kz
    criteria = {"aic": logdet + 2 * free / T, "bic": logdet + np.log(T) * free / T,
                "hq": logdet + 2 * np.log(np.log(T)) * free / T, "fpe": ((T + kz) / max(T - kz, 1)) ** K * np.exp(logdet)}
    k = s.X.shape[1]
    # the posterior covariance of the coefficients is not Sigma (x) Omega here: params and summary use post.Omega,
    # the draws' covariance of vec(B), and ZZinv is left undefined
    est = _Estimate(post.B, U, post.Psi, sigma_ml, np.full((k, k), np.nan), criteria, float(-T / 2 * (K * np.log(2 * np.pi) + logdet + K)))
    hyper = {"tightness": prior.tightness, "cross tightness": prior.cross_tightness}
    hyper.update({n: getattr(prior, n) for n in ("sum_of_coefficients", "initial_observation") if getattr(prior, n) is not None})
    prov = Provenance(
        data_vintages={v: data.variables[v].vintage for v in spec.variables if v in data.variables},
        spec={**to_jsonable(spec), "lags": p},
        method={**to_jsonable(method), "hyperparameters": hyper, "log marginal likelihood": "Chib (1995), from the Gibbs draws",
                **searched},
        sample=(str(a.index[0]), str(a.index[-1])),
        nobs=len(a.index), dropped=0, package_version=__version__, seed=method.seed,
        data_fingerprint=data_fingerprint(data, [*spec.variables, *getattr(spec, "exog", ())], a.index, p),
        resolved={"lags": p, "chosen by": spec.lags} if isinstance(spec.lags, str) else None,
        **_sample_fields(data, span),
    )
    from .var import Cholesky

    return FittedBVAR(spec, p, est, a, Cholesky(), prov, method=method, hyperparameters=hyper, log_ml=post.logml,
                      posterior=post, draws_B=Bs, draws_sigma=Ss)


# ---------------------------------------------------------------------------------------------
# results


@dataclass(frozen=True, eq=False)
class FittedBVAR(FittedVAR):
    """A Bayesian VAR: the posterior mean as the point estimate, with posterior draws for bands."""

    method: Any = None
    hyperparameters: Mapping[str, float] = field(default_factory=dict)
    log_ml: float = np.nan
    posterior: Any = field(default=None, repr=False)
    draws_B: np.ndarray | None = field(default=None, repr=False)
    draws_sigma: np.ndarray | None = field(default=None, repr=False)
    hyperparameter_draws: pd.DataFrame | None = field(default=None, repr=False)  # GLP(mcmc=True): one row per kept draw

    @property
    def stats(self) -> dict[str, float]:
        return {**super().stats, "log_ml": self.log_ml, "draws": float(len(self.draws_B)) if self.draws_B is not None else 0.0}

    def _draw_coefs(self, n: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
        assert self.draws_B is not None and self.draws_sigma is not None
        d, K = len(self.arrays.d_names), len(self.variables)
        count = len(self.draws_B) if not n else min(n, len(self.draws_B))
        return [(_lag_matrices(self.draws_B[i], d, K, self.lags), self.draws_sigma[i]) for i in range(count)]

    @property
    def independent(self) -> bool:
        """Whether the prior is the independent normal-Wishart, sampled by Gibbs."""
        return self.method is not None and isinstance(self.method.prior, NormalWishart)

    @property
    def from_draws(self) -> bool:
        """Whether the point estimates are the draws' means and covariance (the independent prior, or GLP's prior
        integrated over its hyperparameters) rather than a conjugate posterior's."""
        return self.independent or self.hyperparameter_draws is not None

    @property
    def params(self) -> Any:
        if not self.from_draws:
            return super().params
        from .estimation.params import point

        names = [f"{eq}: {reg}" for eq in self.variables for reg in self.regressors]
        return point(names, self.estimate.B.T.reshape(-1), self.posterior.Omega)

    def explosive_share(self) -> float:
        """Share of posterior draws whose companion matrix has a root on or outside the unit circle."""
        K, p = len(self.variables), self.lags
        count = 0
        draws = self._draw_coefs()
        for A, _ in draws:
            companion = np.zeros((K * p, K * p))
            companion[:K, :] = np.hstack(list(A))
            companion[K:, :-K] = np.eye(K * (p - 1))
            count += np.max(np.abs(np.linalg.eigvals(companion))) >= 1
        return count / max(len(draws), 1)

    def summary(self) -> Table:
        e = self.estimate
        if self.from_draws:
            sd = np.sqrt(np.diag(self.posterior.Omega)).reshape(e.B.shape, order="F")
        else:
            sd = np.sqrt(np.outer(np.diag(e.ZZinv), np.diag(e.sigma)))
        rows = [[v for j in range(len(self.variables)) for v in (e.B[i, j], sd[i, j])] for i in range(len(self.regressors))]
        columns = [c for v in self.variables for c in (v, "(sd)")]
        prior = self.method.prior if self.method is not None and self.method.prior is not None else Minnesota()
        hyper = ", ".join(f"{k.replace('_', ' ')} {v:.4g}" for k, v in self.hyperparameters.items())
        notes = [f"Sample: {self.provenance.sample[0]} to {self.provenance.sample[1]}, {self.provenance.nobs} observations"] if self.provenance else []
        from .bvar_hierarchical import GLP

        if self.independent:
            notes += [f"Prior: Minnesota, independent normal-Wishart, own first lag mean {prior.own_mean}, lag decay {prior.decay:g}; "
                      f"{hyper}{' (chosen by marginal likelihood)' if prior.optimize else ''}; Gibbs sampling, {prior.burn} draws discarded",
                      f"Posterior mean and standard deviation, from the draws; log marginal likelihood {self.log_ml:.3f} "
                      "(Chib's estimate from the Gibbs draws)"]
        elif isinstance(prior, GLP):
            how = (f"integrated over by Metropolis-Hastings ({prior.burn} draws discarded, acceptance "
                   f"{self.provenance.method.get('acceptance', float('nan')):.2f}); values shown at the mode"
                   if self.hyperparameter_draws is not None else "at their posterior mode")
            notes += [f"Prior: hierarchical Minnesota (Giannone, Lenza and Primiceri), own first lag mean {prior.own_mean}, lag decay "
                      f"{prior.decay:g}; hyperparameters {how}: {hyper}",
                      f"Posterior mean and standard deviation{', from the draws' if self.hyperparameter_draws is not None else ''}; "
                      f"log marginal likelihood at the mode {self.log_ml:.3f}"]
        else:
            notes += [f"Prior: Minnesota (normal-inverse-Wishart), own first lag mean {prior.own_mean}, lag decay {prior.decay:g}; "
                      f"{hyper}{' (chosen by marginal likelihood)' if prior.optimize else ''}",
                      f"Posterior mean and standard deviation; log marginal likelihood {self.log_ml:.3f}"]
        notes += [f"{len(self.draws_B) if self.draws_B is not None else 0} posterior draws, {self.explosive_share():.1%} explosive; "
                  f"posterior mean {'stable' if self.is_stable else 'NOT stable'} (largest root {self.roots[0]:.4f})"]
        return Table(columns, rows, index=self.regressors, title=f"Bayesian VAR({self.lags}): {', '.join(self.variables)}", notes=notes)

    def __repr__(self) -> str:
        return f"<FittedBVAR BVAR({self.lags}) in {', '.join(self.variables)}; {self.provenance.nobs if self.provenance else '?'} observations>"

    @api()
    def irf(self, periods: int = 20, orthogonal: bool = True, cumulative: bool = False, draws: int | None = None,
            alpha: float = 0.05, seed: int = 0) -> IRF:
        """Posterior median responses with ``1 - alpha`` credible bands, from the posterior draws (all, or the
        first ``draws``; 0 for the responses at the posterior mean without bands). Under sign restrictions, one
        accepted rotation for each posterior draw; with zero restrictions too, ARW's draws, importance-weighted and
        resampled (see ``pymetrix.zerosign``)."""
        from .var import SignRestrictions

        if orthogonal and isinstance(self.identification, SignRestrictions):
            return self._sign_irf(None if draws == 0 else self._draw_coefs(draws), periods, cumulative, alpha, seed,
                                  _scheme_label(self.identification) + ", posterior")
        if draws == 0:
            values = self._responses(self.coefs, self.estimate.sigma, periods, orthogonal, cumulative)
            lower = upper = None
        else:
            sims = np.stack([self._responses(A, S, periods, orthogonal, cumulative) for A, S in self._draw_coefs(draws)])
            values = np.median(sims, axis=0)
            lower, upper = np.quantile(sims, [alpha / 2, 1 - alpha / 2], axis=0)
        label = (_scheme_label(self.identification) if orthogonal else "reduced-form (unit) shocks") + ", posterior median"
        return IRF(values, self.variables, self.variables, lower, upper, cumulative, label, 1 - alpha)

    def _hd_draws(self, draws: int, seed: int) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], str]:
        """(A, Bd, P) for the first ``draws`` posterior draws: under sign (or zero and sign) restrictions, a rotation
        accepted for each draw (``irf``'s rule); otherwise the identification's impact matrix at the draw."""
        from .var import SignRestrictions, _impact, _rotations_for_draws

        scheme = self.identification
        if isinstance(scheme, SignRestrictions) and scheme.narrative:  # narratives need each draw's residuals: the set at the mean
            return super()._hd_draws(draws, seed)
        pairs = self._draw_coefs(draws)
        d = len(self.arrays.d_names)
        Bds = self.draws_B[: len(pairs), :d]  # type: ignore[index]
        if isinstance(scheme, SignRestrictions):
            chosen = _rotations_for_draws(scheme, pairs, self.variables, make_rng(seed))
            return [(pairs[i][0], Bds[i], B) for i, B in chosen], "posterior draws, each with a rotation accepted for it"
        return [(A, Bds[i], _impact(scheme, S, A, self.variables)) for i, (A, S) in enumerate(pairs)], "posterior draws"

    @api()
    def fevd(self, periods: int = 20) -> Any:
        """Variance decomposition at the posterior mean."""
        return super().fevd(periods)

    @api()
    def forecast(self, data: Dataset, span: tuple[Any, Any], alpha: float = 0.05, dynamic: bool = True, draws: int | None = None,
                 seed: int = 0) -> Dataset:
        """Forecasts from the predictive distribution: each posterior draw simulates a path with its own shocks,
        so the intervals include parameter uncertainty. Columns: mean, sd, median and the ``1 - alpha`` interval."""
        from .data import Dataset
        from .metadata import DatasetMeta

        if not dynamic:
            raise SpecificationError("VAR forecasts are dynamic (multi-step); use dynamic=True")
        periods, paths = self.predictive_draws(data, span, draws, seed)
        return self._summarised(data, periods, paths, alpha, DatasetMeta(
            title=f"Bayesian VAR({self.lags}) forecast",
            description=f"Predictive mean and {1 - alpha:.0%} interval from {len(paths)} posterior draws, including parameter uncertainty."))

    @api()
    def conditional_forecast(self, data: Dataset, span: tuple[Any, Any], conditions: Mapping[str, Any], alpha: float = 0.05,
                             draws: int | None = None, seed: int = 0, shocks: Sequence[str] | None = None,
                             shock_conditions: Mapping[str, Any] | None = None, update_posterior: bool = False) -> Dataset:
        """Forecasts given conditions on some variables, from the predictive distribution: for each posterior draw, a
        path that meets the conditions (Waggoner and Zha 1999 for exact ones; a truncated-normal Gibbs sampler for
        ranges), with the shocks the conditions leave free drawn at random, so the intervals include parameter
        uncertainty. Conditions, ``shocks`` and ``shock_conditions`` as for ``FittedVAR.conditional_forecast``; the
        impact matrix of the identification is recomputed for each draw.

        By default the posterior itself is not updated by the conditions (the usual shortcut): each posterior draw
        is conditioned in turn. ``update_posterior=True`` runs Waggoner and Zha's Gibbs sampler instead, which
        alternates a path given the parameters and the conditions with the parameters given the data and that path
        (the path treated as extra observations under the same prior and hyperparameters), so the conditions also
        inform the parameters; it needs every shock free to move (``shocks=None``, no ``shock_conditions``), keeps
        ``draws`` iterations (default: as many as the posterior draws) after 200 of burn-in, and is slower."""
        from . import var_conditions as vc
        from .metadata import DatasetMeta
        from .var import SignRestrictions, _conditional_moments, _impact, _point_path

        periods, D, history = self._forecast_inputs(data, span)
        d, K, H = len(self.arrays.d_names), len(self.variables), len(periods)
        structural = shocks is not None or bool(shock_conditions)
        names = self.shocks if structural else list(self.variables)
        conds = vc.parse(conditions, self.variables, periods) + vc.parse(shock_conditions, names, periods, "shock", on_shock=True)
        vc.check(conds)
        if structural and isinstance(self.identification, SignRestrictions):
            raise SpecificationError("scenarios with chosen shocks from a Bayesian VAR identified by sign restrictions are not "
                                     "available yet: identify it another way, or use a classical VAR")
        if update_posterior and structural:
            raise SpecificationError("update_posterior=True needs every shock free to move: leave out shocks= and shock_conditions=")
        info = vc.record(conds, periods, self.variables, names, shocks)
        how = vc.summary(conds, self.variables, names) + (f", met by the shocks {', '.join(shocks)}" if shocks is not None else "")
        rng = make_rng(seed)
        if update_posterior:
            paths = self._updated_paths(conds, D, history, H, draws, rng)
            info["posterior"] = "updated by the conditions (Waggoner-Zha Gibbs sampler)"
            return self._summarised(data, periods, paths, alpha, DatasetMeta(
                title=f"Bayesian VAR({self.lags}) conditional forecast",
                description=f"Given {how}, with the posterior updated by the conditions (Waggoner-Zha Gibbs sampler): "
                            f"predictive mean and {1 - alpha:.0%} interval from {len(paths)} draws after burn-in.", extra=info))
        coefs = self._draw_coefs(draws)
        Bd_all = self.draws_B[: len(coefs), :d]  # type: ignore[index]
        paths = np.empty((len(coefs), H, K))
        if not structural and all(c.exact for c in conds):  # Waggoner-Zha, draw by draw (as before ranges and shocks=)
            exact = [(c.i, c.k, c.value) for c in conds]
            for n, (A, S) in enumerate(coefs):
                point = _point_path(A, Bd_all[n], D, history, H)
                mean, _, M, free = _conditional_moments(A, S, point, exact)
                paths[n] = mean + (M @ (free @ rng.standard_normal(H * K))).reshape(H, K)
        else:
            driving = vc.driving_mask(shocks, names, H)
            parts = []
            for n, (A, S) in enumerate(coefs):
                point = _point_path(A, Bd_all[n], D, history, H)
                impact = _impact(self.identification, S, A, self.variables) if structural else np.linalg.cholesky(S)
                parts.append(vc.prepare(A, impact, point, conds, driving, 1, rng))
            joint = vc.Prepared(np.vstack([p.y0 for p in parts]), np.vstack([p.c for p in parts]), np.stack([p.W for p in parts]),
                                np.stack([p.S for p in parts]), parts[0].lower, parts[0].upper)
            if joint.c.shape[1]:
                z = vc.box_gibbs(joint.c, joint.S, joint.lower, joint.upper, rng)
                flat = joint.y0 + np.einsum("nij,nj->ni", joint.W, z - joint.c)
            else:
                flat = joint.y0
            paths[:] = flat.reshape(len(coefs), H, K)
        return self._summarised(data, periods, paths, alpha, DatasetMeta(
            title=f"Bayesian VAR({self.lags}) conditional forecast",
            description=f"Given {how} (draw by draw): predictive mean and {1 - alpha:.0%} interval from "
                        f"{len(paths)} posterior draws, including parameter uncertainty. The posterior is not updated by "
                        "the conditions.", extra=info))

    def _updated_paths(self, conds: Any, D: Any, history: Any, H: int, draws: int | None, rng: np.random.Generator,
                       burn: int = 200) -> np.ndarray:
        """Waggoner and Zha's (1999) Gibbs sampler: alternately a path given the parameters and the conditions, and the
        parameters given the data and the path (appended as observations), under the fitted prior."""
        from . import var_conditions as vc
        from .bvar_hierarchical import GLP
        from .var import _point_path

        prior = self.method.prior if self.method is not None and self.method.prior is not None else Minnesota()
        s = _setup(self.spec, self.arrays, self.lags, prior)
        h = self.hyperparameters
        if isinstance(prior, NormalWishart):  # at the hyperparameters fitted (those the search chose, with optimize=True)
            prior = replace(prior, tightness=h["tightness"], cross_tightness=h["cross tightness"],
                            **{n: h[n] for n in ("sum_of_coefficients", "initial_observation") if n in h})
            b, V, Psi, dof, dummies = _independent_prior(s, prior)
        else:
            if isinstance(prior, GLP):  # the prior scales psi are estimated too
                s = replace(s, psi=np.array([h[f"psi {v}"] for v in self.variables]))
            b, omega, Psi, dof, dummies = _prior(s, prior, h["tightness"], h.get("sum_of_coefficients"), h.get("initial_observation"))
        Y0, X0 = (np.vstack([dummies[0], s.Y]), np.vstack([dummies[1], s.X])) if dummies is not None else (s.Y, s.X)
        d, K, p = len(self.arrays.d_names), len(self.variables), self.lags
        n = int(draws) if draws else len(self.draws_B)  # type: ignore[arg-type]
        B, Sigma = self.draws_B[0].copy(), self.draws_sigma[0].copy()  # type: ignore[index]
        driving = np.ones(H * K, bool)
        start = None
        Dm = np.asarray(D, float).reshape(H, d)
        out = np.empty((n, H, K))
        for it in range(burn + n):
            A = _lag_matrices(B, d, K, p)
            point = _point_path(A, B[:d], D, history, H)
            prepared = vc.prepare(A, np.linalg.cholesky(Sigma), point, conds, driving, 1, rng)
            flat, z = vc.draw_paths(prepared, rng, vc.UPDATE_SWEEPS if start is not None else vc.SWEEPS, start)
            start = z if z.shape[1] else None
            Yf = flat.reshape(H, K)
            hist = np.vstack([np.asarray(history, float), Yf])  # y[t-p] ... y[t+H-1]
            Xf = np.column_stack([Dm, *[hist[p - l: p - l + H] for l in range(1, p + 1)]])
            Y, X = np.vstack([Y0, Yf]), np.vstack([X0, Xf])
            if isinstance(prior, NormalWishart):
                B, Sigma = _gibbs_step(Y, X, b, V, Psi, dof, B, rng)
            else:
                post = _niw(Y, X, b, omega, Psi, dof)
                Sigma = stats.invwishart(df=post.df, scale=post.Psi).rvs(random_state=rng).reshape(K, K)
                B = post.B + np.linalg.cholesky(post.Omega) @ rng.standard_normal(post.B.shape) @ np.linalg.cholesky(Sigma).T
            if it >= burn:
                out[it - burn] = Yf
        return out

    def predictive_draws(self, data: Dataset, span: tuple[Any, Any], draws: int | None = None, seed: int = 0) -> tuple[pd.PeriodIndex, np.ndarray]:
        """Simulated forecast paths, one per posterior draw with its own shocks: the periods, and an array
        (draws, periods, variables) in the order of ``variables``. ``forecast`` summarises them; backtests score them."""
        periods, D, history = self._forecast_inputs(data, span)
        d, K = len(self.arrays.d_names), len(self.variables)
        rng = make_rng(seed)
        coefs = self._draw_coefs(draws)
        Bd_all = self.draws_B[: len(coefs), :d]  # type: ignore[index]
        paths = np.empty((len(coefs), len(periods), K))
        for n, (A, S) in enumerate(coefs):
            hist = list(history)
            shocks = rng.standard_normal((len(periods), K)) @ np.linalg.cholesky(S).T
            for i in range(len(periods)):
                y = np.asarray(D[i]) @ Bd_all[n] if d else np.zeros(K)
                for l in range(1, self.lags + 1):
                    y = y + A[l - 1] @ hist[-l]
                y = y + shocks[i]
                paths[n, i] = y
                hist.append(y)
        return periods, paths
