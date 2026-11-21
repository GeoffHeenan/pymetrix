"""Hierarchical priors for Bayesian VARs, and the marginal likelihood of the independent normal-Wishart prior.

    VAR(["y", "pi", "r"], lags=4).fit(data, method=Bayes(prior=GLP()))             # hyperparameters at their posterior mode
    VAR(["y", "pi", "r"], lags=4).fit(data, method=Bayes(prior=GLP(mcmc=True)))    # and integrated over, by MCMC
    VAR(...).fit(data, method=Bayes(prior=NormalWishart(optimize=True)))            # Chib's marginal likelihood, maximised

**GLP** (Giannone, Lenza and Primiceri 2015, "Prior selection for vector autoregressions"). The conjugate Minnesota
prior (``bvar.Minnesota``) with the sum-of-coefficients and single-unit-root dummy observations, whose
hyperparameters are themselves given priors and estimated: the overall tightness lambda, the sum-of-coefficients mu
and the single-unit-root delta each have a gamma hyperprior (by mode and standard deviation: 0.2 and 0.4 for lambda,
1 and 1 for mu and delta, as the authors'), and the diagonal psi of the prior scale of Sigma (which also scales the
Minnesota variances, ``lambda^2 / (l^decay)^2 / psi_j``) an inverse gamma with shape and scale 0.02^2 each. The
hyperparameters' posterior is the marginal likelihood (closed form, the same as ``bvar._niw``'s) times the hyperpriors.
GLP's lag decay is ``l^-2`` on the variances: ``decay=1`` here. Search bounds are the authors': lambda in [1e-4, 5],
mu and delta in [1e-4, 50], psi_j within a factor 100 of the residual variance of an AR(1) of variable j.

- ``mcmc=False``: the hyperparameters at their posterior mode (as GLP's maximisation), and exact draws of the
  coefficients and Sigma given them.
- ``mcmc=True``: random-walk Metropolis-Hastings over the hyperparameters (on their logs, proposal from the inverse
  Hessian at the mode, its scale tuned during the burn-in towards an acceptance of a quarter; GLP aim at 20 to 30%),
  and for each kept hyperparameter draw a draw of the coefficients and Sigma from their conjugate posterior given it.
  Forecasts, impulse responses and their bands then integrate over the hyperparameters.

**Chib's marginal likelihood for the independent normal-Wishart prior** (Chib 1995). With two Gibbs blocks, B and
Sigma, the basic marginal likelihood identity at a point (B*, Sigma*) of high posterior density reads
``log p(Y) = log p(Y | B*, Sigma*) + log p(B*) + log p(Sigma*) - log p(Sigma* | Y) - log p(B* | Sigma*, Y)``. The last
term is exact (B given Sigma is normal); ``p(Sigma* | Y)`` is the average over the Gibbs draws of B of the
inverse-Wishart density of Sigma* given B (Rao-Blackwellised). Two blocks need no reduced run. Dummy observations
(sum-of-coefficients, single-unit-root) are part of the prior, so ``p(Y) = p(Y, Yd) / p(Yd)``: a second Gibbs run, on
the dummies alone, gives ``p(Yd)`` the same way. With ``optimize=True`` the tightness, cross-tightness and any dummy
hyperparameters maximise this estimate (Nelder-Mead on their logs, each evaluation a Gibbs run with the same random
numbers, so the search sees a smooth surface; the search's runs are shorter than the final one).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

import numpy as np
from scipy import linalg, optimize, special

from pymetrix._lazy import stats

from ._random import generator as make_rng
from .errors import SpecificationError
from .forms import Widget

if TYPE_CHECKING:
    from .bvar import _Setup

__all__ = ["GLP", "chib_log_ml", "gamma_shape_scale", "glp_log_ml"]


@dataclass(frozen=True)
class GLP:
    """The hierarchical Minnesota prior of Giannone, Lenza and Primiceri (2015): hyperpriors on the tightness, the
    sum-of-coefficients and single-unit-root hyperparameters and the prior scales psi, which are estimated at their
    posterior mode or integrated over by MCMC."""

    label: ClassVar[str] = "Hierarchical Minnesota prior (Giannone, Lenza and Primiceri)"
    topics: ClassVar[tuple[str, ...]] = ("bayesian-var",)

    mcmc: Annotated[bool, Widget("checkbox", help="integrate over the hyperparameters by MCMC (else: at their posterior mode)")] = False
    sum_of_coefficients: Annotated[bool, Widget("checkbox", help="the sum-of-coefficients prior (mu), with its hyperprior")] = True
    initial_observation: Annotated[bool, Widget("checkbox", help="the single-unit-root prior (delta), with its hyperprior")] = True
    own_mean: Annotated[float | Mapping[str, float], Widget("number", help="prior mean of each variable's own first lag: 1 for levels, 0 for growth rates")] = 1.0
    decay: Annotated[float, Widget("number", min=0, max=4, advanced=True, help="lag l's prior standard deviation shrinks as 1/l**decay (GLP: 1)")] = 1.0
    tightness_mode: Annotated[float, Widget("number", min=1e-3, max=5, advanced=True, help="mode of the gamma hyperprior on lambda")] = 0.2
    tightness_sd: Annotated[float, Widget("number", min=1e-3, max=10, advanced=True, help="standard deviation of the gamma hyperprior on lambda")] = 0.4
    soc_mode: Annotated[float, Widget("number", min=1e-3, max=50, advanced=True, help="mode of the gamma hyperprior on mu")] = 1.0
    soc_sd: Annotated[float, Widget("number", min=1e-3, max=50, advanced=True, help="standard deviation of the gamma hyperprior on mu")] = 1.0
    dio_mode: Annotated[float, Widget("number", min=1e-3, max=50, advanced=True, help="mode of the gamma hyperprior on delta")] = 1.0
    dio_sd: Annotated[float, Widget("number", min=1e-3, max=50, advanced=True, help="standard deviation of the gamma hyperprior on delta")] = 1.0
    psi_shape: Annotated[float, Widget("number", min=0, advanced=True, help="shape of the inverse-gamma hyperprior on each psi")] = 0.02**2
    psi_scale: Annotated[float, Widget("number", min=0, advanced=True, help="scale of the inverse-gamma hyperprior on each psi")] = 0.02**2
    burn: Annotated[int, Widget("number", min=0, max=1_000_000, advanced=True, help="MCMC draws of the hyperparameters discarded first")] = 2000
    deterministic: Annotated[float, Widget("number", advanced=True, help="prior variance of the constant, trend and exogenous terms (loose)")] = 1e6
    scale_lags: Annotated[int, Widget("number", min=1, max=12, advanced=True, help="lags of the autoregressions that centre and bound psi")] = 1

    def __post_init__(self) -> None:
        for name in ("tightness_mode", "tightness_sd", "soc_mode", "soc_sd", "dio_mode", "dio_sd"):
            if not getattr(self, name) > 0:
                raise SpecificationError(f"{name} must be positive")
        if self.psi_shape < 0 or self.psi_scale < 0:
            raise SpecificationError("psi_shape and psi_scale must be zero or more (zero: a flat hyperprior on log psi)")
        if self.burn < 0:
            raise SpecificationError("burn must be zero or more")

    @property
    def optimize(self) -> bool:
        return True


def gamma_shape_scale(mode: float, sd: float) -> tuple[float, float]:
    """The gamma distribution's shape and scale with this mode and standard deviation (GLP's ``GammaCoef``)."""
    r = mode**2 / sd**2
    k = (2 + r + np.sqrt((4 + r) * r)) / 2
    return float(k), float(np.sqrt(sd**2 / k))


def glp_log_ml(Y: np.ndarray, X: np.ndarray, b: np.ndarray, omega: np.ndarray, psi: np.ndarray, d: float) -> float:
    """The log marginal likelihood as GLP's code computes it (``logMLVAR_formin``), by eigenvalues: ``Y = X B + E``,
    ``B | Sigma ~ N(b, Sigma (x) diag(omega))``, ``Sigma ~ IW(diag(psi), d)``. The same number as ``bvar._niw``'s
    with ``Psi = diag(psi)``."""
    T, n = Y.shape
    so = np.sqrt(omega)
    XX = X.T @ X
    A = XX + np.diag(1.0 / omega)
    Bh = linalg.solve(A, X.T @ Y + b / omega[:, None], assume_a="pos")
    E = Y - X @ Bh
    D = Bh - b
    aaa = so[:, None] * XX * so[None, :]
    sp = 1.0 / np.sqrt(psi)
    bbb = sp[:, None] * (E.T @ E + D.T @ (D / omega[:, None])) * sp[None, :]
    ea = np.linalg.eigvalsh((aaa + aaa.T) / 2)
    eb = np.linalg.eigvalsh((bbb + bbb.T) / 2)
    ea = np.where(ea < 1e-12, 0.0, ea) + 1
    eb = np.where(eb < 1e-12, 0.0, eb) + 1
    j = np.arange(n)
    return float(-n * T / 2 * np.log(np.pi) + np.sum(special.gammaln((T + d - j) / 2) - special.gammaln((d - j) / 2))
                 - T / 2 * np.sum(np.log(psi)) - n / 2 * np.sum(np.log(ea)) - (T + d) / 2 * np.sum(np.log(eb)))


# --- GLP -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class _Hyper:
    names: list[str]  # lambda, then mu and delta when used, then psi_1..psi_K
    lows: np.ndarray
    highs: np.ndarray
    k_soc: bool
    k_dio: bool


def _space(s: _Setup, prior: GLP) -> _Hyper:
    K = len(s.psi)
    names = ["tightness"] + (["sum_of_coefficients"] if prior.sum_of_coefficients else []) \
        + (["initial_observation"] if prior.initial_observation else []) + [f"psi[{j}]" for j in range(K)]
    lows = [1e-4] + [1e-4] * (len(names) - K - 1) + list(s.psi / 100)
    highs = [5.0] + [50.0] * (len(names) - K - 1) + list(s.psi * 100)
    return _Hyper(names, np.array(lows), np.array(highs), prior.sum_of_coefficients, prior.initial_observation)


def _unpack(h: np.ndarray, space: _Hyper) -> tuple[float, float | None, float | None, np.ndarray]:
    i = 1
    soc = dio = None
    if space.k_soc:
        soc, i = float(h[i]), i + 1
    if space.k_dio:
        dio, i = float(h[i]), i + 1
    return float(h[0]), soc, dio, np.asarray(h[i:], dtype=float)


def _log_hyperprior(h: np.ndarray, space: _Hyper, prior: GLP) -> float:
    lam, soc, dio, psi = _unpack(h, space)

    def lgamma(x: float, mode: float, sd: float) -> float:
        k, theta = gamma_shape_scale(mode, sd)
        return float((k - 1) * np.log(x) - x / theta - k * np.log(theta) - special.gammaln(k))

    lp = lgamma(lam, prior.tightness_mode, prior.tightness_sd)
    if soc is not None:
        lp += lgamma(soc, prior.soc_mode, prior.soc_sd)
    if dio is not None:
        lp += lgamma(dio, prior.dio_mode, prior.dio_sd)
    a, b = prior.psi_shape, prior.psi_scale
    if a > 0 and b > 0:  # GLP's logIG2pdf
        lp += float(np.sum(a * np.log(b) - (a + 1) * np.log(psi) - b / psi - special.gammaln(a)))
    return lp


def _glp_ml(s: _Setup, prior: GLP, h: np.ndarray, space: _Hyper) -> float:
    """The log marginal likelihood at hyperparameters h, by GLP's formula, the dummies' own normalisation removed."""
    from .bvar import _prior

    lam, soc, dio, psi = _unpack(h, space)
    t = replace(s, psi=psi)
    b, omega, Psi, d, dummies = _prior(t, prior, lam, soc, dio)
    if dummies is None:
        return glp_log_ml(s.Y, s.X, b, omega, psi * (d - len(psi) - 1), d)
    Yd, Xd = dummies
    return (glp_log_ml(np.vstack([Yd, s.Y]), np.vstack([Xd, s.X]), b, omega, psi * (d - len(psi) - 1), d)
            - glp_log_ml(Yd, Xd, b, omega, psi * (d - len(psi) - 1), d))


def _log_post(s: _Setup, prior: GLP, h: np.ndarray, space: _Hyper) -> float:
    if not np.all((h > space.lows) & (h < space.highs)):
        return -np.inf
    try:
        value = _glp_ml(s, prior, h, space) + _log_hyperprior(h, space, prior)
    except (np.linalg.LinAlgError, linalg.LinAlgError, ValueError):
        return -np.inf
    return float(value) if np.isfinite(value) else -np.inf


def glp_mode(s: _Setup, prior: GLP) -> tuple[_Hyper, np.ndarray, float]:
    """The hyperparameters' posterior mode (marginal likelihood times hyperpriors), searched on their logs within the
    bounds: the space, the mode and the log posterior there."""
    space = _space(s, prior)
    start = np.concatenate([[prior.tightness_mode], [prior.soc_mode] * space.k_soc, [prior.dio_mode] * space.k_dio, s.psi])
    start = np.clip(start, space.lows * 1.01, space.highs * 0.99)

    def negative(x: np.ndarray) -> float:
        v = _log_post(s, prior, np.exp(x), space)
        return -v if np.isfinite(v) else 1e300

    bounds = list(zip(np.log(space.lows), np.log(space.highs)))
    best = None
    for x0 in (np.log(start), np.log(np.clip(np.concatenate([[0.5 * prior.tightness_mode], start[1:]]), space.lows * 1.01, space.highs * 0.99))):
        found = optimize.minimize(negative, x0, method="L-BFGS-B", bounds=bounds, options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 2000})
        if best is None or found.fun < best.fun:
            best = found
    assert best is not None
    return space, np.exp(best.x), float(-best.fun)


def glp_mcmc(s: _Setup, prior: GLP, space: _Hyper, mode: np.ndarray, draws: int, rng: np.random.Generator
             ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Random-walk Metropolis-Hastings over the logs of the hyperparameters (the Jacobian included), started at the
    mode; the proposal is the inverse Hessian there, its scale tuned in the burn-in. Returns the kept draws (levels),
    their log posteriors (levels), the acceptance rate after the burn-in and the final scale."""
    from .estimation.numdiff import hessian

    k = len(mode)

    def lp_log(x: np.ndarray) -> float:
        v = _log_post(s, prior, np.exp(x), space)
        return v + float(np.sum(x)) if np.isfinite(v) else -np.inf

    x = np.log(mode)
    try:
        H = hessian(lp_log, x)
        cov = np.linalg.inv(-(H + H.T) / 2)
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        L = np.eye(k) * 0.1
    c = 2.38 / np.sqrt(k)
    lp = lp_log(x)
    out, out_lp = np.empty((draws, k)), np.empty(draws)
    accepted, window = 0, 0
    burn = int(prior.burn)
    for i in range(burn + draws):
        trial = x + c * (L @ rng.standard_normal(k))
        tlp = lp_log(trial)
        ok = bool(np.isfinite(tlp) and np.log(rng.uniform()) < tlp - lp)
        if ok:
            x, lp = trial, tlp
        if i < burn:
            window += ok
            if (i + 1) % 100 == 0:
                c *= float(np.exp(2.0 * (window / 100 - 0.25)))
                window = 0
        else:
            accepted += ok
            out[i - burn] = np.exp(x)
            out_lp[i - burn] = lp - float(np.sum(x))
    return out, out_lp, accepted / max(draws, 1), c


def fit_glp(s: _Setup, prior: GLP, draws: int, seed: Any) -> dict[str, Any]:
    """The GLP posterior: the hyperparameters (mode or MCMC draws) and draws of (B, Sigma) given them."""
    from .bvar import _posterior

    space, mode, top = glp_mode(s, prior)
    rng = make_rng(seed)
    K = s.Y.shape[1]

    def at(h: np.ndarray) -> Any:
        lam, soc, dio, psi = _unpack(h, space)
        return _posterior(replace(s, psi=psi), prior, lam, soc, dio)

    def draw(post: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
        sig = stats.invwishart(df=post.df, scale=post.Psi).rvs(size=n, random_state=rng).reshape(n, K, K)
        cO = np.linalg.cholesky(post.Omega)
        Bs = post.B[None] + np.einsum("ij,njk,nlk->nil", cO, rng.standard_normal((n, *post.B.shape)), np.linalg.cholesky(sig))
        return Bs, sig

    at_mode = at(mode)
    out: dict[str, Any] = {"space": space, "mode": mode, "log_posterior": top, "at_mode": at_mode}
    if not prior.mcmc:
        out["B"], out["sigma"] = draw(at_mode, draws)
        return out
    hs, lps, rate, scale = glp_mcmc(s, prior, space, mode, draws, rng)
    Bs, Ss = np.empty((draws, *at_mode.B.shape)), np.empty((draws, K, K))
    for i, h in enumerate(hs):
        b, sig = draw(at(h), 1)
        Bs[i], Ss[i] = b[0], sig[0]
    out.update(B=Bs, sigma=Ss, hyper_draws=hs, hyper_log_posterior=lps, acceptance=rate, scale=scale)
    return out


# --- Chib's marginal likelihood for the independent normal-Wishart prior --------------------------------------------------

def _gibbs_log_ml(Y: np.ndarray, X: np.ndarray, b: np.ndarray, V: np.ndarray, Psi: np.ndarray, d: float, draws: int, burn: int,
                  rng: np.random.Generator, Bs: np.ndarray | None = None, Ss: np.ndarray | None = None) -> float:
    """Chib's (1995) log marginal likelihood of ``Y = X B + E`` under vec(B) ~ N(vec(b), diag(V)), Sigma ~ IW(Psi, d),
    from Gibbs draws (run here unless given)."""
    from .bvar import _gibbs

    if Bs is None or Ss is None:
        Bs, Ss = _gibbs(Y, X, b, V, Psi, d, draws, burn, rng)
    T, K = Y.shape
    k = X.shape[1]
    B_star = Bs.mean(axis=0)
    S_star = Ss.mean(axis=0)
    S_star = (S_star + S_star.T) / 2
    Si = np.linalg.inv(S_star)
    E = Y - X @ B_star
    logdet_S = np.linalg.slogdet(S_star)[1]
    loglike = -T * K / 2 * np.log(2 * np.pi) - T / 2 * logdet_S - 0.5 * float(np.sum(Si * (E.T @ E)))
    v, bv = V.reshape(-1, order="F"), b.reshape(-1, order="F")
    bs = B_star.reshape(-1, order="F")
    log_prior_B = float(np.sum(stats.norm.logpdf(bs, bv, np.sqrt(v))))
    log_prior_S = float(stats.invwishart.logpdf(S_star, df=d, scale=Psi))
    # p(Sigma* | Y): the average of the inverse-Wishart densities given each draw of B
    dens = np.empty(len(Bs))
    for g, B in enumerate(Bs):
        R = Y - X @ B
        S = Psi + R.T @ R
        dens[g] = stats.invwishart.logpdf(S_star, df=d + T, scale=(S + S.T) / 2)
    top = dens.max()
    log_post_S = float(top + np.log(np.mean(np.exp(dens - top))))
    # p(B* | Sigma*, Y): normal, exact
    P = np.kron(Si, X.T @ X)
    P[np.diag_indices_from(P)] += 1.0 / v
    c = linalg.cho_factor(P)
    mean = linalg.cho_solve(c, bv / v + (X.T @ Y @ Si).reshape(-1, order="F"))
    dev = bs - mean
    log_post_B = float(-k * K / 2 * np.log(2 * np.pi) + np.sum(np.log(np.diag(c[0]))) - 0.5 * dev @ P @ dev)
    return float(loglike + log_prior_B + log_prior_S - log_post_S - log_post_B)


def chib_log_ml(Y: np.ndarray, X: np.ndarray, b: np.ndarray, V: np.ndarray, Psi: np.ndarray, d: float, draws: int, burn: int,
                rng: np.random.Generator, dummies: tuple[np.ndarray, np.ndarray] | None = None,
                Bs: np.ndarray | None = None, Ss: np.ndarray | None = None) -> float:
    """Chib's log marginal likelihood of the data ``Y`` for the independent normal-Wishart prior; ``dummies`` (Yd, Xd)
    are part of the prior: ``log p(Y, Yd) - log p(Yd)``, each by its own Gibbs run. ``Bs``, ``Ss``: Gibbs draws
    already made for (Y with the dummies), reused."""
    if dummies is None:
        return _gibbs_log_ml(Y, X, b, V, Psi, d, draws, burn, rng, Bs, Ss)
    Yd, Xd = dummies
    joint = _gibbs_log_ml(np.vstack([Yd, Y]), np.vstack([Xd, X]), b, V, Psi, d, draws, burn, rng, Bs, Ss)
    return joint - _gibbs_log_ml(Yd, Xd, b, V, Psi, d, draws, burn, rng)


def independent_search(s: _Setup, prior: Any, seed: Any, draws: int, burn: int) -> tuple[dict[str, float], float]:
    """The independent prior's tightness, cross-tightness and dummy hyperparameters that maximise Chib's marginal
    likelihood: Nelder-Mead on their logs, every evaluation with the same random numbers."""
    from .bvar import _independent_prior

    names = ["tightness", "cross_tightness"] + [n for n in ("sum_of_coefficients", "initial_observation") if getattr(prior, n) is not None]
    lows = np.log([1e-4, 1e-3] + [1e-4] * (len(names) - 2))
    highs = np.log([5.0, 1.0] + [50.0] * (len(names) - 2))
    cache: dict[tuple, float] = {}

    def negative(x: np.ndarray) -> float:
        x = np.clip(x, lows, highs)
        key = tuple(np.round(x, 10))
        if key not in cache:
            trial = replace(prior, **dict(zip(names, map(float, np.exp(x)))))
            try:
                b, V, Psi, d, dummies = _independent_prior(s, trial)
                value = chib_log_ml(s.Y, s.X, b, V, Psi, d, draws, burn, make_rng(seed), dummies)
            except (np.linalg.LinAlgError, linalg.LinAlgError, ValueError):
                value = -np.inf
            cache[key] = -value if np.isfinite(value) else 1e300
        return cache[key]

    x0 = np.clip(np.log([getattr(prior, n) for n in names]), lows, highs)
    found = optimize.minimize(negative, x0, method="Nelder-Mead", bounds=list(zip(lows, highs)),
                              options={"xatol": 1e-2, "fatol": 1e-2, "maxfev": 30 * len(names)})
    return dict(zip(names, map(float, np.exp(np.clip(found.x, lows, highs))))), float(-found.fun)
