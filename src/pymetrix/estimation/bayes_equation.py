"""Bayesian estimation of single equations: ``Equation(...).fit(data, method=Bayes(prior={...}))``.

    Equation("c ~ y").fit(data, method=Bayes(prior={"y": Normal(0.8, 0.1), "sigma": InvGamma1(0.5, 1)}))
    Equation("d ~ x", family=Logit()).fit(data, method=Bayes(prior={"x": Normal(0, 2)}, draws=5000, chains=4))

Priors are given by coefficient name (as the fit names them; ``const`` for the intercept), and for a Gaussian
equation on the error's standard deviation (``sigma``, usually ``InvGamma1``) or its variance (``sigma2``, ``InvGamma``).
A coefficient without a prior has a flat one, and with no prior on the error 1/sigma^2 (Jeffreys) is used, so with no
priors at all the posterior mean is least squares. Gaussian equations whose priors are normal (or flat) on the
coefficients and inverse gamma on the error variance are sampled by Gibbs, each step exact: the coefficients given the
variance are normal, the variance given the coefficients inverse gamma (InvGamma1 on sigma is an inverse gamma on
sigma^2). Other priors, and logit, probit and Poisson equations, are sampled by random-walk Metropolis-Hastings, the
proposal shaped by the curvature at the posterior mode and its scale tuned during the burn-in towards a 0.25 acceptance
rate. Each chain has its own random stream; R-hat and effective sample sizes come from the chains (Gelman et al. 2013).

``sampler="nuts"`` samples any of them by the No-U-Turn sampler (``pymetrix.nuts``: Hoffman and Gelman 2014, as Stan
runs it) with analytic gradients of the Gaussian, logit, probit and Poisson log likelihoods and of the priors. It
samples log sigma, and a coefficient whose prior is bounded (Beta, Gamma, the inverse gammas, Uniform) through the
logistic or log of it, with the Jacobian; the chains start near the posterior mode, the mass matrix from the curvature
there, and adapt during the burn-in (Stan's warmup: step size by dual averaging, the diagonal mass matrix in windows).
Divergent transitions after the burn-in are counted and noted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize

from ..errors import SpecificationError
from ..output import Table
from .fitted import FittedEquation
from .params import point
from .._random import generator as make_rng, seed_sequence


@dataclass(frozen=True, eq=False)
class FittedBayesEquation(FittedEquation):
    """A single equation estimated by Bayesian methods: posterior means as the coefficients, the posterior covariance
    as their covariance, and the draws (``draws``: one row per draw, with its chain)."""

    draws: pd.DataFrame | None = field(default=None, repr=False)

    def posterior(self, coverage: float = 0.9) -> pd.DataFrame:
        """Mean, standard deviation, median, highest-posterior-density interval, R-hat, effective sample size and the
        Monte Carlo standard error of the mean (MCSE: s.d. / sqrt(ESS)) of each parameter."""
        from ..models.bayes import hpd, rhat

        d = self.draws
        assert d is not None
        rows = {}
        for name, (x, per, e, se) in _chain_diagnostics(d).items():
            low, high = hpd(x, coverage)
            e = int(round(e)) if np.isfinite(e) else e  # a count of draws: shown whole, as the DSGE posterior shows it
            rows[name] = {"mean": x.mean(), "s.d.": x.std(ddof=1), "median": float(np.median(x)),
                          f"{coverage:.0%} HPD low": low, f"{coverage:.0%} HPD high": high, "R-hat": rhat(per), "ESS": e,
                          "MCSE": se}
        return pd.DataFrame(rows).T

    def summary(self) -> Table:
        post = self.posterior()
        notes = list(self.extra.get("notes", []))
        if self.provenance is not None:
            sample = f"{self.provenance.sample[0]} to {self.provenance.sample[1]}, " if self.provenance.sample else ""
            notes.append(f"Sample: {sample}{self.provenance.nobs} observations ({self.provenance.dropped} dropped)")
        disagree = [n for n, r in post["R-hat"].items() if np.isfinite(r) and r > 1.1]
        if disagree:
            notes.append(f"R-hat above 1.1 for {', '.join(disagree)}: the chains disagree; draw more, or check the priors")
        post["ESS"] = pd.Series([int(v) if np.isfinite(v) else v for v in post["ESS"]], index=post.index, dtype=object)
        return Table.from_frame(post, title=f"Bayesian estimation: {self.equation}", notes=notes)


def _chain_diagnostics(draws: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, float, float]]:
    """For each parameter of ``draws`` (one row per draw, with its chain): its draws, its draws by chain
    (``[chain, draw]``), its effective sample size and the Monte Carlo standard error of its mean."""
    from ..models.bayes import ess, mcse

    chains = draws["chain"].to_numpy()
    out = {}
    for name in [c for c in draws.columns if c != "chain"]:
        x = draws[name].to_numpy()
        per = np.stack([x[chains == c] for c in np.unique(chains)])
        out[name] = (x, per, ess(per), mcse(per))
    return out


def _flat_or_normal(prior: Any) -> bool:
    from .priors import Normal

    return prior is None or isinstance(prior, Normal)


def _ig_shape_scale(prior: Any, key: str) -> tuple[float, float] | None:
    """The inverse gamma (shape, scale) of sigma^2 for a prior on sigma (InvGamma1) or sigma^2 (InvGamma); Jeffreys
    1/sigma^2 is (0, 0); None for any other prior (no exact Gibbs step)."""
    from .priors import InvGamma, InvGamma1

    if prior is None:
        return 0.0, 0.0
    if key == "sigma" and isinstance(prior, InvGamma1):
        nu, s = prior.nu_s  # s / sigma^2 ~ chi-squared(nu): sigma^2 ~ IG(nu / 2, s / 2)
        return nu / 2, s / 2
    if key == "sigma2" and isinstance(prior, InvGamma):
        a = (prior.mean / prior.sd) ** 2 + 2
        return a, prior.mean * (a - 1)
    return None


def _gibbs(y: np.ndarray, X: np.ndarray, w: np.ndarray, priors: list[Any], shape: float, scale: float, draws: int,
           burn: int, rng: np.random.Generator) -> np.ndarray:
    """Gibbs draws of (beta, sigma) for y = X beta + e, e ~ N(0, sigma^2 / w), independent normal (or flat) priors."""
    n, k = X.shape
    prec0 = np.array([0.0 if p is None else 1 / p.sd**2 for p in priors])
    mean0 = np.array([0.0 if p is None else p.mean for p in priors])
    Xw = X * w[:, None]
    XX, Xy = Xw.T @ X, Xw.T @ y
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    out = np.empty((draws, k + 1))
    for i in range(burn + draws):
        e = y - X @ beta
        s2 = (scale + 0.5 * float(e @ (w * e))) / rng.gamma(shape + n / 2)
        P = XX / s2 + np.diag(prec0)
        c = np.linalg.cholesky(P)
        mean = np.linalg.solve(P, Xy / s2 + prec0 * mean0)
        beta = mean + np.linalg.solve(c.T, rng.standard_normal(k))
        if i >= burn:
            out[i - burn, :k], out[i - burn, k] = beta, np.sqrt(s2)
    return out


def _rwmh(logpost: Any, mode: np.ndarray, cov: np.ndarray, draws: int, burn: int,
          rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Random-walk Metropolis-Hastings from the mode, N(0, c^2 cov) steps, c tuned in the burn-in towards a 0.25
    acceptance rate. Returns the draws and the acceptance rate after the burn-in."""
    k = len(mode)
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(k))
    c = 2.38 / np.sqrt(k)
    x, lp = mode.copy(), logpost(mode)
    out = np.empty((draws, k))
    accepted = 0
    for i in range(burn + draws):
        prop = x + c * (L @ rng.standard_normal(k))
        lq = logpost(prop)
        ok = np.log(rng.uniform()) < lq - lp
        if ok:
            x, lp = prop, lq
        if i < burn:
            c *= np.exp((float(ok) - 0.25) / np.sqrt(i + 1))  # Robbins-Monro towards a 0.25 acceptance rate
        else:
            accepted += ok
            out[i - burn] = x
    return out, accepted / draws


def _prior_terms(prior: Any) -> tuple[Any, Any]:
    """A prior's log density and its derivative, as functions of a float, with any constants worked out once."""
    from .priors import Beta, Gamma, InvGamma, InvGamma1, Normal, Uniform

    if isinstance(prior, Normal):
        m, v = float(prior.mean), float(prior.sd) ** 2
        return prior.logpdf, lambda x: -(x - m) / v
    if isinstance(prior, Beta):
        c = prior.mean * (1 - prior.mean) / prior.sd**2 - 1
        a, b = prior.mean * c, (1 - prior.mean) * c
        return prior.logpdf, lambda x: (a - 1) / x - (b - 1) / (1 - x)
    if isinstance(prior, Gamma):
        shape, scale = (prior.mean / prior.sd) ** 2, prior.sd**2 / prior.mean
        return prior.logpdf, lambda x: (shape - 1) / x - 1 / scale
    if isinstance(prior, InvGamma):
        a = (prior.mean / prior.sd) ** 2 + 2
        scale = prior.mean * (a - 1)
        return prior.logpdf, lambda x: -(a + 1) / x + scale / x**2
    if isinstance(prior, InvGamma1):
        from scipy.special import gammaln

        nu, s = prior.nu_s  # solved once, not at every evaluation
        const = float(np.log(2) - gammaln(nu / 2) + (nu / 2) * np.log(s / 2))
        return (lambda x: const - (nu + 1) * np.log(x) - s / (2 * x * x) if x > 0 else -np.inf), lambda x: -(nu + 1) / x + s / x**3
    if isinstance(prior, Uniform):
        return prior.logpdf, lambda x: 0.0

    def numeric(x: float) -> float:
        h = 1e-6 * max(abs(x), 1.0)
        return (prior.logpdf(x + h) - prior.logpdf(x - h)) / (2 * h)

    return prior.logpdf, numeric


def _gradient_posterior(family: Any, gaussian: bool, y: np.ndarray, X: np.ndarray, w: np.ndarray, coef_priors: list[Any],
                        error_key: str, error_prior: Any) -> Any:
    """The log posterior of (beta, log sigma) (or beta alone) and its gradient, analytic: for the No-U-Turn sampler."""
    from .ml import _loglike, _score_hessian

    n, k = X.shape
    coef = [(j, *_prior_terms(p)) for j, p in enumerate(coef_priors) if p is not None]
    err = _prior_terms(error_prior) if error_prior is not None else None
    half_log_w = 0.5 * float(np.sum(np.log(w)))

    def f(theta: np.ndarray) -> tuple[float, np.ndarray]:
        beta = theta[:k]
        g = np.zeros(len(theta))
        lp = 0.0
        for j, logpdf, dlog in coef:
            lp += logpdf(float(beta[j]))
            g[j] += dlog(float(beta[j]))
        eta = X @ beta
        if gaussian:
            sigma2 = float(np.exp(2 * theta[k]))
            e = y - eta
            q = float(np.sum(w * e * e))
            ll = -0.5 * q / sigma2 - n * theta[k] + half_log_w
            g[:k] += X.T @ (w * e) / sigma2
            g[k] += q / sigma2 - n
            if err is not None:  # sampled in log sigma: a prior on sigma gains the Jacobian sigma, one on sigma^2 2 sigma^2
                sigma = float(np.sqrt(sigma2))
                if error_key == "sigma":
                    lp += err[0](sigma) + np.log(sigma)
                    g[k] += sigma * err[1](sigma) + 1
                else:
                    lp += err[0](sigma2) + np.log(2 * sigma2)
                    g[k] += 2 * sigma2 * err[1](sigma2) + 2
        else:
            ll = float(np.sum(w * _loglike(family, y, eta)))
            g[:k] += X.T @ (w * _score_hessian(family, y, eta)[0])
        value = lp + ll
        return (float(value), g) if np.isfinite(value) else (-np.inf, g)

    return f


def _nuts(f: Any, bounds: list[tuple[float, float]], mode: np.ndarray, cov: np.ndarray, draws: int, burn: int, chains: int,
          seed: Any, metric: str = "diag") -> tuple[list[np.ndarray], Any]:
    """NUTS chains for the log posterior ``f`` (value and gradient), each parameter mapped off its bounds; ``metric``
    as ``pymetrix.models.bayes_nuts.start_metric``."""
    from ..models.bayes_nuts import start_metric
    from ..nuts import Transform, sample

    tr = Transform.of(bounds)
    z0 = tr.to_z(mode)
    var_z = np.maximum(np.diag(cov), 1e-12) / tr.dx_dz(z0) ** 2
    inv_metric = start_metric(metric, cov, tr.dx_dz(z0), var_z)

    in_z = tr.wrap(f)
    jitter = make_rng(seed_sequence(seed).spawn(chains + 1)[chains])  # a stream apart from the chains' own
    starts = z0 + 0.2 * np.sqrt(var_z) * jitter.standard_normal((chains, len(z0)))
    from ..progress import report

    run = sample(in_z, starts, draws=draws, warmup=burn, seed=seed, inv_metric=inv_metric,
                 progress=lambda share: report(share, "No-U-Turn sampler"), metric="diag" if metric == "diag" else "dense",
                 adapt_metric=metric != "mode")
    return [np.array([tr.to_x(z) for z in run.draws[c]]) for c in range(chains)], run


def estimate(method: Any, eq: Any, data: Any, span: Any, where: Any, weights: Any, missing: str) -> FittedBayesEquation:
    from .design import linear_design, provenance
    from .families import Gaussian, Logit, Poisson, Probit
    from .ml import _check_outcome, _loglike, _newton

    if eq.has_error_model:
        raise SpecificationError("Bayesian estimation of equations with ARMA errors or GARCH variance is not built")
    if eq.form != "linear":
        raise SpecificationError("Bayesian estimation takes linear equations (any family below); explicit-form equations are not built")
    family = eq.family
    gaussian = isinstance(family, Gaussian)
    if not gaussian and not isinstance(family, (Logit, Probit, Poisson)):
        raise SpecificationError(f"Bayesian estimation takes Gaussian, Logit, Probit or Poisson equations, not {type(family).__name__}")
    if method.sampler not in ("auto", "rwmh", "nuts"):
        raise SpecificationError(f"sampler is 'auto', 'rwmh' or 'nuts', not {method.sampler!r}")
    if method.metric not in ("diag", "dense", "mode"):
        raise SpecificationError(f"metric (NUTS's mass matrix) is 'diag', 'dense' or 'mode', not {method.metric!r}")
    d = linear_design(eq, data, span, where, weights, missing)
    y, X, names = d.y, d.X, list(d.names)
    w = d.w if d.w is not None else np.ones(len(y))
    n, k = X.shape
    prior = dict(method.prior or {})
    if not isinstance(prior, Mapping):
        raise SpecificationError("for an equation, prior is a mapping of coefficient names (and sigma or sigma2) to priors")
    error_keys = {"sigma", "sigma2"} if gaussian else set()
    unknown = sorted(set(prior) - set(names) - error_keys)
    if unknown:
        raise SpecificationError(f"priors for {unknown}, which are not coefficients of the equation ({', '.join(names)})"
                                 + ("" if gaussian else "; only Gaussian equations have sigma"))
    if "sigma" in prior and "sigma2" in prior:
        raise SpecificationError("give a prior on sigma or on sigma2, not both")
    coef_priors = [prior.get(nm) for nm in names]
    error_key = "sigma" if "sigma" in prior else "sigma2"
    error_prior = prior.get(error_key)
    chains, draws, burn = int(method.chains), int(method.draws), int(method.burn)
    if chains < 1 or draws < 10 or burn < 0:
        raise SpecificationError("chains must be at least 1, draws at least 10 and burn zero or more")
    streams = [make_rng(s) for s in seed_sequence(method.seed).spawn(chains)]
    columns = names + (["sigma"] if gaussian else [])
    ig = None
    if gaussian and method.sampler == "auto" and all(_flat_or_normal(p) for p in coef_priors):
        ig = _ig_shape_scale(error_prior, error_key)
    from ..progress import report

    if ig is not None:
        sampler = "Gibbs"
        out = []
        for j, rng in enumerate(streams):
            report(j / chains, f"Gibbs: chain {j + 1} of {chains}")
            out.append(_gibbs(y, X, w, coef_priors, ig[0], ig[1], draws, burn, rng))
        acceptance = 1.0
    else:
        sampler = "No-U-Turn (NUTS)" if method.sampler == "nuts" else "random-walk Metropolis-Hastings"

        def unpack(theta: np.ndarray) -> tuple[np.ndarray, float]:
            return theta[:k], (float(np.exp(theta[k])) if gaussian else 1.0)

        def logpost(theta: np.ndarray) -> float:
            beta, sigma = unpack(theta)
            lp = 0.0
            for p, b in zip(coef_priors, beta):
                if p is not None:
                    lo, hi = p.support
                    if not lo < b < hi:
                        return -np.inf
                    lp += p.logpdf(float(b))
            eta = X @ beta
            if gaussian:
                e = y - eta
                ll = -0.5 * float(np.sum(w * e * e)) / sigma**2 - n * np.log(sigma) + 0.5 * float(np.sum(np.log(w)))
                # sampled in log sigma: 1/sigma^2 on sigma^2 is flat there; a prior on sigma gains the Jacobian sigma, one on
                # sigma^2 the Jacobian 2 sigma^2
                if error_key == "sigma" and error_prior is not None:
                    lp += error_prior.logpdf(sigma) + np.log(sigma)
                elif error_prior is not None:
                    lp += error_prior.logpdf(sigma**2) + np.log(2 * sigma**2)
            else:
                ll = float(np.sum(w * _loglike(family, y, eta)))
            return lp + ll if np.isfinite(ll) else -np.inf

        if gaussian:
            beta0 = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
            e0 = y - X @ beta0
            start = np.append(beta0, 0.5 * np.log(float(e0 @ (w * e0)) / max(n - k, 1)))
        else:
            _check_outcome(family, y)
            start = _newton(family, y, X, w, 100, 1e-10, bhhh=False)[0]
        for j, p in enumerate(coef_priors):  # a start inside every prior's support
            if p is not None and not p.support[0] < start[j] < p.support[1]:
                start[j] = float(np.clip(getattr(p, "mean", start[j]), p.support[0] + 1e-6, p.support[1] - 1e-6))
        found = optimize.minimize(lambda t: -logpost(t), start, method="Nelder-Mead" if len(start) > 8 else "BFGS",
                                  options={"maxiter": 20_000})
        mode = found.x if np.isfinite(found.fun) else start
        from .numdiff import hessian

        H = hessian(logpost, mode)
        try:
            cov = np.linalg.inv(-H)
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            cov = np.diag(np.maximum(np.abs(mode), 0.1) ** 2 * 1e-2)
        out, rates = [], []
        if method.sampler == "nuts":
            bounds = [p.support if p is not None else (-np.inf, np.inf) for p in coef_priors] + ([(-np.inf, np.inf)] if gaussian else [])
            f = _gradient_posterior(family, gaussian, y, X, w, coef_priors, error_key, error_prior)
            out, run = _nuts(f, bounds, mode, cov, draws, burn, chains, method.seed, method.metric)
            if gaussian:
                for o in out:
                    o[:, k] = np.exp(o[:, k])
            rates = list(run.accept.mean(axis=1))
        for j, rng in enumerate(streams if method.sampler != "nuts" else []):
            report(j / chains, f"Metropolis-Hastings: chain {j + 1} of {chains}")
            o, a = _rwmh(logpost, mode, cov, draws, burn, rng)
            if gaussian:
                o[:, k] = np.exp(o[:, k])
            out.append(o)
            rates.append(a)
        acceptance = float(np.mean(rates))
    D = pd.DataFrame(np.vstack(out), columns=columns)
    D.insert(0, "chain", np.repeat(np.arange(1, chains + 1), draws))
    B = D[names].to_numpy()
    beta_mean = B.mean(axis=0)
    eta = X @ beta_mean
    mu = eta if gaussian else family.mean(eta)
    stats = {"nobs": float(n), "k": float(k), "draws": float(draws * chains), "acceptance": acceptance}
    extra: dict[str, Any] = {}
    if sampler.startswith("No-U-Turn"):
        stats["divergences"] = float(run.divergent.sum())
        extra["nuts"] = run.info()
    if gaussian:
        stats["sigma"] = float(D["sigma"].mean())
    for name, (_, _, e, se) in _chain_diagnostics(D).items():  # how many independent draws each mean is worth, and its error
        stats[f"ess_{name}"] = float(e)
        stats[f"mcse_{name}"] = float(se)
    stated =", ".join(f"{nm} {type(p).__name__}({', '.join(f'{v:g}' for v in vars(p).values())})"
                       for nm, p in prior.items() if p is not None) or "none (flat, and 1/sigma^2 on the error)"
    notes = [f"Method: Bayes ({sampler}{'' if gaussian else ', ' + type(family).__name__}); {chains} chains of {draws} draws "
             f"after {burn} discarded" + ("" if sampler == "Gibbs" else f"; acceptance rate {acceptance:.2f}"
                                          if not extra else " (warmup: step size and mass matrix adapted)"),
             f"Priors: {stated}; coefficients without one are flat"]
    if extra:
        notes[1:1] = run.notes()
    return FittedBayesEquation(
        equation=eq, params=point(names, beta_mean, np.atleast_2d(np.cov(B, rowvar=False))), method=method,
        resid=pd.Series(y - mu, index=d.index, name="resid"), fitted=pd.Series(mu, index=d.index, name="fitted"),
        stats=stats, provenance=provenance(eq, method, data, d.index, d.dropped),
        design=pd.DataFrame(X, index=d.index, columns=names), weights=weights,
        extra={"notes": notes, "sampler": sampler, **extra}, draws=D)
