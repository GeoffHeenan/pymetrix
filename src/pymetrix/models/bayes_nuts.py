"""The posterior of a linearised model's parameters by the No-U-Turn sampler: ``fit.mcmc(..., sampler="nuts")``.

Each parameter is mapped off its bounds (and its prior's support) to the real line: the logistic between two finite
bounds, the log above one, with the Jacobian in the density. The log posterior is the Kalman-filter likelihood times the
priors, as for Metropolis-Hastings. Its gradient is analytic where that applies (``pymetrix.models.kalman_score``: the
steady state, the solution and the Kalman filter differentiated exactly, chained through the transform), at the cost of
a few likelihood evaluations a leapfrog step; elsewhere (known-date regimes, the OccBin filter, diffuse starts, growth
models), or with ``gradient="numerical"``, by central differences (``pymetrix.estimation.numdiff.jacobian``), so that
every leapfrog step costs ``2k + 1`` likelihood evaluations (each a model solution and a Kalman filter) for ``k``
parameters. A draw takes ``2**depth`` steps: with a tree depth of 3 to 5 and central differences, a NUTS draw costs 50
to 300 times a Metropolis-Hastings draw for ten parameters. In return the draws are far less autocorrelated: compare
effective sample sizes per second. A finite-difference gradient is not exact, but the leapfrog map keeps volume and is
reversible for any gradient, and each draw is chosen by the exact energy, so the draws keep the posterior. A particle
filter's likelihood is an estimate: NUTS is refused there (particle marginal Metropolis-Hastings samples it).

The chains start near the posterior mode, with the mass matrix from the inverse Hessian there, and adapt in warmup as
Stan does (``pymetrix.nuts``): ``metric="dense"`` (the default) its whole covariance, ``"diag"`` its variances (each
then re-estimated from the warmup draws), ``"mode"`` the whole inverse Hessian kept fixed, only the step size adapting:
a posterior near Gaussian then needs a short warmup. Dense is the default because DSGE posteriors correlate their
parameters (a shock's persistence with its size, a policy rule's coefficients with each other): on the New
Keynesian template it gave 2.5 times the smallest effective sample a second of the diagonal metric, and on Smets and
Wouters (2007), 36 parameters, 2.3 times (three times per gradient; ``docs/validation.md``).

A point where the model cannot be solved, or has no unique stable solution, has zero density: a trajectory that
reaches it stops, counted as a divergence. ``workers`` chains run at once, each in its own process, as
Metropolis-Hastings chains do (by default when the run is long enough to repay starting them); each chain has its own
random stream and starts from no carried-over solution, so the draws are the same either way.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..errors import SpecificationError
from .solvers import ModelError


_STEPS_GUESS = 10  # leapfrog steps a draw takes, for deciding whether chains go to other processes (trees of depth 3 to 4)


class _Target:
    """The log posterior and its gradient in the unconstrained parameters ``z`` (``Transform``), with the Jacobian: the
    analytic gradient chained through the transform, or central differences. A class, so that chains can run in
    other processes; ``reset`` drops the last model solution (a warm start), so each chain's draws are the same
    wherever it runs."""

    def __init__(self, logpost: Callable[[np.ndarray], float], tr: Any, analytic: bool) -> None:
        self.logpost, self.tr, self.analytic = logpost, tr, analytic

    def reset(self) -> None:
        owner = getattr(self.logpost, "__self__", None)
        if hasattr(owner, "forget"):
            owner.forget()

    def value(self, z: np.ndarray) -> float:
        return float(self.logpost(self.tr.to_x(z))) + self.tr.log_jacobian(z)

    def __call__(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        if self.analytic:
            fn = self.__dict__.get("_fn")
            if fn is None:
                from .kalman_score import log_posterior_gradient

                lik = self.logpost.__self__  # type: ignore[attr-defined]
                fn = self.__dict__["_fn"] = self.tr.wrap(lambda x: log_posterior_gradient(lik, x))
            return fn(z)
        from ..estimation.numdiff import jacobian

        lp = self.value(z)
        if not np.isfinite(lp):
            return -np.inf, np.zeros(len(z))
        g = jacobian(lambda u: np.array([self.value(u)]), z)[0]
        return lp, g if np.all(np.isfinite(g)) else np.full(len(z), np.nan)

    def __getstate__(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "_fn"}


METRICS = ("diag", "dense", "mode")


def start_metric(metric: str, cov: Any, dx_dz: np.ndarray, var_z: np.ndarray) -> np.ndarray:
    """The starting inverse metric in the unconstrained parameters: the mode's variances (``"diag"``), or its whole
    covariance, the inverse Hessian, carried through the transform (``"dense"``, ``"mode"``: ``D^-1 cov D^-1`` with
    ``D`` the transform's derivatives), its eigenvalues kept above a millionth of the largest so that it is positive
    definite; the variances alone where the estimate has no covariance."""
    if metric not in METRICS:
        raise SpecificationError(f"metric is 'diag', 'dense' or 'mode', not {metric!r}")
    if metric == "diag":
        return var_z
    C = np.asarray(cov, dtype=float) if cov is not None else None
    if C is None or C.shape != (len(var_z), len(var_z)) or not np.all(np.isfinite(C)):
        if metric == "mode":
            raise ModelError("metric='mode' needs the inverse Hessian at the mode: estimate with a covariance (cov='hessian')")
        return np.diag(var_z)
    C = C / np.outer(dx_dz, dx_dz)
    vals, vecs = np.linalg.eigh(0.5 * (C + C.T))
    top = float(vals.max()) if vals.size else 1.0
    if not top > 0:
        if metric == "mode":
            raise ModelError("metric='mode' needs a positive definite inverse Hessian at the mode")
        return np.diag(var_z)
    vals = np.maximum(vals, 1e-6 * top)
    return (vecs * vals) @ vecs.T


def nuts(fit: Any, draws: int = 1000, chains: int = 2, burn: int | None = None, seed: int | None = None,
         progress: Callable[[float], Any] | None = None, target_accept: float = 0.8, max_depth: int = 10,
         gradient: str = "auto", workers: int | None = None, metric: str = "dense") -> Any:
    """The work behind ``FittedModel.mcmc(..., sampler="nuts")``: a ``Posterior`` with ``sampler == "nuts"``."""
    from .._random import generator as make_rng, seed_sequence
    from ..nuts import Transform, sample
    from ..progress import report
    from ..threads import blas_threads
    from .bayes import Posterior

    if fit.log_posterior is None:
        raise ModelError("this estimate cannot be sampled: estimate the model again with this version of pymetrix")
    if getattr(fit, "particle", None) is not None:
        raise ModelError("NUTS needs the exact log posterior and its gradient; a particle filter's likelihood is an estimate: "
                         "sample it with particle marginal Metropolis-Hastings (sampler='rwmh')")
    names = tuple(fit.params.names)
    missing = [n for n in names if n not in fit.priors]
    if missing:
        raise ModelError(f"give every estimated parameter a prior for the posterior: none for {', '.join(missing)} "
                         "(priors={...} in estimate, or Param.prior)")
    if draws < 10 or chains < 1:
        raise SpecificationError("draws must be at least 10 and chains at least 1")
    burn = draws if burn is None else int(burn)
    bounds = []
    for n in names:
        low, high = fit.bounds.get(n, (-np.inf, np.inf))
        s_low, s_high = getattr(fit.priors[n], "support", (-np.inf, np.inf))
        bounds.append((max(float(low), float(s_low)), min(float(high), float(s_high))))
    tr = Transform.of(bounds)
    mode = np.asarray(fit.params.values, dtype=float)
    inside = np.array([lo < m < hi for m, (lo, hi) in zip(mode, bounds)])
    if not inside.all():
        stuck = [n for n, ok in zip(names, inside) if not ok]
        raise ModelError(f"the mode is on a bound for {', '.join(stuck)}: NUTS samples inside the bounds; widen them, or "
                         "use Metropolis-Hastings")
    z0 = tr.to_z(mode)
    logpost = fit.log_posterior
    cov = fit.params.cov
    sd = np.sqrt(np.abs(np.diag(cov))) if cov is not None else np.full(len(mode), np.nan)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 0.1 * np.maximum(np.abs(mode), 0.1))
    var_z = (sd / tr.dx_dz(z0)) ** 2
    inv_metric = start_metric(metric, cov, tr.dx_dz(z0), var_z)

    from .kalman_score import ready

    lik = getattr(logpost, "__self__", None)
    analytic = lik is not None and hasattr(lik, "free") and ready(lik, mode, gradient)
    f = _Target(logpost, tr, analytic)
    value = f.value
    if not np.isfinite(value(z0)):
        raise ModelError("the log posterior is not finite at the mode")
    jitter = make_rng(seed_sequence(seed).spawn(chains + 1)[chains])  # a stream apart from the chains' own
    starts = z0 + 0.2 * np.sqrt(var_z) * jitter.standard_normal((chains, len(z0)))

    def advance(share: float) -> None:
        if progress is not None:
            progress(share)
        report(share, "NUTS: drawing from the posterior")  # the app's progress bar; Cancel stops

    if workers is None:  # in parallel when the run is long enough to repay starting processes (about a second each)
        import os
        import time

        from .bayes import _PARALLEL_SECONDS

        begin = time.perf_counter()
        for _ in range(3):
            f(z0)
        seconds = (time.perf_counter() - begin) / 3 * chains * (draws + burn) * _STEPS_GUESS
        workers = min(chains, os.cpu_count() or 1) if chains > 1 and seconds > _PARALLEL_SECONDS else 1
    with blas_threads():
        run = sample(f, starts, draws=int(draws), warmup=burn, seed=seed, target_accept=target_accept, max_depth=max_depth,
                     inv_metric=inv_metric, progress=advance, workers=int(workers), metric="diag" if metric == "diag" else "dense",
                     adapt_metric=metric != "mode")
    x = np.array([[tr.to_x(z) for z in chain] for chain in run.draws])
    lp = run.logp - np.array([[tr.log_jacobian(z) for z in chain] for chain in run.draws])
    info = {**run.info(), **({"metric": "dense, the inverse Hessian at the mode (not adapted)"} if metric == "mode" else {}),
            "gradient": "analytic (Kalman-filter score)" if analytic else "central differences", "notes": run.notes()}
    return Posterior(fit, names, x, lp, run.accept.mean(axis=1), run.step_size, burn, seed, sampler="nuts", nuts=info)
