"""Bayesian estimation of linearised models: the posterior by random-walk Metropolis-Hastings, as in Dynare.

    fit = nk.estimate(data, estimate=["kappa", "rho", "sd_ei"], priors={...})   # the posterior mode first
    post = fit.mcmc(draws=20000, chains=2, seed=1)                             # then the posterior around it
    post.summary()                  # means, medians, HPD intervals, R-hat, effective sample sizes, acceptance
    post.marginal_likelihood()      # log marginal data density: modified harmonic mean (Geweke)
    post.laplace()                  # the same, by the Laplace approximation at the mode
    post.irf(20)                    # impulse responses with posterior bands
    post.plot(); post.trace()       # prior and posterior densities; the chains
    fit.identification()            # can the data's moments tell the parameters apart (Iskrev)

The proposal is normal around the current draw with covariance ``c**2`` times the inverse Hessian at the mode
(Dynare's ``mh_jscale``). Each chain starts from an overdispersed draw around the mode; during burn-in ``c``
is tuned towards an acceptance rate of about 0.3, then held fixed for the draws that are kept. Draws outside the
parameters' bounds, or where the model has no steady state or no unique stable solution, are rejected.

Several modes (mode jumps). A random walk scaled to one mode rarely crosses to another far away: on Smets and
Wouters (2007) about once in 10,000 draws. When the estimate found more than one mode (``estimate(...,
starts=n)``), or ``modes=`` names others, a share ``jump`` of the steps (5% by default) instead proposes moving the
draw by the difference between two of the modes, chosen at random: ``x + (m_j - m_i)``. The reverse move is as
likely, so the proposal is symmetric and accepted with the usual Metropolis probability; the draws keep the
posterior exactly, with each mode's share, and the chains cross far more often. Chains start around the modes in
turn. While adapting, a chain's burn-in draws are taken relative to the mode it is in, so the proposal keeps the
shape of one mode.

Diagnostics follow Gelman et al. (2013) and Stan: split R-hat, and effective sample sizes from the chains'
autocorrelations summed in pairs while positive (Geyer's initial monotone sequence). HPD intervals are the
shortest intervals holding the stated share of the draws. The marginal data density is Geweke's modified harmonic
mean with a truncated normal weight, averaged over truncations 0.1 to 0.9 as Dynare reports it; the Laplace
approximation uses the mode and the inverse Hessian.

Identification (Iskrev 2010): the Jacobian of the observables' means and autocovariances with respect to the
parameters. With known-date regimes the moments change over the sample: the Jacobian is of the means, variances and
autocovariances of every period of the sample, stacked, as the time-varying rules imply them (``regime_moments``); a
parameter of a regime that does not occur in the sample (nor is foreseen in it) moves none of them. A rank below the number of parameters means some cannot be told apart; each parameter's
sensitivity (the Jacobian column, scaled by the parameter's value) and its collinearity (the multiple
correlation of its column with the others, and the parameter it is closest to) show which.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._api import api, renamed_arguments
from ..errors import SpecificationError
from ..output import Table
from .solvers import ModelError
from .._random import generator as make_rng, get_random_generator as _current_generator, seed_sequence

__all__ = ["Identification", "Posterior", "ess", "hpd", "identification", "laplace", "mcmc", "mcse", "rhat"]

_TARGET = 0.3  # acceptance rate aimed at while tuning


# --- diagnostics ------------------------------------------------------------------------------------------------

def _autocovariance(x: np.ndarray) -> np.ndarray:
    """Autocovariances of one chain at every lag (by FFT), divided by n."""
    n = len(x)
    centred = x - x.mean()
    size = 2 ** int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(centred, size)
    return np.fft.irfft(f * np.conjugate(f), size)[:n] / n


def ess(chains: np.ndarray) -> float:
    """Effective sample size of one parameter's draws, ``chains[chain, draw]``: Stan's estimator (Geyer's initial
    monotone sequence on the autocorrelations combined across chains)."""
    x = np.atleast_2d(np.asarray(chains, dtype=float))
    m, n = x.shape
    if n < 4:
        return float("nan")
    acov = np.array([_autocovariance(c) for c in x])
    chain_var = acov[:, 0] * n / (n - 1)
    w = chain_var.mean()
    var_plus = w * (n - 1) / n + (x.mean(axis=1).var(ddof=1) if m > 1 else 0.0)
    if var_plus <= 0:
        return float("nan")
    rho = 1 - (w - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    pairs = []
    previous = np.inf
    for t in range(0, n - 1, 2):
        p = rho[t] + rho[t + 1]
        if p <= 0:
            break
        p = min(p, previous)  # monotone
        pairs.append(p)
        previous = p
    tau = -1 + 2 * float(np.sum(pairs))
    return float(m * n / max(tau, 1.0 / np.log10(max(m * n, 10))))


def mcse(chains: np.ndarray) -> float:
    """Monte Carlo standard error of the posterior mean of one parameter's draws, ``chains[chain, draw]``: the
    standard deviation of all the draws over the square root of their effective sample size (``ess``), as Stan and
    ArviZ (``method="mean"``) report it. How far the mean of these draws may be from the posterior mean itself."""
    x = np.atleast_2d(np.asarray(chains, dtype=float))
    e = ess(x)
    if not np.isfinite(e) or e <= 0 or x.size < 2:
        return float("nan")
    return float(x.std(ddof=1) / np.sqrt(e))


def rhat(chains: np.ndarray) -> float:
    """Split R-hat of one parameter's draws, ``chains[chain, draw]``: near 1 when the chains agree."""
    x = np.atleast_2d(np.asarray(chains, dtype=float))
    half = x.shape[1] // 2
    if half < 2:
        return float("nan")
    split = np.vstack([x[:, :half], x[:, -half:]])
    n = split.shape[1]
    w = split.var(axis=1, ddof=1).mean()
    b = n * split.mean(axis=1).var(ddof=1)
    if w <= 0:
        return float("nan")
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


def hpd(draws: np.ndarray, coverage: float = 0.9) -> tuple[float, float]:
    """The shortest interval holding ``coverage`` of the draws."""
    x = np.sort(np.asarray(draws, dtype=float).ravel())
    k = int(np.ceil(coverage * len(x)))
    if k >= len(x):
        return float(x[0]), float(x[-1])
    widths = x[k:] - x[:len(x) - k]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + k])


def _logsumexp(a: np.ndarray) -> float:
    top = float(np.max(a))
    return top + float(np.log(np.sum(np.exp(a - top))))


def _prior_text(prior: Any) -> str:
    return f"{type(prior).__name__}({', '.join(f'{v:g}' for v in vars(prior).values())})" if prior is not None else ""


# --- the posterior ----------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Posterior:
    """Draws from the posterior of a linearised model's parameters: see the module docstring."""

    fit: Any  # the FittedModel at the posterior mode
    names: tuple[str, ...]
    draws: np.ndarray  # [chain, draw, parameter], burn-in discarded
    logpost: np.ndarray  # [chain, draw]: log likelihood + log prior of each draw
    acceptance: np.ndarray  # acceptance rate of each chain, after burn-in
    scale: np.ndarray  # the tuned proposal scale of each chain
    burn: int
    seed: int | None = None
    coverage: float = 0.9
    _cache: dict = field(default_factory=dict, repr=False)
    modes: Any = None  # [mode, parameter]: the modes jumped between (the estimate's first), or None for one mode
    jumps: Any = None  # [chain, 2]: mode jumps proposed and accepted while draws were kept
    jump: float = 0.0  # the share of steps that proposed a jump
    sampler: str = "mh"  # "mh": random-walk Metropolis-Hastings; "smc": sequential Monte Carlo (see smc)
    smc: Any = None  # sequential Monte Carlo: the tempering schedule, effective sizes, acceptance and log marginal likelihood
    nuts: Any = None  # the No-U-Turn sampler's diagnostics: step sizes, divergences, tree depths (sampler "nuts")
    random_generator: str = field(default_factory=lambda: _current_generator())  # the bit generator the draws used

    @property
    def provenance(self) -> Any:
        """The estimate's provenance, with the sampler: draws, chains, burn-in, seed and acceptance."""
        from dataclasses import replace

        prov = getattr(self.fit, "provenance", None)
        if prov is None:
            return None
        chains, draws = self.draws.shape[:2]
        sampler = {"sampler": "random-walk Metropolis-Hastings", "chains": int(chains), "draws": int(draws), "burn": int(self.burn),
                   "acceptance": [round(float(a), 4) for a in self.acceptance]}
        if self.sampler == "pmmh":
            sampler["sampler"] = ("particle marginal Metropolis-Hastings (random walk; a fresh particle-filter likelihood "
                                  "estimate at each proposal, the current draw's kept)")
        if self.sampler == "smc":
            sampler = {"sampler": "sequential Monte Carlo (Herbst-Schorfheide, adaptive tempering)", "particles": int(draws),
                       "stages": len(self.smc["phi"]) - 1, "steps": int(self.smc["steps"]), "seed": self.seed,
                       "log marginal data density": round(float(self.smc["log_ml"]), 4)}
        if self.sampler == "nuts":
            sampler = {"sampler": "No-U-Turn (NUTS, multinomial)",
                       "chains": int(chains), "draws": int(draws), "warmup": int(self.burn),
                       **{k: v for k, v in self.nuts.items() if k not in ("notes", "warmup")}}
        if self.modes is not None:
            sampler.update({"modes": [dict(zip(self.names, map(float, m))) for m in self.modes], "jump": float(self.jump),
                            "jumps accepted": [int(j[1]) for j in self.jumps]})
        return replace(prov, method={**(prov.method or {}), **sampler}, seed=self.seed,
                       random_generator=getattr(self, "random_generator", "pcg64") if self.seed is not None else None)

    def __getitem__(self, name: str) -> np.ndarray:
        """All kept draws of one parameter, chains one after another."""
        return self.draws[:, :, self.names.index(name)].ravel()

    @property
    def mean(self) -> dict[str, float]:
        return {n: float(self[n].mean()) for n in self.names}

    def summary(self) -> Table:
        rows = []
        mode = self.fit.params.as_dict()
        for j, n in enumerate(self.names):
            x = self.draws[:, :, j]
            low, high = hpd(x, self.coverage)
            smc = self.sampler == "smc"
            effective = float(self.smc["ess"][-1]) if smc else ess(x)  # a count of draws: shown whole
            rows.append([_prior_text(self.fit.priors.get(n)), mode[n], float(x.mean()), float(np.median(x)), float(x.std(ddof=1)),
                         low, high, float("nan") if smc else rhat(x), int(round(effective)) if np.isfinite(effective) else effective])
        chains, n_draws = self.draws.shape[:2]
        pct = f"{self.coverage:.0%}"
        rhats = {n: row[7] for n, row in zip(self.names, rows)}  # R-hat is the eighth column
        disagree = [n for n, value in rhats.items() if np.isfinite(value) and value > 1.1]
        notes = [f"Random-walk Metropolis-Hastings: {chains} chain(s) of {n_draws} draws kept after {self.burn} burn-in",
                 f"Acceptance rate: {', '.join(f'{a:.2f}' for a in self.acceptance)}; proposal scale {', '.join(f'{s:.3g}' for s in self.scale)}",
                 f"HPD: the shortest interval holding {pct} of the draws. R-hat near 1 and large effective sample sizes "
                 "suggest the chains have converged"]
        if self.sampler == "smc":
            s = self.smc
            notes = [f"Sequential Monte Carlo: {n_draws} particles from the prior, {len(s['phi']) - 1} tempering stages "
                     f"(each keeping the effective sample size near {s['target']:.0%} of the particles), {s['steps']} mutation "
                     f"step(s) a stage; {s['resampled']} resamplings",
                     f"Acceptance in the last stage {s['acceptance'][-1]:.2f}; proposal scale {s['scale'][-1]:.3g}. Effective draws: the "
                     "particles' effective sample size at the end (R-hat is for chains)",
                     f"HPD: the shortest interval holding {pct} of the draws"]
        if self.sampler == "nuts":
            notes = [*self.nuts["notes"], f"HPD: the shortest interval holding {pct} of the draws. R-hat near 1 and large "
                                          "effective sample sizes suggest the chains have converged"]
        if self.modes is not None:
            proposed, accepted = (int(v) for v in np.sum(self.jumps, axis=0))
            notes.append(f"Mode jumps between {len(self.modes)} modes on {self.jump:.0%} of steps: {accepted} of {proposed} accepted "
                         f"(per chain {', '.join(str(int(j[1])) for j in self.jumps)})")
        try:
            method = "sequential Monte Carlo" if self.sampler == "smc" else "modified harmonic mean"
            notes.append(f"Log marginal data density: {self.marginal_likelihood():.3f} ({method}); "
                         f"{self.laplace():.3f} (Laplace)")
        except (ModelError, ValueError, np.linalg.LinAlgError):
            pass
        if disagree:
            notes.insert(0, f"The chains disagree on {', '.join(disagree[:6])}{' ...' if len(disagree) > 6 else ''} (R-hat above 1.1): "
                            "run longer chains, or more of them, and check for several modes (estimate(..., starts=4))")
        return Table(["prior", "mode", "mean", "median", "std dev", f"HPD {pct} low", f"HPD {pct} high", "R-hat", "eff. draws"],
                     rows, index=list(self.names), title="Posterior distribution", notes=notes)

    def laplace(self) -> float:
        """Log marginal data density by the Laplace approximation at the posterior mode."""
        return laplace(self.fit)

    def marginal_likelihood(self, truncations: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                            method: str | None = None) -> float:
        """Log marginal data density: from sequential Monte Carlo's incremental weights when sampled that way, otherwise
        (or with ``method="mhm"``) Geweke's modified harmonic mean, averaged over the ``truncations``."""
        from scipy import stats

        if method not in (None, "mhm", "smc"):
            raise SpecificationError("method is 'mhm' (modified harmonic mean) or 'smc' (sequential Monte Carlo)")
        if method == "smc" and self.sampler != "smc":
            raise SpecificationError("these draws are from Metropolis-Hastings: the SMC estimate needs fit.smc(...)")
        if self.sampler == "smc" and method != "mhm":
            return float(self.smc["log_ml"])
        key = ("mhm", truncations)
        if key in self._cache:
            return self._cache[key]
        x = self.draws.reshape(-1, len(self.names))
        lp = self.logpost.ravel()
        ok = np.isfinite(lp)
        x, lp = x[ok], lp[ok]
        k = x.shape[1]
        mu = x.mean(axis=0)
        sigma = np.atleast_2d(np.cov(x, rowvar=False))
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            raise ModelError("the draws' covariance is singular: too few distinct draws for the marginal likelihood")
        d = x - mu
        q = np.einsum("ij,jk,ik->i", d, np.linalg.inv(sigma), d)
        estimates = []
        for p in truncations:
            inside = q <= stats.chi2.ppf(p, k)
            log_f = -np.log(p) - 0.5 * k * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * q
            terms = np.where(inside, log_f - lp, -np.inf)
            if not inside.any():
                continue
            estimates.append(-(_logsumexp(terms[inside]) - np.log(len(lp))))
        if not estimates:
            raise ModelError("no draws inside the truncation region")
        value = float(np.mean(estimates))
        self._cache[key] = value
        return value

    def _subsample(self, n: int) -> list[dict[str, float]]:
        x = self.draws.reshape(-1, len(self.names))
        idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).astype(int))
        return [dict(zip(self.names, map(float, x[i]))) for i in idx]

    @api()
    def forecast(self, span: tuple[Any, Any], scenario: Any = None, data: Any = None, tunes: Any = None, draws: int = 500,
                 shocks: bool = True, seed: int | None = None, occbin: bool | None = None, regimes: Any = None) -> Any:
        """Forecasts with parameter uncertainty: the forecast at the posterior mode (``fit.forecast``), with draws at
        ``draws`` posterior draws spread over the chains. At each, the model is solved and its variables filtered from
        the data again (their starting values depend on the parameters), and one path of its shocks is drawn
        (``shocks=False``: parameter uncertainty alone); observed series through the measurement equations, without
        measurement errors. ``Solution.fan``, ``quantiles`` and ``bands`` summarise them. ``occbin``: the model's
        occasionally binding constraints respected at every draw (the OccBin filter's state and OccBin's forecast at
        that draw's parameters); default: when the model was estimated with the OccBin filter. A model estimated with
        known-date regimes is solved at each draw with that draw's values of the regimes' parameters
        (``phi_pi@post``), filtered through their time-varying rules and forecast through them; ``regimes`` adds
        others in the forecast, as ``FittedModel.forecast``. See ``pymetrix.models.stochastic``."""
        from .stochastic import posterior

        return posterior(self, span, scenario, data, tunes, draws, shocks, seed, occbin, regimes)

    def irf(self, periods: int = 20, draws: int = 200, coverage: float | None = None) -> Any:
        """Impulse responses to one-standard-deviation shocks at ``draws`` draws spread over the chains: the
        posterior median, with the central ``coverage`` band (an ``IRF``)."""
        from ..var import IRF

        coverage = self.coverage if coverage is None else coverage
        results = []
        for values in self._subsample(draws):
            _, solution, system = self.fit.evaluate(values)
            results.append(solution.irf(periods, shock_sd=system.shock_sd, shock_corr=getattr(system, "shock_corr", None)))
        stack = np.stack([r.values for r in results])
        tail = (1 - coverage) / 2
        first = results[0]
        return IRF(np.median(stack, axis=0), first.responses, first.shocks, np.quantile(stack, tail, axis=0),
                   np.quantile(stack, 1 - tail, axis=0), identification="model (posterior median and band)", coverage=coverage)

    def plot(self) -> Any:
        """Prior (dashed) and posterior densities of each parameter."""
        import pandas as pd
        from scipy.stats import gaussian_kde

        from ..figures import Figure, Line, Panel

        panels = []
        for n in self.names:
            x = self[n]
            low, high = np.quantile(x, [0.001, 0.999])
            pad = (high - low) * 0.5 or abs(low) * 0.1 or 0.1
            grid = np.linspace(low - pad, high + pad, 200)
            lo_b, hi_b = self.fit.bounds.get(n, (-np.inf, np.inf))
            grid = grid[(grid >= lo_b) & (grid <= hi_b)]
            index = pd.Index(grid, name=n)
            layers: list[Any] = []
            if np.ptp(x) > 0:
                layers.append(Line(pd.Series(gaussian_kde(x)(grid), index=index), "posterior"))
            prior = self.fit.priors.get(n)
            if prior is not None:
                with np.errstate(all="ignore"):
                    dens = np.exp([prior.logpdf(g) for g in grid])
                layers.append(Line(pd.Series(dens, index=index), "prior", style="--"))
            panels.append(Panel(tuple(layers), title=n, legend=len(panels) == 0))
        return Figure(tuple(panels), title="Prior and posterior", columns=min(3, len(panels)))

    def trace(self) -> Any:
        """The kept draws of each parameter, chain by chain."""
        import pandas as pd

        from ..figures import Figure, Line, Panel

        panels = []
        for j, n in enumerate(self.names):
            layers = tuple(Line(pd.Series(self.draws[c, :, j], index=pd.RangeIndex(self.draws.shape[1], name="draw")), f"chain {c + 1}")
                           for c in range(self.draws.shape[0]))
            panels.append(Panel(layers, title=n, legend=j == 0, xlabel="draw"))
        title = "NUTS chains" if self.sampler == "nuts" else "Metropolis-Hastings chains"
        return Figure(tuple(panels), title=title, columns=min(2, len(panels)))

    def to_frame(self, view: str | None = None) -> Any:
        if view is not None:  # one of its views by name, as for every result (pymetrix.views)
            from ..views import _views_to_frame

            return _views_to_frame(self, view)
        import pandas as pd

        chains, n, k = self.draws.shape
        frame = pd.DataFrame(self.draws.reshape(-1, k), columns=list(self.names))
        frame.insert(0, "draw", np.tile(np.arange(n), chains))
        frame.insert(0, "chain", np.repeat(np.arange(1, chains + 1), n))
        frame["logpost"] = self.logpost.ravel()
        return frame

    def __repr__(self) -> str:
        chains, n = self.draws.shape[:2]
        return f"<Posterior: {', '.join(self.names)}; {chains} chain(s) x {n} draws, acceptance {self.acceptance.mean():.2f}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


def laplace(fit: Any) -> float:
    """Log marginal data density by the Laplace approximation at the posterior mode of ``fit``."""
    if "logpost" not in fit.stats:
        raise ModelError("the Laplace approximation needs priors on the estimated parameters (a posterior mode)")
    cov = fit.params.cov
    if cov is None or not np.all(np.isfinite(cov)):
        raise ModelError("no covariance at the mode (a parameter at a bound?): the Laplace approximation needs one")
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ModelError("the covariance at the mode is not positive definite")
    k = len(fit.params.names)
    return float(fit.stats["logpost"] + 0.5 * k * np.log(2 * np.pi) + 0.5 * logdet)


def _proposal_factor(cov: Any, mode: np.ndarray) -> np.ndarray:
    """Cholesky factor of the proposal covariance: the inverse Hessian at the mode, or a diagonal guess."""
    k = len(mode)
    if cov is not None and np.all(np.isfinite(cov)):
        c = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
        for jitter in (0.0, 1e-10, 1e-8, 1e-6):
            try:
                return np.linalg.cholesky(c + jitter * np.eye(k) * max(1.0, float(np.max(np.abs(np.diag(c))))))
            except np.linalg.LinAlgError:
                continue
    sd = np.where(np.isfinite(np.diag(cov)), np.sqrt(np.abs(np.diag(cov))), np.nan) if cov is not None else np.full(k, np.nan)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 0.1 * np.maximum(np.abs(mode), 0.1))
    return np.diag(sd)


def _chain(logpost: Callable[[np.ndarray], float], mode: np.ndarray, factor: np.ndarray, c0: float, scale: float | None,
           burn: int, draws: int, seed: np.random.SeedSequence, adapt: bool = True, modes: np.ndarray | None = None,
           jump: float = 0.0, start: int = 0, progress: Callable[[int], Any] | None = None
           ) -> tuple[np.ndarray, np.ndarray, float, float, tuple[int, int]]:
    """One chain: the kept draws, their log posteriors, the random walk's acceptance rate, the final proposal scale,
    and the mode jumps proposed and accepted among the kept draws. With ``modes`` (several), a share ``jump`` of the
    steps proposes ``x + (modes[j] - modes[i])`` instead, for a random ordered pair; the chain starts around
    ``modes[start]``.

    With ``adapt``, halfway through burn-in the proposal's covariance becomes that of the chain's own burn-in draws
    (from a quarter of the way in), shrunk a tenth towards the inverse Hessian, and its scale is tuned again: on a
    long curved ridge (Smets-Wouters' risk-premium shock and habit) the curvature at the mode is a poor guide, and a
    chain that starts away from the mode would otherwise crawl. Nothing adapts while draws are kept, so the kept
    draws are a Metropolis-Hastings chain with one fixed proposal."""
    owner = getattr(logpost, "__self__", logpost)  # a bound log posterior's likelihood, or a pseudo-marginal one itself
    if hasattr(owner, "forget"):
        owner.forget()  # no solution carried over from elsewhere: the same draws here or in another process
    k = len(mode)
    rng = make_rng(seed)
    kept, kept_lp = np.empty((draws, k)), np.empty(draws)
    shifts = [] if modes is None or len(modes) < 2 else [modes[j] - modes[i] for i in range(len(modes)) for j in range(len(modes)) if i != j]
    centre = mode if not shifts else np.asarray(modes[start % len(modes)], dtype=float)
    x, lp = centre.copy(), logpost(centre)
    for _ in range(100):  # an overdispersed start, as Dynare's mh_init_scale
        trial = centre + 2 * c0 * factor @ rng.standard_normal(k)
        trial_lp = logpost(trial)
        if np.isfinite(trial_lp):
            x, lp = trial, trial_lp
            break
    if not np.isfinite(lp):
        raise ModelError("the log posterior is not finite at the mode")
    c = c0
    window_accepts = window_steps = 0
    accepts = steps = 0
    tried = made = 0
    offset = np.zeros(k)  # the sum of the jumps made: draws less this are around the starting mode (for adapting)
    adapt_at = burn // 2 if adapt and scale is None and burn >= 400 else -1
    history = np.empty((max(adapt_at, 0), k))
    for i in range(burn + draws):
        if i == adapt_at:
            factor, c = _adapted(history[adapt_at // 2:], factor, c0)
        if shifts and rng.uniform() < jump:  # a jump between modes: symmetric, as the reverse shift is as likely
            shift = shifts[int(rng.integers(len(shifts)))]
            trial = x + shift
            trial_lp = logpost(trial)
            tried += i >= burn
            if np.isfinite(trial_lp) and np.log(rng.uniform()) < trial_lp - lp:
                x, lp = trial, trial_lp
                offset += shift
                made += i >= burn
        else:
            trial = x + c * factor @ rng.standard_normal(k)
            trial_lp = logpost(trial)
            moved = bool(np.isfinite(trial_lp) and np.log(rng.uniform()) < trial_lp - lp)
            if moved:
                x, lp = trial, trial_lp
            if i >= burn:
                steps += 1
                accepts += moved
            else:
                window_steps += 1
                window_accepts += moved
            if i < burn and scale is None and window_steps == 100:  # tune towards the target acceptance rate
                rate = window_accepts / 100
                c *= float(np.exp(2.0 * (rate - _TARGET)))
                window_accepts = window_steps = 0
        if i < adapt_at:
            history[i] = x - offset
        if i >= burn:
            kept[i - burn] = x
            kept_lp[i - burn] = lp
        if progress is not None and (i + 1) % 200 == 0:
            progress(200)
    return kept, kept_lp, accepts / max(steps, 1), c, (tried, made)


_MODE_GAP = 5.0  # starts this far below the best log posterior are not jumped to by default


def _modes(fit: Any, names: tuple[str, ...], mode: np.ndarray, factor: np.ndarray, modes: Any) -> np.ndarray | None:
    """The modes to jump between, the estimate's first: from ``modes`` (mappings by name or arrays), or by default
    from the estimate's starts (``estimate(..., starts=n)``) that reached within 5 of the best log posterior, highest
    first: a mode lower than that holds well under 1% of the posterior unless it is far wider, and jumps to it would
    mostly be refused (Smets-Wouters has one 6.3 below). Points within three proposal standard deviations of one
    already kept are the same mode. None when there is only one."""
    if modes is False:
        return None
    given = modes is not None
    if given:
        candidates = list(modes)
    else:
        prov = getattr(fit, "provenance", None)
        starts = (prov.method or {}).get("starts", []) if prov is not None else []
        best = max((float(r.get("log_posterior", -np.inf)) for r in starts), default=-np.inf)
        ranked = sorted(starts, key=lambda r: -float(r.get("log_posterior", -np.inf)))
        candidates = [r["to"] for r in ranked if float(r.get("log_posterior", -np.inf)) > best - _MODE_GAP]
    kept = [mode]
    for c in candidates:
        if isinstance(c, Mapping):
            unknown = [n for n in names if n not in c]
            if unknown:
                if given:
                    raise SpecificationError(f"a mode needs a value for every estimated parameter: none for {', '.join(unknown)}")
                continue
            x = np.array([float(c[n]) for n in names])
        else:
            x = np.asarray(c, dtype=float).ravel()
            if len(x) != len(names):
                raise SpecificationError(f"a mode needs {len(names)} values, one per estimated parameter, in params.names order")
        if not np.isfinite(fit.log_posterior(x)):
            if given:
                raise ModelError("the log posterior is not finite at one of the modes given")
            continue
        if all(np.linalg.norm(np.linalg.solve(factor, x - m)) > 3 for m in kept):
            kept.append(x)
    return np.array(kept) if len(kept) > 1 else None


def _adapted(history: np.ndarray, factor: np.ndarray, c0: float) -> tuple[np.ndarray, float]:
    """The proposal's Cholesky factor from burn-in draws (shrunk a tenth towards the old one) and the starting scale;
    the old ones when the draws moved too little to say anything."""
    k = history.shape[1]
    if len(np.unique(history, axis=0)) <= 2 * k:
        return factor, c0
    old = factor @ factor.T
    cov = 0.9 * np.cov(history, rowvar=False) + 0.1 * old
    try:
        return np.linalg.cholesky(0.5 * (cov + cov.T)), c0
    except np.linalg.LinAlgError:
        return factor, c0


_PARALLEL_SECONDS = 3.0  # estimated run time above which chains go to other processes by default


def mcmc(fit: Any, draws: int = 10000, chains: int = 2, burn: int | None = None, scale: float | None = None,
           seed: int | None = None, progress: Callable[[float], Any] | None = None, workers: int | None = None,
           adapt: bool = True, modes: Any = None, jump: float = 0.05, sampler: str = "rwmh", target_accept: float = 0.8,
           max_depth: int = 10, gradient: str = "auto", metric: str = "dense") -> Posterior:
    """The work behind ``FittedModel.mcmc``: see the module docstring; ``sampler="nuts"``: ``pymetrix.models.bayes_nuts``."""
    if sampler not in ("rwmh", "mh", "nuts"):
        raise SpecificationError(f"sampler is 'rwmh' (random-walk Metropolis-Hastings) or 'nuts' (No-U-Turn), not {sampler!r}")
    from .kalman_score import GRADIENTS

    if gradient not in GRADIENTS:
        raise SpecificationError(f"gradient is one of {', '.join(map(repr, GRADIENTS))}, not {gradient!r}")
    if metric not in ("diag", "dense", "mode"):
        raise SpecificationError(f"metric (NUTS's mass matrix) is 'dense', 'diag' or 'mode', not {metric!r}")
    if sampler == "nuts":
        if _particle_likelihood(fit) is not None:
            raise ModelError("NUTS needs the gradient of an exact likelihood, not a particle filter's noisy estimate: "
                             "sample this posterior with sampler='rwmh' (particle marginal Metropolis-Hastings)")
        from .bayes_nuts import nuts

        return nuts(fit, draws, chains, burn, seed, progress, target_accept, max_depth, gradient, workers, metric)
    if fit.log_posterior is None:
        raise ModelError("this estimate cannot be sampled: estimate the model again with this version of pymetrix")
    names = tuple(fit.params.names)
    missing = [n for n in names if n not in fit.priors]
    if missing:
        raise ModelError(f"give every estimated parameter a prior for the posterior: none for {', '.join(missing)} "
                         "(priors={...} in estimate, or Param.prior)")
    if draws < 10 or chains < 1:
        raise SpecificationError("draws must be at least 10 and chains at least 1")
    if not 0 <= jump < 1:
        raise SpecificationError("jump is the share of steps that propose a jump between modes: from 0 up to (not including) 1")
    burn = draws if burn is None else int(burn)
    k = len(names)
    mode = np.asarray(fit.params.values, dtype=float)
    factor = _proposal_factor(fit.params.cov, mode)
    c0 = float(scale) if scale is not None else 2.38 / np.sqrt(k)
    logpost = fit.log_posterior
    found = _modes(fit, names, mode, factor, modes) if jump > 0 else None
    streams = seed_sequence(seed).spawn(chains)  # one random stream per chain: the same draws wherever it runs
    pseudo = _particle_likelihood(fit)
    if pseudo is not None:  # particle marginal Metropolis-Hastings: each chain's likelihood estimates on fresh random numbers
        from .particle import PseudoMarginal

        jobs = [(PseudoMarginal(pseudo, stream.spawn(1)[0]), mode, factor, c0, scale, burn, draws, stream, adapt, found, jump, j)
                for j, stream in enumerate(streams)]
    else:
        jobs = [(logpost, mode, factor, c0, scale, burn, draws, stream, adapt, found, jump, j) for j, stream in enumerate(streams)]
    total = chains * (burn + draws)
    done = [0]

    from ..progress import report

    def advance(n: int) -> None:
        done[0] += n
        if progress is not None:
            progress(min(1.0, done[0] / total))
        report(min(1.0, done[0] / total), f"MCMC: {done[0]:,} of {total:,} draws")  # the app's progress bar; Cancel stops

    from ..threads import blas_threads

    with blas_threads():
        post = _sample(fit, names, mode, jobs, chains, total, advance, workers, burn, seed, found, jump)
    if pseudo is not None:
        from dataclasses import replace

        post = replace(post, sampler="pmmh")
    return post


def _particle_likelihood(fit: Any) -> Any:
    """The estimate's likelihood when it is the particle filter's (then sampled pseudo-marginally), else None."""
    owner = getattr(fit.log_posterior, "__self__", None)
    return owner if getattr(owner, "particle", None) else None


def _sample(fit: Any, names: tuple[str, ...], mode: np.ndarray, jobs: list[tuple], chains: int, total: int,
            advance: Callable[[int], Any], workers: int | None, burn: int, seed: int | None, found: Any = None,
            jump: float = 0.0) -> Posterior:
    logpost = fit.log_posterior
    if workers is None:  # in parallel when the run is long enough to repay starting processes (about a second each)
        import os
        import time

        start = time.perf_counter()
        for _ in range(3):
            logpost(mode)
        seconds = (time.perf_counter() - start) / 3 * total
        workers = min(chains, os.cpu_count() or 1) if chains > 1 and seconds > _PARALLEL_SECONDS else 1
    results = None
    if workers > 1 and chains > 1:
        from .._processes import run_in_processes

        results = run_in_processes([(_chain, job) for job in jobs], int(workers), lambda j, r: advance(2 * len(r[0])), "chain")
    if results is None:
        results = [_chain(*job, progress=advance) for job in jobs]
    kept = np.stack([r[0] for r in results])
    kept_lp = np.stack([r[1] for r in results])
    acceptance = np.array([r[2] for r in results])
    scales = np.array([r[3] for r in results])
    if found is None:
        return Posterior(fit, names, kept, kept_lp, acceptance, scales, burn, seed)
    return Posterior(fit, names, kept, kept_lp, acceptance, scales, burn, seed, modes=found,
                     jumps=np.array([r[4] for r in results]), jump=jump)


_SMC_PARALLEL_SECONDS = 10.0  # a stage estimated to take longer than this moves its particles in other processes
_SMC_BLOCKS = 8  # particles are moved in this many blocks, each with its own random stream: the same draws in any number of processes


def _mutate(logpost: Callable[[np.ndarray], float], priors: Sequence[Any], lows: np.ndarray, highs: np.ndarray, X: np.ndarray,
            LL: np.ndarray, LP: np.ndarray, factor: np.ndarray, c: float, phi: float, steps: int, stream: Any
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """``steps`` random-walk Metropolis-Hastings moves of a block of particles under the likelihood to the power phi
    times the prior: the particles, their log likelihoods and log priors after, and how many moves were accepted.
    Module level, so that blocks can run in other processes."""
    rng = make_rng(stream)
    X, LL, LP = X.copy(), LL.copy(), LP.copy()
    n, k = X.shape
    moved = 0
    for _ in range(int(steps)):
        proposals = X + c * rng.standard_normal((n, k)) @ factor.T
        logu = np.log(rng.random(n))
        for i in range(n):
            x = proposals[i]
            if not np.all((x > lows) & (x < highs)):
                continue
            post = logpost(x)
            if not np.isfinite(post):
                continue
            lp = float(sum(p.logpdf(v) for p, v in zip(priors, x)))
            ll = post - lp
            if logu[i] < phi * ll + lp - (phi * LL[i] + LP[i]):
                X[i], LL[i], LP[i] = x, ll, lp
                moved += 1
    return X, LL, LP, moved


@renamed_arguments(target="ess_fraction")
def smc(fit: Any, particles: int = 2000, steps: int = 2, ess_fraction: float = 0.5, seed: int | None = None,
        progress: Callable[[float], Any] | None = None, max_stages: int = 1000, workers: int | None = None) -> Posterior:
    """The posterior by sequential Monte Carlo (Herbst and Schorfheide 2014), as Dynare 6's: ``particles`` drawn from
    the prior move to the posterior through the likelihood raised to powers 0 < phi(1) < ... < 1, each step chosen so
    that the particles' weights keep a conditional effective sample size of ``ess_fraction`` (a share of the particles,
    between 0 and 1: higher means more, smaller tempering steps; ``target`` is its old name, deprecated);
    particles are resampled when their effective sample size falls below half, and moved by ``steps`` random-walk
    Metropolis-Hastings steps a stage, the proposal the particles' own covariance scaled towards an acceptance of 0.25.
    The log marginal data density is the sum of the logs of the mean incremental weights, plus the log of the share of
    prior draws inside the parameters' bounds where the model can be solved (so it compares with the modified harmonic
    mean and Laplace estimates, which integrate the posterior over the same region). A ``Posterior``: one "chain" of the
    final particles. ``workers``: processes that move the particles at once (default: as many as pay, when a stage
    takes long enough to repay starting them); the draws are the same either way."""
    if fit.log_posterior is None:
        raise ModelError("this estimate cannot be sampled: estimate the model again with this version of pymetrix")
    if _particle_likelihood(fit) is not None:
        raise ModelError("sequential Monte Carlo on a particle-filter likelihood (SMC squared) is not available: sample the "
                         "posterior with mcmc (particle marginal Metropolis-Hastings)")
    names = tuple(fit.params.names)
    missing = [n for n in names if n not in fit.priors]
    if missing:
        raise ModelError(f"give every estimated parameter a prior for the posterior: none for {', '.join(missing)} "
                         "(priors={...} in estimate, or Param.prior)")
    if particles < 50 or steps < 1 or not 0 < ess_fraction < 1:
        raise SpecificationError("particles is at least 50, steps at least 1, and ess_fraction a share between 0 and 1")
    from ..progress import report
    from ..threads import blas_threads

    k, N = len(names), int(particles)
    priors = [fit.priors[n] for n in names]
    lows = np.array([fit.bounds.get(n, (-np.inf, np.inf))[0] for n in names])
    highs = np.array([fit.bounds.get(n, (-np.inf, np.inf))[1] for n in names])
    rng = make_rng(seed)
    logpost = fit.log_posterior

    def prior_of(x: np.ndarray) -> float:
        return float(sum(p.logpdf(v) for p, v in zip(priors, x)))

    with blas_threads():
        X, LL, LP = np.empty((N, k)), np.empty(N), np.empty(N)
        filled = tried = 0
        while filled < N:  # draws from the prior, where the model can be solved
            tried += 1
            if tried > 1000 * N:
                raise ModelError(f"only {filled} of {N} prior draws give a model that can be solved: check the priors and bounds")
            x = np.array([float(p.rvs(rng)) for p in priors])
            if not np.all((x > lows) & (x < highs)):
                continue
            post = logpost(x)
            if not np.isfinite(post):
                continue
            X[filled], LP[filled] = x, prior_of(x)
            LL[filled] = post - LP[filled]
            filled += 1
        log_ml = float(np.log(N / tried))
        W = np.full(N, 1.0 / N)
        phi, c = 0.0, 0.5
        schedule, sizes, accepted, scales, resampled = [0.0], [float(N)], [], [], 0

        def conditional_ess(d: float) -> float:
            w = np.exp(d * (LL - LL.max()))
            return float((W @ w) ** 2 / (W @ w ** 2))

        while phi < 1.0:
            if len(schedule) > max_stages:
                raise ModelError(f"more than {max_stages} tempering stages: lower ess_fraction, or check the model")
            if conditional_ess(1.0 - phi) >= ess_fraction:
                new = 1.0
            else:
                lo, hi = 0.0, 1.0 - phi
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if conditional_ess(mid) >= ess_fraction else (lo, mid)
                new = phi + lo if lo > 0 else phi + hi * 1e-6
            inc = (new - phi) * LL
            top = inc.max()
            log_ml += float(top + np.log(W @ np.exp(inc - top)))
            W = W * np.exp(inc - top)
            W /= W.sum()
            phi = new
            if 1.0 / float(W @ W) < N / 2:  # systematic resampling
                positions = (rng.random() + np.arange(N)) / N
                idx = np.minimum(np.searchsorted(np.cumsum(W), positions), N - 1)
                X, LL, LP, W = X[idx], LL[idx], LP[idx], np.full(N, 1.0 / N)
                resampled += 1
            mean = W @ X
            cov = (X - mean).T @ ((X - mean) * W[:, None])
            factor = _proposal_factor(np.atleast_2d(cov), mean)
            blocks = np.array_split(np.arange(N), _SMC_BLOCKS)
            streams = seed_sequence(None if seed is None else [seed, len(schedule)]).spawn(len(blocks))
            jobs = [(logpost, priors, lows, highs, X[b], LL[b], LP[b], factor, c, phi, steps, st) for b, st in zip(blocks, streams)]
            if workers is None:  # in processes when a stage takes long enough to repay starting them (about a second each)
                import os
                import time

                start = time.perf_counter()
                logpost(X[0])
                seconds = (time.perf_counter() - start) * N * int(steps)
                workers = min(_SMC_BLOCKS, os.cpu_count() or 1) if seconds > _SMC_PARALLEL_SECONDS else 1
            results = None
            if workers > 1:
                from .._processes import run_in_processes

                results = run_in_processes([(_mutate, job) for job in jobs], int(workers), what="particles")
            if results is None:
                results = [_mutate(*job) for job in jobs]
            for b, (xb, llb, lpb, _) in zip(blocks, results):
                X[b], LL[b], LP[b] = xb, llb, lpb
            moved = sum(r[3] for r in results)
            rate = moved / (N * int(steps))
            c *= 0.95 + 0.10 * np.exp(16 * (rate - 0.25)) / (1 + np.exp(16 * (rate - 0.25)))
            schedule.append(phi)
            sizes.append(1.0 / float(W @ W))
            accepted.append(rate)
            scales.append(c)
            if progress is not None:
                progress(phi)
            report(phi, f"SMC: stage {len(schedule) - 1}, likelihood weight {phi:.3f}")  # the app's progress bar
        if not np.allclose(W, 1.0 / N):  # equal weights at the end, for draws like any posterior's
            positions = (rng.random() + np.arange(N)) / N
            idx = np.minimum(np.searchsorted(np.cumsum(W), positions), N - 1)
            X, LL, LP = X[idx], LL[idx], LP[idx]
            resampled += 1
    info = {"phi": schedule, "ess": sizes, "acceptance": accepted, "scale": scales, "log_ml": log_ml, "steps": int(steps),
            "target": float(ess_fraction), "resampled": resampled, "prior_share": filled / tried}
    return Posterior(fit, names, X[None], (LL + LP)[None], np.array([accepted[-1]]), np.array([scales[-1]]), 0, seed,
                     sampler="smc", smc=info)


# --- identification -------------------------------------------------------------------------------------------

def moments(system: Any, lags: int = 4) -> np.ndarray:
    """The observables' unconditional moments as one vector: their means (``d``), the upper triangle of their
    covariance matrix (measurement errors included), then the autocovariances ``Z T^j P Z'`` for j = 1 to ``lags``,
    each matrix by rows. ``system`` is a state-space form (``LinearSolution.system(...)``, a fitted model's
    ``system``); the moments are what identification differentiates."""
    from scipy import linalg

    T, R, Q, Z, H = system.T, system.R, system.Q, system.Z, system.H
    if np.max(np.abs(np.linalg.eigvals(T))) >= 1 - 1e-8:
        raise ModelError("the model has a unit root: its unconditional moments, used for identification, do not exist")
    P = linalg.solve_discrete_lyapunov(T, R @ Q @ R.T)
    gamma0 = Z @ P @ Z.T + H
    parts = [np.asarray(system.d, dtype=float).ravel(), gamma0[np.triu_indices(len(gamma0))]]
    TjP = P
    for _ in range(lags):
        TjP = T @ TjP
        parts.append((Z @ TjP @ Z.T).ravel())
    return np.concatenate(parts)


_moments = moments  # the name it had before it was public


def regime_moments(solution: Any, system: Any, regimes: Any, periods: Any, lags: int = 4, stacked: bool = False) -> np.ndarray:
    """The observables' moments over a sample through known-date regimes: with the regimes, the rules change over the
    sample, so the moments do too. From the state before the first period (the reference regime's stationary
    distribution, as the regime filter starts), the state's mean and variance go through each period's rule
    (``a(t) = c(t) + T(t) a(t-1)``, ``P(t) = T(t) P(t-1) T(t)' + R(t) Q R(t)'``); each period contributes the
    observables' means, the upper triangle of their variance and their covariances with the ``lags`` periods before
    (``Z T(t)...T(t-j+1) P(t-j) Z'``, those inside the sample), stacked period after period and divided by the square
    root of the number of periods, so that moments the same in every period weigh as one period's. With no regime
    changing in the sample (``periods``), it is ``moments(system, lags)`` exactly, unless ``stacked``: then the
    periods' moments are stacked all the same (as they are where some trial value makes a regime change something,
    so that all trial values give moments of one length)."""
    import pandas as pd

    from .regimes import _matrices, regime_plan

    periods = periods if isinstance(periods, pd.PeriodIndex) else pd.PeriodIndex(periods)
    _, plan = regime_plan(solution, regimes, periods, where="sample")
    if plan is None and not stacked:
        return moments(system, lags)
    if np.max(np.abs(np.linalg.eigvals(system.T))) >= 1 - 1e-8:
        raise ModelError("the model has a unit root: its unconditional moments, used for identification, do not exist")
    n = len(periods)
    if plan is None:
        Ts, cs, Rs = np.repeat(system.T[None], n, 0), np.zeros((n, system.T.shape[0])), np.repeat(system.R[None], n, 0)
    else:
        Ts, cs, Rs = _matrices(plan, system, n)
    a0, P0, _ = system.start("auto")
    Z, H, Q, d = system.Z, system.H, system.Q, np.asarray(system.d, dtype=float).ravel()
    a, P = np.asarray(a0, float), np.asarray(P0, float)
    variances, parts = [], []
    upper = np.triu_indices(len(Z))
    for t in range(n):
        a = cs[t] + Ts[t] @ a
        P = Ts[t] @ P @ Ts[t].T + Rs[t] @ Q @ Rs[t].T
        variances.append(P)
        parts += [d + Z @ a, (Z @ P @ Z.T + H)[upper]]
        product = np.eye(len(a))
        for j in range(1, min(lags, t) + 1):
            product = product @ Ts[t - j + 1]
            parts.append((Z @ product @ variances[t - j] @ Z.T).ravel())
    return np.concatenate(parts) / np.sqrt(n)


@dataclass(frozen=True, eq=False)
class Identification:
    """Iskrev's identification test at a point: see the module docstring."""

    names: tuple[str, ...]
    at: Mapping[str, float]
    jacobian: np.ndarray  # d moments / d parameters, one column per parameter
    rank: int
    singular_values: np.ndarray
    sensitivity: np.ndarray  # norm of each column scaled by the parameter's value
    collinearity: np.ndarray  # multiple correlation of each column with the others
    closest: tuple[str, ...]  # the parameter each is most collinear with
    lags: int
    pairwise: np.ndarray | None = None  # |cosine| of each column with its closest one's (1 = the same effect)
    elasticity: bool = False  # sensitivities and collinearities from the Jacobian in elasticities

    @property
    def identified(self) -> bool:
        return self.rank == len(self.names)

    @property
    def problems(self) -> tuple[str, ...]:
        """Parameters that move no moment, or that move them only as others do."""
        out = []
        top = float(np.max(self.sensitivity)) if len(self.sensitivity) else 0.0
        for n, s, r in zip(self.names, self.sensitivity, self.collinearity):
            if s <= 1e-8 * max(top, 1.0) or r >= 1 - 1e-6:
                out.append(n)
        return tuple(out)

    def summary(self) -> Table:
        rows = []
        for j, n in enumerate(self.names):
            status = "not identified" if n in self.problems else ("highly collinear" if self.collinearity[j] > 0.95 else "identified")
            pair = float(self.pairwise[j]) if self.pairwise is not None else float("nan")
            rows.append([float(self.at[n]), float(self.sensitivity[j]), float(self.collinearity[j]), self.closest[j], pair, status])
        cond = self.singular_values[0] / self.singular_values[-1] if self.singular_values[-1] > 0 else np.inf
        notes = [f"Rank of the Jacobian of the moments: {self.rank} for {len(self.names)} parameters "
                 + ("(all identified)" if self.identified else "(some parameters cannot be told apart)"),
                 f"Moments: means, variances and autocovariances of the observables to lag {self.lags}; condition number {cond:.3g}",
                 "Sensitivity: how much the moments move with the parameter (scaled by its value). Collinearity: the "
                 "multiple correlation of its effect with the other parameters' (1 = the same effect); pairwise: with "
                 "the one parameter most like it alone"]
        if self.elasticity:
            notes.append("In elasticities: each moment's change relative to its size (moments near zero left out)")
        return Table(["value", "sensitivity", "collinearity", "most like", "pairwise", "status"], rows, index=list(self.names),
                     title="Identification (Iskrev)", notes=notes)

    def __repr__(self) -> str:
        return f"<Identification: rank {self.rank} of {len(self.names)}{'' if self.identified else '; problems: ' + ', '.join(self.problems)}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


def identification(fit: Any, lags: int = 4, at: Mapping[str, float] | None = None, elasticity: bool = False) -> Identification:
    """The work behind ``FittedModel.identification``: see the module docstring."""
    if fit.evaluate is None:
        raise ModelError("this estimate cannot be checked: estimate the model again with this version of pymetrix")
    names = tuple(fit.params.names)
    point = {**fit.params.as_dict(), **dict(at or {})}
    lik = getattr(fit.evaluate, "__self__", None)
    # each value's solution started from the one at the point, solved afresh: the same as Model.identification gives
    evaluate = lik.anchored(point) if hasattr(lik, "anchored") else fit.evaluate
    regimes = getattr(fit, "regimes", ())
    if regimes:  # the moments over the sample through the regimes, at each trial value of their parameters
        import pandas as pd

        from .regimes import with_values

        index = fit.data.frame.index[fit.data.sample_mask(fit.span)]
        periods = index if isinstance(index, pd.PeriodIndex) else pd.PeriodIndex(index)

        def moments_at(values: dict[str, float], stacked: bool = False) -> np.ndarray:
            _, solution, system = evaluate(values)
            return regime_moments(solution, system, with_values(regimes, values), periods, lags, stacked)

        return identify(names, point, fit.bounds, None, lags, elasticity, moments_at)
    return identify(names, point, fit.bounds, lambda values: evaluate(values)[2], lags, elasticity)


def identify(names: tuple[str, ...], point: Mapping[str, float], bounds: Mapping[str, tuple[float, float]],
             system_at: Callable[[dict[str, float]], Any] | None, lags: int = 4, elasticity: bool = False,
             moments_at: Callable[..., np.ndarray] | None = None) -> Identification:
    """Iskrev's test of ``names`` at ``point``, with ``system_at({name: value})`` the state-space form at those values:
    what ``FittedModel``, ``Model`` and ``Template.identification`` share. ``moments_at``: the moments at those values
    instead (``regime_moments``, with known-date regimes; ``moments_at(values, stacked=True)`` stacks them period by
    period, as every value is given them when the regimes change something at some values and not at others)."""
    from ..threads import blas_threads

    if moments_at is None:
        assert system_at is not None
        moments_at = lambda values, stacked=False: moments(system_at(values), lags)  # noqa: E731
    with blas_threads():
        return _identification(tuple(names), point, bounds, moments_at, lags, elasticity)


def _identification(names: tuple[str, ...], point: Mapping[str, float], bounds: Mapping[str, tuple[float, float]],
                    moments_at: Callable[..., np.ndarray], lags: int, elasticity: bool) -> Identification:
    theta = np.array([point[n] for n in names], dtype=float)
    steps = []
    for j in range(len(names)):
        h = 1e-5 * max(abs(theta[j]), 1e-2)
        lo_b, hi_b = bounds.get(names[j], (-np.inf, np.inf))
        up, down = theta.copy(), theta.copy()
        up[j] = min(theta[j] + h, hi_b)
        down[j] = max(theta[j] - h, lo_b)
        steps.append((up, down))

    def evaluate(stacked: bool) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        def at(x: np.ndarray) -> np.ndarray:
            values = dict(zip(names, map(float, x)))
            return moments_at(values, stacked=True) if stacked else moments_at(values)

        return at(theta), [(at(up), at(down)) for up, down in steps]

    base, moved = evaluate(False)
    if any(len(v) != len(base) for pair in moved for v in pair):  # regimes that change something at some values only
        base, moved = evaluate(True)
    J = np.empty((len(base), len(names)))
    for j, ((up, down), (m_up, m_down)) in enumerate(zip(steps, moved)):
        J[:, j] = (m_up - m_down) / (up[j] - down[j])
    scaled = J * np.where(np.abs(theta) > 0, np.abs(theta), 1.0)
    if elasticity:  # each moment's change relative to its size, so the largest variances do not dominate
        keep = np.abs(base) > 1e-6 * max(float(np.max(np.abs(base))), 1e-300)
        scaled = scaled[keep] / np.abs(base[keep])[:, None]
    sv = np.linalg.svd(scaled, compute_uv=False)
    tol = max(scaled.shape) * np.finfo(float).eps * 1e4 * (sv[0] if len(sv) else 0.0)
    rank = int(np.sum(sv > tol))
    sensitivity = np.linalg.norm(scaled, axis=0)
    collinearity = np.zeros(len(names))
    pairwise = np.zeros(len(names))
    closest = []
    for j in range(len(names)):
        others = np.delete(scaled, j, axis=1)
        y = scaled[:, j]
        if others.shape[1] == 0 or not np.any(y):
            collinearity[j] = pairwise[j] = 0.0 if others.shape[1] == 0 else 1.0
            closest.append("")
            continue
        beta, *_ = np.linalg.lstsq(others, y, rcond=None)
        fitted = others @ beta
        collinearity[j] = float(np.sqrt(max(0.0, min(1.0, (fitted @ fitted) / (y @ y)))))
        cos = [abs(o @ y) / (np.linalg.norm(o) * np.linalg.norm(y)) if np.linalg.norm(o) > 0 else 0.0 for o in others.T]
        pairwise[j] = float(min(1.0, max(cos)))
        closest.append([n for i, n in enumerate(names) if i != j][int(np.argmax(cos))])
    return Identification(names, {n: float(v) for n, v in zip(names, theta)}, J, rank, sv, sensitivity, collinearity,
                          tuple(closest), lags, pairwise, elasticity)
