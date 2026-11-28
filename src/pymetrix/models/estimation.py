"""Estimating a linearised model: maximum likelihood, or the posterior mode with priors.

    nk = templates.new_keynesian()
    fit = nk.estimate(data)                                     # every calibrated parameter and shock size
    fit = nk.estimate(data, estimate=["kappa", "phi_pi", "sd_ei"], priors={"kappa": Gamma(0.1, 0.05)})
    fit = model.estimate(data, measurement, shocks=["ey", "epi", "ei"], shock_sd={"ey": 0.5})

    fit.params                      # the estimates, with their covariance
    fit.parameters                  # every parameter, with its status: estimated, calibrated or fixed
    fit.shock_sd, fit.stats         # shock sizes; log likelihood (and posterior), AIC, BIC
    fit.solution, fit.system        # the model solved at the estimates, and its state-space form
    fit.smoothed                    # the model's variables estimated from the data
    fit.provenance                  # data, sample, method, fingerprint, warnings, convergence
    fit.mcmc(draws=20000)           # the posterior by Metropolis-Hastings from the mode (a Posterior: see bayes.py)
    fit.identification()            # which parameters the data's moments can pin down (Iskrev)

Each evaluation rebuilds the model with trial values, finds its steady state, linearises it, solves it for its
stable rational-expectations solution and runs the Kalman filter on the observed series of the measurement map.
Values with no steady state or no unique stable solution (Blanchard-Kahn) are ruled out. Free by default: the
calibrated parameters and the standard deviation of each shock, ``sd_<shock>``; measurement-error standard
deviations ``sd_<series>`` are free only if named in ``estimate``. Correlations of the shocks (``shock_corr``,
``{("ey", "ei"): 0.3}``) are ``corr_<shock>_<shock>`` (the two in the model's order of shocks), within (-1, 1): free
when given in ``shock_corr`` or named in ``estimate`` (starting at 0), and penalised where the shocks' correlation
matrix is not positive definite (see ``pymetrix.models.covariance``). Coefficients estimated equation by equation
are held at their fitted values. Bounds come from each parameter's ``Param`` metadata, standard deviations are
positive, and priors come from ``Param.prior`` or ``priors``: with any prior the estimate is the posterior mode.
Standard errors are from the inverse Hessian of the log likelihood (plus log prior); a parameter at a bound gets none.
``cov="none"`` skips the Hessian: no standard errors, and ``mcmc`` starts from a diagonal proposal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize

from .._api import api, call_migration, renamed_arguments
from ..errors import SpecificationError
from ..output import Table, bounds_text
from .solvers import ModelError
from .._random import generator as make_rng, get_random_generator

__all__ = ["FittedModel", "estimate_model"]

_PENALTY = 1e12


def occbin_solution(model: Any, solution: Any, measurement: Any, shock_sd: Mapping[str, float], smoothed: Any, span: Any,
                    scenario: Any, draws: int | None, seed: int | None, history: Any, regimes: Any = None,
                    shock_corr: Mapping[Any, float] | None = None) -> Any:
    """An OccBin forecast as a ``Solution``: the linear forecast's result (its index, history and provenance) with
    OccBin's paths and shocks put in the forecast periods (``occbin_filter.occbin_forecast`` from the state in
    ``smoothed``), and the observed series through ``measurement``; with ``draws``, OccBin's draws (correlated as
    ``shock_corr``). ``regimes``: known-date regimes, combined with the constraints."""
    from .occbin_filter import occbin_forecast
    from .scenario import Scenario

    paths, binding, simulation, shocks = occbin_forecast(model, list(shock_sd), {}, smoothed, span, scenario, draws,
                                                         shock_sd if draws else None, seed, regimes=regimes or None,
                                                         shock_corr=shock_corr if draws else None)
    shell_scenario = Scenario(set=scenario.set, shift=scenario.shift, name=scenario.name) if scenario is not None else None
    base = solution.solve(smoothed, span, shell_scenario)
    frame = base.frame.copy()
    frame.loc[paths.index, paths.columns] = paths.to_numpy()
    if scenario is not None and scenario.free:  # the free shocks OccBin solved for
        steady = dict(solution.model.steady.values)
        for s in scenario.free:
            frame.loc[shocks.index, s] = shocks[s].to_numpy() + steady.get(s, 0.0)
    for name, values in measurement.report(replace(base, frame=frame), history).frame.items():
        frame[name] = values
    spells = {c: [str(p) for p in binding.index[binding[c].to_numpy()]] for c in binding.columns}
    method = {**base.provenance.method, "type": "piecewise-linear (OccBin)", "binding": spells}
    if scenario is not None and scenario.hold:
        method["targets"] = "held by the free shocks through the piecewise solution (Newton's method on the shocks)"
    if regimes:
        from .regimes import as_regimes, regimes_text

        method["known regimes"] = regimes_text(as_regimes(regimes))
    out = replace(base, frame=frame, scenario=scenario if scenario is not None else base.scenario,
                  provenance=replace(base.provenance, method=method))
    if draws:
        from .stochastic import observables

        out = replace(out, simulation=observables(measurement, out, simulation, history))
    return out


@dataclass(frozen=True, eq=False)
class FittedModel:
    """A linearised model estimated from data: see the module docstring."""

    model: Any  # the Model, rebuilt at the estimates
    params: Any  # PointEstimate of the estimated parameters (values and covariance)
    parameters: Any  # ParameterSet: every structural parameter with its status
    shock_sd: Mapping[str, float]
    measurement: Any  # the MeasurementMap, with any estimated measurement errors
    solution: Any  # LinearSolution at the estimates
    system: Any  # StateSpaceSystem at the estimates
    smoothed: Any  # Dataset: the model's variables from the data, in levels
    stats: Mapping[str, float]
    priors: Mapping[str, Any] = field(default_factory=dict)
    bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    method: str = "maximum likelihood"
    provenance: Any = None
    evaluate: Any = field(default=None, repr=False)  # {name: value} -> (model, solution, system) at those values
    log_posterior: Any = field(default=None, repr=False)  # values in params.names order -> log likelihood + log prior (-inf if impossible)
    data: Any = field(default=None, repr=False)
    span: Any = field(default=None, repr=False)
    regimes: tuple[Any, ...] = ()  # known-date regimes, at the estimates (see pymetrix.models.regimes)
    shock_corr: Mapping[tuple[str, str], float] = field(default_factory=dict)  # correlations of the shocks, at the estimates
    ahead: tuple[Any, ...] = ()  # regimes after the sample (a template's), which forecasts follow
    nonlinear: Any = None  # estimated with the particle filter: the solution it ran on (a SecondOrderSolution at order 2)
    particle: Any = None  # estimated with the particle filter: its result at the estimates (a ParticleFilter)

    @property
    def _corr(self) -> Mapping[tuple[str, str], float] | None:
        return getattr(self, "shock_corr", None) or None  # (a fit saved before correlations had none)

    def summary(self) -> Table:
        p = self.params
        from .system import prior_text

        rows = []
        spec_of = getattr(self.model, "spec", lambda name: None)
        for name, value, se in zip(p.names, p.values, p.se):
            low, high = self.bounds.get(name, (-np.inf, np.inf))
            spec = spec_of(name)
            rows.append([float(value), float(se), prior_text(self.priors.get(name)), bounds_text(low, high), spec.label if spec else ""])
        s = self.stats
        filter_name = "particle filter" if getattr(self, "particle", None) is not None else "Kalman filter"
        notes = [f"Method: {self.method}, {filter_name} on {', '.join(self.measurement.observables)}",
                 f"Sample: {self.provenance.sample[0]} to {self.provenance.sample[1]}, {int(s['nobs'])} observations" if self.provenance else "",
                 f"Log likelihood: {s['loglike']:.4f}", *([f"Log posterior: {s['logpost']:.4f}"] if "logpost" in s else []),
                 f"Akaike info criterion: {s['aic']:.3f}", f"Schwarz criterion: {s['bic']:.3f}",
                 "Converged" if self.provenance is None or self.provenance.converged else "Did not converge: check the estimates and starting values"]
        if self.provenance is not None and self.provenance.method.get("covariance") == "none":
            notes.append("Standard errors not computed (cov='none')")
        note = _starts_note(self.provenance.method.get("starts", []) if self.provenance is not None else [])
        if note:
            notes.append(note)
        if self.regimes:
            from .regimes import regimes_text

            notes.append("Regimes: " + regimes_text(self.regimes))
        if getattr(self, "ahead", ()):
            from .regimes import regimes_text

            notes.append("Regimes after the sample, for forecasts: " + regimes_text(self.ahead))
        if self.provenance is not None and self.provenance.warnings:
            notes.append("Warnings: " + "; ".join(self.provenance.warnings))
        return Table(["estimate", "std err", "prior", "bounds", "description"], rows, index=list(p.names), title=f"Estimated model ({self.method})",
                     notes=[n for n in notes if n])

    @api()
    def forecast(self, span: tuple[Any, Any], scenario: Any = None, data: Any = None, tunes: Any = None,
                 draws: int | None = None, seed: int | None = None, occbin: bool | None = None, tune_sd: Any = None,
                 regimes: Any = None) -> Any:
        """Forecasts from the end of the data: the model's variables start at their Kalman-smoothed values (from
        ``data`` if given, filtered again with the estimated model; otherwise the data it was estimated on), follow
        the solution under ``scenario`` (shock paths, anticipated or not; targets held by free shocks), and the
        observed series are added in the data's units through the measurement equations. ``tunes`` imposes
        judgement on the filtered variables (``{"y": {"2024Q4": -1.0}}``; give ``data`` with it), exact unless given a
        standard deviation (``tune_sd``, or a ``(value, sd)`` pair; see ``StateSpaceSystem.filter``). A ``Solution``.
        With ``draws``, a stochastic simulation too: the estimated shocks drawn in each period of the span (at the
        estimates: parameters are not drawn), and the observed series of each draw (``Solution.fan``, ``quantiles``).

        ``occbin``: the model's occasionally binding constraints (its max or min) respected, by OccBin from the
        OccBin-smoothed state (``data`` filtered again with the OccBin filter); default: when the model was estimated
        with the OccBin filter. Scenarios give shock paths (``set``, ``shift``), anticipated unless all are named
        unanticipated, and targets held by free shocks (``hold``, ``free``: anticipated, without draws), solved for
        through the piecewise solution; draws are surprises in each period. ``Solution.provenance.method["binding"]``
        says when each constraint binds.

        ``regimes``: parameters or equations that change at known dates in the forecast (``Regime`` objects or text such
        as ``"2025Q1..2026Q4: rho=1, phi_pi=0"``), after the model's own regimes (those it was estimated with, the last
        of which holds at the end of the data) and those after the sample it keeps (``ahead``, a template's); see
        ``pymetrix.models.regimes``."""
        from .regimes import as_regimes

        if occbin is None:
            occbin = str(self.provenance.method.get("filter", "") if self.provenance else "").startswith("OccBin")
        regimes = (*self.regimes, *getattr(self, "ahead", ()), *as_regimes(regimes))
        if occbin:
            return self._occbin_forecast(span, scenario, data, tunes, draws, seed, regimes)
        if tunes and data is None:
            raise SpecificationError("give the data to filter again with the tunes: forecast(span, data=..., tunes=...)")
        if data is None:
            smoothed = self.smoothed
        elif self.regimes:  # newer data, through the model's regimes
            if tunes:
                raise SpecificationError("tunes are not available with regimes")
            from .regimes import smooth

            smoothed = smooth(self.solution, self.system, self.regimes, data, shocks=False).smoothed
        else:
            smoothed = self.system.filter(data, tunes=tunes, tune_sd=tune_sd)
        path = self.solution.solve(smoothed, span, scenario, draws=draws, shock_sd=self.shock_sd if draws else None, seed=seed,
                                   regimes=regimes or None, shock_corr=self._corr if draws else None)
        history = self.data if data is None else data
        frame = path.frame.copy()
        for name, values in self.measurement.report(path, history).frame.items():  # observed, reported, raw rebuilt
            frame[name] = values
        out = replace(path, frame=frame)
        if draws:
            from .stochastic import observables

            out = replace(out, simulation=observables(self.measurement, out, path.simulation, history))
        return out

    def _occbin_forecast(self, span: Any, scenario: Any, data: Any, tunes: Any, draws: Any, seed: Any, regimes: Any = ()) -> Any:
        """``forecast`` with the constraints: the linear forecast's result, its paths replaced by OccBin's; through
        the known-date ``regimes`` (the model's own and those given)."""
        from ..data import Dataset
        from .occbin_filter import occbin_filter

        if tunes:
            raise SpecificationError("tunes are for the linear filter: forecast with occbin=False to use them")
        if data is not None:
            smoothed = occbin_filter(self, data).smoothed
            smoothed = Dataset.from_frame(smoothed.frame.drop(columns=list(self.shock_sd)))
        elif str(self.provenance.method.get("filter", "") if self.provenance else "").startswith("OccBin"):
            smoothed = self.smoothed
        else:  # estimated with the linear filter: the state the OccBin filter gives at these estimates
            smoothed = occbin_filter(self, self.data, span=self.span).smoothed
        return occbin_solution(self.model, self.solution, self.measurement, self.shock_sd, smoothed, span, scenario, draws, seed,
                               self.data if data is None else data, regimes, self._corr)

    @renamed_arguments(target="ess_fraction")
    @api(version=2)  # 2: target became ess_fraction
    def smc(self, particles: int = 2000, steps: int = 2, ess_fraction: float = 0.5, seed: int | None = None,
            progress: Any = None, workers: int | None = None) -> Any:
        """The posterior distribution by sequential Monte Carlo (a ``Posterior``, as ``mcmc`` gives): particles from the
        prior moved to the posterior through tempered likelihoods, as Dynare 6's (Herbst and Schorfheide 2014), with
        the log marginal data density from the same run. More robust than Metropolis-Hastings where the posterior has
        several modes or ridges; every estimated parameter needs a prior. ``ess_fraction``: the share of the particles
        the weights' conditional effective sample size is kept at in each tempering step (higher: more, smaller
        steps); ``target`` is its old name, deprecated. See ``pymetrix.models.bayes.smc``."""
        from .bayes import smc

        return smc(self, particles, steps, ess_fraction, seed, progress, workers=workers)

    @api()
    def decompose(self, start: str = "first", occbin: bool | None = None) -> Any:
        """The Kalman-smoothed history of the model's variables split into the contributions of each shock and of
        the starting point (a ``Decomposition``); ``start`` as in ``LinearSolution.decompose``. ``occbin`` (default:
        when the model was estimated with the OccBin filter): the OccBin-smoothed history, decomposed with the
        regimes held at the smoothed ones, the binding regimes' constants a part of their own
        (``OccBinFilter.decompose``). A model estimated with known-date regimes is decomposed through their
        time-varying rules, the regimes' constants a part of their own, ``"regimes"`` (``RegimeFilter.decompose``)."""
        if occbin is None:
            occbin = str(self.provenance.method.get("filter", "") if self.provenance else "").startswith("OccBin")
        if occbin:
            from .occbin_filter import occbin_filter

            if start != "first":
                raise SpecificationError("OccBin decompositions start from the first period's state: start='first'")
            method = self.provenance.method if self.provenance else {}
            return occbin_filter(self, self.data, span=self.span, start=method.get("filter_start", "auto"),
                                 presample=int(method.get("presample", 0) or 0)).decompose()
        if self.regimes and _changes(self.solution, self.regimes, self.data.frame.index[self.data.sample_mask(self.span)]):
            from .regimes import smooth

            method = self.provenance.method if self.provenance else {}
            return smooth(self.solution, self.system, self.regimes, self.data, self.span, method.get("filter_start", "auto"),
                          int(method.get("presample", 0) or 0)).decompose(start)
        return self.solution.decompose(self.smoothed, start)

    @api()
    def contributions(self, horizon: int = 0, data: Any = None, tunes: Any = None, tune_sd: Any = None) -> Any:
        """How much each observed series (and each tune) contributed to the smoothed estimates of the model's
        variables, and with ``horizon`` to the forecast from the end of the data (a ``Decomposition``): what moved
        the output gap, the GDP release or inflation? ``data``: newer data, filtered with the estimated model."""
        return self.system.contributions(self.data if data is None else data, self.span if data is None else None, tunes, horizon,
                                         tune_sd)

    def moments(self, lags: int = 4, regime: str | None = None) -> Any:
        """Theoretical moments at the estimates, with the estimated shock sizes (``Moments``). A model estimated with
        known-date regimes has the moments of one regime at a time: ``regime`` names it (its name, or its start), held
        for good at its own steady state (its own stationary solution); default: the base model, before any regime."""
        return self._regime_solution(regime).moments(self.shock_sd, lags, self._corr)

    @api()
    def fevd(self, periods: int = 20, regime: str | None = None) -> Any:
        """Forecast-error variance decomposition at the estimates (an ``FEVD``); ``regime`` as in ``moments``."""
        return self._regime_solution(regime).fevd(periods, self.shock_sd, self._corr)

    def _regime_solution(self, regime: str | None) -> Any:
        """The solution of one known-date regime, held for good (``regimes.regime_solution``), or the base's."""
        if regime is None:
            return self.solution
        from .regimes import regime_solution

        regimes = (*self.regimes, *getattr(self, "ahead", ()))
        if not regimes and regime != "base":
            raise ModelError("the model was estimated without regimes: leave out regime=")
        freq = self.data.frame.index.freq if self.data is not None and isinstance(self.data.frame.index, pd.PeriodIndex) else None
        return regime_solution(self.solution, regimes, str(regime), freq)

    @api()
    def mcmc(self, draws: int = 10000, chains: int = 2, burn: int | None = None, scale: float | None = None,
               seed: int | None = None, progress: Any = None, workers: int | None = None, adapt: bool = True,
               modes: Any = None, jump: float = 0.05, sampler: str = "rwmh", target_accept: float = 0.8, max_depth: int = 10,
               gradient: str = "auto", metric: str = "dense") -> Any:
        """The posterior distribution by random-walk Metropolis-Hastings, started around the posterior mode (a
        ``Posterior``). Every estimated parameter needs a prior. ``draws`` are kept per chain after ``burn``
        (default: as many again, discarded, while the proposal's scale is tuned). ``workers``: how many chains run at
        once in other processes (default: one per chain when the run is long enough to pay; 1 runs them here, one
        after another). The draws are the same either way. ``adapt``: halfway through burn-in, the proposal takes the
        shape of the chain's own draws (see ``pymetrix.models.bayes``). ``modes``: other posterior modes to jump
        between ({name: value} each, or arrays in ``params.names`` order); by default the distinct modes the estimate's
        starts found (``estimate(..., starts=n)``), and ``modes=False`` for none. ``jump``: the share of steps that
        propose a jump when there are several modes. ``sampler="nuts"``: the No-U-Turn sampler instead, ``target_accept``
        and ``max_depth`` its settings (``burn`` its warmup, ``workers`` as above; ``scale``, ``adapt``, ``modes`` and
        ``jump`` are Metropolis-Hastings's), with the analytic gradient of the Kalman filter's likelihood where it
        applies (``gradient="auto"``; ``"analytic"`` insists, ``"numerical"`` takes central differences; see
        ``pymetrix.models.kalman_score``), and ``metric`` its mass matrix: ``"dense"`` (the default) or ``"diag"``
        adapted in warmup from the inverse Hessian at the mode, or ``"mode"`` that inverse Hessian kept (a short warmup
        then suffices); see ``pymetrix.models.bayes_nuts``."""
        from .bayes import mcmc

        return mcmc(self, draws, chains, burn, scale, seed, progress, workers, adapt, modes, jump, sampler, target_accept, max_depth,
                    gradient, metric)

    @api()
    def gradient(self, at: Mapping[str, float] | None = None, method: str = "analytic") -> pd.DataFrame:
        """The gradient of the log likelihood and of the log posterior in the estimated parameters, at the estimates
        or at ``at={name: value}`` (the others at the estimates): one row per parameter. ``method="analytic"``
        differentiates the steady state, the solution and the Kalman filter exactly (``pymetrix.models.kalman_score``;
        refused where it does not apply); ``"numerical"`` takes central differences of the log posterior."""
        from .kalman_score import gradient_frame

        return gradient_frame(self, at, method)

    def laplace(self) -> float:
        """Log marginal data density by the Laplace approximation at the posterior mode (needs priors)."""
        from .bayes import laplace

        return laplace(self)

    @api()
    def identification(self, lags: int = 4, at: Any = None, elasticity: bool = False) -> Any:
        """Whether the data's means and autocovariances (to ``lags``) can tell the estimated parameters apart, at the
        estimates or at other values ``at={name: value}`` (an ``Identification``: Iskrev's rank test, with
        sensitivities, collinearities and each parameter's closest pair). ``elasticity=True`` measures each moment's
        change relative to its size, so that the largest variances do not make every pair of parameters look alike.
        ``Model.identification`` and ``Template.identification`` give the same without data."""
        from .bayes import identification

        return identification(self, lags, at, elasticity)

    def __repr__(self) -> str:
        return f"<FittedModel: {len(self.params.names)} estimated by {self.method}, log likelihood {self.stats['loglike']:.3f}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


def _scale(prior: Any, start: float) -> float:
    """The unit a parameter is optimised in: its prior's standard deviation when finite, else a tenth of its size."""
    if prior is not None:
        sd = getattr(prior, "sd", None)
        if sd is None and hasattr(prior, "low"):
            sd = (prior.high - prior.low) / np.sqrt(12)
        if sd is not None and np.isfinite(sd) and sd > 0:
            return float(sd)
    return float(max(abs(start), 1e-2) * 0.1)


@dataclass(frozen=True, eq=False)
class _Likelihood:
    """The log likelihood (plus log prior) of a linearised model at trial values: what estimation maximises, and what
    the fitted model keeps for sampling and identification. A class rather than local functions, so fitted models
    can be saved (pickled) with a project's results."""

    model: Any
    params: Any
    structural: Mapping[str, float]
    shocks: tuple[str, ...]
    sd: Mapping[str, float]
    errors: Mapping[str, float]
    measurement: Any
    free: tuple[str, ...]
    priors: Mapping[str, Any]
    limits: Mapping[str, tuple[float, float]]
    data: Any
    span: Any
    guess: Mapping[str, float]
    growth: bool
    start: str = "auto"  # the Kalman filter's start: see SystemMatrices.start
    presample: int = 0  # periods filtered but left out of the likelihood
    occbin: bool = False  # the likelihood of the OccBin filter (occasionally binding constraints), not the linear one
    horizon: int = 40  # OccBin: how far ahead each period's regimes are worked out
    regimes: tuple[Any, ...] = ()  # known-date regimes; their values estimated as <parameter>@<regime>
    corr: Mapping[str, float] = field(default_factory=dict)  # correlations of the shocks by name (corr_<a>_<b>), at their values
    corr_pairs: Mapping[str, tuple[str, str]] = field(default_factory=dict)  # each correlation's two shocks
    order: int = 1  # the solution's order: 2 takes the second-order (pruned) solution, with the particle filter
    particle: Mapping[str, Any] = field(default_factory=dict)  # the particle filter's options (particles, seed, proposal, ...); empty: Kalman
    gradient: str = "auto"  # the optimiser's gradient: "auto" (analytic where it applies), "analytic" or "numerical"
    diffuse_terms: str = "include"  # diffuse states: their periods' log likelihood terms in (exact diffuse) or out (see pymetrix.statespace)

    def shock_corr(self, values: Mapping[str, float]) -> dict[tuple[str, str], float]:
        """The shocks' correlations at these values (none for a likelihood saved before correlations)."""
        corr = getattr(self, "corr", None) or {}
        return {self.corr_pairs[n]: float(values.get(n, v)) for n, v in corr.items()}

    def measurement_at(self, values: Mapping[str, float]) -> Any:
        """The measurement map with its errors' sizes (``sd_<series>``) and correlations (``corr_<series>_<series>``) at
        these values."""
        from .covariance import corr_pairs

        errs = {o: values.get(f"sd_{o}", self.errors[f"sd_{o}"]) for o in self.measurement.observables}
        pairs = corr_pairs(self.measurement.observables)
        corr = {pairs[n]: float(values.get(n, v)) for n, v in self.errors.items() if n in pairs}
        return self.measurement.with_errors(errs, corr or None)

    def build(self, values: Mapping[str, float], start: Any = None, fresh: bool = False) -> tuple[Any, Any, Any]:
        """The model, its solution and its state-space form at these values.

        The solution is iterated from a nearby one (``LinearModel.solve(start=...)``) to a tolerance, so its last
        digits depend on where it started. Without ``start``, that is the last solution built here, and this one
        becomes the next start (an estimation's trial values). With ``start`` (``FittedModel.evaluate`` passes the
        solution at the estimates), it starts there and leaves the last solution alone: the result depends on the
        values only, not on what was evaluated before (views, responses, Metropolis-Hastings in between). ``fresh``:
        solved from no start at all, the last solution left alone too."""
        # the same model at these values, sharing its compiled structure (fixed and fitted parameters kept)
        m = self._base().with_parameters({n: v for n, v in values.items() if n in self.structural})
        ss = m.steady_state(guess=dict(self.guess) or None, growth=self.growth)
        linear = m.linearize(ss, shocks=self.shocks)
        if start is not None or fresh:
            solution = linear.solve(start=start)
        else:
            solution = linear.solve(start=self.__dict__.get("_warm"))
            object.__setattr__(self, "_warm", solution)  # the next trial value starts from this solution
        sds = {s: values.get(f"sd_{s}", self.sd[f"sd_{s}"]) for s in self.shocks}
        system = solution.system(self.measurement_at(values), sds, shock_corr=self.shock_corr(values) or None)
        return m, solution, system

    def build_occbin(self, values: Mapping[str, float]) -> tuple[Any, Any]:
        """The model's piecewise solution (``occbin._Piecewise``) and its reference regime's state-space form at these
        values."""
        from .occbin import _Piecewise

        m = self._base().with_parameters({n: v for n, v in values.items() if n in self.structural})
        regimes, first, freq, length = getattr(self, "regimes", ()), None, None, 1
        if regimes:  # known-date regimes in the sample, at these values
            from .regimes import with_values

            index = self.data.frame.index[self.data.sample_mask(self.span)]
            index = index if isinstance(index, pd.PeriodIndex) else pd.PeriodIndex(index)
            regimes, first, freq, length = with_values(regimes, values), index[0], index.freq, len(index)
        pw = _Piecewise(m, self.shocks, dict(self.guess), self.horizon, regimes=regimes, first=first, freq=freq, length=length)
        pw.tie = True
        sds = {s: values.get(f"sd_{s}", self.sd[f"sd_{s}"]) for s in self.shocks}
        return pw, pw.solution.system(self.measurement_at(values), sds, shock_corr=self.shock_corr(values) or None)

    def build_nonlinear(self, values: Mapping[str, float]) -> tuple[Any, Any, dict[str, float], Any]:
        """The model, its solution of the estimation's order (a ``SecondOrderSolution`` at order 2), the shock sizes and
        the measurement map with its errors, at these values: what the particle filter runs on."""
        m = self._base().with_parameters({n: v for n, v in values.items() if n in self.structural})
        ss = m.steady_state(guess=dict(self.guess) or None, growth=self.growth)
        sds = {s: values.get(f"sd_{s}", self.sd[f"sd_{s}"]) for s in self.shocks}
        if getattr(self, "order", 1) == 2:
            from .perturbation import second_order

            solution = second_order(m, self.shocks, sds, steady=ss, shock_corr=self.shock_corr(values) or None)
        else:
            solution = m.linearize(ss, shocks=self.shocks).solve()
        return m, solution, sds, self.measurement_at(values)

    def particle_loglike(self, values: Mapping[str, float]) -> float:
        """The particle filter's log likelihood at these values, with the estimation's seed (common random numbers),
        or the seed a pseudo-marginal chain has just set (``particle.PseudoMarginal``)."""
        from .particle import _Engine, rules_of, run

        _, solution, sds, measurement = self.build_nonlinear(values)
        if self.order == 2:
            rules = rules_of(solution)
        else:
            rules = rules_of(solution, sds, self.shock_corr(values) or None)
        engine = _Engine(rules, measurement)
        observed = self.__dict__.get("_observed_pf")
        if observed is None:
            frame = self.measurement.observed(self.data, self.span)
            trends = self.measurement.trend_values(frame.index) if self.measurement.trend_exprs else np.zeros(frame.shape)
            observed = (frame.to_numpy(float), trends)
            object.__setattr__(self, "_observed_pf", observed)
        o = self.particle
        errors = np.array([float(measurement.errors.get(n, 0.0)) for n in measurement.observables])
        seed = self.__dict__.get("_pf_seed", o.get("seed", 0))
        return float(run(engine, observed[0], errors, observed[1], int(o["particles"]), seed, float(o.get("resample", 0.5)),
                         str(o.get("proposal", "bootstrap")), o.get("burn"), self.presample)["loglike"])

    def loglike(self, values: Mapping[str, float]) -> float:
        """The log likelihood at these values: the Kalman filter's, the OccBin filter's with ``occbin``, or the particle
        filter's (``particle``)."""
        if getattr(self, "particle", None):
            return self.particle_loglike(values)
        if self.occbin:
            from .occbin_filter import piecewise_filter

            pw, system = self.build_occbin(values)
        else:
            _, solution, system = self.build(values)
        observed = self.__dict__.get("_observed")
        if observed is None:  # the data do not change between trial values: take the observed series once
            observed = system.observations(self.data, self.span)
            object.__setattr__(self, "_observed", observed)
        if self.occbin:
            return piecewise_filter(pw, system, observed[0], self.start, self.presample)["loglike"]
        if self.regimes:
            from .regimes import regime_loglike, with_values

            return regime_loglike(solution, system, with_values(self.regimes, values), observed[0], observed[1],
                                  self.start, self.presample)
        return system.loglike_observed(*observed, start=self.start, presample=self.presample,
                                       diffuse_terms=getattr(self, "diffuse_terms", "include"))

    def _base(self) -> Any:
        """The model at the starting parameter values, built once: every trial value is a ``with_parameters`` copy."""
        base = self.__dict__.get("_base_model")
        if base is None:
            from .model import Model

            base = Model(self.model.equations, params=self.params, roles=self.model.roles)
            object.__setattr__(self, "_base_model", base)
        return base

    def __getstate__(self) -> dict[str, Any]:
        """Saved without the last solution (a starting point only, and large) and the cached model (rebuilt)."""
        return {k: v for k, v in self.__dict__.items() if k not in ("_warm", "_base_model")}

    def evaluator(self, solution: Any) -> Any:
        """``FittedModel.evaluate`` (``evaluate``), each build starting from ``solution``, the one at the estimates."""
        object.__setattr__(self, "_estimates", solution)
        return self.evaluate

    def evaluate(self, values: Mapping[str, float]) -> tuple[Any, Any, Any]:
        """``build`` started from the solution at the estimates: the same at the same values whatever was evaluated
        before (a saved result recomputes exactly the numbers it showed), and the estimation's warm start untouched."""
        return self.build(values, start=self.__dict__.get("_estimates"))

    def anchored(self, point: Mapping[str, float]) -> Callable[[Mapping[str, float]], tuple[Any, Any, Any]]:
        """``build`` at values near ``point``, each started from the solution at ``point`` solved afresh: what an
        identification test evaluates, so its finite differences depend on the values only, the same from a fitted
        model as from the model before any data (whatever either evaluated before), and the warm starts untouched."""
        anchor = self.build(point, fresh=True)[1]
        return lambda values: self.build(values, start=anchor)

    def forget(self) -> None:
        """Drop the last solution, so the next evaluation starts afresh (each Metropolis-Hastings chain does, so
        chains give the same draws in this process or another)."""
        self.__dict__.pop("_warm", None)

    def log_prior(self, values: Mapping[str, float]) -> float:
        return float(sum(prior.logpdf(values[n]) for n, prior in self.priors.items()))

    def objective(self, x: np.ndarray) -> float:
        """Minus the log posterior (log likelihood without priors), or a large penalty where it does not exist."""
        values = dict(zip(self.free, map(float, x)))
        if any(not (self.limits[n][0] <= values[n] <= self.limits[n][1]) for n in self.free):
            return _PENALTY
        try:
            value = self.loglike(values) + self.log_prior(values)
        except (ModelError, ValueError, np.linalg.LinAlgError, ZeroDivisionError, OverflowError):
            return _PENALTY
        return -value if np.isfinite(value) else _PENALTY

    def log_posterior(self, x: Any) -> float:
        value = self.objective(np.asarray(x, dtype=float))
        return -np.inf if value >= _PENALTY else -value

    def objective_gradient(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """``objective`` and its gradient, analytic (``pymetrix.models.kalman_score``) where it applies."""
        from .kalman_score import objective_gradient

        return objective_gradient(self, x)


def _starts_note(starts: Sequence[Mapping[str, Any]]) -> str:
    """What several starts found: how many reached the best value, and the other modes (more than 0.01 lower)."""
    if len(starts) < 2:
        return ""
    key = "log_posterior" if "log_posterior" in starts[0] else "log_likelihood"
    values = sorted((float(r[key]) for r in starts), reverse=True)
    best = values[0]
    reached = sum(v > best - 0.01 for v in values)
    others = sorted({round(v, 2) for v in values if v <= best - 0.01}, reverse=True)
    text = f"{len(starts)} starts: {reached} reached the best {key.replace('_', ' ')} ({best:.4f})"
    return text + (f"; others stopped at {', '.join(f'{v:.2f}' for v in others)} (other modes, or not converged)" if others else "")


_PARALLEL_SECONDS = 5.0  # a start taking longer than this sends the other starts to other processes (by default)
_POLISH_MAX = 8  # derivative-free polish (Nelder-Mead) only for this many parameters or fewer: hopeless in more


def _optimise(lik: _Likelihood, x0: np.ndarray, scale: np.ndarray, bounds: Sequence[tuple[float, float]], max_iter: int
              ) -> dict[str, Any]:
    """Minus the log posterior minimised from ``x0``, in units of ``scale`` relative to it: the point, the value, the
    evaluations and whether the optimiser converged. Module level, so that each start can run in its own process.

    BFGS (a full quasi-Newton picture of the curvature, like Dynare's csminwel) on unbounded transforms of the
    parameters: logistic between two bounds, exponential above one. It follows the long ridges of medium-sized models
    where limited-memory methods crawl (Smets-Wouters: Dynare's mode in 87 iterations, where L-BFGS-B took 557 to a
    lower one). Then L-BFGS-B in the parameters themselves, which settles any parameter pressed against a bound (where
    the transform is flat), and for a few parameters a derivative-free polish if that has not converged."""
    from ..progress import report

    k = len(x0)
    tried = [0]

    def scaled(z: np.ndarray) -> float:
        tried[0] += 1
        report(None, f"estimating: likelihood evaluation {tried[0]:,}")  # the app's progress bar; Cancel stops here
        return lik.objective(x0 + z * scale)

    lows = np.array([(lo - x0[i]) / scale[i] if np.isfinite(lo) else -np.inf for i, (lo, _) in enumerate(bounds)])
    highs = np.array([(hi - x0[i]) / scale[i] if np.isfinite(hi) else np.inf for i, (_, hi) in enumerate(bounds)])
    two, low_only, high_only = np.isfinite(lows) & np.isfinite(highs), np.isfinite(lows) & ~np.isfinite(highs), ~np.isfinite(lows) & np.isfinite(highs)

    def to_z(u: np.ndarray) -> np.ndarray:
        z = np.array(u, dtype=float)
        with np.errstate(over="ignore"):
            z[two] = lows[two] + (highs[two] - lows[two]) / (1 + np.exp(-u[two]))
            z[low_only] = lows[low_only] + np.exp(np.minimum(u[low_only], 700))
            z[high_only] = highs[high_only] - np.exp(np.minimum(u[high_only], 700))
        return z

    def to_u(z: np.ndarray) -> np.ndarray:
        u = np.array(z, dtype=float)
        share = np.clip((z[two] - lows[two]) / (highs[two] - lows[two]), 1e-12, 1 - 1e-12)
        u[two] = np.log(share / (1 - share))
        u[low_only] = np.log(np.maximum(z[low_only] - lows[low_only], 1e-300))
        u[high_only] = np.log(np.maximum(highs[high_only] - z[high_only], 1e-300))
        return u

    evaluations = 0
    if getattr(lik, "particle", None):
        # a particle filter's likelihood is noisy and has small jumps (resampling) even with common random numbers:
        # finite-difference gradients are no guide. Nelder-Mead on the transformed parameters (Dynare's simplex,
        # mode_compute=7), restarted from where it stopped until a restart gains nothing.
        u = to_u(np.zeros(k))
        fun = np.inf
        success = False
        for _ in range(4):
            simplex = np.vstack([u, u + 0.5 * np.eye(k)])
            run = optimize.minimize(lambda v: scaled(to_z(v)), u, method="Nelder-Mead",
                                    options={"maxiter": max_iter, "maxfev": max_iter * 2, "xatol": 1e-3, "fatol": 1e-3,
                                             "initial_simplex": simplex})
            evaluations += int(run.nfev)
            gained = fun - float(run.fun)
            if run.fun < fun:
                u, fun = run.x, float(run.fun)
            success = bool(run.success)
            if gained < 1e-2:
                break
        return {"x": x0 + to_z(u) * scale, "fun": fun, "nfev": evaluations, "success": success}
    from .kalman_score import ready

    if ready(lik, x0):  # the analytic gradient, through the transforms by the chain rule
        def scaled_jac(z: np.ndarray) -> tuple[float, np.ndarray]:
            tried[0] += 1
            report(None, f"estimating: likelihood evaluation {tried[0]:,} (with its gradient)")
            value, g = lik.objective_gradient(x0 + z * scale)
            return value, g * scale

        def dz_du(u: np.ndarray) -> np.ndarray:
            d = np.ones(len(u))
            with np.errstate(over="ignore"):
                s_ = 1 / (1 + np.exp(-u[two]))
                d[two] = (highs[two] - lows[two]) * s_ * (1 - s_)
                d[low_only] = np.exp(np.minimum(u[low_only], 700))
                d[high_only] = -np.exp(np.minimum(u[high_only], 700))
            return d

        def u_jac(u: np.ndarray) -> tuple[float, np.ndarray]:
            value, g = scaled_jac(to_z(u))
            return value, g * dz_du(u)

        first = optimize.minimize(u_jac, to_u(np.zeros(k)), jac=True, method="BFGS", options={"maxiter": max_iter, "gtol": 1e-6})
        settle, settle_jac = scaled_jac, True
    else:
        first = optimize.minimize(lambda u: scaled(to_z(u)), to_u(np.zeros(k)), method="BFGS", options={"maxiter": max_iter, "gtol": 1e-6})
        settle, settle_jac = scaled, None
    evaluations += int(first.nfev)
    z_bounds = [(None if not np.isfinite(lo) else lo, None if not np.isfinite(hi) else hi) for lo, hi in zip(lows, highs)]
    options = {"maxiter": max_iter, "maxfun": max_iter * 20, "ftol": 1e-12, "gtol": 1e-7}
    best = optimize.minimize(settle, np.clip(to_z(first.x), lows, highs), jac=settle_jac, method="L-BFGS-B", bounds=z_bounds,
                             options=options)
    evaluations += int(best.nfev)
    if best.fun > first.fun:  # the settling step made nothing better
        best.x, best.fun = to_z(first.x), first.fun
    if not best.success and k <= _POLISH_MAX:  # a few parameters on a ridge: polish without derivatives, then again
        polish = optimize.minimize(scaled, best.x, method="Nelder-Mead", bounds=z_bounds,
                                   options={"maxiter": max_iter * 10, "xatol": 1e-7, "fatol": 1e-9})
        evaluations += int(polish.nfev)
        if polish.fun <= best.fun:
            again = optimize.minimize(settle, polish.x, jac=settle_jac, method="L-BFGS-B", bounds=z_bounds, options=options)
            evaluations += int(again.nfev)
            best = again if again.fun <= polish.fun else polish
    return {"x": x0 + np.asarray(best.x) * scale, "fun": float(best.fun), "nfev": evaluations, "success": bool(best.success)}


def _start_points(lik: _Likelihood, x0: np.ndarray, scale: np.ndarray, starts: int, seed: int | None) -> list[np.ndarray]:
    """``x0`` and ``starts - 1`` more: each parameter drawn from its prior within its bounds (around ``x0`` by its
    scale where it has no prior), kept only where the model can be solved."""
    rng = make_rng(seed)
    lows = np.array([lik.limits[n][0] for n in lik.free])
    highs = np.array([lik.limits[n][1] for n in lik.free])
    points = [x0]
    for _ in range(200 * starts):
        if len(points) == starts:
            break
        x = np.empty(len(x0))
        for i, name in enumerate(lik.free):
            prior = lik.priors.get(name)
            for _ in range(100):
                v = float(prior.rvs(rng)) if prior is not None and hasattr(prior, "rvs") else x0[i] + scale[i] * rng.standard_normal()
                if lows[i] < v < highs[i]:
                    break
            else:
                v = x0[i]
            x[i] = v
        if lik.objective(x) < _PENALTY:
            points.append(x)
    if len(points) < starts:
        raise ModelError(f"only {len(points)} of {starts} starting points where the model can be solved were found")
    return points


def estimate_model(model: Any, data: Any, measurement: Any, shocks: Sequence[str] | None = None, shock_sd: Mapping[str, float] | None = None,
                   estimate: Sequence[str] | None = None, priors: Mapping[str, Any] | None = None,
                   bounds: Mapping[str, tuple[float, float]] | None = None, span: Any = None, guess: Mapping[str, float] | None = None,
                   cov: str = "classic", max_iter: int = 2000, growth: bool = False, start: str = "auto",
                   presample: int = 0, starts: int = 1, seed: int | None = None, workers: int | None = None,
                   occbin: bool = False, horizon: int = 40, regimes: Any = None,
                   shock_corr: Mapping[Any, float] | None = None, order: int = 1, filter: str | None = None,
                   particles: int = 2000, gradient: str = "auto", smooth_paths: int = 100,
                   diffuse_terms: str = "include") -> FittedModel:
    """The work behind ``Model.estimate``: see the module docstring."""
    from .._capture import capture
    from ..threads import blas_threads

    if cov not in ("classic", "none"):
        raise ModelError(f"cov={cov!r}: use 'classic' (the inverse Hessian) or 'none' (no standard errors, no Hessian)")
    with capture() as caught, blas_threads():
        fit = _estimate(model, data, measurement, shocks, shock_sd, estimate, priors, bounds, span, guess, cov, max_iter, growth,
                        start, presample, starts, seed, workers, occbin, horizon, regimes, shock_corr, order, filter, particles,
                        gradient, smooth_paths, diffuse_terms)
    if caught and fit.provenance is not None:
        fit = replace(fit, provenance=replace(fit.provenance, warnings=(*fit.provenance.warnings, *caught)))
    return fit


def _setup(model: Any, measurement: Any, shocks: Sequence[str] | None, shock_sd: Mapping[str, float] | None,
           estimate: Sequence[str] | None, priors: Mapping[str, Any] | None, bounds: Mapping[str, tuple[float, float]] | None,
           regimes: Sequence[Any] = (), shock_corr: Mapping[Any, float] | None = None
           ) -> tuple[Any, tuple[str, ...], Any, dict[str, float], dict[str, float], dict[str, float], dict[str, float], list[str],
                      dict[str, Any], dict[str, tuple[float, float]], dict[str, Any], tuple[Any, Any, Any]]:
    """What estimation and identification share: the measurement map, the shocks, the starting values of every
    candidate (calibrated parameters, sd_<shock>, sd_<series>, corr_<a>_<b>), the free ones, their priors, their limits,
    the candidates' metadata (``Param`` specs), and the correlations (by name, each name's pair, and those given).
    ``regimes`` add their values, ``<parameter>@<regime>``."""
    from .measurement import MeasurementMap

    if not isinstance(measurement, MeasurementMap):
        measurement = MeasurementMap.direct(list(measurement))
    if not measurement.observables:
        raise ModelError("the measurement map observes no series (it only reports some): estimation needs measurement "
                         "equations for observed series")
    shocks = tuple(shocks or model.compile().shocks)
    if not shocks:
        raise ModelError("name the model's shocks (shocks=[...]) or give them Role.SHOCK in roles=")
    params = model.parameters
    structural = {p.name: p.value for p in params if p.status == "calibrated"}
    sd = {f"sd_{s}": float((shock_sd or {}).get(s, 1.0)) for s in shocks}
    errors = {f"sd_{o}": float(measurement.errors.get(o, 0.0)) for o in measurement.observables}
    from .covariance import corr_pairs, normalise
    from .regimes import regime_values

    pairs = corr_pairs(shocks)
    # correlations of measurement errors, corr_<series>_<series> (the series in the map's order): free only if named
    error_pairs = {n: p for n, p in corr_pairs(measurement.observables).items() if n not in pairs}
    known = getattr(measurement, "error_corr", None) or {}
    errors.update({n: float(known.get(p, 0.0)) for n, p in error_pairs.items()})
    given = normalise(shocks, shock_corr)
    unnamed = [p for p in given if f"corr_{p[0]}_{p[1]}" not in pairs]
    if unnamed:
        raise ModelError(f"the correlation of {unnamed[0][0]} and {unnamed[0][1]} has no name of its own: corr_{unnamed[0][0]}_{unnamed[0][1]} "
                         "would name another pair of shocks too (rename a shock)")
    corr = {n: given.get(p, 0.0) for n, p in pairs.items()}
    by_regime = regime_values(regimes)
    candidates = {**structural, **sd, **errors, **corr, **by_regime}
    spec_of = getattr(model, "spec", lambda name: params[name].spec if name in params else None)
    specs = {n: spec_of(n) for n in candidates}
    held = {n for n, s in specs.items() if s is not None and s.fixed}  # Param(fixed=True): kept at its value
    given_names = [f"corr_{a}_{b}" for a, b in given]  # correlations given are estimated by default, as shock sizes
    free = list(estimate) if estimate is not None else [n for n in [*structural, *sd, *given_names, *by_regime] if n not in held]
    unknown = [n for n in free if n not in candidates]
    if unknown:
        fitted = [n for n in unknown if n in params and params[n].status != "calibrated"]
        if fitted:
            raise ModelError(f"{', '.join(fitted)} are {params[fitted[0]].status} (fitted equation by equation, or fixed); "
                             "only calibrated parameters and shock sizes are estimated here")
        raise ModelError(f"{', '.join(unknown)} are not parameters, shock sizes (sd_<shock>), correlations of shocks "
                         f"(corr_<shock>_<shock>, in the order {', '.join(shocks)}), measurement errors (sd_<series>, corr_<series>_<series>) "
                         "or regimes' values (<parameter>@<regime>) of this model")
    if not free:
        raise ModelError("nothing to estimate")
    fixed_names = set(params.by_status("fixed")) | held
    clash = sorted(set(free) & fixed_names)
    if clash:
        raise ModelError(f"{', '.join(clash)} are fixed; remove them from estimate=")
    priors = {**{n: specs[n].prior for n in free if specs.get(n) is not None and specs[n].prior is not None}, **dict(priors or {})}
    limits = {}
    for n in free:
        spec = specs.get(n)
        if spec is not None and spec.limits != (-np.inf, np.inf):
            low, high = spec.limits
        else:
            own = n.split("@", 1)[0]  # a regime's value: its parameter's bounds
            low, high = (params[n].bounds if n in params else (-1.0, 1.0) if n in corr or n in error_pairs
                         else params[own].bounds if "@" in n and own in params
                         else (-np.inf, np.inf) if "@" in n else (0.0, np.inf))
        limits[n] = tuple((bounds or {}).get(n, (low, high)))
        if n in corr or n in error_pairs:  # a correlation stays within (-1, 1), whatever bounds are given
            limits[n] = (max(limits[n][0], -1.0), min(limits[n][1], 1.0))
        support = getattr(priors.get(n), "support", (-np.inf, np.inf))  # a parameter stays where its prior is positive
        limits[n] = (max(limits[n][0], support[0]), min(limits[n][1], support[1]))
    return measurement, shocks, params, structural, sd, errors, candidates, free, priors, limits, specs, (corr, pairs, given)


def model_identification(model: Any, measurement: Any, params: Sequence[str] | None = None, shocks: Sequence[str] | None = None,
                         shock_sd: Mapping[str, float] | None = None, lags: int = 4, at: Mapping[str, float] | None = None,
                         elasticity: bool = False, guess: Mapping[str, float] | None = None, growth: bool = False,
                         regimes: Any = None, span: Any = None) -> Any:
    """The work behind ``Model.identification``: Iskrev's test at the calibration (or ``at``), without data, as
    ``FittedModel.identification`` does at the same point; with ``regimes``, over the sample ``span``."""
    from .bayes import identify, regime_moments
    from .regimes import as_regimes, check_starts, with_values

    regimes = as_regimes(regimes)
    periods, inside = None, ()
    if regimes:
        if span is None:
            raise SpecificationError("with regimes, give the sample the moments are of: span=('2000Q1', '2019Q4')")
        if growth:
            raise ModelError("regimes are for models without growth trends: not with growth=")
        from ..data.sample import as_span

        first, last = as_span(span)
        start = pd.Period(str(first))
        periods = pd.period_range(start, pd.Period(str(last), freq=start.freq), freq=start.freq)
        inside = tuple(r for r in regimes if pd.Period(str(r.start), freq=start.freq) <= periods[-1])  # later ones: forecasts'
        check_starts(inside, periods, "sample")
    elif span is not None:
        raise SpecificationError("span= is the sample of the regimes' moments: give regimes= too, or leave it out")
    measurement, shocks, model_params, structural, sd, errors, candidates, free, priors, limits, _, _ = _setup(
        model, measurement, shocks, shock_sd, params, None, None, regimes)
    lik = _Likelihood(model, model_params, structural, shocks, sd, errors, measurement, tuple(free), priors, limits, None, None,
                      dict(guess or {}), growth)
    point = {n: candidates[n] for n in free}
    unknown = sorted(set(at or {}) - set(free))
    if unknown:
        raise ModelError(f"at= gives {', '.join(unknown)}, which are not among the parameters tested ({', '.join(free)})")
    point.update({n: float(v) for n, v in (at or {}).items()})
    build = lik.anchored(point)  # each value's solution started from the one at the point, as a fitted model's test
    if periods is not None:
        def moments_at(values: dict[str, float], stacked: bool = False) -> np.ndarray:
            _, solution, system = build(values)
            return regime_moments(solution, system, with_values(inside, values), periods, lags, stacked)

        return identify(tuple(free), point, limits, None, lags, elasticity, moments_at)
    return identify(tuple(free), point, limits, lambda values: build(values)[2], lags, elasticity)


def _estimate(model: Any, data: Any, measurement: Any, shocks: Sequence[str] | None, shock_sd: Mapping[str, float] | None,
              estimate: Sequence[str] | None, priors: Mapping[str, Any] | None, bounds: Mapping[str, tuple[float, float]] | None,
              span: Any, guess: Mapping[str, float] | None, cov: str, max_iter: int, growth: bool = False, start: str = "auto",
              presample: int = 0, starts: int = 1, seed: int | None = None, workers: int | None = None, occbin: bool = False,
              horizon: int = 40, regimes: Any = None, shock_corr: Mapping[Any, float] | None = None, order: int = 1,
              filter: str | None = None, particles: int = 2000, gradient: str = "auto", smooth_paths: int = 100,
              diffuse_terms: str = "include") -> FittedModel:
    from .._capture import fingerprint
    from .._version import __version__
    from ..estimation.numdiff import hessian
    from ..estimation.params import point
    from ..metadata import Provenance
    from ..statespace import _at_optimum

    from .regimes import as_regimes, check_starts

    regimes = as_regimes(regimes)
    y_index = data.frame.index[data.sample_mask(span)]
    if regimes:
        if growth:
            raise ModelError("regimes are estimated for models without growth trends: not with growth=")
        check_starts(regimes, y_index if isinstance(y_index, pd.PeriodIndex) else pd.PeriodIndex(y_index), "sample")
    measurement, shocks, params, structural, sd, errors, candidates, free, priors, limits, specs, (corr, pairs, given) = _setup(
        model, measurement, shocks, shock_sd, estimate, priors, bounds, regimes, shock_corr)
    steady_guess = dict(guess or {})

    from ..statespace import STARTS

    if start not in STARTS:
        raise SpecificationError(f"start is one of {', '.join(STARTS)}, not {start!r}")
    if presample < 0:
        raise SpecificationError("presample is a number of periods, at least 0")
    from ..statespace import check_diffuse_terms

    if check_diffuse_terms(diffuse_terms) == "exclude" and (occbin or regimes or order == 2 or filter not in (None, "kalman")):
        raise SpecificationError("diffuse_terms='exclude' is for the Kalman filter's likelihood: not with occbin, regimes or "
                                 "the particle filter")
    if occbin and (growth or getattr(measurement, "trend_exprs", None)):
        raise ModelError("estimation with the OccBin filter is for models without growth or deterministic trends")
    particle = _particle_options(order, filter, particles, seed, occbin, regimes, growth, start, measurement, free, errors)
    from .kalman_score import GRADIENTS

    if gradient not in GRADIENTS:
        raise SpecificationError(f"gradient is one of {', '.join(map(repr, GRADIENTS))}, not {gradient!r}")
    if int(smooth_paths) < 0:
        raise SpecificationError("smooth_paths is a number of paths of the particle smoother: at least 0 (0: filtered means)")
    lik = _Likelihood(model, params, structural, shocks, sd, errors, measurement, tuple(free), priors, limits, data, span,
                      steady_guess, growth, start, int(presample), bool(occbin), int(horizon), regimes,
                      {n: v for n, v in corr.items() if n in free or v != 0}, pairs, int(order), particle, gradient, diffuse_terms)
    build, log_prior, objective = lik.build, lik.log_prior, lik.objective

    initial = {**candidates, **{n: float(s.start) for n, s in specs.items() if s is not None and s.start is not None}}
    x0 = np.array([min(max(initial[n], lo + 1e-6 if np.isfinite(lo) else -np.inf), hi - 1e-6 if np.isfinite(hi) else np.inf)
                   for n, (lo, hi) in ((n, limits[n]) for n in free)])
    if objective(x0) >= _PENALTY:
        raise ModelError("the model cannot be solved, or the data are impossible under it, at the starting values; "
                         "check the calibration, the measurement equations and the data")
    # optimise in comparable units: each parameter relative to its start, in prior standard deviations (or a tenth
    # of its size), so that parameters of very different magnitudes (0.005 and 0.99) do not stop the search early
    scale = np.array([_scale(priors.get(n), x0[i]) for i, n in enumerate(free)])
    bounds_list = [limits[n] for n in free]
    points = [x0] if starts <= 1 else _start_points(lik, x0, scale, starts, seed)
    jobs = [(lik, x, scale, bounds_list, max_iter) for x in points]
    import time

    began = time.perf_counter()
    runs = [_optimise(*jobs[0])]  # the starting values first, here: how long it takes decides where the others run
    rest = None
    if len(jobs) > 1 and (workers is None or workers > 1) and (workers is not None or time.perf_counter() - began > _PARALLEL_SECONDS):
        import os

        from .._processes import run_in_processes

        rest = run_in_processes([(_optimise, job) for job in jobs[1:]], min(len(jobs) - 1, workers or os.cpu_count() or 1), what="start")
    runs += rest if rest is not None else [_optimise(*job) for job in jobs[1:]]
    best_run = int(np.argmin([r["fun"] for r in runs]))
    best = runs[best_run]
    if not np.isfinite(best["fun"]) or best["fun"] >= _PENALTY:
        raise ModelError("no start reached a point where the model can be solved")
    converged = bool(best["success"]) or _at_optimum(objective, best["x"], bounds_list)
    best_x = np.asarray(best["x"], dtype=float)
    values = dict(zip(free, map(float, best_x)))
    start_report = [{"log_posterior" if priors else "log_likelihood": -float(r["fun"]), "evaluations": int(r["nfev"]),
                     "from": dict(zip(free, map(float, pt))), "to": dict(zip(free, map(float, r["x"])))}
                    for pt, r in zip(points, runs)]
    loglike = lik.loglike(values)  # as optimised
    m, solution, system = build(values)

    covariance = np.full((len(free), len(free)), np.nan)
    if cov == "classic":
        interior = [i for i, n in enumerate(free) if not (np.isclose(values[n], limits[n][0], atol=1e-5) or np.isclose(values[n], limits[n][1], atol=1e-5))]
        if interior:
            def f(x: np.ndarray) -> float:
                return -objective(np.array([x[interior.index(i)] if i in interior else best_x[i] for i in range(len(free))]))

            from .kalman_score import hessian_from_gradient, ready

            try:
                H = (_wide_hessian(f, best_x[interior], scale[interior], [limits[free[i]] for i in interior]) if particle
                     else hessian_from_gradient(lik, best_x, interior) if ready(lik, best_x) else hessian(f, best_x[interior]))
                sub = np.linalg.inv(-H)
                if np.all(np.diag(sub) > 0):
                    covariance[np.ix_(interior, interior)] = sub
            except (np.linalg.LinAlgError, ValueError):
                pass

    se = dict(zip(free, np.sqrt(np.where(np.diag(covariance) > 0, np.diag(covariance), np.nan))))
    updated = params.with_values({n: values[n] for n in free if n in structural}, status="estimated")
    updated = type(updated)(tuple(replace(p, se=float(se[p.name])) if p.name in se else p for p in updated))
    nobs = int(len(y_index))
    k = len(free)
    stats = {"loglike": float(loglike), "nobs": float(nobs), "k": float(k), "aic": -2 * loglike + 2 * k, "bic": -2 * loglike + k * np.log(nobs)}
    if priors:
        stats["logpost"] = float(loglike + log_prior(values))
    shock_sizes = {s: values.get(f"sd_{s}", sd[f"sd_{s}"]) for s in shocks}
    used_measurement = lik.measurement_at(values)
    sources = list(measurement.data_series())
    method = ("posterior mode" if priors else "maximum likelihood") + (" (OccBin filter)" if occbin else "")
    if particle:
        method += f" (particle filter, {'second' if order == 2 else 'first'} order)"
    provenance = Provenance(
        data_vintages={n: data.variables[n].vintage for n in sources if n in data.variables},
        spec={"equations": [e.name for e in model.compile().equations], "measurement": list(measurement.equations), "shocks": list(shocks)},
        method={"type": method, "solution": "piecewise-linear (OccBin)" if occbin else "first-order rational expectations",
                "filter": "OccBin (piecewise Kalman)" if occbin else "Kalman", "optimizer": "BFGS on transformed parameters, then L-BFGS-B",
                "gradient": _gradient_note(lik, best_x),
                "evaluations": int(sum(r["nfev"] for r in runs)), "estimated": list(free), "filter_start": start,
                "presample": int(presample), **({"diffuse_terms": diffuse_terms} if diffuse_terms != "include" else {}),
                "starts": start_report, **({"covariance": "none"} if cov == "none" else {}),
                **(_particle_method(particle, order) if particle else {})},
        sample=(str(y_index[0]), str(y_index[-1])), nobs=nobs, dropped=0, package_version=__version__,
        **({"seed": particle["seed"], "random_generator": get_random_generator()} if particle else {}),
        sample_source=data.sample_record(span)[0], sample_definition=data.sample_record(span)[1],
        data_fingerprint=fingerprint(data.frame.loc[y_index, sources]), converged=converged,
    )
    nonlinear = filtered_pf = None
    if particle:
        from .particle import particle_filter

        _, nonlinear, sizes, pf_measurement = lik.build_nonlinear(values)
        filtered_pf = particle_filter(nonlinear, data, pf_measurement, None if order == 2 else sizes, span, particle["particles"],
                                      particle["seed"], particle["resample"], particle["proposal"], particle["burn"],
                                      int(presample), shock_corr=None if order == 2 else (lik.shock_corr(values) or None),
                                      smooth_paths=int(smooth_paths))
        # the particle smoother's means (data of every period); without it, the filtered means (data to each period)
        smoothed = filtered_pf.smoothed if smooth_paths else filtered_pf.filtered
    elif occbin:
        from .occbin_filter import occbin_filter

        from ..data import Dataset

        frame = occbin_filter(m, data, used_measurement, shock_sizes, span, start, horizon, int(presample), shocks,
                              regimes=_fitted_regimes(regimes, values) or None,
                              shock_corr={p: r for p, r in lik.shock_corr(values).items() if r != 0} or None).smoothed.frame
        smoothed = Dataset.from_frame(frame.drop(columns=list(shocks)))  # the columns the linear filter gives
    elif regimes and _changes(solution, _fitted_regimes(regimes, values), y_index):
        from .regimes import smooth

        filtered = smooth(solution, system, _fitted_regimes(regimes, values), data, span, start, int(presample), shocks=False)
        smoothed = filtered.smoothed
        provenance = replace(provenance, method={**provenance.method, "regimes": [
            {"regime": name, "from": str(p[0]), "to": str(p[-1])} for name, p in _spells(filtered.regime)]})
    else:  # without regimes, or none that changes anything: the Kalman smoother, as always
        smoothed = system.filter(data, span)
    shock_correlations = {p: r for p, r in lik.shock_corr(values).items() if r != 0}
    return FittedModel(m, point(free, best_x, covariance), updated, shock_sizes, used_measurement, solution, system, smoothed, stats,
                       priors, limits, method, provenance, lik.evaluator(solution), lik.log_posterior, data, span,
                       _fitted_regimes(regimes, values), shock_correlations, nonlinear=nonlinear, particle=filtered_pf)


_PARTICLE_FILTERS = ("kalman", "particle", "guided")


def _gradient_note(lik: _Likelihood, x: np.ndarray) -> str:
    """What the provenance says about the optimiser's gradient."""
    from .kalman_score import ready, supported

    if getattr(lik, "particle", None):
        return "none (Nelder-Mead)"
    if ready(lik, x):
        return "analytic (Kalman-filter score); Hessian by central differences of it"
    reason = "asked for" if getattr(lik, "gradient", "auto") == "numerical" else (supported(lik) or "not available at the estimates")
    return f"central differences ({reason})"


def _wide_hessian(f: Any, x: np.ndarray, scale: np.ndarray, bounds: Sequence[tuple[float, float]]) -> np.ndarray:
    """The Hessian of a noisy function (a particle filter's log likelihood) by central differences over about one
    standard error of each parameter rather than a tiny step, so that the noise does not swamp the curvature: steps
    start at a tenth of each parameter's scale and are set twice to the standard error the diagonal implies."""
    k = len(x)
    h = 0.1 * np.asarray(scale, float)
    f0 = f(x)
    for _ in range(2):
        for i in range(k):
            lo, hi = bounds[i]
            h[i] = max(min(h[i], 0.9 * (x[i] - lo), 0.9 * (hi - x[i])), 1e-10)
            e = np.zeros(k)
            e[i] = h[i]
            d2 = (f(x + e) - 2 * f0 + f(x - e)) / h[i] ** 2
            if np.isfinite(d2) and d2 < 0:
                h[i] = 1.0 / np.sqrt(-d2)
    for i in range(k):
        lo, hi = bounds[i]
        h[i] = max(min(h[i], 0.9 * (x[i] - lo), 0.9 * (hi - x[i])), 1e-10)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = h[i], h[j]
            if i == j:
                H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h[i] ** 2
            else:
                H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


def _particle_options(order: int, filter: str | None, particles: int, seed: int | None, occbin: bool, regimes: Any,
                      growth: bool, start: str, measurement: Any, free: Sequence[str], errors: Mapping[str, float]) -> dict[str, Any]:
    """The particle filter's options for estimation ({} for the Kalman filter), checked: ``order`` 1 or 2,
    ``filter`` "kalman", "particle" (bootstrap) or "guided" (default: Kalman at first order, particle at second)."""
    if order not in (1, 2):
        raise SpecificationError("order is 1 (first-order solution) or 2 (second order, with the particle filter)")
    chosen = filter if filter is not None else ("particle" if order == 2 else "kalman")
    if chosen not in _PARTICLE_FILTERS:
        raise SpecificationError(f"filter is one of {', '.join(map(repr, _PARTICLE_FILTERS))}, not {filter!r}")
    if chosen == "kalman":
        if order == 2:
            raise SpecificationError("a second-order solution is not linear: estimate it with filter='particle' (or 'guided')")
        return {}
    if int(particles) < 10:
        raise SpecificationError("particles is the number of particles: at least 10")
    for given, what in ((occbin, "the OccBin filter (occbin=True)"), (regimes, "known-date regimes"), (growth, "growth trends")):
        if given:
            raise SpecificationError(f"the particle filter does not take {what}: estimate with the Kalman filter instead")
    if start not in ("auto", "stationary"):
        raise SpecificationError(f"the particle filter starts from the stationary distribution: start={start!r} is for the Kalman filter")
    lacking = [o for o in measurement.observables if not errors.get(f"sd_{o}", 0.0) > 0 and f"sd_{o}" not in free]
    if lacking:
        raise ModelError(f"the particle filter needs a measurement error for every observed series: none for {', '.join(lacking)} "
                         "(MeasurementMap(..., errors=...), or estimate sd_<series> with a positive starting value)")
    return {"particles": int(particles), "seed": 0 if seed is None else int(seed), "proposal": "guided" if chosen == "guided" else "bootstrap",
            "resample": 0.5, "burn": None}


def _particle_method(particle: Mapping[str, Any], order: int) -> dict[str, Any]:
    """What the provenance says about the particle filter."""
    return {"solution": "second-order perturbation (pruned)" if order == 2 else "first-order rational expectations",
            "filter": f"particle ({particle['proposal']}, {particle['particles']} particles, systematic resampling when the "
                      f"effective sample size is below {particle['resample']:g} of them)",
            "particles": particle["particles"], "particle_seed": particle["seed"],
            "optimizer": "Nelder-Mead on transformed parameters, restarted (the particle filter's likelihood is noisy)",
            "likelihood": "particle filter estimate with common random numbers (fixed seed): noisy, see pymetrix.models.particle"}


def _fitted_regimes(regimes: Any, values: Any) -> tuple[Any, ...]:
    from .regimes import with_values

    return with_values(regimes, values) if regimes else ()


def _changes(solution: Any, regimes: Any, index: Any) -> bool:
    """Whether the regimes change the model somewhere in the sample."""
    from .regimes import regime_plan

    return regime_plan(solution, regimes, index if isinstance(index, pd.PeriodIndex) else pd.PeriodIndex(index), "sample")[1] is not None


def _spells(regime: Any) -> list[tuple[str, list[Any]]]:
    """Runs of periods in the same regime: (name, periods)."""
    out: list[tuple[str, list[Any]]] = []
    for period, name in regime.items():
        if out and out[-1][0] == name:
            out[-1][1].append(period)
        else:
            out.append((name, [period]))
    return out


@call_migration("pymetrix:fitted_model.smc", 1)
def _smc_ess_fraction(arguments: dict[str, Any]) -> dict[str, Any]:  # version 2: target became ess_fraction
    if "target" in arguments:
        arguments["ess_fraction"] = arguments.pop("target")
    return arguments
