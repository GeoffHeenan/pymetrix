"""Template models: small, standard models to learn from, check and copy.

    from pymetrix.models import templates
    nk = templates.new_keynesian()               # three equations: output gap, inflation, policy rate
    nk.check()                                   # steady state and Blanchard-Kahn
    solution = nk.solve()                        # decision rules, impulse responses, simulations
    system = nk.system()                         # state-space form, with the template's measurement equations
    system.filter(data)                          # the unobserved variables, from data
    templates.new_keynesian(phi=1.2).model       # change a parameter; .model is an ordinary Model to edit

Templates are examples, not classes to inherit: each is a ``Model`` of ordinary equations with calibrated
parameters, the shocks, their sizes, a starting guess for the steady state, and measurement equations that
link the model to the kind of data it is usually fitted to. Copy the equations to start a model of your own.

- ``new_keynesian``: the textbook three-equation model (Galí 2015, chapter 3), variables in percent
  deviations from the steady state, quarterly. Observed: annualised inflation, the policy rate, GDP growth.
- ``rbc``: the stochastic growth model with log utility, full depreciation of the time endowment and a
  technology shock (King, Plosser and Rebelo 1988), as in the Dynare examples. Variables in levels.
- ``gap_model``: a small semi-structural model of the kind used in central banks' quarterly projection
  systems (Berg, Karam and Laxton 2006): output gap, inflation with forward- and backward-looking terms, a
  policy rule, and the real interest rate. Observed: inflation and the policy rate; the output gap is
  unobserved, which is what the Kalman filter estimates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .._api import api
from ..errors import SpecificationError
from ..estimation import Equation
from ..metadata import Param, Role
from .measurement import MeasurementMap
from .model import Model

__all__ = ["Template", "gap_model", "new_keynesian", "rbc"]


@dataclass(frozen=True, eq=False)
class Template:
    """A template model with what it needs to be solved and taken to data."""

    name: str
    model: Model
    shocks: tuple[str, ...]
    shock_sd: Mapping[str, float]
    measurement: MeasurementMap | None = None
    guess: Mapping[str, float] = field(default_factory=dict)
    description: str = ""
    labels: Mapping[str, str] = field(default_factory=dict)  # descriptions of variables, where given
    shock_corr: Mapping[Any, float] = field(default_factory=dict)  # correlations of the shocks, {(a, b): r}; none by default
    regimes: tuple[Any, ...] = ()  # known-date regimes (pymetrix.models.regimes), which solve, estimate and forecasts follow

    def check(self, data: Any = None, span: Any = None) -> Any:
        return self.model.check(data, span, shocks=self.shocks, steady=True, guess=dict(self.guess) or None)

    def steady_state(self) -> Any:
        return self.model.steady_state(guess=dict(self.guess) or None)

    @api()
    def solve(self, order: int = 1) -> Any:
        """The rational-expectations solution: first order (a ``LinearSolution``), or second order with the
        template's shock sizes (a ``SecondOrderSolution``: risk corrections, quadratic terms). A first-order solution
        keeps the template's ``regimes``: its ``solve(data, span)`` follows them unless given others."""
        if order == 2:
            from .perturbation import second_order

            return second_order(self.model, self.shocks, self.shock_sd, steady=self.steady_state(), shock_corr=self.shock_corr or None)
        if order != 1:
            raise SpecificationError("order is 1 or 2")
        sol = self.model.linearize(self.steady_state(), shocks=self.shocks).solve()
        return replace(sol, regimes=tuple(self.regimes)) if self.regimes else sol

    @api()
    def system(self, measurement: MeasurementMap | None = None, shock_sd: Mapping[str, float] | None = None,
               measurement_sd: Mapping[str, float] | None = None) -> Any:
        """The state-space form (a ``StateSpaceSystem``): the template's measurement equations and shock sizes unless
        others are given (``shock_sd`` overrides some or all of the template's), with its shocks' correlations."""
        m = measurement or self.measurement
        if m is None:
            raise SpecificationError(f"the {self.name} template has no measurement equations; give some")
        return self.solve().system(m, {**self.shock_sd, **(shock_sd or {})}, measurement_sd, shock_corr=self.shock_corr or None)

    @api()
    def identification(self, params: Any = None, lags: int = 4, at: Mapping[str, float] | None = None,
                       elasticity: bool = False, measurement: MeasurementMap | None = None, regimes: Any = None,
                       span: Any = None) -> Any:
        """Whether data on the template's observables could tell ``params`` apart, before any data exist: Iskrev's
        test at its calibration and shock sizes (or ``at``), as ``estimate(data, params).identification(lags, at)``
        gives at the same point. With a sample ``span``, through known-date ``regimes`` (default: the template's own)
        as ``Model.identification``. An ``Identification``."""
        if regimes is None and span is not None:
            regimes = self.regimes or None
        return self.model.identification(measurement or self.measurement, params, self.shocks, self.shock_sd, lags, at,
                                         elasticity, guess=dict(self.guess) or None, regimes=regimes, span=span)

    @api()
    def estimate(self, data: Any, estimate: Any = None, priors: Mapping[str, Any] | None = None, span: Any = None,
                 measurement: MeasurementMap | None = None, **kw: Any) -> Any:
        """Estimate the template's parameters and shock sizes from data (a ``FittedModel``), with its measurement
        equations and shocks; its shock sizes (and correlations, ``shock_corr``) are the starting values. Unless
        ``regimes`` are given, the template's own: those that start inside the sample are estimated with, those
        after it are kept for the fit's forecasts (``FittedModel.ahead``)."""
        kw.setdefault("shock_corr", dict(self.shock_corr) or None)
        ahead: tuple[Any, ...] = ()
        if "regimes" not in kw and self.regimes:
            kw["regimes"], ahead = _within(self.regimes, data, span)
        fit = self.model.estimate(data, measurement or self.measurement, self.shocks, self.shock_sd, estimate, priors,
                                  span=span, guess=dict(self.guess) or None, **kw)
        return replace(fit, ahead=ahead) if ahead else fit

    @api()
    def dsge_var(self, data: Any, lags: int = 4, lambda_: Any = 1.0, lambdas: Any = None, estimate: Any = None,
                 priors: Mapping[str, Any] | None = None, span: Any = None, measurement: MeasurementMap | None = None,
                 **kw: Any) -> Any:
        """A DSGE-VAR(lambda) with the template's measurement equations, shocks and shock sizes (a ``FittedDSGEVAR``):
        ``Model.dsge_var``; see ``pymetrix.models.dsge_var``."""
        kw.setdefault("shock_corr", dict(self.shock_corr) or None)
        return self.model.dsge_var(data, measurement or self.measurement, lags, lambda_, lambdas, self.shocks, self.shock_sd,
                                   estimate, priors, span=span, guess=dict(self.guess) or None, **kw)

    def __repr__(self) -> str:
        return f"<Template {self.name}: {len(self.model.endogenous)} equations, shocks {', '.join(self.shocks)}>"


def _within(regimes: Any, data: Any, span: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """The regimes that start by the sample's last period (estimated with), and those after it (for forecasts)."""
    import pandas as pd

    from .regimes import _period

    index = data.frame.index[data.sample_mask(span)]
    index = index if isinstance(index, pd.PeriodIndex) else pd.PeriodIndex(index)
    last = index[-1]
    inside = tuple(r for r in regimes if _period(r.start, index.freq, "the regime start") <= last)
    return inside, tuple(r for r in regimes if r not in inside)


def _bounded(name: str, low: float, high: float) -> Param:
    return Param(name, bounds=(low, high))


@api("pymetrix:templates.new_keynesian")
def new_keynesian(sigma: float = 1.0, beta: float = 0.99, kappa: float = 0.1, rho: float = 0.8, phi_pi: float = 1.5,
                  phi_y: float = 0.5, pi_ss: float = 2.0, r_ss: float = 1.0, growth: float = 2.0,
                  shock_sd: Mapping[str, float] | None = None, lower_bound: float | None = None) -> Template:
    """The three-equation New Keynesian model. ``phi_pi`` above one satisfies the Taylor principle;
    below one the solution is not unique (``check()`` says so). ``lower_bound`` (percent a year, e.g. 0) puts an
    effective lower bound on the policy rate: ``i ~ max(rule, i_floor)``, solved by ``occbin``."""
    params = {"sigma": sigma, "beta": beta, "kappa": kappa, "rho": rho, "phi_pi": phi_pi, "phi_y": phi_y}
    rule = "rho*i[-1] + (1 - rho)*(phi_pi*pi + phi_y*y) + ei"
    policy_params: list[Any] = [_bounded("rho", 0, 1), "phi_pi", "phi_y"]
    if lower_bound is not None:  # the annual rate r_ss + pi_ss + 4 i is at least lower_bound
        params["i_floor"] = (lower_bound - r_ss - pi_ss) / 4
        rule = f"max({rule}, i_floor)"
        policy_params.append("i_floor")
    model = Model([
        Equation("y ~ y[+1] - (1/sigma)*(i - pi[+1]) + ey", params=[_bounded("sigma", 0, 1e9)], name="IS"),
        Equation("pi ~ beta*pi[+1] + kappa*y + epi", params=[_bounded("beta", 0, 1), _bounded("kappa", 0, 1e9)], name="Phillips"),
        Equation(f"i ~ {rule}", params=policy_params, name="policy"),
    ], params=params, roles={"ey": Role.SHOCK, "epi": Role.SHOCK, "ei": Role.SHOCK})
    measurement = MeasurementMap(
        ["infl = pi_ss + 4*pi", "rate = r_ss + pi_ss + 4*i", "growth = g + 4*(y - y[-1])"],
        params={"pi_ss": pi_ss, "r_ss": r_ss, "g": growth},
        units={"infl": "percent a year", "rate": "percent a year", "growth": "percent a year"},
        labels={"infl": "inflation", "rate": "policy interest rate", "growth": "GDP growth"},
    )
    return Template("new_keynesian", model, ("ey", "epi", "ei"), dict(shock_sd or {"ey": 0.5, "epi": 0.3, "ei": 0.2}), measurement,
                    description="Output gap, inflation and the policy rate in percent deviations; quarterly.")


@api("pymetrix:templates.rbc")
def rbc(alpha: float = 0.33, beta: float = 0.99, delta: float = 0.025, shock_sd: Mapping[str, float] | None = None) -> Template:
    """The stochastic growth model: consumption, capital (end of period) and output, with a technology shock."""
    model = Model([
        Equation("c ~ c[+1] / (beta*(1 + alpha*k^(alpha - 1) - delta))", params=["alpha", "beta", "delta"], name="Euler"),
        Equation("y ~ exp(e)*k[-1]^alpha", params=["alpha"], name="production"),
        Equation("k ~ y + (1 - delta)*k[-1] - c", params=["delta"], name="capital"),
    ], params={"alpha": alpha, "beta": beta, "delta": delta}, roles={"e": Role.SHOCK})
    k = ((1 / beta - 1 + delta) / alpha) ** (1 / (alpha - 1))
    guess = {"k": k, "y": k**alpha, "c": k**alpha - delta * k}
    measurement = MeasurementMap(["log_y = log(y)", "log_c = log(c)"], labels={"log_y": "log output", "log_c": "log consumption"})
    return Template("rbc", model, ("e",), dict(shock_sd or {"e": 0.01}), measurement, guess,
                    description="Consumption, capital and output in levels, with a technology shock.")


@api("pymetrix:templates.gap_model")
def gap_model(a1: float = 0.75, a2: float = 0.1, a3: float = 0.1, b1: float = 0.25, b2: float = 0.2, g1: float = 0.7,
              g2: float = 1.5, g3: float = 0.5, rr_bar: float = 1.0, pi_target: float = 2.0,
              shock_sd: Mapping[str, float] | None = None) -> Template:
    """A small semi-structural projection model: output gap, inflation (percent a year), policy rate, real rate."""
    params = {"a1": a1, "a2": a2, "a3": a3, "b1": b1, "b2": b2, "g1": g1, "g2": g2, "g3": g3, "rr_bar": rr_bar, "pi_target": pi_target}
    model = Model([
        Equation("y_gap ~ a1*y_gap[-1] + a2*y_gap[+1] - a3*(rr[-1] - rr_bar) + e_y", params=["a1", "a2", "a3", "rr_bar"], name="aggregate demand"),
        Equation("pi ~ b1*pi[+1] + (1 - b1)*pi[-1] + b2*y_gap + e_pi", params=["b1", "b2"], name="Phillips"),
        Equation("i ~ g1*i[-1] + (1 - g1)*(rr_bar + pi[+1] + g2*(pi[+1] - pi_target) + g3*y_gap) + e_i",
                 params=["g1", "g2", "g3", "rr_bar", "pi_target"], name="policy"),
        Equation("rr = i - pi[+1]", name="real rate"),
    ], params=params, roles={"e_y": Role.SHOCK, "e_pi": Role.SHOCK, "e_i": Role.SHOCK})
    measurement = MeasurementMap(["infl = pi", "rate = i"], errors={"infl": 0.1},
                                 labels={"infl": "inflation, percent a year", "rate": "policy interest rate"})
    guess = {"y_gap": 0.0, "pi": pi_target, "i": rr_bar + pi_target, "rr": rr_bar}
    return Template("gap_model", model, ("e_y", "e_pi", "e_i"), dict(shock_sd or {"e_y": 0.5, "e_pi": 0.6, "e_i": 0.3}),
                    measurement, guess, description="Output gap (unobserved), inflation, policy rate and real rate.")
