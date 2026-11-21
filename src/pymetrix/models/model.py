"""Model: a set of fitted, calibrated or identity equations, solved together."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from .._api import api
from .._serial import to_jsonable
from .._capture import data_fingerprint
from .._version import __version__
from ..data import Dataset
from ..errors import SpecificationError
from ..estimation import Equation, FittedEquation
from ..expr import Expr, Name, Num, diff, evaluate
from ..metadata import Param, Provenance, Role
from ..results import deprecated_alias
from .scenario import Scenario
from .solution import Solution
from .solvers import Fallback, FirstOrder, GaussSeidel, ModelError, Newton, PerfectForesight

_ZERO = Num(0.0)


@dataclass(frozen=True)
class ModelEquation:
    name: str
    dependent: str  # the variable it determines (was target)
    residual: Expr  # zero when the equation holds; parameters already substituted
    stochastic: bool

    def __setstate__(self, state: dict[str, Any]) -> None:  # pickled (in a project's cached results) before the rename
        if "target" in state:
            state["dependent"] = state.pop("target")
        self.__dict__.update(state)

    target = deprecated_alias("dependent", "target", why="the variable the equation determines; target is its old name")


@dataclass(frozen=True)
class Block:
    variables: tuple[str, ...]
    simultaneous: bool


@dataclass(frozen=True, eq=False)
class Model:
    code_positional: ClassVar[int] = 1  # generated code gives the equations by position: Model([...])
    equations: Sequence[Equation | FittedEquation]
    roles: Mapping[str, Role] | None = None
    over: str | None = None  # an entity axis: the equations solved for each entity of a panel (see solve)
    params: Mapping[str, float] | Any = None  # calibrated values for equations not estimated (a dict or a ParameterSet)
    links: Sequence[Any] = ()  # with over: variables each entity takes from the others (Link; see pymetrix.models.links)
    specs: Sequence[Param] = ()  # parameter metadata (priors, bounds, labels, symbols...), also for sd_<shock>, sd_<series>, corr_<a>_<b>

    model_equations: tuple[ModelEquation, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from ..systems import FittedSystem
        from .system import compile_model

        items: list[Any] = []
        held: list[Model] = []
        for item in self.equations:  # a fitted system brings its equations and identities
            inner = _held_model(item)
            if inner is not None:  # a model (or a template or XML model read, holding one) brings its equations
                held.append(inner)
                items.extend(inner.equations)
            else:
                items.extend([*item.equations, *item.system.identities] if isinstance(item, FittedSystem) else [item])
        object.__setattr__(self, "equations", tuple(items))
        if held:  # and its parameter values and roles, under those given here
            object.__setattr__(self, "params", _merged_params([m.params for m in held], self.params))
            roles = {k: v for m in held for k, v in (m.roles or {}).items()}
            if roles:
                object.__setattr__(self, "roles", {**roles, **(self.roles or {})})
            if not self.specs:
                object.__setattr__(self, "specs", tuple(s for m in held for s in m.specs))
        if self.links:
            from .links import Link

            object.__setattr__(self, "links", tuple(self.links))
            if not self.over:
                raise SpecificationError("links join the entities of a model solved per entity: give over= (e.g. over='country')")
            if not all(isinstance(link, Link) for link in self.links):
                raise SpecificationError("links are Link('ys', 'y', weights) objects")
        system = compile_model(self.equations, self.params, self.roles)
        if self.specs:
            system = _with_specs(system, self.specs)
            object.__setattr__(self, "specs", tuple(self.specs))
        object.__setattr__(self, "_system", system)
        residuals = system.residuals()
        object.__setattr__(self, "model_equations", tuple(ModelEquation(e.name, e.dependent, r, e.stochastic)
                                                          for e, r in zip(system.equations, residuals)))
        from .system import innovation_name

        object.__setattr__(self, "_innovations", {  # equations with MA terms: their innovations' history, from the fit
            innovation_name(item.equation.label): item.resid for item in self.equations
            if isinstance(item, FittedEquation) and item.equation.errors is not None and item.equation.errors.ma_lags})

    def __getstate__(self) -> dict[str, Any]:
        """Pickled without the structure shared by ``with_parameters`` copies (compiled code, which does not pickle;
        it is worked out again on first use)."""
        return {k: v for k, v in self.__dict__.items() if k != "_shared"}

    def with_parameters(self, values: Mapping[str, float]) -> Model:
        """The same model at other parameter values, sharing everything that does not depend on them: the compiled
        equations, their derivatives, and the structure the steady state and linearisation work out. What estimation
        and Metropolis-Hastings use at each trial value, instead of building the model again from its equations."""
        from dataclasses import replace

        system = self._system  # type: ignore[attr-defined]
        _ = self.exogenous, self.has_leads  # worked out once here, so every copy shares them
        params = system.parameters.with_values(dict(values))
        new = object.__new__(Model)
        new.__dict__.update(self.__dict__)
        # params as given: a dict stays a dict (with these values), a ParameterSet the updated set
        object.__setattr__(new, "params", {**self.params, **values} if isinstance(self.params, Mapping) else params)
        object.__setattr__(new, "_system", replace(system, parameters=params))
        object.__setattr__(new, "model_equations", _Substituted(new))
        object.__setattr__(new, "_shared", self.__dict__.setdefault("_shared", {}))
        return new

    def compile(self) -> Any:
        """The model as a ``ModelSystem``: equations as residuals with symbolic parameters, every variable with
        its role, the parameters with their values and status, and the lead and lag structure."""
        return self._system  # type: ignore[attr-defined]

    def spec(self, name: str) -> Param | None:
        """The metadata of a parameter, or of a shock size or measurement error (``sd_...``), if any was given. A regime's
        value (``phi_pi@post``, see ``pymetrix.models.regimes``) has its own spec if given, else its parameter's."""
        params = self.compile().parameters
        if name in params:
            return params[name].spec
        own = next((s for s in self.specs if s.name == name), None)
        if own is None and "@" in name:
            return self.spec(name.split("@", 1)[0])
        return own

    @property
    def parameters(self) -> Any:
        """The model's parameters (a ``ParameterSet``): estimated coefficients, named by equation, and calibrated ones."""
        return self.compile().parameters

    @api()
    def check(self, data: Dataset | None = None, span: tuple[Any, Any] | None = None, shocks: Sequence[str] = (),
              steady: bool | None = None, guess: Mapping[str, float] | None = None, scenario: Scenario | None = None) -> Any:
        """Diagnostics before solving (a ``ModelCheck``): equations and their targets, solvability of simultaneous
        blocks, parameters, timing, shocks, and with ``data`` and ``span``, whether the data cover the solution, and
        the inputs each equation lacks in each period (``ModelCheck.missing``, as ``missing_inputs`` gives it, under
        ``scenario`` if one is given). For models with leads (or ``steady=True``), also the steady state and the
        Blanchard-Kahn conditions."""
        from .system import check_model

        return check_model(self, data, span, shocks, steady, guess, scenario)

    @api()
    def missing_inputs(self, data: Dataset, span: tuple[Any, Any], scenario: Scenario | None = None) -> pd.DataFrame:
        """The inputs each equation lacks in each period of ``span``, before solving: a table with one row per period,
        equation and input (``period``, ``equation``, ``input``, the input written as ``x[-1]``), empty when every
        equation can be evaluated. Inputs are the exogenous series and, with their lags, the endogenous ones before the
        span (the starting values); the endogenous values the solve determines in the span are not inputs, nor are
        leads past it (the terminal condition gives them). ``scenario`` is applied first: its set and shifted paths,
        fixed and held variables (whose equations are not solved where they are fixed; a fixed variable without a value
        is reported against its equation), partial fixes (``Scenario(partial=...)``) and free variables."""
        return missing_inputs(self, data, span, scenario)

    @classmethod
    def from_file(cls, path: str | Path, **options: Any) -> Model:
        """Read a model file into a Model, by its extension: ``.mod`` (Dynare, ``read_mod``), ``.model`` (IRIS,
        ``read_iris``), ``.prg`` (an EViews program, ``read_eviews``) or ``.xml`` (an FRB/US-style model file,
        ``read_eviews_xml``). ``options`` go to the reader (e.g. ``params=`` for IRIS and EViews, ``mce=`` for XML).
        The reader's own result (a ``Template`` or ``DynareModel``) also keeps the shocks, their sizes and the
        measurement: call the reader directly when those are needed."""
        import importlib

        p = Path(path)
        readers = {".mod": ("dynare", "read_mod"), ".model": ("iris", "read_iris"),
                   ".prg": ("eviews", "read_eviews"), ".xml": ("eviews", "read_eviews_xml")}
        ext = p.suffix.lower()
        if ext not in readers:
            raise SpecificationError(f"cannot tell the format of {p.name!r} from its extension: Model.from_file reads "
                                     f"{', '.join(readers)} files (for EViews model text in another file, use read_eviews)")
        if not p.is_file():
            raise FileNotFoundError(f"no model file {str(p)!r}")
        module, reader = readers[ext]
        if ext == ".mod" and options:
            raise SpecificationError(f"read_mod takes no options (given {', '.join(options)})")
        return getattr(importlib.import_module(f".{module}", __package__), reader)(p, **options).model

    @property
    def endogenous(self) -> tuple[str, ...]:
        return tuple(e.dependent for e in self._system.equations)  # type: ignore[attr-defined]

    @property
    def exogenous(self) -> tuple[str, ...]:
        cached = self.__dict__.get("_exogenous")  # the equations never change: work it out once
        if cached is None:
            names = set().union(*(e.residual.variables() for e in self._system.equations))  # type: ignore[attr-defined]
            cached = tuple(sorted(names - set(self.endogenous)))
            object.__setattr__(self, "_exogenous", cached)
        return cached

    @property
    def has_leads(self) -> bool:
        cached = self.__dict__.get("_has_leads")
        if cached is None:
            cached = any(n.lag > 0 for e in self._system.equations for n in e.residual.refs())  # type: ignore[attr-defined]
            object.__setattr__(self, "_has_leads", cached)
        return cached

    def blocks(self) -> list[Block]:
        """Recursive and simultaneous blocks, in solution order (Tarjan's algorithm)."""
        endo = set(self.endogenous)
        deps = {
            e.dependent: sorted({n.id for n in e.residual.refs() if n.lag == 0 and n.id in endo})
            for e in self.model_equations
        }
        index: dict[str, int] = {}
        low: dict[str, int] = {}
        stack: list[str] = []
        on_stack: set[str] = set()
        out: list[Block] = []

        def strong(v: str) -> None:
            index[v] = low[v] = len(index)
            stack.append(v)
            on_stack.add(v)
            for w in deps[v]:
                if w not in index:
                    strong(w)
                    low[v] = min(low[v], low[w])
                elif w in on_stack:
                    low[v] = min(low[v], index[w])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                out.append(Block(tuple(sorted(comp)), len(comp) > 1))

        for v in self.endogenous:
            if v not in index:
                strong(v)
        return out

    def steady_state(self, exogenous: Mapping[str, float] | None = None, guess: Mapping[str, float] | None = None,
                     growth: bool = False, fix: Mapping[str, float] | None = None) -> Any:
        """The steady state: every variable constant over time, exogenous variables at ``exogenous``
        (missing ones at zero, as for shocks). ``guess`` gives starting values. ``growth=True`` finds a balanced
        growth path instead (levels and growth rates; see ``pymetrix.models.growth``); ``fix`` sets levels the model
        does not pin down."""
        from .linear import steady_state

        return steady_state(self, exogenous, guess, growth, fix)

    @api()
    def calibrate(self, targets: Mapping[str, float], free: Sequence[str], growth: bool = False,
                  exogenous: Mapping[str, float] | None = None, guess: Mapping[str, float] | None = None,
                  fix: Mapping[str, float] | None = None) -> Any:
        """Choose the ``free`` parameters so that the steady state hits ``targets`` (expressions of the variables, e.g.
        ``{"k/y": 10}``); a ``Calibration`` with the new model. See ``pymetrix.models.calibrate``."""
        from .calibrate import calibrate

        return calibrate(self, targets, free, growth, exogenous, guess, fix)

    def in_logs(self, *names: str) -> Model:
        """The same model with these endogenous variables (all, without names) written as ``exp(x)`` in every
        equation, so that ``x`` is the log of the variable: levels that grow by a factor ``(1 + g)`` each period then
        grow by ``log(1 + g)``, a balanced growth path ``steady_state(growth=True)`` finds; first-order solutions are
        in log deviations (as IRIS's log variables and Dynare's ``loglinear``)."""
        from dataclasses import replace

        from ..estimation import Equation
        from .dynare import _logs

        targets = list(names) or list(self.endogenous)
        unknown = [v for v in targets if v not in self.endogenous]
        if unknown:
            raise ModelError(f"{unknown} are not endogenous variables of the model")
        eqs = []
        for e in self.equations:
            if not isinstance(e, Equation):
                raise ModelError(f"{e!r} is a fitted equation: write it as an Equation to take logs of its variables")
            eqs.append(replace(e, text=_logs(e.text, targets)))  # parameters, priors and names kept
        return Model(eqs, roles=self.roles, over=self.over, params=self.params)

    @api()
    def linearize(self, at: Any = None, shocks: Sequence[str] = (), exogenous: Mapping[str, float] | None = None,
                  guess: Mapping[str, float] | None = None) -> Any:
        """The first-order approximation around a steady state (default: ``steady_state()``), as a
        ``LinearModel`` in deviations from it. ``shocks`` are the exogenous variables that are random (default:
        those with ``Role.SHOCK`` in ``roles``); ``exogenous`` and ``guess`` are passed to ``steady_state``."""
        from .linear import linearize

        return linearize(self, at if at is not None else self.steady_state(exogenous, guess), shocks or self.compile().shocks)

    def estimate(self, data: Dataset, measurement: Any, shocks: Sequence[str] | None = None, shock_sd: Mapping[str, float] | None = None,
                 estimate: Sequence[str] | None = None, priors: Mapping[str, Any] | None = None,
                 bounds: Mapping[str, tuple[float, float]] | None = None, span: tuple[Any, Any] | None = None,
                 guess: Mapping[str, float] | None = None, growth: bool = False, start: str = "auto", presample: int = 0,
                 max_iter: int = 2000, starts: int = 1, seed: int | None = None, workers: int | None = None,
                 occbin: bool = False, horizon: int = 40, cov: str = "classic", regimes: Any = None,
                 shock_corr: Mapping[Any, float] | None = None, order: int = 1, filter: str | None = None,
                 particles: int = 2000, gradient: str = "auto", smooth_paths: int = 100,
                 diffuse_terms: str = "include") -> Any:
        """Estimate the linearised model from data (a ``FittedModel``): maximum likelihood, or the posterior mode when
        any estimated parameter has a prior. ``measurement`` links the data to the model (a ``MeasurementMap``, or
        model variables observed as they are). ``start`` is the Kalman filter's start (``auto``, ``stationary``,
        ``diffuse`` or ``scaled``, Dynare's ``lik_init=2``); ``presample`` periods are filtered but left out of the
        likelihood; ``max_iter`` caps the optimiser's iterations. ``starts`` optimises from that many points (the starting
        values, then draws from the priors), in ``workers`` processes at once, and keeps the best: every start's result is
        in ``provenance.method["starts"]``, and the summary says when they found different modes. ``occbin=True`` takes the
        likelihood of the OccBin filter instead, for a model with occasionally binding constraints (its own ``max`` or
        ``min``): see ``pymetrix.models.occbin_filter``; ``horizon`` is how far ahead each period's regimes are worked
        out. ``cov="none"`` skips the inverse Hessian (no standard errors), for speed when only the mode is wanted.
        ``regimes``: parameters or equations that change at known dates in the sample (``Regime`` objects or text such as
        ``"2009Q1: phi_pi=1.2"``), filtered through the time-varying rules they give; their values are estimated as
        ``<parameter>@<regime>`` (``phi_pi@2009Q1``; see ``pymetrix.models.regimes``). ``shock_corr`` gives the shocks' correlations (``{("ey", "ei"): 0.3}``): estimated
        as ``corr_ey_ei``, as shock sizes are (see ``pymetrix.models.covariance``). ``order=2`` estimates the second-order
        (pruned) solution on the likelihood of the particle filter (``filter="particle"``, bootstrap, the default at
        order 2, or ``"guided"``) with ``particles`` particles and common random numbers from ``seed`` (0 if None): every
        observed series needs a measurement error; ``filter="particle"`` at order 1 checks it against the Kalman filter;
        ``smooth_paths`` paths of the particle smoother give the fit's ``smoothed`` (0: the filtered means instead).
        See ``pymetrix.models.particle``. ``gradient``: the optimiser's and the Hessian's gradient, ``"auto"`` (analytic
        where it applies, see ``pymetrix.models.kalman_score``, else finite differences), ``"analytic"`` (refused where
        it does not apply) or ``"numerical"``. ``diffuse_terms``, with diffuse states (unit roots, or ``start="diffuse"``):
        ``"include"`` (default) maximises the exact diffuse log likelihood of Durbin and Koopman (2012, section 7.2.2),
        with the diffuse periods' terms; ``"exclude"`` leaves them out (statsmodels' default convention; Kalman filter
        only), see ``pymetrix.statespace``. See ``pymetrix.models.estimation``."""
        from .estimation import estimate_model

        return estimate_model(self, data, measurement, shocks, shock_sd, estimate, priors, bounds, span, guess, growth=growth,
                              start=start, presample=presample, max_iter=max_iter, starts=starts, seed=seed, workers=workers,
                              occbin=occbin, horizon=horizon, cov=cov, regimes=regimes, shock_corr=shock_corr, order=order,
                              filter=filter, particles=particles, gradient=gradient,
                              smooth_paths=smooth_paths, diffuse_terms=diffuse_terms)

    @api()
    def dsge_var(self, data: Dataset, measurement: Any, lags: int = 4, lambda_: Any = 1.0, lambdas: Sequence[float] | None = None,
                 shocks: Sequence[str] | None = None, shock_sd: Mapping[str, float] | None = None, estimate: Sequence[str] | None = None,
                 priors: Mapping[str, Any] | None = None, bounds: Mapping[str, tuple[float, float]] | None = None,
                 span: tuple[Any, Any] | None = None, guess: Mapping[str, float] | None = None, max_iter: int = 2000,
                 shock_corr: Mapping[Any, float] | None = None) -> Any:
        """A DSGE-VAR(lambda) (Del Negro and Schorfheide 2004): a VAR(``lags``) with a constant in the observables, its
        prior from the model's population moments weighted as ``lambda T`` observations, and the model's parameters
        estimated by the VAR's marginal likelihood (a ``FittedDSGEVAR``: the posterior mode, then ``.mcmc(...)``).
        ``lambda_``: a number (fixed, at least ``(k + n) / T``); ``"grid"``, the best marginal data density over ``lambdas``
        (lambda-hat); or a prior (``Gamma(1, 0.5)``), lambda then estimated as ``lambda``. The rest as in ``estimate``.
        See ``pymetrix.models.dsge_var``."""
        from .dsge_var import dsge_var

        return dsge_var(self, data, measurement, lags, lambda_, lambdas, shocks, shock_sd, estimate, priors, bounds, span, guess,
                        max_iter, shock_corr)

    @api()
    def identification(self, measurement: Any, params: Sequence[str] | None = None, shocks: Sequence[str] | None = None,
                       shock_sd: Mapping[str, float] | None = None, lags: int = 4, at: Mapping[str, float] | None = None,
                       elasticity: bool = False, guess: Mapping[str, float] | None = None, growth: bool = False,
                       regimes: Any = None, span: Any = None) -> Any:
        """Whether the observables' moments could tell ``params`` apart, before any data exist: Iskrev's test at the
        calibration (or ``at={name: value}``), as ``estimate(...).identification(lags, at)`` gives at the same point.
        ``params`` are named as in ``estimate`` (default: the calibrated parameters and every ``sd_<shock>``);
        ``measurement``, ``shocks`` and ``shock_sd`` as there. ``regimes`` (known-date regimes, their values named
        ``<parameter>@<regime>``) over a sample ``span`` (``("2000Q1", "2019Q4")``): the moments of every period of it
        through the regimes' time-varying rules (``bayes.regime_moments``); regimes that start after the span are
        left out, as estimation leaves them to forecasts. An ``Identification``."""
        from .estimation import model_identification

        return model_identification(self, measurement, params, shocks, shock_sd, lags, at, elasticity, guess, growth,
                                    regimes, span)

    @api()
    def add_factors(self, data: Dataset, span: tuple[Any, Any] | None = None) -> dict[str, pd.Series]:
        """Tracking add factors: by equation name, each equation's residual at the data in each period of ``span``
        (default: every period where it can be computed), as EViews's ``addinit`` sets them. Solved with
        ``Scenario(adjust=model.add_factors(data))`` the model reproduces the data (a baseline projection, say)
        exactly, under any solver, so a scenario's other changes read as deviations from that baseline."""
        if data.kind != "time_series":
            raise ModelError("add factors are computed on a single time series")
        frame = data.frame
        missing = sorted({v for v in (*self.endogenous, *self.exogenous) if v not in frame.columns})
        if missing:
            raise ModelError(f"{', '.join(missing)} not in the data: every variable of the model is needed")
        columns = {c: frame[c].to_numpy(dtype=float) for c in {*self.endogenous, *self.exogenous}}
        nrows = len(frame)

        def resolve(name: str, lag: int) -> np.ndarray:
            out = np.full(nrows, np.nan)
            values = columns[name]
            if lag >= 0:
                out[:nrows - lag] = values[lag:] if lag else values
            else:
                out[-lag:] = values[:lag]
            return out

        if span is not None:
            from ..data.sample import as_span

            first, last = (pd.Period(str(p), freq=frame.index.freq) for p in as_span(span))
            keep = (frame.index >= first) & (frame.index <= last)
        else:
            keep = np.ones(nrows, dtype=bool)
        out = {}
        for e in self.model_equations:
            values = np.broadcast_to(np.asarray(evaluate(e.residual, resolve), dtype=float), (nrows,))
            series = pd.Series(values, index=frame.index)[keep]
            out[e.name] = series[np.isfinite(series.to_numpy())]
        return out

    @api()
    def solve(
        self,
        data: Dataset,
        span: tuple[Any, Any] | None,
        scenario: Scenario | None = None,
        solver: Newton | GaussSeidel | Fallback | PerfectForesight | FirstOrder | None = None,
        draws: int | None = None,
        seed: int | None = None,
        errors: str | pd.DataFrame | Dataset | Mapping[str, pd.Series] = "normal",
        shock_sd: Mapping[str, float] | None = None,
        coefficients: bool = False,
        entity_params: Mapping[Any, Mapping[str, float]] | None = None,
        regimes: Any = None,
        shock_corr: Mapping[Any, float] | None = None,
        error_scale: Mapping[Any, float] | None = None,
        workers: int | None = None,
        ordering: str = "joint",
    ) -> Solution:
        """Solve over ``span`` under ``scenario``. With ``draws``, a stochastic simulation as well: the deterministic
        solution with that many draws (``Solution.simulation``; ``quantiles``, ``mean``, ``sd``, ``fan``). A
        backward-looking model draws its fitted equations' errors (``errors``: ``"normal"`` or ``"bootstrap"``;
        ``shock_sd`` by equation; ``coefficients=True`` draws their coefficients too), or resamples a sample of
        residuals given (``errors=`` a DataFrame, periods by equation, a whole period at a time; or a Dataset of them,
        over its active sample, or the add factors ``add_factors`` gives over a historical span, in the periods every
        equation has one), with ``error_scale``
        multiplying the errors of some periods (``{"2025Q1": 0.5}``) and ``workers`` processes solving the draws; a
        first-order solution draws its
        shocks, with ``shock_sd`` by shock and correlations ``shock_corr`` (``{("ey", "ei"): 0.3}``). ``seed`` makes the
        draws reproducible. See ``pymetrix.models.stochastic``.

        With ``over`` (an entity axis of a panel ``data``), the model is solved for each entity on its own time series:
        with the same parameters, or each entity's in ``entity_params`` (``{"UK": {"cons.y": 0.6}}``); ``scenario`` is
        one for all or ``{entity: Scenario}``; draws take seed + the entity's position. Returns a ``PanelSolution``.
        With ``links`` (foreign variables, world aggregates), the entities are solved jointly, each draw too: see
        ``pymetrix.models.links``.

        ``regimes`` (first-order solver, or perfect foresight): parameters or equations that change at known dates
        (``Regime`` objects or text such as ``"2025Q1..2026Q4: rho=1, phi_pi=0"``); see ``pymetrix.models.regimes``.
        With ``PerfectForesight`` the stacked system's equations switch at the regimes' dates, the nonlinear model's
        own (``pymetrix.models.perfect_foresight``), known in advance unless a regime is a surprise (``expected=False``),
        solved again from its start.

        ``ordering`` (a backward-looking model solved period by period): ``"joint"`` (default) gives the solver all the
        equations each period; ``"blocks"`` splits them into recursive and simultaneous blocks first, solves each
        recursive equation once, in order, and iterates only within the simultaneous blocks, each with the solver given
        (``Newton``, ``GaussSeidel`` or a ``Fallback``, block by block); see ``pymetrix.models.ordering``. The same
        solution, with far fewer evaluations for a model that is mostly recursive. ``Solution.convergence`` then
        reports the blocks iterated on and the recursive equations by period, and the blocks in ``.attrs["blocks"]``."""
        if ordering not in ("joint", "blocks"):
            raise SpecificationError(f"ordering is 'joint' or 'blocks', not {ordering!r}")
        if ordering == "blocks" and (regimes or self.links or isinstance(solver, (FirstOrder, PerfectForesight))):
            raise SpecificationError("ordering='blocks' is for backward-looking models solved period by period (Newton, "
                                     "GaussSeidel or a Fallback of them), without regimes or links")
        if shock_corr and (self.over or not (draws and isinstance(solver, FirstOrder))):
            raise SpecificationError("shock_corr is for the shocks drawn from a first-order solution (solver=FirstOrder(), draws=...); "
                                     "a backward-looking model's errors are correlated as their residuals were")
        errors = _residual_sample(errors)
        sample = isinstance(errors, pd.DataFrame)
        if (sample or error_scale or workers) and (self.over or not draws or isinstance(solver, FirstOrder)):
            raise SpecificationError("a residual sample (errors=<DataFrame>), error_scale and workers are for the stochastic "
                                     "simulation (draws=...) of a backward-looking model, for one time series")
        if workers is not None and int(workers) < 1:
            raise SpecificationError("workers is the number of processes: 1 or more")
        if regimes:
            if self.over or not isinstance(solver, (FirstOrder, PerfectForesight)):
                raise SpecificationError("regimes are solved with the first-order solver (solver=FirstOrder()) or perfect "
                                         "foresight (solver=PerfectForesight()), for one time series")
            if data.kind != "time_series":
                raise ModelError("a model is solved on a single time series")
            if span is None:
                raise ModelError("give the simulation span, e.g. span=('2025Q1', '2030Q4')")
            if isinstance(solver, PerfectForesight):
                if draws:
                    raise SpecificationError("stochastic simulation of a model with expectations needs its first-order solution "
                                             "(solver=FirstOrder()); nonlinear perfect foresight is solved for one path")
                return self._solve(data, span, scenario or Scenario(name="baseline"), solver, regimes)
            if solver.growth:
                raise ModelError("regimes are for models without growth trends")
            sol = self.linearize(self.steady_state(), shocks=self.exogenous).solve()
            return sol.solve(data, span, scenario, draws=draws, shock_sd=shock_sd if draws else None, seed=seed, regimes=regimes,
                             shock_corr=shock_corr)
        if self.over:
            return _solve_entities(self, data, span, scenario, solver, draws, seed, errors, shock_sd, coefficients, entity_params,
                                   ordering)
        if entity_params:
            raise SpecificationError("entity_params are for a model solved per entity: Model(..., over='country')")
        solver = solver or Newton()
        scenario = scenario or Scenario(name="baseline")
        if draws:  # what a stochastic simulation can do, said before solving
            if int(draws) < 2:
                raise SpecificationError("draws is the number of simulations: at least 2")
            if not sample and errors not in ("normal", "bootstrap"):
                raise SpecificationError("errors is 'normal' or 'bootstrap', or a DataFrame of residuals to resample")
            if isinstance(solver, FirstOrder) and (errors != "normal" or coefficients):
                raise SpecificationError("a first-order solution draws its shocks (normal, with shock_sd): bootstrapped errors "
                                         "and coefficient draws are for backward-looking models")
            if not isinstance(solver, FirstOrder) and (isinstance(solver, PerfectForesight) or self.has_leads):
                raise SpecificationError("stochastic simulation of a model with expectations needs its first-order solution "
                                         "(solver=FirstOrder()); nonlinear perfect foresight is solved for one path")
        base = self._solve(data, span, scenario, solver, ordering=ordering)
        if not draws:
            return base
        from . import stochastic

        if isinstance(solver, FirstOrder):
            sims = stochastic.first_order(base.model, base, int(draws), shock_sd, seed, shock_corr=shock_corr)
        else:
            sims = stochastic.backward(self, data, span, scenario, solver, base, int(draws), seed, errors, shock_sd, coefficients,
                                       error_scale, workers)
        return with_simulation(base, sims)

    def _solve(self, data: Dataset, span: tuple[Any, Any] | None, scenario: Scenario, solver: Any, regimes: Any = None,
               ordering: str = "joint") -> Solution:
        if data.kind != "time_series":
            raise ModelError("a model is solved on a single time series; for a panel, solve per entity with Model(..., over='country')")
        if span is None:
            raise ModelError("give the simulation span, e.g. span=('2025Q1', '2030Q4')")
        if isinstance(solver, FirstOrder):
            steady = self.steady_state(growth=solver.growth)
            return self.linearize(steady, shocks=self.exogenous).solve().solve(data, span, scenario)
        if self.has_leads and not isinstance(solver, PerfectForesight):
            raise ModelError("the model has leads (x[+1]); use solver=PerfectForesight() or FirstOrder()")
        if isinstance(solver, PerfectForesight):
            from .perfect_foresight import simulate_stacked, simulate_with_surprises

            if scenario.unanticipated:  # expectation errors: solved again each time a surprise arrives
                return simulate_with_surprises(self, data, span, scenario, solver, regimes)
            return simulate_stacked(self, data, span, scenario, solver, regimes)
        # without leads nothing is expected, so whether a shock is anticipated makes no difference
        return _simulate(self, data, span, scenario, solver, ordering)


def _residual_sample(errors: Any) -> Any:
    """``solve``'s ``errors`` as the DataFrame of residuals it resamples, when a sample is given: a Dataset's series
    over its active sample, or add factors by equation (``add_factors``) in the periods they all have; a name
    ("normal", "bootstrap") as it is."""
    if isinstance(errors, (str, pd.DataFrame)):
        return errors
    if isinstance(errors, Dataset):
        if errors.kind != "time_series":
            raise SpecificationError("a residual sample is a single time series: periods by equation")
        return errors.frame[errors.sample_mask()]
    if isinstance(errors, Mapping) and all(isinstance(v, pd.Series) for v in errors.values()):
        return pd.DataFrame(dict(errors)).dropna()  # the periods every equation has one, as the bootstrap's
    return errors


def _held_model(item: Any) -> Model | None:
    """The Model an item of ``Model([...])`` stands for, when it is one: a Model, or a template or XML model read
    (``read_eviews``, ``read_eviews_xml``) holding a backward-looking one. Equations and systems are not."""
    if isinstance(item, Model):
        return item
    if isinstance(item, (Equation, FittedEquation)):
        return None
    inner = getattr(item, "model", None)
    return inner if isinstance(inner, Model) else None


def _merged_params(inner: Sequence[Any], given: Any) -> Any:
    """The parameter values of the models a Model is built from, those ``given`` winning."""
    from .system import ParameterSet

    values: dict[str, float] = {}
    for p in inner:
        values.update(p.values() if isinstance(p, ParameterSet) else dict(p or {}))
    if isinstance(given, ParameterSet):
        rest = {k: v for k, v in values.items() if k not in given}
        return ParameterSet((*given.parameters, *ParameterSet.from_values(rest).parameters)) if rest else given
    values.update(dict(given or {}))
    return values or None


def _with_specs(system: Any, specs: Sequence[Param]) -> Any:
    """The compiled parameters with the model's metadata, over any their equations gave; specs for shock sizes and
    measurement errors (``sd_...``) stay with the model (``Model.spec``)."""
    from dataclasses import replace

    if not all(isinstance(s, Param) for s in specs):
        raise SpecificationError("specs are Param('rho', prior=..., bounds=..., label=...) objects")
    names = [s.name for s in specs]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise SpecificationError(f"more than one spec for {', '.join(twice)}")
    params = system.parameters
    unknown = [n for n in names if n not in params and not n.startswith(("sd_", "corr_")) and "@" not in n]
    if unknown:
        raise SpecificationError(f"specs for {', '.join(unknown)}, which are not parameters of the model (nor sd_<shock>, "
                                 f"sd_<series>, corr_<shock>_<shock> or a regime's value, name@regime); parameters: {', '.join(params.names)}")
    by_name = {s.name: s for s in specs}
    merged = tuple(replace(p, spec=by_name[p.name].merged(p.spec)) if p.name in by_name else p for p in params)
    return replace(system, parameters=type(params)(merged))


def _solve_entities(model: Model, data: Dataset, span: Any, scenario: Any, solver: Any, draws: Any, seed: Any, errors: str,
                    shock_sd: Any, coefficients: bool, entity_params: Any, ordering: str = "joint") -> Any:
    """The model solved for each entity of a panel on its own time series (see ``Model.solve``)."""
    from dataclasses import replace

    from ..data import EntityAxis
    from .solution import PanelSolution

    over = model.over
    if data.kind != "panel" or over not in [a.name for a in data.axes]:
        raise ModelError(f"a model solved over {over!r} needs a panel with that entity axis; this data has {', '.join(a.name for a in data.axes)}")
    entities = list(dict.fromkeys(data.frame.index.get_level_values(over)))
    for name, given in (("entity_params", entity_params), ("scenario", scenario if isinstance(scenario, Mapping) else None)):
        unknown = [k for k in (given or {}) if k not in entities]
        if unknown:
            raise ModelError(f"{name} names {', '.join(map(str, unknown))}, not {over} values of the data")
    base = replace(model, over=None, links=())  # compiled once; entity parameters share it
    solutions, rounds = {}, 0
    if model.links:  # solved jointly: each entity's links from the others' solutions
        from .links import check_links, solve_linked

        check_links(model.links, base, list(data.frame.columns))
        if draws:  # what a stochastic simulation of linked entities can do, said before solving
            if int(draws) < 2:
                raise SpecificationError("draws is the number of simulations: at least 2")
            if errors not in ("normal", "bootstrap"):
                raise SpecificationError("errors is 'normal' or 'bootstrap'")
            if isinstance(solver, (FirstOrder, PerfectForesight)) or base.has_leads:
                raise SpecificationError("stochastic simulation of linked entities is for backward-looking models: solve a model "
                                         "with expectations without draws, or without links")
            if coefficients:
                raise SpecificationError("coefficient draws with links are not available: draw the errors only "
                                         "(coefficients=False), or solve without links")
        solutions, rounds = solve_linked(base, model.links, data, span, scenario, solver, entity_params, entities, str(over))
        if draws:
            from .links import draw_linked

            sims = draw_linked(base, model.links, data, span, scenario, solver, entity_params, entities, str(over), solutions,
                               int(draws), seed, errors, shock_sd)
            solutions = {key: with_simulation(s, sims[key]) for key, s in solutions.items()}
    else:
        for i, key in enumerate(entities):
            own = base.with_parameters(entity_params[key]) if entity_params and key in entity_params else base
            own_scenario = scenario.get(key) if isinstance(scenario, Mapping) else scenario
            try:
                solutions[key] = own.solve(data.entity(key, by=over), span, own_scenario, solver, draws,
                                           None if seed is None else seed + i, errors, shock_sd, coefficients,
                                           **({"ordering": ordering} if ordering != "joint" else {}))
            except (ModelError, SpecificationError) as exc:
                raise ModelError(f"{over} {key}: {exc}") from exc
    first = next(iter(solutions.values()))
    frame = pd.concat({k: s.frame for k, s in solutions.items()}, names=[over])
    return PanelSolution(frame.sort_index(), (EntityAxis(over), *first.axes), first.variables, None, first.meta,
                         solutions=solutions, over=str(over), links=tuple(link.name for link in model.links), rounds=rounds)


# simulation


def missing_inputs(model: Model, data: Dataset, span: Any, scenario: Scenario | None = None) -> pd.DataFrame:
    """``Model.missing_inputs``."""
    if data.kind != "time_series":
        raise ModelError("a model is solved on a single time series")
    if span is None:
        raise ModelError("give the simulation span, e.g. span=('2025Q1', '2030Q4')")
    sc = scenario or Scenario(name="baseline")
    absent = [v for v in (*model.exogenous, *sc.free) if v not in data.frame.columns]
    if absent:  # reported as missing in every period that needs them, not refused
        data = data.with_series(**{v: pd.Series(np.nan, index=data.frame.index) for v in dict.fromkeys(absent)})
    prep = _prepare(model, data, span, sc, lenient=True)
    arrays, nrows = prep.arrays, len(prep.full)
    positions = np.flatnonzero(prep.rows)
    first, last = int(positions[0]), int(positions[-1])
    solved = {k: set(prep.layouts[k][1]) for k in range(len(prep.layouts))} if prep.layouts else None
    base = set(prep.unknowns)

    def unknown(name: str, p: int) -> bool:
        """Whether the solve determines ``name`` in row ``p``."""
        if first <= p <= last:
            return name in (base if solved is None or prep.layout is None else solved[int(prep.layout[p])])
        return p > last and name in model.endogenous  # leads past the span: the terminal condition's
    out: list[tuple[Any, str, str]] = []
    for pos in positions:
        eqs = prep.at(int(pos))[0]
        kept = {e.name for e in eqs}
        for e in model.model_equations:
            if e.name not in kept:  # fixed here: the fixed value is its input
                if not np.isfinite(arrays[e.dependent][pos]):
                    out.append((prep.full[pos], e.name, e.dependent))
                continue
            lacking = []
            for n in sorted(e.residual.refs(), key=lambda r: (r.id, r.lag)):
                p = int(pos) + n.lag
                if unknown(n.id, p):
                    continue
                if not (0 <= p < nrows and np.isfinite(arrays[n.id][p])):
                    lacking.append(str(n))
            out += [(prep.full[pos], e.name, x) for x in lacking]
    frame = pd.DataFrame(out, columns=["period", "equation", "input"])
    if not len(frame):
        frame["period"] = pd.PeriodIndex([], freq=prep.full.freq)
    return frame


def _path(value: Any, periods: pd.PeriodIndex) -> np.ndarray:
    if value is None:
        return np.full(len(periods), np.nan)
    if isinstance(value, Mapping):  # {period: value}
        value = pd.Series({pd.Period(str(k), freq=periods.freq): float(v) for k, v in value.items()}, dtype=float)
    if isinstance(value, pd.Series):
        s = value
        if not isinstance(s.index, pd.PeriodIndex):
            s = pd.Series(s.to_numpy(), index=pd.PeriodIndex(s.index.astype(str), freq=periods.freq))
        return s.reindex(periods).to_numpy(float)
    return np.full(len(periods), float(value))


class _PeriodSystem:
    """One period's equations as functions of that period's unknowns. With ``compiled`` (``pymetrix.models.fast``),
    the residuals and the Jacobian are each one compiled call for all the equations, the other values (lags, leads,
    exogenous) gathered once per period; otherwise the expression trees are evaluated (the reference)."""

    def __init__(self, eqs, unknowns, arrays, adjust, nrows, compiled=None):
        self.eqs = eqs
        self.equation_names = [e.name for e in eqs]
        self.unknowns = unknowns
        self.uidx = {u: j for j, u in enumerate(unknowns)}
        self.arrays = arrays
        self.adjust = adjust
        self.nrows = nrows
        self.own_index = [self.uidx.get(e.dependent, -1) for e in eqs]
        self.own_diff = [diff(e.residual, Name(e.dependent, 0)) for e in eqs]
        self.pos = 0
        self.x = np.zeros(len(unknowns))
        self.x0 = self.x
        self._last: tuple[np.ndarray, np.ndarray] | None = None  # the last residuals computed, and where
        self.report: dict[str, Any] = {}  # how the solver solved this period (test, solver, jacobians), if it says
        self.compiled = None
        if compiled is not None:
            self._compile(*compiled)

    @cached_property
    def jac(self) -> list[tuple[int, int, Any]]:
        return [
            (i, j, d)
            for i, e in enumerate(self.eqs)
            for j, u in enumerate(self.unknowns)
            if (d := diff(e.residual, Name(u, 0))) != _ZERO
        ]

    def _compile(self, fns: Any, params: np.ndarray) -> None:
        """The compiled residuals, and the derivatives with respect to this period's unknowns compiled on their own."""
        from ..expr.compile import compile_exprs

        wanted = [(i, self.uidx[name], d) for (i, name, lag), d in fns._derivatives if lag == 0 and name in self.uidx]
        jacobian = compile_exprs(tuple(d for _, _, d in wanted), fns.coefs)
        self.jac_rows = np.array([i for i, _, _ in wanted], dtype=int)
        self.jac_cols = np.array([j for _, j, _ in wanted], dtype=int)

        def plan(refs: Any) -> tuple[np.ndarray, np.ndarray]:
            """For each slot: the unknown it is (or -1), and the lag to read from the data otherwise."""
            return (np.array([self.uidx[n] if lag == 0 and n in self.uidx else -1 for n, lag in refs], dtype=int),
                    np.array([lag for _, lag in refs], dtype=int))

        self.compiled = (fns.residuals, plan(fns.residuals.refs), jacobian, plan(jacobian.refs), params)
        self._fixed: tuple[np.ndarray, np.ndarray] | None = None

    def _gather(self, refs: Any, which: np.ndarray, lags: np.ndarray) -> np.ndarray:
        out = np.full(len(refs), np.nan)
        for k, ((name, _), j, lag) in enumerate(zip(refs, which, lags)):
            if j < 0:
                p = self.pos + int(lag)
                if 0 <= p < self.nrows:
                    out[k] = self.arrays[name][p]
        return out

    def _values(self, x: np.ndarray, part: int) -> np.ndarray:
        which = self.compiled[2 * part + 1][0]
        if self._fixed is None:
            res, jac = self.compiled[0], self.compiled[2]
            self._fixed = (self._gather(res.refs, *self.compiled[1]), self._gather(jac.refs, *self.compiled[3]))
        values = self._fixed[part].copy()
        known = which >= 0
        values[known] = x[which[known]]
        return values

    def resolve(self, name: str, lag: int) -> float:
        if lag == 0 and name in self.uidx:
            return self.x[self.uidx[name]]
        p = self.pos + lag
        return self.arrays[name][p] if 0 <= p < self.nrows else np.nan

    def at(self, pos: int) -> None:
        self.pos = pos
        self._last = None
        self.report = {}
        self._fixed = None
        self._own_fixed: list[np.ndarray] | None = None
        x0 = []
        for u in self.unknowns:
            v = self.arrays[u][pos]
            if not np.isfinite(v) and pos > 0:
                v = self.arrays[u][pos - 1]
            x0.append(v if np.isfinite(v) else 1.0)
        self.x0 = np.array(x0)

    def residuals(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        if self.compiled is not None:
            fn, params = self.compiled[0], self.compiled[4]
            F = np.array(fn(self._values(x, 0), params), dtype=float) - np.array([a[self.pos] for a in self.adjust])
        else:
            F = np.array([evaluate(e.residual, self.resolve) - a[self.pos] for e, a in zip(self.eqs, self.adjust)], dtype=float)
        self._last = (x.copy(), F)
        return F

    def _largest(self, F: np.ndarray) -> tuple[float, str]:
        if not len(F):
            return 0.0, ""
        size = np.where(np.isfinite(F), np.abs(F), np.inf)
        i = int(np.argmax(size))
        return float(size[i]), self.equation_names[i]

    def last_residuals(self) -> tuple[float | None, str | None]:
        """The largest absolute residual the solver last computed in this period, and its equation."""
        return (None, None) if self._last is None or self._last[0].shape != self.x0.shape else self._largest(self._last[1])

    def final_residuals(self, x: np.ndarray) -> tuple[float, str]:
        """The largest absolute residual at the solution ``x``, and its equation: the solver's last residuals when they
        were computed there (Newton and Gauss-Seidel end so), else computed again."""
        last = self._last
        F = last[1] if last is not None and np.array_equal(last[0], x) else self.residuals(x)
        return self._largest(F)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        J = np.zeros((len(self.eqs), len(self.unknowns)))
        if self.compiled is not None:
            fn, params = self.compiled[2], self.compiled[4]
            np.add.at(J, (self.jac_rows, self.jac_cols), np.array(fn(self._values(x, 1), params), dtype=float))
            return J
        for i, j, d in self.jac:
            J[i, j] = evaluate(d, self.resolve)
        return J

    @cached_property
    def _own(self) -> list[tuple[Any, np.ndarray, np.ndarray]]:
        """Each equation's residual and derivative in its own variable, compiled together (parameters as numbers)."""
        from ..expr.compile import compile_exprs

        out = []
        for e, d in zip(self.eqs, self.own_diff):
            fn = compile_exprs((e.residual, d))
            which = np.array([self.uidx[n] if lag == 0 and n in self.uidx else -1 for n, lag in fn.refs], dtype=int)
            out.append((fn, which, np.array([lag for _, lag in fn.refs], dtype=int)))
        return out

    def own(self, i: int, x: np.ndarray) -> tuple[float, float, int]:
        self.x = x
        if self.compiled is not None:
            fn, which, lags = self._own[i]
            if self._own_fixed is None:
                self._own_fixed = [self._gather(f.refs, w, g) for f, w, g in self._own]
            values = self._own_fixed[i].copy()
            known = which >= 0
            values[known] = x[which[known]]
            f, d = fn(values)
            return float(f) - self.adjust[i][self.pos], float(d), self.own_index[i]
        f = evaluate(self.eqs[i].residual, self.resolve) - self.adjust[i][self.pos]
        return float(f), float(evaluate(self.own_diff[i], self.resolve)), self.own_index[i]

    def missing_inputs(self) -> list[str]:
        out = set()
        for e in self.eqs:
            for n in e.residual.refs():
                if n.lag == 0 and n.id in self.uidx:
                    continue
                if not np.isfinite(self.resolve(n.id, n.lag)):
                    out.add(str(n))
        return sorted(out)


@dataclass
class _Prepared:
    """A simulation set up: the periods, the data as arrays (with the scenario applied), the
    equations still to solve, their unknowns and add factors."""

    full: pd.PeriodIndex
    frame: pd.DataFrame
    arrays: dict[str, np.ndarray]
    rows: np.ndarray
    eqs: list[ModelEquation]
    unknowns: list[str]
    adjust: list[np.ndarray]
    start: pd.Period
    end: pd.Period
    # partial fixes and holds (Scenario(partial=...)): each period's equations, unknowns and add factors, and which of
    # them each row of ``full`` takes (-1 outside the span). One layout, every row's, without them: then eqs, unknowns
    # and adjust are it; with several, they are the layout with no partial fix or hold in force.
    layouts: list[tuple[list[ModelEquation], list[str], list[np.ndarray]]] = field(default_factory=list)
    layout: np.ndarray | None = None

    @property
    def varying(self) -> bool:
        """Whether the equations solved change from period to period (partial fixes or holds)."""
        return len(self.layouts) > 1

    def at(self, pos: int) -> tuple[list[ModelEquation], list[str], list[np.ndarray]]:
        """The equations, unknowns and add factors of row ``pos``."""
        if not self.layouts or self.layout is None:
            return self.eqs, self.unknowns, self.adjust
        return self.layouts[int(self.layout[pos])]


def _simulate(model: Model, data: Dataset, span: tuple[Any, Any], sc: Scenario, solver: Any, ordering: str = "joint") -> Solution:
    prep = _prepare(model, data, span, sc)
    full, rows, arrays = prep.full, prep.rows, prep.arrays
    from ..threads import blas_threads
    from .fast import functions, parameter_values
    from .solvers import describe, uses_options

    systems: dict[int, _PeriodSystem] = {}  # one per layout of the equations (partial fixes and holds), built when first met
    ordered: dict[int, Any] = {}  # ordering="blocks": each layout's blocks

    def build(eqs: list[ModelEquation], unknowns: list[str], adjust: list[np.ndarray]) -> _PeriodSystem:
        fns = functions(model, eqs)
        return _PeriodSystem(eqs, unknowns, arrays, adjust, len(full), None if fns is None else (fns, parameter_values(model, fns)))

    def system_at(pos: int) -> _PeriodSystem:
        k = -1 if prep.layout is None else int(prep.layout[pos])
        if k not in systems:
            systems[k] = build(*prep.at(pos))
        return systems[k]

    def blocks_at(pos: int, system: _PeriodSystem) -> Any:
        from .ordering import BlockSolve

        k = -1 if prep.layout is None else int(prep.layout[pos])
        if k not in ordered:
            ordered[k] = BlockSolve(system, lambda e, u: build([system.eqs[i] for i in e], [system.unknowns[j] for j in u],
                                                               [system.adjust[i] for i in e]))
        return ordered[k]

    blocks = ordering == "blocks"
    positions = np.flatnonzero(rows)
    record = _Convergence(len(positions), detailed=blocks or uses_options(solver), blocks=blocks)
    name = describe(solver) if record.detailed else ""
    total = 0
    with blas_threads():  # one period's Jacobian is small: BLAS's threads cost more than they save (pymetrix.threads)
        for t, pos in enumerate(positions):
            system = system_at(int(pos))
            system.at(int(pos))
            gaps = system.missing_inputs()
            if gaps:
                record.failed(t, 0, None, None)
                raise _with_convergence(ModelError(f"{full[pos]}: missing values for {gaps}"), record, full[positions])
            plan = blocks_at(int(pos), system) if blocks else None
            started = time.perf_counter()
            try:
                x, iterations = plan.solve_period(solver, int(pos)) if plan is not None else solver.solve_period(system)
            except ModelError as exc:
                record.seconds[t] = time.perf_counter() - started
                record.failed(t, getattr(exc, "iterations", None), *(getattr(exc, "last", None) or system.last_residuals()),
                              system.report, plan)
                raise _with_convergence(exc, record, full[positions])
            record.seconds[t] = time.perf_counter() - started
            record.solved(t, iterations, *(plan.final if plan is not None else system.final_residuals(x)),
                          {"solver": name, **system.report}, plan)
            total += iterations
            for j, u in enumerate(system.unknowns):
                arrays[u][pos] = x[j]
    table = record.frame(full[positions])
    if blocks:  # the blocks in solution order, by name (by layout, with partial fixes or holds)
        table.attrs["blocks"] = next(iter(ordered.values())).names() if len(ordered) == 1 else \
            {k: plan.names() for k, plan in ordered.items()}
    return _solution(model, data, sc, solver, prep, total, table, ordering)


class _Convergence:
    """Each period's iterations, largest absolute residual and its equation, and whether it converged, filled in as a
    period-by-period solve goes (``Solution.convergence``). ``detailed``: also which test stopped each period, the
    solver that solved it and the Jacobians evaluated; ``blocks``: also the blocks iterated on and the recursive
    equations solved once (``ordering="blocks"``). ``seconds``: each period's solve, timed by ``time.perf_counter``
    around the solver's call (building the period's system is not in it), kept in the table's ``.attrs["seconds"]``,
    one a row, not as a column."""

    def __init__(self, periods: int, detailed: bool = False, blocks: bool = False) -> None:
        self.iterations = np.zeros(periods, dtype=np.int64)
        self.known = np.ones(periods, dtype=bool)  # whether the iterations are known (a solver's error may not say)
        self.residual = np.full(periods, np.nan)
        self.equation: list[str] = [""] * periods
        self.converged = np.zeros(periods, dtype=bool)
        self.detailed, self.blocks = detailed, blocks
        self.test: list[str] = [""] * periods
        self.solver: list[str] = [""] * periods
        self.jacobians = np.zeros(periods, dtype=np.int64)
        self.simultaneous = np.zeros(periods, dtype=np.int64)
        self.recursive = np.zeros(periods, dtype=np.int64)
        self.seconds = np.zeros(periods)
        self.done = 0

    def _details(self, t: int, report: Mapping[str, Any] | None, plan: Any) -> None:
        report = report or {}
        self.test[t] = str(report.get("test", ""))
        self.solver[t] = str(report.get("solver", ""))
        self.jacobians[t] = int(report.get("jacobians", 0))
        if plan is not None:
            self.simultaneous[t], self.recursive[t] = plan.simultaneous, plan.recursive

    def solved(self, t: int, iterations: int, residual: float, equation: str, report: Mapping[str, Any] | None = None,
               plan: Any = None) -> None:
        self.iterations[t], self.residual[t], self.equation[t], self.converged[t] = iterations, residual, equation, True
        self._details(t, report, plan)
        self.done = t + 1

    def failed(self, t: int, iterations: int | None, residual: float | None, equation: str | None,
               report: Mapping[str, Any] | None = None, plan: Any = None) -> None:
        self.known[t] = iterations is not None
        self.iterations[t] = iterations or 0
        self.residual[t] = np.nan if residual is None else residual
        self.equation[t] = equation or ""
        self._details(t, {k: v for k, v in (report or {}).items() if k != "solver"}, plan)
        self.done = t + 1

    def frame(self, periods: pd.PeriodIndex) -> pd.DataFrame:
        n = self.done
        iterations = pd.array(self.iterations[:n], dtype="Int64")
        iterations[~self.known[:n]] = pd.NA
        out = convergence_frame(periods[:n], iterations, self.residual[:n], self.equation[:n], self.converged[:n])
        if self.detailed:
            out["test"] = self.test[:n]
            out["solver"] = self.solver[:n]
            out["jacobians"] = pd.array(self.jacobians[:n], dtype="Int64")
        if self.blocks:
            out["blocks"] = pd.array(self.simultaneous[:n], dtype="Int64")
            out["recursive"] = pd.array(self.recursive[:n], dtype="Int64")
        out.attrs["seconds"] = [float(v) for v in self.seconds[:n]]  # metadata: the columns are as without it
        return out


def convergence_frame(periods: pd.PeriodIndex, iterations: Any, residual: Any, equation: Sequence[str], converged: Any) -> pd.DataFrame:
    """``Solution.convergence``: by period, the solver's iterations, the largest absolute residual at the end and the
    equation it belongs to, and whether the period converged."""
    index = pd.PeriodIndex(periods, name=periods.name or "time")
    return pd.DataFrame({"iterations": pd.array(iterations, dtype="Int64"), "residual": np.asarray(residual, dtype=float),
                         "equation": list(equation), "converged": np.asarray(converged, dtype=bool)}, index=index)


def _with_convergence(exc: Exception, record: _Convergence, periods: pd.PeriodIndex) -> Exception:
    """``exc`` carrying the convergence of the periods solved before it and of the one it stopped in (``convergence``)."""
    exc.convergence = record.frame(periods)  # type: ignore[attr-defined]
    return exc


def _solution(model: Model, data: Dataset, sc: Scenario, solver: Any, prep: _Prepared, iterations: int,
              convergence: pd.DataFrame | None = None, ordering: str = "joint") -> Solution:
    out = prep.frame.copy()
    for c, values in prep.arrays.items():
        out[c] = values
    method = to_jsonable(solver)
    if ordering != "joint":
        method = {**method, "ordering": ordering} if isinstance(method, Mapping) else {"solver": method, "ordering": ordering}
    provenance = Provenance(
        data_vintages={v: data.variables[v].vintage for v in model.exogenous if v in data.variables},
        spec={"equations": [e.name for e in model.model_equations], "scenario": sc.name},
        method=method,
        sample=(str(prep.start), str(prep.end)),
        nobs=int(prep.rows.sum()),
        dropped=0,
        package_version=__version__,
        data_fingerprint=data_fingerprint(data),
    )
    return Solution(out, data.axes, data.variables, meta=data.meta, model=model, scenario=sc, converged=True, iterations=iterations, provenance=provenance,
                    convergence=convergence)


def _prepare(model: Model, data: Dataset, span: tuple[Any, Any], sc: Scenario, lenient: bool = False) -> _Prepared:
    """The simulation set up; ``lenient``: a fixed or held variable without values is left so (for ``missing_inputs``)."""
    if sc.hold_sd or sc.free_sd or sc.ranges:
        raise ModelError("soft holds (hold_sd=), ranges (hold_lower=, hold_upper=) and least-norm free shocks (free_sd=) need a first-order solution: "
                         "solve with solver=FirstOrder(), or linearize().solve() first")
    t = data.time_axis
    assert t is not None
    idx = data.frame.index
    from ..data.sample import as_span
    span = as_span(span)  # '1990Q1:2019Q4' or a pair
    start = pd.Period(str(span[0]), freq=idx.freq)
    end = pd.Period(str(span[1]), freq=idx.freq)
    full = pd.period_range(min(idx.min(), start), max(idx.max(), end), freq=idx.freq, name=t.name)
    frame = data.frame.reindex(full)
    endo, exo = model.endogenous, model.exogenous
    missing = [v for v in exo if v not in frame.columns and v not in sc.free]
    if missing:
        raise ModelError(f"exogenous variable(s) {missing} are not in the data")
    innovations = getattr(model, "_innovations", {})
    for v in (*endo, *sc.free):
        if v not in frame.columns:
            history = innovations.get(v)
            frame[v] = history.reindex(full).to_numpy(float) if history is not None else np.nan
    arrays = {c: frame[c].to_numpy(dtype=float, copy=True) for c in {*endo, *exo}}
    rows = np.asarray((full >= start) & (full <= end))
    if innovations:
        _continue_innovations(model, arrays, int(np.argmax(rows)))
    periods = full[rows]

    def check(names, allowed, what):
        bad = [v for v in names if v not in allowed]
        if bad:
            raise ModelError(f"{what}: {bad} are not {'exogenous' if allowed is exo else 'endogenous'} in this model")

    check(sc.set, exo, "set=")
    check(sc.shift, exo, "shift=")
    check(sc.free, exo, "free=")
    check(sc.fix, endo, "fix=")
    check(sc.hold, endo, "hold=")
    for v, value in sc.set.items():
        p = _path(value, periods)
        arrays[v][rows] = np.where(np.isfinite(p), p, arrays[v][rows])
    for v, value in sc.shift.items():
        arrays[v][rows] += np.nan_to_num(_path(value, periods))
    partial = _partial_names(sc)
    active: dict[str, np.ndarray] = {}  # partial fixes and holds: the periods of the span they are in force
    for v, value in {**sc.fix, **sc.hold}.items():
        p = _path(value, periods)
        if v in partial:
            given = np.isfinite(p) if value is not None else np.isfinite(arrays[v][rows])
            active[v] = given
            arrays[v][rows] = np.where(given & np.isfinite(p), p, arrays[v][rows])
            continue
        arrays[v][rows] = np.where(np.isfinite(p), p, arrays[v][rows])
        if not np.all(np.isfinite(arrays[v][rows])) and not lenient:
            raise ModelError(f"{v!r} is fixed or targeted but has no value in some simulation periods")
    if any(v in sc.hold for v in active) and len(sc.free) != len(sc.hold):
        raise ModelError(f"{len(sc.hold)} held variables but {len(sc.free)} free variables: with partial holds each hold= "
                         "is paired with one free= variable, in order")
    names = {e.name for e in model.model_equations}
    unknown_adjust = set(sc.adjust) - names
    if unknown_adjust:
        raise ModelError(f"adjust= names {sorted(unknown_adjust)} are not equation names {sorted(names)}")
    factors = {e.name: np.nan_to_num(_path(sc.adjust[e.name], periods)) for e in model.model_equations if e.name in sc.adjust}

    def layout(on: frozenset[str]) -> tuple[list[ModelEquation], list[str], list[np.ndarray]]:
        """The equations solved, their unknowns and add factors, with the partial fixes and holds ``on`` in force."""
        fixed = {v for v in sc.fix if v not in active or v in on}
        held = {v for v in sc.hold if v not in active or v in on}
        eqs = [e for e in model.model_equations if e.dependent not in fixed]
        frees = [f for h, f in zip(sc.hold, sc.free) if h in held] if active else list(sc.free)
        unknowns = [e.dependent for e in eqs if e.dependent not in held] + frees
        if len(unknowns) != len(eqs):
            raise ModelError(f"{len(eqs)} equations but {len(unknowns)} unknowns; each hold= needs one free= variable")
        adjust = []
        for e in eqs:
            a = np.zeros(len(full))
            if e.name in factors:
                a[rows] = factors[e.name]
            adjust.append(a)
        return eqs, unknowns, adjust

    if not active:
        eqs, unknowns, adjust = layout(frozenset())
        return _Prepared(full, frame, arrays, rows, eqs, unknowns, adjust, start, end)
    order = list(active)
    keys = [frozenset(v for v in order if active[v][k]) for k in range(len(periods))]
    distinct = list(dict.fromkeys(keys))
    layouts = [layout(key) for key in distinct]
    where = np.full(len(full), -1, dtype=int)
    where[rows] = [distinct.index(key) for key in keys]
    eqs, unknowns, adjust = layouts[0] if len(layouts) == 1 else layout(frozenset())
    return _Prepared(full, frame, arrays, rows, eqs, unknowns, adjust, start, end, layouts, where)


def _partial_names(sc: Scenario) -> set[str]:
    """The fixed and held variables ``Scenario(partial=...)`` applies only where their paths have values."""
    given = getattr(sc, "partial", False)  # a Scenario pickled before partial has none
    if given is True:
        return {*sc.fix, *sc.hold}
    if not given:
        return set()
    names = [given] if isinstance(given, str) else list(given)
    bad = [v for v in names if v not in sc.fix and v not in sc.hold]
    if bad:
        raise ModelError(f"partial= names {bad}, which are not fixed or held in this scenario")
    return set(names)


def with_simulation(base: Solution, sims: Any) -> Solution:
    """``base`` with a stochastic simulation's draws, and how they were made in its provenance."""
    from dataclasses import replace

    if base.provenance is None:
        return replace(base, simulation=sims)
    method = base.provenance.method
    method = {**method} if isinstance(method, Mapping) else {"solver": method}
    return replace(base, simulation=sims, provenance=replace(base.provenance, method={**method, "stochastic": dict(sims.info)}))


def _continue_innovations(model: Model, arrays: dict[str, np.ndarray], first: int) -> None:
    """The ARMA innovations of the periods after an equation's estimation sample and before the simulation, from the
    data: each from its identity, ``e = residual`` (the equation's lagged innovations among its terms)."""
    for v in getattr(model, "_innovations", {}):
        eq = next(e for e in model.model_equations if e.dependent == v)
        values = arrays[v]
        known = np.flatnonzero(np.isfinite(values[:first]))
        if not len(known):
            continue
        for pos in range(int(known[-1]) + 1, first):
            def resolve(name: str, lag: int, pos: int = pos) -> float:
                if name == v and lag == 0:
                    return 0.0
                p = pos + lag
                return float(arrays[name][p]) if name in arrays and 0 <= p < len(arrays[name]) else np.nan

            value = -float(evaluate(eq.residual, resolve))  # the identity is e - residual = 0
            if not np.isfinite(value):
                break  # data missing: the simulation's own check reports it
            values[pos] = value


class _Substituted(Sequence):
    """A model's equations with its parameter values substituted, worked out only when something asks for them (the
    compiled solvers do not): ``Model.with_parameters`` would otherwise substitute every equation at every trial."""

    def __init__(self, model: Model) -> None:
        self._model = model
        self._items: tuple[Any, ...] | None = None

    def _all(self) -> tuple[Any, ...]:
        if self._items is None:
            system = self._model._system  # type: ignore[attr-defined]
            self._items = tuple(ModelEquation(e.name, e.dependent, r, e.stochastic) for e, r in zip(system.equations, system.residuals()))
        return self._items

    def __getitem__(self, i: Any) -> Any:
        return self._all()[i]

    def __len__(self) -> int:
        return len(self._model._system.equations)  # type: ignore[attr-defined]

    def __iter__(self) -> Any:
        return iter(self._all())
