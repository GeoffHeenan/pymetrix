"""Linear rational-expectations models: steady state, linearisation and the stable solution.

``model.steady_state()`` finds the values at which every variable is constant;
``model.linearize(ss, shocks=[...])`` gives the first-order approximation in
deviations from it::

    A0 E[x(t+1)] + A1 x(t) + A2 x(t-1) + B e(t) = 0

and ``LinearModel.solve()`` the unique stable solution (the decision rules)::

    x(t) = P x(t-1) + Q e(t)

found by the QZ (generalised Schur) decomposition, as in Klein (2000) and
Uhlig (1999). The Blanchard-Kahn conditions are checked: with too few stable
roots there is no stable solution; with too many it is not unique
(indeterminacy, e.g. a Taylor rule that responds less than one for one to
inflation). Leads and lags beyond one become auxiliary variables. The solution
gives impulse responses, simulations, and a state-space form for the Kalman
filter (estimating unobserved variables from data, as in policy-model filtration).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import linalg, optimize

from .._api import api
from ..expr import Name, diff, evaluate
from ..output import Table
from .solvers import ModelError
from .._random import generator as make_rng

if TYPE_CHECKING:
    from ..data import Dataset
    from .model import Model

__all__ = ["LinearModel", "LinearSolution", "SteadyState", "linearize", "steady_state"]


# ---------------------------------------------------------------------------------------------
# steady state


@dataclass(frozen=True, eq=False)
class SteadyState:
    """Values of the endogenous and exogenous variables when every variable is constant, or, with ``growth``, on a
    balanced growth path: ``x(t) = values[x] + growth[x] * t`` (``t`` counts periods from the reference period)."""

    values: Mapping[str, float]
    model: Any = field(default=None, repr=False)
    growth: Mapping[str, float] = field(default_factory=dict)  # per period; variables not listed do not grow
    undetermined: tuple[str, ...] = ()  # levels the model does not pin down (fix= sets them)

    def __getitem__(self, name: str) -> float:
        return self.values[name]

    def at(self, name: str, t: float = 0.0) -> float:
        """The value of ``name`` ``t`` periods after the reference period."""
        return self.values[name] + self.growth.get(name, 0.0) * t

    def table(self) -> Table:
        endo = set(self.model.endogenous) if self.model is not None else set(self.values)
        if not self.growth:
            rows = [[k, v, "endogenous" if k in endo else "exogenous"] for k, v in self.values.items()]
            return Table(["variable", "value", ""], rows, title="Steady state")
        rows = [[k, v, self.growth.get(k, 0.0), "endogenous" if k in endo else "exogenous",
                 "not determined" if k in self.undetermined else ""] for k, v in self.values.items()]
        notes = [f"Levels not determined by the model (only their growth): {', '.join(self.undetermined)}; set them with fix="]             if self.undetermined else []
        return Table(["variable", "level", "growth per period", "", "note"], rows, title="Steady state with growth", notes=notes)

    def __str__(self) -> str:
        return str(self.table())

    def _repr_html_(self) -> str:
        return self.table()._repr_html_()


def _static(expr: Any) -> Any:
    """The expression with every lag and lead removed (x[-1], x[+1] -> x)."""
    return expr.map(lambda n: Name(n.id, 0) if isinstance(n, Name) and n.lag else None)


def steady_state(model: Model, exogenous: Mapping[str, float] | None = None, guess: Mapping[str, float] | None = None,
                 growth: bool = False, fix: Mapping[str, float] | None = None) -> SteadyState:
    if growth or fix:
        from .growth import steady_state_with_growth

        return steady_state_with_growth(model, exogenous, guess, fix)
    exogenous = dict(exogenous or {})
    endo = list(model.endogenous)
    values = {v: float(exogenous.get(v, 0.0)) for v in model.exogenous}
    from .fast import ENABLED, functions, parameter_values

    fns = functions(model)
    # the interpreter's residuals and Jacobian, only when the compiled ones are not there
    residuals = [_static(e.residual) for e in model.model_equations] if fns is None else []
    jac = [] if ENABLED else [[diff(r, Name(u, 0)) for u in endo] for r in residuals]

    def resolve_with(x: np.ndarray):
        current = {**values, **dict(zip(endo, x))}
        return lambda name, lag: current[name]

    def F(x: np.ndarray) -> np.ndarray:
        resolve = resolve_with(x)
        return np.array([evaluate(r, resolve) for r in residuals], dtype=float)

    def J(x: np.ndarray) -> np.ndarray:
        resolve = resolve_with(x)
        return np.array([[evaluate(d, resolve) for d in row] for row in jac], dtype=float)

    if fns is not None:  # the same residuals and Jacobian, compiled
        p = parameter_values(model, fns)
        where = {u: j for j, u in enumerate(endo)}

        def slots(compiled: Any, x: np.ndarray) -> list[float]:
            return [x[where[name]] if name in where else values[name] for name, _ in compiled.refs]

        def F(x: np.ndarray) -> np.ndarray:  # noqa: F811
            return np.array(fns.static(slots(fns.static, x), p), dtype=float)

        def J(x: np.ndarray) -> np.ndarray:  # noqa: F811
            out = np.zeros((len(model._system.equations), len(endo)))
            for (i, j), d in zip(fns.static_pairs, fns.static_jacobian(slots(fns.static_jacobian, x), p)):
                out[i, j] = d
            return out

    x0 = np.array([float((guess or {}).get(u, 1.0)) for u in endo])
    found = _newton_step(F, J, x0)  # one exact step solves a linear model (most Dynare imports): no root search
    if found is None:
        found = optimize.root(F, x0, jac=J, method="hybr", options={"xtol": 1e-12})
    if not found.success or np.max(np.abs(F(found.x))) > 1e-8:
        found = optimize.root(F, x0, jac=J, method="lm", options={"xtol": 1e-14})
    if np.max(np.abs(F(found.x))) > 1e-8:
        worst = int(np.argmax(np.abs(F(found.x))))
        raise ModelError(f"no steady state found from these starting values (equation {model._system.equations[worst].name!r} "
                         f"off by {F(found.x)[worst]:.3g}); give guess={{...}}, or check the model has one")
    solved = {u: (0.0 if abs(v) < 1e-14 else float(v)) for u, v in zip(endo, found.x)}
    return SteadyState({**solved, **values}, model)



def _newton_step(F: Any, J: Any, x0: np.ndarray) -> Any:
    """One exact Newton step from ``x0``, kept when it solves the equations (to 1e-10): a linear model's steady state
    in one Jacobian and one solve, where the root search takes several of each. None otherwise, and the search runs
    as before, from ``x0``."""
    from types import SimpleNamespace

    try:
        with np.errstate(all="ignore"):
            jac = J(x0)
            if jac.size == 0 or jac.shape[0] != jac.shape[1] or not np.all(np.isfinite(jac)):
                return None
            x1 = x0 + np.linalg.solve(jac, -F(x0))
            if np.all(np.isfinite(x1)) and np.max(np.abs(F(x1)), initial=0.0) <= 1e-10:
                return SimpleNamespace(x=x1, success=True)
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError, FloatingPointError, OverflowError):
        pass
    return None

# ---------------------------------------------------------------------------------------------
# linearisation


@dataclass(frozen=True, eq=False)
class LinearModel:
    """``A0 E[x(t+1)] + A1 x(t) + A2 x(t-1) + B e(t) = 0`` in deviations from the steady state."""

    A0: np.ndarray
    A1: np.ndarray
    A2: np.ndarray
    B: np.ndarray
    variables: tuple[str, ...]  # the model's endogenous variables, then auxiliary ones for longer lags and leads
    shocks: tuple[str, ...]
    steady: SteadyState
    observed: tuple[str, ...] = ()  # the model's own endogenous variables (not auxiliary)
    c: np.ndarray | None = None  # a constant term (+ c): equations that do not hold at the steady state (an OccBin regime)

    @api()
    def solve(self, start: LinearSolution | None = None) -> LinearSolution:
        """The unique stable solution ``x(t) = P x(t-1) + Q e(t)`` (Blanchard-Kahn checked).

        ``start``: a solution of the same model at nearby parameter values (the previous draw of an estimation). The
        solution is then found by iterating ``P = -(A0 P + A1)^-1 A2`` from it, much cheaper than the generalised
        Schur form on a large model, and checked: its own roots stable, the model's other roots explosive. If the
        iteration is slow or a check fails, the Schur form is used, as without ``start``."""
        if start is not None:
            warm = _warm_solve(self, start)
            if warm is not None:
                return warm
        return _solve(self)

    def roots(self) -> tuple[np.ndarray, int]:
        """The generalised eigenvalues of the model's pencil (``inf`` for infinite ones) and the number of predetermined
        variables: a unique stable solution has exactly that many roots inside the unit circle (Blanchard-Kahn). Found
        without solving, so they are there to see when the conditions fail."""
        F, G, m, _ = _pencil(self, _structure(self))
        if not len(F):
            return np.zeros(0), m
        alpha, beta = linalg.eigvals(G, F, homogeneous_eigvals=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(beta) > 1e-12, alpha / beta, np.inf), m


def _residual_system(model: Model, ss: SteadyState, shocks: tuple[str, ...], transform: Any = None
                     ) -> tuple[list[Any], tuple[str, ...], list[str], dict[str, float], Any]:
    """The equations as residuals in variables with at most one lag and one lead (auxiliary variables for longer
    ones), the variables, the model's own endogenous ones, steady-state values, and a resolver for evaluation."""
    bad = [s for s in shocks if s not in model.exogenous]
    if bad:
        raise ModelError(f"shocks must be exogenous variables of the model; {', '.join(bad)} are not")
    endo = list(model.endogenous)
    # auxiliary variables for lags and leads beyond one: x[-2] = x_l1[-1], x[+2] = x_f1[+1]
    residuals = [e.residual if transform is None else transform(e.residual) for e in model.model_equations]
    extra_eqs: list[Any] = []
    aux: list[str] = []
    seen: dict[str, list[int]] = {}
    for e in residuals:  # every variable's lags and leads, in one pass
        for r in e.refs():
            seen.setdefault(r.id, []).append(r.lag)
    for u in list(endo):
        lags = seen.get(u, [])
        deepest, highest = -min(lags, default=0), max(lags, default=0)
        for k in range(1, deepest):
            name = f"{u}_lag{k}"
            aux.append(name)
            prev = Name(u, -1) if k == 1 else Name(f"{u}_lag{k - 1}", -1)
            extra_eqs.append(_sub(Name(name, 0), prev))
        for k in range(1, highest):
            name = f"{u}_lead{k}"
            aux.append(name)
            nxt = Name(u, 1) if k == 1 else Name(f"{u}_lead{k - 1}", 1)
            extra_eqs.append(_sub(Name(name, 0), nxt))

    endo_set = set(endo)

    def rename(expr: Any) -> Any:
        def change(n: Any) -> Any:
            if not isinstance(n, Name) or n.id not in endo_set:
                return None
            if n.lag < -1:
                return Name(f"{n.id}_lag{-n.lag - 1}", -1)
            if n.lag > 1:
                return Name(f"{n.id}_lead{n.lag - 1}", 1)
            return None
        return expr.map(change)

    residuals = [rename(r) for r in residuals] + extra_eqs
    variables = tuple(endo + aux)
    values = dict(ss.values)
    for name in aux:  # auxiliaries equal their variable in the steady state
        base, _, tail = name.rpartition("_")
        k = int(tail[3:]) if tail.startswith("lag") else -int(tail[4:])  # x_lag2 is x two periods back; x_lead1 one ahead
        values[name] = values[base] - (getattr(ss, "growth", {}) or {}).get(base, 0.0) * k  # on the growth path
    for r in residuals:
        for ref in r.refs():
            if ref.lag != 0 and ref.id in shocks:  # shocks are exogenous (checked above)
                raise ModelError(f"shock {ref.id} enters with a lag or lead ({ref}); give it its own variable")

    def resolve(name: str, lag: int) -> float:
        return values[name] + (getattr(ss, "growth", {}) or {}).get(name.rsplit("_lag", 1)[0].rsplit("_lead", 1)[0], 0.0) * lag

    return residuals, variables, endo, values, resolve


def _compiled_system(model: Model, ss: SteadyState, shocks: tuple[str, ...]) -> tuple[list[Any], tuple[str, ...], list[str], dict[str, float], Any]:
    """``_residual_system`` for the compiled path: the model's own equations are handled by the compiled derivatives,
    so only the auxiliary equations (for lags and leads beyond one) are kept as expressions, and the structure, which
    does not depend on parameter values, is worked out once per model and shared by ``with_parameters`` copies. The
    references come from the symbolic residuals: substituting values never removes one (``0*x[-2]`` stays)."""
    shared = model.__dict__.setdefault("_shared", {})
    key = ("linear structure", shocks)
    if key not in shared:
        symbolic = type("_Symbolic", (), {})()
        symbolic.exogenous, symbolic.endogenous = model.exogenous, model.endogenous
        symbolic.model_equations = [type("_E", (), {"residual": e.residual})() for e in model._system.equations]
        residuals, variables, endo, _, _ = _residual_system(symbolic, SteadyState({**{v: 0.0 for v in model.endogenous},
                                                                                     **{v: 0.0 for v in model.exogenous}}), shocks)
        first = len(model._system.equations)
        shared[key] = ([None] * first + residuals[first:], variables, endo)
    residuals, variables, endo = shared[key]
    values = dict(ss.values)
    growth = getattr(ss, "growth", {}) or {}
    for name in variables[len(endo):]:  # auxiliaries equal their variable in the steady state
        base, _, tail = name.rpartition("_")
        k = int(tail[3:]) if tail.startswith("lag") else -int(tail[4:])
        values[name] = values[base] - growth.get(base, 0.0) * k

    def resolve(name: str, lag: int) -> float:
        return values[name] + growth.get(name.rsplit("_lag", 1)[0].rsplit("_lead", 1)[0], 0.0) * lag

    return list(residuals), variables, list(endo), values, resolve


def linearize(model: Model, ss: SteadyState, shocks: Sequence[str] = (), transform: Any = None,
              constant: bool = False) -> LinearModel:
    """``transform`` rewrites each equation's residual first (OccBin: one branch of each max or min); with
    ``constant``, equations that do not hold at the steady state keep their value there as the constant ``c``."""
    shocks = tuple(shocks)
    from .fast import functions as _compiled

    if transform is None and _compiled(model) is not None:
        residuals, variables, endo, values, resolve = _compiled_system(model, ss, shocks)
    else:
        residuals, variables, endo, values, resolve = _residual_system(model, ss, shocks, transform)
    n = len(variables)
    A0, A1, A2 = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    B = np.zeros((n, len(shocks)))
    from .fast import functions, parameter_values

    fns = functions(model) if transform is None else None  # rewritten equations: the interpreter
    first = 0
    c = np.zeros(n)
    if fns is not None:  # the model's own equations, compiled; auxiliary ones below
        p = parameter_values(model, fns)
        scale = 1e-8 * max(1.0, max(abs(v) for v in values.values()))
        trend = getattr(ss, "growth", {}) or {}
        at_steady = fns.residuals([values[name] + trend.get(name, 0.0) * lag for name, lag in fns.residuals.refs], p)
        for i, value in enumerate(at_steady):
            if abs(float(value)) > scale:
                if constant:  # a regime's equations away from the reference steady state: their constant
                    c[i] = float(value)
                    continue
                raise ModelError(f"equation {i + 1} does not hold at the steady state; is it really a steady state?")
        for (i, name, lag), d in zip(fns.pairs, fns.derivatives([values[nm] + trend.get(nm, 0.0) * lg for nm, lg in fns.derivatives.refs], p)):
            if name in endo:
                aux = name if -1 <= lag <= 1 else (f"{name}_lag{-lag - 1}" if lag < -1 else f"{name}_lead{lag - 1}")
                {1: A0, 0: A1, -1: A2}[max(-1, min(1, lag))][i, variables.index(aux)] += float(d)
            elif name in shocks:
                B[i, shocks.index(name)] += float(d)
        first = len(model._system.equations)
    for i, r in enumerate(residuals):
        if i < first:
            continue
        value = float(evaluate(r, resolve))
        if abs(value) > 1e-8 * max(1.0, max(abs(v) for v in values.values())):
            if not constant:
                raise ModelError(f"equation {i + 1} does not hold at the steady state; is it really a steady state?")
            c[i] = value
        for ref in r.refs():
            d = float(evaluate(diff(r, ref), resolve))
            if ref.id in variables:
                target = {1: A0, 0: A1, -1: A2}[ref.lag]
                target[i, variables.index(ref.id)] += d
            elif ref.id in shocks:
                B[i, shocks.index(ref.id)] += d
    return LinearModel(A0, A1, A2, B, variables, shocks, ss, tuple(endo), c if constant else None)


def _sub(a: Any, b: Any) -> Any:
    from ..expr import BinOp

    return BinOp("-", a, b)


# ---------------------------------------------------------------------------------------------
# the stable solution


@dataclass(frozen=True, eq=False)
class LinearSolution:
    """Decision rules ``x(t) = P x(t-1) + Q e(t)`` in deviations from the steady state."""

    P: np.ndarray
    Q: np.ndarray
    model: LinearModel
    eigenvalues: np.ndarray
    regimes: tuple[Any, ...] = ()  # known-date regimes that solve() follows by default (a template's)

    @property
    def variables(self) -> tuple[str, ...]:
        return self.model.variables

    @property
    def shocks(self) -> tuple[str, ...]:
        return self.model.shocks

    @property
    def endogenous(self) -> tuple[str, ...]:
        """The model's own variables (without the auxiliary ones for longer lags and leads)."""
        return tuple(self.model.observed)

    def rules(self) -> Table:
        """The decision rules: each variable on lagged variables and current shocks."""
        cols = [_lagged(v) for v in self.variables] + list(self.shocks)
        M = np.c_[self.P, self.Q]
        keep = [j for j in range(M.shape[1]) if np.any(np.abs(M[:, j]) > 1e-12)]
        rows = [[v] + [M[i, j] for j in keep] for i, v in enumerate(self.model.observed)]
        return Table(["", *[cols[j] for j in keep]], rows, title="Decision rules (deviations from the steady state)")

    @property
    def n_stable(self) -> int:
        """The number of stable roots (modulus below one; unit roots count as stable, as the solver counts them)."""
        return int(np.sum(np.abs(self.eigenvalues) < 1 + _UNIT))

    @property
    def n_predetermined(self) -> int:
        """The number of predetermined variables: those that appear lagged (Blanchard-Kahn's count to match)."""
        return len(_structure(self.model).lagged)

    def roots(self) -> Table:
        """The generalised eigenvalues: as many stable ones as variables that appear lagged (Blanchard-Kahn)."""
        mod = np.abs(self.eigenvalues)
        rows = [[i + 1, float(m), "stable" if m < 1 else "unit root" if m < 1 + _UNIT else "unstable"]
                for i, m in enumerate(np.sort(mod)) if np.isfinite(m)]
        bk = f"Blanchard-Kahn: stable roots {self.n_stable} (unit roots included), predetermined variables {self.n_predetermined}"
        return Table(["", "modulus", ""], rows, title="Roots", notes=[bk, f"{np.sum(~np.isfinite(mod))} infinite roots"])

    @api()
    def irf(self, periods: int = 20, shock_sd: Mapping[str, float] | None = None, shock_corr: Mapping[Any, float] | None = None,
            shocks: Sequence[str] | None = None) -> Any:
        """Responses to each shock (one unit, or ``shock_sd``), as deviations from the steady state. With correlated
        shocks (``shock_corr``, ``{("ey", "ei"): 0.3}``), orthogonalised as Dynare's ``stoch_simul``: the response to
        each shock is a column of ``Q`` times the Cholesky factor of the covariance, the shocks in the model's order
        (see ``pymetrix.models.covariance``). ``shocks``: only the responses to these shocks, in this order (all by
        default); the responses to each are the same as among all, the orthogonalisation included."""
        from ..errors import SpecificationError
        from ..var import IRF
        from .covariance import shock_factor

        if periods is None or int(periods) < 0 or int(periods) != periods:
            raise SpecificationError(f"periods is the horizon: a whole number at least 0, not {periods!r}")
        periods = int(periods)
        chosen = list(self.shocks) if shocks is None else [shocks] if isinstance(shocks, str) else list(shocks)
        unknown = [s for s in chosen if s not in self.shocks]
        if unknown or not chosen:
            raise SpecificationError(f"the solution has no shock {unknown[0]!r}; its shocks are {', '.join(self.shocks)}"
                                     if unknown else "shocks: give at least one shock, or leave it out for all")
        sd = np.array([float((shock_sd or {}).get(s, 1.0)) for s in self.shocks])
        values = np.empty((periods + 1, len(self.model.observed), len(self.shocks)))
        factor = shock_factor(self.shocks, sd, shock_corr)
        x = self.Q * sd if factor is None else self.Q @ factor
        k = len(self.model.observed)
        for h in range(periods + 1):
            values[h] = x[:k]
            x = self.P @ x
        if chosen != list(self.shocks):
            values = values[:, :, [self.shocks.index(s) for s in chosen]]
        return IRF(values, list(self.model.observed), chosen,
                   identification="rational-expectations solution" + ("" if factor is None else ", correlated shocks orthogonalised (Cholesky)"))

    def simulate(self, periods: int, shock_sd: Mapping[str, float], seed: int | None = None, levels: bool = True,
                 shock_corr: Mapping[Any, float] | None = None) -> pd.DataFrame:
        """A stochastic simulation: shocks drawn with standard deviations ``shock_sd`` (and correlations
        ``shock_corr``); levels or deviations."""
        from .covariance import shock_factor

        rng = make_rng(seed)
        sd = np.array([float(shock_sd.get(s, 0.0)) for s in self.shocks])
        factor = shock_factor(self.shocks, sd, shock_corr)
        x = np.zeros(len(self.variables))
        rows = []
        for _ in range(periods):
            x = self.P @ x + self.Q @ (sd * rng.normal(size=len(sd)) if factor is None else factor @ rng.normal(size=len(sd)))
            rows.append(x[: len(self.model.observed)].copy())
        frame = pd.DataFrame(rows, columns=list(self.model.observed))
        if levels:
            frame += pd.Series({v: self.model.steady.values[v] for v in self.model.observed})
        return frame

    def moments(self, shock_sd: Mapping[str, float] | None = None, lags: int = 4, shock_corr: Mapping[Any, float] | None = None) -> Any:
        """Standard deviations, autocorrelations and variance shares implied by the solution (``Moments``); shocks with
        standard deviation 1 unless ``shock_sd`` says otherwise, uncorrelated unless ``shock_corr`` says otherwise.
        See ``pymetrix.models.moments``."""
        from .moments import moments

        return moments(self, shock_sd, lags, shock_corr)

    def decompose(self, levels: Any, start: str = "first", regimes: Any = None) -> Any:
        """Paths of the model's variables (a DataFrame or Dataset in levels, e.g. smoothed estimates) split into the
        shocks' contributions and the effect of the starting point (a ``Decomposition``). ``start="first"`` counts the
        first period's state, its shocks included, as the initial condition; ``"presample"`` takes the first period as
        the pre-sample state x(-1) and decomposes from the next, so that period's shocks go to the shocks.
        ``regimes``: known-date regimes over the path (``Regime`` objects or text), decomposed through their
        time-varying rules with their constants a part of their own, ``"regimes"`` (see ``pymetrix.models.decompose``)."""
        from .decompose import decompose

        return decompose(self, levels.frame if hasattr(levels, "frame") else levels, start, regimes)

    def fevd(self, periods: int = 20, shock_sd: Mapping[str, float] | None = None, shock_corr: Mapping[Any, float] | None = None) -> Any:
        """The share of each variable's h-step forecast-error variance due to each shock (an ``FEVD``); with correlated
        shocks (``shock_corr``), of the shocks orthogonalised by Cholesky in the model's order, as Dynare."""
        from .moments import fevd

        return fevd(self, periods, shock_sd, shock_corr)

    def solve(self, data: Dataset | None, span: tuple[Any, Any], scenario: Any = None, draws: int | None = None,
              shock_sd: Mapping[str, float] | None = None, seed: int | None = None, regimes: Any = None,
              shock_corr: Mapping[Any, float] | None = None) -> Any:
        """Simulate over ``span`` under a ``Scenario``: shock paths (anticipated unless named ``unanticipated``), and
        targets held by free shocks (exogenize and endogenize). Starting values come from ``data`` (levels), or the
        steady state when ``data`` is None. See ``pymetrix.models.plans``. With ``draws``, a stochastic simulation as
        well: shocks drawn in each period with standard deviations ``shock_sd`` and correlations ``shock_corr``
        (``pymetrix.models.stochastic``).

        ``regimes``: parameters or equations that change at known dates (``Regime`` objects, or text such as
        ``"2025Q1..2026Q4: rho=1, phi_pi=0"``), known in advance unless a regime says otherwise; the path follows the
        time-varying decision rules they give; by default, the solution's own (a template's). See
        ``pymetrix.models.regimes``."""
        from .plans import _solve_plan

        if regimes is None:
            regimes = getattr(self, "regimes", ()) or None

        base, used, plan = _solve_plan(self, data, span, scenario, regimes)
        if not draws:
            return base
        from ..errors import SpecificationError
        from .model import with_simulation
        from .stochastic import first_order

        if int(draws) < 2:
            raise SpecificationError("draws is the number of simulations: at least 2")
        rules = None if plan is None else [(P, Q) for P, _, Q in plan.period_rules(len(plan.periods))]
        return with_simulation(base, first_order(used, base, int(draws), shock_sd, seed, rules, shock_corr=shock_corr))

    @api()
    def system(self, measurement: Any, shock_sd: Mapping[str, float] | None = None, measurement_sd: Mapping[str, float] | None = None,
               shock_corr: Mapping[Any, float] | None = None) -> Any:
        """The solution in state-space form (a ``StateSpaceSystem``). ``measurement`` is a ``MeasurementMap``
        linking data to the model, or a list of model variables observed as they are, in levels. Measurement
        equations are linearised at the steady state (exact when they are linear); lags in them add lagged
        states. ``shock_sd`` and ``measurement_sd`` are standard deviations; ``shock_corr`` correlations of the shocks
        (``{("ey", "ei"): 0.3}``: the filter's shock covariance then has them off its diagonal)."""
        from ..statespace import StateSpaceSystem
        from .covariance import normalise, shock_covariance
        from .measurement import MeasurementMap

        if measurement is None:
            raise ModelError("give measurement equations (a MeasurementMap) or the model variables observed")
        shock_sd = dict(shock_sd or {})  # unit standard deviations unless given
        if not isinstance(measurement, MeasurementMap):
            measurement = MeasurementMap.direct(list(measurement), measurement_sd)
        elif measurement_sd:
            measurement = measurement.with_errors(measurement_sd)
        if not measurement.observables:
            raise ModelError("the measurement map observes no series (it only reports some): a state-space form needs "
                             "measurement equations for observed series")
        missing = sorted(measurement.variables() - set(self.model.observed))
        if missing:
            raise ModelError(f"{', '.join(missing)} are not variables of the model")
        k, L = len(self.variables), measurement.max_lag
        keys = [(v, -lag) for lag in range(L + 1) for v in self.variables]
        names = tuple(v if lag == 0 else f"{v}[{lag:+d}]" for v, lag in keys)
        n = k * (L + 1)
        T = np.zeros((n, n))
        T[:k, :k] = self.P
        for lag in range(1, L + 1):  # each lagged copy is last period's state
            T[lag * k:(lag + 1) * k, (lag - 1) * k:lag * k] = np.eye(k)
        R = np.zeros((n, len(self.shocks)))
        R[:k] = self.Q
        steady = dict(self.model.steady.values)
        for v in self.variables:  # auxiliary variables equal their variable in the steady state
            steady.setdefault(v, steady.get(v.rsplit("_", 1)[0], 0.0))
        Z, d = measurement.linearize(keys, steady)
        H = measurement.error_covariance()  # diagonal unless error_corr correlates two series' errors
        Qcov = np.diag([float(shock_sd.get(s, 1.0)) ** 2 for s in self.shocks])
        corr = {k: v for k, v in normalise(self.shocks, shock_corr).items() if v != 0}
        if corr:
            Qcov = shock_covariance(self.shocks, np.array([float(shock_sd.get(s, 1.0)) for s in self.shocks]), corr)
        observed = tuple(self.model.observed)
        growth = dict(getattr(self.model.steady, "growth", {}) or {})
        base = [v if v in growth else v.rsplit("_", 1)[0] for v, _ in keys]  # an auxiliary lag grows with its variable
        trend = np.array([growth.get(b, 0.0) for b in base]) if growth else None
        lagged = np.flatnonzero(np.any(self.P != 0, axis=0))  # P's other columns are zero: their roots are zero
        block = self.P[np.ix_(lagged, lagged)]
        unit_roots = bool(block.size) and float(np.max(np.abs(np.linalg.eigvals(block)))) >= 1 - 1e-9
        stationary = () if unit_roots else tuple(True for _ in range(n))  # with unit roots the filter starts them diffuse
        return StateSpaceSystem(names, measurement.observables, tuple(self.shocks), T, np.zeros(n), R, Qcov, Z, d, H,
                                stationary, measurement=measurement, shock_sd=dict(shock_sd), shock_corr=corr,
                                levels={v: steady[v] for v in observed}, variables=observed, trend=trend,
                                trend_lags=np.array([lag for _, lag in keys], dtype=float) if growth else None)

    def filter(self, data: Dataset, observables: Sequence[str], shock_sd: Mapping[str, float],
               measurement_sd: Mapping[str, float] | None = None, span: Any = None, shock_corr: Mapping[Any, float] | None = None) -> pd.DataFrame:
        """Estimate every model variable from the observed ones with the Kalman smoother (in levels):
        e.g. the output gap and neutral rate behind observed output, inflation and interest rates.
        ``observables`` is a list of model variables observed as they are, or a ``MeasurementMap``."""
        return self.system(observables, shock_sd, measurement_sd, shock_corr).filter(data, span).frame[list(self.model.observed)]

    def particle_filter(self, data: Dataset, measurement: Any, shock_sd: Mapping[str, float] | None = None, span: Any = None,
                        particles: int = 2000, seed: int | None = 0, resample: float = 0.5, proposal: str = "bootstrap",
                        burn: int | None = None, presample: int = 0, quantiles: Sequence[float] = (0.05, 0.5, 0.95),
                        shock_corr: Mapping[Any, float] | None = None, measurement_sd: Mapping[str, float] | None = None,
                        smooth_paths: int = 0, candidates: int | None = 200) -> Any:
        """The particle filter on the first-order solution (a ``ParticleFilter``): with linear measurement equations an
        estimate of what the Kalman filter computes exactly, a check on both; mostly for second-order solutions
        (``SecondOrderSolution.particle_filter``). ``smooth_paths`` > 0 adds the particle smoother. See
        ``pymetrix.models.particle``."""
        from .particle import particle_filter

        return particle_filter(self, data, measurement, shock_sd, span, particles, seed, resample, proposal, burn, presample,
                               quantiles, shock_corr, measurement_sd, smooth_paths, candidates)

    def plot(self, periods: int = 20, shock_sd: Mapping[str, float] | None = None, shock_corr: Mapping[Any, float] | None = None,
             **kw: Any) -> Any:
        """Impulse responses to every shock, as a grid of panels."""
        return self.irf(periods, shock_sd, shock_corr).plot(**kw)

    def summary(self) -> Table:
        return self.rules()

    def _repr_html_(self) -> str:
        return self.rules()._repr_html_()

    def __str__(self) -> str:
        return str(self.rules())


def _lagged(name: str) -> str:
    """``x[-1]``, with auxiliary variables shown as what they stand for: ``x_lag1[-1]`` is ``x[-2]``."""
    base, _, tail = name.rpartition("_")
    for prefix, sign in (("lag", -1), ("lead", 1)):
        if base and tail.startswith(prefix) and tail[len(prefix):].isdigit():
            k = sign * int(tail[len(prefix):]) - 1
            return base if k == 0 else f"{base}[{k:+d}]"
    return f"{name}[-1]"


_UNIT = 1e-6  # roots within this of one are unit roots: kept, as stable (a trending model has them)


def _solve(lin: LinearModel) -> LinearSolution:
    """QZ on the smallest pencil of ``A0 x(t+1) + A1 x(t) + A2 x(t-1) = 0``, as Dynare builds it.

    Variables are static (only at ``t``), predetermined (lagged: L) or forward-looking (led: F); a variable can be both.
    The static ones are taken out first: a QR of their columns of ``A1`` leaves ``n - s`` equations without them, and
    they follow from the others at the end. The pencil is on ``y(t) = [x_L(t-1); x_F(t)]``, of size ``|L| + |F|``
    (identities and variables that are neither lagged nor led add no roots), with one extra equation for each variable
    in both. The real generalised Schur form is used (several times faster than the complex one). A unique stable
    solution has exactly ``|L|`` stable roots (Blanchard-Kahn). Tests check it against QZ on the full companion pencil."""
    n = len(lin.variables)
    st = _structure(lin)
    lagged, forward = st.lagged, st.forward
    F, G, m, f = _pencil(lin, st)
    size = m + f

    def stable(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:  # unit roots count as stable (trends, as in IRIS)
        return np.abs(alpha) < np.abs(beta) * (1 + _UNIT)

    if size:
        AA, BB, alpha, beta, Q, Zq = linalg.ordqz(G, F, sort=stable, output="real")
    else:
        AA = BB = Zq = np.zeros((0, 0))
        alpha = beta = np.zeros(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        eig = np.where(np.abs(beta) > 1e-12, alpha / beta, np.inf)
    n_stable = int(np.sum(np.abs(eig) < 1 + _UNIT))
    if n_stable < m:
        raise ModelError(f"no stable solution: {n_stable} stable roots for {m} predetermined variables (Blanchard-Kahn). "
                         "Too many explosive roots: check signs and the strength of policy responses")
    if n_stable > m:
        raise ModelError(f"the solution is not unique: {n_stable} stable roots for {m} predetermined variables (indeterminacy, "
                         "Blanchard-Kahn). E.g. a Taylor rule that responds less than one for one to inflation")
    P = np.zeros((n, n))
    if m:
        Z11, Z21 = Zq[:m, :m], Zq[m:, :m]
        if np.linalg.cond(Z11) > 1e12:
            raise ModelError("the stable roots do not determine the lagged variables (Z11 singular): the solution is not unique")
        try:  # on the stable subspace y = Z1 w: w(t+1) = BB11^-1 AA11 w(t), and x_L(t-1) = Z11 w(t)
            step = np.linalg.solve(BB[:m, :m], AA[:m, :m])
        except np.linalg.LinAlgError:
            raise ModelError("the stable roots do not determine the lagged variables: the solution is not unique") from None
        P[np.ix_(lagged, lagged)] = np.linalg.solve(Z11.T, (Z11 @ step).T).T  # Z11 BB11^-1 AA11 Z11^-1
        if f:
            P[np.ix_(forward, lagged)] = np.linalg.solve(Z11.T, Z21.T).T  # Z21 Z11^-1
        _statics(P, st)
    return _finish(lin, P, eig)


def _pencil(lin: LinearModel, st: _Structure) -> tuple[np.ndarray, np.ndarray, int, int]:
    """``F y(t+1) = G y(t)`` on ``y(t) = [x_L(t-1); x_F(t)]`` (see ``_solve``), with ``|L|`` and ``|F|``."""
    n = len(lin.variables)
    lagged, forward = st.lagged, st.forward
    m, f, s = len(lagged), len(forward), len(st.static)
    a0, a1, a2 = st.A0[s:], st.A1[s:], st.A2[s:]
    pos_l = {j: i for i, j in enumerate(lagged)}
    pos_f = {j: m + i for i, j in enumerate(forward)}
    both = [j for j in forward if j in pos_l]
    # F y(t+1) = G y(t): a0_F x_F(t+1) + a1_L x_L(t) = -a2_L x_L(t-1) - a1_F' x_F'(t)  (F': led, not lagged),
    # and x_b(t) is in both y(t+1) and y(t) for each variable b that is both lagged and led
    size = m + f
    F = np.zeros((size, size))
    G = np.zeros((size, size))
    F[:n - s, :m] = a1[:, lagged]
    F[:n - s, m:] = a0[:, forward]
    G[:n - s, :m] = -a2[:, lagged]
    for j in forward:
        if j not in pos_l:
            G[:n - s, pos_f[j]] = -a1[:, j]
    for r, b in enumerate(both, start=n - s):
        F[r, pos_l[b]] = 1.0
        G[r, pos_f[b]] = 1.0
    return F, G, m, f


@dataclass(frozen=True)
class _Structure:
    """Which variables are lagged, led and static, and the equations rotated so that the first ``s`` hold the static
    variables (with ``R11``, upper triangular, their coefficients) and the rest do not."""

    lagged: list[int]
    forward: list[int]
    static: list[int]
    dynamic: list[int]
    R11: np.ndarray
    A0: np.ndarray
    A1: np.ndarray
    A2: np.ndarray


def _structure(lin: LinearModel) -> _Structure:
    A0, A1, A2 = lin.A0, lin.A1, lin.A2
    led = np.any(A0 != 0, axis=0)
    lagged_mask = np.any(A2 != 0, axis=0)
    static_mask = ~led & ~lagged_mask
    static = list(np.flatnonzero(static_mask))
    s = len(static)
    R11 = np.zeros((0, 0))
    if s:  # rotate the equations so that the first s hold the static variables and the rest do not
        Qs_, R = linalg.qr(A1[:, static])
        R11 = R[:s, :s]
        diag = np.abs(np.diag(R11))
        if diag.size and diag.min() <= 1e-12 * max(1.0, float(np.max(np.abs(A1)))):
            raise ModelError("the variables that appear without lags or leads are not determined by the equations: "
                             "check for two equations that say the same thing")
        A0, A1, A2 = Qs_.T @ A0, Qs_.T @ A1, Qs_.T @ A2
    return _Structure(list(np.flatnonzero(lagged_mask)), list(np.flatnonzero(led)), static, list(np.flatnonzero(~static_mask)),
                      R11, A0, A1, A2)


def _statics(P: np.ndarray, st: _Structure) -> None:
    """The static variables' rows of ``P`` from the others: ``R11 x_S(t) = -(A0_F x_F(t+1) + A1_D x_D(t) +
    A2_L x_L(t-1))`` in the rotated first ``s`` equations."""
    s = len(st.static)
    if not s or not st.lagged:
        return
    lagged, forward, dynamic = st.lagged, st.forward, st.dynamic
    PL = P[np.ix_(lagged, lagged)]
    rhs = (st.A0[:s][:, forward] @ P[np.ix_(forward, lagged)] @ PL + st.A1[:s][:, dynamic] @ P[np.ix_(dynamic, lagged)]
           + st.A2[:s][:, lagged])
    P[np.ix_(st.static, lagged)] = -linalg.solve_triangular(st.R11, rhs)


def _finish(lin: LinearModel, P: np.ndarray, eig: np.ndarray) -> LinearSolution:
    """The impact of shocks, ``Q = -(A0 P + A1)^-1 B``, and a check that ``P`` solves the model."""
    n = len(lin.variables)
    M = lin.A0 @ P + lin.A1
    try:
        Qs = -np.linalg.solve(M, lin.B) if lin.B.size else np.zeros((n, 0))
    except np.linalg.LinAlgError:
        raise ModelError("the impact of shocks is not determined (A0 P + A1 is singular)") from None
    check = lin.A0 @ P @ P + lin.A1 @ P + lin.A2
    if np.max(np.abs(check)) > 1e-7 * max(1.0, np.max(np.abs(lin.A1))):
        raise ModelError("the solution does not satisfy the model (numerical problem)")
    return LinearSolution(P, Qs, lin, eig)


_WARM_BUDGET = 30.0  # iterations a warm start may take, in units of the cost of QZ relative to one iteration (calibrated)
_WARM_TOL = 1e-13  # relative change in P at which the iteration has converged


def _warm_solve(lin: LinearModel, start: LinearSolution) -> LinearSolution | None:
    """The solution by fixed-point iteration from a nearby one, or None to use the generalised Schur form.

    The static variables are taken out as for QZ; on the remaining ``d`` equations and variables,
    ``P = -(A0 P + A1)^-1 A2`` converges to the stable solution from a start close to it, at the rate of the largest
    stable root over the smallest explosive one. It hands over to QZ when a step does not halve the last, or when the
    iterations would cost more than QZ on its pencil (of size ``|L| + |F|``, against ``d`` for each iteration).
    Blanchard-Kahn is checked on the result through ``A0 l^2 + A1 l + A2 = (A0 l + A0 P + A1)(l I - P)``: the roots
    of ``P`` must be stable and those of ``A0 l + A0 P + A1`` (``l = -1/mu``, ``mu`` the eigenvalues of
    ``(A0 P + A1)^-1 A0`` on the led variables) explosive. Both are small eigenproblems."""
    n = len(lin.variables)
    if start.model.variables != lin.variables or start.P.shape != (n, n):
        return None
    try:
        st = _structure(lin)
    except ModelError:
        return None
    lagged, forward, dynamic = st.lagged, st.forward, st.dynamic
    if not lagged:
        return None
    s, d = len(st.static), len(dynamic)
    budget = max(3, int(_WARM_BUDGET * ((len(lagged) + len(forward)) / d) ** 3))
    at = {j: i for i, j in enumerate(dynamic)}
    lag_d = [at[j] for j in lagged]
    fwd_d = [at[j] for j in forward]
    a0, a1, a2 = st.A0[s:][:, dynamic], st.A1[s:][:, dynamic], st.A2[s:][:, dynamic]
    a0F, a2L = a0[:, fwd_d], a2[:, lag_d]
    Pd = np.zeros((d, d))
    Pd[:, lag_d] = start.P[np.ix_(dynamic, lagged)]
    previous = np.inf
    with np.errstate(all="ignore"):
        for i in range(budget):
            try:
                new = -np.linalg.solve(a0F @ Pd[fwd_d] + a1, a2L)
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(new)):
                return None
            step = float(np.max(np.abs(new - Pd[:, lag_d])))
            Pd[:, lag_d] = new
            if step <= _WARM_TOL * max(1.0, float(np.max(np.abs(new)))):
                break
            if i >= 2 and step > 0.5 * previous:  # converging slowly (roots close to the unit circle): QZ is cheaper
                return None
            previous = step
        else:
            return None
        stable = np.linalg.eigvals(Pd[np.ix_(lag_d, lag_d)])
        if np.any(np.abs(stable) >= 1 + _UNIT):
            return None
        try:
            mu = np.linalg.eigvals(np.linalg.solve(a0 @ Pd + a1, a0F)[fwd_d]) if fwd_d else np.zeros(0)
        except np.linalg.LinAlgError:
            return None
        if np.any(np.abs(mu) * (1 + _UNIT) >= 1):  # another root stable: not unique (or the iteration found the wrong one)
            return None
        other = np.where(mu != 0, -1 / np.where(mu != 0, mu, 1), np.inf)
    P = np.zeros((n, n))
    P[np.ix_(dynamic, lagged)] = Pd[:, lag_d]
    _statics(P, st)
    return _finish(lin, P, np.concatenate([stable, other]))
