"""Stochastic simulation: a model solved many times with random errors, for the distribution of its paths.

    sol = model.solve(data, ("2025Q1", "2027Q4"), draws=1000, seed=1)     # errors drawn for each fitted equation
    sol.quantiles([0.05, 0.5, 0.95]); sol.mean(); sol.sd(); sol.fan()

**Backward-looking models** (fitted equations and identities, solved period by period). Each draw adds an error to
each fitted equation in every period of the span, as an add factor (the equation as estimated, ``lhs = rhs + e``),
and solves the model again, nonlinear equations and all. The errors are

- ``errors="normal"``: normal, with each equation's standard error of regression (SSR / (n - k)) and the correlations
  of the equations' residuals over the periods they share (EViews' default);
- ``errors="bootstrap"``: the residuals themselves, a whole period of them at a time (so their correlations and
  shapes are kept), drawn with replacement from the periods every equation has a residual;
- ``errors=<a DataFrame>``: a sample of residuals given (periods by equation, columns named by equation, as
  ``Model.add_factors`` returns them for a historical span), resampled the same way, a whole period of them at a time:
  the equations shocked are the frame's columns, fitted or not. This is how the Federal Reserve's ``stochsim.prg``
  draws FRB/US's shocks, from the historical errors of its stochastic equations (de-meaned first: the frame is used as
  given).

``error_scale`` multiplies the errors drawn in some periods (``{"2026Q2": 0.5}``: the first quarter's shocks halved, as
``stochsim.prg`` does when that quarter is already partly known). Each draw's errors are kept with the draws
(``Simulations.errors``, draws by periods by equation, in the order of ``Simulations.error_names``). ``workers=n``
solves the draws in ``n`` processes, each a share of them; the errors are drawn first, in this process, so the draws
are the same for any number of workers (not with GARCH errors, whose draws are solved here).

``shock_sd`` gives an equation (by name) another standard deviation (correlations kept), or one to an equation that
was not estimated (calibrated equations have no errors otherwise). ``coefficients=True`` also draws each fitted
equation's coefficients, once per draw, from the normal with their estimated covariance (equations independently).
The draws are solved together, period by period, by Newton's method; a draw that does not converge is dropped (its
paths are missing) and counted. Fixed variables and targets hold in every draw. The equations are compiled once
(``pymetrix.models.fast``) and evaluated for all the draws at once, so a large model (FRB/US's 284 equations) takes
seconds a period for hundreds of draws.

**First-order solutions** of models with expectations (``solver=FirstOrder()``, or ``LinearSolution.solve``). The
decision rules are linear, so each draw is the deterministic solution plus the response to shocks drawn in every
period of the span, unanticipated, with standard deviations ``shock_sd`` (by shock): exact, and fast.

The result is the deterministic ``Solution`` with its draws (``Solution.simulation``): ``quantiles``, ``mean``,
``sd``, ``bands`` and ``fan`` summarise them. The draws are reproducible with ``seed``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..errors import SpecificationError
from ..expr import Name, Num, evaluate
from ..expr.diff import diff
from .solvers import ModelError
from .._random import generator as make_rng

__all__ = ["Simulations"]


@dataclass(frozen=True, eq=False)
class Simulations:
    """The draws of a stochastic simulation: ``values[draw, period, variable]`` over ``periods``, and how they were
    made (``info``: draws, seed, errors, failures, each equation's or shock's standard deviation)."""

    values: np.ndarray
    names: tuple[str, ...]
    periods: pd.PeriodIndex
    info: Mapping[str, Any] = field(default_factory=dict)
    errors: np.ndarray | None = None  # the errors drawn (a backward-looking model): draws x periods x equations
    error_names: tuple[str, ...] = ()  # the equations of ``errors``' last axis

    def __getitem__(self, name: str) -> pd.DataFrame:
        """One variable's draws: periods by draws."""
        j = self.names.index(name)
        return pd.DataFrame(self.values[:, :, j].T, index=self.periods, columns=pd.RangeIndex(self.values.shape[0], name="draw"))

    def quantile(self, q: float) -> pd.DataFrame:
        return pd.DataFrame(np.nanquantile(self.values, q, axis=0), index=self.periods, columns=list(self.names))

    def error(self, equation: str) -> pd.DataFrame:
        """One equation's errors as drawn: periods by draws."""
        if self.errors is None or equation not in self.error_names:
            raise SpecificationError(f"no errors drawn for {equation!r}" + (f"; drawn for {', '.join(self.error_names)}" if self.error_names else ""))
        j = self.error_names.index(equation)
        return pd.DataFrame(self.errors[:, :, j].T, index=self.periods, columns=pd.RangeIndex(self.errors.shape[0], name="draw"))


def _rng(seed: int | None) -> np.random.Generator:
    return make_rng(seed)


# ---------------------------------------------------------------------------------------------
# first-order solutions: the deterministic path plus simulated deviations


def first_order(sol: Any, base: Any, draws: int, shock_sd: Mapping[str, float] | None, seed: int | None,
                rules: Any = None, shock_corr: Mapping[Any, float] | None = None) -> Simulations:
    """Draws around ``base`` (``sol.solve``'s deterministic Solution): shocks unanticipated in each period of its span,
    normal with standard deviations ``shock_sd`` and correlations ``shock_corr`` (``{(a, b): r}``); ``rules``: each
    period's ``(P, Q)`` when they change with known-date regimes."""
    from .covariance import normalise, shock_factor
    shocks = list(sol.shocks)
    unknown = sorted(set(shock_sd or {}) - set(shocks))
    if unknown:
        raise SpecificationError(f"shock_sd names {', '.join(unknown)}, which are not shocks of this solution ({', '.join(shocks)})")
    sd = np.array([float((shock_sd or {}).get(s, 0.0)) for s in shocks])
    if not np.any(sd > 0):
        raise SpecificationError("a stochastic simulation of a first-order solution needs the shocks' standard deviations: "
                                 f"shock_sd={{{', '.join(repr(s) + ': ...' for s in shocks[:3])}}}")
    if base.scenario is not None and (base.scenario.hold or base.scenario.ranges):
        raise SpecificationError("targets (exogenize and endogenize) with a stochastic simulation are not supported for "
                                 "first-order solutions: simulate without targets, or use a backward-looking model")
    first, last = base.span
    periods = pd.period_range(first, last, freq=base.frame.index.freq)
    observed = list(sol.model.observed)
    variables = list(sol.variables)
    where = [variables.index(v) for v in observed]
    factor = shock_factor(shocks, sd, shock_corr)  # None: independent shocks, drawn as they always were
    rng = _rng(seed)
    x = np.zeros((draws, len(variables)))
    dev = np.empty((draws, len(periods), len(observed)))
    for t in range(len(periods)):
        P, Q = (sol.P, sol.Q) if rules is None else rules[t]
        z = rng.standard_normal((draws, len(shocks)))
        x = x @ P.T + (z * sd if factor is None else z @ factor.T) @ Q.T
        dev[:, t] = x[:, where]
    levels = base.frame.loc[periods, observed].to_numpy(float)
    info = {"method": "first-order solution: unanticipated shocks in each period", "draws": int(draws), "seed": seed,
            "shock_sd": {s: float(v) for s, v in zip(shocks, sd) if v > 0}, "failed": 0}
    if factor is not None:
        info["shock_corr"] = {f"{a},{b}": r for (a, b), r in normalise(shocks, shock_corr).items() if r != 0}
    return Simulations(levels[None, :, :] + dev, tuple(observed), periods, info)


# ---------------------------------------------------------------------------------------------
# backward-looking models: errors as add factors, all draws solved together


def _fitted(model: Any) -> dict[str, Any]:
    """The model's fitted equations, by equation name."""
    from ..estimation import FittedEquation

    return {item.equation.label: item for item in model.equations if isinstance(item, FittedEquation)}


def _garch(fit: Any) -> bool:
    return getattr(fit.equation, "variance", None) is not None


def _errors(model: Any, names: list[str], periods: int, draws: int, kind: str, shock_sd: Mapping[str, float] | None,
            rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    """Errors for the stochastic equations ``names`` (draws x periods x equations), how they were made, and the
    equations with GARCH variance, whose columns are standardised draws, scaled period by period in ``backward``."""
    from scipy import stats

    if kind not in ("normal", "bootstrap"):
        raise SpecificationError("errors is 'normal' or 'bootstrap'")
    fitted = _fitted(model)
    override = dict(shock_sd or {})
    unknown = sorted(set(override) - set(names))
    if unknown:
        raise SpecificationError(f"shock_sd names {', '.join(unknown)}, which are not stochastic equations solved here "
                                 f"({', '.join(names) or 'none'})")
    with_fit = [n for n in names if n in fitted]
    garch = [n for n in with_fit if _garch(fitted[n])]
    fixed = sorted(set(garch) & set(override))
    if fixed:
        raise SpecificationError(f"{', '.join(fixed)} have GARCH variance, which sets the size of their errors: leave them out of shock_sd")
    sd = {n: float(fitted[n].stats["se_regression"]) for n in with_fit if n not in garch}
    standard = {n: (fitted[n].extra["std_resid"] if n in garch else fitted[n].resid / sd[n]).astype(float) for n in with_fit}
    common = pd.concat(standard, axis=1).dropna() if with_fit else pd.DataFrame()
    scale = {n: (override[n] / sd[n] if sd[n] > 0 else 0.0) if n in override else 1.0 for n in sd}
    out = np.zeros((draws, periods, len(names)))
    info: dict[str, Any] = {"errors": kind, "sd": {}, "not drawn": [n for n in names if n not in fitted and n not in override]}
    if with_fit:
        cols = [names.index(n) for n in with_fit]
        if kind == "bootstrap":
            if len(common) < 10:
                raise ModelError(f"the fitted equations share {len(common)} periods of residuals: too few to bootstrap "
                                 "(use errors='normal')")
            values = common.to_numpy(float)[rng.integers(0, len(common), size=(draws, periods))]  # standardised, a period at a time
            info["periods resampled"] = len(common)
        else:
            if len(with_fit) > 1 and len(common) < len(with_fit) + 2:
                raise ModelError(f"the fitted equations share {len(common)} periods of residuals: too few for their "
                                 "correlations (give shock_sd and simulate them one at a time, or fit them on a common sample)")
            corr = common.corr().to_numpy(float) if len(with_fit) > 1 else np.ones((1, 1))
            values = rng.standard_normal((draws, periods, len(with_fit))) @ np.linalg.cholesky(corr + 1e-12 * np.eye(len(with_fit))).T
            for j, n in enumerate(with_fit):  # t errors: t margins, the normal draws' correlations kept (a Gaussian copula)
                nu = fitted[n].params.as_dict().get("nu") if n in garch else None
                if nu:
                    values[:, :, j] = stats.t.ppf(stats.norm.cdf(values[:, :, j]), nu) * np.sqrt((nu - 2) / nu)
            if len(with_fit) > 1:
                info["correlations from"] = len(common)
        for j, n in enumerate(with_fit):
            if n not in garch:
                values[:, :, j] *= sd[n] * scale[n]
        out[:, :, cols] = values
        info["sd"].update({n: sd[n] * scale[n] for n in sd})
        if garch:
            info["GARCH"] = garch
    for n in names:
        if n not in fitted and n in override:
            out[:, :, names.index(n)] = rng.standard_normal((draws, periods)) * float(override[n])
            info["sd"][n] = float(override[n])
    return out, info, garch


def _resampled(sample: pd.DataFrame, equations: list[str], periods: int, draws: int,
               rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    """Errors resampled from a sample of residuals given (periods by equation), a whole period at a time."""
    names = [str(c) for c in sample.columns]
    unknown = [n for n in names if n not in equations]
    if unknown:
        raise SpecificationError(f"the residual sample has columns {', '.join(unknown[:5])}, which are not equations solved "
                                 "here: name its columns by equation (as Model.add_factors does)")
    if len(set(names)) != len(names):
        raise SpecificationError("the residual sample names an equation twice")
    values = sample.to_numpy(float)
    if len(values) < 2:
        raise SpecificationError("the residual sample needs at least 2 periods to resample")
    gaps = [n for j, n in enumerate(names) if not np.all(np.isfinite(values[:, j]))]
    if gaps:
        raise SpecificationError(f"the residual sample has missing values for {', '.join(gaps[:5])}: give complete periods")
    drawn = values[rng.integers(0, len(values), size=(draws, periods))]  # whole periods: draws x periods x equations
    info: dict[str, Any] = {"errors": "resampled from the residuals given, a period at a time", "periods resampled": len(values)}
    if isinstance(sample.index, pd.PeriodIndex):
        info["residuals from"] = f"{sample.index[0]} to {sample.index[-1]}"
    return drawn, info, names


def _scales(error_scale: Mapping[Any, float] | None, periods: pd.PeriodIndex) -> np.ndarray:
    """Each period's multiplier of the errors drawn."""
    out = np.ones(len(periods))
    for key, value in (error_scale or {}).items():
        p = pd.Period(str(key), freq=periods.freq)
        if p not in periods:
            raise SpecificationError(f"error_scale names {p}, which is not in the simulation span ({periods[0]} to {periods[-1]})")
        if not np.isfinite(float(value)) or float(value) < 0:
            raise SpecificationError(f"error_scale for {p} is a multiplier of 0 or more, not {value!r}")
        out[periods.get_loc(p)] = float(value)
    return out


class _Batch:
    """One period's equations for the draws (each draw a row): residuals and the Jacobian in this period's unknowns,
    for all the draws or some (``rows``, their indices), compiled (``pymetrix.models.fast``) or, where an equation
    cannot be compiled, interpreted."""

    def __init__(self, model: Any, eqs: list[Any], unknowns: list[str], residuals: list[Any], arrays: dict[str, np.ndarray],
                 params: dict[str, Any], draws: int):
        from .fast import functions

        self.unknowns, self.arrays, self.params, self.D = unknowns, arrays, params, draws
        self.uidx = {u: j for j, u in enumerate(unknowns)}
        self.residual_exprs = residuals
        fns = functions(model, eqs)
        self.compiled = None
        if fns is not None:
            from ..expr.compile import compile_exprs

            wanted = [(i, self.uidx[name], d) for (i, name, lag), d in fns._derivatives if lag == 0 and name in self.uidx]
            jacobian = compile_exprs(tuple(d for _, _, d in wanted), fns.coefs)
            self.rows = np.array([i for i, _, _ in wanted], dtype=int)
            self.cols = np.array([j for _, j, _ in wanted], dtype=int)
            self.compiled = (fns.residuals, jacobian, [params[c] for c in fns.coefs])
        else:
            self.jac = [(i, j, d) for i, r in enumerate(residuals) for j, u in enumerate(unknowns)
                        if (d := diff(r, Name(u, 0))) != Num(0.0)]
        self.pos = 0

    def at(self, pos: int) -> None:
        self.pos = pos
        if self.compiled is not None:
            self._fixed = [self._gather(f.refs) for f in self.compiled[:2]]

    def _value(self, name: str, lag: int) -> Any:
        p = self.pos + lag
        values = self.arrays[name]
        if not 0 <= p < values.shape[-1]:
            return np.nan
        return values[:, p] if values.ndim == 2 else values[p]

    def _gather(self, refs: Any) -> list[Any]:
        """Each slot's value: None for this period's unknowns (filled per call), the data otherwise."""
        return [None if lag == 0 and name in self.uidx else self._value(name, lag) for name, lag in refs]

    @staticmethod
    def _take(value: Any, rows: np.ndarray | None) -> Any:
        return value[rows] if rows is not None and isinstance(value, np.ndarray) and value.ndim == 1 else value

    def _slots(self, part: int, x: np.ndarray, rows: np.ndarray | None) -> list[Any]:
        refs = self.compiled[part].refs
        xs = x if rows is None else x[rows]
        return [xs[:, self.uidx[name]] if v is None else self._take(v, rows) for (name, _), v in zip(refs, self._fixed[part])]

    def _resolve(self, x: np.ndarray, rows: np.ndarray | None) -> Any:
        xs = x if rows is None else x[rows]

        def resolve(name: str, lag: int) -> Any:
            if lag == 0 and name in self.uidx:
                return xs[:, self.uidx[name]]
            return self._take(self._value(name, lag), rows)
        return resolve

    def _params(self, rows: np.ndarray | None) -> Any:
        if self.compiled is not None:
            return [self._take(v, rows) for v in self.compiled[2]]
        return {k: self._take(v, rows) for k, v in self.params.items()}

    def residuals(self, x: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        n = self.D if rows is None else len(rows)
        if self.compiled is not None:
            values = self.compiled[0](self._slots(0, x, rows), self._params(rows))
        else:
            resolve, params = self._resolve(x, rows), self._params(rows)
            values = [evaluate(r, resolve, params) for r in self.residual_exprs]
        return np.stack([np.broadcast_to(np.asarray(v, dtype=float), (n,)) for v in values], axis=1)

    def jacobian(self, x: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        n = self.D if rows is None else len(rows)
        J = np.zeros((n, len(self.residual_exprs), len(self.unknowns)))
        if self.compiled is not None:
            values = self.compiled[1](self._slots(1, x, rows), self._params(rows))
            if len(values):
                J[:, self.rows, self.cols] = np.stack([np.broadcast_to(np.asarray(v, dtype=float), (n,)) for v in values], axis=1)
            return J
        resolve, params = self._resolve(x, rows), self._params(rows)
        for i, j, d in self.jac:
            J[:, i, j] = np.broadcast_to(evaluate(d, resolve, params), (n,))
        return J


def _chord(batch: _Batch, residual: Any, x: np.ndarray, F: np.ndarray, live: np.ndarray, tol: float,
           iterations: int = 40) -> None:
    """Newton's method with one Jacobian for all the draws (their mean, factored once): each step is then a
    triangular solve, cheap for a large model. The first step may raise the residuals (the draws' own Jacobians differ
    from the mean); after it, a draw whose residuals stop shrinking is put back where it started, for Newton's method
    with its own Jacobian, as are those not settled after ``iterations`` steps (from where they got to). ``x`` and
    ``F`` are updated in place."""
    from scipy.linalg import lu_factor, lu_solve

    norm = np.max(np.abs(F), axis=1, initial=0.0)
    rows = live & np.isfinite(norm) & (norm >= tol)
    if rows.sum() < 2:
        return
    J = batch.jacobian(x, np.flatnonzero(rows)).mean(axis=0)
    if not np.all(np.isfinite(J)):
        return
    try:
        lu = lu_factor(J, check_finite=False)
    except (ValueError, np.linalg.LinAlgError):
        return
    if not np.all(np.isfinite(lu[0])) or np.min(np.abs(np.diag(lu[0]))) == 0.0:
        return
    x0, F0, norm0 = x.copy(), F.copy(), norm.copy()
    grew = np.zeros(len(x), dtype=int)
    for it in range(iterations):
        active = np.flatnonzero(rows & (norm >= tol))
        if not len(active):
            break
        x[active] -= lu_solve(lu, F[active].T, check_finite=False).T
        F[active] = residual(x, active)
        new = np.max(np.abs(F[active]), axis=1, initial=0.0)
        grew[active] = np.where(~(new < norm[active]), grew[active] + (it > 0), 0)
        stalled = active[~np.isfinite(new) | (grew[active] >= 2)]  # for Newton: from here, or from the start if worse
        norm[active] = new
        back = stalled[~(norm[stalled] < norm0[stalled])]
        x[back], F[back], norm[back] = x0[back], F0[back], norm0[back]
        rows[stalled] = False


class _GarchPath:
    """One equation's GARCH variance, draw by draw: started from its fit's last innovations and variances, carried over
    any periods between the estimation sample and the simulation with the innovations the data imply."""

    def __init__(self, fit: Any, residual: Any, arrays: dict[str, np.ndarray], full: pd.PeriodIndex, first: int):
        v = fit.params.as_dict()
        spec = fit.equation.variance
        self.omega = v["omega"]
        self.alpha = [v[f"alpha[{i}]"] for i in range(1, spec.arch + 1)]
        self.gamma = [v[f"gamma[{i}]"] for i in range(1, spec.asym + 1)]
        self.beta = [v[f"beta[{i}]"] for i in range(1, spec.garch + 1)]
        m = max(len(self.alpha), len(self.gamma), 1)
        e = fit.resid.to_numpy(float)
        h = fit.extra["volatility"].to_numpy(float) ** 2
        self.e = list(e[-m:])
        self.h = list(h[-max(len(self.beta), 1):])
        last = full.get_loc(fit.resid.index[-1]) if fit.resid.index[-1] in full else first - 1
        for pos in range(last + 1, first):  # the gap: innovations from the data, the variance carried forward
            def resolve(name: str, lag: int, pos: int = pos) -> float:
                p = pos + lag
                values = arrays.get(name)
                return float(values[p]) if values is not None and values.ndim == 1 and 0 <= p < len(values) else np.nan

            innovation = float(evaluate(residual, resolve))  # the equation's residual is its innovation
            if not np.isfinite(innovation):
                raise ModelError(f"{fit.equation.label}: the data before the simulation do not give its innovations, which "
                                 "its GARCH variance needs")
            self.h.append(self.variance())
            self.e.append(innovation)

    def variance(self) -> Any:
        h = self.omega + sum(a * self.e[-i] ** 2 for i, a in enumerate(self.alpha, 1))
        h = h + sum(g * self.e[-i] ** 2 * (np.asarray(self.e[-i]) < 0) for i, g in enumerate(self.gamma, 1))
        return h + sum(b * self.h[-j] for j, b in enumerate(self.beta, 1))

    def step(self, z: np.ndarray) -> np.ndarray:
        """This period's innovations, draw by draw: the standardised draws ``z`` scaled by each draw's variance."""
        h = self.variance()
        e = np.sqrt(h) * z
        self.e.append(e)
        self.h.append(h)
        return e


def _coefficient_draws(model: Any, draws: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Each fitted equation's coefficients, drawn from the normal with their estimated covariance."""
    out: dict[str, np.ndarray] = {}
    for label, fit in _fitted(model).items():
        names = list(fit.params.names)
        cov = np.asarray(fit.params.cov, dtype=float)
        mean = np.array([fit.params.as_dict()[n] for n in names])
        if not np.all(np.isfinite(cov)):
            raise ModelError(f"equation {label!r} has no coefficient covariance to draw from")
        values = rng.multivariate_normal(mean, cov, size=draws, method="eigh")
        out.update({f"{label}.{n}": values[:, j] for j, n in enumerate(names)})
    return out


def backward(model: Any, data: Any, span: Any, scenario: Any, solver: Any, base: Any, draws: int, seed: int | None,
             errors: Any, shock_sd: Mapping[str, float] | None, coefficients: bool,
             error_scale: Mapping[Any, float] | None = None, workers: int | None = None) -> Simulations:
    """Draws of a backward-looking model around ``base`` (its deterministic Solution); see the module docstring."""
    from .model import _prepare

    prep = _prepare(model, data, span, scenario)
    rng = _rng(seed)
    positions = np.flatnonzero(prep.rows)
    T, D = len(positions), int(draws)
    scales = _scales(error_scale, prep.full[positions])
    eqs = prep.eqs
    if isinstance(errors, pd.DataFrame):
        if shock_sd:
            raise SpecificationError("shock_sd rescales errors drawn from fitted equations: with a residual sample given, "
                                     "scale its columns before passing it")
        E, info, stochastic = _resampled(errors, [e.name for e in eqs], T, D, rng)
        garch: list[str] = []
    else:
        stochastic = [e.name for e in eqs if e.stochastic]
        E, info, garch = _errors(model, stochastic, T, D, errors, shock_sd, rng)
    if error_scale:
        info["error_scale"] = {str(p): float(v) for p, v in zip(prep.full[positions], scales) if v != 1.0}
    drawn: dict[str, np.ndarray] = {}
    if coefficients:
        drawn = _coefficient_draws(model, D, rng)
        info["coefficients"] = "drawn from their estimated covariance, each equation separately"
    E = E * scales[None, :, None]
    frame = base.frame
    chunks = [c for c in np.array_split(np.arange(D), min(int(workers or 1), D)) if len(c)]
    results = None
    if len(chunks) > 1 and not garch:  # the draws split between processes, each solving its own (the errors drawn here)
        from .._processes import run_in_processes

        jobs = [(_solve_draws, (model, data, span, scenario, solver, frame, E[c], stochastic, {k: v[c] for k, v in drawn.items()}))
                for c in chunks]
        results = run_in_processes(jobs, len(chunks), what="draws")
        if results is not None:
            values = np.concatenate([r[0] for r in results])
            failed = np.concatenate([r[1] for r in results])
            names = results[0][2]
            info["workers"] = len(chunks)
    if results is None:
        values, failed, names = _solve_draws(model, data, span, scenario, solver, frame, E, stochastic, drawn, garch)
    if failed.all():
        raise ModelError("no draw of the stochastic simulation could be solved: check the deterministic solution, and "
                         "the size of the errors (shock_sd)")
    info.update({"method": "backward-looking model: errors as add factors, each draw solved by Newton's method",
                 "draws": D, "seed": seed, "failed": int(failed.sum())})
    return Simulations(values, names, prep.full[positions], info, E, tuple(stochastic))


def _solve_draws(model: Any, data: Any, span: Any, scenario: Any, solver: Any, frame: pd.DataFrame, E: np.ndarray,
                 stochastic: list[str], drawn: Mapping[str, np.ndarray], garch: Sequence[str] = ()) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Each draw solved, period by period, with its errors ``E`` (draws x periods x the ``stochastic`` equations) as
    add factors and its coefficients (``drawn``), from the deterministic solution (``frame``): the draws' values
    (draws x periods x variables), which failed, and the variables' names. ``E`` is filled in for GARCH equations."""
    from ..threads import blas_threads
    from .model import _prepare

    prep = _prepare(model, data, span, scenario)
    positions = np.flatnonzero(prep.rows)
    T, D = len(positions), E.shape[0]
    system = model.compile()
    symbolic = {e.name: e.residual for e in system.equations}
    eqs = prep.eqs
    index = {n: k for k, n in enumerate(stochastic)}
    fitted = _fitted(model)
    substituted = {e.name: e.residual for e in model.model_equations}
    paths = {n: _GarchPath(fitted[n], substituted[n], prep.arrays, prep.full, int(positions[0])) for n in garch}
    params: dict[str, Any] = {**dict(system.parameters.values()), **drawn}
    layouts = prep.layouts or [(eqs, prep.unknowns, prep.adjust)]  # partial fixes and holds: the equations by period
    arrays: dict[str, np.ndarray] = dict(prep.arrays)
    for u in dict.fromkeys(u for _, own, _ in layouts for u in own):  # each draw its own path, from the deterministic solution
        start = frame[u].reindex(prep.full).to_numpy(float) if u in frame else arrays[u]
        arrays[u] = np.tile(start, (D, 1))
    batches: dict[int, _Batch] = {}
    failed = np.zeros(D, dtype=bool)
    tol, max_iter = float(getattr(solver, "tol", 1e-10)), max(int(getattr(solver, "max_iter", 50)), 50)
    with blas_threads():  # many small systems: BLAS's threads cost more than they save (pymetrix.threads)
        for t, pos in enumerate(positions):
            for n, path in paths.items():  # GARCH: this period's variance from each draw's own past innovations
                E[:, t, index[n]] = path.step(E[:, t, index[n]])
            k = 0 if prep.layout is None else int(prep.layout[pos])
            eqs, unknowns, adjust = layouts[k]
            if k not in batches:
                batches[k] = _Batch(model, eqs, unknowns, [symbolic[e.name] for e in eqs], arrays, params, D)
            batch = batches[k]
            batch.at(int(pos))
            x = np.nan_to_num(np.stack([arrays[u][:, pos] for u in unknowns], axis=1), nan=1.0)  # the deterministic solution
            add = np.stack([np.full(D, adjust[i][pos]) + (E[:, t, index[e.name]] if e.name in index else 0.0)
                            for i, e in enumerate(eqs)], axis=1)

            def residual(x: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
                return batch.residuals(x, rows) - (add if rows is None else add[rows])

            live = ~failed
            F = residual(x)
            _chord(batch, residual, x, F, live, tol)  # most draws settle here; the rest by Newton's method, each its own
            for _ in range(max_iter + 1):
                bad = ~np.all(np.isfinite(F), axis=1)
                failed |= bad & live
                live &= ~bad
                todo = np.flatnonzero(live & (np.max(np.abs(F), axis=1, initial=0.0) >= tol))
                if not len(todo):
                    break
                J = batch.jacobian(x, todo)
                Ft = F[todo]
                nonfinite = ~np.isfinite(J)
                for j in np.flatnonzero(nonfinite.any(axis=(0, 1))):  # an overflowing derivative: a forward difference
                    h = 1e-7 * np.maximum(1.0, np.abs(x[todo, j]))
                    shifted = x.copy()
                    shifted[todo, j] += h
                    column = (residual(shifted, todo) - Ft) / h[:, None]
                    J[:, :, j] = np.where(nonfinite[:, :, j], column, J[:, :, j])
                step = np.zeros((len(todo), x.shape[1]))
                ok = np.ones(len(todo), dtype=bool)
                try:
                    step = np.linalg.solve(J, -Ft[..., None])[..., 0]
                except np.linalg.LinAlgError:  # some draw's Jacobian is singular: solve them one by one, dropping those
                    for r in range(len(todo)):
                        try:
                            step[r] = np.linalg.solve(J[r], -Ft[r])
                        except np.linalg.LinAlgError:
                            ok[r] = False
                    failed[todo[~ok]], live[todo[~ok]] = True, False
                    todo, step = todo[ok], step[ok]
                before = x[todo].copy()
                x[todo] = before + step
                F[todo] = residual(x, todo)
                for _ in range(40):  # a step to where the equations are undefined (the log of a negative level) is halved
                    undefined = ~np.all(np.isfinite(F[todo]), axis=1)
                    if not undefined.any():
                        break
                    rows = todo[undefined]
                    step[undefined] *= 0.5
                    x[rows] = before[undefined] + step[undefined]
                    F[rows] = residual(x, rows)
            else:
                done = np.max(np.abs(F), axis=1) < tol
                failed |= live & ~done
            for j, u in enumerate(unknowns):
                arrays[u][:, pos] = np.where(failed, np.nan, x[:, j])
    names = tuple(dict.fromkeys([*model.endogenous, *scenario.free]))
    values = np.empty((D, T, len(names)))
    for j, n in enumerate(names):
        a = arrays[n]
        values[:, :, j] = a[:, positions] if a.ndim == 2 else np.broadcast_to(a[positions], (D, T))
    values[failed] = np.nan
    return values, failed, names


def observables(measurement: Any, solution: Any, sims: Simulations, data: Any = None) -> Simulations:
    """``sims`` with the observed series each draw implies, through ``measurement`` (evaluated exactly, every draw at
    once; lags reach into the solution's history), and the reported ones; with ``data`` and the measurement's
    transforms, the raw series rebuilt in each draw too."""
    frame = solution.frame
    rows = frame.index.get_indexer(sims.periods)
    D = sims.values.shape[0]
    columns: dict[str, np.ndarray] = {}

    def column(name: str) -> np.ndarray:
        if name not in columns:
            base = frame[name].to_numpy(float) if name in frame else np.full(len(frame), np.nan)
            values = np.tile(base, (D, 1))
            if name in sims.names:
                values[:, rows] = sims.values[:, :, sims.names.index(name)]
            columns[name] = values
        return columns[name]

    def resolve(name: str, lag: int) -> np.ndarray:
        values = column(name)
        out = np.full((D, len(rows)), np.nan)
        src = rows + lag
        ok = (src >= 0) & (src < values.shape[1])
        out[:, ok] = values[:, src[ok]]
        return out

    params = dict(measurement.params)
    equations = (*measurement.measurements, *getattr(measurement, "reports", ()))
    extra = [np.broadcast_to(np.asarray(evaluate(m.rhs, resolve, params), float), (D, len(rows))) for m in equations]
    names = [m.observable for m in equations]
    if measurement.trend_exprs:
        trend = measurement.trend_values(sims.periods)
        observed = [m.observable for m in measurement.measurements]
        extra = [e + trend[:, observed.index(n)] if n in measurement.trend_exprs else e for e, n in zip(extra, names)]
    if data is not None and getattr(measurement, "transform_exprs", None):
        observed = list(measurement.observables)
        _, raw = measurement._reverse(observed, sims.periods, np.stack([extra[names.index(o)] for o in observed], axis=2), data)
        rebuilt = [measurement.transform_exprs[o][1] for o in observed if o in measurement.transform_exprs]
        extra += [raw[:, :, j] for j in range(len(rebuilt))]
        names += rebuilt
    keep = [n for n in names if n not in sims.names]
    values = np.concatenate([sims.values, np.stack([extra[names.index(n)] for n in keep], axis=2)], axis=2) if keep else sims.values
    return Simulations(values, (*sims.names, *keep), sims.periods, sims.info)


def _occbin_draw(fit: Any, model: Any, solution: Any, system: Any, span: Any, scenario: Any, data: Any, shocks: bool,
                 seed: int, regimes: tuple[Any, ...] = (), extra: tuple[Any, ...] = ()) -> tuple[Any, Simulations]:
    """One posterior draw's OccBin forecast: the OccBin filter and smoother at the draw's parameters for the state,
    then OccBin's path from there (one path of shocks drawn, as surprises, with ``shocks``); its observed series.
    ``regimes``: the model's known-date regimes at the draw's values (filtered through); ``extra``: those after the
    sample, for the forecast."""
    from ..data import Dataset
    from .estimation import occbin_solution
    from .occbin_filter import occbin_filter

    method = fit.provenance.method if fit.provenance else {}
    history = fit.data if data is None else data
    names = list(fit.shock_sd)
    result = occbin_filter(model, history, system.measurement, dict(system.shock_sd), fit.span if data is None else None,
                           method.get("filter_start", "auto"), presample=int(method.get("presample", 0) or 0), shocks=names,
                           regimes=regimes or None, shock_corr=getattr(system, "shock_corr", None) or None)
    smoothed = Dataset.from_frame(result.smoothed.frame.drop(columns=names))
    path = occbin_solution(model, solution, system.measurement, dict(system.shock_sd), smoothed, span, scenario,
                           1 if shocks else None, seed, history, (*regimes, *extra), getattr(system, "shock_corr", None) or None)
    if shocks:
        if path.simulation is None or not len(path.simulation.values):
            raise ModelError("the draw's OccBin path does not settle")
        return path, path.simulation
    observed = list(solution.model.observed)
    periods = pd.period_range(*path.span, freq=path.frame.index.freq)
    sims = Simulations(path.frame.loc[periods, observed].to_numpy(float)[None], tuple(observed), periods)
    return path, observables(system.measurement, path, sims, history)


def posterior(post: Any, span: Any, scenario: Any, data: Any, tunes: Any, draws: int, shocks: bool, seed: int | None,
              occbin: bool | None = None, regimes: Any = None) -> Any:
    """Forecasts with parameter uncertainty: at ``draws`` posterior draws, spread evenly over the chains, the model is
    solved again, its variables filtered again from the data for their starting values, and (``shocks``) one path of
    its shocks drawn; the observed series through the measurement equations. The deterministic Solution at the
    posterior mode, with these draws.

    ``occbin`` (default: when the model was estimated with the OccBin filter, as ``FittedModel.forecast``): the
    model's occasionally binding constraints respected at every draw, its state from the OccBin filter and smoother
    at that draw's parameters and its path OccBin's forecast from there (``estimation.occbin_solution``), with the
    shocks drawn as surprises in each period.

    Known-date regimes (a model estimated with them): at each draw, the regimes take the draw's values
    (``<parameter>@<regime>``), the state is filtered through their time-varying rules (``regimes.smooth``), and the
    forecast follows them, the fit's regimes after the sample (``ahead``) and ``regimes`` after them."""
    from ..progress import report
    from .model import with_simulation
    from .plans import _solve_plan
    from .regimes import as_regimes, with_values

    fit = post.fit
    if occbin is None:
        occbin = str(fit.provenance.method.get("filter", "") if fit.provenance else "").startswith("OccBin")
    if occbin and tunes:
        raise SpecificationError("tunes are for the linear filter: forecast with occbin=False to use them")
    if tunes and data is None:
        raise SpecificationError("give the data to filter again with the tunes: forecast(span, data=..., tunes=...)")
    if int(draws) < 2:
        raise SpecificationError("draws is the number of posterior draws: at least 2")
    if shocks and scenario is not None and (scenario.hold or scenario.ranges):
        raise SpecificationError("targets with shocks drawn are not supported: forecast with shocks=False to hold targets "
                                 "at each parameter draw")
    own = tuple(getattr(fit, "regimes", ()))
    extra = (*getattr(fit, "ahead", ()), *as_regimes(regimes))
    if tunes and own:
        raise SpecificationError("tunes are not available with regimes")
    base = fit.forecast(span, scenario, data, tunes, occbin=bool(occbin), regimes=as_regimes(regimes) or None)  # at the posterior mode
    rng = _rng(seed)
    samples = post._subsample(int(draws))
    paths: list[np.ndarray] = []
    names: tuple[str, ...] = ()
    failed = 0
    for k, values in enumerate(samples):
        report(None, f"forecast at posterior draw {k + 1} of {len(samples)}")  # the app's progress bar; Cancel stops here
        try:
            model, solution, system = fit.evaluate(values)
            drawn = with_values(own, values) if own else ()  # the regimes at this draw's values
            if occbin:
                path, sims = _occbin_draw(fit, model, solution, system, span, scenario, data, shocks, int(rng.integers(2**63)),
                                          drawn, extra)
                names = names or sims.names
                paths.append(sims.values[0][:, [sims.names.index(n) for n in names]])
                continue
            if drawn:
                from .regimes import smooth

                history = fit.data if data is None else data
                smoothed = smooth(solution, system, drawn, history, fit.span if data is None else None, shocks=False).smoothed
                path, used, plan = _solve_plan(solution, smoothed, span, scenario, (*drawn, *extra))
            else:
                smoothed = system.filter(fit.data, fit.span, tunes=tunes) if data is None else system.filter(data, tunes=tunes)
                path, used, plan = _solve_plan(solution, smoothed, span, scenario, extra or None)
            if shocks:
                rules = None if plan is None else [(P, Q) for P, _, Q in plan.period_rules(len(plan.periods))]
                sims = first_order(used, path, 1, system.shock_sd, int(rng.integers(2**63)), rules=rules,
                                   shock_corr=getattr(system, "shock_corr", None))
            else:
                observed = list(solution.model.observed)
                periods = pd.period_range(*path.span, freq=path.frame.index.freq)
                sims = Simulations(path.frame.loc[periods, observed].to_numpy(float)[None], tuple(observed), periods)
            sims = observables(system.measurement, path, sims, fit.data if data is None else data)
        except (ModelError, ValueError, np.linalg.LinAlgError):  # a draw the model cannot be solved at
            failed += 1
            continue
        names = names or sims.names
        paths.append(sims.values[0][:, [sims.names.index(n) for n in names]])
    if len(paths) < 2:
        raise ModelError(f"the model could be solved at {len(paths)} of {len(samples)} posterior draws: too few to forecast")
    info = {"method": "posterior draws (spread evenly over the chains), each solved and filtered again"
                      + (" with OccBin (the constraints respected)" if occbin else "")
                      + (", with one path of shocks" if shocks else ", without shocks (parameter uncertainty alone)"),
            "draws": len(paths), "parameter draws": len(samples), "seed": seed, "failed": failed, "shocks": bool(shocks),
            "shock_sd": "at each draw" if shocks else {}}
    first, last = base.span
    return with_simulation(base, Simulations(np.stack(paths), names, pd.period_range(first, last, freq=base.frame.index.freq), info))
