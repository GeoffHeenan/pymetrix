"""State-space models: written as equations, filtered and smoothed with the Kalman filter.

A model is written the way economists write it, with declared parameters::

    uc = StateSpace(
        states=["trend", "slope", "cycle"],
        transition=[
            "trend = trend[-1] + slope[-1] + e_trend",
            "slope = slope[-1] + e_slope",
            "cycle = phi1*cycle[-1] + phi2*cycle[-2] + e_cycle",
        ],
        measurement=["y = trend + cycle + u_y"],
        shocks={"e_trend": "sd_trend", "e_slope": "sd_slope", "e_cycle": "sd_cycle", "u_y": "sd_y"},
        params=["phi1", "phi2", "sd_trend", "sd_slope", "sd_cycle", "sd_y"],
    )
    fit = uc.fit(data)                 # maximum likelihood (posterior mode with priors)
    fit.smoothed["cycle"]              # two-sided estimates of the states
    fit.filtered["trend"]              # one-sided (real-time) estimates

Transition equations give each state in terms of lagged states and shocks;
lags beyond one are handled by adding lagged copies of the state. Measurement
equations link observed series (or expressions of them) to current or lagged
states, plus optional measurement errors. Each shock has a standard deviation:
a parameter (estimated, kept positive) or a number. Parameters are declared;
``Param(..., bounds=, prior=)`` sets bounds and priors.

The system matrices follow Durbin and Koopman: ``y[t] = d + Z a[t] + eps[t]``,
``a[t+1] = c + T a[t] + R eta[t]``, with ``eps ~ N(0, H)`` and ``eta ~ N(0, Q)``.
Initial states are stationary when every root of ``T`` is inside the unit
circle, and diffuse (unknown) otherwise.

The log likelihood with diffuse states is the exact diffuse likelihood of Durbin and Koopman (2012, sections 5.2
and 7.2.2): with ``d`` diffuse periods (until every diffuse state is pinned down by the data), ``p_t`` series observed
in period ``t``, innovations ``v_t`` with variance ``F_t`` after the diffuse periods, and in a diffuse period the
variance's diffuse part ``F_inf,t`` (one observation at a time; ``F_star,t`` and ``v_t`` where ``F_inf,t`` is zero)::

    log L = - 1/2 sum_{t <= d} [ p_t log 2 pi + sum_i w_{t,i} ]                            (the diffuse periods)
            - 1/2 sum_{t > d}  [ p_t log 2 pi + log |F_t| + v_t' F_t^-1 v_t ]
    w_{t,i} = log F_inf,t,i                     where F_inf,t,i > 0
            = log F_star,t,i + v^2 / F_star,t,i  where F_inf,t,i = 0

This is the default (``diffuse_terms="include"``), as statsmodels' exact diffuse filter computes it
(``initialize_diffuse``, ``use_exact_diffuse=True``). ``diffuse_terms="exclude"`` leaves the diffuse periods' terms out, the likelihood of the later
observations given the first ``d`` (Harvey 1989, section 3.4.3): what statsmodels' ``UnobservedComponents`` and
``SARIMAX`` report by default, whose approximately diffuse start (a large initial variance) drops the first ``d``
periods (``loglikelihood_burn``). The difference is the diffuse periods' terms; where the diffuse part of the system
(``Z`` and ``T`` on the diffuse states) has no parameters, as in the unobserved-components models, it is a constant,
``d/2 log 2 pi + 1/2 sum log F_inf`` for one series, and the estimates and standard errors are the same either way:
only the level of the log likelihood and of the information criteria moves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
from scipy import linalg, optimize

from ._api import api
from .errors import NotFoundError, SpecificationError
from ._capture import data_fingerprint, records_warnings, with_effective_sample
from ._version import __version__
from .expr import Coef, Expr, ExpressionError, Name, Num, diff, evaluate, parse_expression
from .forms import Widget
from .metadata import Param, Provenance
from .output import Table
from .plugins import register
from ._random import generator as make_rng

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["FittedStateSpace", "KalmanOutput", "KalmanRun", "StateSpace", "StateSpaceSystem", "SystemMatrices", "UnobservedComponents",
           "kalman_filter", "kalman_smoother"]

KAPPA = 1e6  # variance of approximately diffuse initial states, per squared unit of the data (see SystemMatrices.initial)
SCALED_VARIANCE = 10.0  # the "scaled" start: every state at zero with this variance (Dynare's lik_init=2, Harvey_scale_factor)
STARTS = ("auto", "stationary", "diffuse", "scaled")
DIFFUSE_TERMS = ("include", "exclude")  # the diffuse periods' log likelihood terms: in (Durbin and Koopman) or out (Harvey; statsmodels' default)
_LOG2PI = np.log(2 * np.pi)


# ---------------------------------------------------------------------------------------------
# the numbers: matrices, filter and smoother


@dataclass(frozen=True, eq=False)
class SystemMatrices:
    """``y[t] = d + Z a[t] + eps[t]``, ``a[t+1] = c + T a[t] + R eta[t]``; ``eps ~ N(0, H)``, ``eta ~ N(0, Q)``."""

    states: tuple[str, ...]
    observables: tuple[str, ...]
    shocks: tuple[str, ...]
    T: np.ndarray
    c: np.ndarray
    R: np.ndarray
    Q: np.ndarray
    Z: np.ndarray
    d: np.ndarray
    H: np.ndarray
    stationary: tuple[bool, ...] = ()  # states declared stationary (with their lags); empty = none declared

    def start(self, mode: str = "auto") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The initial state: mean, variance of its known part, and which states are diffuse (unknown).

        ``auto``: declared-stationary states start from their stationary distribution and the others are
        diffuse; with none declared, all states are stationary if every root of ``T`` is inside the unit
        circle, and all are diffuse otherwise. ``stationary`` and ``diffuse`` force one or the other; ``scaled`` starts
        every state at zero with variance ``SCALED_VARIANCE`` (Dynare's ``lik_init=2``)."""
        k = len(self.states)
        if mode == "scaled":
            return np.zeros(k), SCALED_VARIANCE * np.eye(k), np.zeros(k, bool)
        if mode not in STARTS:
            raise SpecificationError(f"the filter's start is one of {', '.join(STARTS)}, not {mode!r}")
        if mode == "stationary":
            S = np.ones(k, bool)
        elif mode == "diffuse":
            S = np.zeros(k, bool)
        elif any(self.stationary):
            S = np.array(self.stationary, bool)
        else:
            roots = np.abs(np.linalg.eigvals(self.T)) if k else np.array([])
            S = np.full(k, bool(np.all(roots < 1 - 1e-8)))
        D = ~S
        a0, P = np.zeros(k), np.zeros((k, k))
        if S.any():
            if np.any(self.T[np.ix_(S, D)]):
                raise SpecificationError("stationary states cannot depend on non-stationary ones")
            T_SS = self.T[np.ix_(S, S)]
            if np.any(np.abs(np.linalg.eigvals(T_SS)) >= 1 - 1e-8):
                raise SpecificationError("the stationary states are not stationary at these parameter values")
            a0[S] = np.linalg.solve(np.eye(int(S.sum())) - T_SS, self.c[S])
            RQR = (self.R @ self.Q @ self.R.T)[np.ix_(S, S)]
            block = linalg.solve_discrete_lyapunov(T_SS, RQR)
            P[np.ix_(S, S)] = (block + block.T) / 2
        return a0, P, D

    def initial(self, mode: str = "auto", scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, int]:
        """Initial mean and variance with diffuse states approximated by a large variance, and how
        many observations to leave out of the likelihood (the approximate start of the native filter).

        Diffuse states start at zero with variance ``KAPPA * scale**2``; ``scale`` is the size of the
        data (its largest absolute value), so the start is equally uninformative for any units."""
        a0, P, D = self.start(mode)
        kappa = KAPPA * max(1.0, float(scale) ** 2)
        return a0, P + kappa * np.diag(D.astype(float)), int(D.sum())


@dataclass(frozen=True, eq=False)
class StateSpaceSystem(SystemMatrices):
    """A solved model in state-space form, as a result: its matrices with names, the measurement equations
    linking it to data, and the shock sizes. ``filter`` estimates the model's variables from data (the Kalman
    smoother), in levels; ``loglike`` is the likelihood of the data, the basis of estimation.

    Made by ``LinearSolution.system(measurement, shock_sd)``. States are the model's variables in deviations from
    the steady state, with lagged copies where the measurement equations use lags (``y[-1]``)."""

    measurement: Any = None  # the MeasurementMap
    shock_sd: Mapping[str, float] = field(default_factory=dict)
    shock_corr: Mapping[tuple[str, str], float] = field(default_factory=dict)  # correlations of the shocks, if any (off Q's diagonal)
    levels: Mapping[str, float] = field(default_factory=dict)  # steady-state values, added back to report levels
    variables: tuple[str, ...] = ()  # the model's own variables (not lagged copies or auxiliaries)
    trend: np.ndarray | None = None  # growth per period of each state's variable, for a model on a growth path
    trend_lags: np.ndarray | None = None  # the lag of each state (0, -1, ...): its trend is growth * (t + lag)

    def _offset(self, n: int, Z: np.ndarray | None = None) -> np.ndarray:
        """What the growth path adds to each observation (n x observables), t = 0 at the first one: the measurement
        equations applied to the trend part of each state's level."""
        Z = self.Z if Z is None else Z
        if self.trend is None or not np.any(self.trend):
            return np.zeros((n, Z.shape[0]))
        t = np.arange(n, dtype=float)[:, None]
        return (self.trend[None, :] * (t + self.trend_lags[None, :])) @ Z.T

    def _dtrend(self, index: Any, observables: int) -> np.ndarray:
        """The deterministic trends of the measurement equations in these periods (periods x observables, zero for
        observables without one, and for tunes)."""
        out = np.zeros((len(index), observables))
        trends = getattr(self.measurement, "trend_exprs", None)
        if trends:
            values = self.measurement.trend_values(index)
            out[:, :values.shape[1]] = values
        return out

    def _trend_of(self, n: int) -> np.ndarray:
        """The growth path of the model's own variables (n x variables), added to the levels reported."""
        if self.trend is None:
            return np.zeros((n, len(self.variables)))
        return np.arange(n, dtype=float)[:, None] * self.trend[None, :len(self.variables)]

    def observations(self, data: Dataset, span: Any = None) -> tuple[np.ndarray, pd.Index]:
        """The observed series as a matrix, in the order of the measurement equations."""
        if self.measurement is not None:  # made from the data by the measurement map: renamed, or transformed from raw series
            made = self.measurement.observed(data, span)
            return np.column_stack([made[o].to_numpy(float) for o in self.observables]), made.index
        frame = data.frame[data.sample_mask(span)]
        names = list(self.observables)
        missing = [n for n in names if n not in frame.columns]
        if missing:
            raise SpecificationError(f"the data have no {', '.join(missing)} (observed series of the measurement equations)")
        return np.column_stack([frame[n].to_numpy(float) for n in names]), frame.index

    @api("pymetrix:state_space_system.filter")
    def filter(self, data: Dataset, span: Any = None, tunes: Mapping[str, Any] | None = None,
               tune_sd: Mapping[str, Any] | None = None, shocks: bool = False) -> Dataset:
        """The model's variables estimated from the data by the Kalman smoother, in levels, with each observed
        series as the model fits it (``<series>_fitted``).

        ``tunes`` imposes judgement on model variables, observed or not: ``{"y_gap": {"2024Q4": -1.0}}`` (or a
        Series), each value treated as an exact observation of that variable in that period, so the filter moves
        everything else to be consistent with it (as IRIS's filter tunes do). A tune can be soft: give it a standard
        deviation, ``{"y_gap": {"2024Q4": (-1.0, 0.25)}}`` (or a list ``[-1.0, 0.25]``) or ``tune_sd={"y_gap": {"2024Q4": 0.25}}`` (or one number
        for all of a variable's tunes); it is then an observation with that error, in its own period only. A standard
        deviation of None or 0 is exact, as before.

        ``shocks=True`` adds the smoothed shocks e(t) of ``x(t) = P x(t-1) + Q e(t)`` as columns named after the
        shocks, in their own units, as a solution's paths carry them. They come from the smoothed state
        disturbances, so the shocks need not be states; e(0) is E[e(0) | y] through the states that start from their
        stationary distribution."""
        from .data import Dataset as _Dataset

        y, index = self.observations(data, span)
        system = self
        if tunes:
            system, y, _ = self._tuned(tunes, y, index, tune_sd)
        y = y - system._dtrend(index, y.shape[1])  # deterministic trends of the measurement equations
        out = run(y - system._offset(len(y), system.Z), system, "auto")  # stationary states from their distribution; unit roots diffuse
        k = len(self.variables)
        frame = pd.DataFrame(out.a_smooth[:, :k] + np.array([self.levels.get(v, 0.0) for v in self.variables]) + self._trend_of(len(y)),
                             index=index, columns=list(self.variables))
        fitted = self.d + out.a_smooth @ self.Z.T + self._offset(len(y)) + self._dtrend(index, len(self.observables))
        for j, o in enumerate(self.observables):
            frame[f"{o}_fitted"] = fitted[:, j]
        if self.measurement is not None:
            frame = self.measurement.with_reported(frame)
        if shocks:
            taken = [s for s in self.shocks if s in frame.columns]
            if taken:
                raise SpecificationError(f"shocks {', '.join(taken)} have the names of other columns of the filtered data")
            e = self._shocks(system, out)
            for j, s in enumerate(self.shocks):
                frame[s] = e[:, j]
        return _Dataset.from_frame(frame)

    def _shocks(self, system: SystemMatrices, out: KalmanRun) -> np.ndarray:
        """The smoothed shocks e(t) by period: e(t) = eta(t-1) for t >= 1 (the disturbance moving the state into t; when R
        has full column rank, taken from the smoothed states, R e(t) = a(t|n) - c - T a(t-1|n), which is the same
        estimate without the engines' own rounding); e(0)
        from the start, ``E[e(0) | y] = Q R_S' P_SS^-1 (a(0|n) - a0)_S`` over the states that start from their stationary
        distribution (the shock is independent of what came before; diffuse states say nothing about it)."""
        eta = out.eta_smooth
        n = len(out.a_smooth)
        e = np.zeros((n, len(self.shocks)))
        if n > 1:
            R = system.R
            if len(self.shocks) and np.linalg.matrix_rank(R) == R.shape[1]:  # each shock seen in the states: exactly
                step = out.a_smooth[1:] - system.c - out.a_smooth[:-1] @ system.T.T  # = R e(t), so the identity holds
                e[1:] = np.linalg.lstsq(R, step.T, rcond=None)[0].T             # to rounding error on either engine
            else:
                e[1:] = eta[:-1]
        a0, P0, D = system.start("auto")
        S = ~D
        if S.any() and len(self.shocks):
            gap = (out.a_smooth[0] - a0)[S]
            e[0] = system.Q @ system.R[S].T @ np.linalg.lstsq(P0[np.ix_(S, S)], gap, rcond=None)[0]
        return e

    def contributions(self, data: Dataset, span: Any = None, tunes: Mapping[str, Any] | None = None, horizon: int = 0,
                      tune_sd: Mapping[str, Any] | None = None) -> Any:
        """The Kalman-smoothed estimate of each model variable (deviation from its steady state) split into what
        each observed series contributed, each tune if any, and the starting point ("start"); with ``horizon``,
        the forecast from the end of the data too. A ``Decomposition`` titled "Data contributions".

        Given which values are missing, the smoother is linear in the data: a series' contribution is the smoother
        run with that series as observed and every other one at the value the model expects with no news (its
        steady state, on the growth path), less the smoother with all of them there. The parts add up to the
        smoothed path exactly, period by period; forecasts carry each part forward by the transition matrix."""
        from .models.decompose import Decomposition

        y, index = self.observations(data, span)
        system = self
        if tunes:
            system, y, _ = self._tuned(tunes, y, index, tune_sd)
        y = y - system._offset(len(y), system.Z) - system._dtrend(index, y.shape[1])
        seen = ~np.isnan(y)
        base = np.where(seen, np.broadcast_to(system.d, y.shape), np.nan)
        start = run(base, system, "auto").a_smooth
        names: list[str] = []
        parts: list[np.ndarray] = []
        for j, name in enumerate(system.observables):  # soft tunes are one row per value: their parts add up by variable
            only = base.copy()
            only[:, j] = y[:, j]
            part = run(only, system, "auto").a_smooth - start
            if name in names:
                parts[names.index(name)] = parts[names.index(name)] + part
            else:
                names.append(name)
                parts.append(part)
        if horizon:
            Tm, c = system.T, np.asarray(system.c, dtype=float)

            def carry(path: np.ndarray, constant: bool) -> np.ndarray:
                rows, state = [], path[-1]
                for _ in range(horizon):
                    state = Tm @ state + (c if constant else 0.0)
                    rows.append(state)
                return np.vstack([path, rows])

            parts = [carry(p, False) for p in parts]
            start = carry(start, True)
            if isinstance(index, pd.PeriodIndex):
                index = index.append(pd.period_range(index[-1] + 1, periods=horizon, freq=index.freq))
            else:
                index = pd.RangeIndex(len(index) + horizon)
        k = len(self.variables)
        contributions = {v: pd.DataFrame({**{n: p[:, i] for n, p in zip(names, parts)}, "start": start[:, i]}, index=index)
                         for i, v in enumerate(self.variables)}
        deviations = pd.DataFrame({v: contributions[v].sum(axis=1) for v in self.variables[:k]}, index=index)
        return Decomposition(contributions, deviations, pd.DataFrame(index=index), "Data contributions")

    def _tuned(self, tunes: Mapping[str, Any], y: np.ndarray, index: pd.Index,
               tune_sd: Mapping[str, Any] | None = None) -> tuple[StateSpaceSystem, np.ndarray, list[str]]:
        """The system with each tuned variable added as observations, and the data with the tunes; and the names.

        A variable whose tunes are all exact is one observation without error, as always. One with a soft tune (a
        standard deviation above zero, in a ``(value, sd)`` pair or in ``tune_sd``) is one observation per tuned
        value, seen only in its own period, with variance ``sd ** 2`` (zero for the exact ones among them)."""
        from dataclasses import replace

        unknown = [v for v in [*tunes, *(tune_sd or {})] if v not in self.variables]
        if unknown:
            raise SpecificationError(f"tunes: {', '.join(unknown)} are not variables of the model ({', '.join(self.variables)})")
        stray = [v for v in (tune_sd or {}) if v not in tunes]
        if stray:
            raise SpecificationError(f"tune_sd: {', '.join(stray)} have no tunes")

        def periods(values: Any) -> list[Any]:
            return [pd.Period(str(p), freq=index.freq) if isinstance(index, pd.PeriodIndex) else p for p in values]

        rows, columns, names, variances = [], [], [], []
        for v, values in tunes.items():
            pairs = list(values.items()) if isinstance(values, (pd.Series, Mapping)) else list(dict(values).items())
            keys = periods([k for k, _ in pairs])
            outside = [str(p) for p in keys if p not in index]
            if outside:
                raise SpecificationError(f"tunes for {v}: {', '.join(outside[:3])} are outside the data's periods")
            pair = [isinstance(x, (tuple, list)) for _, x in pairs]  # (value, sd), or [value, sd] as a saved call has it
            if any(isinstance(x, (tuple, list)) and len(x) != 2 for _, x in pairs):
                raise SpecificationError(f"tunes for {v}: give a value, or a (value, standard deviation) pair")
            level = [x[0] if is_pair else x for is_pair, (_, x) in zip(pair, pairs)]
            sd = [x[1] if is_pair else None for is_pair, (_, x) in zip(pair, pairs)]
            given = (tune_sd or {}).get(v)
            if given is not None:  # one number for all of v's tunes, or by period
                if isinstance(given, (Mapping, pd.Series)):
                    by_period = dict(zip(periods(list(given.keys())), list(given.values())))
                    extra = [str(p) for p in by_period if p not in keys]
                    if extra:
                        raise SpecificationError(f"tune_sd for {v}: {', '.join(extra[:3])} have no tune")
                    sd = [by_period.get(p, s) for p, s in zip(keys, sd)]
                else:
                    sd = [given] * len(keys)
            sd = [0.0 if s is None else float(s) for s in sd]
            if any(s < 0 or not np.isfinite(s) for s in sd):
                raise SpecificationError(f"tune standard deviations for {v} must be zero or positive")
            row = np.zeros(len(self.states))
            row[list(self.states).index(v)] = 1.0
            values_at = pd.Series([float(x) for x in level], index=keys).reindex(index).to_numpy(float) - self.levels.get(v, 0.0)
            if not any(sd):  # every tune exact: one observation without error, as before
                rows.append(row)
                columns.append(values_at)
                names.append(f"{v} (tune)")
                variances.append(0.0)
                continue
            for p, s in zip(keys, sd):  # soft: each value its own observation, seen in its period only
                column = np.full(len(index), np.nan)
                at = index.get_loc(p)
                column[at] = values_at[at]
                rows.append(row)
                columns.append(column)
                names.append(f"{v} (tune)")
                variances.append(s * s)
        Z = np.vstack([self.Z, *rows])
        d = np.concatenate([self.d, np.zeros(len(rows))])
        H = linalg.block_diag(self.H, np.diag(variances))
        tuned = replace(self, observables=(*self.observables, *names), Z=Z, d=d, H=H)
        return tuned, np.column_stack([y, *columns]), names

    def loglike(self, data: Dataset, span: Any = None, diffuse_terms: str = "include") -> float:
        """The log likelihood of the observed series under the model (the Kalman filter). With diffuse states,
        ``diffuse_terms`` includes the diffuse periods' terms (the exact diffuse likelihood, the default) or leaves
        them out (``"exclude"``, statsmodels' default convention): see ``pymetrix.statespace``."""
        return self.loglike_observed(*self.observations(data, span), diffuse_terms=diffuse_terms)

    def loglike_observed(self, y: np.ndarray, index: Any, start: str = "auto", presample: int = 0,
                         diffuse_terms: str = "include") -> float:
        """The log likelihood of observations already taken from the data (``observations``): estimation takes them
        once and evaluates this many times. ``start`` is the filter's start (see ``SystemMatrices.start``);
        ``presample`` periods are filtered but left out of the likelihood (Dynare's ``presample``); ``diffuse_terms``
        as in ``loglike``."""
        return loglike(y - self._offset(len(y)) - self._dtrend(index, y.shape[1]), self.needed(start), start, presample,
                       diffuse_terms)

    def needed(self, start: str = "auto") -> SystemMatrices:
        """The system on the states the likelihood needs: those observed and those they depend on, as Dynare's
        ``kalman_algo`` restricts the state to the predetermined and observed variables. Exact for a stationary system
        (the others never reach an observation, and the stationary distribution of the rest is their own); a system with
        diffuse states is returned whole, since how many states start diffuse changes the likelihood."""
        if start != "scaled" and not (self.stationary and all(self.stationary)):  # scaled: every state starts alike
            return self
        need = np.any(self.Z != 0, axis=0)
        while True:
            more = need | np.any(self.T[need] != 0, axis=0)
            if np.array_equal(more, need):
                break
            need = more
        if need.all():
            return self
        keep = np.flatnonzero(need)
        states = tuple(self.states[i] for i in keep)
        R, Q, shocks = self.R[keep], self.Q, self.shocks
        if R.shape[1] > len(keep):  # more shocks than states left: their combined variance, one per state
            R, Q, shocks = np.eye(len(keep)), R @ Q @ R.T, states
        return SystemMatrices(states, self.observables, shocks, self.T[np.ix_(keep, keep)], self.c[keep], R, Q,
                              self.Z[:, keep], self.d, self.H, tuple(True for _ in keep))

    def _matrix(self, M: np.ndarray, rows: Sequence[str], cols: Sequence[str], title: str) -> Table:
        return Table(list(cols), [list(map(float, r)) for r in M], index=list(rows), title=title)

    def transition(self) -> Table:
        return self._matrix(self.T, self.states, self.states, "Transition: state(t) on state(t-1)")

    def shock_impact(self) -> Table:
        return self._matrix(self.R, self.states, self.shocks, "Shocks: state(t) on shocks(t)")

    def measurement_matrix(self) -> Table:
        rows = [[float(self.d[i]), *map(float, self.Z[i])] for i in range(len(self.observables))]
        return Table(["constant", *self.states], rows, index=list(self.observables),
                     title="Measurement: observed = constant + loadings on the states (linearised at the steady state)")

    def covariances(self) -> Table:
        rows = [[s, "shock", float(np.sqrt(self.Q[i, i]))] for i, s in enumerate(self.shocks)]
        rows += [[o, "measurement error", float(np.sqrt(self.H[i, i]))] for i, o in enumerate(self.observables)]
        notes = [f"Correlations: {', '.join(f'{a}-{b} {r:g}' for (a, b), r in self.shock_corr.items())}"] \
            if getattr(self, "shock_corr", None) else []  # (a system saved before correlations had none)
        return Table(["", "kind", "standard deviation"], rows, title="Shocks and measurement errors", notes=notes)

    def summary(self) -> Table:
        return self.measurement.table() if self.measurement is not None else self.measurement_matrix()

    def __repr__(self) -> str:
        return (f"<StateSpaceSystem: {len(self.states)} states, {len(self.observables)} observed ({', '.join(self.observables)}), "
                f"{len(self.shocks)} shocks>")

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


@dataclass(frozen=True, eq=False)
class KalmanOutput:
    """Filter output by period: predicted (``a[t] | y[..t-1]``) and filtered (``a[t] | y[..t]``) states,
    innovations ``v`` and their variances ``F`` (NaN where unobserved), and log-likelihood terms."""

    a_pred: np.ndarray
    P_pred: np.ndarray
    a_filt: np.ndarray
    P_filt: np.ndarray
    v: np.ndarray
    F: np.ndarray
    loglike_t: np.ndarray
    burn: int

    @property
    def loglike(self) -> float:
        return float(np.sum(self.loglike_t[self.burn:]))


_RICCATI_TOL = 1e-12  # relative change in the predicted state variance below which it has converged


def kalman_filter(y: np.ndarray, m: SystemMatrices, a0: np.ndarray, P0: np.ndarray, burn: int = 0) -> KalmanOutput:
    """The Kalman filter. Rows of ``y`` are periods; NaN entries are unobserved (handled exactly).

    Once the predicted state variance stops changing (to ``_RICCATI_TOL``, relative) in a period with every series
    observed, the gain and innovation variance are constant: later fully observed periods update only the mean, as
    Dynare does with ``riccati_tol``. A period with a missing observation goes back to the full recursion."""
    y = np.atleast_2d(np.asarray(y, dtype=float))
    n, p = y.shape
    k = len(a0)
    a_pred, a_filt = np.empty((n, k)), np.empty((n, k))
    P_pred, P_filt = np.empty((n, k, k)), np.empty((n, k, k))
    v, F = np.full((n, p), np.nan), np.full((n, p, p), np.nan)
    ll = np.zeros(n)
    RQR = m.R @ m.Q @ m.R.T
    a, P = np.asarray(a0, float), np.asarray(P0, float)
    steady: tuple[Any, ...] | None = None  # (chol F, chol F^-1 Z P, log det F, F, filtered P) once the variance has converged
    from .progress import check

    for t in range(n):
        if t % 50 == 49:
            check()  # Cancel in the app stops a long filter here (it costs nothing outside the app)
        a_pred[t], P_pred[t] = a, P
        seen = ~np.isnan(y[t])
        if steady is not None and seen.all():
            L, W, logdet, Ft, Pf = steady
            vt = y[t] - m.d - m.Z @ a
            w = np.linalg.solve(L, vt)
            a = a + W.T @ w
            ll[t] = -0.5 * (p * _LOG2PI + logdet + w @ w)
            v[t], F[t] = vt, Ft
            a_filt[t], P_filt[t] = a, Pf
            a = m.c + m.T @ a
            continue
        steady = None
        if p == 1 and seen[0]:  # one observed series: scalar algebra (the common case, and much faster)
            z = m.Z[0]
            vt = y[t, 0] - m.d[0] - z @ a
            Pz = P @ z
            Ft = z @ Pz + m.H[0, 0]
            if not Ft > 0:
                raise SpecificationError(f"the innovation variance is not positive in period {t}")
            a = a + Pz * (vt / Ft)
            P = P - np.outer(Pz, Pz) / Ft
            ll[t] = -0.5 * (_LOG2PI + np.log(Ft) + vt * vt / Ft)
            v[t, 0], F[t, 0, 0] = vt, Ft
            L, W, logdet = np.array([[np.sqrt(Ft)]]), Pz[None, :] / np.sqrt(Ft), np.log(Ft)
        elif seen.any():
            every = seen.all()  # NumPy throughout (no SciPy wrappers): this runs once a period, many times in estimation
            Zt, dt, Ht = (m.Z, m.d, m.H) if every else (m.Z[seen], m.d[seen], m.H[np.ix_(seen, seen)])
            vt = (y[t] if every else y[t, seen]) - dt - Zt @ a
            PZ = P @ Zt.T
            Ft = Zt @ PZ + Ht
            Ft = (Ft + Ft.T) / 2
            try:
                L = np.linalg.cholesky(Ft)
            except np.linalg.LinAlgError:
                raise SpecificationError(f"the innovation variance is not positive definite in period {t}") from None
            # solves with the Cholesky factor, never F^-1 itself: F is badly conditioned while diffuse states settle
            X = np.linalg.solve(L, np.column_stack((vt, PZ.T)))
            w, W = X[:, 0], X[:, 1:]  # L^-1 v and L^-1 Z P
            a = a + W.T @ w
            P = P - W.T @ W
            P = (P + P.T) / 2
            logdet = 2 * np.sum(np.log(np.diag(L)))
            ll[t] = -0.5 * (len(vt) * _LOG2PI + logdet + w @ w)
            if every:
                v[t], F[t] = vt, Ft
            else:
                v[t, seen] = vt
                F[t][np.ix_(seen, seen)] = Ft
        a_filt[t], P_filt[t] = a, P
        a = m.c + m.T @ a
        P_next = m.T @ P @ m.T.T + RQR
        if seen.all() and np.max(np.abs(P_next - P_pred[t])) <= _RICCATI_TOL * max(1.0, float(np.max(np.abs(P_next)))):
            steady = (L, W, logdet, F[t].copy(), P.copy())
        P = P_next
    return KalmanOutput(a_pred, P_pred, a_filt, P_filt, v, F, ll, burn)


def check_diffuse_terms(diffuse_terms: str) -> str:
    """``diffuse_terms`` checked: ``include`` or ``exclude`` (see the module docstring)."""
    if diffuse_terms not in DIFFUSE_TERMS:
        raise SpecificationError(f"diffuse_terms is 'include' (Durbin and Koopman's exact diffuse likelihood) or 'exclude' "
                                 f"(without the diffuse periods' terms, statsmodels' default), not {diffuse_terms!r}")
    return diffuse_terms


def diffuse_loglike(y: np.ndarray, m: SystemMatrices, mode: str = "auto", diffuse_terms: str = "include") -> tuple[float, int]:
    """Exact diffuse log likelihood (Durbin and Koopman 2012, section 7.2.2): diffuse states' starts are unknown.

    Observations are processed one at a time (which needs uncorrelated measurement errors).
    While some state is still unidentified, an observation adds ``-(log 2 pi + log F_inf) / 2``;
    afterwards the usual Kalman filter takes over. Returns the log likelihood and the number
    of diffuse periods. Unlike a large finite initial variance, this is smooth in the parameters,
    so numerical derivatives (and so estimation) are reliable. ``diffuse_terms="exclude"`` leaves out
    every term of the diffuse periods (see the module docstring).
    """
    ll, diffuse, first = _diffuse_parts(y, m, mode)
    return (ll if check_diffuse_terms(diffuse_terms) == "include" else ll - first), diffuse


def _diffuse_parts(y: np.ndarray, m: SystemMatrices, mode: str = "auto") -> tuple[float, int, float]:
    """The exact diffuse log likelihood, the number of diffuse periods and the sum of those periods' terms."""
    if np.count_nonzero(m.H - np.diag(np.diag(m.H))):
        raise SpecificationError("exact diffuse filtering needs uncorrelated measurement errors")
    y = np.atleast_2d(np.asarray(y, dtype=float))
    n = len(y)
    k = len(m.T)
    a, P_star, D = m.start(mode)
    P_inf = np.diag(D.astype(float))
    RQR = m.R @ m.Q @ m.R.T
    ll = 0.0
    tol = 1e-8
    for t in range(n):
        if np.max(np.abs(P_inf)) <= tol:  # every state identified: the usual filter from here
            rest = kalman_filter(y[t:], m, a, P_star, 0)
            return ll + rest.loglike, t, ll
        for i in np.flatnonzero(~np.isnan(y[t])):
            z = m.Z[i]
            v = y[t, i] - m.d[i] - z @ a
            M_inf, M_star = P_inf @ z, P_star @ z
            F_inf, F_star = z @ M_inf, z @ M_star + m.H[i, i]
            if F_inf > tol:
                a = a + M_inf * (v / F_inf)
                P_star = P_star + np.outer(M_inf, M_inf) * (F_star / F_inf**2) - (np.outer(M_star, M_inf) + np.outer(M_inf, M_star)) / F_inf
                P_inf = P_inf - np.outer(M_inf, M_inf) / F_inf
                ll += -0.5 * (_LOG2PI + np.log(F_inf))
            elif F_star > 0:
                a = a + M_star * (v / F_star)
                P_star = P_star - np.outer(M_star, M_star) / F_star
                ll += -0.5 * (_LOG2PI + np.log(F_star) + v * v / F_star)
        a = m.c + m.T @ a
        P_star = m.T @ P_star @ m.T.T + RQR
        P_inf = m.T @ P_inf @ m.T.T
    return ll, n, ll


def kalman_smoother(kf: KalmanOutput, m: SystemMatrices) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-interval (Rauch-Tung-Striebel) smoother: states and variances given all the data."""
    n, k = kf.a_filt.shape
    a_s, P_s = kf.a_filt.copy(), kf.P_filt.copy()
    for t in range(n - 2, -1, -1):
        P_next = kf.P_pred[t + 1]
        J = np.linalg.lstsq(P_next, m.T @ kf.P_filt[t], rcond=None)[0].T  # P_filt T' P_next^-1 (P_next may be singular)
        a_s[t] = kf.a_filt[t] + J @ (a_s[t + 1] - kf.a_pred[t + 1])
        P_s[t] = kf.P_filt[t] + J @ (P_s[t + 1] - P_next) @ J.T
        P_s[t] = (P_s[t] + P_s[t].T) / 2
    return a_s, P_s


def disturbance_smoother(kf: KalmanOutput, m: SystemMatrices, a_smooth: np.ndarray) -> np.ndarray:
    """Smoothed state disturbances ``E[eta(t) | y] = Q R' P(t+1|t)^-1 (a(t+1|n) - a(t+1|t))`` (periods x shocks; the last
    period's is zero: nothing after it says anything about it). Since ``a(t+1) - c - T a(t) = R eta(t)`` for every
    realisation, ``R eta(t|n) = a(t+1|n) - c - T a(t|n)`` holds exactly for the smoothed means (pseudo-inverses where
    ``P(t+1|t)`` is singular, as with lagged copies of states)."""
    n = len(a_smooth)
    out = np.zeros((n, len(m.shocks)))
    QR = m.Q @ m.R.T
    for t in range(n - 1):
        gap = a_smooth[t + 1] - kf.a_pred[t + 1]
        out[t] = QR @ np.linalg.lstsq(kf.P_pred[t + 1], gap, rcond=None)[0]
    return out


# ---------------------------------------------------------------------------------------------
# the engine: statsmodels' compiled filter and smoother when installed (exact diffuse start, fast),
# otherwise the filter above. Both are checked against each other in the tests.

ENGINE = "auto"  # "auto" (statsmodels if installed), "statsmodels" or "native"


@dataclass(frozen=True, eq=False)
class KalmanRun:
    """Filtered, predicted and smoothed states with their variances (periods first), innovations,
    the log likelihood and the number of diffuse periods."""

    a_pred: np.ndarray
    P_pred: np.ndarray
    a_filt: np.ndarray
    P_filt: np.ndarray
    a_smooth: np.ndarray | None
    P_smooth: np.ndarray | None
    v: np.ndarray
    F: np.ndarray
    loglike: float
    diffuse: int
    engine: str
    eta_smooth: np.ndarray | None = None  # smoothed state disturbances E[eta(t) | y] (periods x shocks), eta(t) moving a(t) to a(t+1)
    loglike_diffuse: float = 0.0  # the diffuse periods' part of ``loglike`` (their terms; 0 without diffuse states)

    def loglike_as(self, diffuse_terms: str = "include") -> float:
        """The log likelihood with the diffuse periods' terms included (``loglike``) or left out."""
        return self.loglike if check_diffuse_terms(diffuse_terms) == "include" else self.loglike - self.loglike_diffuse


def _use_statsmodels() -> bool:
    if ENGINE == "native":
        return False
    import importlib.util

    found = importlib.util.find_spec("statsmodels") is not None
    if ENGINE == "statsmodels" and not found:
        raise ImportError("the statsmodels state-space engine needs statsmodels: pip install statsmodels (a pymetrix requirement)")
    return found


def _sm_model(y: np.ndarray, m: SystemMatrices, mode: str) -> Any:
    from statsmodels.tsa.statespace.mlemodel import MLEModel

    k = len(m.states)
    shocks = max(len(m.shocks), 1)
    model = MLEModel(y, k_states=k, k_posdef=shocks)
    R = m.R if len(m.shocks) else np.zeros((k, 1))
    Q = m.Q if len(m.shocks) else np.zeros((1, 1))
    for key, value in (("design", m.Z), ("obs_intercept", m.d), ("obs_cov", m.H), ("transition", m.T),
                       ("state_intercept", m.c), ("selection", R), ("state_cov", Q)):
        model.ssm[key] = value
    a0, P0, D = m.start(mode)
    if D.all():
        model.ssm.initialize_diffuse()
    elif not D.any():
        model.ssm.initialize_known(a0, P0)
    else:  # diffuse states come first in the layout (see StateSpace._parse)
        from statsmodels.tsa.statespace.initialization import Initialization

        nd = int(D.sum())
        init = Initialization(k)
        init.set((0, nd), "diffuse")
        init.set((nd, k), "known", constant=a0[nd:], stationary_cov=P0[nd:, nd:])
        model.ssm.initialization = init
    return model


def run(y: np.ndarray, m: SystemMatrices, mode: str = "auto", smooth: bool = True) -> KalmanRun:
    """Filter (and smooth) ``y`` with system ``m``: exact diffuse start when there are unit roots."""
    import warnings

    y = np.atleast_2d(np.asarray(y, dtype=float))
    if _use_statsmodels():
        model = _sm_model(y, m, mode)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = model.smooth([]) if smooth else model.filter([])
        n = len(y)
        a_s = r.smoothed_state.T if smooth else None
        P_s = np.moveaxis(r.smoothed_state_cov, 2, 0) if smooth else None
        eta = (r.smoothed_state_disturbance.T[:, :len(m.shocks)] if len(m.shocks) else np.zeros((n, 0))) if smooth else None
        return KalmanRun(r.predicted_state[:, :n].T, np.moveaxis(r.predicted_state_cov[:, :, :n], 2, 0), r.filtered_state.T,
                         np.moveaxis(r.filtered_state_cov, 2, 0), a_s, P_s, r.forecasts_error.T, np.moveaxis(r.forecasts_error_cov, 2, 0),
                         float(r.llf), int(r.nobs_diffuse), "statsmodels", eta, float(np.sum(r.llf_obs[:int(r.nobs_diffuse)])))
    a0, P0, burn = m.initial(mode, _scale(y))
    kf = kalman_filter(y, m, a0, P0, burn)
    a_s, P_s = kalman_smoother(kf, m) if smooth else (None, None)
    eta = disturbance_smoother(kf, m, a_s) if smooth else None
    first = 0.0
    if burn:
        try:
            ll, diffuse, first = _diffuse_parts(y, m, mode)
        except ValueError:  # correlated measurement errors: the approximately diffuse likelihood, without the first periods
            ll, diffuse = kf.loglike, burn
    else:
        ll, diffuse = kf.loglike, 0
    return KalmanRun(kf.a_pred, kf.P_pred, kf.a_filt, kf.P_filt, a_s, P_s, kf.v, kf.F, ll, diffuse, "native", eta, first)


def loglike(y: np.ndarray, m: SystemMatrices, mode: str = "auto", presample: int = 0, diffuse_terms: str = "include") -> float:
    """The log likelihood alone (what estimation evaluates many times), without the first ``presample`` periods.

    ``diffuse_terms``: the diffuse periods' terms included (the exact diffuse likelihood, the default) or left out
    (``"exclude"``, statsmodels' default convention); see the module docstring. Without diffuse states both are the same.

    A degenerate model, one that never pins down its states or predicts some observation
    with zero variance (e.g. every shock variance zero), has no likelihood: ``-inf``.
    """
    import warnings

    exclude = check_diffuse_terms(diffuse_terms) == "exclude"
    y = np.atleast_2d(np.asarray(y, dtype=float))
    compiled = _compiled_loglike(y, m, mode, presample)
    if compiled is not None:
        return compiled
    if _use_statsmodels():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = _sm_model(y, m, mode).filter([])
        diffuse, F, ll = int(r.nobs_diffuse), np.moveaxis(r.forecasts_error_cov, 2, 0), float(r.llf)
        if presample:
            ll = float(np.sum(r.llf_obs[presample:]))
        if exclude and diffuse > presample:
            ll = float(np.sum(r.llf_obs[diffuse:]))
    else:
        if presample:  # the period-by-period likelihood of the filter (approximate where states start diffuse)
            a0, P0, burn = m.initial(mode, _scale(y))
            kf = kalman_filter(y, m, a0, P0, burn)
            diffuse, F, ll = burn, kf.F, float(np.sum(kf.loglike_t[max(burn, presample):]))
        else:
            out = run(y, m, mode, smooth=False)
            diffuse, F, ll = out.diffuse, out.F, out.loglike_as(diffuse_terms)
    if diffuse > len(m.states):
        return -np.inf
    variances = np.einsum("tii->ti", F[diffuse:])
    seen = ~np.isnan(y[diffuse:])
    if np.any(variances[seen] <= 1e-12) or not np.isfinite(ll):
        return -np.inf
    return ll


def _compiled_loglike(y: np.ndarray, m: SystemMatrices, mode: str, presample: int) -> float | None:
    """The log likelihood by the compiled filter (``pymetrix.models._kalman_compiled``, with numba installed) where
    no state starts diffuse; None otherwise (statsmodels' filter, or the native one, then runs). The same likelihood
    as statsmodels' to rounding, except that once the predicted variance has converged (``_RICCATI_TOL``) it is
    reused, as the native filter and Dynare do."""
    if ENGINE != "auto":
        return None
    from .models import _kalman_compiled

    if not _kalman_compiled.available():
        return None
    a0, P0, D = m.start(mode)
    if D.any():
        return None
    k = len(m.states)
    RQR = m.R @ m.Q @ m.R.T if len(m.shocks) else np.zeros((k, k))
    c = np.ascontiguousarray
    ll, smallest, fail = _kalman_compiled.kernels().loglike(
        c(y), ~np.isnan(y), c(m.T, dtype=float), c(m.c, dtype=float), c(RQR, dtype=float), c(m.Z, dtype=float),
        c(m.d, dtype=float), c(m.H, dtype=float), c(a0, dtype=float), c(P0, dtype=float), int(presample), _RICCATI_TOL)
    if fail >= 0 or not smallest > 1e-12 or not np.isfinite(ll):
        return -np.inf
    return float(ll)


# ---------------------------------------------------------------------------------------------
# the specification: equations with declared parameters


def _lag_name(state: str, lag: int) -> str:
    return state if lag == 0 else f"{state}[-{lag}]"


@dataclass(frozen=True)
class _Linear:
    """An equation's right-hand side as coefficients (expressions in the parameters)."""

    on: dict[tuple[str, int], Expr]  # (state, lag) -> coefficient
    shocks: dict[str, Expr]
    const: Expr


@dataclass(frozen=True, eq=False)
class StateSpace:
    """A linear state-space model, written as transition and measurement equations (see the module)."""

    label: ClassVar[str] = "State space"
    topics: ClassVar[tuple[str, ...]] = ("state-space",)

    states: Sequence[str]
    transition: Sequence[str]
    measurement: Sequence[str]
    shocks: Mapping[str, str | float]
    params: Sequence[str | Param] = ()
    stationary: Sequence[str] = ()  # states that are stationary, e.g. a cycle or gap: they start from their distribution
    initial: Literal["auto", "diffuse", "stationary"] = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "transition", tuple(self.transition))
        object.__setattr__(self, "measurement", tuple(self.measurement))
        object.__setattr__(self, "stationary", tuple(self.stationary))
        strange = [s for s in self.stationary if s not in self.states]
        if strange:
            raise ExpressionError("stationary names " + ", ".join(strange) + ", which are not states")
        object.__setattr__(self, "shocks", dict(self.shocks))
        specs = tuple(p if isinstance(p, Param) else Param(p) for p in self.params)
        object.__setattr__(self, "params", specs)
        names = [p.name for p in specs]
        if len(set(names)) != len(names):
            raise ExpressionError("a parameter is declared twice")
        clash = set(names) & (set(self.states) | set(self.shocks))
        if clash:
            raise ExpressionError(f"{', '.join(sorted(clash))} cannot be both a parameter and a state or shock")
        for shock, sd in self.shocks.items():
            if isinstance(sd, str) and sd not in names:
                raise ExpressionError(f"the standard deviation of {shock} is {sd!r}, which is not a declared parameter")
        self._parse()

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)  # type: ignore[union-attr]

    # parsing

    def _linear(self, text: str, rhs: Expr, current_states: bool) -> _Linear:
        on: dict[tuple[str, int], Expr] = {}
        shocks: dict[str, Expr] = {}
        for ref in rhs.refs():
            if ref.id in self.states:
                if ref.lag > 0:
                    raise ExpressionError(f"{text!r}: leads ({ref}) are not allowed in a state-space model")
                if not current_states and ref.lag == 0:
                    raise ExpressionError(f"{text!r}: the right of a transition equation uses lagged states, e.g. {ref.id}[-1]")
                target = on
            elif ref.id in self.shocks:
                if ref.lag != 0:
                    raise ExpressionError(f"{text!r}: shocks enter without lags ({ref})")
                target = shocks
            else:
                raise ExpressionError(f"{text!r}: {ref.id!r} is not a state, a shock or a declared parameter")
            coef = diff(rhs, ref)
            if any(r.id in self.states or r.id in self.shocks for r in coef.refs()):
                raise ExpressionError(f"{text!r} is not linear in the states and shocks")
            key: Any = (ref.id, -ref.lag) if target is on else ref.id
            target[key] = coef
        const = rhs.map(lambda n: Num(0.0) if isinstance(n, Name) else None)
        return _Linear(on, shocks, const)

    def _parse(self) -> None:
        params = self.param_names
        transitions: dict[str, _Linear] = {}
        for text in self.transition:
            lhs_text, sep, rhs_text = text.partition("=")
            lhs = lhs_text.strip()
            if not sep or lhs not in self.states:
                raise ExpressionError(f"{text!r}: a transition equation is 'state = ...' for one of the states {list(self.states)}")
            if lhs in transitions:
                raise ExpressionError(f"{lhs} has two transition equations")
            transitions[lhs] = self._linear(text, parse_expression(rhs_text, params), current_states=False)
        missing = [s for s in self.states if s not in transitions]
        if missing:
            raise ExpressionError(f"no transition equation for {', '.join(missing)}")
        measurements: list[tuple[str, Expr, _Linear]] = []
        for text in self.measurement:
            lhs_text, sep, rhs_text = text.partition("=")
            if not sep:
                raise ExpressionError(f"{text!r}: a measurement equation is 'observed series = ...'")
            lhs = parse_expression(lhs_text.strip())
            if lhs.variables() & (set(self.states) | set(self.shocks)):
                raise ExpressionError(f"{text!r}: the left of a measurement equation is observed data")
            measurements.append((lhs_text.strip(), lhs, self._linear(text, parse_expression(rhs_text, params), current_states=True)))
        trans_shocks = {s for lin in transitions.values() for s in lin.shocks}
        meas_shocks = {s for _, _, lin in measurements for s in lin.shocks}
        both = trans_shocks & meas_shocks
        if both:
            raise ExpressionError(f"{', '.join(sorted(both))} cannot be both a state shock and a measurement error")
        unused = set(self.shocks) - trans_shocks - meas_shocks
        if unused:
            raise ExpressionError(f"shocks {', '.join(sorted(unused))} appear in no equation")
        # lagged copies of states: s[-j] is a state when a transition uses s[-(j+1)] or a measurement uses s[-j]
        depth = {s: 0 for s in self.states}
        for lin in transitions.values():
            for s, lag in lin.on:
                depth[s] = max(depth[s], lag - 1)
        for _, _, lin in measurements:
            for s, lag in lin.on:
                depth[s] = max(depth[s], lag)
        def block(states: list[str]) -> list[str]:
            return states + [_lag_name(s, j) for s in states for j in range(1, depth[s] + 1)]

        # diffuse states first, stationary ones (with their lags) after: initial blocks are contiguous
        layout = block([s for s in self.states if s not in self.stationary]) + block([s for s in self.states if s in self.stationary])
        object.__setattr__(self, "_transitions", transitions)
        object.__setattr__(self, "_measurements", measurements)
        object.__setattr__(self, "_depth", depth)
        object.__setattr__(self, "_layout", tuple(layout))
        object.__setattr__(self, "_trans_shocks", tuple(s for s in self.shocks if s in trans_shocks))
        object.__setattr__(self, "_meas_shocks", tuple(s for s in self.shocks if s in meas_shocks))

    @property
    def observables(self) -> tuple[str, ...]:
        return tuple(name for name, _, _ in self._measurements)  # type: ignore[attr-defined]

    # matrices for given parameter values

    def _sd(self, shock: str, values: Mapping[str, float]) -> float:
        sd = self.shocks[shock]
        return float(values[sd]) if isinstance(sd, str) else float(sd)

    def system(self, values: Mapping[str, float]) -> SystemMatrices:
        """The system matrices at parameter ``values``."""
        missing = [p for p in self.param_names if p not in values]
        if missing:
            raise NotFoundError(f"no value for {', '.join(missing)}")

        def num(e: Expr) -> float:
            return float(evaluate(e, _no_data, values))

        layout = self._layout  # type: ignore[attr-defined]
        index = {name: i for i, name in enumerate(layout)}
        k, p = len(layout), len(self._measurements)  # type: ignore[attr-defined]
        ts, ms = self._trans_shocks, self._meas_shocks  # type: ignore[attr-defined]
        T, c, R = np.zeros((k, k)), np.zeros(k), np.zeros((k, len(ts)))
        for state, lin in self._transitions.items():  # type: ignore[attr-defined]
            row = index[state]
            for (s, lag), coef in lin.on.items():
                T[row, index[_lag_name(s, lag - 1)]] += num(coef)
            for shock, coef in lin.shocks.items():
                R[row, ts.index(shock)] = num(coef)
            c[row] = num(lin.const)
        for s, deep in self._depth.items():  # type: ignore[attr-defined]
            for j in range(1, deep + 1):
                T[index[_lag_name(s, j)], index[_lag_name(s, j - 1)]] = 1.0
        Q = np.diag([self._sd(s, values) ** 2 for s in ts]) if ts else np.zeros((0, 0))
        Z, d, G = np.zeros((p, k)), np.zeros(p), np.zeros((p, len(ms)))
        for row, (_, _, lin) in enumerate(self._measurements):  # type: ignore[attr-defined]
            for (s, lag), coef in lin.on.items():
                Z[row, index[_lag_name(s, lag)]] += num(coef)
            for shock, coef in lin.shocks.items():
                G[row, ms.index(shock)] = num(coef)
            d[row] = num(lin.const)
        H = G @ np.diag([self._sd(s, values) ** 2 for s in ms]) @ G.T if ms else np.zeros((p, p))
        stationary = tuple(name.split("[")[0] in self.stationary for name in layout) if self.stationary else ()
        return SystemMatrices(tuple(layout), self.observables, ts, T, c, R, Q, Z, d, H, stationary)

    # data

    def _observed(self, data: Dataset, span: Any) -> tuple[pd.Index, np.ndarray]:
        if data.kind != "time_series":
            raise SpecificationError("state-space models need a single time series")
        frame = data.frame[data.sample_mask(span)]
        sub = type(data)(frame, data.axes, data.variables)
        cols = []
        for name, lhs, _ in self._measurements:  # type: ignore[attr-defined]
            value = evaluate(lhs, sub.lagged)
            cols.append(np.broadcast_to(np.asarray(value, dtype=float), (len(frame),)))
        y = np.column_stack(cols) if cols else np.zeros((len(frame), 0))
        observed = ~np.isnan(y).all(axis=1)
        if not observed.any():
            raise SpecificationError("no observations of the measured series in the sample")
        first, last = np.flatnonzero(observed)[[0, -1]]
        return frame.index[first : last + 1], y[first : last + 1]

    # estimation and filtering

    def _bounds(self, spec: Param) -> tuple[float, float]:
        low, high = spec.bounds
        if spec.name in {sd for sd in self.shocks.values() if isinstance(sd, str)}:
            low = max(low, 0.0)  # standard deviations are not negative
        return low, high

    def filter(self, data: Dataset, values: Mapping[str, float], span: Any = None, diffuse_terms: str = "include") -> FittedStateSpace:
        """Filter and smooth at given (calibrated) parameter values, without estimating. ``diffuse_terms`` as in ``fit``."""
        index, y = self._observed(data, span)
        return self._result(data, index, y, dict(values), None, estimated=(), diffuse_terms=check_diffuse_terms(diffuse_terms))

    def loglike(self, data: Dataset, values: Mapping[str, float], span: Any = None, diffuse_terms: str = "include") -> float:
        """The log likelihood at these values; ``diffuse_terms`` as in ``fit``."""
        index, y = self._observed(data, span)
        return _loglike(self, y, self.system(values), check_diffuse_terms(diffuse_terms))[0]

    @api("pymetrix:state_space.fit")
    @records_warnings
    def fit(self, data: Dataset, span: Any = None, start: Mapping[str, float] | None = None,
            fixed: Mapping[str, float] | None = None, cov: Literal["classic", "none"] = "classic",
            max_iter: int = 2000, diffuse_terms: Literal["include", "exclude"] = "include") -> FittedStateSpace:
        """Estimate the free parameters by maximum likelihood, or by posterior mode when parameters
        have priors. ``fixed`` sets parameters (e.g. ``{"sd_slope": 0}``); ``start`` gives starting values.

        ``diffuse_terms``, with diffuse states: ``"include"`` (default) maximises the exact diffuse log likelihood of
        Durbin and Koopman (2012, section 7.2.2), with the diffuse periods' terms; ``"exclude"`` leaves those terms out
        (Harvey's likelihood of the later observations given the first ``d``, what statsmodels' ``UnobservedComponents``
        reports by default). The log likelihood, AIC and BIC follow; where the diffuse part of the system has no
        parameters (unobserved-components models) the two differ by a constant, ``d/2 log 2 pi + 1/2 sum log F_inf``
        for one series, and the estimates and standard errors are the same. ``criteria`` gives either from a fit."""
        diffuse_terms = check_diffuse_terms(diffuse_terms)
        index, y = self._observed(data, span)
        fixed = dict(fixed or {})
        unknown = set(fixed) - set(self.param_names)
        if unknown:
            raise NotFoundError(f"{', '.join(sorted(unknown))} are not parameters of the model")
        free = [p for p in self.params if p.name not in fixed]  # type: ignore[union-attr]
        starts = {**_default_starts(self, y), **{p.name: p.start for p in self.params if p.start is not None}, **dict(start or {})}  # type: ignore[union-attr]
        bounds = [self._bounds(p) for p in free]
        x0 = np.array([min(max(starts[p.name], lo + 1e-6 if np.isfinite(lo) else -np.inf), hi - 1e-6 if np.isfinite(hi) else np.inf)
                       for p, (lo, hi) in zip(free, bounds)])

        from .progress import report

        tried = [0]

        def objective(x: np.ndarray) -> float:
            tried[0] += 1
            report(None, f"estimating: likelihood evaluation {tried[0]:,}")  # the app's progress bar; Cancel stops here
            values = {**fixed, **{p.name: float(v) for p, v in zip(free, x)}}  # type: ignore[union-attr]
            try:
                value = _loglike(self, y, self.system(values), diffuse_terms)[0] + _log_prior(self, values)
            except (ValueError, np.linalg.LinAlgError):
                return 1e12
            return -value if np.isfinite(value) else 1e12

        if free:
            scipy_bounds = [(lo if np.isfinite(lo) else None, hi if np.isfinite(hi) else None) for lo, hi in bounds]
            best = optimize.minimize(objective, x0, method="L-BFGS-B", bounds=scipy_bounds,
                                     options={"maxiter": max_iter, "maxfun": max_iter * 20, "ftol": 1e-12, "gtol": 1e-7})
            if not best.success:  # stopped early (often on a flat ridge): polish without derivatives, then try again
                polish = optimize.minimize(objective, best.x, method="Nelder-Mead", bounds=scipy_bounds,
                                           options={"maxiter": max_iter * 10, "xatol": 1e-7, "fatol": 1e-9})
                if polish.fun <= best.fun:
                    best = optimize.minimize(objective, polish.x, method="L-BFGS-B", bounds=scipy_bounds,
                                             options={"maxiter": max_iter, "maxfun": max_iter * 20, "ftol": 1e-12, "gtol": 1e-7})
                    if polish.fun < best.fun:
                        best = polish
            estimate = {**fixed, **{p.name: float(v) for p, v in zip(free, best.x)}}  # type: ignore[union-attr]
            converged = bool(best.success) or _at_optimum(objective, best.x, bounds)
        else:
            estimate, converged = dict(fixed), True
        covariance = None
        if free and cov == "classic":
            covariance = _covariance(self, y, estimate, [p.name for p in free], bounds, diffuse_terms)  # type: ignore[union-attr]
        return self._result(data, index, y, estimate, covariance, estimated=tuple(p.name for p in free), converged=converged,  # type: ignore[union-attr]
                            diffuse_terms=diffuse_terms)

    def _result(self, data: Dataset, index: pd.Index, y: np.ndarray, values: dict[str, float], covariance: np.ndarray | None,
                estimated: tuple[str, ...], converged: bool = True, diffuse_terms: str = "include") -> FittedStateSpace:
        from .data import Dataset
        from .estimation.params import point

        m = self.system(values)
        kf = run(y, m, self.initial)
        names = list(estimated)
        cov_full = covariance if covariance is not None else None
        params = point(names, np.array([values[n] for n in names]), cov_full) if names else point([], np.array([]), None)
        ll, diffuse = kf.loglike_as(diffuse_terms), kf.diffuse
        nobs = int(np.sum(~np.isnan(y).all(axis=1)))
        k_free = len(names)
        stats = {"loglike": ll, "nobs": float(nobs), "k": float(k_free), "aic": -2 * ll + 2 * k_free,
                 "bic": -2 * ll + k_free * np.log(max(nobs, 1)), "diffuse periods": float(diffuse)}
        if not converged:
            stats["converged"] = 0.0
        frame_index = pd.Index(index, name=data.time_axis.name)  # type: ignore[union-attr]

        def states_frame(a: np.ndarray, P: np.ndarray) -> Dataset:
            df = pd.DataFrame(a, index=frame_index, columns=list(m.states))
            for i, s in enumerate(m.states):
                df[f"{s}_se"] = np.sqrt(np.maximum(P[:, i, i], 0.0))
            return Dataset(df, data.axes)

        prov = Provenance(data_vintages={v: data.variables[v].vintage for v in data.names if v in data.variables},
                          spec={"type": "StateSpace", "states": list(self.states), "transition": list(self.transition),
                                "measurement": list(self.measurement)},
                          method={"type": "Kalman filter", "estimation": "posterior mode" if _has_priors(self) else "maximum likelihood",
                                  "initial": self.initial, "backend": _backend(kf.engine),
                                  **({"diffuse_terms": diffuse_terms} if diffuse_terms != "include" else {})},
                          sample=(str(frame_index[0]), str(frame_index[-1])), nobs=nobs, dropped=0, package_version=__version__,
                          data_fingerprint=data_fingerprint(data, sorted(set().union(*(lhs.variables() for _, lhs, _ in self._measurements))), frame_index),
                          backend={"name": "statsmodels", "version": _backend(kf.engine).partition(" ")[2], "wraps": "statsmodels MLEModel (Kalman filter)"}
                          if kf.engine == "statsmodels" else {"name": "pymetrix", "version": __version__, "wraps": "pymetrix Kalman filter"})
        observed = frame_index[~np.isnan(y).all(axis=1)]  # periods with an observation; the filter runs through the others
        prov = with_effective_sample(prov, data, observed, span=(str(frame_index[0]), str(frame_index[-1])),
                                     variables=sorted(set().union(*(lhs.variables() for _, lhs, _ in self._measurements))))
        if not converged:
            prov = replace(prov, messages=(*prov.messages, "the optimiser did not converge"))
        return FittedStateSpace(self, params, values, m, kf, states_frame(kf.a_filt, kf.P_filt), states_frame(kf.a_smooth, kf.P_smooth),
                                states_frame(kf.a_pred, kf.P_pred), y, frame_index, stats, prov, data.axes)


def _loglike(model: StateSpace, y: np.ndarray, m: SystemMatrices, diffuse_terms: str = "include") -> tuple[float, int]:
    return loglike(y, m, model.initial, diffuse_terms=diffuse_terms), 0


def _at_optimum(objective: Any, x: np.ndarray, bounds: list[tuple[float, float]], tol: float = 1e-3) -> bool:
    """True if no free direction improves the objective: the projected gradient is small
    (a parameter at a bound may have a gradient pointing out of bounds)."""
    f0 = objective(x)
    for i, (lo, hi) in enumerate(bounds):
        h = 1e-6 * max(1.0, abs(x[i]))
        up, down = x.copy(), x.copy()
        up[i] = min(x[i] + h, hi)
        down[i] = max(x[i] - h, lo)
        slope_up = (objective(up) - f0) / (up[i] - x[i]) if up[i] > x[i] else np.inf
        slope_down = (f0 - objective(down)) / (x[i] - down[i]) if down[i] < x[i] else -np.inf
        if slope_up < -tol * max(1.0, abs(f0)) or slope_down > tol * max(1.0, abs(f0)):
            return False
    return True


def _backend(engine: str) -> str:
    if engine == "statsmodels":
        from .adapters import backend

        return backend("statsmodels")
    return f"pymetrix {__version__}"


def _scale(y: np.ndarray) -> float:
    """The size of the data, for the diffuse initial variance."""
    finite = np.abs(y[np.isfinite(y)])
    return float(finite.max()) if finite.size else 1.0


def _no_data(name: str, lag: int) -> Any:
    raise ExpressionError(f"{name} is not a parameter")


def _has_priors(model: StateSpace) -> bool:
    return any(p.prior is not None for p in model.params)  # type: ignore[union-attr]


def _log_prior(model: StateSpace, values: Mapping[str, float]) -> float:
    return float(sum(p.prior.logpdf(values[p.name]) for p in model.params if p.prior is not None))  # type: ignore[union-attr]


def _default_starts(model: StateSpace, y: np.ndarray) -> dict[str, float]:
    """Rough starting values: standard deviations from the data's changes, others inside their bounds."""
    changes = np.diff(y, axis=0)
    scale = float(np.nanstd(changes)) if np.isfinite(changes).any() else 1.0
    scale = scale if scale > 0 else 1.0
    sds = {sd for sd in model.shocks.values() if isinstance(sd, str)}
    out = {}
    for p in model.params:  # type: ignore[union-attr]
        low, high = model._bounds(p)
        if p.name in sds:
            out[p.name] = scale / 2
        elif np.isfinite(low) and np.isfinite(high):
            out[p.name] = low + 0.5 * (high - low)
        elif np.isfinite(low):
            out[p.name] = low + 0.5
        else:
            out[p.name] = 0.1
    return out


def _covariance(model: StateSpace, y: np.ndarray, values: dict[str, float], names: list[str],
                bounds: list[tuple[float, float]], diffuse_terms: str = "include") -> np.ndarray | None:
    """Inverse of the negative Hessian of the log likelihood (plus log prior) in the parameters.
    Parameters at a bound (e.g. a standard deviation of zero) get no standard error (NaN)."""
    from .estimation.numdiff import hessian

    interior = [i for i, (n, (lo, hi)) in enumerate(zip(names, bounds))
                if not (np.isclose(values[n], lo, atol=1e-5) or np.isclose(values[n], hi, atol=1e-5))]
    cov = np.full((len(names), len(names)), np.nan)
    if not interior:
        return cov
    inner = [names[i] for i in interior]

    def f(x: np.ndarray) -> float:
        vals = {**values, **dict(zip(inner, x))}
        return _loglike(model, y, model.system(vals), diffuse_terms)[0] + _log_prior(model, vals)

    try:
        H = hessian(f, np.array([values[n] for n in inner]))
        sub = np.linalg.inv(-H)
    except (np.linalg.LinAlgError, ValueError):
        return cov
    if np.any(np.diag(sub) < 0):
        return cov
    cov[np.ix_(interior, interior)] = sub
    return cov


# ---------------------------------------------------------------------------------------------
# results


@dataclass(frozen=True, eq=False)
class FittedStateSpace:
    """A filtered (and possibly estimated) state-space model.

    ``smoothed`` and ``filtered`` are datasets of the states with standard errors
    (``trend``, ``trend_se``, ...); ``predicted`` holds one-step-ahead predictions.
    """

    model: StateSpace
    params: Any
    values: dict[str, float]
    matrices: SystemMatrices
    kalman: KalmanRun = field(repr=False)
    filtered: Any = field(repr=False)
    smoothed: Any = field(repr=False)
    predicted: Any = field(repr=False)
    y: np.ndarray = field(repr=False)
    index: pd.Index = field(repr=False)
    stats: dict[str, float] = field(default_factory=dict)
    provenance: Provenance | None = None
    axes: Any = None

    def innovations(self) -> pd.DataFrame:
        """Standardised one-step-ahead prediction errors of the observed series (should look like white noise)."""
        out = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            for i, name in enumerate(self.matrices.observables):
                z = self.kalman.v[:, i] / np.sqrt(self.kalman.F[:, i, i])
                z[: self.kalman.diffuse] = np.nan  # the diffuse start: no prediction to speak of
                out[name] = z
        return pd.DataFrame(out, index=self.index)

    def criteria(self, diffuse_terms: str | None = None) -> dict[str, float]:
        """The log likelihood and the information criteria (AIC, BIC, Hannan-Quinn) at these parameter values, with the
        diffuse periods' terms included (``"include"``, the exact diffuse likelihood of Durbin and Koopman) or left
        out (``"exclude"``, statsmodels' default convention); None: as the fit was made. AIC and BIC equal ``stats``'
        for the fit's own convention. Compare criteria across models only in one convention, and only between models
        with the same diffuse periods (see ``pymetrix.statespace``)."""
        method = self.provenance.method if self.provenance is not None else {}
        terms = check_diffuse_terms(diffuse_terms or method.get("diffuse_terms", "include"))
        ll = self.kalman.loglike if terms == "include" else self.kalman.loglike - getattr(self.kalman, "loglike_diffuse", 0.0)
        from .estimation import criteria as c

        k, n = self.stats.get("k", 0.0), max(self.stats.get("nobs", 1.0), 1.0)
        return {"loglike": float(ll), "aic": c.aic(ll, k), "bic": c.bic(ll, k, n), "hq": c.hq(ll, k, n) if n > 1 else float("nan"),
                "diffuse periods": float(self.kalman.diffuse)}

    def summary(self) -> Table:
        rows: list[list[Any]] = []
        names = list(self.params.names)
        se = self.params.se if names else []
        for i, n in enumerate(names):
            rows.append([n, self.values[n], se[i] if len(se) else None])
        for n, v in self.values.items():
            if n not in names:
                rows.append([n, v, "fixed"])
        notes = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in self.stats.items()]
        estimation = self.provenance.method["estimation"] if self.provenance else ""
        return Table(["parameter", "value", "std err"], rows, title=f"State space ({estimation}): {', '.join(self.model.states)}", notes=notes)

    @api()
    def forecast(self, data: Dataset, span: tuple[Any, Any], alpha: float = 0.05, dynamic: bool = True) -> Dataset:
        """Forecasts of the observed series (and states) for the periods in ``span``, with intervals.

        The filter runs on the data up to the start of the span and on beyond it with
        the observations treated as missing, so forecasts condition on the data before
        the span."""
        from scipy import stats as sps

        from .data import Dataset

        if not dynamic:
            raise SpecificationError("state-space forecasts are dynamic (multi-step from the end of the data)")
        freq = self.index.freq
        from .data.sample import as_span
        span = as_span(span)  # '1990Q1:2019Q4' or a pair
        start, end = pd.Period(str(span[0]), freq=freq), pd.Period(str(span[1]), freq=freq)
        first = self.index[0]
        periods = pd.period_range(first, end, freq=freq, name=self.index.name)
        model = self.model
        frame = data.frame.reindex(periods)
        sub = Dataset(frame, data.axes, data.variables)
        cols = [np.broadcast_to(np.asarray(evaluate(lhs, sub.lagged), dtype=float), (len(frame),)) for _, lhs, _ in model._measurements]  # type: ignore[attr-defined]
        y = np.column_stack(cols)
        y[np.asarray(periods >= start)] = np.nan
        m = self.matrices
        kf = run(y, m, model.initial, smooth=False)
        z = sps.norm.ppf(1 - alpha / 2)
        keep = np.asarray(periods >= start)
        out = {}
        for i, name in enumerate(m.observables):
            mean = m.d[i] + kf.a_pred[:, :] @ m.Z[i]
            var = np.einsum("j,tjk,k->t", m.Z[i], kf.P_pred, m.Z[i]) + m.H[i, i]
            se = np.sqrt(np.maximum(var, 0))
            out[name], out[f"{name}_se"] = mean, se
            out[f"{name}_lower"], out[f"{name}_upper"] = mean - z * se, mean + z * se
        for j, s in enumerate(m.states):
            if s in model.states:
                out[s] = kf.a_pred[:, j]
        df = pd.DataFrame(out, index=periods)[keep]
        return Dataset(df, data.axes)

    def plot(self, which: Literal["smoothed", "filtered"] = "smoothed") -> Any:
        """The estimated states with 95% bands: smoothed (two-sided) or filtered (real-time)."""
        from .figures import states_figure

        return states_figure(self, which)

    def irf(self, periods: int = 20) -> pd.DataFrame:
        return impulse_responses(self.matrices, self.model, periods)

    def simulate(self, periods: int, seed: int | None = None) -> pd.DataFrame:
        return simulate(self.matrices, self.model, periods, seed)

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()

    def __str__(self) -> str:
        return str(self.summary())


def impulse_responses(m: SystemMatrices, model: StateSpace, periods: int = 20) -> pd.DataFrame:
    """Responses of the states and observed series to a one-standard-deviation shock, by horizon.
    Columns are (shock, variable)."""
    out = {}
    for j, shock in enumerate(m.shocks):
        a = m.R[:, j] * np.sqrt(m.Q[j, j])
        path = []
        for _ in range(periods + 1):
            path.append(a)
            a = m.T @ a
        path = np.array(path)
        for i, s in enumerate(m.states):
            if s in model.states:
                out[(shock, s)] = path[:, i]
        for i, name in enumerate(m.observables):
            out[(shock, name)] = path @ m.Z[i]
    frame = pd.DataFrame(out)
    frame.index.name = "horizon"
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["shock", "variable"])
    return frame


def simulate(m: SystemMatrices, model: StateSpace, periods: int, seed: int | None = None) -> pd.DataFrame:
    """A simulated path of the states and observed series: stationary states start from their
    distribution, diffuse ones (trends) from zero."""
    rng = make_rng(seed)
    a0, P0, _ = m.start(model.initial)
    a = rng.multivariate_normal(a0, P0) if len(a0) else a0
    rows = []
    for _ in range(periods):
        eps = rng.multivariate_normal(np.zeros(len(m.observables)), m.H) if len(m.observables) else np.zeros(0)
        y = m.d + m.Z @ a + eps
        rows.append(np.r_[a, y])
        eta = rng.multivariate_normal(np.zeros(len(m.shocks)), m.Q) if len(m.shocks) else np.zeros(0)
        a = m.c + m.T @ a + m.R @ eta
    cols = list(m.states) + list(m.observables)
    frame = pd.DataFrame(rows, columns=cols)
    return frame[[c for c in cols if c in model.states or c in m.observables]]


# ---------------------------------------------------------------------------------------------
# unobserved components: the common models, ready-made

_TRENDS = {
    "local level": (["trend = trend[-1] + e_trend"], ["trend"]),
    "random walk with drift": (["trend = drift + trend[-1] + e_trend"], ["trend"]),
    "local linear trend": (["trend = trend[-1] + slope[-1] + e_trend", "slope = slope[-1] + e_slope"], ["trend", "slope"]),
    "smooth trend": (["trend = trend[-1] + slope[-1]", "slope = slope[-1] + e_slope"], ["trend", "slope"]),
}


@dataclass(frozen=True)
class UnobservedComponents:
    """A series as trend + cycle + irregular, estimated by maximum likelihood with the Kalman filter.

    Choose the trend (local level, random walk with drift, local linear trend, or smooth trend, whose
    trend has no shock of its own and gives an HP-like trend), the order of an autoregressive cycle
    (0 for none) and whether to include an irregular (white-noise) component. The smoothed trend and
    cycle are two-sided estimates; the filtered ones use only past data (real-time).

    The split into trend and cycle is identified only by the model's assumptions: different trend specifications
    fit the data about equally well yet give quite different cycles, so treat the cycle as model-dependent and
    compare specifications. A trend with its own shocks absorbs much of the low-frequency movement and leaves a
    small, noisy cycle; the smooth trend forces more of the movement into the cycle. A variance estimated at
    zero makes that component deterministic. Smoothed estimates near the end of the sample are revised as new
    data arrive, which matters when the latest gap is used for policy; the filtered ones are what was known then.
    """

    label: ClassVar[str] = "Unobserved components"
    topics: ClassVar[tuple[str, ...]] = ("state-space", "trend-cycle-filters")

    series: Annotated[str, Widget("text", help="the series to decompose, e.g. log real GDP")]
    trend: Annotated[Literal["local level", "random walk with drift", "local linear trend", "smooth trend"],
                     Widget("choice", help="local level: random walk; with drift: plus a constant growth; local linear: "
                                         "with a random-walk slope; smooth: slope shocks only")] = "local linear trend"
    cycle: Annotated[int, Widget("number", min=0, max=4, help="order of the autoregressive cycle; 0 = no cycle")] = 2
    irregular: Annotated[bool, Widget("checkbox", help="include a white-noise irregular component")] = True

    def spec(self) -> StateSpace:
        transition, states = _TRENDS[self.trend]
        transition, states = list(transition), list(states)
        params: list[Param | str] = []
        shocks: dict[str, str | float] = {}
        if "e_trend" in " ".join(transition):
            shocks["e_trend"] = "sd_trend"
            params.append(Param("sd_trend", bounds=(0, np.inf)))
        if "e_slope" in " ".join(transition):
            shocks["e_slope"] = "sd_slope"
            params.append(Param("sd_slope", bounds=(0, np.inf)))
        if "drift" in transition[0]:
            params.insert(0, "drift")
        measured = "trend"
        if self.cycle:
            phis = [f"phi{i}" for i in range(1, self.cycle + 1)]
            terms = " + ".join(f"{phi}*cycle[-{i}]" for i, phi in enumerate(phis, 1))
            transition.append(f"cycle = {terms} + e_cycle")
            states.append("cycle")
            shocks["e_cycle"] = "sd_cycle"
            params += [Param(phi, bounds=(-2.0, 2.0) if self.cycle > 1 else (-0.999, 0.999), start=0.5 if i == 0 else 0.0)
                       for i, phi in enumerate(phis)]
            params.append(Param("sd_cycle", bounds=(0, np.inf)))
            measured += " + cycle"
        if self.irregular:
            measured += " + u"
            shocks["u"] = "sd_irregular"
            params.append(Param("sd_irregular", bounds=(0, np.inf)))
        return StateSpace(states, transition, [f"{self.series} = {measured}"], shocks, params,
                          stationary=["cycle"] if self.cycle else [])

    @api("pymetrix:unobserved_components.fit")
    def fit(self, data: Dataset, span: Any = None, **options: Any) -> FittedStateSpace:
        return self.spec().fit(data, span=span, **options)


register("models", UnobservedComponents)
