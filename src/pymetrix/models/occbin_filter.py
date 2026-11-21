"""The OccBin filter: the Kalman filter and smoother for models with occasionally binding constraints, and their
likelihood, as Dynare 6's piecewise Kalman filter (Giovannini, Pfeiffer and Ratto 2021).

    result = occbin_filter(zlb_template, data)          # the template's own measurement equations and shock sizes
    result.smoothed                                     # the model's variables from the data, in levels, with the shocks
    result.binding                                      # when each constraint binds, on the smoothed path
    result.loglike
    fit = zlb_template.estimate(data, ["kappa", "rho"], occbin=True)   # estimated with this likelihood
    occbin_filter(zlb_template, data, smoother="iterated").regimes      # regimes worked out again on the smoothed path
    result.decompose()                                  # shock decomposition, the regimes held at the smoothed ones

In each period the regimes expected from the filtered state, with no further shocks, give that period's decision rule
x(t) = c(t) + P(t) x(t-1) + Q(t) e(t) (``occbin``'s time-varying rules, found backwards from the end of the horizon);
the Kalman filter predicts and updates with it. The shock it then estimates may change the regime: OccBin is solved
again from the filtered state with that shock, and the period filtered again with the regimes it gives, until they
no longer change. The likelihood is the sum of the prediction errors' densities; the smoother runs backwards through
the same time-varying rules. Measurement equations are those of the reference regime, linearised at its steady
state. The regime of each period depends on the state through its filtered mean, as in Dynare.

With ``smoother="iterated"`` (Dynare 6's OccBin smoother), the regimes are then worked out again on the smoothed
path, each period's from the smoothed state before it and the smoothed shock, and the filter and smoother run again
with them, until they no longer change: the smoothed path then satisfies the constraints period by period.

Correlated shocks (``shock_corr``) enter as the full covariance Q of the filter; the smoothed shocks are then the
likeliest given each period's state, ``Q R' (R Q R')^+ (a(t|n) - c(t) - T(t) a(t-1|n))`` (least squares without
correlations, as before), and decompositions are of the shocks orthogonalised by Cholesky. Known-date regimes
(``regimes``) in the sample combine with the constraints period by period (``occbin._Piecewise``); forecasts take them
too, and draw correlated shocks through the Cholesky factor of their covariance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._api import api
from ..output import Table
from .solvers import ModelError

__all__ = ["OccBinFilter", "occbin_filter"]

_REGIME_ITERATIONS = 50
BINDING = "binding constraints"  # the decomposition's part for the binding regimes' constants


@dataclass(frozen=True, eq=False)
class OccBinFilter:
    """The OccBin filter's result: see the module docstring."""

    smoothed: Any  # Dataset: the model's variables in levels, fitted observables and the smoothed shocks
    filtered: pd.DataFrame  # the model's variables in levels, from data up to each period
    binding: pd.DataFrame  # True where each constraint binds: the regimes the filter chose (data to each period), used by the smoother too
    ex_post: pd.DataFrame  # True where each constraint binds when evaluated on the smoothed path
    loglike: float
    constraints: tuple[str, ...]
    iterations: int  # the most regime iterations a period needed
    observables: tuple[str, ...] = ()
    cycles: int = 0  # periods whose regime guesses cycled, settled by the likelier regime
    smoother: str = "filter"  # "filter": smoothed with the filter's regimes; "iterated": regimes worked out on the smoothed path
    smoothed_binding: pd.DataFrame | None = None  # True where each constraint binds in the regimes the smoother used
    smoother_iterations: int = 0  # iterated: the smoother runs until its regimes settled
    smoother_cycled: bool = False  # iterated: its regimes cycled, and those the smoothed path contradicts least were kept
    smoother_kept: int = 0  # iterated: periods whose regimes from the smoothed path make the data impossible, kept as the filter's
    rules: Any = field(default=None, repr=False)  # the smoother's time-varying rules and smoothed states and shocks (decompose)

    @property
    def regimes(self) -> pd.DataFrame:
        """When each constraint binds in the regimes the smoothed path was computed with: the filter's, or with
        ``smoother="iterated"`` those settled on the smoothed path."""
        return self.binding if self.smoothed_binding is None else self.smoothed_binding

    def decompose(self, orthogonal: bool = True) -> Any:
        """The smoothed path split into the shocks' contributions, as a ``Decomposition``: with the regimes held at
        those the smoother used, each shock's contribution and the starting point's go through the smoother's
        time-varying rules x(t) = c(t) + T(t) x(t-1) + R(t) e(t), and the constants c(t) of the binding regimes (where
        a constraint holds the variable, as a rate at its floor) are a part of their own, ``"binding constraints"``;
        with known-date regimes, their constants are in that part too. The parts add up to the path exactly. In a
        model with a constraint the contributions depend on the regimes (the shocks' effects are not additive across
        regimes): this is the decomposition given the smoothed regimes. When nothing binds it is the linear
        decomposition. With correlated shocks the parts are of the shocks orthogonalised by the Cholesky factor of
        their covariance, in the model's order, as impulse responses are (``orthogonal=False``: of the shocks
        themselves)."""
        from .decompose import path_decomposition

        r = self.rules
        if r is None:
            raise ModelError("this result has no smoother rules: run occbin_filter again")
        factor = r.get("factor") if orthogonal else None
        return path_decomposition(r["T"], r["c"], r["R"], r["a_s"], r["e_s"], self.binding.index, r["variables"], r["observed"],
                                  r["shocks"], BINDING, "Shock decomposition (OccBin, regimes held at the smoothed ones)", factor)

    def summary(self) -> Table:
        index = self.binding.index
        span = f"{index[0]} to {index[-1]}" if len(index) else "no periods"
        iterated = self.smoother == "iterated"
        rows = []
        for c in self.constraints:
            used = [i + 1 for i, b in enumerate(self.binding[c]) if b]
            row = [c, len(used), _labels(index, used)]
            if iterated:
                settled = [i + 1 for i, b in enumerate(self.regimes[c]) if b]
                row += [len(settled), _labels(index, settled)]
            rows.append(row + [int((self.regimes[c] != self.ex_post[c]).sum())])
        notes = [f"Sample: {span}", f"Log likelihood: {self.loglike:.4f}", f"Observed: {', '.join(self.observables)}",
                 f"Regimes settled in at most {self.iterations} iterations a period"
                 + (f"; in {self.cycles} periods the guesses cycled and the likelier regime was taken" if self.cycles else "")]
        if iterated:
            notes.append(f"Smoother iterated on the smoothed path: its regimes settled after {self.smoother_iterations} runs"
                         + ("; they cycled, and those the smoothed path contradicts least were kept" if self.smoother_cycled else "")
                         + (f"; in {self.smoother_kept} periods the regimes the smoothed path implies make the data impossible "
                            "(a series observed without error off its bound), and the filter's were kept" if self.smoother_kept else ""))
            notes.append("Binding: the regimes the filter chose with data up to each period. Smoothed: the regimes worked out "
                         "again on the smoothed path, which the smoother used. Ex post: whether the constraint binds when "
                         "evaluated on the smoothed path")
        else:
            notes.append("Binding: the regimes the filter chose with data up to each period, which the smoother keeps. Ex post: "
                         "whether the constraint binds when evaluated on the smoothed path, which later data can revise")
        columns = ["constraint", "periods binding", "when"] + (["periods binding, smoothed", "when, smoothed"] if iterated else [])
        return Table(columns + ["periods differing ex post"], rows, title="OccBin filter", notes=notes)

    def plot(self, variables: Sequence[str] | str | None = None) -> Any:
        """The smoothed paths, with the periods in which a constraint binds shaded."""
        names = [variables] if isinstance(variables, str) else list(variables or list(self.filtered.columns)[:4])
        return self.smoothed.plot(names)

    def __repr__(self) -> str:
        binding = ", ".join(f"{c}: {int(self.regimes[c].sum())} periods" for c in self.constraints)
        return f"<OccBinFilter {len(self.binding)} periods, log likelihood {self.loglike:.3f}; binding {binding}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


def _labels(index: pd.Index, periods: list[int]) -> str:
    from .occbin import _spells

    text = _spells(periods)
    if not text:
        return "never"
    def name(k: str) -> str:
        return str(index[int(k) - 1])
    return ", ".join("–".join(name(k) for k in part.split("-")) for part in text.split(", "))


def _setup(obj: Any, measurement: Any, shock_sd: Mapping[str, float] | None, shocks: Sequence[str] | None) -> tuple[Any, ...]:
    """The model, its shocks, the steady-state guess, the measurement map and the shock sizes, from a template, a
    Dynare or IRIS model, a fitted model or a Model."""
    from .estimation import FittedModel
    from .measurement import MeasurementMap
    from .model import Model

    if isinstance(obj, FittedModel):
        model, names = obj.model, tuple(shocks or obj.shock_sd)
        measurement = measurement or obj.measurement
        shock_sd = {**dict(obj.shock_sd), **dict(shock_sd or {})}
        guess = {}
    elif isinstance(obj, Model):
        model, names, guess = obj, tuple(shocks or obj.compile().shocks), {}
    else:
        model = getattr(obj, "model", None)
        if not isinstance(model, Model):
            raise TypeError(f"the OccBin filter runs on a template, a Dynare or IRIS model, a fitted model or a Model, not {type(obj).__name__}")
        names = tuple(shocks or getattr(obj, "shocks", None) or getattr(obj, "exogenous", ()))
        guess = dict(getattr(obj, "guess", None) or {})
        if measurement is None:
            varobs = getattr(obj, "varobs", ())
            measurement = getattr(obj, "measurement", None) or (
                MeasurementMap.direct(list(varobs), dict(getattr(obj, "measurement_sd", {}) or {})) if varobs else None)
        shock_sd = {**dict(getattr(obj, "shock_sd", {}) or {}), **dict(shock_sd or {})}
    if not names:
        raise ModelError("name the model's shocks (shocks=[...]) or give them Role.SHOCK")
    if measurement is None:
        raise ModelError("give measurement equations (a MeasurementMap) or the model variables observed")
    if not isinstance(measurement, MeasurementMap):
        measurement = MeasurementMap.direct(list(measurement))
    return model, names, guess, measurement, dict(shock_sd or {})


def piecewise_filter(pw: Any, system: Any, y: np.ndarray, start: str = "auto", presample: int = 0,
                     regimes: Sequence[np.ndarray] | None = None) -> dict[str, Any]:
    """The OccBin filter on observations ``y`` (periods x observables, in data units less the measurement
    constants), ``pw`` the model's piecewise solution (``occbin._Piecewise``) and ``system`` the reference regime's
    state-space form: the log likelihood, the filtered and predicted states, the period rules and the regimes.
    ``regimes`` (one horizon x constraints array a period, as ``"regimes"`` in the result) fixes each period's
    expected regimes instead of working them out from the filtered state: the Kalman filter with those rules."""
    k, m = pw.n, len(pw.names)
    a0, P0, D = system.start(start)
    if D.any():
        raise ModelError("the OccBin filter is for stationary models: the reference regime has unit roots")
    T0, R0, Qc, Z, d, H = system.T, system.R, system.Q, system.Z, system.d, system.H
    ns, nT = T0.shape[0], len(y)
    a_filt, P_filt = np.empty((nT, ns)), np.empty((nT, ns, ns))
    a_pred, P_pred = np.empty((nT, ns)), np.empty((nT, ns, ns))
    Ts, cs, Rs = np.empty((nT, ns, ns)), np.empty((nT, ns)), np.empty((nT, ns, m))
    binds_used = np.zeros((nT, len(pw.constraints)), dtype=bool)
    expected: list[np.ndarray] = []  # each period's regimes, that period and those expected after it
    loglike, most = 0.0, 0
    a_prev, P_prev = np.asarray(a0, float), np.asarray(P0, float)
    zero = np.zeros((1, m))
    guess = None
    cycles = 0

    def step(binds: np.ndarray, a_prev: np.ndarray, P_prev: np.ndarray, t: int) -> dict[str, Any]:
        """Period t filtered with the regimes ``binds``: its rule, prediction, update, shock estimate and density."""
        Ps, Ds, Q1 = pw.rules(binds, zero, offset=t)
        T, R, c = T0.copy(), R0.copy(), np.zeros(ns)
        T[:k, :k], R[:k], c[:k] = Ps[0], Q1, Ds[0]
        ap = c + T @ a_prev
        Pp = T @ P_prev @ T.T + R @ Qc @ R.T
        obs = np.isfinite(y[t])
        out = {"T": T, "R": R, "c": c, "ap": ap, "Pp": Pp, "af": ap, "Pf": Pp, "e": np.zeros(m), "back": a_prev,
               "logl": 0.0, "observed": obs.any(), "impossible": ""}
        if obs.any():
            Zt = Z[obs]
            v = y[t, obs] - d[obs] - Zt @ ap
            F = Zt @ Pp @ Zt.T + H[np.ix_(obs, obs)]
            try:
                Fi, logdet, rank = _pinv_logdet(0.5 * (F + F.T), v, t)
            except ModelError as exc:  # the data are impossible under these regimes
                out.update(logl=-np.inf, impossible=str(exc))
                return out
            Fi_v = Fi @ v
            K_rows = Pp @ Zt.T
            out.update(af=ap + K_rows @ Fi_v, Pf=Pp - K_rows @ Fi @ K_rows.T,
                       e=Qc @ R.T @ Zt.T @ Fi_v,  # E[e(t) | data to t]: e(t) is independent of x(t-1)
                       back=a_prev + P_prev @ T.T @ Zt.T @ Fi_v,  # E[x(t-1) | data to t]: with e(t), the rule gives af
                       logl=-0.5 * (rank * np.log(2 * np.pi) + logdet + v @ Fi_v))
        return out

    for t in range(nT):
        if regimes is not None:  # given: the period filtered once with them
            binds, iteration = np.asarray(regimes[t], dtype=bool), 1
            result = step(binds, a_prev, P_prev, t)
        else:
            binds, result, iteration, cycled = _settle(pw, step, a_prev, P_prev, t, guess, zero, k)
            cycles += cycled
        if result["impossible"]:
            exc = ModelError(result["impossible"])
            exc.period = t  # type: ignore[attr-defined]
            raise exc
        most = max(most, iteration)
        if result["observed"] and t >= presample:
            loglike += result["logl"]
        a_pred[t], P_pred[t], a_filt[t] = result["ap"], result["Pp"], result["af"]
        P_filt[t] = 0.5 * (result["Pf"] + result["Pf"].T)
        Ts[t], cs[t], Rs[t] = result["T"], result["c"], result["R"]
        binds_used[t] = binds[0]
        expected.append(binds)
        guess = np.vstack([binds[1:], binds[-1:] & False])  # next period's first guess: this one's, a period on
        a_prev, P_prev = a_filt[t], P_filt[t]
    return {"loglike": float(loglike), "a_filt": a_filt, "P_filt": P_filt, "a_pred": a_pred, "P_pred": P_pred, "T": Ts,
            "c": cs, "R": Rs, "binds": binds_used, "iterations": most, "cycles": cycles, "regimes": expected, "a0": np.asarray(a0, float),
            "P0": np.asarray(P0, float), "Q": Qc}


def _settle(pw: Any, step: Any, a_prev: np.ndarray, P_prev: np.ndarray, t: int, guess: np.ndarray | None, zero: np.ndarray,
            k: int) -> tuple[np.ndarray, dict[str, Any], int, int]:
    """Period t's regimes, worked out from the filtered state: filtered with a guess, OccBin solved again from the
    estimates of the state and shock, until the regimes no longer change (or cycle: the likelier is taken)."""
    # first guess: last period's regimes, a period on (at the start, the regimes expected with no shock)
    binds = guess if guess is not None else pw.solve_from(a_prev[:k], zero, offset=t)[2]
    tried: dict[bytes, tuple[np.ndarray, dict[str, Any]]] = {}
    for iteration in range(1, _REGIME_ITERATIONS + 1):
        result = step(binds, a_prev, P_prev, t)
        tried[binds.tobytes()] = (binds, result)
        _, _, new = pw.solve_from(result["back"][:k], result["e"][None, :], binds, offset=t)
        if np.array_equal(new, binds):
            return binds, result, iteration, 0
        if new.tobytes() in tried:  # the guesses cycle: take the regimes under which the period is likeliest
            binds, result = max(tried.values(), key=lambda item: item[1]["logl"])
            return binds, result, iteration, 1
        binds = new
    raise ModelError(f"the regimes do not settle in period {t + 1}: the filter cannot place the constraint there")


def _pinv_logdet(F: np.ndarray, v: np.ndarray, t: int) -> tuple[np.ndarray, float, int]:
    """The prediction-error variance's pseudo-inverse, log pseudo-determinant and rank. At a binding constraint an
    observed variable can be fixed (a policy rate at its floor, observed without error): its variance is zero, and the
    density is the normal on the other directions, the prediction error in the fixed ones having to be zero."""
    w, V = np.linalg.eigh(F)
    tol = max(F.shape) * np.finfo(float).eps * max(1.0, float(np.max(np.abs(w)))) * 1e4
    keep = w > tol
    if not keep.all():
        off = V[:, ~keep].T @ v
        if np.max(np.abs(off)) > 1e-6 * max(1.0, float(np.max(np.abs(v)))):
            raise ModelError(f"period {t + 1}: an observation the model fixes (no shock or measurement error moves it, as a "
                             "rate at its floor) is not where the model puts it; give that series a measurement error")
    Vk = V[:, keep]
    return Vk @ np.diag(1.0 / w[keep]) @ Vk.T, float(np.sum(np.log(w[keep]))), int(keep.sum())


def piecewise_smoother(run: Mapping[str, Any], Qc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The smoothed states (Rauch-Tung-Striebel through the filter's time-varying rules) and the smoothed shocks,
    e(t) from R(t) e(t) = a(t|n) - c(t) - T(t) a(t-1|n), with the pre-sample state smoothed the same way (with
    correlated shocks, the likeliest e(t) that gives the gap, ``Q R' (R Q R')^+ gap``)."""
    a_s, shocks, _ = _smooth(run)
    return a_s, shocks


def _smooth(run: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``piecewise_smoother``, with the smoothed pre-sample state too."""
    a_f, P_f, a_p, P_p, T, c, R = (run[k] for k in ("a_filt", "P_filt", "a_pred", "P_pred", "T", "c", "R"))
    nT = len(a_f)
    a_s = a_f.copy()
    for t in range(nT - 2, -1, -1):
        J = P_f[t] @ T[t + 1].T @ np.linalg.pinv(P_p[t + 1])
        a_s[t] = a_f[t] + J @ (a_s[t + 1] - a_p[t + 1])
    J0 = run["P0"] @ T[0].T @ np.linalg.pinv(P_p[0])
    before = run["a0"] + J0 @ (a_s[0] - a_p[0])
    prev = np.vstack([before, a_s[:-1]])
    shocks = np.empty((nT, R.shape[2]))
    Q = run.get("Q")
    correlated = Q is not None and _correlated(Q)
    for t in range(nT):
        gap = a_s[t] - c[t] - T[t] @ prev[t]
        if correlated:  # the shocks' mean given the gap: lstsq's least norm is not the likeliest with correlations
            shocks[t] = Q @ R[t].T @ np.linalg.pinv(R[t] @ Q @ R[t].T) @ gap
        else:
            shocks[t] = np.linalg.lstsq(R[t], gap, rcond=None)[0]
    return a_s, shocks, before


def _correlated(Q: np.ndarray) -> bool:
    """Whether a shock covariance has anything off its diagonal."""
    return bool(np.any(Q - np.diag(np.diag(Q))))


def _place_muted(pw: Any, run: Mapping[str, Any], a_s: np.ndarray, e_s: np.ndarray, before: np.ndarray, sd: np.ndarray,
                 Q: np.ndarray | None = None) -> np.ndarray:
    """The smoothed shocks, with those a binding regime mutes placed where the regime holds. At a binding constraint
    some shocks do nothing (a policy shock with the rate at its floor): the data say nothing of them, and the smoother
    leaves them at zero, which can put the rule above the floor although the regime has it below. In such a period
    they are moved, in the directions that leave the smoothed variables unchanged, by the least (in standard
    deviations) that puts the constraint's branches level: the nearest values to zero consistent with the regime
    (the mode of their distribution given the data and the regime). With correlated shocks (``Q``, off its diagonal)
    the distance is ``z'N'Q^-1 N z``: the same mode, the shocks' covariance in full."""
    k = pw.n
    e_s = e_s.copy()
    full = np.vstack([before[:k], a_s[:, :k]])
    binds = run["binds"]
    weight = np.where(sd > 0, sd, 1.0)
    precision = np.linalg.pinv(Q) if Q is not None and _correlated(Q) else np.diag(weight**-2.0)
    for t in range(len(a_s)):
        if not binds[t].any():
            continue
        R = run["R"][t]
        _, sv, Vt = np.linalg.svd(R)
        rank = int(np.sum(sv > 1e-10 * max(1.0, float(sv.max()) if sv.size else 1.0)))
        N = Vt[rank:].T  # shocks that leave every state unchanged in this period
        if N.shape[1] == 0:
            continue
        for _ in range(3):  # the branches are linear in the shocks for linear rules: one step, then checked
            E = e_s[: t + 1]
            wrong = pw.binding(full[: t + 2], E, tie=True)[t] != binds[t]
            if not wrong.any():
                break
            rows = np.flatnonzero(wrong)
            g0 = pw.branches(full[: t + 2], E, t)[rows]
            G = np.empty((len(rows), N.shape[1]))
            for c in range(N.shape[1]):
                bumped = e_s[: t + 1].copy()
                bumped[t] += N[:, c]
                G[:, c] = pw.branches(full[: t + 2], bumped, t)[rows] - g0
            if np.linalg.matrix_rank(G) < len(rows):
                break  # the muted shocks do not move this constraint: nothing to place
            W = np.linalg.inv(N.T @ precision @ N)  # least change in standard deviations: z'N'S^-2 N z
            z = -W @ G.T @ np.linalg.solve(G @ W @ G.T, g0)
            e_s[t] = e_s[t] + N @ z
    return e_s


def iterated_smoother(pw: Any, system: Any, y: np.ndarray, run: Mapping[str, Any], start: str = "auto", presample: int = 0,
                      max_iter: int = _REGIME_ITERATIONS) -> dict[str, Any]:
    """The OccBin smoother iterated on the smoothed path, as Dynare 6's OccBin smoother (Giovannini, Pfeiffer and
    Ratto 2021): from the filter's run, each period's regimes are worked out again by OccBin from the smoothed state
    of the period before and the smoothed shock (the regimes of that period and those expected after it), the
    Kalman filter and smoother run again with those rules, and so on until the regimes no longer change. If they
    cycle, the regimes the smoothed path contradicts least are kept (``cycled``). Returns the last run with the
    smoothed states, shocks and pre-sample state, the iterations and whether it cycled."""
    k = pw.n
    sd = np.sqrt(np.clip(np.diag(system.Q), 0.0, None))
    regimes = [np.asarray(r, dtype=bool) for r in run["regimes"]]
    pinned: set[int] = set()  # periods whose regimes from the smoothed path make the data impossible: kept as they were
    seen: dict[bytes, dict[str, Any]] = {}
    for iteration in range(1, max_iter + 1):
        a_s, e_s, before = _smooth(run)
        e_s = _place_muted(pw, run, a_s, e_s, before, sd, system.Q)
        prev = np.vstack([before, a_s[:-1]])
        new = []
        for t in range(len(a_s)):
            try:
                new.append(regimes[t] if t in pinned else pw.solve_from(prev[t, :k], e_s[t][None, :], regimes[t], offset=t)[2])
            except ModelError:  # no regimes settle from this state and shock: keep the period's
                new.append(regimes[t])
        ex_post = pw.binding(np.vstack([before[:k], a_s[:, :k]]), e_s, tie=True)
        state = {**run, "a_s": a_s, "e_s": e_s, "before": before, "iterations": iteration, "cycled": False, "pinned": len(pinned),
                 "contradicted": int((ex_post != run["binds"]).any(axis=1).sum())}
        if all(np.array_equal(a, b) for a, b in zip(new, regimes)):
            return state
        key = b"".join(r.tobytes() for r in new)
        seen[b"".join(r.tobytes() for r in regimes)] = state
        if key in seen:  # the regimes cycle: keep those the smoothed path contradicts in the fewest periods
            best = min(seen.values(), key=lambda s: (s["contradicted"], -s["loglike"]))
            return {**best, "iterations": iteration, "cycled": True}
        while True:
            try:
                run = piecewise_filter(pw, system, y, start, presample, regimes=new)
                break
            except ModelError as exc:
                t = getattr(exc, "period", None)
                if t is None or t in pinned:
                    raise
                pinned.add(t)
                new[t] = regimes[t]
        regimes = new
    raise ModelError(f"the smoother's regimes do not settle in {max_iter} iterations")


@api("pymetrix:occbin_filter")
def occbin_filter(obj: Any, data: Any, measurement: Any = None, shock_sd: Mapping[str, float] | None = None, span: Any = None,
                  start: str = "auto", horizon: int = 40, presample: int = 0, shocks: Sequence[str] | None = None,
                  smoother: str = "filter", regimes: Any = None, shock_corr: Mapping[Any, float] | None = None) -> OccBinFilter:
    """The OccBin filter and smoother of ``obj`` (a template, a Dynare or IRIS model, a fitted model, or a Model with
    ``measurement`` and ``shocks``) on ``data``: see the module docstring. ``shock_sd`` gives shock sizes (default:
    the object's); ``horizon`` how far ahead each period's regimes are worked out; ``presample`` periods are
    filtered but left out of the likelihood. ``smoother``: ``"filter"`` smooths with the regimes the filter chose;
    ``"iterated"`` works them out again on the smoothed path until they settle, as Dynare 6's OccBin smoother.
    ``regimes``: known-date regimes in the sample (``pymetrix.models.regimes``; default: a fitted model's own, or a
    template's that start by the sample's end), combined with the constraints: each period's matrices are those of its
    known regime and of the constraints' regime. ``shock_corr``: the shocks' correlations (default: the object's),
    the filter's shock covariance in full."""
    from ..data import Dataset
    from .estimation import FittedModel
    from .occbin import _Piecewise
    from .regimes import as_regimes, check_starts

    if smoother not in ("filter", "iterated"):
        raise ModelError(f"smoother is 'filter' (the filter's regimes) or 'iterated' (worked out on the smoothed path), not {smoother!r}")
    model, names, guess, measurement, sds = _setup(obj, measurement, shock_sd, shocks)
    if isinstance(obj, FittedModel):
        regimes = obj.regimes if regimes is None else (*obj.regimes, *as_regimes(regimes))
        corr = obj._corr if shock_corr is None else shock_corr
    else:
        if regimes is None and getattr(obj, "regimes", None):  # a template's own, those that start by the sample's end
            from .templates import _within

            regimes = _within(obj.regimes, data, span)[0] or None
        corr = getattr(obj, "shock_corr", None) if shock_corr is None else shock_corr
    regimes = as_regimes(regimes)
    first = freq = None
    periods = 1
    if regimes:
        index = data.frame.index[data.sample_mask(span)]
        index = index if isinstance(index, pd.PeriodIndex) else pd.PeriodIndex(index)
        check_starts(regimes, index, "sample")
        first, freq, periods = index[0], index.freq, len(index)
    pw = _Piecewise(model, names, guess, int(horizon), regimes=regimes, first=first, freq=freq, length=periods)
    pw.tie = True
    system = pw.solution.system(measurement, sds, shock_corr=corr or None)
    if system.trend is not None and np.any(system.trend) or getattr(measurement, "trend_exprs", None):
        raise ModelError("the OccBin filter is for models without growth trends or deterministic trends in the measurement")
    y, index = system.observations(data, span)
    run = piecewise_filter(pw, system, y, start, presample)
    if smoother == "iterated":
        final = iterated_smoother(pw, system, y, run, start, presample)
    else:
        a_s, e_s, before = _smooth(run)
        final = {**run, "a_s": a_s, "e_s": e_s, "before": before, "iterations": 0, "cycled": False, "pinned": 0}
    a_s, e_s = final["a_s"], final["e_s"]
    k = pw.n
    observed = list(pw.base.observed)
    keep = [pw.index[v] for v in observed]
    levels = np.array([pw.steady[v] for v in observed])
    frame = pd.DataFrame(a_s[:, keep] + levels, index=index, columns=observed)
    fitted = system.d + a_s @ system.Z.T
    for j, o in enumerate(system.observables):
        frame[f"{o}_fitted"] = fitted[:, j]
    for j, s in enumerate(names):
        frame[s] = e_s[:, j]
    if system.measurement is not None:
        frame = system.measurement.with_reported(frame)
    filtered = pd.DataFrame(run["a_filt"][:, keep] + levels, index=index, columns=observed)
    before = final["before"][:k]
    ex_post = pw.binding(np.vstack([before, a_s[:, :k]]), e_s, tie=True)
    labels = [c.label for c in pw.constraints]
    rules = {"T": final["T"][:, :k, :k], "c": final["c"][:, :k], "R": final["R"][:, :k], "a_s": a_s[:, :k], "e_s": e_s,
             "variables": tuple(pw.variables), "observed": tuple(observed), "shocks": tuple(names), "steady": dict(pw.steady)}
    if _correlated(system.Q):
        from .covariance import shock_factor

        rules["factor"] = shock_factor(names, np.array([float(system.shock_sd[s]) for s in names]), corr)
    return OccBinFilter(Dataset.from_frame(frame), filtered, pd.DataFrame(run["binds"], index=index, columns=labels),
                        pd.DataFrame(ex_post, index=index, columns=labels), run["loglike"], tuple(labels),
                        int(run["iterations"]), tuple(system.observables), int(run["cycles"]), smoother,
                        pd.DataFrame(final["binds"], index=index, columns=labels), int(final["iterations"]), bool(final["cycled"]),
                        int(final["pinned"]), rules)


def occbin_forecast(model: Any, shocks: Sequence[str], guess: Mapping[str, float], smoothed: Any, span: Any, scenario: Any = None,
                    draws: int | None = None, shock_sd: Mapping[str, float] | None = None, seed: int | None = None,
                    horizon: int = 60, regimes: Any = None, shock_corr: Mapping[Any, float] | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame, Any, pd.DataFrame]:
    """Piecewise-linear forecasts over ``span`` from the state in ``smoothed`` (levels) in the period before it: the
    model's variables in levels, when each constraint binds, with ``draws`` the draws (``Simulations``), and the
    shocks (periods x shocks, deviations). ``scenario`` gives shock paths (``set``, ``shift``): anticipated, unless all
    are named in ``unanticipated``; and targets held by free shocks (``hold``, ``free``), anticipated: the free shocks
    are solved for through the piecewise solution (``_hold``). With draws, each draw adds shocks drawn in each period
    with ``shock_sd`` and correlations ``shock_corr`` (through the Cholesky factor of their covariance), as surprises,
    the scenario's included. ``regimes``: known-date regimes (those before the span set the reference, the regime the
    starting state is in), combined with the constraints."""
    from .covariance import normalise, shock_factor
    from .occbin import _Piecewise, _simulate
    from .plans import _path, _periods, _state_of
    from .stochastic import Simulations, _rng

    sc = scenario
    if sc is not None and (sc.fix or sc.adjust or getattr(sc, "hold_sd", None) or getattr(sc, "free_sd", None)
                           or getattr(sc, "ranges", False)):
        raise ModelError("OccBin forecasts take shock paths (set=, shift=) and targets held by free shocks (hold=, free=); "
                         "fix=, adjust= and soft targets are for linear plans: forecast with occbin=False for them")
    holding = sc is not None and bool(sc.hold or sc.free)
    if holding and sc.unanticipated:
        raise ModelError("OccBin forecasts hold targets with anticipated shocks only (the whole path known from the first "
                         "period): leave unanticipated empty, or forecast with occbin=False")
    if holding and draws:
        raise ModelError("OccBin forecasts with targets held have no draws: the free shocks are solved for one path")
    frame = smoothed.frame
    periods = _periods(span, frame.index.freq)
    pw = _Piecewise(model, tuple(shocks), guess, horizon, regimes=regimes, first=periods[0], freq=periods.freq, length=len(periods))
    names = pw.names
    before = periods[0] - 1
    if before not in frame.index:
        raise ModelError(f"the forecast starts the period after the data end: {frame.index[-1] + 1}, not {periods[0]}")
    observed = list(pw.base.observed)
    x0 = np.zeros(pw.n)
    for j, v in enumerate(pw.variables):
        base, lag = _state_of(v, observed)
        if base is None:
            continue
        at = before - lag
        value = frame[base].get(at, np.nan) if base in frame.columns else np.nan
        if not np.isfinite(value):
            raise ModelError(f"the forecast needs {base} in {at}, from the smoothed data")
        x0[j] = float(value) - pw.steady[base]
    E = np.zeros((len(periods), len(names)))
    for k, s in enumerate(names):
        if sc is not None and s in sc.set:
            p = _path(sc.set[s], periods)
            E[:, k] = np.where(np.isfinite(p), p - pw.steady.get(s, 0.0), 0.0)
        if sc is not None and s in sc.shift:
            E[:, k] += np.nan_to_num(_path(sc.shift[s], periods))
    surprises = set(sc.unanticipated) if sc is not None else set()
    given = {s for k, s in enumerate(names) if np.any(E[:, k])}
    if surprises and given - surprises:
        raise ModelError("OccBin forecasts take the scenario's shocks all anticipated or all unanticipated: name all of "
                         f"{', '.join(sorted(given))} in unanticipated, or none")
    keep = [pw.index[v] for v in observed]
    levels = np.array([pw.steady[v] for v in observed])
    if holding:
        E, dev = _hold(pw, x0, E, sc, periods, observed)
    else:
        dev, _ = _simulate(pw, x0, E, anticipated=not surprises)
    paths = pd.DataFrame(dev[:, keep] + levels, index=periods, columns=observed)
    binding = pd.DataFrame(pw.binding(np.vstack([x0, dev]), E),
                           index=periods, columns=[c.label for c in pw.constraints])
    simulation = None
    if draws:
        sd = np.array([float((shock_sd or {}).get(s, 0.0)) for s in names])
        factor = shock_factor(names, sd, shock_corr)  # None: independent shocks, drawn as they always were
        rng = _rng(seed)
        values = np.empty((int(draws), len(periods), len(observed)))
        failed = 0
        for d in range(int(draws)):
            drawn = E + (rng.standard_normal(E.shape) * sd if factor is None else rng.standard_normal(E.shape) @ factor.T)
            try:
                values[d] = _simulate(pw, x0, drawn, anticipated=False)[0][:, keep] + levels
            except ModelError:  # a draw whose regimes do not settle: left out
                values[d] = np.nan
                failed += 1
        info = {"method": "OccBin: shocks drawn in each period, as surprises (the scenario's too)", "draws": int(draws), "seed": seed,
                "shock_sd": {s: float(v) for s, v in zip(names, sd) if v > 0}, "failed": failed}
        if factor is not None:
            info["shock_corr"] = {f"{a},{b}": r for (a, b), r in normalise(names, shock_corr).items() if r != 0}
        ok = ~np.isnan(values).any(axis=(1, 2))
        simulation = Simulations(values[ok], tuple(observed), periods, info)
    return paths, binding, simulation, pd.DataFrame(E, index=periods, columns=list(names))


def _hold(pw: Any, x0: np.ndarray, E: np.ndarray, sc: Any, periods: pd.PeriodIndex, observed: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Targets held by free shocks, all anticipated, through the piecewise solution: Newton's method on the free
    shocks. With the regimes fixed the path is linear in the shocks, so each step solves the targets exactly under
    the current regimes; OccBin then works out the regimes of the new shock path, and the steps go on until they no
    longer change. The shocks (periods x shocks) and the path (periods x variables, deviations)."""
    from .plans import _path

    names = pw.names
    if pw.arrivals:
        raise ModelError("OccBin forecasts hold targets with regimes known in advance: a surprise regime (expected=False) "
                         "makes the path be solved again when it starts; forecast without targets, or with the regime expected")
    bad = [v for v in sc.hold if v not in observed]
    if bad:
        raise ModelError(f"hold=: {', '.join(bad)} are not variables of the model ({', '.join(observed)})")
    bad = [s for s in sc.free if s not in names]
    if bad:
        raise ModelError(f"free=: {', '.join(bad)} are not shocks of the model ({', '.join(names)})")
    if len(sc.free) != len(sc.hold):
        raise ModelError(f"{len(sc.hold)} held variables but {len(sc.free)} free shocks; pair each hold= with one free= shock, in order")
    if len(periods) > pw.H:
        raise ModelError(f"the forecast is longer than the OccBin horizon ({pw.H} periods)")
    conditions, unknowns = [], []
    for (v, values), s in zip(sc.hold.items(), sc.free):
        p = _path(values, periods)
        for t in np.flatnonzero(np.isfinite(p)):
            conditions.append((int(t), pw.index[v], float(p[t]) - pw.steady[v]))
            unknowns.append((int(t), names.index(s)))
    T = len(periods)
    binds = np.zeros((pw.H, len(pw.constraints)), dtype=bool)
    seen: set[bytes] = set()
    for _ in range(pw.max_iter):
        base = pw.path(x0, E, binds)
        M = np.empty((len(conditions), len(unknowns)))
        for c, (t, k) in enumerate(unknowns):  # under fixed regimes the path is linear in the shocks: exact columns
            bumped = E.copy()
            bumped[t, k] += 1.0
            response = pw.path(x0, bumped, binds) - base
            M[:, c] = [response[tt, j] for tt, j, _ in conditions]
        if np.linalg.matrix_rank(M) < len(unknowns):
            raise ModelError("the free shocks cannot hold the targets under the regimes OccBin finds: a shock does nothing "
                             "while a constraint binds (a policy shock with the rate at its floor), or does not move its "
                             "target; hold another variable, free another shock, or forecast with occbin=False")
        gap = np.array([value - base[t, j] for t, j, value in conditions])
        E = E.copy()
        for (t, k), z in zip(unknowns, np.linalg.solve(M, gap)):
            E[t, k] += z
        X = pw.path(x0, E, binds)
        new = pw.check(X, x0, E)
        if np.array_equal(new, binds):
            if new[-1].any():
                raise ModelError(f"the constraint still binds at the end of the horizon ({pw.H} periods): give a longer horizon")
            return E, X[:T]
        if new.tobytes() in seen:
            raise ModelError("holding the targets, the regimes cycle without settling: no OccBin path holds them")
        seen.add(new.tobytes())
        binds = new
    raise ModelError(f"holding the targets, the regimes do not settle in {pw.max_iter} iterations")
