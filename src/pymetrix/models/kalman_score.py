"""The analytic gradient (score) of a linearised model's Kalman-filter log likelihood in its estimated parameters.

    fit = model.estimate(data, measurement, priors=...)       # BFGS with this gradient; the Hessian from it
    fit.gradient()                                            # d log likelihood (and log posterior) / d parameter
    post = fit.mcmc(draws=1000, sampler="nuts")               # NUTS with this gradient (gradient="numerical": as before)

The log likelihood of a first-order solution goes through four maps, each differentiated exactly here: the first
three for every estimated parameter at once (forward mode), the filter backwards (reverse mode):

1. **The steady state.** The static residuals ``f(ybar, theta) = 0`` hold at every parameter value, so
   ``d ybar / d theta = -f_y^-1 f_theta`` (the implicit-function theorem), with ``f_y`` the static Jacobian and
   ``f_theta`` the residuals' derivatives in the parameters, both from the equations' symbolic derivatives.
2. **The linearisation** ``A0 E x(t+1) + A1 x(t) + A2 x(t-1) + B e(t) = 0``: each entry is a first derivative of a
   residual evaluated at the steady state, so its derivative is the second derivative in the parameter plus the second
   derivatives in each variable times that variable's steady-state derivative (symbolic second derivatives, compiled
   once per model and shared by every trial value).
3. **The solution** ``x(t) = P x(t-1) + Q e(t)``. ``P`` solves ``A0 P^2 + A1 P + A2 = 0``; differentiating,
   ``(A0 P + A1) dP + A0 dP P = -(dA0 P^2 + dA1 P + dA2)`` (Iskrev 2010; Ratto and Iskrev 2011), a generalised
   Sylvester equation in the columns of ``P`` that are not zero (the lagged variables). It is solved column by column
   on the complex Schur form of ``P``'s lagged block: each column one linear system with ``I + s_jj (A0 P + A1)^-1 A0``,
   which is regular exactly when the Blanchard-Kahn conditions hold (the model's other roots are outside the unit
   circle). Then ``dQ = -(A0 P + A1)^-1 (dB + (dA0 P + A0 dP + dA1) Q)``.
4. **The Kalman filter** on the state-space form (lagged copies of the states where the measurement equations have
   lags), differentiated in reverse mode: the filter runs forwards (``_filter``: the variance recursion, which does not
   depend on the data, one *state* per period in which the variance is updated, then the means), then its adjoint runs
   backwards (``_adjoint``: the means' adjoint over periods, ``abar(t) = L' abar(t+1) + Z' F^-1 v``, then the variance's
   over states, ``Pbar(s) = L' Pbar(s+1) L + C(s)`` with ``L = T (I - K Z)``), giving the likelihood's derivatives in
   every entry of ``T``, ``Z``, ``d``, ``H``, ``R Q R'`` and the initial variance at the cost of a few small matrix
   products a period, whatever the number of parameters. Those are then contracted with each parameter's derivatives
   of the system matrices (the measurement matrices' through the steady state, from the second derivatives of the
   measurement equations; the shocks' covariance ``D C D`` in their standard deviations and correlations; the
   measurement errors' variances). The stationary initial variance ``P0 = T P0 T' + R Q R'`` passes its adjoint on
   through one more Lyapunov equation, ``X = T' X T + P0bar``; the scaled start is constant. Where the filter stops
   updating the variance (it has converged, as Dynare's ``riccati_tol``) the periods after reuse the converged state,
   so the gradient is that of the likelihood as computed, exactly; missing observations are the full-size recursion
   with their rows of ``Z`` set to zero (the same likelihood). The backward variance recursion contracts (``L`` is the
   stable closed-loop transition): the forward-mode recursion it replaced (``dP`` carried through the non-Joseph update)
   amplified rounding errors and overflowed on Smets and Wouters (2007) after about a hundred periods.

The cost, measured (``docs/validation.md``): a gradient for 7 parameters of the New Keynesian template about 2 log
posterior evaluations (6.7 ms against 3.3 ms), and for the 36 of Smets and Wouters (2007) about 3.5 (41 ms against 12
ms), against ``2k + 1`` for central differences, and exact rather than accurate to about 1e-7. The pure-Python parts are
kept small: the filter calls LAPACK's Cholesky factorisation and triangular inverse directly, every product that does
not need the loop is done for all periods at once, and the model's compiled derivatives share their common
subexpressions (``pymetrix.expr.compile``). With numba installed (``pymetrix[fast]``) the filter's and the adjoint's loops over
periods run compiled (``pymetrix.models._kalman_compiled``; equal to these NumPy loops to 1e-10), about halving the
gradient's time on small models. Optimisers then use it (BFGS on the transformed parameters, with the chain
rule through the transform), the Hessian at the mode is the central difference of the gradient (``2k`` gradients,
accurate to about 1e-8 where differences of the likelihood itself give about 1e-4), and NUTS takes it for every
leapfrog step.

Not covered, where the finite-difference gradient is used as before: known-date regimes, the OccBin filter, the
particle filter (its likelihood is an estimate), balanced-growth models, filters with diffuse states (unit roots, or
``start="diffuse"``), equations that cannot be compiled, and a steady state the static equations do not pin down
(singular static Jacobian). ``supported`` says why for a given estimate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import linalg
from scipy.linalg import lapack

from ..errors import PyMetrixError
from ..expr import Name, diff
from ..expr.compile import compile_exprs
from ..expr.nodes import Coef
from .solvers import ModelError

__all__ = ["NoGradient", "loglike_score", "supported"]

_LOG2PI = float(np.log(2 * np.pi))
_ZERO = "0"


class NoGradient(PyMetrixError):
    """The analytic gradient does not cover this likelihood (the message says why): use finite differences. Internal:
    callers fall back, or ``ready`` turns it into a ``ModelError`` when the analytic gradient was asked for."""


def supported(lik: Any) -> str | None:
    """Why the analytic gradient does not cover this estimate's likelihood, or None when it does (checked before any
    evaluation: the steady state and the filter's start are checked at each one)."""
    from .estimation import _Likelihood

    if type(lik).loglike is not _Likelihood.loglike:  # a DSGE-VAR's p(Y | theta, lambda), say: not the Kalman filter's
        return "this likelihood is not the Kalman filter's: it has no analytic gradient here"
    if getattr(lik, "particle", None):
        return "the particle filter's likelihood is an estimate (noisy): it has no analytic gradient"
    if getattr(lik, "occbin", False):
        return "the OccBin filter's likelihood is piecewise: it has no analytic gradient here"
    if getattr(lik, "regimes", ()):
        return "known-date regimes are not covered by the analytic gradient"
    if getattr(lik, "growth", False):
        return "balanced-growth models are not covered by the analytic gradient"
    if getattr(lik, "start", "auto") == "diffuse":
        return "a diffuse start is not covered by the analytic gradient"
    if any("@" in n for n in lik.free):
        return "regimes' values are not covered by the analytic gradient"
    return None


# --- symbolic second derivatives, compiled once per model ------------------------------------------------------------

def _second(m: Any, fns: Any) -> dict[str, Any]:
    """The compiled second derivatives of the model's residuals (each first derivative of the linearisation in each
    variable and parameter) and the static residuals' derivatives in the parameters, kept with the model's compiled
    functions (shared by its ``with_parameters`` copies)."""
    if "kalman score" in fns._cache:  # the model's compiled functions are shared by its with_parameters copies
        return fns._cache["kalman score"]
    from .fast import _static

    exprs, plan = [], []
    for e, (_, d) in enumerate(fns._derivatives):
        for ref in sorted(d.refs(), key=lambda r: (r.id, r.lag)):
            dd = diff(d, Name(ref.id, ref.lag))
            if str(dd) != _ZERO:
                plan.append((e, "v", ref.id))
                exprs.append(dd)
        for c in sorted(d.params()):
            dd = diff(d, Coef(c))
            if str(dd) != _ZERO:
                plan.append((e, "c", c))
                exprs.append(dd)
    s_exprs, s_plan = [], []
    for i, r in enumerate(fns.residual_exprs):
        s = _static(r)
        for c in sorted(s.params()):
            dd = diff(s, Coef(c))
            if str(dd) != _ZERO:
                s_plan.append((i, c))
                s_exprs.append(dd)
    out = {"second": compile_exprs(tuple(exprs), fns.coefs), "plan": tuple(plan),
           "static": compile_exprs(tuple(s_exprs), fns.coefs), "static_plan": tuple(s_plan)}
    fns._cache["kalman score"] = out
    return out


def _slots(compiled: Any, values: Mapping[str, float]) -> list[float]:
    return [float(values[name]) for name, _ in compiled.refs]


# --- the pieces ----------------------------------------------------------------------------------------------------

def _structural(m: Any, lin: Any, structural: list[str], K: int, columns: Mapping[str, int]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """dA0, dA1, dA2 (K x n x n), dB (K x n x shocks) and d ybar (endogenous x K), and the endogenous' positions."""
    from .fast import functions, parameter_values

    fns = functions(m)
    if fns is None:
        raise NoGradient("the model's equations cannot be compiled")
    ss = lin.steady
    if getattr(ss, "growth", None):
        raise NoGradient("balanced-growth models are not covered by the analytic gradient")
    if lin.c is not None:
        raise NoGradient("equations with a constant term are not covered")
    values = dict(ss.values)
    endo = list(m.endogenous)
    where = {u: j for j, u in enumerate(endo)}
    p = parameter_values(m, fns)
    sec = _second(m, fns)
    n_eq = len(fns.residual_exprs)
    Js = np.zeros((n_eq, len(endo)))
    for (i, j), d in zip(fns.static_pairs, fns.static_jacobian(_slots(fns.static_jacobian, values), p)):
        Js[i, j] = float(d)
    Ft = np.zeros((n_eq, K))
    if sec["static_plan"]:
        for (i, c), v in zip(sec["static_plan"], sec["static"](_slots(sec["static"], values), p)):
            if c in columns:
                Ft[i, columns[c]] += float(v)
    if Js.shape[0] != Js.shape[1] or np.linalg.cond(Js) > 1e12:
        raise NoGradient("the static equations do not pin down the steady state (singular Jacobian)")
    dybar = -np.linalg.solve(Js, Ft)  # endogenous x K
    entries = fns._derivatives
    dE = np.zeros((len(entries), K))
    if sec["plan"]:
        for (e, kind, what), v in zip(sec["plan"], sec["second"](_slots(sec["second"], values), p)):
            v = float(v)
            if kind == "c":
                if what in columns:
                    dE[e, columns[what]] += v
            elif what in where:
                dE[e] += v * dybar[where[what]]
    variables = list(lin.variables)
    vpos = {v: i for i, v in enumerate(variables)}
    shocks = list(lin.shocks)
    n = len(variables)
    dA = np.zeros((3, K, n, n))  # A0 (lead), A1, A2 (lag)
    dB = np.zeros((K, n, len(shocks)))
    endo_set = set(endo)
    for e in np.flatnonzero(np.any(dE != 0, axis=1)).tolist():
        (i, name, lag), _ = entries[e]
        if name in endo_set:
            aux = name if -1 <= lag <= 1 else (f"{name}_lag{-lag - 1}" if lag < -1 else f"{name}_lead{lag - 1}")
            dA[{1: 0, 0: 1, -1: 2}[max(-1, min(1, lag))], :, i, vpos[aux]] += dE[e]
        elif name in shocks:
            dB[:, i, shocks.index(name)] += dE[e]
    return dA[0], dA[1], dA[2], dB, dybar, where


def _solution(lin: Any, P: np.ndarray, Qm: np.ndarray, dA0: np.ndarray, dA1: np.ndarray, dA2: np.ndarray,
              dB: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dP and dQ (K x n x n, K x n x shocks) by the implicit-function theorem on ``A0 P^2 + A1 P + A2 = 0``."""
    A0, A1, A2 = lin.A0, lin.A1, lin.A2
    K, n = dA0.shape[0], A0.shape[0]
    has_lag = np.any(A2 != 0, axis=0)
    lagged = np.flatnonzero(has_lag)
    if not has_lag.all() and np.any(dA2[:, :, ~has_lag] != 0):
        raise NoGradient("a variable's lag has a coefficient of zero at these values but not nearby")
    M = A0 @ P + A1
    try:
        Minv = np.linalg.inv(M)  # small, and solved against many right-hand sides
    except np.linalg.LinAlgError:
        raise NoGradient("A0 P + A1 is singular") from None
    dP = np.zeros((K, n, n))
    m = len(lagged)
    if m:
        P2 = P @ P
        C = dA0 @ P2 + dA1 @ P + dA2  # K x n x n
        CL = C[:, :, lagged]
        D = -(Minv @ CL)  # K x n x m
        Kmat = Minv @ A0
        S, U = linalg.schur(P[np.ix_(lagged, lagged)], output="complex")
        # Kmat = V Tk V* (its complex Schur form): each column's system I + s_jj Kmat is then triangular, I + s_jj Tk,
        # in the coordinates W = V* Y, and the earlier columns' terms Kmat Y become Tk W
        Tk, V = linalg.schur(Kmat, output="complex")
        DU = V.conj().T @ (D @ U)  # n x ... per parameter: K x n x m
        W = np.zeros((m, n, K), dtype=complex)  # column j of W(k) as W[j][:, k]
        diag, trtrs = np.diag_indices(n), lapack.ztrtrs  # LAPACK's triangular solve directly
        for j in range(m):
            rhs = DU[:, :, j].T
            if j:
                rhs = rhs - Tk @ np.tensordot(S[:j, j], W[:j], axes=(0, 0))
            Uj = S[j, j] * Tk
            Uj[diag] += 1.0
            W[j], info = trtrs(Uj, rhs)
            if info:
                raise NoGradient("the solution's derivative is not determined (Blanchard-Kahn on the edge)")
        Y = (U.conj() @ (V @ W).reshape(m, n * K)).reshape(m, n, K)  # column i of dP(k) as Y[i][:, k]
        dP[:, :, lagged] = Y.real.transpose(2, 1, 0)
    dM = dA0 @ P + A0 @ dP + dA1
    rhs = dB + dM @ Qm  # K x n x shocks
    return dP, -(Minv @ rhs)


_PLANS: dict[int, tuple[Any, Any, list]] = {}  # the measurement equations' symbolic derivatives, by their rows' identity


def _measurement_plan(measurement: Any) -> list:
    """Each measurement equation's first derivatives in its variables, and their derivatives in the variables (the
    symbolic work, done once for a measurement map's equations: copies with other errors share them)."""
    rows, params = measurement.measurements, measurement.params
    found = _PLANS.get(id(rows))
    if found is not None and found[0] is rows and found[1] is params:
        return found[2]
    plan = []
    for mm in rows:
        rhs = mm.rhs.substitute(params)
        terms = []
        for ref in rhs.refs():
            d1 = diff(rhs, ref)
            seconds = [(d2, ref2.id) for ref2 in d1.refs() if str(d2 := diff(d1, ref2)) != _ZERO]
            terms.append((ref.id, ref.lag, d1, seconds))
        plan.append(terms)
    if len(_PLANS) > 64:
        _PLANS.clear()
    _PLANS[id(rows)] = (rows, params, plan)
    return plan


def _measurement(measurement: Any, keys: list[tuple[str, int]], steady: Mapping[str, float], dybar: np.ndarray,
                 where: Mapping[str, int], K: int) -> tuple[np.ndarray, np.ndarray]:
    """dZ (K x observables x states) and dd (K x observables): the measurement equations' derivatives through the
    steady state (``MeasurementMap.linearize`` differentiated)."""
    from ..expr import evaluate

    values = {k: float(v) for k, v in steady.items()}
    resolve = lambda name, lag: values[name]  # noqa: E731 - the steady state: every lag at the same value
    index = {s: j for j, s in enumerate(keys)}
    plan = _measurement_plan(measurement)
    dZ = np.zeros((K, len(plan), len(keys)))
    dd = np.zeros((K, len(plan)))
    zero = np.zeros(K)
    for i, terms in enumerate(plan):
        for name, lag, d1, seconds in terms:
            if name in where:
                dd[:, i] += float(evaluate(d1, resolve)) * dybar[where[name]]
            col = index[(name, lag)]
            for d2, name2 in seconds:
                dZ[:, i, col] += float(evaluate(d2, resolve)) * (dybar[where[name2]] if name2 in where else zero)
    return dZ, dd


# --- the filter and its adjoint --------------------------------------------------------------------------------------

def _filter(y: np.ndarray, T: np.ndarray, RQR: np.ndarray, Z: np.ndarray, d: np.ndarray, H: np.ndarray, P0: np.ndarray,
            presample: int) -> dict[str, Any]:
    """The Kalman filter from mean zero and variance ``P0``, as ``statespace.kalman_filter`` (missing observations, the
    converged-variance shortcut), arranged for its adjoint: the variance recursion first (it does not depend on the data),
    one *state* for each period in which the variance is updated, then the means. A period with missing observations
    is the full-size recursion with the missing rows of ``Z``, ``d`` and ``y`` set to zero and those of ``H`` to the
    identity: the same likelihood, gain and variances as the reduced recursion, and every state the same shape. Once
    the variance has converged (``_RICCATI_TOL``), each later period with every series observed reuses the converged
    period's state: no variance work at all. Returns the log likelihood and what ``_adjoint`` needs."""
    from ..statespace import _RICCATI_TOL

    n_t, n = y.shape[0], T.shape[0]
    observed = ~np.isnan(y)
    fast = _kernels()
    if fast is not None:
        T, RQR, Z, H = (np.ascontiguousarray(M, dtype=float) for M in (T, RQR, Z, H))
        state, P_s, G, Li, Pf_s, updated, fail = fast.variance_pass(T, RQR, Z, H, np.ascontiguousarray(P0, dtype=float),
                                                                    observed, _RICCATI_TOL)
        if fail >= 0:
            raise ModelError(f"the innovation variance is not positive definite in period {fail}")
    else:
        state, P_s, G, Li, Pf_s, updated = _variance(T, RQR, Z, H, P0, observed, _RICCATI_TOL)
    M = observed[updated].astype(float)  # states x observables: 1 where observed
    Finv = np.swapaxes(Li, 1, 2) @ Li
    Kf = G @ Finv
    TK = T @ Kf
    L = T - (TK * M[:, None, :]) @ Z  # the means' transition: a(t+1) = L a(t) + T K (y(t) - d), Z's observed rows
    diag = np.einsum("sii->si", Li)  # 1 / the Cholesky factor's diagonal
    logdet = -2 * np.sum(np.log(diag), axis=1)
    Fdiag = np.einsum("pn,snp->sp", Z, G) + np.diag(H)  # the innovation variances (read where observed)
    smallest = float(np.min(np.where(M > 0, Fdiag, np.inf))) if M.size else np.inf
    u = np.where(observed, y, 0.0) - M[state] * d  # y - d, zero where missing
    push = np.einsum("tnp,tp->tn", TK[state], u)  # T K (y - d), period by period
    if fast is not None:
        a = fast.means(np.ascontiguousarray(L), push, state)
    else:
        a = np.zeros((n_t + 1, n))
        Lt, pushes = list(L), list(push)
        at = a[0]
        for t, s in enumerate(state.tolist()):
            at = Lt[s].dot(at) + pushes[t]
            a[t + 1] = at
        a = a[:-1]
    v = u - observed * (a @ Z.T)
    e = np.einsum("tpq,tq->tp", Finv[state], v)
    w = (np.arange(n_t) >= presample).astype(float)
    seen = observed.sum(axis=1)
    loglike = -0.5 * float(np.sum(w * (seen * _LOG2PI + logdet[state] + np.einsum("tp,tp->t", v, e))))
    return {"loglike": loglike, "smallest": smallest, "state": state, "P": P_s, "G": G, "Finv": Finv, "Kf": Kf,
            "Pf": Pf_s, "M": M, "L": L, "a": a, "v": v, "e": e, "w": w}


def _kernels() -> Any:
    """The compiled loops (``pymetrix.models._kalman_compiled``) when numba is installed and they are not switched
    off; None for the NumPy loops."""
    from . import _kalman_compiled

    return _kalman_compiled.kernels() if _kalman_compiled.available() else None


def _variance(T: np.ndarray, RQR: np.ndarray, Z: np.ndarray, H: np.ndarray, P0: np.ndarray, observed: np.ndarray,
              tol: float) -> tuple[np.ndarray, ...]:
    """The variance recursion of ``_filter`` in NumPy: per period the state it uses; per state the predicted variance,
    ``G = P Z'``, the inverse Cholesky factor of the innovation variance, the filtered variance, and its period."""
    n_t, n = observed.shape[0], T.shape[0]
    complete = observed.all(axis=1)
    TT = T.T
    potrf, trtri = lapack.dpotrf, lapack.dtrtri  # LAPACK directly: numpy.linalg's checks cost more than the work here
    state = np.empty(n_t, dtype=int)
    Ps, Gs, Lis, Pfs, updated = [], [], [], [], []
    shapes: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    P = np.asarray(P0, float)
    trace = float(P.trace())
    frozen = -1
    full = (Z, Z.T, H)
    for t in range(n_t):
        if frozen >= 0 and complete[t]:
            state[t] = frozen
            continue
        frozen = -1
        if complete[t]:
            Zt, ZtT, Ht = full
        else:
            key = observed[t].tobytes()
            if key not in shapes:
                m = observed[t].astype(float)
                Zm = Z * m[:, None]
                shapes[key] = (Zm, Zm.T, H * np.outer(m, m) + np.diag(1.0 - m))
            Zt, ZtT, Ht = shapes[key]
        G = P.dot(ZtT)
        c, info = potrf(Zt.dot(G) + Ht, lower=1, clean=1)
        if info:
            raise ModelError(f"the innovation variance is not positive definite in period {t}")
        Li = trtri(c, lower=1)[0]
        B = Li.dot(G.T)
        Pf = P - B.T.dot(B)
        P_next = T.dot(Pf).dot(TT) + RQR
        state[t] = len(Ps)
        Ps.append(P)
        Gs.append(G)
        Lis.append(Li)
        Pfs.append(Pf)
        updated.append(t)
        trace_next = float(P_next.trace())
        # converged as statespace.kalman_filter decides (every entry's change within the tolerance); first a test that
        # every converged variance passes (a variance's largest entries are on its diagonal, so the change in the trace
        # is at most n times the tolerance), which most periods fail at the cost of a trace
        if complete[t] and abs(trace_next - trace) <= n * tol * max(1.0, trace_next) and \
                np.abs(P_next - P).max() <= tol * max(1.0, float(np.abs(P_next).max())):
            frozen = state[t]
        P, trace = P_next, trace_next
    return state, np.array(Ps), np.array(Gs), np.array(Lis), np.array(Pfs), np.array(updated, dtype=int)


def _adjoint(f: Mapping[str, Any], T: np.ndarray, Z: np.ndarray) -> dict[str, np.ndarray]:
    """The log likelihood's derivatives in ``T``, ``Z``, ``d``, ``H``, ``R Q R'`` and the initial variance ``P0`` (each
    entry of each matrix, as the filter computes it), by the filter run backwards (reverse mode): the means' adjoint
    recursion over periods, then the variance's over states. Every period costs a few small matrix products whatever
    the number of parameters; the parameters come in once, contracting these with each one's derivatives."""
    state, w, v, e, a = f["state"], f["w"], f["v"], f["e"], f["a"]
    Kf, Finv, G, P, Pf, M, L = f["Kf"], f["Finv"], f["G"], f["P"], f["Pf"], f["M"], f["L"]
    n_t, n = a.shape
    S, p = M.shape
    # the means: abar(t) = L' abar(t+1) + w Z' e, from abar(n_t) = 0
    ze = w[:, None] * ((M[state] * e) @ Z)
    fast = _kernels()
    LT = list(np.swapaxes(L, 1, 2))
    if fast is not None:
        L = np.ascontiguousarray(L)
        abar = fast.backward_means(L, np.ascontiguousarray(ze), state)
    else:
        abar = np.zeros((n_t + 1, n))
        at = abar[n_t]
        order = state.tolist()
        for t in range(n_t - 1, -1, -1):
            at = LT[order[t]].dot(at) + ze[t]
            abar[t] = at
    after = abar[1:]  # the adjoint of a(t+1), period by period
    afbar = after @ T  # of the filtered mean a(t) + K v(t)
    af = a + np.einsum("tnp,tp->tn", Kf[state], v)
    Tbar = after.T @ af
    vbar = np.einsum("tnp,tn->tp", Kf[state], afbar) - w[:, None] * e
    dbar = -np.sum(M[state] * vbar, axis=0)
    Zbar = -(M[state] * vbar).T @ a
    # each state's share: the sum over the periods it serves (one each, except the converged state)
    # (the states run in order: a converged state serves the periods after its own until one has a missing value)
    first = np.flatnonzero(np.r_[True, state[1:] != state[:-1]])
    Kfbar = np.add.reduceat(afbar[:, :, None] * v[:, None, :], first, axis=0)
    wv = w[:, None] * v
    Finvbar = -0.5 * np.add.reduceat(wv[:, :, None] * v[:, None, :], first, axis=0)
    ldbar = -0.5 * np.add.reduceat(w, first)
    # the variances, state by state backwards: Pbar(s) from Pbar(s+1). P is symmetric and so are its perturbations,
    # so Pbar is kept symmetric (its symmetric part is all that counts); then, with Kf = G Finv,
    #   Fbar = A + Kf' Pfbar Kf,  A = ldbar Finv - Finv (Finvbar + G' Kfbar) Finv  (symmetrised)
    #   Gbar = Kfbar Finv - 2 Pfbar Kf + Z' Fbar,   Pbar = Pfbar + sym(Gbar Z),   Pfbar = T' Pbar(s+1) T
    # which collect into Pbar(s) = L' Pbar(s+1) L + C, with L = T (I - Kf Z) the means' transition and
    # C = sym(Kfbar Finv Z) + Z' A Z: three products a state in the loop, and Fbar, Gbar for every state after it
    A = ldbar[:, None, None] * Finv - Finv @ (Finvbar + np.swapaxes(G, 1, 2) @ Kfbar) @ Finv
    A = 0.5 * (A + np.swapaxes(A, 1, 2))
    Bs = Kfbar @ Finv
    C = (Bs * M[:, None, :]) @ Z
    C = 0.5 * (C + np.swapaxes(C, 1, 2)) + Z.T @ (A * M[:, :, None] * M[:, None, :]) @ Z
    if fast is not None:
        Pnext, Pn = fast.backward_variance(L, np.ascontiguousarray(C))
    else:
        Pn = np.zeros((n, n))
        Pnext = [Pn] * S
        Ll, LTl, Cl = list(L), LT, list(C)
        for s in range(S - 1, -1, -1):
            Pnext[s] = Pn
            Pn = LTl[s].dot(Pn).dot(Ll[s]) + Cl[s]
        Pnext = np.array(Pnext)
    PT = Pnext @ T
    Pfbar = T.T @ PT
    PK = Pfbar @ Kf
    Fbars = A + np.swapaxes(Kf, 1, 2) @ PK
    Gbars = Bs - 2 * PK + Z.T @ (M[:, :, None] * Fbars)
    # P(s+1) = T Pf(s) T' + R Q R': T's share is 2 Pbar T Pf (both symmetric), summed over states
    Tbar += 2 * np.swapaxes(PT, 0, 1).reshape(n, S * n) @ Pf.reshape(S * n, n)
    Ztbar = Fbars @ np.swapaxes(G, 1, 2) + np.swapaxes(Gbars, 1, 2) @ P
    Zbar += np.einsum("sp,spn->pn", M, Ztbar)
    Hbar = np.einsum("sp,spq,sq->pq", M, Fbars, M)
    return {"T": Tbar, "Z": Zbar, "d": dbar, "H": Hbar, "RQR": Pnext.sum(axis=0), "P0": Pn}


_KRON_MAX = 12  # states up to which the Lyapunov equations are solved by one factorisation of I - T (x) T, kept


def _lyapunov(T: np.ndarray, C: np.ndarray, adjoint: bool = False) -> np.ndarray:
    """``X = T X T' + C`` (or ``X = T' X T + C``, the stationary variance's adjoint): for a few states by one LU
    factorisation of ``I - T (x) T`` shared by the two (cached on ``T``'s values), otherwise ``solve_discrete_lyapunov``."""
    n = T.shape[0]
    if n > _KRON_MAX:
        return linalg.solve_discrete_lyapunov(T.T if adjoint else T, C)
    global _LU
    key = T.tobytes()
    cached = _LU
    if cached is None or cached[0] != key:
        cached = _LU = (key, linalg.lu_factor(np.eye(n * n) - np.kron(T, T)))
    return linalg.lu_solve(cached[1], C.reshape(-1), trans=1 if adjoint else 0).reshape(n, n)


_LU: tuple[bytes, Any] | None = None


def _score_filter(y: np.ndarray, T: np.ndarray, RQR: np.ndarray, Z: np.ndarray, d: np.ndarray, H: np.ndarray,
                  P0: np.ndarray, dT: np.ndarray, dRQR: np.ndarray, dZ: np.ndarray, dd: np.ndarray, dH: np.ndarray,
                  presample: int, stationary: bool) -> tuple[float, np.ndarray]:
    """The log likelihood and its gradient (K) by the Kalman filter from mean zero and variance ``P0``: the filter
    forwards, its adjoint backwards, then the adjoints contracted with each parameter's derivatives of the system
    matrices. ``stationary``: ``P0`` is the stationary variance (``P0 = T P0 T' + R Q R'``), whose adjoint is one more
    Lyapunov equation, ``X = T' X T + P0bar``, giving ``T`` and ``R Q R'`` their share; otherwise ``P0`` is fixed."""
    f = _filter(y, T, RQR, Z, d, H, P0, presample)
    K = dT.shape[0]
    if f["smallest"] <= 1e-12 or not np.isfinite(f["loglike"]):
        return -np.inf, np.full(K, np.nan)
    bar = _adjoint(f, T, Z)
    Tbar, RQRbar = bar["T"], bar["RQR"]
    if stationary and T.shape[0]:
        P0bar = (bar["P0"] + bar["P0"].T) / 2  # P0 is symmetrised as it is computed
        X = _lyapunov(T, P0bar, adjoint=True)
        Tbar = Tbar + (X + X.T) @ T @ P0
        RQRbar = RQRbar + X
    grad = (np.einsum("kij,ij->k", dT, Tbar) + np.einsum("kij,ij->k", dRQR, RQRbar) + dd @ bar["d"]
            + np.einsum("kij,ij->k", dH, bar["H"]))
    if np.any(dZ):
        grad += np.einsum("kij,ij->k", dZ, bar["Z"])
    return f["loglike"], grad


# --- the whole map ---------------------------------------------------------------------------------------------------

def loglike_score(lik: Any, values: Mapping[str, float]) -> tuple[float, np.ndarray]:
    """The log likelihood at ``values`` (the estimated parameters by name, as ``lik.free``) and its gradient in them,
    in the order of ``lik.free``. Raises ``NoGradient`` where the analytic gradient does not apply."""
    from .covariance import correlation_matrix

    reason = supported(lik)
    if reason:
        raise NoGradient(reason)
    free = list(lik.free)
    K = len(free)
    values = {**{n: float(v) for n, v in values.items()}}
    m, solution, system = lik.build(values)
    lin = solution.model
    shocks = list(lin.shocks)
    observables = list(system.observables)
    structural = [n for n in free if n in lik.structural]
    columns = {n: free.index(n) for n in structural}
    kinds: dict[int, tuple[str, Any]] = {}
    pairs = getattr(lik, "corr_pairs", {}) or {}
    for j, name in enumerate(free):
        if name in lik.structural:
            continue
        hits = []
        if name.startswith("sd_") and name[3:] in shocks:
            hits.append(("shock", shocks.index(name[3:])))
        if name.startswith("sd_") and name[3:] in observables:
            hits.append(("error", observables.index(name[3:])))
        if name in pairs:
            a_, b_ = pairs[name]
            hits.append(("corr", (shocks.index(a_), shocks.index(b_))))
        if len(hits) != 1:
            raise NoGradient(f"{name} is not a parameter the analytic gradient knows")
        kinds[j] = hits[0]

    dA0, dA1, dA2, dB, dybar, where = _structural(m, lin, structural, K, columns)
    dP, dQ = _solution(lin, solution.P, solution.Q, dA0, dA1, dA2, dB)

    # the state-space form, as LinearSolution.system builds it
    k = len(lin.variables)
    L = system.measurement.max_lag
    n = k * (L + 1)
    keys = [(v, -lag) for lag in range(L + 1) for v in lin.variables]
    dT = np.zeros((K, n, n))
    dT[:, :k, :k] = dP
    dR = np.zeros((K, n, len(shocks)))
    dR[:, :k] = dQ
    steady = dict(lin.steady.values)
    for v in lin.variables:
        steady.setdefault(v, steady.get(v.rsplit("_", 1)[0], 0.0))
    dZ, dd = _measurement(system.measurement, keys, steady, dybar, where, K)
    dH = np.zeros((K, len(observables), len(observables)))
    Q = system.Q
    dQc = np.zeros((K, len(shocks), len(shocks)))
    sd = np.sqrt(np.diag(Q))
    C = correlation_matrix(shocks, system.shock_corr) if system.shock_corr else None
    C = np.eye(len(shocks)) if C is None else C
    for j, (kind, at) in kinds.items():
        if kind == "error":
            dH[j, at, at] = 2 * float(system.measurement.errors.get(observables[at], 0.0))
        elif kind == "shock":
            row = C[at] * sd
            dQc[j, at, :] += row
            dQc[j, :, at] += row
        else:
            a_, b_ = at
            dQc[j, a_, b_] = dQc[j, b_, a_] = sd[a_] * sd[b_]
    R = system.R
    RQ = R @ Q
    dRQR = dR @ RQ.T
    dRQR = dRQR + np.swapaxes(dRQR, 1, 2) + R @ dQc @ R.T
    RQR = RQ @ R.T
    T, Z = system.T, system.Z

    # the states the likelihood needs (StateSpaceSystem.needed), with the derivatives' patterns too
    start = getattr(lik, "start", "auto")
    all_stationary = bool(system.stationary) and all(system.stationary)
    if start == "scaled" or all_stationary:
        need = np.any(Z != 0, axis=0) | np.any(dZ != 0, axis=(0, 1))
        pattern = (T != 0) | np.any(dT != 0, axis=0)
        while True:
            more = need | np.any(pattern[need], axis=0)
            if np.array_equal(more, need):
                break
            need = more
        keep = np.flatnonzero(need)
        T, RQR, Z = T[np.ix_(keep, keep)], RQR[np.ix_(keep, keep)], Z[:, keep]
        dT, dRQR, dZ = dT[:, keep][:, :, keep], dRQR[:, keep][:, :, keep], dZ[:, :, keep]
    ns = T.shape[0]
    if start == "scaled":
        from ..statespace import SCALED_VARIANCE

        P0 = SCALED_VARIANCE * np.eye(ns)
    else:
        if not all_stationary:
            roots = np.abs(np.linalg.eigvals(T)) if ns else np.zeros(0)
            if not (start == "stationary" or np.all(roots < 1 - 1e-8)):
                raise NoGradient("states with unit roots start diffuse: not covered by the analytic gradient")
        if ns and np.max(np.abs(np.linalg.eigvals(T))) >= 1 - 1e-8:
            raise NoGradient("the states are not stationary at these values")
        P0 = _lyapunov(T, RQR)
        P0 = (P0 + P0.T) / 2

    observed = lik.__dict__.get("_observed")
    if observed is None:
        observed = system.observations(lik.data, lik.span)
        object.__setattr__(lik, "_observed", observed)
    y, index = observed
    y = np.atleast_2d(np.asarray(y, float)) - system._offset(len(y)) - system._dtrend(index, np.shape(y)[1])
    return _score_filter(y, T, RQR, Z, system.d, system.H, P0, dT, dRQR, dZ, dd, dH, int(getattr(lik, "presample", 0)),
                         start != "scaled")


def prior_gradient(priors: Mapping[str, Any], free: list[str], x: np.ndarray) -> np.ndarray:
    """The log prior's gradient: each prior's log density differentiated by central differences in its own parameter
    (one-dimensional closed forms: accurate to about 1e-10, at a negligible cost), one-sided at its support's edge."""
    g = np.zeros(len(free))
    for i, name in enumerate(free):
        prior = priors.get(name)
        if prior is None:
            continue
        xi = float(x[i])
        h = 1e-6 * max(1.0, abs(xi))
        lo, hi = getattr(prior, "support", (-np.inf, np.inf))
        up, down = min(xi + h, hi - 1e-15) if np.isfinite(hi) else xi + h, max(xi - h, lo + 1e-15) if np.isfinite(lo) else xi - h
        if up <= down:
            continue
        g[i] = (float(prior.logpdf(up)) - float(prior.logpdf(down))) / (up - down)
    return g


# --- what estimation and sampling call -------------------------------------------------------------------------------

GRADIENTS = ("auto", "analytic", "numerical")
_CAUGHT = (ModelError, ValueError, np.linalg.LinAlgError, ZeroDivisionError, OverflowError, FloatingPointError)


def ready(lik: Any, x: np.ndarray | None = None, gradient: str | None = None) -> bool:
    """Whether to use the analytic gradient for this likelihood (tried at ``x`` when given): ``gradient`` (default the
    estimate's own setting) ``"numerical"`` never, ``"auto"`` where it applies, ``"analytic"`` where it applies and
    refused (``ModelError``, saying why) where it does not."""
    mode = gradient or getattr(lik, "gradient", "auto")
    if mode == "numerical":
        return False
    reason = supported(lik)
    if reason is None and x is not None:
        try:
            loglike_score(lik, dict(zip(lik.free, map(float, x))))
        except NoGradient as exc:
            reason = str(exc)
        except _CAUGHT:
            pass  # a point where the model cannot be solved: the gradient may still apply elsewhere
    if reason is None:
        return True
    if mode == "analytic":
        raise ModelError(f"no analytic gradient: {reason} (use gradient='auto' or 'numerical')")
    return False


def objective_gradient(lik: Any, x: np.ndarray) -> tuple[float, np.ndarray]:
    """Minus the log posterior (``_Likelihood.objective``) and its gradient: analytic, with central differences at a
    point the analytic gradient does not cover (a steady state not pinned down there, say); the penalty with a zero
    gradient where the model cannot be solved."""
    from .estimation import _PENALTY

    x = np.asarray(x, dtype=float)
    free = list(lik.free)
    values = dict(zip(free, map(float, x)))
    zero = np.zeros(len(free))
    if any(not (lik.limits[n][0] <= values[n] <= lik.limits[n][1]) for n in free):
        return _PENALTY, zero
    try:
        with np.errstate(all="ignore"):  # a point where the model cannot be solved: the penalty below, quietly
            ll, g = loglike_score(lik, values)
        value = ll + lik.log_prior(values)
    except NoGradient:
        from ..estimation.numdiff import jacobian

        value = lik.objective(x)
        if value >= _PENALTY:
            return _PENALTY, zero
        return value, jacobian(lambda u: np.array([lik.objective(u)]), x)[0]
    except _CAUGHT:
        return _PENALTY, zero
    if not np.isfinite(value) or not np.all(np.isfinite(g)):
        return _PENALTY, zero
    return -float(value), -(g + prior_gradient(lik.priors, free, x))


def log_posterior_gradient(lik: Any, x: np.ndarray) -> tuple[float, np.ndarray]:
    """The log posterior (log likelihood plus log prior; -inf where impossible) and its gradient."""
    from .estimation import _PENALTY

    value, g = objective_gradient(lik, x)
    if value >= _PENALTY:
        return -np.inf, np.full(len(g), np.nan)
    return -value, -g


def hessian_from_gradient(lik: Any, x: np.ndarray, interior: list[int]) -> np.ndarray:
    """The Hessian of the log posterior in the ``interior`` parameters (the others held at ``x``) by central differences
    of the analytic gradient, symmetrised: ``2k`` gradients, each exact, so accurate to about the cube root of machine
    precision squared (1e-8 relative) where second differences of the log posterior give about 1e-4."""
    from ..estimation.numdiff import jacobian

    x = np.asarray(x, dtype=float)

    def g(z: np.ndarray) -> np.ndarray:
        full = x.copy()
        full[interior] = z
        value, grad = log_posterior_gradient(lik, full)
        if not np.isfinite(value):
            raise ModelError("the log posterior is not finite next to the mode")
        return grad[interior]

    H = jacobian(g, x[interior])
    return (H + H.T) / 2


def gradient_frame(fit: Any, at: Mapping[str, float] | None = None, method: str = "analytic") -> Any:
    """The work behind ``FittedModel.gradient``: a DataFrame, one row per estimated parameter, with the value and the
    gradients of the log likelihood and the log posterior."""
    import pandas as pd

    from ..errors import SpecificationError

    if method not in ("analytic", "numerical"):
        raise SpecificationError(f"method is 'analytic' or 'numerical', not {method!r}")
    lik = getattr(fit.log_posterior, "__self__", None)
    if lik is None or not hasattr(lik, "free"):
        raise ModelError("this estimate keeps no likelihood: estimate the model again with this version of pymetrix")
    names = list(lik.free)
    unknown = sorted(set(at or {}) - set(names))
    if unknown:
        raise SpecificationError(f"at= gives {', '.join(unknown)}, which are not estimated parameters ({', '.join(names)})")
    point = {**dict(zip(names, map(float, fit.params.values))), **{n: float(v) for n, v in (at or {}).items()}}
    x = np.array([point[n] for n in names])
    if method == "analytic":
        ready(lik, None, "analytic")
        ll, g = loglike_score(lik, point)
    else:
        from ..estimation.numdiff import jacobian

        ll = lik.loglike(point)
        g = jacobian(lambda u: np.array([lik.loglike(dict(zip(names, map(float, u))))]), x)[0]
    gp = g + prior_gradient(lik.priors, names, x) if lik.priors else np.full(len(names), np.nan)
    frame = pd.DataFrame({"value": x, "d log likelihood": g, "d log posterior": gp}, index=pd.Index(names, name="parameter"))
    frame.attrs["log likelihood"] = float(ll)
    frame.attrs["method"] = method
    return frame
