"""Optimal policy in linear(ised) models with a quadratic loss: commitment, discretion, and optimal simple rules.

    nk = templates.new_keynesian()
    com = optimal_policy(nk, "i", {"pi": 1.0, "y": 0.25}, kind="commitment")   # the policy rule is replaced
    dis = optimal_policy(nk, "i", {"pi": 1.0, "y": 0.25}, kind="discretion")
    com.irf(20).plot(); com.loss(); com.solution.rules()
    best = osr(nk, ["phi_pi", "phi_y"], {"pi": 1.0, "y": 0.25}, bounds={"phi_pi": (1.01, 5), "phi_y": (0, 3)})
    best.values, best.loss, best.template.solve()

The loss is the discounted sum of ``x' W x``, with ``W`` diagonal: a weight for each variable named in ``loss``
(the instrument too, to penalise its moves). The model's own policy rule, the equation that determines the
instrument (``rule=`` names it if the instrument is not its target), is dropped: the remaining equations are the
constraints, and the instrument is set optimally.

Commitment (from the timeless perspective) adds a Lagrange multiplier for each constraint and solves the first-order
conditions ``W x(t) + A1' l(t) + A0' l(t-1) / beta + beta A2' E l(t+1) = 0`` with the constraints as one linear
rational-expectations model (the multipliers are named ``mult_<equation>``). Discretion is the Markov-perfect
equilibrium of Dennis (2007): each period the policymaker minimises ``x' (W + beta S) x`` subject to the constraints,
taking future policy ``E x(t+1) = H x(t)`` as given; ``H`` and the value matrix ``S`` are iterated to a fixed point.
Optimal simple rules choose a rule's coefficients to minimise the weighted unconditional variances (Dynare's
``osr``); coefficients that leave no unique stable solution are ruled out.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy import linalg, optimize

from .._api import api
from ..errors import SpecificationError
from ..output import Table
from .solvers import ModelError

__all__ = ["OSRResult", "OptimalPolicy", "optimal_policy", "osr"]


def _parts(obj: Any) -> tuple[Any, tuple[str, ...], dict[str, float], dict[str, float]]:
    from .model import Model

    if isinstance(obj, Model):
        return obj, tuple(obj.compile().shocks), {}, {}
    model = getattr(obj, "model", None)
    if not isinstance(model, Model):
        raise TypeError(f"optimal policy needs a template, a Dynare or IRIS model, or a Model, not {type(obj).__name__}")
    shocks = tuple(getattr(obj, "shocks", None) or getattr(obj, "exogenous", ()))
    return model, shocks, dict(getattr(obj, "shock_sd", {}) or {}), dict(getattr(obj, "guess", None) or {})


def _weights(lin: Any, loss: Mapping[str, float]) -> np.ndarray:
    unknown = [v for v in loss if v not in lin.observed]
    if unknown:
        raise ModelError(f"the loss names {unknown}, which are not variables of the model")
    W = np.zeros((len(lin.variables), len(lin.variables)))
    for v, w in loss.items():
        if w < 0:
            raise ModelError(f"loss weights cannot be negative ({v}: {w})")
        i = lin.variables.index(v)
        W[i, i] = float(w)
    return W


@dataclass(frozen=True, eq=False)
class OptimalPolicy:
    """The optimal-policy solution: see the module docstring."""

    kind: str  # "commitment" or "discretion"
    instrument: str
    weights: Mapping[str, float]
    beta: float
    solution: Any  # LinearSolution over the model's variables (and, under commitment, the multipliers)
    shock_sd: Mapping[str, float] = field(default_factory=dict)
    dropped: str = ""  # the policy rule that was replaced

    def irf(self, periods: int = 20, shock_sd: Mapping[str, float] | None = None) -> Any:
        return self.solution.irf(periods, shock_sd or self.shock_sd or None)

    def moments(self, lags: int = 4) -> Any:
        return self.solution.moments(self.shock_sd or None, lags)

    def loss(self) -> float:
        """The weighted unconditional variances, ``sum w_v Var(v)``: the loss per period on average."""
        m = self.moments()
        return float(sum(w * m.covariance.loc[v, v] for v, w in self.weights.items()))

    def summary(self) -> Table:
        m = self.moments()
        rows = [[v, w, float(np.sqrt(m.covariance.loc[v, v]))] for v, w in self.weights.items()]
        notes = [f"Optimal policy under {self.kind}, instrument {self.instrument}" + (f" (replacing the rule {self.dropped!r})" if self.dropped else ""),
                 f"Discount factor {self.beta:g}; loss per period on average (weighted variances): {self.loss():.6g}"]
        return Table(["weight", "std. dev."], [r[1:] for r in rows], index=[r[0] for r in rows], title="Optimal policy", notes=notes)

    def __repr__(self) -> str:
        return f"<OptimalPolicy: {self.kind}, instrument {self.instrument}, loss {self.loss():.4g}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


def _constraints(model: Any, shocks: tuple[str, ...], instrument: str, rule: str | None, guess: Mapping[str, float]) -> tuple[Any, int, str]:
    from .linear import linearize

    if instrument not in model.endogenous:
        raise ModelError(f"the instrument {instrument!r} is not an endogenous variable of the model")
    eqs = model.model_equations
    if rule is not None:
        matches = [i for i, e in enumerate(eqs) if e.name == rule]
        if not matches:
            raise ModelError(f"no equation named {rule!r}; the equations are {', '.join(e.name for e in eqs)}")
    else:
        matches = [i for i, e in enumerate(eqs) if e.dependent == instrument]
        if not matches:
            raise ModelError(f"no equation determines {instrument!r}: name the policy rule to replace with rule=")
    ss = model.steady_state(guess=dict(guess) or None)
    lin = linearize(model, ss, shocks)
    return lin, matches[0], eqs[matches[0]].name


def _commitment(lin: Any, drop: int, W: np.ndarray, beta: float) -> Any:
    from .linear import LinearModel, SteadyState

    keep = [i for i in range(len(lin.variables)) if i != drop]
    A0, A1, A2, B = lin.A0[keep], lin.A1[keep], lin.A2[keep], lin.B[keep]
    n, m = A0.shape[1], A0.shape[0]
    Z = np.zeros
    A0z = np.block([[A0, Z((m, m))], [Z((n, n)), beta * A2.T]])
    A1z = np.block([[A1, Z((m, m))], [W, A1.T]])
    A2z = np.block([[A2, Z((m, m))], [Z((n, n)), A0.T / beta]])
    Bz = np.vstack([B, Z((n, B.shape[1]))])
    names = [f"mult_{j}" for j in range(m)]
    steady = SteadyState({**lin.steady.values, **{nm: 0.0 for nm in names}}, lin.steady.model)
    big = LinearModel(A0z, A1z, A2z, Bz, tuple(lin.variables) + tuple(names), lin.shocks, steady, lin.observed)
    return big.solve()


def _discretion(lin: Any, drop: int, W: np.ndarray, beta: float, tol: float = 1e-12, max_iter: int = 10000) -> Any:
    from .linear import LinearSolution

    keep = [i for i in range(len(lin.variables)) if i != drop]
    A0, A1, A2, B = lin.A0[keep], lin.A1[keep], lin.A2[keep], lin.B[keep]
    n, m = A0.shape[1], A0.shape[0]
    H, S = np.zeros((n, n)), np.zeros((n, n))
    for _ in range(max_iter):
        C = A0 @ H + A1  # the constraints, given E x(t+1) = H x(t)
        K = np.block([[W + beta * S, C.T], [C, np.zeros((m, m))]])
        try:
            sol = np.linalg.solve(K, np.vstack([np.zeros((n, n + B.shape[1])), -np.hstack([A2, B])]))
        except np.linalg.LinAlgError:
            raise ModelError("the policymaker's problem has no unique solution (the bordered system is singular)") from None
        H_new, G = sol[:n, :n], sol[:n, n:]
        S_new = H_new.T @ (W + beta * S) @ H_new
        if np.max(np.abs(H_new - H)) < tol and np.max(np.abs(S_new - S)) < tol * max(1.0, np.max(np.abs(S_new))):
            if np.max(np.abs(np.linalg.eigvals(H_new))) >= 1:
                raise ModelError("the discretionary equilibrium found is not stable")
            return LinearSolution(H_new, G, lin, np.linalg.eigvals(H_new))
        H, S = H_new, S_new
    raise ModelError(f"the discretionary equilibrium did not converge in {max_iter} iterations")


@api("pymetrix:optimal_policy")
def optimal_policy(obj: Any, instrument: str, loss: Mapping[str, float], kind: str = "commitment", beta: float = 0.99,
                   rule: str | None = None) -> OptimalPolicy:
    """Optimal policy for ``obj`` (a template, a Dynare or IRIS model, or a Model): see the module docstring."""
    if kind not in ("commitment", "discretion"):
        raise SpecificationError("kind is 'commitment' or 'discretion'")
    if not 0 < beta < 1:
        raise SpecificationError("the discount factor beta is between 0 and 1")
    model, shocks, shock_sd, guess = _parts(obj)
    lin, drop, dropped = _constraints(model, shocks, instrument, rule, guess)
    W = _weights(lin, loss)
    solution = _commitment(lin, drop, W, beta) if kind == "commitment" else _discretion(lin, drop, W, beta)
    return OptimalPolicy(kind, instrument, dict(loss), beta, solution, shock_sd, dropped)


# --- optimal simple rules ----------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class OSRResult:
    """An optimal simple rule: the coefficients, the loss, and the model with them."""

    values: Mapping[str, float]
    loss: float
    template: Any  # the template (or model) with the optimal coefficients
    weights: Mapping[str, float]
    converged: bool = True

    def summary(self) -> Table:
        return Table(["value"], [[v] for v in self.values.values()], index=list(self.values), title="Optimal simple rule",
                     notes=[f"Loss (weighted unconditional variances): {self.loss:.6g}; weights "
                            + ", ".join(f"{k} {w:g}" for k, w in self.weights.items()),
                            "Converged" if self.converged else "Did not converge: check the bounds and starting values"])

    def __repr__(self) -> str:
        return f"<OSRResult: {', '.join(f'{k}={v:.4g}' for k, v in self.values.items())}; loss {self.loss:.4g}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


def _rule_loss(obj: Any, values: Mapping[str, float], loss: Mapping[str, float]) -> float:
    from ..sweeps import _with_values

    from .model import Model

    candidate = _with_values(obj, values)
    if isinstance(candidate, Model):  # a model on its own: its first-order solution, shocks of size 1
        sol = candidate.linearize(shocks=candidate.compile().shocks).solve()
    else:
        sol = candidate.solve()
    shock_sd = getattr(candidate, "shock_sd", None) or None
    m = sol.moments(shock_sd)
    return float(sum(w * m.covariance.loc[v, v] for v, w in loss.items()))


@api("pymetrix:osr")
def osr(obj: Any, params: Sequence[str], loss: Mapping[str, float], bounds: Mapping[str, tuple[float, float]] | None = None,
        start: Mapping[str, float] | None = None) -> OSRResult:
    """The coefficients ``params`` of a template's (or Dynare or IRIS model's) policy rule that minimise the weighted
    unconditional variances: see the module docstring."""
    from ..sweeps import _with_values

    params = list(params)
    model, _, _, _ = _parts(obj)
    current = dict(model.parameters.as_dict()) if hasattr(model.parameters, "as_dict") else {p.name: p.value for p in model.parameters}
    missing = [p for p in params if p not in current]
    if missing:
        raise ModelError(f"{missing} are not parameters of the model")
    x0 = np.array([float((start or {}).get(p, current[p])) for p in params])
    box = [tuple((bounds or {}).get(p, (-np.inf, np.inf))) for p in params]
    penalty = 1e10
    from ..progress import report

    tried = [0]

    def objective(x: np.ndarray) -> float:
        tried[0] += 1
        report(None, f"searching rule coefficients: evaluation {tried[0]:,}")  # the app's progress bar; Cancel stops here
        if any(not (lo <= v <= hi) for v, (lo, hi) in zip(x, box)):
            return penalty
        try:
            value = _rule_loss(obj, dict(zip(params, map(float, x))), loss)
        except (ModelError, ValueError, np.linalg.LinAlgError):
            return penalty
        return value if np.isfinite(value) else penalty

    if objective(x0) >= penalty:
        raise ModelError("the model has no unique stable solution at the starting coefficients: give start= or bounds=")
    best = optimize.minimize(objective, x0, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 4000})
    polish = optimize.minimize(objective, best.x, method="L-BFGS-B",
                               bounds=[(lo if np.isfinite(lo) else None, hi if np.isfinite(hi) else None) for lo, hi in box])
    if polish.fun <= best.fun:
        best = polish
    values = dict(zip(params, map(float, best.x)))
    return OSRResult(values, float(best.fun), _with_values(obj, values), dict(loss), bool(best.success))
