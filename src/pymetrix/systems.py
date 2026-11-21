"""Systems of equations: SUR, three-stage least squares, and their single-equation counterparts.

``System([...]).fit(data, method=SUR())`` estimates several equations together::

    sys = System([Equation("c ~ y + c[-1]"), Equation("i ~ y[-1] + r"), Equation("y = c + i + g")])
    sys.fit(data, method=SUR())                  # correlated errors across equations (Zellner)
    sys.fit(data, method=ThreeSLS())             # and endogenous regressors (Zellner-Theil)
    sys.fit(data, method=SUR(), restrictions="c.y = i.y[-1]")
    sys.fit(data, method=FIML())                 # full-information maximum likelihood (Koopmans; Klein)

Identities are not estimated, but their left-hand sides count as endogenous for
3SLS. Linear-form coefficients are named ``equation.coefficient`` (``c.y``);
explicit-form parameters keep their own names, so a parameter used in more than
one equation is a cross-equation restriction. Further restrictions, linear or
not, are given as text and imposed exactly.

Every method minimises ``sum_t u_t' S u_t`` over the common sample, with the
residuals ``u_t`` of all equations at t: ``S`` is the identity for OLS and 2SLS
and the inverse residual covariance for SUR and 3SLS. For 2SLS and 3SLS the
residuals' derivatives are projected on the instruments: by default the
constant and every regressor that does not involve the current value of an
endogenous variable (left-hand sides of the system, plus ``endogenous=``), plus
``instruments=``. Nonlinear equations and restrictions are handled by
Gauss-Newton; linear ones converge in one step. Iterated SUR is maximum
likelihood under normal errors.

FIML maximises the likelihood of the whole simultaneous system under normal errors, concentrated in the error
covariance: sum_t log |det J_t| - n/2 log |U'U / n|, with J_t the derivative of every equation's residual (the
identities' included) with respect to the current endogenous variables, so there must be as many equations as
endogenous variables. It starts from 3SLS (or least squares) and needs no instruments; standard errors come from
the inverse Hessian. Exactly identified equations give 2SLS; a single structural equation beside the reduced form
gives LIML.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from ._api import api
from ._capture import data_fingerprint, records_warnings, with_effective_sample
from ._serial import to_jsonable
from ._version import __version__
from .adapters import native
from .errors import NotFoundError, SpecificationError
from .estimation.covariance import default_hac_lags
from .estimation.design import as_series, sample
from .estimation.fitted import FittedEquation
from .estimation.params import PointEstimate, point
from .expr import Coef, Expr, Name, diff, evaluate
from .expr.nodes import ExpressionError
from .forms import Widget
from .metadata import Provenance
from .output import Table
from .plugins import register

if TYPE_CHECKING:
    from .data import Dataset
    from .estimation.equation import Equation

__all__ = ["FIML", "FittedSystem", "SUR", "System", "SystemOLS", "ThreeSLS", "TwoSLS"]

Cov = Literal["classic", "white", "hac"]


# ---------------------------------------------------------------------------------------------
# methods


@dataclass(frozen=True)
class _SystemMethod:
    weighted: ClassVar[bool] = False  # S = inverse residual covariance (SUR, 3SLS)
    instrumented: ClassVar[bool] = False  # project on the instruments (2SLS, 3SLS)

    cov: Annotated[Cov, Widget("choice", help="classic; white: robust to heteroskedasticity; hac: and to serial correlation")] = "classic"
    iterate: Annotated[bool, Widget("checkbox", help="re-estimate the residual covariance until the estimates settle")] = False
    df_correction: Annotated[bool, Widget("checkbox", advanced=True, help="divide residual cross-products by sqrt((n-k_i)(n-k_j)), not n")] = False
    hac_lags: Annotated[int | None, Widget("number", min=0, max=48, advanced=True, help="Newey-West lags; empty = 4(n/100)^(2/9)")] = None
    tol: Annotated[float, Widget("number", advanced=True, help="convergence tolerance")] = 1e-8
    max_iter: Annotated[int, Widget("number", min=1, max=10_000, advanced=True, help="most iterations")] = 500


@dataclass(frozen=True)
class SystemOLS(_SystemMethod):
    """Each equation by least squares, but jointly, so that parameters can be shared or restricted across
    equations. Standard errors allow for errors correlated across equations."""

    label: ClassVar[str] = "Least squares (each equation)"
    topics: ClassVar[tuple[str, ...]] = ("systems", "least-squares")


@dataclass(frozen=True)
class TwoSLS(_SystemMethod):
    """Each equation by two-stage least squares with the system's instruments, jointly (shared parameters
    and restrictions allowed)."""

    label: ClassVar[str] = "Two-stage least squares (each equation)"
    topics: ClassVar[tuple[str, ...]] = ("systems", "instrumental-variables")
    instrumented: ClassVar[bool] = True


@dataclass(frozen=True)
class SUR(_SystemMethod):
    """Seemingly unrelated regressions (Zellner): generalised least squares allowing for errors correlated
    across equations. More precise than equation-by-equation least squares when the equations have
    different regressors and their errors are correlated. Iterated, it is maximum likelihood."""

    label: ClassVar[str] = "Seemingly unrelated regressions (SUR)"
    topics: ClassVar[tuple[str, ...]] = ("systems",)
    weighted: ClassVar[bool] = True


@dataclass(frozen=True)
class ThreeSLS(_SystemMethod):
    """Three-stage least squares (Zellner and Theil): SUR on equations whose regressors are endogenous,
    using the system's instruments. Consistent with endogenous regressors, and efficient among
    instrumental-variable estimators of the system when the errors are homoskedastic."""

    label: ClassVar[str] = "Three-stage least squares (3SLS)"
    topics: ClassVar[tuple[str, ...]] = ("systems", "instrumental-variables")
    weighted: ClassVar[bool] = True
    instrumented: ClassVar[bool] = True


@dataclass(frozen=True)
class FIML(_SystemMethod):
    """Full-information maximum likelihood: the whole simultaneous system, identities included, by maximum
    likelihood under normal errors. Efficient when the model is right, but a misspecified equation can bias the
    others; needs as many equations (with identities) as endogenous variables, and no instruments."""

    label: ClassVar[str] = "Full-information maximum likelihood (FIML)"
    topics: ClassVar[tuple[str, ...]] = ("systems", "instrumental-variables", "maximum-likelihood")
    weighted: ClassVar[bool] = True
    fiml: ClassVar[bool] = True


for _name, _cls in (("ols", SystemOLS), ("2sls", TwoSLS), ("sur", SUR), ("3sls", ThreeSLS), ("fiml", FIML)):
    register("systems", _cls, name=_name)


# ---------------------------------------------------------------------------------------------
# the system


_REF = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")


@dataclass(frozen=True, eq=False)
class System:
    """Equations estimated together. ``equations`` is a list (named by their targets) or a mapping of names."""

    code_positional: ClassVar[int] = 1  # generated code gives the first field by position: System(...)

    equations: Sequence[Equation] | Mapping[str, Equation]
    instruments: Sequence[str] = ()  # extra instruments (expressions), for 2SLS and 3SLS
    endogenous: Sequence[str] = ()  # extra endogenous variables, besides the left-hand sides
    restrictions: str = ""  # e.g. "c.y = i.y, a + b = 1"

    names: tuple[str, ...] = field(init=False, repr=False)
    stochastic: tuple[Equation, ...] = field(init=False, repr=False)
    identities: tuple[Equation, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.equations, Mapping):
            named = [(str(k), v) for k, v in self.equations.items()]
        else:
            named = [(eq.label, eq) for eq in self.equations]
        for name, eq in named:
            if not hasattr(eq, "form"):
                raise TypeError(f"a System takes Equations, not {type(eq).__name__} ({name})")
            if not eq.is_identity and (eq.fixed_effects or eq.categoricals or eq.has_error_model):
                raise ExpressionError(f"{eq.text!r}: system estimation does not support fe(), C(), ARMA errors or GARCH yet")
            if not _REF.fullmatch(f"{name}.x") or "." in name:
                raise ExpressionError(f"equation name {name!r} must be a plain name (letters, digits, _); give a mapping of names")
        stochastic = [(n, e) for n, e in named if not e.is_identity]
        if len(stochastic) < 1:
            raise ExpressionError("a system needs at least one equation to estimate")
        names = [n for n, _ in stochastic]
        dups = sorted({n for n in names if names.count(n) > 1})
        if dups:
            raise ExpressionError(f"more than one equation is named {dups}; give a mapping of names")
        object.__setattr__(self, "equations", tuple(e for _, e in named) if not isinstance(self.equations, Mapping) else dict(named))
        object.__setattr__(self, "names", tuple(names))
        object.__setattr__(self, "stochastic", tuple(e for _, e in stochastic))
        object.__setattr__(self, "identities", tuple(e for _, e in named if e.is_identity))
        object.__setattr__(self, "instruments", tuple(self.instruments))
        object.__setattr__(self, "endogenous", tuple(self.endogenous))

    # names of parameters

    def _rhs(self, i: int) -> Expr:
        """Equation i's right-hand side with system parameter names (``name.coef`` for linear form)."""
        eq, name = self.stochastic[i], self.names[i]
        if eq.form == "explicit":
            return eq.rhs
        return eq.linear_rhs().map(lambda n: Coef(f"{name}.{n.id}") if isinstance(n, Coef) else None)

    def _own(self, i: int) -> list[tuple[str, str]]:
        """(system name, equation's own name) of equation i's parameters."""
        eq, name = self.stochastic[i], self.names[i]
        if eq.form == "explicit":
            return [(p, p) for p in eq.param_names]
        return [(f"{name}.{c}", c) for c in eq.coefficient_names()]

    @property
    def param_names(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for i in range(len(self.stochastic)):
            for full, _ in self._own(i):
                seen.setdefault(full, None)
        return tuple(seen)

    @property
    def endogenous_variables(self) -> tuple[str, ...]:
        targets = [e.require_dependent() for e in self.stochastic + self.identities]
        return tuple(dict.fromkeys(targets + list(self.endogenous)))

    def instrument_list(self) -> list[Expr]:
        """The constant is added separately; these are the regressors that are not endogenous, then ``instruments``."""
        from .expr import parse_expression

        endo = set(self.endogenous_variables)

        def exogenous(e: Expr) -> bool:
            return not any(r.lag == 0 and r.id in endo for r in e.refs())

        out: dict[str, Expr] = {}
        for eq in self.stochastic + self.identities:
            if eq.form == "linear":
                candidates: list[Expr] = [t.expr for t in eq.terms]
            else:
                candidates = sorted(eq.rhs.refs(), key=str)
            for c in candidates:
                if c.refs() and exogenous(c):
                    out.setdefault(str(c), c)
        for text in self.instruments:
            node = parse_expression(text)
            out.setdefault(str(node), node)
        return list(out.values())

    def _restrictions(self) -> list[Expr]:
        from .estimation.restrictions import parse_restrictions, split_restrictions

        if not self.restrictions.strip():
            return []
        names = self.param_names
        for part in split_restrictions(self.restrictions):
            unknown = [m.group(0) for m in _REF.finditer(part)  # a dotted name that does not begin a coefficient's name
                       if not any(part.startswith(n, m.start()) for n in names)]
            if unknown:
                raise ExpressionError(f"unknown coefficient(s) {unknown} in restriction {part!r}; coefficients are {list(names)}")
        return [expr for _, expr in parse_restrictions(self.restrictions, names)]

    # estimation

    @api("pymetrix:system.fit")
    @records_warnings
    def fit(self, data: Dataset, method: Any = None, span: tuple[Any, Any] | None = None, where: str | None = None,
            start: Mapping[str, float] | None = None, missing: Literal["drop", "raise"] = "drop") -> FittedSystem:
        """Estimate the system over the sample common to all its equations."""
        from .estimation.families import Gaussian

        method = method if method is not None else SUR()
        for eq in self.stochastic:
            eq.check_names(data)
            if not isinstance(eq.family, Gaussian):
                raise ExpressionError(f"{eq.text!r}: system estimation needs Gaussian equations")
        fit = _estimate(self, data, method, span, where, start, missing)
        variables = sorted(set().union(*(eq.variables() for eq in self.stochastic)))
        object.__setattr__(fit, "provenance", with_effective_sample(fit.provenance, data, fit.resid.index, span=span, where=where,
                                                                    variables=variables))
        return fit

    def __repr__(self) -> str:
        return f"<System: {len(self.stochastic)} equations ({', '.join(self.names)}), {len(self.identities)} identities>"


# ---------------------------------------------------------------------------------------------
# estimation


@dataclass(frozen=True, eq=False)
class _Arrays:
    resolve: Any
    n: int
    index: pd.Index
    dropped: int
    Z: np.ndarray | None
    z_names: list[str]


def _arrays(system: System, data: Dataset, span: Any, where: str | None, method: Any, missing: str) -> _Arrays:
    index = data.frame.index
    cols: dict[str, pd.Series] = {}
    refs = set()
    for i, eq in enumerate(system.stochastic):
        refs |= eq.lhs.refs() | system._rhs(i).refs()
    for r in sorted(refs, key=str):
        cols[str(r)] = data.lagged(r.id, r.lag).astype(float)
    z_names: list[str] = []
    if method.instrumented:
        cols["__z__const"] = pd.Series(1.0, index=index)
        z_names.append("const")
        for e in system.instrument_list():
            unknown = sorted(e.variables() - set(data.names))
            if unknown:
                raise ExpressionError(f"instrument {e} uses names that are not series: {unknown}")
            cols[f"__z__{e}"] = as_series(evaluate(e, data.lagged), index)
            z_names.append(str(e))
    used, dropped = sample(pd.DataFrame(cols, index=index), data.sample_mask(span, where), missing)
    arrays = {(r.id, r.lag): used[str(r)].to_numpy(float) for r in refs}
    Z = used[[f"__z__{z}" for z in z_names]].to_numpy(float) if z_names else None
    return _Arrays(lambda name, lag: arrays[(name, lag)], len(used), used.index, dropped, Z, z_names)


def _estimate(system: System, data: Dataset, method: Any, span: Any, where: str | None,
              start: Mapping[str, float] | None, missing: str) -> FittedSystem:
    if getattr(method, "fiml", False):
        return _fiml(system, data, method, span, where, start, missing)
    a = _arrays(system, data, span, where, method, missing)
    n, m = a.n, len(system.stochastic)
    names = list(system.param_names)
    k = len(names)
    pos = {p: j for j, p in enumerate(names)}
    rhs = [system._rhs(i) for i in range(m)]
    lhs = [np.broadcast_to(np.asarray(evaluate(eq.lhs, a.resolve), float), (n,)) for eq in system.stochastic]
    derivs = [[(pos[p], diff(rhs[i], Coef(p))) for p in rhs[i].params()] for i in range(m)]
    restr = system._restrictions()
    rgrad = [[(pos[p], diff(r, Coef(p))) for p in r.params()] for r in restr]

    if a.Z is not None:
        L = a.Z.shape[1]
        if np.linalg.matrix_rank(a.Z) < L:
            raise SpecificationError(f"the instruments are collinear: {a.z_names}")
        own = [len(system._own(i)) for i in range(m)]
        short = [system.names[i] for i in range(m) if own[i] > L]
        if short:
            raise SpecificationError(f"too few instruments ({L}) for {', '.join(short)}: each equation needs at least as many as its coefficients")
        Q, _ = np.linalg.qr(a.Z)
        project = lambda M: Q @ (Q.T @ M)  # noqa: E731
    else:
        project = lambda M: M  # noqa: E731

    def residuals(theta: np.ndarray) -> np.ndarray:
        values = dict(zip(names, theta))
        return np.column_stack([lhs[i] - np.broadcast_to(np.asarray(evaluate(rhs[i], a.resolve, values), float), (n,)) for i in range(m)])

    def jacobian(theta: np.ndarray) -> np.ndarray:
        """d u[t, i] / d theta[j] as an (n, m, k) array, projected on the instruments for IV methods."""
        values = dict(zip(names, theta))
        G = np.zeros((n, m, k))
        for i in range(m):
            for j, d in derivs[i]:
                G[:, i, j] = -np.broadcast_to(np.asarray(evaluate(d, a.resolve, values), float), (n,))
        if a.Z is not None:
            G = project(G.reshape(n, m * k)).reshape(n, m, k)
        return G

    def constraints(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = dict(zip(names, theta))
        g = np.array([float(evaluate(r, lambda *_: np.nan, values)) for r in restr])
        R = np.zeros((len(restr), k))
        for q, terms in enumerate(rgrad):
            for j, d in terms:
                R[q, j] = float(evaluate(d, lambda *_: np.nan, values))
        return g, R

    theta = np.zeros(k)
    for i, eq in enumerate(system.stochastic):  # nonlinear equations start from their own least-squares fit
        if eq.form == "explicit" and not start:
            try:
                single = eq.fit(data, span=span, where=where)
                for p, v in single.params.as_dict().items():
                    theta[pos[p]] = v
            except Exception:  # noqa: BLE001 - fall back to zeros
                pass
    for p, v in (start or {}).items():
        if p not in pos:
            raise SpecificationError(f"start gives {p!r}, which is not a parameter; parameters are {names}")
        theta[pos[p]] = float(v)

    def solve(theta: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, int]:
        """Gauss-Newton on sum_t u_t' S u_t (with projected derivatives), restrictions imposed exactly."""
        for it in range(1, method.max_iter + 1):
            U = residuals(theta)
            G = jacobian(theta)
            B = np.einsum("tim,ij,tjn->mn", G, S, G)
            grad = np.einsum("tim,ij,tj->m", G, S, U)
            if restr:
                g, R = constraints(theta)
                KKT = np.block([[B, R.T], [R, np.zeros((len(restr), len(restr)))]])
                sol = np.linalg.lstsq(KKT, np.r_[-grad, -g], rcond=None)[0]
                step = sol[:k]
            else:
                step = np.linalg.lstsq(B, -grad, rcond=None)[0]
            if not np.all(np.isfinite(step)):
                raise SpecificationError("the estimation failed (a step was not finite); check the starting values")
            current = _objective(U, S, project)
            scale = 1.0
            for _ in range(40):  # backtrack unless the step lowers the objective (or meets new restrictions)
                trial = theta + scale * step
                Ut = residuals(trial)
                if np.all(np.isfinite(Ut)) and (restr or _objective(Ut, S, project) <= current * (1 + 1e-12)):
                    break
                scale /= 2
            theta = trial
            if np.max(np.abs(scale * step)) < method.tol * (1 + np.max(np.abs(theta))):
                return theta, it
        raise SpecificationError(f"the estimates did not converge in {method.max_iter} iterations")

    first, iterations = solve(theta, np.eye(m))
    theta = first
    sigma = _sigma(residuals(theta), system, method, n)
    rounds = 1
    if method.weighted:
        while True:
            S = np.linalg.inv(sigma)
            new, its = solve(theta, S)
            iterations += its
            rounds += 1
            change = np.max(np.abs(new - theta)) / (1 + np.max(np.abs(new)))
            theta = new
            new_sigma = _sigma(residuals(theta), system, method, n)
            if not method.iterate or change < method.tol:
                if method.iterate:
                    sigma = new_sigma
                break
            sigma = new_sigma
            if rounds > method.max_iter:
                raise SpecificationError("iterating on the residual covariance did not converge")
        S = np.linalg.inv(sigma)
    else:
        S = np.eye(m)
    U = residuals(theta)
    G = jacobian(theta)
    sigma_final = _sigma(U, system, method, n)
    cov = _covariance(G, U, S, sigma_final if not method.weighted else sigma, method, constraints(theta)[1] if restr else None)
    return _result(system, data, method, a, theta, cov, U, sigma_final, iterations, rounds, S, project, len(restr))


def _fiml(system: System, data: Dataset, method: Any, span: Any, where: str | None, start: Mapping[str, float] | None,
          missing: str) -> FittedSystem:
    """Full-information maximum likelihood, from 3SLS (or least squares, when the instruments are too few)."""
    from scipy import optimize

    if system.restrictions.strip():
        raise SpecificationError("FIML with restrictions is not supported yet: share parameter names across equations instead")
    if method.cov != "classic" or method.df_correction:
        raise SpecificationError("FIML's standard errors come from the inverse Hessian: cov='classic', without df_correction")
    endo = list(system.endogenous_variables)
    eqs = list(system.stochastic) + list(system.identities)
    if len(eqs) != len(endo):
        raise SpecificationError(f"FIML needs as many equations, identities included, as endogenous variables: {len(eqs)} "
                                 f"equations for {len(endo)} ({', '.join(endo)})")
    m = len(system.stochastic)
    exprs = [system.stochastic[i].lhs - system._rhs(i) for i in range(m)] + [eq.lhs - eq.rhs for eq in system.identities]
    refs = set().union(*(e.refs() for e in exprs))
    index = data.frame.index
    cols = {str(r): data.lagged(r.id, r.lag).astype(float) for r in sorted(refs, key=str)}
    used, dropped = sample(pd.DataFrame(cols, index=index), data.sample_mask(span, where), missing)
    arrays = {(r.id, r.lag): used[str(r)].to_numpy(float) for r in refs}
    resolve = lambda name, lag: arrays[(name, lag)]  # noqa: E731
    n = len(used)
    names = list(system.param_names)
    jac = [[diff(e, Name(g, 0)) for g in endo] for e in exprs]
    constant = all(not d.refs() for row in jac for d in row)

    def residuals(theta: np.ndarray) -> np.ndarray:
        values = dict(zip(names, theta))
        return np.column_stack([np.broadcast_to(np.asarray(evaluate(exprs[i], resolve, values), float), (n,)) for i in range(m)])

    def log_jacobian(theta: np.ndarray) -> float:
        values = dict(zip(names, theta))
        if constant:
            J = np.array([[float(evaluate(d, resolve, values)) for d in row] for row in jac])
            sign, logdet = np.linalg.slogdet(J)
            return n * logdet if sign != 0 else -np.inf
        J = np.stack([np.stack([np.broadcast_to(np.asarray(evaluate(d, resolve, values), float), (n,)) for d in row], axis=-1)
                      for row in jac], axis=1)
        sign, logdet = np.linalg.slogdet(J)
        return float(logdet.sum()) if np.all(sign != 0) else -np.inf

    def loglike(theta: np.ndarray) -> float:
        U = residuals(theta)
        if not np.all(np.isfinite(U)):
            return -np.inf
        sign, logdet = np.linalg.slogdet(U.T @ U / n)
        if sign <= 0:
            return -np.inf
        return log_jacobian(theta) - n / 2 * (m * (1 + np.log(2 * np.pi)) + logdet)

    try:
        first = _estimate(system, data, ThreeSLS(max_iter=method.max_iter), span, where, start, missing)
    except SpecificationError:
        first = _estimate(system, data, SystemOLS(max_iter=method.max_iter), span, where, start, missing)
    theta0 = np.array([first.params.as_dict()[p] for p in names])
    f = lambda th: -loglike(th) / n  # noqa: E731
    found = optimize.minimize(f, theta0, method="BFGS", options={"gtol": 1e-10, "maxiter": 20 * method.max_iter})
    polish = optimize.minimize(f, found.x, method="Nelder-Mead",
                               options={"xatol": method.tol, "fatol": 1e-15, "maxiter": 200 * len(names) * 10})
    theta = polish.x if polish.fun <= found.fun else found.x
    if not np.isfinite(f(theta)):
        raise SpecificationError("FIML failed: the likelihood is not finite at the estimates (a singular Jacobian?)")
    k = len(names)
    H = np.empty((k, k))
    h = 1e-3 * np.maximum(np.abs(theta), 0.1)  # not smaller: the log likelihood is large, and differences cancel
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = h[i], h[j]
            H[i, j] = H[j, i] = (loglike(theta + ei + ej) - loglike(theta + ei - ej) - loglike(theta - ei + ej)
                                 + loglike(theta - ei - ej)) / (4 * h[i] * h[j])
    cov = np.linalg.pinv(-H)
    U = residuals(theta)
    sigma = U.T @ U / n
    a = _Arrays(resolve, n, used.index, dropped, None, [])
    fit = _result(system, data, method, a, theta, cov, U, sigma, int(found.nit), 1, np.linalg.inv(sigma), lambda M: M, 0)
    fit.stats["loglike"] = float(loglike(theta))
    fit.stats["log_jacobian"] = float(log_jacobian(theta))
    return fit


def _objective(U: np.ndarray, S: np.ndarray, project: Any) -> float:
    PU = project(U)
    return float(np.einsum("ti,ij,tj->", PU, S, PU))


def _sigma(U: np.ndarray, system: System, method: Any, n: int) -> np.ndarray:
    cross = U.T @ U
    if not method.df_correction:
        return cross / n
    k = np.array([len(system._own(i)) for i in range(U.shape[1])], float)
    return cross / np.sqrt(np.outer(n - k, n - k))


def _covariance(G: np.ndarray, U: np.ndarray, S: np.ndarray, sigma: np.ndarray, method: Any, R: np.ndarray | None) -> np.ndarray:
    """Sandwich ``B^-1 M B^-1`` with ``B = sum G_t' S G_t``; restrictions through the null space of R."""
    n, m, k = G.shape
    B = np.einsum("tim,ij,tjn->mn", G, S, G)
    if method.cov == "classic":
        M = np.einsum("tim,ij,jl,lk,tkn->mn", G, S, sigma, S, G)
    else:
        h = np.einsum("tim,ij,tj->tm", G, S, U)  # score contributions
        M = h.T @ h
        if method.cov == "hac":
            lags = default_hac_lags(n) if method.hac_lags is None else method.hac_lags
            for lag in range(1, lags + 1):
                gamma = h[lag:].T @ h[:-lag]
                M += (1 - lag / (lags + 1)) * (gamma + gamma.T)
    N = np.eye(k) if R is None or not len(R) else _null_space(R)
    inner = np.linalg.pinv(N.T @ B @ N)
    return N @ inner @ N.T @ M @ N @ inner @ N.T


def _null_space(R: np.ndarray) -> np.ndarray:
    _, s, vt = np.linalg.svd(R)
    rank = int(np.sum(s > 1e-10 * max(1.0, s.max(initial=0.0))))
    return vt[rank:].T


# ---------------------------------------------------------------------------------------------
# results


def _result(system: System, data: Dataset, method: Any, a: _Arrays, theta: np.ndarray, cov: np.ndarray, U: np.ndarray,
            sigma: np.ndarray, iterations: int, rounds: int, S: np.ndarray, project: Any, n_restrictions: int) -> FittedSystem:
    from .estimation.design import ls_stats, provenance

    names = list(system.param_names)
    params = point(names, theta, cov)
    n, m = U.shape
    fits = []
    for i, eq in enumerate(system.stochastic):
        own = system._own(i)
        idx = [names.index(full) for full, _ in own]
        p = point([o for _, o in own], theta[idx], cov[np.ix_(idx, idx)])
        y = np.broadcast_to(np.asarray(evaluate(eq.lhs, a.resolve), float), (n,))
        u = U[:, i]
        stats = ls_stats(y, u, None, len(idx), n - len(idx), centered=True, time_series=data.kind == "time_series", within=False)
        fits.append(FittedEquation(eq, p, method, resid=pd.Series(u, index=a.index, name=system.names[i]),
                                   fitted=pd.Series(y - u, index=a.index, name=system.names[i]), stats=stats,
                                   provenance=provenance(eq, method, data, a.index, a.dropped),
                                   extra={"system": system.names, "instrument_names": a.z_names} if a.Z is not None else {"system": system.names}))
    corr = sigma / np.sqrt(np.outer(np.diag(sigma), np.diag(sigma)))
    stats: dict[str, float] = {"nobs": float(n), "equations": float(m), "k": float(len(names) - n_restrictions),
                               "restrictions": float(n_restrictions), "iterations": float(iterations),
                               "log_det_sigma": float(np.linalg.slogdet(sigma)[1])}
    stats["loglike"] = float(-n / 2 * (m * (1 + np.log(2 * np.pi)) + stats["log_det_sigma"]))
    if m > 1:
        lm = n * float(np.sum(np.triu(corr, 1) ** 2))
        stats["bp_diagonal"] = lm
        stats["bp_diagonal_df"] = m * (m - 1) / 2
        stats["bp_diagonal_pvalue"] = float(sps.chi2.sf(lm, m * (m - 1) / 2))
    if a.Z is not None and method.weighted:
        df = m * a.Z.shape[1] - (len(names) - n_restrictions)
        Sj = np.linalg.inv(sigma)
        J = _objective(U, Sj, project)
        if df > 0:
            stats["j"] = J
            stats["j_df"] = float(df)
            stats["j_pvalue"] = float(sps.chi2.sf(J, df))
    t = data.time_axis
    span = (str(a.index.get_level_values(t.name).min()), str(a.index.get_level_values(t.name).max())) if t is not None else None
    variables = sorted(set().union(*(eq.variables() for eq in system.stochastic)))
    prov = Provenance(data_vintages={v: data.variables[v].vintage for v in variables if v in data.variables},
                      spec={"type": "System", "equations": {nm: eq.text for nm, eq in zip(system.names, system.stochastic)},
                            "identities": [eq.text for eq in system.identities], "instruments": list(system.instruments),
                            "endogenous": list(system.endogenous), "restrictions": system.restrictions},
                      method=to_jsonable(method), sample=span, nobs=n, dropped=a.dropped, package_version=__version__,
                      data_fingerprint=data_fingerprint(data, variables), backend=native.backend_record(f"pymetrix.{type(method).__name__}"))
    labels = list(system.names)
    return FittedSystem(system, method, params, tuple(fits), pd.DataFrame(sigma, index=labels, columns=labels),
                        pd.DataFrame(U, index=a.index, columns=labels), stats, prov, tuple(a.z_names))


@dataclass(frozen=True, eq=False)
class FittedSystem:
    """Estimates of a system: all parameters with their joint covariance, and each equation as a ``FittedEquation``."""

    system: System
    method: Any
    params: PointEstimate
    equations: tuple[FittedEquation, ...]
    sigma: pd.DataFrame  # residual covariance across equations
    resid: pd.DataFrame
    stats: Mapping[str, float]
    provenance: Provenance
    instrument_names: tuple[str, ...] = ()

    def __getitem__(self, name: str) -> FittedEquation:
        try:
            return self.equations[self.system.names.index(name)]
        except ValueError:
            raise NotFoundError(f"no equation {name!r}; equations are {list(self.system.names)}") from None

    @property
    def names(self) -> tuple[str, ...]:
        return self.system.names

    @property
    def correlation(self) -> pd.DataFrame:
        d = np.sqrt(np.diag(self.sigma.to_numpy()))
        return self.sigma / np.outer(d, d)

    def summary(self) -> Table:
        p = self.params
        z = p.values / p.se
        pv = 2 * sps.norm.sf(np.abs(z))
        rows = [[v, s, zz, q] for v, s, zz, q in zip(p.values, p.se, z, pv)]
        s = self.stats
        notes = [f"Method: {getattr(self.method, 'label', type(self.method).__name__)}"
                 + (", iterated" if getattr(self.method, "iterate", False) and getattr(self.method, "weighted", False) else "")
                 + f"; covariance {self.method.cov}"]
        sample = f"{self.provenance.sample[0]} to {self.provenance.sample[1]}, " if self.provenance.sample else ""
        notes.append(f"Sample: {sample}{self.provenance.nobs} observations ({self.provenance.dropped} dropped), common to all equations")
        if self.system.restrictions:
            notes.append(f"Restrictions: {self.system.restrictions}")
        if self.instrument_names:
            notes.append("Instruments: " + ", ".join(self.instrument_names))
        for name, fit in zip(self.names, self.equations):
            notes.append(f"{name}: {fit.equation.text}   R2 {fit.stats['r2']:.4f}, S.E. {fit.stats['se_regression']:.4f}"
                         + (f", DW {fit.stats['durbin_watson']:.3f}" if "durbin_watson" in fit.stats else ""))
        if "bp_diagonal" in s:
            notes.append(f"Breusch-Pagan test of a diagonal covariance: {s['bp_diagonal']:.3f} (df {s['bp_diagonal_df']:.0f}, p = {s['bp_diagonal_pvalue']:.4f})")
        if "j" in s:
            notes.append(f"Hansen-Sargan test of the overidentifying restrictions: {s['j']:.3f} (df {s['j_df']:.0f}, p = {s['j_pvalue']:.4f})")
        notes += [f"Log likelihood: {s['loglike']:.3f}", f"Log det residual covariance: {s['log_det_sigma']:.4f}"]
        title = f"System: {', '.join(self.names)}"
        return Table(["coef", "std err", "z", "p>|z|"], rows, index=list(p.names), title=title, notes=notes)

    def equation_stats(self) -> Table:
        keys = ("nobs", "k", "r2", "r2_adj", "se_regression", "ssr", "durbin_watson")
        rows = [[name] + [fit.stats.get(k, np.nan) for k in keys] for name, fit in zip(self.names, self.equations)]
        return Table(["equation", "obs", "coefficients", "R2", "adj. R2", "S.E.", "SSR", "DW"], rows, title="Equations")

    def wald(self, restriction: str) -> Any:
        """Wald test of restrictions across equations, e.g. ``"c.y = i.y"`` or ``"a + b = 1"``."""
        from .stats.equation_tests import Wald

        shim = SimpleNamespace(params=self.params, stats={})  # Wald reads any coefficient name (c.dlog(y) too)
        return Wald(restriction).run(shim)  # type: ignore[arg-type]

    def test_diagonal(self) -> Any:
        """Breusch-Pagan LM test that the errors are uncorrelated across equations (then SUR gains nothing)."""
        from .stats import TestResult

        if "bp_diagonal" not in self.stats:
            raise SpecificationError("a single equation has no cross-equation correlation to test")
        s = self.stats
        return TestResult("Breusch-Pagan (diagonal covariance)", "the errors are uncorrelated across equations",
                          s["bp_diagonal"], s["bp_diagonal_df"], s["bp_diagonal_pvalue"], "chi2")

    def model(self, *others: Any, **kw: Any) -> Any:
        """A ``Model`` of the estimated equations, the system's identities, and ``others``."""
        from .models import Model

        return Model([*self.equations, *self.system.identities, *others], **kw)

    def plot(self) -> Any:
        """Actual and fitted values of each equation."""
        from .figures import Figure, Line, Panel

        panels = []
        for name, fit in zip(self.names, self.equations):
            actual = (fit.fitted + fit.resid).rename("actual")
            panels.append(Panel((Line(actual, "actual"), Line(fit.fitted, "fitted")), title=name))
        return Figure(tuple(panels), columns=2 if len(panels) > 1 else 1)

    def __repr__(self) -> str:
        return f"<FittedSystem {', '.join(self.names)}, {type(self.method).__name__}, {len(self.params)} parameters>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()

    def __str__(self) -> str:
        return str(self.summary())

