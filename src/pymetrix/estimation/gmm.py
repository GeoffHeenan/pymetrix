"""Generalised method of moments for one equation: moments E[z_t u_t(theta)] = 0, with u the equation's residual.

    euler = Equation("1 ~ beta * (c / c[-1])^(-gamma) * gross_r", params=["beta", "gamma"])    # an Euler equation
    euler.fit(data, method=GMM(instruments=["c[-1]/c[-2]", "gross_r[-1]"], cov="hac"))
    Equation("y ~ x + w").fit(data, method=GMM(instruments=["w", "z1", "z2"]))                # linear: two-step IV-GMM

The instruments are a constant (unless ``constant=False``) and the expressions listed: list every instrument,
including regressors that are their own. The first step weights the moments by (Z'Z / n)^-1, which is two-stage
least squares; ``steps="two"`` re-weights once by the inverse of their estimated covariance S, ``"iterated"`` until
the estimates settle, ``"cue"`` is Hansen, Heaton and Yaron's continuously updated estimator (S evaluated at every
theta; with ``cov="classic"`` in a linear model it is LIML), and ``weights="identity"`` keeps W = I in a single step.

Tests and estimators beside the fit: ``c_test`` (difference-in-Sargan: are some instruments valid?),
``anderson_rubin`` and ``stock_wright`` (tests of parameter values that stay valid when the instruments are weak, and
the confidence sets they give), and ``minimum_distance`` (classical minimum distance from estimates and their
covariance). S is heteroskedasticity-robust
(``cov="white"``, the default), assumes homoskedastic errors (``"classic"``) or allows autocorrelation as well
(``"hac"``, Newey-West with the Bartlett kernel). Hansen's J statistic tests the over-identifying restrictions.

``MomentGMM`` takes moment conditions written as a Python function of the parameters and the data instead of an
equation and instruments (several equations at once, moments of a distribution): see its docstring.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from pymetrix._lazy import stats as sps

from .._api import api
from ..errors import SpecificationError
from ..forms import Widget
from ..plugins import register

if TYPE_CHECKING:
    from ..data import Dataset
    from .equation import Equation
    from .fitted import FittedEquation


@dataclass(frozen=True)
class GMM:
    """Generalised method of moments: the equation's residual is orthogonal to the instruments (Hansen 1982).

    List every instrument, the exogenous regressors included; the constant is added unless switched off. With as
    many instruments as parameters GMM is just IV; with more, the efficient weighting gives estimates asymptotically
    at least as precise as 2SLS when the errors are heteroskedastic or (hac) autocorrelated, and Hansen's J tests
    the over-identifying restrictions: a small p-value says some instruments are invalid or the model is wrong, but
    a large one cannot show that all are valid. The instruments must be relevant as well as valid: with weak
    instruments the estimates are biased and the standard errors too small. Many instruments, relative to the
    sample, make the two-step estimates biased towards least squares, weaken the J test and understate the standard
    errors; keep the set small. The continuously updated estimator (CUE) is less biased with many or weak
    instruments, but its objective can have several minima.
    """

    maturity: ClassVar[str] = "beta"
    label: ClassVar[str] = "GMM"
    topics: ClassVar[tuple[str, ...]] = ("instrumental-variables",)
    references: ClassVar[tuple[str, ...]] = (
        ("Hansen, L. P. (1982). Large sample properties of generalized method of moments estimators. Econometrica 50(4), 1029-1054. doi:10.2307/1912775"),
        ("Hansen, L. P., Heaton, J. and Yaron, A. (1996). Finite-sample properties of some alternative GMM estimators. Journal of Business and Economic Statistics 14(3), 262-280. doi:10.1080/07350015.1996.10524656"),
    )
    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation
        from pymetrix.estimation import GMM

        card = Dataset.example("card")           # Card (1995): returns to schooling, college proximity as instruments
        fit = Equation("lwage ~ educ + exper + black").fit(
            card, method=GMM(instruments=["nearc4", "nearc2", "exper", "black"]))
        print(fit.summary())                     # ends with Hansen's J test of the one over-identifying restriction
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("coef, std err, z", "the estimates with standard errors from the moments' covariance S (white, classic or hac)"),
        ("Instruments", "every instrument used, the constant included: count them against the parameters"),
        ("moments for parameters", "more moments than parameters means over-identified, and the J test below applies"),
        ("Hansen's J", "chi-squared with (moments - parameters) degrees of freedom; a small p-value rejects the "
                       "over-identifying restrictions (an invalid instrument or a wrong model), a large one does not "
                       "prove the instruments valid"),
        ("fit.stats", "J, J_df and J_pvalue for use in code, with moments and k (the parameters)"),
    )
    kind: ClassVar[str] = "iv"  # the Estimate dialog's instruments field

    instruments: Annotated[Sequence[str], Widget("", help="every instrument, exogenous regressors included (the constant is "
                                                     "added): at least one per parameter")] = ()
    weights: Annotated[Literal["optimal", "identity"], Widget("choice", help="optimal: weighted by the moments' covariance; identity: W = I, one step")] = "optimal"
    steps: Annotated[Literal["two", "iterated", "cue"], Widget("choice", help="two-step, iterated until the estimates settle, or continuously updated (CUE)")] = "two"
    cov: Annotated[str, Widget("", help="covariance S of the moments, which weights them and gives the standard errors: "
                                        "white, classic or hac")] = "white"  # "white" (robust), "classic" (homoskedastic) or "hac"
    hac_lags: Annotated[int, Widget("number", min=0, max=100, advanced=True, help="Newey-West lags; 0 = automatic")] = 0
    constant: Annotated[bool, Widget("checkbox", advanced=True, help="a constant among the instruments")] = True
    start: Annotated[Mapping[str, float] | None, Widget("", help="starting values for explicit-form equations, e.g. beta=0.99, "
                                                                "gamma=2; empty = 0.1 or the Param's")] = None
    max_iter: Annotated[int, Widget("number", min=1, max=1000, advanced=True, help="iterations of the iterated estimator")] = 100
    tol: Annotated[float, Widget("number", min=1e-14, max=1e-2, advanced=True,
                                 help="iterated estimator: stop when the estimates change by less than this (relative)")] = 1e-8
    endog: Annotated[Sequence[str], Widget("", help="not used by GMM: list the instruments in full instead")] = field(default=(), repr=False)  # accepted from the dialog's IV fields; GMM needs only instruments

    def estimate(self, eq: Equation, data: Dataset, *, span: Any = None, where: Any = None, weights: Any = None,
                 missing: str = "drop") -> FittedEquation:
        return _gmm(self, eq, data, span, where, weights, missing)


@dataclass
class _Moments:
    """The residual u(theta), its derivative G = du/dtheta and the instruments Z, on the estimation sample."""

    u: Any
    G: Any
    Z: np.ndarray
    z_names: list[str]
    names: list[str]
    index: pd.Index
    lhs: np.ndarray
    start: np.ndarray
    bounds: tuple[np.ndarray, np.ndarray]
    dropped: int


def _moments(m: GMM, eq: Equation, data: Dataset, span: Any, where: Any, weights: Any, missing: str) -> _Moments:
    from .design import explicit_design, linear_design
    from .methods import _Nonlinear, _start, instrument_series

    if weights:
        raise SpecificationError("GMM does not take weights")
    if not m.instruments:
        raise SpecificationError("GMM needs instruments=[...]: list every instrument (the constant is added)")
    series = instrument_series(list(m.instruments), data)
    if eq.form == "linear":
        if eq.fixed_effects or eq.categoricals:
            raise NotImplementedError("GMM with fe() or C() terms")
        d = linear_design(eq, data, span, where, None, missing, instruments=series)
        y, X = d.y, d.X

        def u(theta: np.ndarray) -> np.ndarray:
            return y - X @ theta

        def G(theta: np.ndarray) -> np.ndarray:
            return -X

        Zx, z_names, index, names, lhs, dropped = d.Z, list(d.z_names), d.index, list(d.names), y, d.dropped
        start = np.zeros(X.shape[1])
        bounds = (np.full(len(names), -np.inf), np.full(len(names), np.inf))
    else:
        d = explicit_design(eq, data, span, where, None, missing, instruments=series)
        nl = _Nonlinear(eq, d)
        u, G = nl.f, nl.jac
        Zx, z_names, index, names, lhs, dropped = d.Z, list(d.z_names), d.index, list(nl.names), nl.lhs(eq), d.dropped
        start = _start(eq, m.start)
        bounds = (np.array([p.bounds[0] for p in eq.param_specs]), np.array([p.bounds[1] for p in eq.param_specs]))
    n = len(index)
    parts = ([np.ones((n, 1))] if m.constant else []) + ([Zx] if Zx is not None else [])
    if not parts:
        raise SpecificationError("no instruments")
    Z = np.column_stack(parts)
    z_names = (["const"] if m.constant else []) + z_names
    if Z.shape[1] < len(names):
        raise SpecificationError(f"the equation is underidentified: {Z.shape[1]} instruments for {len(names)} parameters")
    if np.linalg.matrix_rank(Z) < Z.shape[1]:
        raise SpecificationError("the instruments are collinear: drop one")
    return _Moments(u, G, Z, z_names, names, index, np.broadcast_to(lhs, (n,)), np.asarray(start, float), bounds, dropped)


def _S(kind: str, Z: np.ndarray, u: np.ndarray, lags: int) -> np.ndarray:
    """The covariance of the moments sqrt(n) mean(z u): robust, homoskedastic or Newey-West."""
    from .covariance import default_hac_lags

    n = len(u)
    g = Z * u[:, None]
    if kind == "classic":
        return (u @ u / n) * (Z.T @ Z) / n
    S = g.T @ g / n
    if kind == "hac":
        L = lags or default_hac_lags(n)
        for j in range(1, L + 1):
            Gamma = g[j:].T @ g[:-j] / n
            S = S + (1 - j / (L + 1)) * (Gamma + Gamma.T)
    elif kind != "white":
        raise SpecificationError(f"GMM's cov is 'white', 'classic' or 'hac', not {kind!r}")
    return S


def _cue_objective(m: GMM, mo: _Moments, theta: np.ndarray) -> float:
    """n gbar(theta)' S(theta)^-1 gbar(theta): the continuously updated objective."""
    u = mo.u(theta)
    if not np.all(np.isfinite(u)):
        return 1e300
    gbar = mo.Z.T @ u / len(u)
    try:
        return float(len(u) * gbar @ np.linalg.solve(_S(m.cov, mo.Z, u, m.hac_lags), gbar))
    except np.linalg.LinAlgError:
        return 1e300


def _cue(m: GMM, mo: _Moments, theta0: np.ndarray) -> np.ndarray:
    """The continuously updated estimate, from the two-step one (BFGS, then Nelder-Mead to polish)."""
    from scipy.optimize import minimize

    f = lambda th: _cue_objective(m, mo, th)  # noqa: E731
    first = minimize(f, theta0, method="BFGS", options={"gtol": 1e-10, "maxiter": 2000})
    best = first if first.fun <= f(theta0) else type(first)(x=theta0, fun=f(theta0))
    polish = minimize(f, best.x, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    return polish.x if polish.fun <= best.fun else best.x


def _minimise(mo: _Moments, W: np.ndarray, theta0: np.ndarray, linear: bool) -> np.ndarray:
    """The theta minimising n gbar' W gbar, gbar = Z'u / n: closed form when linear, else least squares on W^1/2 gbar."""
    n = len(mo.index)
    if linear:
        X = -mo.G(theta0)
        y = mo.u(np.zeros_like(theta0))
        A = X.T @ mo.Z @ W @ mo.Z.T @ X
        return np.linalg.solve(A, X.T @ mo.Z @ W @ mo.Z.T @ y)
    C = np.linalg.cholesky(W).T  # W = C'C

    def r(theta: np.ndarray) -> np.ndarray:
        return C @ (mo.Z.T @ mo.u(theta)) / np.sqrt(n)

    def jac(theta: np.ndarray) -> np.ndarray:
        return C @ (mo.Z.T @ mo.G(theta)) / np.sqrt(n)

    lo, hi = mo.bounds
    bounded = bool(np.isfinite(lo).any() or np.isfinite(hi).any())
    if not np.all(np.isfinite(r(theta0))):
        raise SpecificationError("the equation cannot be evaluated at the starting values; set start={...}")
    found = least_squares(r, theta0, jac=jac, bounds=(lo, hi) if bounded else (-np.inf, np.inf), method="trf",
                          xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=5000)
    return found.x


def _gmm(m: GMM, eq: Equation, data: Dataset, span: Any, where: Any, weights: Any, missing: str) -> FittedEquation:
    from .design import provenance
    from .fitted import FittedEquation
    from .params import point

    if m.cov == "hac" and data.time_axis is None:
        raise SpecificationError("HAC weighting needs a time series")
    mo = _moments(m, eq, data, span, where, weights, missing)
    linear = eq.form == "linear"
    n, L, p = len(mo.index), mo.Z.shape[1], len(mo.names)
    W = np.eye(L) if m.weights == "identity" else np.linalg.inv(mo.Z.T @ mo.Z / n)
    theta = _minimise(mo, W, mo.start, linear)
    steps = 1
    if m.steps == "cue" and m.weights != "optimal":
        raise SpecificationError("the continuously updated estimator weights optimally: weights='optimal'")
    if m.weights == "optimal":
        rounds = 1 if m.steps in ("two", "cue") else m.max_iter
        for _ in range(rounds):
            S = _S(m.cov, mo.Z, mo.u(theta), m.hac_lags)
            W = np.linalg.inv(S)
            new = _minimise(mo, W, theta, linear)
            change = np.max(np.abs(new - theta)) / (1 + np.max(np.abs(theta)))
            theta, steps = new, steps + 1
            if change < m.tol:
                break
    if m.steps == "cue":
        theta = _cue(m, mo, theta)
        steps += 1
        W = np.linalg.inv(_S(m.cov, mo.Z, mo.u(theta), m.hac_lags))
    u = mo.u(theta)
    S = _S(m.cov, mo.Z, u, m.hac_lags)
    D = mo.Z.T @ mo.G(theta) / n  # derivative of the mean moments
    bread = np.linalg.inv(D.T @ W @ D)
    cov = bread @ D.T @ W @ S @ W @ D @ bread / n
    gbar = mo.Z.T @ u / n
    # Hansen's J: the minimised objective, n gbar' W gbar, with the weights of the last step (efficient weighting);
    # with W = I that is no test, so S at the estimates is used instead
    J = float(n * gbar @ (W @ gbar if m.weights == "optimal" else np.linalg.solve(S, gbar)))
    df = L - p
    stats = {"nobs": float(n), "k": float(p), "moments": float(L), "J": J, "J_pvalue": float(sps.chi2.sf(J, df)) if df > 0 else float("nan"),
             "J_df": float(df), "steps": float(steps), "ssr": float(u @ u)}
    if m.cov == "hac":
        from .covariance import default_hac_lags

        stats["hac_lags"] = float(m.hac_lags or default_hac_lags(n))
    weighting = ("W = I, one step" if m.weights == "identity" else "continuously updated (CUE), S " + m.cov if m.steps == "cue" else
                 f"{'two-step' if m.steps == 'two' else f'iterated ({steps} steps)'} efficient weighting, S {m.cov}")
    notes = [f"GMM: {L} moments for {p} parameters; {weighting}.",
             (f"Hansen's J = {J:.4f}, chi-squared({df}), p = {stats['J_pvalue']:.4f}: a small p-value rejects the over-identifying restrictions."
              if df > 0 else "Exactly identified: no over-identifying restrictions to test.")]
    return FittedEquation(
        equation=eq, params=point(mo.names, theta, cov), method=m,
        resid=pd.Series(u, index=mo.index, name="resid"), fitted=pd.Series(mo.lhs - u, index=mo.index, name="fitted"),
        stats=stats, provenance=provenance(eq, m, data, mo.index, mo.dropped, extra_variables=list(m.instruments)),
        design=pd.DataFrame(-mo.G(theta), index=mo.index, columns=mo.names),
        extra={"instruments": pd.DataFrame(mo.Z, index=mo.index, columns=mo.z_names), "notes": notes, "converged": True})


register("methods", GMM)


# ---------------------------------------------------------------------------------------------
# GMM on moment functions written in Python


def _S_moments(kind: str, g: np.ndarray, lags: int, center: bool) -> np.ndarray:
    """The covariance of sqrt(n) times the mean of the moment contributions ``g`` (n x L): robust or Newey-West."""
    from .covariance import default_hac_lags

    n = len(g)
    if center:
        g = g - g.mean(axis=0)
    S = g.T @ g / n
    if kind == "hac":
        L = lags or default_hac_lags(n)
        for j in range(1, L + 1):
            Gamma = g[j:].T @ g[:-j] / n
            S = S + (1 - j / (L + 1)) * (Gamma + Gamma.T)
    elif kind != "white":
        raise SpecificationError(f"GMM on moment functions takes cov='white' or 'hac', not {kind!r} (there is no residual "
                                 "to be homoskedastic)")
    return S


@dataclass(frozen=True)
class MomentGMM:
    """GMM on moment conditions written as a Python function: ``moments(params, frame)`` returns each period's (or
    unit's) contributions to the moments, an n x L array or DataFrame, one row per row of ``frame``; GMM sets their
    means to zero as closely as the weights allow (Hansen 1982).

        def euler(p, d):                       # E[(beta (c/c[-1])^-gamma R - 1) z] = 0
            u = p["beta"] * (d.c / d.c.shift(1)) ** -p["gamma"] * d.r - 1
            return np.column_stack([u, u * d.c.shift(1) / d.c.shift(2), u * d.r.shift(1)])

        MomentGMM(euler, start={"beta": 0.99, "gamma": 2.0}).fit(data)

    The function gets the whole frame (so lags reach before the span); the rows used are those in ``span`` whose
    moments are all finite at the starting values. The first step weights the moments equally (W = I, as statsmodels'
    GMM); ``steps="two"`` re-weights once by the inverse of their covariance S, ``"iterated"`` until the estimates
    settle, ``"cue"`` continuously updates S. S is robust (``cov="white"``) or Newey-West (``"hac"``), from the
    contributions as they are or ``center``-ed. Derivatives are numerical. Hansen's J tests the over-identifying
    restrictions. Written in code only: a function cannot be recorded as a dialog step."""

    moments: Any
    start: Mapping[str, float]
    weights: Literal["optimal", "identity"] = "optimal"
    steps: Literal["two", "iterated", "cue"] = "two"
    cov: Literal["white", "hac"] = "white"
    hac_lags: int = 0
    center: bool = False
    bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    max_iter: int = 100
    tol: float = 1e-8

    def __post_init__(self) -> None:
        if not callable(self.moments):
            raise SpecificationError("moments is a function: moments(params, frame) -> an n x L array of moment contributions")
        if not self.start:
            raise SpecificationError("give the parameters' starting values: start={'beta': 0.99, ...}")
        unknown = sorted(set(self.bounds) - set(self.start))
        if unknown:
            raise SpecificationError(f"bounds name {', '.join(unknown)}, which are not parameters (start)")
        if self.weights not in ("optimal", "identity") or self.steps not in ("two", "iterated", "cue"):
            raise SpecificationError("weights is 'optimal' or 'identity'; steps 'two', 'iterated' or 'cue'")
        if self.steps == "cue" and self.weights != "optimal":
            raise SpecificationError("the continuously updated estimator weights optimally: weights='optimal'")

    def fit(self, data: Any, span: Any = None) -> FittedGMM:
        """Estimate from ``data`` (a Dataset or DataFrame) over ``span``: a ``FittedGMM``."""
        return _moment_gmm(self, data, span)


@dataclass(frozen=True, eq=False)
class FittedGMM:
    """GMM estimates from moment functions: ``params`` (values and covariance), ``stats`` (J, its p-value and degrees
    of freedom, observations, moments, steps), ``contributions`` (the moments at the estimates, by row) and notes."""

    params: Any
    stats: Mapping[str, float]
    contributions: pd.DataFrame
    method: MomentGMM
    notes: tuple[str, ...] = ()
    provenance: Any = None

    @property
    def J(self) -> float:
        return float(self.stats["J"])

    def summary(self) -> Any:
        from ..output import Table

        p = self.params
        z = np.asarray(p.values) / np.asarray(p.se)
        rows = [[float(v), float(se), float(t), float(2 * sps.norm.sf(abs(t)))] for v, se, t in zip(p.values, p.se, z)]
        means = self.contributions.mean()
        notes = [*self.notes, "Moments at the estimates (means): " + ", ".join(f"{k} {v:.3g}" for k, v in means.items())]
        return Table(["coef", "std err", "z", "p>|z|"], rows, index=list(p.names), title="GMM on moment conditions", notes=notes)

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        return self.params.to_frame()

    def __repr__(self) -> str:
        return f"<FittedGMM: {len(self.params.names)} parameters, {int(self.stats['moments'])} moments, J = {self.J:.4g}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


def _moment_gmm(m: MomentGMM, data: Any, span: Any) -> FittedGMM:
    from scipy.optimize import minimize

    from .._capture import data_fingerprint
    from .._version import __version__
    from ..data import Dataset
    from ..metadata import Provenance
    from .params import point

    dataset = data if isinstance(data, Dataset) else None  # a plain DataFrame (a cross-section, say) is used as it is
    frame = dataset.frame if dataset is not None else pd.DataFrame(data)
    timed = dataset.time_axis is not None if dataset is not None else isinstance(frame.index, pd.PeriodIndex)
    names = list(m.start)
    theta0 = np.array([float(m.start[k]) for k in names])

    def contributions(theta: np.ndarray) -> Any:
        out = m.moments(dict(zip(names, map(float, theta))), frame)
        return out

    first = contributions(theta0)
    columns = list(first.columns) if isinstance(first, pd.DataFrame) else None
    g0 = np.asarray(first, dtype=float)
    if g0.ndim == 1:
        g0 = g0[:, None]
    if g0.shape[0] != len(frame):
        raise SpecificationError(f"the moment function returned {g0.shape[0]} rows for a frame of {len(frame)}: one row per row")
    rows = np.all(np.isfinite(g0), axis=1)
    if span is not None:
        if dataset is not None:
            rows &= np.asarray(dataset.sample_mask(span))
        elif timed:
            from ..data.sample import as_span

            first, last = (pd.Period(str(v), freq=frame.index.freq) for v in as_span(span))
            rows &= np.asarray((frame.index >= first) & (frame.index <= last))
        else:
            raise SpecificationError("span= needs periods: a Dataset with a time axis, or a frame with a PeriodIndex")
    n, L, k = int(rows.sum()), g0.shape[1], len(names)
    if L < k:
        raise SpecificationError(f"underidentified: {L} moments for {k} parameters")
    if n <= L:
        raise SpecificationError(f"{n} usable rows for {L} moments: too few")
    if m.cov == "hac" and not timed:
        raise SpecificationError("HAC weighting needs a time series")
    moment_names = columns or [f"m{i + 1}" for i in range(L)]

    def g(theta: np.ndarray) -> np.ndarray:
        out = np.asarray(contributions(theta), dtype=float)
        return (out[:, None] if out.ndim == 1 else out)[rows]

    def gbar(theta: np.ndarray) -> np.ndarray:
        return g(theta).mean(axis=0)

    def jacobian(theta: np.ndarray) -> np.ndarray:
        D = np.empty((L, k))
        for j in range(k):
            h = 1e-6 * max(1.0, abs(theta[j]))
            up, down = theta.copy(), theta.copy()
            up[j] += h
            down[j] -= h
            D[:, j] = (gbar(up) - gbar(down)) / (2 * h)
        return D

    lo = np.array([m.bounds.get(nm, (-np.inf, np.inf))[0] for nm in names], dtype=float)
    hi = np.array([m.bounds.get(nm, (-np.inf, np.inf))[1] for nm in names], dtype=float)
    bounded = bool(np.isfinite(lo).any() or np.isfinite(hi).any())

    def minimise(W: np.ndarray, start: np.ndarray) -> np.ndarray:
        C = np.linalg.cholesky(W).T

        def r(theta: np.ndarray) -> np.ndarray:
            value = C @ gbar(theta) * np.sqrt(n)
            return value if np.all(np.isfinite(value)) else np.full(L, 1e150)

        found = least_squares(r, start, jac=lambda th: C @ jacobian(th) * np.sqrt(n),
                              bounds=(lo, hi) if bounded else (-np.inf, np.inf), method="trf", xtol=1e-12, ftol=1e-12,
                              gtol=1e-12, max_nfev=5000)
        return found.x

    def S_at(theta: np.ndarray) -> np.ndarray:
        return _S_moments(m.cov, g(theta), m.hac_lags, m.center)

    W = np.eye(L)
    theta = minimise(W, theta0)
    steps = 1
    if m.weights == "optimal":
        for _ in range(1 if m.steps in ("two", "cue") else m.max_iter):
            W = np.linalg.inv(S_at(theta))
            new = minimise(W, theta)
            change = np.max(np.abs(new - theta)) / (1 + np.max(np.abs(theta)))
            theta, steps = new, steps + 1
            if change < m.tol:
                break
    if m.steps == "cue":
        def objective(th: np.ndarray) -> float:
            if bounded and (np.any(th < lo) or np.any(th > hi)):
                return 1e300
            try:
                gb = gbar(th)
                value = float(n * gb @ np.linalg.solve(S_at(th), gb))
            except (np.linalg.LinAlgError, ValueError):
                return 1e300
            return value if np.isfinite(value) else 1e300

        best = minimize(objective, theta, method="BFGS", options={"gtol": 1e-10, "maxiter": 2000})
        polish = minimize(objective, best.x, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
        theta = polish.x if polish.fun <= best.fun else best.x
        steps += 1
        W = np.linalg.inv(S_at(theta))
    S = S_at(theta)
    D = jacobian(theta)
    bread = np.linalg.inv(D.T @ W @ D)
    cov = bread @ D.T @ W @ S @ W @ D @ bread / n
    gb = gbar(theta)
    J = float(n * gb @ (W @ gb if m.weights == "optimal" else np.linalg.solve(S, gb)))
    df = L - k
    stats = {"nobs": float(n), "k": float(k), "moments": float(L), "J": J, "J_df": float(df),
             "J_pvalue": float(sps.chi2.sf(J, df)) if df > 0 else float("nan"), "steps": float(steps)}
    if m.cov == "hac":
        from .covariance import default_hac_lags

        stats["hac_lags"] = float(m.hac_lags or default_hac_lags(n))
    weighting = ("W = I, one step" if m.weights == "identity" else f"continuously updated (CUE), S {m.cov}" if m.steps == "cue" else
                 f"{'two-step' if m.steps == 'two' else f'iterated ({steps} steps)'} efficient weighting, S {m.cov}"
                 + (" (centred)" if m.center else ""))
    notes = (f"GMM on moment conditions: {L} moments for {k} parameters, {n} rows; {weighting}; numerical derivatives.",
             f"Hansen's J = {J:.4f}, chi-squared({df}), p = {stats['J_pvalue']:.4f}: a small p-value rejects the over-identifying "
             "restrictions." if df > 0 else "Exactly identified: no over-identifying restrictions to test.")
    index = frame.index[rows]
    provenance = Provenance(data_vintages={}, spec={"moments": getattr(m.moments, "__name__", "moments"), "parameters": names,
                                                     "moment names": moment_names},
                            method={"estimator": "GMM on moment functions", "weights": m.weights, "steps": m.steps, "cov": m.cov,
                                    "center": m.center, "steps taken": steps},
                            sample=(str(index[0]), str(index[-1])), nobs=n, dropped=int(len(frame) - n), package_version=__version__,
                            data_fingerprint=data_fingerprint(dataset) if dataset is not None else None)
    return FittedGMM(point(names, theta, cov), stats, pd.DataFrame(g(theta), index=index, columns=moment_names), m, notes, provenance)


@api("pymetrix:c_test")
def c_test(equation: Equation, data: Dataset, method: GMM, suspect: Sequence[str], span: Any = None) -> Any:
    """The C (difference-in-Sargan) test that the ``suspect`` instruments are valid, given the others: Hansen's J with all
    the instruments less J without the suspect ones, both minimised with the full set's S estimated from the two-step
    residuals (the block of it for the subset), so C is never negative,
    chi-squared with as many degrees of freedom as suspect instruments (Eichenbaum, Hansen and Singleton 1988; Hayashi
    2000)."""
    from dataclasses import replace

    from ..stats.base import TestResult

    suspect = [suspect] if isinstance(suspect, str) else list(suspect)
    missing = [s for s in suspect if s not in method.instruments]
    if missing:
        raise SpecificationError(f"{', '.join(missing)} are not among the instruments")
    full = _moments(method, equation, data, span, None, None, "drop")
    linear = equation.form == "linear"
    n = len(full.index)
    W = np.linalg.inv(full.Z.T @ full.Z / n)
    theta = _minimise(full, W, full.start, linear)
    for _ in range(2):  # S from the two-step residuals; both J statistics are minima with that same S
        S = _S(method.cov, full.Z, full.u(theta), method.hac_lags)
        Sinv = np.linalg.inv(S)
        theta = _minimise(full, Sinv, theta, linear)

    def J(mo: _Moments, Winv: np.ndarray, th: np.ndarray) -> float:
        g = mo.Z.T @ mo.u(th) / n
        return float(n * g @ Winv @ g)

    J_full = J(full, Sinv, theta)
    keep = [i for i, name in enumerate(full.z_names) if name not in [str(s) for s in suspect]]
    if len(keep) == len(full.z_names):  # instrument names may be shown differently: match by position after the constant
        offset = 1 if method.constant else 0
        drop = {offset + list(method.instruments).index(s) for s in suspect}
        keep = [i for i in range(len(full.z_names)) if i not in drop]
    sub = replace(full, Z=full.Z[:, keep], z_names=[full.z_names[i] for i in keep])
    if sub.Z.shape[1] < len(full.names):
        raise SpecificationError("without the suspect instruments the equation is underidentified")
    S_sub = np.linalg.inv(S[np.ix_(keep, keep)])
    theta_sub = _minimise(sub, S_sub, theta, linear)
    J_sub = J(sub, S_sub, theta_sub)
    C = J_full - J_sub
    df = len(full.z_names) - len(keep)
    return TestResult("C (difference-in-Sargan) test", f"the instruments {', '.join(suspect)} are valid", float(C), df,
                      float(sps.chi2.sf(C, df)), "chi2", details={"J (all instruments)": J_full, "J (without them)": J_sub})


@api("pymetrix:stock_wright")
def stock_wright(equation: Equation, data: Dataset, method: GMM, values: Mapping[str, float], span: Any = None) -> Any:
    """Stock and Wright's (2000) S test that every parameter has the given ``values``: the continuously updated GMM
    objective there, chi-squared with as many degrees of freedom as moments, whatever the instruments' strength."""
    from ..stats.base import TestResult

    mo = _moments(method, equation, data, span, None, None, "drop")
    missing = [k for k in mo.names if k not in values]
    if missing:
        raise SpecificationError(f"give a value for every parameter: {', '.join(missing)} missing")
    theta = np.array([float(values[k]) for k in mo.names])
    S = _cue_objective(method, mo, theta)
    L = mo.Z.shape[1]
    return TestResult("Stock-Wright S test", "the parameters take the given values", float(S), L, float(sps.chi2.sf(S, L)), "chi2",
                      details={k: float(values[k]) for k in mo.names})


@api("pymetrix:anderson_rubin")
def anderson_rubin(equation: Equation, data: Dataset, endog: str, instruments: Sequence[str], value: float, cov: str = "classic",
                   span: Any = None) -> Any:
    """The Anderson-Rubin (1949) test that the coefficient on ``endog`` is ``value``: regress y - value * endog on the
    other regressors and the excluded ``instruments``, and test that the instruments' coefficients are zero (an F test,
    exact with normal errors; ``cov="white"`` for a robust Wald test). Valid however weak the instruments."""
    from ..stats.base import TestResult
    from .design import linear_design
    from .methods import instrument_series

    if equation.form != "linear":
        raise SpecificationError("the Anderson-Rubin test here is for linear equations")
    series = instrument_series(list(instruments), data)
    d = linear_design(equation, data, span, None, None, "drop", instruments=series)
    names = list(d.names)
    label = endog if endog in names else next((nm for nm in names if nm.replace(" ", "") == endog.replace(" ", "")), None)
    if label is None:
        raise SpecificationError(f"{endog!r} is not a regressor of the equation: {', '.join(names)}")
    j = names.index(label)
    y = d.y - value * d.X[:, j]
    X1 = np.delete(d.X, j, axis=1)
    Z2 = d.Z
    R = np.column_stack([X1, Z2])
    b, *_ = np.linalg.lstsq(R, y, rcond=None)
    e = y - R @ b
    n, k1, k2 = len(y), X1.shape[1], Z2.shape[1]
    if cov == "classic":
        e0 = y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
        F = ((e0 @ e0 - e @ e) / k2) / (e @ e / (n - k1 - k2))
        stat, p, dist, df = float(F), float(sps.f.sf(F, k2, n - k1 - k2)), "F", (k2, n - k1 - k2)
    elif cov == "white":
        RRi = np.linalg.inv(R.T @ R)
        V = RRi @ (R * (e**2)[:, None]).T @ R @ RRi * n / (n - k1 - k2)
        g = b[k1:]
        stat = float(g @ np.linalg.solve(V[k1:, k1:], g))
        p, dist, df = float(sps.chi2.sf(stat, k2)), "chi2", k2
    else:
        raise SpecificationError("cov is 'classic' (the exact F test) or 'white' (robust)")
    return TestResult("Anderson-Rubin test", f"the coefficient on {endog} is {value:g}", stat, df, p, dist,
                      details={"value": float(value), "instruments": k2})


@api("pymetrix:anderson_rubin_set")
def anderson_rubin_set(equation: Equation, data: Dataset, endog: str, instruments: Sequence[str], grid: Sequence[float],
                       alpha: float = 0.05, cov: str = "classic", span: Any = None) -> pd.DataFrame:
    """The Anderson-Rubin confidence set by test inversion: the ``grid`` values not rejected at ``alpha`` (with each
    p-value). It may be bounded, unbounded or empty, as weak identification allows."""
    rows = [{"value": float(v), "p-value": anderson_rubin(equation, data, endog, instruments, float(v), cov, span).p_value} for v in grid]
    out = pd.DataFrame(rows)
    out["in set"] = out["p-value"] > alpha
    return out


@dataclass(frozen=True, eq=False)
class MinimumDistance:
    """Classical minimum distance: structural parameters fitted to reduced-form estimates."""

    names: tuple[str, ...]
    values: np.ndarray
    cov: np.ndarray = field(repr=False)
    distance: float = 0.0
    df: int = 0

    stage = "estimate"

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def summary(self) -> Any:
        from ..output import Table

        rows = [[float(v), float(s), float(v / s)] for v, s in zip(self.values, self.se)]
        notes = (f"Distance {self.distance:.4f}, chi-squared({self.df}), p = {sps.chi2.sf(self.distance, self.df):.4f}: a small p-value rejects "
                 "the restrictions." if self.df > 0 else "Exactly identified: no restrictions to test.",)
        return Table(["estimate", "std err", "z"], rows, index=list(self.names), index_name="parameter", title="Minimum distance", notes=notes)

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from ..views import _views_to_frame

            return _views_to_frame(self, view)
        return pd.DataFrame({"estimate": self.values, "se": self.se}, index=list(self.names))

    def __repr__(self) -> str:
        return f"<MinimumDistance {dict(zip(self.names, np.round(self.values, 4)))}>"


def minimum_distance(estimates: Sequence[float], cov: Any, restriction: Any, theta0: Sequence[float], names: Sequence[str] | None = None) -> MinimumDistance:
    """Classical minimum distance (Chamberlain 1982): theta minimising (pi - h(theta))' V^-1 (pi - h(theta)), with ``pi``
    reduced-form ``estimates``, V their covariance and h the ``restriction``; standard errors (H' V^-1 H)^-1, H the
    Jacobian of h; the minimised distance is chi-squared with len(pi) - len(theta) degrees of freedom."""
    from scipy.optimize import least_squares as _ls

    pi = np.asarray(estimates, float)
    V = np.atleast_2d(np.asarray(cov, float))
    theta0 = np.asarray(theta0, float)
    if V.shape != (len(pi), len(pi)):
        raise SpecificationError("the covariance must be square, one row for each estimate")
    if len(theta0) > len(pi):
        raise SpecificationError("more parameters than estimates: not identified")
    C = np.linalg.cholesky(np.linalg.inv(V)).T  # V^-1 = C'C
    found = _ls(lambda th: C @ (pi - np.asarray(restriction(th), float)), theta0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    theta = found.x

    def jacobian(theta):
        H = np.empty((len(pi), len(theta)))
        for j in range(len(theta)):
            h = 1e-6 * max(abs(theta[j]), 1e-3)
            up, dn = theta.copy(), theta.copy()
            up[j] += h
            dn[j] -= h
            H[:, j] = (np.asarray(restriction(up), float) - np.asarray(restriction(dn), float)) / (2 * h)
        return H

    def distance(theta):
        r = C @ (pi - np.asarray(restriction(theta), float))
        return float(r @ r)

    H = jacobian(theta)
    for _ in range(3):  # Gauss-Newton polish: the optimiser can stop short of the tolerance it was given
        step = np.linalg.lstsq(C @ H, C @ (pi - np.asarray(restriction(theta), float)), rcond=None)[0]
        if not distance(theta + step) < distance(theta):
            break
        theta = theta + step
        H = jacobian(theta)
    cov_theta = np.linalg.inv(H.T @ np.linalg.solve(V, H))
    d = pi - np.asarray(restriction(theta), float)
    names = tuple(names) if names is not None else tuple(f"theta{i}" for i in range(len(theta)))
    return MinimumDistance(names, theta, cov_theta, float(d @ np.linalg.solve(V, d)), len(pi) - len(theta))
