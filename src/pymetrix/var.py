"""Vector autoregressions: estimation, lag selection, identification, impulse responses,
variance decompositions and forecasts.

    spec = VAR(["dly", "inf", "r"], lags="bic", max_lags=8)
    spec.select_order(data)
    var = spec.fit(data)
    var.irf(20).plot()
    var.identify(Cholesky(["r", "inf", "dly"])).fevd(20)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from ._api import api, call_migration, renamed_arguments
from ._capture import (
    _sample_fields,
    data_fingerprint,
    records_warnings,
    with_effective_sample,
)
from ._planned import planned
from ._random import generator as make_rng
from ._serial import to_jsonable
from ._version import __version__
from .adapters import native
from .errors import SpecificationError
from .estimation.params import PointEstimate, point
from .forms import Widget
from .metadata import Provenance
from .output import Figure, Table
from .plugins import register

if TYPE_CHECKING:
    from .data import Dataset
    from .stats import TestResult

Criterion = Literal["aic", "bic", "hq", "fpe"]


# identification schemes


@dataclass(frozen=True)
class Cholesky:
    """Recursive identification; ``order`` lists the variables from most to least exogenous
    (default: the VAR's own order)."""

    code_positional: ClassVar[int] = 1  # generated code gives the order by position: Cholesky(['y', 'c'])
    order: Sequence[str] = ()


@dataclass(frozen=True)
class LongRun:
    """Blanchard-Quah: the long-run effects of the shocks are lower triangular."""


@dataclass(frozen=True)
class SignRestrictions:
    """Set identification by the signs of responses (Uhlig 2005; Rubio-Ramírez, Waggoner and Zha 2010).

    ``signs`` names each identified shock and the signs its responses must have, on impact and for ``horizon``
    periods after: ``{"supply": {"gy": "+", "pi": "-"}, "demand": {"gy": "+", "pi": "+"}}``. Random orthogonal
    rotations of the Cholesky factor (uniform, by QR) are kept when some column, or its negative, satisfies each
    shock's signs, until ``rotations`` are accepted (or ``max_tries`` drawn). Shocks not named are left
    unidentified. Responses are the pointwise median of the accepted draws; the impact matrix (for variance
    decompositions) is the accepted draw closest to that median (Fry and Pagan's median target).

    ``narrative_signs`` and ``narrative_major`` add Antolin-Diaz and Rubio-Ramirez's narrative restrictions: a named
    shock's sign at a date, or that it was the largest contributor to a variable's surprise at a date (see
    ``pymetrix.identification_extra``).

    ``zeros`` and ``long_run_zeros`` add zero restrictions (Arias, Rubio-Ramírez and Waggoner 2018): the variables a
    shock does not move on impact, or in the long run (the cumulated response, for variables in differences):
    ``zeros={"monetary": ["gy", "pi"]}``. A shock may have zeros and no signs. With zeros, rotations are drawn column
    by column in the null space of the zeros (ARW's algorithm), each column signed by its first sign restriction and
    kept if every sign holds (shocks keep their columns: no search over columns), with ARW's importance weights; the
    kept draws are resampled by weight (see ``pymetrix.zerosign``). Ordered by their number of zeros, the first shock
    may have n - 1, the next n - 2, and so on."""

    code_positional: ClassVar[int] = 1  # generated code gives the signs by position: SignRestrictions({...}, horizon=1)
    signs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)  # shock -> {variable: "+" or "-" (or +1 / -1)}
    horizon: int = 0
    rotations: int = 1000
    max_tries: int = 200_000
    seed: int = 0
    narrative_signs: Mapping[str, Mapping[str, str]] = field(default_factory=dict)  # shock -> {date: "+" or "-"}
    narrative_major: Mapping[str, Mapping[str, str]] = field(default_factory=dict)  # shock -> {date: the variable it moved most}
    zeros: Mapping[str, Sequence[str]] = field(default_factory=dict)  # shock -> the variables it does not move on impact
    long_run_zeros: Mapping[str, Sequence[str]] = field(default_factory=dict)  # shock -> those it does not move in the long run

    @property
    def narrative(self) -> bool:
        return bool(self.narrative_signs or self.narrative_major)

    @property
    def has_zeros(self) -> bool:
        return bool(self.zeros or self.long_run_zeros)

    @property
    def named(self) -> list[str]:
        """The identified shocks: those with signs, then any with zeros only."""
        return list(dict.fromkeys([*self.signs, *self.zeros, *self.long_run_zeros]))

    def __post_init__(self) -> None:
        for name in ("zeros", "long_run_zeros"):
            given = dict(getattr(self, name) or {})
            tidy = {str(k): tuple(str(v).strip() for v in ([vs] if isinstance(vs, str) else vs) if str(v).strip()) for k, vs in given.items()}
            object.__setattr__(self, name, {k: v for k, v in tidy.items() if v})
        if not self.signs and not self.has_zeros:
            raise SpecificationError("sign restrictions need at least one shock, with the signs of its responses")
        clean: dict[str, dict[str, int]] = {}
        for shock, responses in (self.signs or {}).items():
            if not responses and str(shock) not in self.zeros and str(shock) not in self.long_run_zeros:
                raise SpecificationError(f"the shock {shock!r} has no sign restrictions")
            clean[str(shock)] = {}
            for variable, sign in responses.items():
                value = {"+": 1, "-": -1}.get(sign.strip(), sign) if isinstance(sign, str) else sign
                if value not in (1, -1):
                    raise SpecificationError(f"the sign of {variable} for {shock} is {sign!r}: use '+' or '-'")
                clean[str(shock)][str(variable)] = int(value)
        object.__setattr__(self, "signs", clean)
        if self.horizon < 0 or self.rotations < 1 or self.max_tries < self.rotations:
            raise SpecificationError("sign restrictions need horizon >= 0, rotations >= 1 and max_tries >= rotations")

    def shock_names(self, variables: Sequence[str]) -> list[str]:
        named = self.named
        return named + [f"unidentified {i}" for i in range(1, len(variables) - len(named) + 1)]

    def check(self, variables: Sequence[str]) -> None:
        named = self.named
        if len(named) > len(variables):
            raise SpecificationError(f"{len(named)} shocks named, but the VAR has {len(variables)} variables")
        restricted = {v for r in self.signs.values() for v in r} | {v for r in [*self.zeros.values(), *self.long_run_zeros.values()] for v in r}
        unknown = sorted(restricted - set(variables))
        if unknown:
            raise SpecificationError(f"sign restrictions on {', '.join(unknown)}, which the VAR does not have ({', '.join(variables)})")
        clash = sorted(set(named) & set(variables))
        if clash:
            raise SpecificationError(f"name the shocks apart from the variables: {', '.join(clash)}")
        for shock, vs in self.zeros.items():
            both = sorted(set(vs) & set(self.signs.get(shock, {})))
            if both:
                raise SpecificationError(f"{shock} has both a sign and a zero on impact for {', '.join(both)}")
        if self.has_zeros:
            from .zerosign import generation_order

            generation_order([len(self.zeros.get(k, ())) + len(self.long_run_zeros.get(k, ())) for k in self.shock_names(variables)])


from .identification_extra import (
    MaxShare,  # noqa: E402 - the maximum forecast-error-variance share
)
from .structural import (
    AB,  # noqa: E402 - restrictions on A and B, estimated by maximum likelihood
)

Identification = Cholesky | LongRun | SignRestrictions | AB | MaxShare


# specification


def _exog_names(spec: Any) -> list[str]:
    """The series an exogenous expression of a VAR uses."""
    from .expr import parse_expression

    return sorted(set().union(*(parse_expression(e).variables() for e in getattr(spec, "exog", ()) or ()))) if getattr(spec, "exog", ()) else []


@dataclass(frozen=True, eq=False)
class VAR:
    """Vector autoregression, estimated equation by equation by least squares.

    Each variable is regressed on lags of all of them, so the VAR summarises the data's joint dynamics with few
    restrictions; its coefficients are hard to read one by one, and it is used through impulse responses,
    variance decompositions, causality tests and forecasts. Responses to shocks need an identification: the
    default Cholesky scheme lets a variable respond within the period only to the shocks of variables listed
    before it, so the order of the variables is an assumption and the responses can change with it. The lag
    length is a trade-off: too few leave autocorrelation in the residuals, too many spend k squared parameters
    per lag and make the estimates and forecasts noisy (BIC tends to choose fewer lags than AIC). Check that the
    model is stable (every root inside the unit circle) before reading its responses. Integrated variables can
    stay in levels, but some tests then lose their usual distributions; difference them, or use a VECM if they
    cointegrate.
    """

    code_positional: ClassVar[int] = 1  # generated code gives the first field by position: VAR(...)

    label: ClassVar[str] = "VAR"
    topics: ClassVar[tuple[str, ...]] = ('var',)

    variables: Annotated[Sequence[str], Widget("text", help="the endogenous series, in identification order")]
    lags: Annotated[int | Criterion, Widget("text", help="number of lags, or a criterion to choose them: aic, bic, hq, fpe")]
    max_lags: Annotated[int, Widget("number", min=1, max=48, help="largest lag searched when lags is a criterion")] = 8
    exog: Annotated[Sequence[str], Widget("text", help="exogenous expressions, e.g. dlog(oil)")] = ()
    trend: Literal["n", "c", "ct"] = "c"
    by: Annotated[str | None, Widget("hidden", help="the entity axis, for a panel VAR")] = None
    factors: Annotated[Any, Widget("hidden", help="not used: fit a factor-augmented VAR with favar(...)")] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "exog", tuple(self.exog))
        if len(self.variables) < 1 or len(set(self.variables)) != len(self.variables):
            raise SpecificationError("a VAR needs one or more distinct variables")
        if isinstance(self.lags, str) and self.lags not in ("aic", "bic", "hq", "fpe"):
            raise SpecificationError("lags must be a number or one of 'aic', 'bic', 'hq', 'fpe'")

    @api("pymetrix:var.select_order")
    def select_order(self, data: Dataset, max_lags: int | None = None, span: tuple | None = None) -> LagSelection:
        """Information criteria for 0..max_lags lags, all on the sample that max_lags allows."""
        if self.factors is not None:
            raise SpecificationError(_FAVAR_HINT)
        max_lags = max_lags if max_lags is not None else self.max_lags
        if self.by:  # Andrews and Lu's criteria from Hansen's J (see pymetrix.panelvar.select_order)
            from .panelvar import select_order as panel_select_order

            return panel_select_order(self, data, max_lags, span)
        arrays = _arrays(self, data, span, max_lags)
        rows = []
        for p in range(max_lags + 1):
            est = _estimate(arrays, p)
            rows.append([est.criteria[c] for c in ("aic", "bic", "hq", "fpe")])
        frame = pd.DataFrame(rows, index=pd.Index(range(max_lags + 1), name="lags"), columns=["AIC", "BIC", "HQ", "FPE"])
        best = {c.lower(): int(frame[c].idxmin()) for c in frame.columns}
        notes = [f"{len(arrays.index)} observations ({arrays.index[0]} to {arrays.index[-1]}) for every lag length"]
        notes += [f"Best by {c.upper()}: {p} lag{'s' * (p != 1)}" for c, p in best.items()]
        return LagSelection(Table.from_frame(frame, title=f"VAR lag selection: {', '.join(self.variables)}", notes=notes), best)

    @api("pymetrix:var.fit")
    @records_warnings
    def fit(self, data: Dataset, span: tuple | None = None, method: Any = None, transform: Literal["fod", "within"] = "fod",
            instrument_lags: int | None = None) -> FittedVAR:
        """Estimate the VAR on ``data`` over ``span``: by least squares, equation by equation, or with ``method=Bayes(...)``
        and a prior (e.g. Minnesota). A lag criterion (``lags="bic"``) chooses the lags first, up to ``max_lags``.
        With ``by`` (an entity axis), a panel VAR: GMM on forward orthogonal deviations (``transform``, ``instrument_lags``;
        see ``pymetrix.panelvar``)."""
        if self.by:
            if method is not None:
                planned("Other estimators for panel VARs", 3)
            from .panelvar import fit_panel
            from .panelvar import select_order as panel_select_order

            p = self.lags
            if isinstance(p, str):  # bic, aic, hq: Andrews and Lu's MBIC, MAIC, MQIC (or Lutkepohl's for transform="within")
                if p == "fpe":
                    raise SpecificationError("panel VARs choose lags by 'bic', 'aic' or 'hq' (Andrews and Lu's MBIC, MAIC, MQIC)")
                p = panel_select_order(self, data, self.max_lags, span, transform).best[p]
            return fit_panel(self, data, span, int(p), transform, instrument_lags)
        if self.factors is not None:
            raise SpecificationError(_FAVAR_HINT)
        from .estimation import OLS, Bayes

        if method is not None and not isinstance(method, (OLS, Bayes, Restricted)):
            planned(f"VAR estimation by {type(method).__name__}", 5)
        p = self.select_order(data, self.max_lags, span).best[self.lags] if isinstance(self.lags, str) else self.lags
        if p < 1:
            raise SpecificationError("the selected lag length is 0; a VAR needs at least one lag (set lags= explicitly)")
        arrays = _arrays(self, data, span, p)
        if isinstance(method, Bayes):
            from .bvar import fit_bayes

            return fit_bayes(self, data, span, method, p, arrays)
        if isinstance(method, Restricted):
            free = method.mask(self.variables, arrays.d_names + [f"{v}[-{l}]" for l in range(1, p + 1) for v in self.variables],
                               len(arrays.d_names))
            est = _estimate_restricted(arrays, p, free, method.iterate, method.max_iter)
        else:
            est = _estimate(arrays, p)
        prov = Provenance(
            data_vintages={v: data.variables[v].vintage for v in self.variables if v in data.variables},
            spec={**to_jsonable(self), "lags": p},
            method={"type": "restricted (feasible GLS)" if isinstance(method, Restricted) else "OLS"},
            sample=(str(arrays.index[0]), str(arrays.index[-1])),
            nobs=len(arrays.index),
            dropped=0,
            package_version=__version__,
            data_fingerprint=data_fingerprint(data, [*self.variables, *getattr(self, "exog", ())], arrays.index, p),
            resolved={"lags": p, "chosen by": self.lags} if isinstance(self.lags, str) else None,
            backend=native.backend_record("pymetrix.VAR"),
            **_sample_fields(data, span),
        )
        prov = with_effective_sample(prov, data, arrays.index, span=span, variables=[*self.variables, *_exog_names(self)])
        return FittedVAR(self, p, est, arrays, Cholesky(), prov, method if isinstance(method, Restricted) else None)


@dataclass(frozen=True, eq=False)
class LagSelection:
    table: Table
    best: dict[str, int]
    provenance: Any = None  # set where a library chose the lags (VECM.select_order)

    def __str__(self) -> str:
        return str(self.table)

    def _repr_html_(self) -> str:
        return self.table._repr_html_()


# estimation core (NumPy only, so bootstraps are fast)


@dataclass(frozen=True, eq=False)
class _Arrays:
    Y: np.ndarray  # (T, K) current values
    L: np.ndarray  # (T, K * max_lags) lags, lag-major: y[t-1] (K columns), y[t-2], ...
    D: np.ndarray  # (T, d) deterministic terms and exogenous variables
    d_names: list[str]
    index: pd.PeriodIndex
    variables: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class _Estimate:
    B: np.ndarray  # (kz, K): rows are regressors (deterministic first, then lags), columns equations
    U: np.ndarray
    sigma: np.ndarray  # residual covariance, degrees-of-freedom adjusted
    sigma_ml: np.ndarray
    ZZinv: np.ndarray
    criteria: dict[str, float]
    loglike: float
    cov: np.ndarray | None = None  # under restrictions: the covariance of vec(B), equation by equation (zero where restricted)
    free: np.ndarray | None = None  # under restrictions: which coefficients are estimated (kz, K)


@dataclass(frozen=True)
class Restricted:
    """Zero restrictions on a VAR's coefficients, estimated by feasible GLS (seemingly unrelated regressions,
    Lutkepohl 2005, section 5.2): each equation's own regressors; the residual covariance from least squares on them,
    then GLS (``iterate=True``: until the estimates settle, the maximum-likelihood estimate).

    ``zeros`` maps an equation to the regressors left out of it: a variable (all its lags), one lag (``"r[-2]"``), or
    a deterministic or exogenous term (``"const"``). ``block_exogenous`` variables' equations leave out the other
    variables' lags (a small open economy's world block, say)."""

    zeros: Mapping[str, Sequence[str]] = field(default_factory=dict)
    block_exogenous: Sequence[str] = ()
    iterate: bool = False
    max_iter: int = 100

    def mask(self, variables: Sequence[str], regressors: Sequence[str], d: int) -> np.ndarray:
        """Which coefficients (regressor, equation) are estimated."""
        variables, regressors = list(variables), list(regressors)
        free = np.ones((len(regressors), len(variables)), dtype=bool)
        unknown = [v for v in [*self.zeros, *self.block_exogenous] if v not in variables]
        if unknown:
            raise SpecificationError(f"{', '.join(unknown)} are not equations of the VAR: {', '.join(variables)}")
        for eq, names in self.zeros.items():
            j = variables.index(eq)
            for name in ([names] if isinstance(names, str) else names):
                rows = [i for i, r in enumerate(regressors) if r == name or (i >= d and r.split("[")[0] == name)]
                if not rows:
                    raise SpecificationError(f"{name!r} is not a regressor of the VAR (a variable, a lag such as r[-2], or const)")
                free[rows, j] = False
        for eq in self.block_exogenous:
            j = variables.index(eq)
            free[[i for i, r in enumerate(regressors) if i >= d and r.split("[")[0] != eq], j] = False
        if not free.any(axis=0).all():
            raise SpecificationError("every equation needs at least one regressor")
        return free


def _estimate_restricted(a: _Arrays, p: int, free: np.ndarray, iterate: bool = False, max_iter: int = 100,
                         Y: np.ndarray | None = None, L: np.ndarray | None = None) -> _Estimate:
    """Feasible GLS with each equation's own regressors (``free``): least squares first, then GLS with the residual
    covariance U'U / T (repeated with ``iterate``). Sigma for responses is U'U over sqrt((T - k_i)(T - k_j)), which is
    the unrestricted VAR's when nothing is restricted."""
    Y = a.Y if Y is None else Y
    L = a.L if L is None else L
    T, K = Y.shape
    Z = np.column_stack([a.D, L[:, : K * p]])
    kz = Z.shape[1]
    cols = [np.flatnonzero(free[:, j]) for j in range(K)]
    B = np.zeros((kz, K))
    for j, c in enumerate(cols):  # least squares, equation by equation
        B[c, j] = np.linalg.lstsq(Z[:, c], Y[:, j], rcond=None)[0]
    sizes = [len(c) for c in cols]
    offsets = np.r_[0, np.cumsum(sizes)]
    for _ in range(max_iter if iterate else 1):
        U = Y - Z @ B
        Sinv = np.linalg.inv(U.T @ U / T)
        M = np.zeros((offsets[-1], offsets[-1]))
        rhs = np.zeros(offsets[-1])
        for i, ci in enumerate(cols):
            for j, cj in enumerate(cols):
                M[offsets[i]:offsets[i + 1], offsets[j]:offsets[j + 1]] = Sinv[i, j] * Z[:, ci].T @ Z[:, cj]
            rhs[offsets[i]:offsets[i + 1]] = Z[:, ci].T @ (Y @ Sinv[i])
        beta = np.linalg.solve(M, rhs)
        new = np.zeros_like(B)
        for j, c in enumerate(cols):
            new[c, j] = beta[offsets[j]:offsets[j + 1]]
        done = np.max(np.abs(new - B)) < 1e-12
        B = new
        if done:
            break
    U = Y - Z @ B
    inv_M = np.linalg.inv(M)
    cov = np.zeros((K * kz, K * kz))
    positions = np.concatenate([j * kz + c for j, c in enumerate(cols)])  # vec(B), equation by equation
    cov[np.ix_(positions, positions)] = inv_M
    k = np.array(sizes, dtype=float)
    sigma = U.T @ U / np.sqrt(np.outer(T - k, T - k))
    sigma_ml = U.T @ U / T
    logdet = np.linalg.slogdet(sigma_ml)[1]
    nfree = float(k.sum())
    criteria = {"aic": logdet + 2 * nfree / T, "bic": logdet + np.log(T) * nfree / T, "hq": logdet + 2 * np.log(np.log(T)) * nfree / T}
    loglike = -T / 2 * (K * np.log(2 * np.pi) + logdet + K)
    ZZinv = np.linalg.inv(Z.T @ Z)
    return _Estimate(B, U, sigma, sigma_ml, ZZinv, criteria, float(loglike), cov, free)


def _reestimate(fit: Any, a: _Arrays, p: int, Y: np.ndarray, L: np.ndarray) -> _Estimate:
    """A bootstrap sample's estimate, under the fit's restrictions if it has any."""
    e = fit.estimate
    if getattr(e, "free", None) is not None:
        r = fit.restriction
        return _estimate_restricted(a, p, e.free, r.iterate, r.max_iter, Y=Y, L=L)
    return _estimate(a, p, Y=Y, L=L)


def _arrays(spec: VAR, data: Dataset, span: tuple | None, max_lags: int) -> _Arrays:
    from .estimation.design import instrument_series

    if data.kind != "time_series":
        raise SpecificationError("a VAR needs a single time series; for a panel VAR, give the entity axis: VAR(..., by='firm')")
    missing = [v for v in spec.variables if v not in data]
    if missing:
        raise SpecificationError(f"variables {missing} are not in the data")
    cols: dict[str, pd.Series] = {}
    for v in spec.variables:
        cols[v] = data[v].astype(float)
    for lag in range(1, max_lags + 1):
        for v in spec.variables:
            cols[f"{v}[-{lag}]"] = data.lagged(v, -lag).astype(float)
    exog = instrument_series(spec.exog, data)
    cols.update(exog)
    frame = pd.DataFrame(cols)[data.sample_mask(span)]
    frame = frame[np.isfinite(frame.to_numpy(dtype=float)).all(axis=1)]
    if frame.empty:
        raise SpecificationError("no complete observations for the VAR")
    index = frame.index
    if len(pd.period_range(index[0], index[-1], freq=index.freq)) != len(index):
        raise SpecificationError("the VAR sample has gaps; VARs need consecutive periods")
    T = len(frame)
    det, names = [], []
    if spec.trend in ("c", "ct"):
        det.append(np.ones(T))
        names.append("const")
    if spec.trend == "ct":
        det.append(np.arange(1.0, T + 1))
        names.append("trend")
    for name in exog:
        det.append(frame[name].to_numpy(float))
        names.append(name)
    K = len(spec.variables)
    return _Arrays(
        Y=frame[list(spec.variables)].to_numpy(float),
        L=frame[[f"{v}[-{lag}]" for lag in range(1, max_lags + 1) for v in spec.variables]].to_numpy(float).reshape(T, K * max_lags),
        D=np.column_stack(det) if det else np.zeros((T, 0)),
        d_names=names,
        index=index,
        variables=tuple(spec.variables),
    )


def _estimate(a: _Arrays, p: int, Y: np.ndarray | None = None, L: np.ndarray | None = None) -> _Estimate:
    Y = a.Y if Y is None else Y
    L = a.L if L is None else L
    T, K = Y.shape
    Z = np.column_stack([a.D, L[:, : K * p]])
    kz = Z.shape[1]
    if kz:
        if np.linalg.matrix_rank(Z) < kz:
            raise SpecificationError("the VAR regressors are perfectly collinear")
        B = np.linalg.lstsq(Z, Y, rcond=None)[0]
        ZZinv = np.linalg.inv(Z.T @ Z)
    else:
        B, ZZinv = np.zeros((0, K)), np.zeros((0, 0))
    U = Y - Z @ B
    sigma_ml = U.T @ U / T
    sigma = U.T @ U / (T - kz)
    logdet = np.linalg.slogdet(sigma_ml)[1]
    free = K * kz
    criteria = {
        "aic": logdet + 2 * free / T,
        "bic": logdet + np.log(T) * free / T,
        "hq": logdet + 2 * np.log(np.log(T)) * free / T,
        "fpe": ((T + kz) / (T - kz)) ** K * np.exp(logdet),
    }
    loglike = -T / 2 * (K * np.log(2 * np.pi) + logdet + K)
    from .estimation import criteria as _criteria

    # pymetrix's totals beside Lütkepohl's (lag selection): k = K * m, the coefficients, not the covariance
    criteria.update({"aic_total": _criteria.aic(loglike, free), "bic_total": _criteria.bic(loglike, free, T),
                     "hq_total": _criteria.hq(loglike, free, T)})
    return _Estimate(B, U, sigma, sigma_ml, ZZinv, criteria, float(loglike))


def _lag_matrices(B: np.ndarray, d: int, K: int, p: int) -> np.ndarray:
    """A[l] (K x K) with A[l][i, j] = effect of variable j at lag l + 1 in equation i."""
    return np.stack([B[d + l * K : d + (l + 1) * K, :].T for l in range(p)])


def _ma(A: np.ndarray, horizons: int) -> np.ndarray:
    p, K, _ = A.shape
    phi = np.zeros((horizons + 1, K, K))
    phi[0] = np.eye(K)
    for i in range(1, horizons + 1):
        for j in range(1, min(i, p) + 1):
            phi[i] += phi[i - j] @ A[j - 1]
    return phi


def _point_path(A: np.ndarray, Bd: np.ndarray, D: Sequence[Sequence[float]], history: Sequence[np.ndarray], periods: int) -> np.ndarray:
    """The forecast path (periods x variables) with no shocks, from the last ``lags`` observations in ``history``."""
    p, K, _ = A.shape
    hist = [np.asarray(h, float) for h in history]
    out = np.empty((periods, K))
    for i in range(periods):
        y = np.asarray(D[i]) @ Bd if len(Bd) else np.zeros(K)
        for lag in range(1, p + 1):
            y = y + A[lag - 1] @ hist[-lag]
        out[i] = y
        hist.append(y)
    return out


def _condition_list(conditions: Mapping[str, Any], variables: Sequence[str], periods: pd.PeriodIndex) -> list[tuple[int, int, float]]:
    """(period index, variable index, value) for each condition: a number holds a variable over every forecast period,
    a mapping or series of period -> value over those periods."""
    out = []
    for name, path in conditions.items():
        if name not in variables:
            raise SpecificationError(f"{name} is not a variable of this VAR ({', '.join(variables)})")
        k = list(variables).index(name)
        items = [(q, path) for q in periods] if np.isscalar(path) else list(dict(path).items())
        for q, value in items:
            period = pd.Period(str(q), freq=periods.freq)
            if period not in periods:
                raise SpecificationError(f"the condition on {name} in {period} is outside the forecast ({periods[0]} to {periods[-1]})")
            if not np.isfinite(float(value)):
                raise SpecificationError(f"the condition on {name} in {period} is not a number")
            out.append((periods.get_loc(period), k, float(value)))
    if not out:
        raise SpecificationError("give at least one condition, e.g. {'r': 2.0} or {'r': {'2025Q1': 2.0}}")
    if len({(i, k) for i, k, _ in out}) < len(out):
        raise SpecificationError("a variable has two conditions in the same period")
    return out


def _conditional_moments(A: np.ndarray, sigma: np.ndarray, point: np.ndarray, conds: Sequence[tuple[int, int, float]]
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Waggoner and Zha (1999): the path given exact conditions. With future y = point + M e (e ~ N(0, I), M from the
    moving-average weights and the Cholesky factor of sigma) and conditions R y = r, the shocks that meet them on
    average are e* = D'(DD')^-1 (r - R point), D = R M. Returns the conditional mean path, the conditional covariance
    of the stacked path (period-major), M, and the projection (I - D'(DD')^-1 D) of the shocks left free."""
    H, K = point.shape
    phi = _ma(A, H - 1)
    P = np.linalg.cholesky(sigma)
    M = np.zeros((H * K, H * K))
    for i in range(H):
        for j in range(i + 1):
            M[i * K:(i + 1) * K, j * K:(j + 1) * K] = phi[i - j] @ P
    R = np.zeros((len(conds), H * K))
    q = np.empty(len(conds))
    for row, (i, k, value) in enumerate(conds):
        R[row, i * K + k] = 1.0
        q[row] = value - point[i, k]
    Dm = R @ M
    DD = Dm @ Dm.T
    if np.linalg.matrix_rank(DD) < len(conds):
        raise SpecificationError("the conditions cannot all be met: some fix a variable that earlier conditions already determine")
    gain = Dm.T @ np.linalg.inv(DD)
    e_star = gain @ q
    free = np.eye(H * K) - gain @ Dm
    mean = point + (M @ e_star).reshape(H, K)
    cov = M @ free @ M.T
    return mean, cov, M, free


def _impact(scheme: Any, sigma: np.ndarray, A: np.ndarray, variables: Sequence[str], resid: Any = None) -> np.ndarray:
    """Impact matrix P: column k is the response on impact to structural shock k (one standard deviation)."""
    K = len(variables)
    if isinstance(scheme, Cholesky):
        order = list(scheme.order) or list(variables)
        if sorted(order) != sorted(variables):
            raise SpecificationError(f"the Cholesky order must list each of {list(variables)} once")
        perm = [list(variables).index(v) for v in order]
        P = np.zeros((K, K))
        P[np.ix_(perm, perm)] = np.linalg.cholesky(sigma[np.ix_(perm, perm)])
        return P
    if isinstance(scheme, LongRun):
        A1 = np.eye(K) - A.sum(axis=0)
        Xi = np.linalg.inv(A1)
        return A1 @ np.linalg.cholesky(Xi @ sigma @ Xi.T)
    if isinstance(scheme, MaxShare):
        from .identification_extra import max_share_impact

        return max_share_impact(scheme, sigma, A, variables)
    if isinstance(scheme, AB):
        from .structural import estimate_ab

        return estimate_ab(scheme, sigma, A, 100, variables, standard_errors=False).impact
    if isinstance(scheme, SignRestrictions):  # a set: its median-target member
        draws = _accepted(scheme, sigma, A, variables, scheme.rotations, make_rng(scheme.seed), resid)
        phi = _ma(A, max(scheme.horizon, 20))
        return _median_target(draws, np.stack([phi @ B for B in draws]))
    planned(f"{type(scheme).__name__} identification", 3)


def sign_rotations(scheme: SignRestrictions, sigma: np.ndarray, A: np.ndarray, variables: Sequence[str], want: int,
                   rng: np.random.Generator) -> tuple[list[np.ndarray], int]:
    """Impact matrices that satisfy the sign restrictions (columns: the named shocks in order, then the
    unidentified ones), up to ``want`` of them, and how many rotations were drawn. Each B has B B' = sigma. With zero
    restrictions too, ARW's draws resampled by their weights (``_zero_sign_set``)."""
    from .progress import report

    if scheme.has_zeros:
        return _zero_sign_set(scheme, sigma, A, variables, want, rng)

    K = len(variables)
    index = {v: i for i, v in enumerate(variables)}
    checks = [[(index[v], s) for v, s in responses.items()] for responses in scheme.signs.values()]
    P0 = np.linalg.cholesky(sigma)
    phi = _ma(A, scheme.horizon)  # (horizon + 1, K, K): responses to unit reduced-form shocks
    accepted: list[np.ndarray] = []
    tries = 0
    while len(accepted) < want and tries < scheme.max_tries:
        tries += 1
        if tries % 500 == 0:
            report(len(accepted) / want, f"sign restrictions: {len(accepted)} of {want} accepted, {tries:,} rotations drawn")
        Q, R = np.linalg.qr(rng.standard_normal((K, K)))
        B = P0 @ (Q * np.sign(np.diag(R)))  # uniform (Haar) rotations
        responses = phi @ B
        used: list[int] = []
        columns = []
        for restriction in checks:  # each named shock takes the first unused column (or its negative) with its signs
            found = None
            for j in (j for j in range(K) if j not in used):
                for flip in (1.0, -1.0):
                    if all(np.all(flip * sign * responses[:, i, j] > 0) for i, sign in restriction):
                        found = (j, flip)
                        break
                if found:
                    break
            if found is None:
                break
            used.append(found[0])
            columns.append(found[1] * B[:, found[0]])
        else:
            accepted.append(np.column_stack(columns + [B[:, j] for j in range(K) if j not in used]))
    return accepted, tries


def _zero_sign_set(scheme: SignRestrictions, sigma: np.ndarray, A: np.ndarray, variables: Sequence[str], want: int,
                   rng: np.random.Generator) -> tuple[list[np.ndarray], int]:
    """Under zero and sign restrictions: ``want`` impact matrices at (A, sigma), drawn by ARW's algorithm until ``want``
    are kept and resampled by their importance weights; and how many rotations were drawn."""
    from .zerosign import resample, zero_sign_draws

    kept, logw, tries = zero_sign_draws(scheme, sigma, A, variables, scheme.max_tries, rng, want)
    return (resample(kept, logw, want, rng)[0] if kept else []), tries


def _rotations_for_draws(scheme: SignRestrictions, pairs: Sequence[tuple[np.ndarray, np.ndarray]], variables: Sequence[str],
                         rng: np.random.Generator) -> list[tuple[int, np.ndarray]]:
    """An impact matrix for draws of (A, sigma) (bootstrap or posterior), as (the draw's index, B). Signs alone: an
    accepted rotation for each draw (none for a draw where none is found). With zeros, ARW's algorithm: a rotation for
    each draw in turn, kept with its importance weight when its signs hold, in passes over the draws until as many
    are kept as there are draws (or ``max_tries`` rotations drawn), then resampled by weight to that number."""
    pairs = list(pairs)
    if not scheme.has_zeros:
        out = []
        for i, (A, S) in enumerate(pairs):
            accepted, _ = sign_rotations(scheme, S, A, variables, 1, rng)
            if accepted:
                out.append((i, accepted[0]))
        return out
    from .progress import report
    from .zerosign import resample, zero_sign_draws

    kept: list[tuple[int, np.ndarray]] = []
    logw: list[float] = []
    tries = 0
    while len(kept) < len(pairs) and tries < scheme.max_tries:
        report(len(kept) / len(pairs), f"zero and sign restrictions: {len(kept)} of {len(pairs)} kept, {tries:,} rotations drawn")
        for i, (A, S) in enumerate(pairs):
            got, lw, _ = zero_sign_draws(scheme, S, A, variables, 1, rng)
            kept += [(i, B) for B in got]
            logw += lw
        tries += len(pairs)
    return resample(kept, logw, len(pairs), rng)[0] if kept else []


def _accepted(scheme: SignRestrictions, sigma: np.ndarray, A: np.ndarray, variables: Sequence[str], want: int,
              rng: np.random.Generator, resid: Any = None) -> list[np.ndarray]:
    if scheme.has_zeros:
        draws, tries = _zero_sign_set(scheme, sigma, A, variables, want, rng)
    else:
        draws, tries = sign_rotations(scheme, sigma, A, variables, want, rng)
    if not draws:
        raise SpecificationError(f"no rotation satisfies the sign restrictions in {tries:,} tries: check that they can hold together")
    if scheme.narrative:
        from .identification_extra import apply_narrative, narrative_check

        if resid is None:
            raise SpecificationError("narrative restrictions need the VAR's residuals: bands come from the identified set (draws=0)")
        draws = apply_narrative(draws, narrative_check(scheme, resid, variables), rng)
    return draws


def _median_target(draws: Sequence[np.ndarray], responses: np.ndarray) -> np.ndarray:
    """The accepted impact matrix whose responses are closest to the pointwise median, each response standardised by
    its spread across the draws (Fry and Pagan, 2011)."""
    median = np.median(responses, axis=0)
    spread = responses.std(axis=0)
    spread[spread == 0] = 1.0
    distance = (((responses - median) / spread) ** 2).sum(axis=(1, 2, 3))
    return draws[int(np.argmin(distance))]


# results


@dataclass(frozen=True, eq=False)
class FittedVAR:
    # a VAR's aic, bic, hq and fpe are Lütkepohl's (as statsmodels, EViews and Stata report them for lag selection);
    # aic_total, bic_total and hq_total are pymetrix's totals, comparable with single equations
    stat_definitions: ClassVar[dict[str, str]] = {"aic": "aic.lutkepohl", "bic": "bic.lutkepohl", "hq": "hq.lutkepohl",
                                                  "fpe": "fpe.lutkepohl", "loglike": "loglike.gaussian"}

    spec: VAR
    lags: int
    estimate: _Estimate = field(repr=False)
    arrays: _Arrays = field(repr=False)
    identification: Any = field(default_factory=Cholesky)
    provenance: Provenance | None = None
    restriction: Any = None  # Restricted(...), when coefficients are restricted to zero

    # basic results

    @property
    def variables(self) -> tuple[str, ...]:
        return self.arrays.variables

    @property
    def regressors(self) -> list[str]:
        return self.arrays.d_names + [f"{v}[-{l}]" for l in range(1, self.lags + 1) for v in self.variables]

    @property
    def params(self) -> PointEstimate:
        names = [f"{eq}: {reg}" for eq in self.variables for reg in self.regressors]
        e = self.estimate
        cov = e.cov if getattr(e, "cov", None) is not None else np.kron(e.sigma, e.ZZinv)
        return point(names, e.B.T.reshape(-1), cov)

    @property
    def coefs(self) -> np.ndarray:
        """Lag matrices, shape (lags, K, K)."""
        return _lag_matrices(self.estimate.B, len(self.arrays.d_names), len(self.variables), self.lags)

    @property
    def sigma(self) -> pd.DataFrame:
        return pd.DataFrame(self.estimate.sigma, index=list(self.variables), columns=list(self.variables))

    @property
    def resid(self) -> Dataset:
        from .data import Dataset

        return Dataset.from_frame(pd.DataFrame(self.estimate.U, index=self.arrays.index, columns=list(self.variables)))

    @property
    def fitted(self) -> Dataset:
        """The fitted values of each equation (the data less the residuals)."""
        from .data import Dataset

        return Dataset.from_frame(pd.DataFrame(self.arrays.Y - self.estimate.U, index=self.arrays.index, columns=list(self.variables)))

    @property
    def stats(self) -> dict[str, float]:
        e = self.estimate
        return {"nobs": float(len(self.arrays.index)), "lags": float(self.lags), "loglike": e.loglike, **e.criteria}

    @property
    def eigenvalues(self) -> np.ndarray:
        """The companion matrix's eigenvalues (complex), largest modulus first."""
        return _companion_eigenvalues(self.coefs)

    @property
    def roots(self) -> np.ndarray:
        """Moduli of the companion-matrix eigenvalues, largest first."""
        return np.abs(self.eigenvalues)

    @api()
    def residual_correlogram(self, lags: int = 12) -> ResidualCorrelogram:
        """Auto- and cross-correlations of the residuals up to ``lags``, with ±1.96/sqrt(T) bands; ``table()`` and
        ``plot()``. The portmanteau test (``test("portmanteau")``) sums them up."""
        return residual_correlogram(self.resid.frame[list(self.variables)].to_numpy(float), self.variables, lags,
                                    f"Residual correlogram: VAR({self.lags}) in {', '.join(self.variables)}")

    def roots_plot(self) -> Figure:
        """The companion matrix's eigenvalues in the complex plane, with the unit circle: all inside for a stable VAR."""
        from .figures import roots_figure

        state = "stable: every root inside the unit circle" if self.is_stable else "NOT stable: a root on or outside the unit circle"
        return roots_figure(self.eigenvalues, f"VAR({self.lags}) roots", f"Largest modulus {self.roots[0]:.4f}; {state}.")

    @property
    def is_stable(self) -> bool:
        return bool(self.roots[0] < 1)

    def summary(self) -> Table:
        e = self.estimate
        if getattr(e, "cov", None) is not None:  # restricted: zero standard errors where a coefficient is set to zero
            se = np.sqrt(np.diag(e.cov)).reshape(len(self.variables), -1).T
        else:
            se = np.sqrt(np.outer(np.diag(e.ZZinv), np.diag(e.sigma)))
        rows = []
        for i, reg in enumerate(self.regressors):
            row = []
            for j in range(len(self.variables)):
                row += [e.B[i, j], se[i, j]]
            rows.append(row)
        columns = [c for v in self.variables for c in (v, "(se)")]
        notes = [f"Sample: {self.provenance.sample[0]} to {self.provenance.sample[1]}, {self.provenance.nobs} observations"] if self.provenance else []
        names = {"aic": "Akaike info criterion", "bic": "Schwarz criterion", "hq": "Hannan-Quinn criterion",
                 "fpe": "Final prediction error", "aic_total": "Akaike info criterion (total)",
                 "bic_total": "Schwarz criterion (total)", "hq_total": "Hannan-Quinn criterion (total)"}
        # per observation (Lütkepohl's, as statsmodels and EViews give a VAR's); the totals are comparable with single
        # equations' (pymetrix's own). One note each, so they group as a fitted equation's do.
        notes += [f"{names.get(k, k.upper())}: {v:.6g}" for k, v in e.criteria.items()]
        notes.append(f"Log likelihood: {e.loglike:.3f}")
        notes.append(f"Largest root: {self.roots[0]:.4f} ({'stable' if self.is_stable else 'NOT stable'})")
        if getattr(e, "free", None) is not None:
            held = int((~e.free).sum())
            notes.append(f"{held} coefficient{'s' * (held != 1)} restricted to zero; feasible GLS"
                         + (" iterated to maximum likelihood" if self.restriction is not None and self.restriction.iterate else ""))
        return Table(columns, rows, index=self.regressors, title=f"VAR({self.lags}): {', '.join(self.variables)}", notes=notes)

    def __repr__(self) -> str:
        return f"<FittedVAR VAR({self.lags}) in {', '.join(self.variables)}; {self.provenance.nobs if self.provenance else '?'} observations>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()

    # structural analysis

    @api()
    def identify(self, scheme: Identification) -> FittedVAR:
        if isinstance(scheme, SignRestrictions):
            scheme.check(self.variables)
        _impact(scheme, self.estimate.sigma, self.coefs, self.variables, self._resid_frame())  # validate now
        return replace(self, identification=scheme)

    def _resid_frame(self) -> pd.DataFrame:
        return self.resid.frame[list(self.variables)]

    def impact(self) -> np.ndarray:
        return _impact(self.identification, self.estimate.sigma, self.coefs, self.variables, self._resid_frame())

    @api()
    def heteroskedasticity_svar(self, breaks: Sequence[Any]) -> Any:
        """The shocks identified by shifts in their variances at ``breaks`` (Rigobon 2003; see
        ``pymetrix.identification_extra``)."""
        from .identification_extra import heteroskedasticity_svar

        return heteroskedasticity_svar(self, breaks)

    @renamed_arguments(target="normalise_on")
    @api(version=2)
    def proxy_svar(self, data: Dataset, instrument: str, normalise_on: str | None = None, periods: int = 20,
                   scale: Literal["sd", "unit"] = "sd", draws: int = 500, block: int | None = None, alpha: float = 0.05,
                   seed: int = 0) -> Any:
        """The shock of ``normalise_on`` (default: the VAR's first variable) identified by the external ``instrument``:
        its impact, responses with moving-block bootstrap bands, and the first stage (see ``pymetrix.proxy``). With
        ``scale="unit"`` the shock moves ``normalise_on`` by one on impact. (``normalise_on`` was called ``target``; the
        old name still works, with a DeprecationWarning.)"""
        from .proxy import proxy_svar

        return proxy_svar(self, data, instrument, normalise_on, periods, scale, draws, block, alpha, seed)

    def structural(self) -> Any:
        """Under A-B restrictions: A and B by maximum likelihood, with standard errors, the log likelihood and the
        test of any over-identifying restrictions (see ``pymetrix.structural``)."""
        from .structural import estimate_ab

        if not isinstance(self.identification, AB):
            raise SpecificationError("identify the VAR by A-B restrictions first: var.identify(AB(A=..., B=...))")
        return estimate_ab(self.identification, self.estimate.sigma, self.coefs, self.provenance.nobs, self.variables)

    @property
    def shocks(self) -> list[str]:
        """The structural shocks' names: the variables', or those the sign restrictions give."""
        if isinstance(self.identification, (SignRestrictions, MaxShare)):
            return self.identification.shock_names(self.variables)
        return list(self.variables)

    def sign_draws(self, rotations: int | None = None) -> list[np.ndarray]:
        """The impact matrices the sign restrictions accept at the estimates: the identified set, as drawn."""
        scheme = self.identification
        if not isinstance(scheme, SignRestrictions):
            raise SpecificationError("identify the VAR by sign restrictions first: var.identify(SignRestrictions(...))")
        return _accepted(scheme, self.estimate.sigma, self.coefs, self.variables, rotations or scheme.rotations,
                         make_rng(scheme.seed), self._resid_frame())

    def _sign_irf(self, pairs: Any, periods: int, cumulative: bool, alpha: float, seed: int, label: str) -> IRF:
        """Responses under sign restrictions: the median of the rotations accepted at the estimates, with bands from
        them (the identified set) or, given draws of (A, sigma), from one accepted rotation for each draw."""
        scheme = self.identification
        draws = self.sign_draws()
        point = np.stack([_ma(self.coefs, periods) @ B for B in draws])
        sims, detail = point, f"{len(draws)} accepted rotations; the bands span the identified set"
        if pairs is not None:
            pairs = list(pairs)
            out = [_ma(pairs[i][0], periods) @ B for i, B in _rotations_for_draws(scheme, pairs, self.variables, make_rng(seed))]  # type: ignore[arg-type]
            if not out:
                raise SpecificationError("no draw of the VAR has a rotation satisfying the sign restrictions")
            sims, detail = np.stack(out), f"{len(draws)} accepted rotations; bands from {len(out)} draws of the VAR"
        if cumulative:
            point, sims = np.cumsum(point, axis=1), np.cumsum(sims, axis=1)
        lower, upper = np.quantile(sims, [alpha / 2, 1 - alpha / 2], axis=0)
        return IRF(np.median(point, axis=0), self.variables, self.shocks, lower, upper, cumulative, f"{label} ({detail})", 1 - alpha)

    @api()
    def irf(self, periods: int = 20, orthogonal: bool = True, cumulative: bool = False, draws: int = 500, alpha: float = 0.05, seed: int = 0) -> IRF:
        """Impulse responses to one-standard-deviation structural shocks (or unit reduced-form
        shocks with ``orthogonal=False``). Bands: parametric bootstrap with ``draws`` replications.
        Under sign restrictions: the median of the accepted rotations, with bands from a rotation for each
        bootstrap draw (or, with ``draws=0``, spanning the identified set at the estimates); with zero restrictions
        too, ARW's draws, importance-weighted and resampled."""
        if orthogonal and isinstance(self.identification, SignRestrictions):
            pairs = self._bootstrap(draws, seed) if draws and not self.identification.narrative else None  # narratives need the residuals
            return self._sign_irf(pairs, periods, cumulative, alpha, seed, _scheme_label(self.identification))
        values = self._responses(self.coefs, self.estimate.sigma, periods, orthogonal, cumulative)
        lower = upper = None
        if draws:
            sims = np.stack([self._responses(A, S, periods, orthogonal, cumulative) for A, S in self._bootstrap(draws, seed)])
            lower, upper = np.quantile(sims, [alpha / 2, 1 - alpha / 2], axis=0)
        label = _scheme_label(self.identification) if orthogonal else "reduced-form (unit) shocks"
        shocks = self.shocks if orthogonal else self.variables
        return IRF(values, self.variables, shocks, lower, upper, cumulative, label, 1 - alpha)

    @api()
    def girf(self, periods: int = 20, cumulative: bool = False, draws: int = 500, alpha: float = 0.05, seed: int = 0) -> IRF:
        """Generalised impulse responses (Pesaran and Shin 1998; Koop, Pesaran and Potter 1996): the response to a
        one-standard-deviation shock in each equation, the other shocks at their expected values given it, so
        Phi_h Sigma e_j / sqrt(sigma_jj). Invariant to the ordering; shock j's response is the Cholesky response with
        variable j first. The shocks are correlated, so there is no variance decomposition."""
        values = self._generalised(self.coefs, self.estimate.sigma, periods, cumulative)
        lower = upper = None
        if draws:
            sims = np.stack([self._generalised(A, S, periods, cumulative) for A, S in self._bootstrap(draws, seed)])
            lower, upper = np.quantile(sims, [alpha / 2, 1 - alpha / 2], axis=0)
        return IRF(values, self.variables, self.variables, lower, upper, cumulative, "generalised (Pesaran-Shin)", 1 - alpha)

    @staticmethod
    def _generalised(A: np.ndarray, sigma: np.ndarray, periods: int, cumulative: bool) -> np.ndarray:
        out = _ma(A, periods) @ (sigma / np.sqrt(np.diag(sigma)))
        return np.cumsum(out, axis=0) if cumulative else out

    def _responses(self, A: np.ndarray, sigma: np.ndarray, periods: int, orthogonal: bool, cumulative: bool) -> np.ndarray:
        phi = _ma(A, periods)
        out = phi @ _impact(self.identification, sigma, A, self.variables) if orthogonal else phi
        return np.cumsum(out, axis=0) if cumulative else out

    def _bootstrap(self, draws: int, seed: int):
        """Parametric bootstrap: simulate from the fitted VAR with Gaussian shocks, re-estimate."""
        a, e = self.arrays, self.estimate
        T, K = a.Y.shape
        p, d = self.lags, a.D.shape[1]
        A = self.coefs
        Bd = e.B[:d]
        chol = np.linalg.cholesky(e.sigma)
        start = a.L[0, : K * p].reshape(p, K)  # y[t0-1], y[t0-2], ...
        rng = make_rng(seed)
        from .progress import report

        for k in range(draws):
            report(k / draws, f"bootstrap draw {k + 1} of {draws}")  # the app's progress bar; Cancel stops here
            shocks = rng.standard_normal((T, K)) @ chol.T
            history = list(start[::-1])  # oldest first
            Y = np.empty((T, K))
            for t in range(T):
                y = a.D[t] @ Bd + shocks[t]
                for l in range(1, p + 1):
                    y = y + A[l - 1] @ history[-l]
                Y[t] = y
                history.append(y)
            hist = np.vstack([start[::-1], Y])
            L = np.hstack([hist[p - l : p - l + T] for l in range(1, p + 1)])
            est = _reestimate(self, a, p, Y, L)
            yield _lag_matrices(est.B, d, K, p), est.sigma

    @api()
    def fevd(self, periods: int = 20) -> FEVD:
        """Forecast-error variance decomposition under the current identification."""
        theta = _ma(self.coefs, periods - 1) @ self.impact()
        contributions = np.cumsum(theta**2, axis=0)  # (h, variable, shock)
        shares = contributions / contributions.sum(axis=2, keepdims=True)
        return FEVD(shares, self.variables, self.shocks, _scheme_label(self.identification))

    @api()
    def forecast(self, data: Dataset, span: tuple[Any, Any], alpha: float = 0.05, dynamic: bool = True) -> Dataset:
        """Dynamic forecasts from the period before ``span``; intervals reflect shock uncertainty only."""
        from .data import Dataset
        from .estimation.design import instrument_series
        from .metadata import DatasetMeta

        if not dynamic:
            raise SpecificationError("VAR forecasts are dynamic (multi-step); use dynamic=True")
        e = self.estimate
        periods, D, history = self._forecast_inputs(data, span)
        K, p = len(self.variables), self.lags
        d = len(self.arrays.d_names)
        Bd, A = e.B[:d], self.coefs
        out = np.empty((len(periods), K))
        for i in range(len(periods)):
            y = np.asarray(D[i]) @ Bd if d else np.zeros(K)
            for l in range(1, p + 1):
                y = y + A[l - 1] @ history[-l]
            out[i] = y
            history.append(y)
        phi = _ma(A, len(periods) - 1)
        mse = np.cumsum(np.einsum("hij,jk,hlk->hil", phi, e.sigma, phi), axis=0)
        se = np.sqrt(np.diagonal(mse, axis1=1, axis2=2))
        z = sps.norm.ppf(1 - alpha / 2)
        cols = {}
        for j, v in enumerate(self.variables):
            cols[v] = out[:, j]
            cols[f"{v}_se"] = se[:, j]
            cols[f"{v}_lower"] = out[:, j] - z * se[:, j]
            cols[f"{v}_upper"] = out[:, j] + z * se[:, j]
        meta = DatasetMeta(title=f"VAR({p}) forecast", description="Intervals reflect shock uncertainty only, not parameter uncertainty.")
        return Dataset(pd.DataFrame(cols, index=periods), data.axes, meta=meta)

    @api()
    def conditional_forecast(self, data: Dataset, span: tuple[Any, Any], conditions: Mapping[str, Any], alpha: float = 0.05,
                             draws: int | None = None, seed: int = 0, shocks: Sequence[str] | None = None,
                             shock_conditions: Mapping[str, Any] | None = None) -> Dataset:
        """Forecasts given conditions on some variables: exact paths (Waggoner and Zha 1999) or ranges, met by all the
        shocks or by the identified ``shocks`` named (a structural scenario), with intervals.

        ``conditions`` maps a variable to a condition held over every forecast period or to period -> condition
        (``{"r": {"2025Q1": 2.0, "2025Q2": 2.25}}``). A condition is a number (exact) or a range:
        ``Bounds(3, 4)``, ``Bounds(lower=3)``, ``Bounds(upper=4)``, ``(3, 4)`` or ``{"lower": 3, "upper": 4}``.

        With every shock free to move (``shocks=None``), the shocks that meet exact conditions are the most likely
        ones, so the result is the conditional expectation given the conditions and does not depend on how shocks are
        identified. ``shocks`` names the identified shocks that may move to meet them; the others keep their
        unconditional distribution (Antolin-Diaz, Petrella and Rubio-Ramirez 2021). ``shock_conditions`` conditions
        the structural shocks themselves, as ``conditions`` (``{"r": 0.0}``: no policy shocks).

        Exact conditions alone give the closed form: columns ``<v>``, ``<v>_se``, ``<v>_lower``, ``<v>_upper``
        (mean +/- z se); conditioned values have no uncertainty. With a range, the forecast is simulated (``draws``
        paths, default 5,000, with ``seed``; see ``pymetrix.var_conditions``) and the columns are the mean, sd,
        median and the ``1 - alpha`` quantile interval; every draw meets every bound. Intervals reflect the shocks,
        not parameter uncertainty (a Bayesian VAR's include that)."""
        from . import var_conditions as vc
        from .data import Dataset
        from .metadata import DatasetMeta

        e = self.estimate
        periods, D, history = self._forecast_inputs(data, span)
        d, K, H = len(self.arrays.d_names), len(self.variables), len(periods)
        point = _point_path(self.coefs, e.B[:d], D, history, H)
        structural = shocks is not None or bool(shock_conditions)
        names = self.shocks if structural else list(self.variables)
        conds = vc.parse(conditions, self.variables, periods) + vc.parse(shock_conditions, names, periods, "shock", on_shock=True)
        vc.check(conds)
        info = vc.record(conds, periods, self.variables, names, shocks)
        how = vc.summary(conds, self.variables, names) + (f", met by the shocks {', '.join(shocks)}" if shocks is not None else "")
        title = f"VAR({self.lags}) conditional forecast"
        if all(c.exact for c in conds):
            if not structural:  # Waggoner-Zha
                exact = [(c.i, c.k, c.value) for c in conds]
                mean, cov, _, _ = _conditional_moments(self.coefs, e.sigma, point, exact)
            else:
                impact = self.impact()
                mean, cov = vc.moments(self.coefs, impact, point, conds, vc.driving_mask(shocks, names, H))
            se = np.sqrt(np.clip(np.diag(cov), 0, None)).reshape(mean.shape)
            for c in conds:  # known exactly (rounding would leave a variance of about 1e-14)
                if not c.on_shock:
                    se[c.i, c.k], mean[c.i, c.k] = 0.0, c.value
            z = sps.norm.ppf(1 - alpha / 2)
            cols = {}
            for j, v in enumerate(self.variables):
                cols[v] = mean[:, j]
                cols[f"{v}_se"] = se[:, j]
                cols[f"{v}_lower"] = mean[:, j] - z * se[:, j]
                cols[f"{v}_upper"] = mean[:, j] + z * se[:, j]
            meta = DatasetMeta(title=title, description=(
                f"Given {how} (Waggoner-Zha: the most likely shocks that meet them). Intervals reflect the shocks the "
                "conditions leave free, not parameter uncertainty."), extra=info)
            return Dataset(pd.DataFrame(cols, index=periods), data.axes, meta=meta)
        impact = self.impact() if structural else np.linalg.cholesky(e.sigma)
        rng = make_rng(seed)
        n = int(draws) if draws else 5000
        prepared = vc.prepare(self.coefs, impact, point, conds, vc.driving_mask(shocks, names, H), n, rng)
        paths = vc.draw_paths(prepared, rng)[0].reshape(n, H, K)
        return self._summarised(data, periods, paths, alpha, DatasetMeta(title=title, description=(
            f"Given {how}: mean and {1 - alpha:.0%} interval of {n} simulated paths (ranges by Gibbs sampling from the "
            "truncated normal). Intervals reflect the shocks, not parameter uncertainty."), extra=info))

    def _summarised(self, data: Dataset, periods: pd.PeriodIndex, paths: np.ndarray, alpha: float, meta: Any) -> Dataset:
        """Simulated paths (draws, periods, variables) as forecast columns: mean, sd, median and the 1 - alpha interval."""
        from .data import Dataset

        cols: dict[str, np.ndarray] = {}
        lo, hi = np.quantile(paths, [alpha / 2, 1 - alpha / 2], axis=0)
        mean, sd, median = paths.mean(axis=0), paths.std(axis=0, ddof=1), np.median(paths, axis=0)
        for j, v in enumerate(self.variables):
            cols[v] = mean[:, j]
            cols[f"{v}_se"] = sd[:, j]
            cols[f"{v}_median"] = median[:, j]
            cols[f"{v}_lower"] = lo[:, j]
            cols[f"{v}_upper"] = hi[:, j]
        return Dataset(pd.DataFrame(cols, index=periods), data.axes, meta=meta)

    def _forecast_inputs(self, data: Dataset, span: tuple[Any, Any]) -> tuple[pd.PeriodIndex, list[list[float]], list[np.ndarray]]:
        """The forecast periods, their deterministic and exogenous terms, and the last ``lags`` observations."""
        from .data import Dataset
        from .estimation.design import instrument_series

        a = self.arrays
        freq = data.frame.index.freq
        from .data.sample import as_span
        span = as_span(span)  # '1990Q1:2019Q4' or a pair
        start, end = pd.Period(str(span[0]), freq=freq), pd.Period(str(span[1]), freq=freq)
        periods = pd.period_range(start, end, freq=freq, name=data.time_axis.name)  # type: ignore[union-attr]
        K, p = len(self.variables), self.lags
        history = []
        for l in range(p, 0, -1):
            row = data.frame.reindex([start - l])[list(self.variables)].to_numpy(float)[0]
            if not np.all(np.isfinite(row)):
                raise SpecificationError(f"data for {start - l} is needed to start the forecast")
            history.append(row)
        ext = Dataset(data.frame.reindex(periods.union(data.frame.index)), data.axes, data.variables)
        D = []
        for i, period in enumerate(periods):
            terms = []
            if self.spec.trend in ("c", "ct"):
                terms.append(1.0)
            if self.spec.trend == "ct":
                terms.append(float(len(a.index) + (period - a.index[-1]).n))
            for name, s in instrument_series(self.spec.exog, ext).items():
                value = s.loc[period]
                if not np.isfinite(value):
                    raise SpecificationError(f"exogenous {name} is missing for {period}")
                terms.append(value)
            D.append(terms)
        return periods, D, history

    @api()
    def historical_decomposition(self, draws: int = 0, alpha: float = 0.05, seed: int = 0) -> Any:
        """Each variable over the estimation sample, split into the contributions of the identified structural shocks
        and the baseline: the path from the first observations and the deterministic and exogenous terms, with no
        shocks. The parts add up to the data exactly.

        The structural shocks are the residuals through the impact matrix of the current identification,
        ``e(t) = P^-1 u(t)``; shock j's contribution is ``sum over s <= t of Phi(t - s) P[:, j] e_j(s)``. Under sign
        restrictions the impact matrix is the median-target member of the identified set. Returns a
        ``Decomposition``: ``contributions[variable]`` (periods x shocks), ``deviations`` (the data less the
        baseline), ``baseline`` and ``shocks``; ``plot()`` draws the contributions as stacked bars.

        With ``draws``, ``1 - alpha`` bands for each contribution, pointwise quantiles over the decompositions of the
        data under each draw: a Bayesian VAR's posterior draws (each with its own coefficients, residuals, baseline and
        impact matrix: under sign or zero and sign restrictions, a rotation accepted for that draw), or under sign
        restrictions the identified set at the estimates (``draws`` accepted rotations). Every draw's parts add up to
        the data. The point decomposition is unchanged (at a BVAR's posterior mean); ``lower``, ``median`` and
        ``upper`` hold the bands, ``bands_table()`` and ``plot_bands()`` show them."""
        a, K = self.arrays, len(self.variables)
        d = len(a.d_names)
        A, Bd = self.coefs, self.estimate.B[:d]
        P = self.impact()
        if P.shape != (K, K):
            raise SpecificationError(f"a historical decomposition needs as many shocks as variables ({K}); the identification gives {P.shape[1]}")
        U, baseline = self._hd_parts(A, Bd)  # at the coefficients used (a BVAR's posterior mean)
        result = _historical(A, P, U, a.Y, baseline, a.index, self.variables, self.shocks, _scheme_label(self.identification))
        if not draws:
            return result
        if not 0 < alpha < 1:
            raise SpecificationError(f"alpha is between 0 and 1, e.g. 0.05; got {alpha}")
        contributions, _, source = self._hd_samples(int(draws), seed)
        lower, median, upper = np.quantile(contributions, [alpha / 2, 0.5, 1 - alpha / 2], axis=0)
        names = list(self.shocks)

        def frames(x: np.ndarray) -> dict[str, pd.DataFrame]:
            return {v: pd.DataFrame(x[:, i, :], index=a.index, columns=names) for i, v in enumerate(self.variables)}

        return replace(result, lower=frames(lower), median=frames(median), upper=frames(upper), coverage=1 - alpha,
                       band_source=f"{len(contributions)} {source}")

    def _hd_parts(self, A: np.ndarray, Bd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The residuals and the baseline (the path with no shocks) of the data under coefficients (A, Bd)."""
        a, K, p = self.arrays, len(self.variables), self.lags
        T = len(a.Y)
        lags = [a.L[:, (l - 1) * K : l * K] for l in range(1, p + 1)]
        U = a.Y - a.D @ Bd - sum(L @ A[l].T for l, L in enumerate(lags))
        baseline = np.empty((T, K))
        history = list(a.L[0, : K * p].reshape(p, K)[::-1])  # y[t0-p], ..., y[t0-1]
        for t in range(T):
            y = a.D[t] @ Bd
            for l in range(1, p + 1):
                y = y + A[l - 1] @ history[-l]
            baseline[t] = y
            history.append(y)
        return U, baseline

    def _hd_draws(self, draws: int, seed: int) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], str]:
        """(A, Bd, P) for each draw the bands come from, and what they are: here, the identified set at the estimates."""
        scheme = self.identification
        if not isinstance(scheme, SignRestrictions):
            raise SpecificationError("bands for a historical decomposition come from a Bayesian VAR's posterior draws, or from "
                                     "the identified set under sign restrictions; this VAR is point-identified")
        d = len(self.arrays.d_names)
        Bd = self.estimate.B[:d]
        return [(self.coefs, Bd, B) for B in self.sign_draws(draws)], "accepted rotations at the estimates (the identified set)"

    def _hd_samples(self, draws: int, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
        """The decomposition under each draw: contributions (draws, periods, variable, shock) and baselines (draws,
        periods, variable). For each draw the contributions and the baseline add up to the data."""
        inputs, source = self._hd_draws(draws, seed)
        if not inputs:
            raise SpecificationError("no draw gives an impact matrix for the bands")
        T, K = self.arrays.Y.shape
        contributions = np.empty((len(inputs), T, K, K))
        baselines = np.empty((len(inputs), T, K))
        for n, (A, Bd, P) in enumerate(inputs):
            U, baselines[n] = self._hd_parts(A, Bd)
            contributions[n] = _contributions(A, P, U)
        return contributions, baselines, source

    # tests

    @api()
    def test(self, name: str, *args: Any, **opts: Any) -> TestResult:
        from .stats import run_test

        return run_test(self, name, *args, **opts)

    def tests(self) -> list[str]:
        from .stats import available_tests

        return available_tests(self)


@dataclass(frozen=True, eq=False)
class ResidualCorrelogram:
    """Auto- and cross-correlations of a VAR's or VECM's residuals: ``values[h - 1, i, j]`` is the correlation of
    residual i in period t with residual j in period t - h (demeaned, divided by T, as statsmodels' ``resid_acorr``);
    under no autocorrelation each lies within ±1.96/sqrt(T) with probability about 0.95."""

    values: np.ndarray
    variables: Sequence[str]
    nobs: int
    title: str = "Residual correlogram"

    @property
    def band(self) -> float:
        return 1.96 / np.sqrt(self.nobs)

    def table(self) -> Table:
        lags = range(1, len(self.values) + 1)
        cols = {f"{a}, {b}[-h]": self.values[:, i, j] for i, a in enumerate(self.variables) for j, b in enumerate(self.variables)}
        frame = pd.DataFrame(cols, index=pd.Index(list(lags), name="h"))
        outside = int((np.abs(self.values) > self.band).sum())
        notes = [f"{self.nobs} observations; approximate 95% band ±{self.band:.3f}; {outside} of {self.values.size} outside it",
                 "Column 'a, b[-h]': the correlation of a's residual with b's h periods earlier"]
        return Table.from_frame(frame, title=self.title, notes=notes)

    def plot(self) -> Figure:
        """A panel for each pair (a row per residual, a column per lagged residual): bars by lag within the band."""
        from .figures import Bars, Panel, Rule

        lags = pd.Index(range(1, len(self.values) + 1), name="lag")
        panels = []
        for i, a in enumerate(self.variables):
            for j, b in enumerate(self.variables):
                layers = (Rule(0.0), Bars(pd.Series(self.values[:, i, j], index=lags), f"{a}, {b}"), Rule(self.band, "--"), Rule(-self.band, "--"))
                panels.append(Panel(layers, title=f"{a}, {b}[-h]", legend=False))
        return Figure(tuple(panels), title=self.title, columns=len(self.variables))

    def __str__(self) -> str:
        return str(self.table())

    def _repr_html_(self) -> str:
        return self.table()._repr_html_()


def residual_correlogram(U: np.ndarray, variables: Sequence[str], lags: int, title: str) -> ResidualCorrelogram:
    """The correlogram of residuals ``U`` (T x K)."""
    T = len(U)
    lags = int(lags)
    if not 1 <= lags < T:
        raise SpecificationError(f"lags must be between 1 and {T - 1}")
    x = U - U.mean(axis=0)
    cov0 = x.T @ x / T
    scale = np.sqrt(np.outer(np.diag(cov0), np.diag(cov0)))
    values = np.stack([(x[h:].T @ x[:-h] / T) / scale for h in range(1, lags + 1)])
    return ResidualCorrelogram(values, tuple(variables), T, title)


_FAVAR_HINT = ("a factor-augmented VAR is fitted by favar(data, panel=[...], observed=[...], factors=3 or 'icp2', lags=...): "
               "it estimates the factors from the panel itself (Estimate → VARs and local projections → Factor-augmented VAR)")


def _companion_eigenvalues(A: np.ndarray) -> np.ndarray:
    """The eigenvalues of the companion matrix of a VAR with coefficients ``A`` (lags, K, K), largest modulus first."""
    p, K = A.shape[0], A.shape[1]
    companion = np.zeros((K * p, K * p))
    companion[:K, :] = np.hstack(list(A))
    companion[K:, :-K] = np.eye(K * (p - 1))
    z = np.linalg.eigvals(companion)
    return z[np.argsort(-np.abs(z), kind="stable")]


def _historical(A: np.ndarray, P: np.ndarray, U: np.ndarray, Y: np.ndarray, baseline: np.ndarray, index: pd.PeriodIndex,
                variables: Sequence[str], shocks: Sequence[str], label: str) -> Any:
    """A historical decomposition from the levels-VAR coefficients ``A``, the impact matrix ``P``, the residuals, the
    data and the path without shocks: shocks ``P^-1 u(t)``, each one's contribution through the moving-average
    weights ``Phi(h) P``."""
    from .models.decompose import Decomposition

    T = len(U)
    e = np.linalg.solve(P, U.T).T
    theta = _ma(A, T - 1) @ P  # theta[h, variable, shock]
    contributions = np.stack([np.einsum("sij,sj->ij", theta[t::-1], e[: t + 1]) for t in range(T)])
    names = list(shocks)
    parts = {v: pd.DataFrame(contributions[:, i, :], index=index, columns=names) for i, v in enumerate(variables)}
    base = pd.DataFrame(baseline, index=index, columns=list(variables))
    return Decomposition(parts, pd.DataFrame(Y, index=index, columns=list(variables)) - base, pd.DataFrame(e, index=index, columns=names),
                         title=f"Historical decomposition ({label})",
                         measure="relative to the baseline: deterministic terms and initial conditions", baseline=base)


def _contributions(A: np.ndarray, P: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Each shock's contribution (periods, variable, shock) by the VAR's recursion: x_j(t) = sum_l A_l x_j(t - l) +
    P[:, j] e_j(t) from zero, with e(t) = P^-1 u(t)."""
    p, K = A.shape[0], A.shape[1]
    T = len(U)
    e = np.linalg.solve(P, U.T).T
    out = np.zeros((T, K, K))
    for t in range(T):
        x = P * e[t]
        for l in range(1, min(p, t) + 1):
            x = x + A[l - 1] @ out[t - l]
        out[t] = x
    return out


def _scheme_label(scheme: Any) -> str:
    if isinstance(scheme, Cholesky):
        return f"Cholesky ({', '.join(scheme.order)})" if scheme.order else "Cholesky (VAR order)"
    if isinstance(scheme, MaxShare):
        return (f"Max share of {scheme.variable}'s forecast-error variance, horizons {scheme.start} to {scheme.horizon}"
                + (" (no impact effect: a news shock)" if scheme.zero_impact else ""))
    if isinstance(scheme, AB):
        return "A-B restrictions (maximum likelihood)" + (", with long-run restrictions" if scheme.long_run is not None else "")
    if isinstance(scheme, SignRestrictions):
        parts = "; ".join(f"{shock}: " + ", ".join(f"{v}{'+' if s > 0 else '-'}" for v, s in r.items()) for shock, r in scheme.signs.items() if r)
        if not scheme.has_zeros:
            return f"Sign restrictions ({parts}; horizons 0 to {scheme.horizon})" + (" with narrative restrictions" if scheme.narrative else "")
        zeros = "; ".join([f"{shock}: {', '.join(vs)} zero on impact" for shock, vs in scheme.zeros.items()]
                          + [f"{shock}: {', '.join(vs)} zero in the long run" for shock, vs in scheme.long_run_zeros.items()])
        signs = f"{parts}; horizons 0 to {scheme.horizon}; " if parts else ""
        return (f"Zero and sign restrictions ({signs}{zeros}; ARW importance weights)"
                + (" with narrative restrictions" if scheme.narrative else ""))
    return type(scheme).__name__


def _horizon_rows(n: int) -> list[int]:
    return [h for h in (0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 40) if h < n] or list(range(n))


@dataclass(frozen=True, eq=False)
class IRF:
    """Responses ``values[h, response, shock]``; optional bootstrap bands."""

    values: np.ndarray
    responses: Sequence[str]
    shocks: Sequence[str]
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    cumulative: bool = False
    identification: str = ""
    coverage: float = 0.95

    def __getitem__(self, key: tuple[str, str]) -> pd.Series:
        response, shock = key
        i, k = list(self.responses).index(response), list(self.shocks).index(shock)
        return pd.Series(self.values[:, i, k], index=pd.RangeIndex(len(self.values), name="horizon"), name=f"{response} <- {shock}")

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        """Every response (and band) by horizon; or one of its views by name ("Selected horizons")."""
        if view is not None:  # one of its views by name, as for every result (pymetrix.views)
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        cols = {}
        for i, r in enumerate(self.responses):
            for k, s in enumerate(self.shocks):
                cols[f"{r} <- {s}"] = self.values[:, i, k]
                if self.lower is not None:
                    cols[f"{r} <- {s} lower"] = self.lower[:, i, k]
                    cols[f"{r} <- {s} upper"] = self.upper[:, i, k]  # type: ignore[index]
        return pd.DataFrame(cols, index=pd.RangeIndex(len(self.values), name="horizon"))

    def table(self, shock: str | None = None) -> Table:
        shocks = [shock] if shock else list(self.shocks)
        rows_h = _horizon_rows(len(self.values))
        cols = [f"{r} <- {s}" for s in shocks for r in self.responses]
        frame = self.to_frame().loc[rows_h, cols]
        kind = "Cumulative responses" if self.cumulative else "Impulse responses"
        notes = [f"Identification: {self.identification}"]
        if self.lower is not None:
            notes.append(f"{self.coverage:.0%} bands available via to_frame() (parametric bootstrap)")
        return Table.from_frame(frame, title=kind, notes=notes)

    def plot(self, response: str | None = None, shock: str | None = None) -> Figure:
        """Responses (rows) to shocks (columns), with bootstrap bands when available."""
        from .figures import responses_figure

        return responses_figure(self, [response] if response else None, [shock] if shock else None)

    def heatmap(self, horizons: Sequence[int] | None = None) -> Figure:
        """Every response to every shock at once: a row per response and shock, a column per horizon (the selected
        horizons by default), coloured on a scale symmetric about zero."""
        from .figures import heatmap_figure

        rows_h = list(horizons) if horizons is not None else _horizon_rows(len(self.values))
        frame = pd.DataFrame({f"{r} <- {s}": self.values[rows_h, i, k] for k, s in enumerate(self.shocks) for i, r in enumerate(self.responses)},
                             index=pd.Index(rows_h, name="horizon")).T
        kind = "Cumulative responses" if self.cumulative else "Impulse responses"
        return heatmap_figure(frame, title=f"{kind} by horizon: {self.identification}", label="response")

    def __repr__(self) -> str:
        return f"<IRF {len(self.values) - 1} periods, {len(self.shocks)} shocks, {self.identification}>"

    def _repr_html_(self) -> str:
        return self.table()._repr_html_()


@dataclass(frozen=True, eq=False)
class FEVD:
    """Shares ``values[h, variable, shock]`` of the h+1-step forecast-error variance."""

    values: np.ndarray
    variables: Sequence[str]
    shocks: Sequence[str]
    identification: str = ""
    lower: np.ndarray | None = None  # bands, like values: quantiles of the shares over draws (BVAR-SV)
    upper: np.ndarray | None = None
    coverage: float | None = None  # the bands' probability, e.g. 0.68
    band_source: str = ""  # what the draws are, e.g. "500 posterior draws"

    @property
    def has_bands(self) -> bool:
        return self.lower is not None and self.upper is not None

    def bands_table(self, variable: str) -> Table:
        """The variable's shares (%) by horizon with their bands: for each shock the point share, then the lower and
        upper quantiles over the draws."""
        if not self.has_bands:
            raise SpecificationError("this variance decomposition has no bands")
        i = list(self.variables).index(variable)
        rows_h = [h - 1 for h in _horizon_rows(len(self.values) + 1) if h >= 1]
        cols = {}
        for j, s in enumerate(self.shocks):
            cols[s] = 100 * self.values[rows_h, i, j]
            cols[f"{s} lower"] = 100 * self.lower[rows_h, i, j]  # type: ignore[index]
            cols[f"{s} upper"] = 100 * self.upper[rows_h, i, j]  # type: ignore[index]
        frame = pd.DataFrame(cols, index=pd.Index([h + 1 for h in rows_h], name="horizon"))
        notes = [f"{self.coverage:.0%} bands: pointwise quantiles over {self.band_source}" if self.coverage else self.band_source,
                 "The point shares add up to 100 at each horizon; the bands' ends do not"]
        return Table.from_frame(frame, title=f"Variance decomposition (%): {variable}, bands", notes=notes)

    def table(self, variable: str | None = None) -> Table:
        variables = [variable] if variable else list(self.variables)
        rows_h = [h for h in _horizon_rows(len(self.values) + 1) if h >= 1]
        frames = []
        for v in variables:
            i = list(self.variables).index(v)
            frames.append(pd.DataFrame(100 * self.values[[h - 1 for h in rows_h], i, :], index=pd.Index(rows_h, name="horizon"), columns=[f"{v}: {s}" for s in self.shocks]))
        notes = [f"Identification: {self.identification}"] + ([f"Bands ({self.coverage:.0%}) from {self.band_source}: bands_table(variable)"] if self.has_bands else [])
        return Table.from_frame(pd.concat(frames, axis=1), title="Variance decomposition (%)", notes=notes)

    def heatmap(self, horizon: int | None = None) -> Figure:
        """Each variable's forecast-error variance split by shock, in %, at ``horizon`` steps (the last by default):
        a row per variable, a column per shock; each row adds up to 100."""
        from .figures import heatmap_figure

        h = len(self.values) if horizon is None else int(horizon)
        if not 1 <= h <= len(self.values):
            raise SpecificationError(f"horizon is 1 to {len(self.values)}")
        frame = pd.DataFrame(100 * self.values[h - 1], index=list(self.variables), columns=list(self.shocks))
        return heatmap_figure(frame, title=f"Variance decomposition at {h} steps, % ({self.identification})", label="% of the variance",
                              vmin=0.0, vmax=100.0, fmt="{:.0f}")

    def __repr__(self) -> str:
        return f"<FEVD {len(self.values)} periods, {self.identification}>"

    def _repr_html_(self) -> str:
        return self.table()._repr_html_()


@call_migration("pymetrix:fitted_var.proxy_svar", 1)
def _proxy_svar_normalise_on(arguments: dict[str, Any]) -> dict[str, Any]:  # version 2: target became normalise_on
    arguments["normalise_on"] = arguments.pop("target", None)
    return arguments


register("models", VAR)
