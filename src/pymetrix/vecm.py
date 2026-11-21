"""Vector error-correction models and Johansen cointegration tests (statsmodels backend).

    spec = VECM(["ly", "lc", "li"], lags=2, rank="trace", deterministic="co")
    spec.johansen(data)             # trace and maximum-eigenvalue tests
    vecm = spec.fit(data)
    vecm.summary(); vecm.irf(20); vecm.forecast(data, ("2025Q1", "2026Q4"))

``deterministic`` uses statsmodels' codes, which correspond to the usual
five Johansen cases: "n" (1: none), "ci" (2: constant restricted to the
cointegrating relations), "co" (3: unrestricted constant), "coli" (4:
unrestricted constant, trend restricted to the relations), "colo" (5:
unrestricted constant and trend).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd

from .results import BackendResult
from ._api import api
from .errors import SpecificationError
from ._serial import to_jsonable
from ._capture import data_fingerprint, records_warnings, with_effective_sample
from ._version import __version__
from .adapters import require
from .adapters import statsmodels as statsmodels_adapter
from .adapters.base import AdapterSpec, BackendOptions, check_backend_options, with_backend
from .forms import Widget
from .plugins import register
from .metadata import Provenance
from .output import Table
from .var import FEVD, IRF, Cholesky, LagSelection, _impact, _ma, _scheme_label
from ._random import generator as make_rng

if TYPE_CHECKING:
    from .data import Dataset
    from .stats import TestResult

Deterministic = Literal["n", "ci", "co", "coli", "colo"]
CASES = {"n": 1, "ci": 2, "co": 3, "coli": 4, "colo": 5}
# Johansen test deterministic order used for each case when choosing the rank
_DET_ORDER = {"n": -1, "ci": 0, "co": 0, "coli": 1, "colo": 1}


def _frame(variables: Sequence[str], data: Dataset, span: tuple | None, exog: Sequence[str] = (), exog_coint: Sequence[str] = ()):
    from .estimation.design import instrument_series

    if data.kind != "time_series":
        raise SpecificationError("VECMs and Johansen tests need a single time series")
    missing = [v for v in variables if v not in data]
    if missing:
        raise SpecificationError(f"variables {missing} are not in the data")
    cols = {v: data[v].astype(float) for v in variables}
    ex = instrument_series(exog, data)
    exc = instrument_series(exog_coint, data)
    frame = pd.DataFrame({**cols, **{f"__x__{k}": v for k, v in ex.items()}, **{f"__c__{k}": v for k, v in exc.items()}})
    frame = frame[data.sample_mask(span)]
    frame = frame[np.isfinite(frame.to_numpy(dtype=float)).all(axis=1)]
    index = frame.index
    if len(index) == 0:
        raise SpecificationError("no complete observations")
    if len(pd.period_range(index[0], index[-1], freq=index.freq)) != len(index):
        raise SpecificationError("the sample has gaps; VECMs need consecutive periods")
    endog = frame[list(variables)].to_numpy(float)
    X = frame[[c for c in frame.columns if c.startswith("__x__")]].to_numpy(float) if ex else None
    XC = frame[[c for c in frame.columns if c.startswith("__c__")]].to_numpy(float) if exc else None
    return endog, X, XC, index


@dataclass(frozen=True, eq=False)
class JohansenResult:
    """Trace and maximum-eigenvalue statistics for each null rank r = 0 .. K-1."""

    table: Table
    rank_trace: int
    rank_maxeig: int
    eigenvalues: np.ndarray
    signif: float
    provenance: Provenance | None = None

    def __str__(self) -> str:
        return str(self.table)

    def _repr_html_(self) -> str:
        return self.table._repr_html_()


def johansen(data: Dataset, variables: Sequence[str], lags: int = 1, det_order: Literal[-1, 0, 1] = 0,
             signif: float = 0.05, span: tuple | None = None) -> JohansenResult:
    """Johansen cointegration tests (statsmodels ``coint_johansen``). ``lags``: lags in differences;
    ``det_order``: -1 none, 0 constant, 1 linear trend. Critical values are tabulated for 10%, 5% and 1%."""
    vecm = require("statsmodels.tsa.vector_ar.vecm")
    endog, _, _, index = _frame(variables, data, span)
    with statsmodels_adapter.ADAPTER.run(statsmodels_adapter.JOHANSEN) as run:
        res = run.call("test", vecm.coint_johansen, endog, det_order=det_order, k_ar_diff=lags)
    level = {0.10: 0, 0.05: 1, 0.01: 2}.get(round(signif, 2))
    if level is None:
        raise SpecificationError("signif must be 0.10, 0.05 or 0.01 (the tabulated critical values)")
    K = len(variables)
    rows = []
    for r in range(K):
        rows.append([r, float(np.real(res.eig[r])), res.lr1[r], *res.cvt[r], res.lr2[r], *res.cvm[r]])
    columns = ["r0", "eigenvalue", "trace", "trace 10%", "trace 5%", "trace 1%", "max-eig", "max-eig 10%", "max-eig 5%", "max-eig 1%"]

    def rank(stats: np.ndarray, crit: np.ndarray) -> int:
        for r in range(K):
            if stats[r] < crit[r, level]:
                return r
        return K

    r_trace, r_max = rank(res.lr1, res.cvt), rank(res.lr2, res.cvm)
    det = {-1: "no deterministic terms", 0: "constant", 1: "linear trend"}[det_order]
    notes = [
        f"{', '.join(variables)}; {lags} lagged difference{'s' * (lags != 1)}; {det}; {len(index)} observations",
        f"Rank at {signif:.0%}: trace {r_trace}, maximum eigenvalue {r_max} (sequential tests from r0 = 0)",
    ]
    table = Table(columns, [[int(row[0]), *row[1:]] for row in rows], title="Johansen cointegration test", notes=notes)
    spec = {"variables": list(variables), "lags": lags, "det_order": det_order, "signif": signif}
    prov = _backend_provenance(data, variables, index, lags + 1, span, {"test": "johansen", "settings": spec}, run.record)
    return JohansenResult(table, r_trace, r_max, np.real(np.asarray(res.eig)), signif, prov)  # NumPy 2.5: complex, zero imaginary part


def _backend_provenance(data: Dataset, variables: Sequence[str], index: pd.Index, taken: int, span: tuple | None,
                        method: dict[str, Any], record: Any) -> Provenance:
    """Where a test or a lag selection run by statsmodels comes from: the series and rows it was given, the first
    ``taken`` of them presample (the lagged differences), and what statsmodels was given and said."""
    prov = Provenance(
        data_vintages={v: data.variables[v].vintage for v in variables if v in data.variables},
        spec={"series": list(variables)}, method=method, sample=(str(index[0]), str(index[-1])), nobs=len(index) - taken,
        dropped=0, package_version=__version__, data_fingerprint=data_fingerprint(data, list(variables), index))
    prov = with_backend(prov, record)
    return with_effective_sample(prov, data, index[taken:], span=span, variables=variables, presample=index[:taken])


@dataclass(frozen=True, eq=False)
class VECM:
    """Vector error-correction model (statsmodels), with the cointegrating rank chosen by Johansen tests or given.

    For series that are each integrated of order one but share long-run relations (cointegration): the
    differences respond to last period's deviations from those relations. beta holds the relations, normalised on
    the first variables, so the order of the variables decides how they read; alpha holds the speeds of
    adjustment, and a variable whose alpha is near zero does not correct towards the relations (it is weakly
    exogenous). The Johansen tests that choose the rank depend on the lags and on the deterministic case, and
    over-reject in small samples; check that the rank makes economic sense. If the tests find no cointegration a
    VAR in differences is the model; if they find full rank the series look stationary and a VAR in levels is.
    Impulse responses of a VECM need not die out: shocks can move the levels permanently.
    """

    code_positional: ClassVar[int] = 1  # generated code gives the first field by position: VECM(...)

    label: ClassVar[str] = "VECM"
    wraps: ClassVar[str] = "statsmodels.tsa.vector_ar.vecm.VECM"
    docs: ClassVar[str] = "https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.vecm.VECM.html"
    topics: ClassVar[tuple[str, ...]] = ('cointegration',)
    requires: ClassVar[tuple[str, ...]] = ("statsmodels",)

    variables: Annotated[Sequence[str], Widget("text", help="the endogenous series (at least two)")]
    lags: Annotated[int, Widget("number", min=0, max=24, help="lagged differences (one fewer than the lags of the VAR in levels); select_order compares them")] = 1
    rank: Annotated[int | Literal["trace", "maxeig"], Widget("text", help="number of cointegrating relations, or trace / maxeig to test for it")] = "trace"
    deterministic: Annotated[Deterministic, Widget("choice", help="n none; ci constant in the relations; co constant outside; coli, colo add a linear trend inside or outside")] = "co"
    exog: Annotated[Sequence[str], Widget("text", advanced=True, help="unrestricted exogenous expressions")] = ()
    exog_coint: Annotated[Sequence[str], Widget("text", advanced=True, help="exogenous expressions inside the cointegrating relations")] = ()
    seasons: Annotated[int, Widget("number", min=0, max=52, advanced=True, help="seasonal dummies, e.g. 4 for quarterly data; 0 = none")] = 0
    first_season: Annotated[int, Widget("number", min=0, max=51, advanced=True, help="position of the first observation in the seasonal cycle (0 to seasons - 1), aligning the dummies")] = 0
    signif: Annotated[float, Widget("number", min=0.01, max=0.1, step=0.01, advanced=True, help="significance level for choosing the rank: 0.10, 0.05 or 0.01")] = 0.05
    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other statsmodels arguments; model.x for the constructor")] = field(default_factory=dict)

    adapter_spec: ClassVar[AdapterSpec] = statsmodels_adapter.VECM

    def __post_init__(self) -> None:
        check_backend_options(self)
        for name in ("variables", "exog", "exog_coint"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len(self.variables) < 2:
            raise SpecificationError("a VECM needs at least two variables")
        if self.deterministic not in CASES:
            raise SpecificationError(f"deterministic must be one of {list(CASES)}")

    def johansen(self, data: Dataset, span: tuple | None = None) -> JohansenResult:
        return johansen(data, self.variables, self.lags, _DET_ORDER[self.deterministic], self.signif, span)  # type: ignore[arg-type]

    @api("pymetrix:vecm.select_order")
    def select_order(self, data: Dataset, max_lags: int = 8, span: tuple | None = None) -> LagSelection:
        """Information criteria for lags (in differences) 0..max_lags (statsmodels ``select_order``)."""
        vecm = require("statsmodels.tsa.vector_ar.vecm")
        endog, X, XC, index = _frame(self.variables, data, span, self.exog, self.exog_coint)
        with statsmodels_adapter.ADAPTER.run(statsmodels_adapter.VECM_SELECT_ORDER) as run, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = run.call("test", vecm.select_order, endog, max_lags, deterministic=self.deterministic, seasons=self.seasons,
                           exog=X, exog_coint=XC)
        keys = [("aic", "AIC"), ("bic", "BIC"), ("hqic", "HQ"), ("fpe", "FPE")]
        frame = pd.DataFrame({label: [res.ics[k][p] for p in range(len(res.ics[k]))] for k, label in keys})
        frame.index = pd.Index(range(len(frame)), name="lags")
        best = {label.lower(): int(res.selected_orders[k]) for k, label in keys}
        notes = [f"Lags in differences; {len(index)} observations; case {CASES[self.deterministic]}"]
        notes += [f"Best by {c.upper()}: {p}" for c, p in best.items()]
        # statsmodels compares every order on the same rows: the first max_lags + 1 are presample for all of them
        prov = _backend_provenance(data, self.variables, index, max_lags + 1, span,
                                   {"type": "VECM lag selection", "settings": {**to_jsonable(self), "max_lags": max_lags}}, run.record)
        return LagSelection(Table.from_frame(frame, title=f"VECM lag selection: {', '.join(self.variables)}", notes=notes), best, prov)

    @api("pymetrix:vecm.fit")
    @records_warnings
    def fit(self, data: Dataset, span: tuple | None = None) -> FittedVECM:
        vecm = require("statsmodels.tsa.vector_ar.vecm")
        endog, X, XC, index = _frame(self.variables, data, span, self.exog, self.exog_coint)
        rank = self.rank
        johansen_result = None
        if isinstance(rank, str):
            johansen_result = self.johansen(data, span)
            rank = johansen_result.rank_trace if rank == "trace" else johansen_result.rank_maxeig
            if rank == 0:
                raise SpecificationError("the Johansen tests find no cointegration (rank 0); model the differences with a VAR instead")
            if rank == len(self.variables):
                raise SpecificationError("the Johansen tests find full rank: the variables look stationary; use a VAR in levels")
        with statsmodels_adapter.ADAPTER.run(self.adapter_spec, self) as run:
            model = run.call("model", vecm.VECM, endog, exog=X, exog_coint=XC, k_ar_diff=self.lags, coint_rank=rank,
                             deterministic=self.deterministic, seasons=self.seasons, first_season=self.first_season)
            res = run.call("fit", model.fit)
        prov = Provenance(
            data_vintages={v: data.variables[v].vintage for v in self.variables if v in data.variables},
            spec={**to_jsonable(self), "rank": rank},
            method={"type": "VECM (Johansen ML)"},
            sample=(str(index[0]), str(index[-1])),
            nobs=len(index),
            dropped=0,
            package_version=__version__,
            data_fingerprint=data_fingerprint(data, list(self.variables), index, self.lags + 1),
        )
        prov = with_backend(prov, run.record)
        # statsmodels takes the first lags + 1 rows of the sample for the lagged differences: they are presample
        taken = len(index) - int(res.nobs)
        prov = with_effective_sample(prov, data, index[taken:], span=span, variables=self.variables, presample=index[:taken])
        return FittedVECM(self, int(rank), res, index, Cholesky(), prov, johansen_result)


@dataclass(frozen=True, eq=False)
class FittedVECM(BackendResult):
    spec: VECM
    rank: int
    results: Any = field(repr=False)  # statsmodels VECMResults
    index: pd.PeriodIndex = field(repr=False)
    identification: Any = field(default_factory=Cholesky)
    provenance: Provenance | None = None
    johansen: JohansenResult | None = None

    needs_backend: ClassVar[bool] = True  # forecasts and impulse responses call statsmodels' VECMResults: kept when pickled; projects recompute it

    @property
    def backend_result(self) -> Any:
        """The library's own result object (``results``)."""
        return self.results

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self.spec.variables)

    @property
    def alpha(self) -> pd.DataFrame:
        """Adjustment (loading) coefficients: variables x relations."""
        return pd.DataFrame(self.results.alpha, index=list(self.variables), columns=self._relations())

    @property
    def beta(self) -> pd.DataFrame:
        """Cointegrating vectors, normalised on the first variables."""
        return pd.DataFrame(self.results.beta, index=list(self.variables), columns=self._relations())

    def _relations(self) -> list[str]:
        return [f"ce{i + 1}" for i in range(self.rank)]

    @property
    def sigma(self) -> pd.DataFrame:
        return pd.DataFrame(self.results.sigma_u, index=list(self.variables), columns=list(self.variables))

    @property
    def coefs(self) -> np.ndarray:
        """Coefficients of the equivalent VAR in levels, shape (lags + 1, K, K)."""
        return np.asarray(self.results.var_rep)

    @property
    def eigenvalues(self) -> np.ndarray:
        """The eigenvalues of the companion matrix of the levels VAR, largest modulus first: K - rank of them are
        exactly one (the common stochastic trends)."""
        from .var import _companion_eigenvalues

        return _companion_eigenvalues(self.coefs)

    @property
    def roots(self) -> np.ndarray:
        """Moduli of the companion-matrix eigenvalues, largest first."""
        return np.abs(self.eigenvalues)

    @api()
    def residual_correlogram(self, lags: int = 12) -> Any:
        """Auto- and cross-correlations of the residuals up to ``lags``, with ±1.96/sqrt(T) bands (see
        ``FittedVAR.residual_correlogram``)."""
        from .var import residual_correlogram

        return residual_correlogram(np.asarray(self.results.resid), self.variables, lags,
                                    f"Residual correlogram: VECM (rank {self.rank}) in {', '.join(self.variables)}")

    def roots_plot(self) -> Any:
        """The levels VAR's companion eigenvalues with the unit circle: K - rank on it, the rest inside if the
        relations are stationary."""
        from .figures import roots_figure

        K = len(self.variables)
        inside = self.roots[K - self.rank :]
        largest = float(inside.max(initial=0.0))
        note = (f"{K - self.rank} unit root(s) from {K} variables and rank {self.rank}; the largest other modulus is {largest:.4f}"
                + ("." if largest < 1 else ", on or outside the unit circle: the rank or the lags may be wrong."))
        return roots_figure(self.eigenvalues, f"VECM roots (rank {self.rank})", note)

    def relations(self) -> Any:
        """The cointegrating relations over the sample: beta' y(t), with any constant, trend or exogenous terms
        restricted to them (cases ci and coli), as they enter the error correction of the next period's changes.
        Stationary around zero if the variables cointegrate as estimated."""
        from statsmodels.tsa.vector_ar.vecm import _linear_trend

        from .data import Dataset

        r, spec = self.results, self.spec
        Y = np.asarray(r.model.endog, dtype=float)
        T, p = len(np.asarray(r.resid)), spec.lags + 1
        levels = Y[p - 1 : -1]  # y(t - 1) for each period t of the estimation sample
        terms = []
        if "ci" in spec.deterministic:
            terms.append(np.ones(T))
        if "li" in spec.deterministic:
            terms.append(_linear_trend(T, p, coint=True))
        exog_coint = getattr(r.model, "exog_coint", None)
        if exog_coint is not None:
            terms.append(np.asarray(exog_coint)[-T - 1 : -1].T)
        values = levels @ np.asarray(r.beta)
        if terms:
            values = values + np.vstack(terms).T @ np.asarray(r.det_coef_coint)
        index = self.index[p - 1 : -1]
        return Dataset.from_frame(pd.DataFrame(values, index=index, columns=self._relations()))

    def _reduced_rank_data(self, start: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.PeriodIndex, int]:
        """The differences, the lagged levels (with restricted terms) and the other regressors, as the fit's design
        (statsmodels' ``_endog_matrices``), their periods, and where the recursions start (``start``, or a third of
        the way in, but no earlier than the moments allow)."""
        from statsmodels.tsa.vector_ar.vecm import _endog_matrices

        r, spec = self.results, self.spec
        m = r.model
        _, dY, Y1, dX = _endog_matrices(np.asarray(m.endog, dtype=float).T, getattr(m, "exog", None), getattr(m, "exog_coint", None),
                                        spec.lags, spec.deterministic, spec.seasons, spec.first_season)
        K, T = dY.shape
        index = self.index[-T:]
        first = T // 3 if start is None else int(index.get_loc(pd.Period(str(start), freq=index.freq)))
        least = dX.shape[0] + Y1.shape[0] + K + 2  # the smallest sample the moments allow
        first = max(first, least - 1)
        if first >= T:
            raise SpecificationError(f"the recursions need samples of at least {least} observations; this one has {T}")
        return dY, Y1, dX, index, first

    @staticmethod
    def _partial(A: np.ndarray, X: np.ndarray) -> np.ndarray:
        """The residuals of A's rows on X's."""
        return A if X.shape[0] == 0 else A - np.linalg.lstsq(X.T, A.T, rcond=None)[0].T @ X

    @api()
    def recursive_eigenvalues(self, start: Any = None, base: Literal["Z", "R"] = "Z") -> Dataset:
        """Johansen's eigenvalues estimated on samples ending each period from ``start`` (default: a third of the way
        into the sample) to the end, as Hansen and Johansen (1999) use to check that the cointegration is stable:
        the first ``rank`` should settle and stay put.

        ``base="Z"`` re-estimates everything on each sample; ``"R"`` partials the short-run, deterministic and
        exogenous terms out once, at the full-sample estimates, and recomputes only the reduced-rank step. Columns
        ``lambda1`` ... ``lambdaK``, largest first, indexed by each sample's last period."""
        from .data import Dataset
        from .metadata import DatasetMeta

        if base not in ("Z", "R"):
            raise SpecificationError("base is 'Z' (everything re-estimated) or 'R' (short run at the full-sample estimates)")
        dY, Y1, dX, index, first = self._reduced_rank_data(start)
        K, T = dY.shape
        partial = self._partial

        def eigenvalues(R0: np.ndarray, R1: np.ndarray) -> np.ndarray:
            n = R0.shape[1]
            S00, S01, S11 = R0 @ R0.T / n, R0 @ R1.T / n, R1 @ R1.T / n
            lam = np.linalg.eigvals(np.linalg.solve(S11, S01.T @ np.linalg.solve(S00, S01)))
            return np.sort(np.real(lam))[::-1][:K]

        if base == "R":  # partialled out once, over the whole sample
            R0_all, R1_all = partial(dY, dX), partial(Y1, dX)
        rows = []
        for n in range(first + 1, T + 1):
            if base == "Z":
                rows.append(eigenvalues(partial(dY[:, :n], dX[:, :n]), partial(Y1[:, :n], dX[:, :n])))
            else:
                rows.append(eigenvalues(R0_all[:, :n], R1_all[:, :n]))
        frame = pd.DataFrame(rows, index=index[first:], columns=[f"lambda{i + 1}" for i in range(K)])
        meta = DatasetMeta(title=f"Recursive eigenvalues ({base} representation): {', '.join(self.variables)}",
                           description=f"Johansen's eigenvalues on samples from {index[0]} to each period; rank {self.rank}.")
        plain = Dataset.from_frame(frame)
        return Dataset(plain.frame, plain.axes, meta=meta)

    @api()
    def fluctuation_test(self, start: Any = None, draws: int = 199, seed: int = 0) -> TestResult:
        """A fluctuation test of the constancy of the first ``rank`` eigenvalues, from their recursive estimates with
        the short run held at the full-sample estimates (Hansen and Johansen's R representation).

        Each eigenvalue is taken as xi = ln(lambda / (1 - lambda)), and the statistic is the largest over the
        recursions of sqrt(T) (t / T) |xi(t) - xi(T)|, for the sum of the first ``rank`` xi's (and, in
        ``details["by eigenvalue"]``, each one). Its p-value is from ``draws`` bootstrap samples of the fitted VECM,
        whose parameters are constant: the residuals resampled, the series rebuilt, the recursions run again.
        ``details["paths"]`` holds each statistic's path, ``details["critical values"]`` the bootstrap 5% values."""
        from .stats import TestResult

        dY, Y1, dX, index, first = self._reduced_rank_data(start)
        observed, paths = self._fluctuations(dY, Y1, dX, first)
        sims = np.array([self._fluctuations(*self._design(Y), first)[0] for Y in self._bootstrap_levels(draws, seed)])
        r = self.rank
        names = [f"eigenvalue {i + 1}" for i in range(r)] + [f"sum of the first {r}"]
        p = (1 + (sims >= observed).sum(axis=0)) / (draws + 1)
        five = np.quantile(sims, 0.95, axis=0)
        rows = pd.DataFrame({"statistic": observed, "p-value": p, "5% critical value": five}, index=names)
        return TestResult(
            name="Fluctuation test of the eigenvalues", null="the cointegrating eigenvalues are constant over the sample",
            statistic=float(observed[-1]), df=None, p_value=float(p[-1]), distribution=f"bootstrap ({draws} draws)",
            critical_values={q: float(np.quantile(sims[:, -1], 1 - a)) for q, a in (("10%", 0.10), ("5%", 0.05), ("1%", 0.01))},
            details={"by eigenvalue": rows, "paths": pd.DataFrame(paths, index=index[first:], columns=names),
                     "critical values": pd.Series(five, index=names), "start": str(index[first])},
            provenance=self.provenance)

    def _design(self, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The fit's design matrices (differences, lagged levels with restricted terms, other regressors) for levels Y."""
        from statsmodels.tsa.vector_ar.vecm import _endog_matrices

        m, spec = self.results.model, self.spec
        _, dY, Y1, dX = _endog_matrices(Y.T, getattr(m, "exog", None), getattr(m, "exog_coint", None), spec.lags, spec.deterministic,
                                        spec.seasons, spec.first_season)
        return dY, Y1, dX

    def _fluctuations(self, dY: np.ndarray, Y1: np.ndarray, dX: np.ndarray, first: int) -> tuple[np.ndarray, np.ndarray]:
        """sqrt(T) (t / T) |xi(t) - xi(T)| for each of the first ``rank`` eigenvalues and their sum, along the
        R-representation recursions; their largest values and their paths."""
        R0, R1 = self._partial(dY, dX), self._partial(Y1, dX)
        T, r = dY.shape[1], self.rank

        def xi(n: int) -> np.ndarray:
            a, b = R0[:, :n], R1[:, :n]
            S00, S01, S11 = a @ a.T / n, a @ b.T / n, b @ b.T / n
            lam = np.sort(np.real(np.linalg.eigvals(np.linalg.solve(S11, S01.T @ np.linalg.solve(S00, S01)))))[::-1][:r]
            x = np.log(lam / (1 - lam))
            return np.r_[x, x.sum()]

        full = xi(T)
        paths = np.array([np.sqrt(T) * (n / T) * np.abs(xi(n) - full) for n in range(first + 1, T + 1)])
        return paths.max(axis=0), paths

    def _bootstrap_levels(self, draws: int, seed: int):
        """Levels rebuilt from the fitted VECM with resampled, centred residuals (as the impulse responses' bootstrap)."""
        from .progress import report

        U = np.asarray(self.results.resid)
        U = U - U.mean(axis=0)
        det = self._deterministic()
        rng = make_rng(seed)
        for k in range(draws):
            report(k / draws, f"bootstrap draw {k + 1} of {draws}")
            yield self._simulate(U[rng.integers(0, len(U), len(U))], det)

    def fluctuation_plot(self, start: Any = None, draws: int = 199, seed: int = 0) -> Any:
        """The fluctuation statistic's path for each eigenvalue and their sum, with its bootstrap 5% critical value."""
        from .figures import Figure, Line, Panel, Rule

        test = self.fluctuation_test(start, draws, seed)
        paths, five = test.details["paths"], test.details["critical values"]
        panels = tuple(Panel((Line(paths[c], c), Rule(float(five[c]), "--")), title=c, legend=False) for c in paths.columns)
        return Figure(panels, title="Fluctuation test of the eigenvalues", columns=2 if len(panels) > 3 else 1,
                      footnote=f"Dashed: the bootstrap 5% critical value; for the sum, statistic {test.statistic:.3f}, p = {test.p_value:.3f}.")

    def recursive_eigenvalues_plot(self, start: Any = None, base: Literal["Z", "R"] = "Z") -> Any:
        """The first ``rank`` recursive eigenvalues (``recursive_eigenvalues``), each in a panel."""
        from .figures import Figure, Line, Panel

        frame = self.recursive_eigenvalues(start, base).frame
        panels = tuple(Panel((Line(frame[c], c),), title=c.replace("lambda", "eigenvalue "), legend=False) for c in frame.columns[: self.rank])
        return Figure(panels, title=f"Recursive eigenvalues ({base} representation)", columns=2 if len(panels) > 3 else 1,
                      footnote=f"Samples from {self.index[-len(np.asarray(self.results.resid))]} to each period; stable cointegration keeps them steady.")

    def relations_plot(self) -> Any:
        """Each cointegrating relation over the sample, with zero."""
        from .figures import Figure, Line, Panel, Rule

        frame = self.relations().frame
        panels = tuple(Panel((Rule(0.0), Line(frame[c], c)), title=c, legend=False) for c in frame.columns)
        return Figure(panels, title=f"Cointegrating relations: {', '.join(self.variables)}", columns=2 if len(panels) > 3 else 1)

    @property
    def resid(self) -> Dataset:
        from .data import Dataset

        resid = np.asarray(self.results.resid)
        return Dataset.from_frame(pd.DataFrame(resid, index=self.index[-len(resid):], columns=list(self.variables)))

    @property
    def stats(self) -> dict[str, float]:
        return {"nobs": float(self.results.nobs), "rank": float(self.rank), "lags": float(self.spec.lags), "loglike": float(np.real(self.results.llf))}  # NumPy 2.5: eig gives complex with zero imaginary part

    def summary(self) -> Table:
        r = self.results
        rows, index = [], []
        for i, v in enumerate(self.variables):
            for j in range(self.rank):
                rows.append(["beta", f"ce{j + 1}", r.beta[i, j], r.stderr_beta[i, j] if i >= self.rank else np.nan])
                index.append(v)
        for i, v in enumerate(self.variables):
            for j in range(self.rank):
                rows.append(["alpha", f"ce{j + 1}", r.alpha[i, j], r.stderr_alpha[i, j]])
                index.append(v)
        notes = []
        if self.provenance:
            notes.append(f"Sample: {self.provenance.sample[0]} to {self.provenance.sample[1]}, {self.provenance.nobs} observations")
        notes += [f"Rank: {self.rank}", f"Lagged differences: {self.spec.lags}",
                  f"Deterministic terms: case {CASES[self.spec.deterministic]} ({self.spec.deterministic})",
                  f"Log likelihood: {float(np.real(r.llf)):.3f}"]
        notes.append("beta: cointegrating vectors, normalised (first rank rows fixed); alpha: adjustment speeds. Short-run terms: short_run().")
        return Table(["block", "relation", "coef", "std err"], rows, index=index, title=f"VECM: {', '.join(self.variables)}", notes=notes)

    def short_run(self) -> Table:
        r = self.results
        K, p = len(self.variables), self.spec.lags
        names = [f"d{v}[-{l}]" for l in range(1, p + 1) for v in self.variables]
        gamma = np.asarray(r.gamma)  # K x K*p
        se = np.asarray(r.stderr_gamma)
        rows = [[gamma[i, j], se[i, j]] for i in range(K) for j in range(K * p)]
        index = [f"d{v}: {n}" for v in self.variables for n in names]
        return Table(["coef", "std err"], rows, index=index, title="VECM short-run coefficients")

    def __repr__(self) -> str:
        return f"<FittedVECM rank {self.rank}, {self.spec.lags} lagged differences: {', '.join(self.variables)}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()

    @api()
    def identify(self, scheme: Any) -> FittedVECM:
        if not isinstance(scheme, Cholesky):
            raise SpecificationError("VECMs support Cholesky identification (long-run restrictions need a stationary VAR)")
        _impact(scheme, np.asarray(self.results.sigma_u), self.coefs, self.variables)
        return replace(self, identification=scheme)

    def impact(self) -> np.ndarray:
        return _impact(self.identification, np.asarray(self.results.sigma_u), self.coefs, self.variables)

    @api()
    def irf(self, periods: int = 20, orthogonal: bool = True, cumulative: bool = False, draws: int = 0, alpha: float = 0.05,
            seed: int = 0, **_: Any) -> IRF:
        """Impulse responses of the levels, with ``draws`` bootstrap replications for (1 - alpha) percentile bands.

        The bootstrap resamples the centred residuals, builds each series again from the estimated VECM (its
        cointegrating relations, adjustment, short-run, deterministic and exogenous terms) from the first lags + 1
        observations, and re-estimates it with the same rank, lags and deterministic case, as JMulTi and R's vars."""
        values = self._responses(self.coefs, np.asarray(self.results.sigma_u), periods, orthogonal, cumulative)
        lower = upper = None
        if draws:
            sims = np.stack([self._responses(A, S, periods, orthogonal, cumulative) for A, S in self._bootstrap(draws, seed)])
            lower, upper = np.quantile(sims, [alpha / 2, 1 - alpha / 2], axis=0)
        label = _scheme_label(self.identification) if orthogonal else "reduced-form (unit) shocks"
        if draws:
            label += f"; residual bootstrap, {draws} draws"
        return IRF(values, self.variables, self.variables, lower, upper, cumulative, label, 1 - alpha)

    def _responses(self, A: np.ndarray, sigma: np.ndarray, periods: int, orthogonal: bool, cumulative: bool) -> np.ndarray:
        phi = _ma(A, periods)
        out = phi @ _impact(self.identification, sigma, A, self.variables) if orthogonal else phi
        return np.cumsum(out, axis=0) if cumulative else out

    def _deterministic(self) -> np.ndarray:
        """Each period's deterministic and exogenous part of the differences (constants, trends, seasonal dummies,
        exogenous terms, inside and outside the relations): the differences less the residuals, the error correction
        and the short-run terms. It does not depend on the endogenous series, so it carries over to bootstrap samples."""
        r = self.results
        Y = np.asarray(r.model.endog, dtype=float)
        dY = np.diff(Y, axis=0)
        U = np.asarray(r.resid)
        T, p = len(U), self.spec.lags
        first = len(Y) - T  # the first lags + 1 observations are presample
        gamma = np.asarray(r.gamma)
        K = Y.shape[1]
        out = np.empty((T, K))
        for t in range(T):
            i = first + t  # the level observed at i; its difference is dY[i - 1]
            short = sum(gamma[:, l * K : (l + 1) * K] @ dY[i - 2 - l] for l in range(p))
            out[t] = dY[i - 1] - U[t] - r.alpha @ (r.beta.T @ Y[i - 1]) - short
        return out

    def _simulate(self, U: np.ndarray, det: np.ndarray) -> np.ndarray:
        """The levels the VECM gives with residuals ``U``, from the actual presample observations."""
        r = self.results
        Y0 = np.asarray(r.model.endog, dtype=float)
        T, p, K = len(U), self.spec.lags, Y0.shape[1]
        first = len(Y0) - T
        gamma = np.asarray(r.gamma)
        Y = Y0.copy()
        for t in range(T):
            i = first + t
            short = sum(gamma[:, l * K : (l + 1) * K] @ (Y[i - 1 - l] - Y[i - 2 - l]) for l in range(p))
            Y[i] = Y[i - 1] + r.alpha @ (r.beta.T @ Y[i - 1]) + short + det[t] + U[t]
        return Y

    def _bootstrap(self, draws: int, seed: int):
        """Residual bootstrap: (levels VAR coefficients, residual covariance) re-estimated on each resampled series."""
        from .progress import report

        vecm = require("statsmodels.tsa.vector_ar.vecm")
        r, spec = self.results, self.spec
        U = np.asarray(r.resid)
        U = U - U.mean(axis=0)
        det = self._deterministic()
        rng = make_rng(seed)
        exog, exog_coint = getattr(r.model, "exog", None), getattr(r.model, "exog_coint", None)
        for k in range(draws):
            report(k / draws, f"bootstrap draw {k + 1} of {draws}")  # the app's progress bar; Cancel stops here
            Y = self._simulate(U[rng.integers(0, len(U), len(U))], det)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = vecm.VECM(Y, exog=exog, exog_coint=exog_coint, k_ar_diff=spec.lags, coint_rank=self.rank,
                                deterministic=spec.deterministic, seasons=spec.seasons, first_season=spec.first_season).fit()
            yield np.asarray(res.var_rep), np.asarray(res.sigma_u)

    @api()
    def historical_decomposition(self) -> Any:
        """The levels over the estimation sample, split into the contributions of the identified shocks and the
        baseline: the path the VECM gives from the first lags + 1 observations with its deterministic and exogenous
        terms and no shocks. The parts add up to the data exactly. Shocks are the residuals through the impact matrix;
        their contributions go through the levels VAR, so permanent shocks keep contributing. A ``Decomposition``,
        as ``FittedVAR.historical_decomposition``."""
        from .var import _historical

        U = np.asarray(self.results.resid)
        Y = np.asarray(self.results.model.endog, dtype=float)[-len(U):]
        baseline = self._simulate(np.zeros_like(U), self._deterministic())[-len(U):]
        return _historical(self.coefs, self.impact(), U, Y, baseline, self.index[-len(U):], self.variables, self.variables,
                           _scheme_label(self.identification))

    @api()
    def fevd(self, periods: int = 20) -> FEVD:
        theta = _ma(self.coefs, periods - 1) @ self.impact()
        contributions = np.cumsum(theta**2, axis=0)
        return FEVD(contributions / contributions.sum(axis=2, keepdims=True), self.variables, self.variables, _scheme_label(self.identification))

    @api()
    def forecast(self, data: Dataset, span: tuple[Any, Any], alpha: float = 0.05, dynamic: bool = True) -> Dataset:
        """Forecasts of the levels from the end of the estimation sample (statsmodels ``predict``)."""
        from .data import Dataset
        from .estimation.design import instrument_series
        from .metadata import DatasetMeta

        if not dynamic:
            raise SpecificationError("VECM forecasts are dynamic (multi-step); use dynamic=True")

        freq = self.index.freq
        from .data.sample import as_span
        span = as_span(span)  # '1990Q1:2019Q4' or a pair
        start, end = pd.Period(str(span[0]), freq=freq), pd.Period(str(span[1]), freq=freq)
        if start != self.index[-1] + 1:
            raise SpecificationError(f"VECM forecasts start right after the estimation sample: use a span starting {self.index[-1] + 1}")
        periods = pd.period_range(start, end, freq=freq, name=data.time_axis.name)  # type: ignore[union-attr]
        ext = Dataset(data.frame.reindex(periods.union(data.frame.index)), data.axes, data.variables)

        def future(texts: Sequence[str]) -> np.ndarray | None:
            if not texts:
                return None
            cols = [s.reindex(periods).to_numpy(float) for s in instrument_series(texts, ext).values()]
            out = np.column_stack(cols)
            if not np.all(np.isfinite(out)):
                raise SpecificationError("exogenous variables are missing for some forecast periods")
            return out

        mean, lower, upper = self.results.predict(steps=len(periods), alpha=alpha, exog_fc=future(self.spec.exog), exog_coint_fc=future(self.spec.exog_coint))
        cols = {}
        for j, v in enumerate(self.variables):
            cols[v], cols[f"{v}_lower"], cols[f"{v}_upper"] = mean[:, j], lower[:, j], upper[:, j]
        meta = DatasetMeta(title="VECM forecast", description="Intervals reflect shock uncertainty only, not parameter uncertainty.")
        return Dataset(pd.DataFrame(cols, index=periods), data.axes, meta=meta)

    @api()
    def test(self, name: str, *args: Any, **opts: Any) -> TestResult:
        from .stats import run_test

        return run_test(self, name, *args, **opts)

    def tests(self) -> list[str]:
        from .stats import available_tests

        return available_tests(self)

    @api()
    def test_beta(self, H: Any) -> TestResult:
        """Johansen's LR test that the cointegrating vectors lie in the columns of ``H`` (beta = H phi): one row for each
        variable, then the restricted constant or trend if the case has one (see ``pymetrix.vecm_restrictions``)."""
        from .vecm_restrictions import test_beta

        return test_beta(self, H)

    @api()
    def test_weak_exogeneity(self, variables: Sequence[str]) -> TestResult:
        """Johansen's LR test that ``variables`` do not adjust to the cointegrating relations (their rows of alpha
        are zero), so they are weakly exogenous for beta."""
        from .vecm_restrictions import test_weak_exogeneity

        return test_weak_exogeneity(self, variables)


register("models", VECM)
