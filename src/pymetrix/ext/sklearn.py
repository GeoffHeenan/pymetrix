"""Reference plugin: shrinkage, bagging and machine-learning estimators, backed by scikit-learn.

    Equation("dgdp ~ dgdp[-1] + spread[-1] + pmi[-1] + ...").fit(data, method=Lasso())        # penalty by time-series CV
    Equation(...).fit(data, method=Ridge()); method=ElasticNet(l1_ratio=0.5); method=Bagging()
    Equation(...).fit(data, method=RandomForest()); method=GradientBoosting(); method=NeuralNetwork(); method=NearestNeighbours()

Shrinkage and bagging (Elliott and Timmermann 2016, chapters 5 and 6) return ordinary fitted equations: their
coefficients apply to the regressors as written, so forecasts (static and dynamic), backtests, combinations and
reports work as for least squares; they have no standard errors. The penalty, when not given, is chosen by
time-series cross-validation: the sample is split in ``folds`` expanding windows, each validated on the periods
after it (never on the past), and the penalty with the smallest average squared error is used. Regressors are
standardized (within each training window) so the penalty treats them alike; the intercept is not penalized.

The nonlinear learners (chapter 11: random forests and boosted regression trees, neural networks as sieves, and
nearest neighbours as local averaging) return a ``FittedLearner``: in-sample fit, the same cross-validation's error
(used as the forecast standard error), permutation importance of each regressor, and forecasts, which for
equations with lags of their own target are built recursively from their own earlier forecasts.

Bundled with pymetrix; written against the Extension API only, so it can move to its own package unchanged.
Install with ``pip install pymetrix[ml]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from .._api import api
from .._random import generator as make_rng
from .._version import __version__
from ..adapters import require
from ..adapters import sklearn as specs
from ..adapters.base import (
    AdapterSpec,
    BackendOptions,
    check_backend_options,
    with_backend,
)
from ..errors import SpecificationError
from ..forms import Widget
from ..output import Table
from ..plugins import PluginInfo, register
from ..results import BackendResult, deprecated_alias

if TYPE_CHECKING:
    from ..data import Dataset
    from ..estimation.equation import Equation
    from ..estimation.fitted import FittedEquation

PLUGIN = PluginInfo(
    name="pymetrix-sklearn",
    version=__version__,
    requires={"sklearn": ">=1.3"},
    license="BSD-3-Clause",
    maintainer="Geoffrey Heenan",
    description="Ridge, lasso, elastic net and bagging; random forests, boosted trees, neural networks and nearest neighbours",
    install="pip install pymetrix[ml]",
)

__all__ = ["Bagging", "ElasticNet", "FittedLearner", "GradientBoosting", "Lasso", "NearestNeighbours", "NeuralNetwork", "RandomForest",
           "Ridge", "time_series_folds"]

_FOLDS = Widget("number", min=2, max=20, advanced=True, help="cross-validation windows (expanding, each validated on the periods after it)")


def time_series_folds(n: int, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits of n ordered observations: (train, test) index arrays, each test block after its training
    window (scikit-learn's TimeSeriesSplit)."""
    ms = require("sklearn.model_selection")
    if folds < 2 or n < folds + 2:
        raise SpecificationError(f"{n} observations: too few for {folds} cross-validation windows")
    return [(np.asarray(a), np.asarray(b)) for a, b in ms.TimeSeriesSplit(n_splits=int(folds)).split(np.zeros(n))]


def _check(eq: Equation, method: str) -> None:
    from ..estimation.families import Gaussian

    if eq.form != "linear" or eq.fixed_effects:
        raise SpecificationError(f"{method} needs a linear-form equation without fe()")
    if not isinstance(eq.family, Gaussian) or eq.has_error_model:
        raise SpecificationError(f"{method} needs a Gaussian equation without ARMA or GARCH terms")


def _design(eq: Equation, data: Dataset, span: Any, where: Any, weights: Any, missing: str, method: str) -> Any:
    from ..estimation.design import linear_design

    _check(eq, method)
    if weights:
        raise SpecificationError(f"{method} does not take weights")
    if data.kind != "time_series":
        raise SpecificationError(f"{method} validates on later periods, so it needs time-series data")
    d = linear_design(eq, data, span, where, None, missing)
    regressors = [j for j, n in enumerate(d.names) if n != "const"]
    if not regressors:
        raise SpecificationError(f"{method} needs at least one regressor besides the constant")
    return d, regressors


# ---------------------------------------------------------------------------------------------
# shrinkage: ridge, lasso, elastic net


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mean) / sd, mean, sd


def _penalized(kind: str, alpha: float, l1_ratio: float, X: np.ndarray, y: np.ndarray, intercept: bool, run: Any) -> tuple[float, np.ndarray]:
    """Intercept and coefficients on the original scale, from a fit on standardized regressors (intercept unpenalized)."""
    lm = require("sklearn.linear_model")
    Z, mean, sd = _standardize(X)
    if not intercept:
        Z, mean = X / sd, np.zeros(X.shape[1])
    if kind == "ridge":
        model = run.call("model", lm.Ridge, alpha=alpha, fit_intercept=intercept)
    elif kind == "lasso":
        model = run.call("model", lm.Lasso, alpha=alpha, fit_intercept=intercept)  # max_iter, tol: fixed in the adapter spec
    else:
        model = run.call("model", lm.ElasticNet, alpha=alpha, l1_ratio=l1_ratio, fit_intercept=intercept)
    model.fit(Z, y)
    beta = np.asarray(model.coef_, float) / sd
    c = float(model.intercept_) - float(beta @ mean) if intercept else 0.0
    return c, beta


def _grid(kind: str, l1_ratio: float, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(y)
    Z, _, _ = _standardize(X)
    if kind == "ridge":
        return n * np.logspace(-5, 2, 36)
    top = float(np.max(np.abs(Z.T @ (y - y.mean())))) / (n * (1.0 if kind == "lasso" else max(l1_ratio, 1e-3)))
    return top * np.logspace(0, -4, 40)


def _cross_validate(kind: str, l1_ratio: float, X: np.ndarray, y: np.ndarray, intercept: bool, folds: int, run: Any) -> tuple[float, float, pd.DataFrame]:
    """The penalty with the smallest average out-of-sample squared error over expanding windows, its RMSE, and the path."""
    splits = time_series_folds(len(y), folds)
    rows = []
    for alpha in _grid(kind, l1_ratio, X, y):
        errors = []
        for train, test in splits:
            c, b = _penalized(kind, alpha, l1_ratio, X[train], y[train], intercept, run)
            errors.append(y[test] - (c + X[test] @ b))
        rows.append({"penalty": float(alpha), "cv MSE": float(np.mean(np.concatenate(errors) ** 2))})
    path = pd.DataFrame(rows)
    best = path.loc[path["cv MSE"].idxmin()]
    return float(best["penalty"]), float(np.sqrt(best["cv MSE"])), path


def _shrinkage_fit(method: Any, kind: str, eq: Equation, data: Dataset, span: Any, where: Any, weights: Any, missing: str) -> FittedEquation:
    from dataclasses import replace

    from ..estimation.design import ls_stats, provenance
    from ..estimation.fitted import FittedEquation
    from ..estimation.params import point

    d, regressors = _design(eq, data, span, where, weights, missing, method.label)
    X = d.X[:, regressors]
    l1 = float(getattr(method, "l1_ratio", 1.0))
    path = None
    with specs.ADAPTER.run(method.adapter_spec, method) as run:
        if method.alpha:
            alpha, cv_rmse = float(method.alpha), float("nan")
        else:
            alpha, cv_rmse, path = _cross_validate(kind, l1, X, d.y, eq.intercept, method.folds, run)
        c, beta = _penalized(kind, alpha, l1, X, d.y, eq.intercept, run)  # last: the options recorded are the final fit's
    coefs = np.zeros(len(d.names))
    coefs[regressors] = beta
    if eq.intercept:
        coefs[d.names.index("const")] = c
    u = d.y - d.X @ coefs
    n = len(d.y)
    active = int(np.sum(coefs[regressors] != 0)) + int(eq.intercept)
    stats = ls_stats(d.y, u, None, active, n - active, eq.intercept, True, False)
    stats.update({"penalty": alpha, "cv_rmse": cv_rmse, "nonzero": float(int(np.sum(beta != 0))), "df_resid": float(n - active)})
    prov = with_backend(provenance(eq, method, data, d.index, d.dropped), run.record, plugin=f"{PLUGIN.name} {PLUGIN.version}")
    prov = replace(prov, resolved={"penalty": alpha} if not method.alpha else None)
    chosen = "chosen by time-series cross-validation" if not method.alpha else "given"
    notes = [f"{method.label}: penalty {alpha:.6g} ({chosen}); regressors standardized, intercept not penalized. "
             f"{int(np.sum(beta != 0))} of {len(beta)} coefficients are nonzero.",
             "Shrunk coefficients have no standard errors" + (f"; cross-validated RMSE {cv_rmse:.6g}." if np.isfinite(cv_rmse) else ".")]
    extra = {"notes": notes}
    if path is not None:
        extra["cv_path"] = path
    return FittedEquation(eq, point(d.names, coefs), method, resid=pd.Series(u, index=d.index, name="resid"),
                          fitted=pd.Series(d.y - u, index=d.index, name="fitted"), stats=stats, provenance=prov,
                          design=pd.DataFrame(d.X, index=d.index, columns=d.names), extra=extra)


@dataclass(frozen=True)
class Ridge:
    """Ridge regression: least squares with a penalty on the sum of squared coefficients, shrinking them all toward zero.

    Shrinkage trades a little bias for less variance, which pays off in forecasting with many correlated or weak
    predictors. The coefficients are biased towards zero by design and have no standard errors, so read them as a
    forecasting rule, not as effects. The penalty matters most: left at 0 it is chosen by expanding-window
    cross-validation, whose path is in ``fit.extra["cv_path"]``; the notes give the penalty used. Ridge keeps every
    regressor and spreads weight over correlated ones, so it suits many predictors that each carry a little
    information.
    """

    maturity: ClassVar[str] = "beta"
    label: ClassVar[str] = "Ridge"
    kind: ClassVar[str] = "shrinkage"
    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)
    references: ClassVar[tuple[str, ...]] = (
        ("Hoerl, A. E. and Kennard, R. W. (1970). Ridge regression: biased estimation for nonorthogonal problems. Technometrics 12(1), 55-67. doi:10.1080/00401706.1970.10488634"),
    )
    requires: ClassVar[tuple[str, ...]] = ("sklearn",)
    wraps: ClassVar[str] = "sklearn.linear_model.Ridge"
    adapter_spec: ClassVar[AdapterSpec] = specs.RIDGE

    alpha: Annotated[float, Widget("number", min=0, max=1e9, help="the penalty (on standardized regressors); 0 = choose by cross-validation")] = 0.0
    folds: Annotated[int, _FOLDS] = 5

    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other scikit-learn arguments of the estimator")] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_backend_options(self)

    def estimate(self, eq: Equation, data: Dataset, *, span: Any = None, where: Any = None, weights: Any = None, missing: str = "drop") -> FittedEquation:
        return _shrinkage_fit(self, "ridge", eq, data, span, where, weights, missing)


@dataclass(frozen=True)
class Lasso:
    """The lasso: least squares with a penalty on the sum of absolute coefficients, which sets some exactly to zero (selection).

    Shrinkage trades a little bias for less variance, which pays off in forecasting with many correlated or weak
    predictors. The coefficients are biased towards zero by design and have no standard errors, so read them as a
    forecasting rule, not as effects. The penalty matters most: left at 0 it is chosen by expanding-window
    cross-validation, whose path is in ``fit.extra["cv_path"]``; the notes give the penalty used and how many
    coefficients are nonzero. Among highly correlated predictors the lasso tends to keep one and drop the rest,
    somewhat arbitrarily, so which regressors survive should not be read as evidence of which matter.
    """

    maturity: ClassVar[str] = "beta"
    label: ClassVar[str] = "Lasso"
    kind: ClassVar[str] = "shrinkage"
    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)
    references: ClassVar[tuple[str, ...]] = (
        ("Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society, Series B (Methodological) 58(1), 267-288. doi:10.1111/j.2517-6161.1996.tb02080.x"),
    )
    requires: ClassVar[tuple[str, ...]] = ("sklearn",)
    wraps: ClassVar[str] = "sklearn.linear_model.Lasso"
    adapter_spec: ClassVar[AdapterSpec] = specs.LASSO

    alpha: Annotated[float, Widget("number", min=0, max=1e9, help="the penalty (on standardized regressors); 0 = choose by cross-validation")] = 0.0
    folds: Annotated[int, _FOLDS] = 5

    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other scikit-learn arguments of the estimator")] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_backend_options(self)

    def estimate(self, eq: Equation, data: Dataset, *, span: Any = None, where: Any = None, weights: Any = None, missing: str = "drop") -> FittedEquation:
        return _shrinkage_fit(self, "lasso", eq, data, span, where, weights, missing)


@dataclass(frozen=True)
class ElasticNet:
    """The elastic net: a mix of the lasso's and ridge's penalties (``l1_ratio`` 1 is the lasso, near 0 ridge).

    Shrinkage trades a little bias for less variance, which pays off in forecasting with many correlated or weak
    predictors. The coefficients are biased towards zero by design and have no standard errors, so read them as a
    forecasting rule, not as effects. The penalty matters most: left at 0 it is chosen by expanding-window
    cross-validation, whose path is in ``fit.extra["cv_path"]``; the notes give the penalty used and how many
    coefficients are nonzero. The ridge part keeps groups of correlated predictors together where the lasso would
    pick one; only the penalty is cross-validated, so ``l1_ratio`` is yours to set.
    """

    maturity: ClassVar[str] = "beta"
    label: ClassVar[str] = "Elastic net"
    kind: ClassVar[str] = "shrinkage"
    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)
    references: ClassVar[tuple[str, ...]] = (
        ("Zou, H. and Hastie, T. (2005). Regularization and variable selection via the elastic net. Journal of the Royal Statistical Society, Series B (Statistical Methodology) 67(2), 301-320. doi:10.1111/j.1467-9868.2005.00503.x"),
    )
    requires: ClassVar[tuple[str, ...]] = ("sklearn",)
    wraps: ClassVar[str] = "sklearn.linear_model.ElasticNet"
    adapter_spec: ClassVar[AdapterSpec] = specs.ELASTIC_NET

    alpha: Annotated[float, Widget("number", min=0, max=1e9, help="the penalty (on standardized regressors); 0 = choose by cross-validation")] = 0.0
    l1_ratio: Annotated[float, Widget("number", min=0.01, max=1.0, step=0.1, help="the lasso's share of the penalty")] = 0.5
    folds: Annotated[int, _FOLDS] = 5

    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other scikit-learn arguments of the estimator")] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_backend_options(self)
        if not 0 < self.l1_ratio <= 1:
            raise SpecificationError("l1_ratio must be above 0 and at most 1")

    def estimate(self, eq: Equation, data: Dataset, *, span: Any = None, where: Any = None, weights: Any = None, missing: str = "drop") -> FittedEquation:
        return _shrinkage_fit(self, "elastic", eq, data, span, where, weights, missing)


# ---------------------------------------------------------------------------------------------
# bagging


@dataclass(frozen=True)
class Bagging:
    """Bootstrap aggregation of a pretest (Inoue and Kilian 2008), averaging the coefficients over bootstrap samples.

    In each moving-block bootstrap sample, keep the regressors whose t-statistic exceeds ``critical``, re-estimate
    with them, and average the coefficients (zero when dropped).

    A pretest alone (keep only the significant regressors) makes the forecast jump whenever a t-statistic crosses
    the critical value; averaging the decision over bootstrap samples smooths it and lowers the forecast variance,
    most usefully when many predictors are individually weak. The moving blocks keep the serial dependence of time
    series within each sample. The averaged coefficients are a forecasting rule with no standard errors; the notes
    give how often each regressor was kept.
    """

    maturity: ClassVar[str] = "beta"
    label: ClassVar[str] = "Bagging"
    kind: ClassVar[str] = "shrinkage"
    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)
    references: ClassVar[tuple[str, ...]] = (
        "Breiman, L. (1996). Bagging predictors. Machine Learning 24(2), 123-140. doi:10.1023/a:1018054314350",
        ("Inoue, A. and Kilian, L. (2008). How useful is bagging in forecasting economic time series? A case study of U.S. consumer price inflation. Journal of the American Statistical Association 103(482), 511-522. doi:10.1198/016214507000000473"),
    )
    requires: ClassVar[tuple[str, ...]] = ()

    critical: Annotated[float, Widget("number", min=0, max=10, step=0.1, help="the pretest's critical |t| (1.96: 5% two-sided)")] = 1.96
    reps: Annotated[int, Widget("number", min=10, max=10000, advanced=True, help="bootstrap samples averaged; more reduce the simulation noise in the coefficients")] = 200
    block: Annotated[int, Widget("number", min=0, max=100, advanced=True, help="block length of the moving-block bootstrap; 0 = n^(1/3)")] = 0
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="seed of the bootstrap draws: the same seed gives the same coefficients")] = 0

    def estimate(self, eq: Equation, data: Dataset, *, span: Any = None, where: Any = None, weights: Any = None, missing: str = "drop") -> FittedEquation:
        from dataclasses import replace

        from ..estimation.design import ls_stats, provenance
        from ..estimation.fitted import FittedEquation
        from ..estimation.params import point

        d, regressors = _design(eq, data, span, where, weights, missing, "Bagging")
        n, k = d.X.shape
        block = int(self.block) or max(1, int(round(n ** (1 / 3))))
        rng = make_rng(self.seed)
        coefs = np.zeros((int(self.reps), k))
        kept = np.zeros(k)
        always = [j for j in range(k) if j not in regressors]  # the constant
        for r in range(int(self.reps)):
            starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
            rows = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
            coefs[r], keep = _pretest(d.X[rows], d.y[rows], always, self.critical)
            kept += keep
        beta = coefs.mean(axis=0)
        full, full_keep = _pretest(d.X, d.y, always, self.critical)
        u = d.y - d.X @ beta
        stats = ls_stats(d.y, u, None, k, n - k, eq.intercept, True, False)
        stats.update({"df_resid": float(n - k), "reps": float(self.reps)})
        prov = provenance(eq, self, data, d.index, d.dropped)
        prov = replace(prov, seed=self.seed)
        share = ", ".join(f"{d.names[j]} {kept[j] / self.reps:.0%}" for j in regressors)
        notes = [f"Bagged pretest: {self.reps} moving-block bootstrap samples (block {block}); |t| > {self.critical:g} keeps a regressor.",
                 f"Kept in: {share}.",
                 "Pretest on the full sample keeps: " + (", ".join(d.names[j] for j in regressors if full_keep[j]) or "none") + "."]
        return FittedEquation(eq, point(d.names, beta), self, resid=pd.Series(u, index=d.index, name="resid"),
                              fitted=pd.Series(d.y - u, index=d.index, name="fitted"), stats=stats, provenance=prov,
                              design=pd.DataFrame(d.X, index=d.index, columns=d.names),
                              extra={"notes": notes, "inclusion": {d.names[j]: kept[j] / self.reps for j in regressors}})


def _pretest(X: np.ndarray, y: np.ndarray, always: list[int], critical: float) -> tuple[np.ndarray, np.ndarray]:
    """OLS, keep the regressors with |t| > critical (and ``always``), OLS again on those: coefficients (0 if dropped), and which were kept."""
    n, k = X.shape
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ b
    s2 = float(u @ u) / max(n - k, 1)
    se = np.sqrt(np.maximum(np.diag(s2 * np.linalg.pinv(X.T @ X)), 0))
    t = np.divide(b, se, out=np.zeros_like(b), where=se > 0)
    keep = np.abs(t) > critical
    keep[always] = True
    out = np.zeros(k)
    if keep.any():
        out[keep] = np.linalg.lstsq(X[:, keep], y, rcond=None)[0]
    return out, keep


# ---------------------------------------------------------------------------------------------
# nonlinear learners


@dataclass(frozen=True, eq=False)
class FittedLearner(BackendResult):
    """A machine-learning regression of an equation's target on its regressors: fit, cross-validated error, importance
    of each regressor, and forecasts."""

    equation: Any
    method: Any
    model: Any = field(repr=False)
    names: list[str] = field(default_factory=list)  # the regressors, as in the equation
    y: pd.Series = field(repr=False, default_factory=pd.Series)
    X: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    cv_rmse: float = float("nan")
    cv_resid: pd.Series = field(repr=False, default_factory=pd.Series)  # out-of-sample errors of the cross-validation windows
    provenance: Any = None

    needs_backend: ClassVar[bool] = True  # predictions call the fitted scikit-learn estimator: kept when pickled; projects recompute it

    @property
    def backend_result(self) -> Any:
        """The library's own result object (``model``)."""
        return self.model

    @property
    def dependent(self) -> str:
        return self.equation.require_dependent()

    target = deprecated_alias("dependent", "target", why="the dependent variable; target is its old name")

    @property
    def fitted(self) -> pd.Series:
        return pd.Series(self.model.predict(self.X.to_numpy(float)), index=self.y.index, name="fitted")

    @property
    def resid(self) -> pd.Series:
        return (self.y - self.fitted).rename("resid")

    @property
    def stats(self) -> dict[str, float]:
        u = self.resid.to_numpy()
        y = self.y.to_numpy()
        cv = self.cv_resid.to_numpy()
        test_y = self.y.loc[self.cv_resid.index].to_numpy()
        return {"nobs": float(len(y)), "rmse": float(np.sqrt(np.mean(u**2))), "r2": float(1 - u @ u / np.sum((y - y.mean()) ** 2)),
                "cv_rmse": self.cv_rmse, "cv_r2": float(1 - cv @ cv / np.sum((test_y - test_y.mean()) ** 2)) if len(cv) else float("nan")}

    def summary(self) -> Table:
        s = self.stats
        rows = [["observations", int(s["nobs"])], ["in-sample RMSE", s["rmse"]], ["in-sample R-squared", s["r2"]],
                ["cross-validated RMSE", s["cv_rmse"]], ["cross-validated R-squared", s["cv_r2"]]]
        sample = f"{self.y.index[0]} to {self.y.index[-1]}"
        return Table(["", "value"], rows, title=f"{self.equation.text}: {self.method.label}",
                     notes=(f"Regressors: {', '.join(self.names)}. Sample {sample}.",
                            f"Cross-validation: {self.method.folds} expanding windows, each validated on the periods after it. "
                            "In-sample fit flatters flexible learners: judge them by the cross-validated error.",
                            "Forecast standard errors are the cross-validated RMSE (one step ahead)."))

    def importance(self, reps: int = 10, seed: int = 0) -> Table:
        """Permutation importance: how much the in-sample squared error rises when each regressor is shuffled."""
        inspection = require("sklearn.inspection")
        result = inspection.permutation_importance(self.model, self.X.to_numpy(float), self.y.to_numpy(float), n_repeats=int(reps),
                                                   random_state=seed, scoring="neg_mean_squared_error")
        order = np.argsort(-result.importances_mean)
        rows = [[float(result.importances_mean[j]), float(result.importances_std[j])] for j in order]
        return Table(["rise in MSE", "sd"], rows, index=[self.names[j] for j in order], index_name="regressor",
                     title="Permutation importance", notes=(f"{reps} shuffles of each regressor; larger means the learner relies on it more.",))

    def _features(self, data: Dataset) -> pd.DataFrame:
        from ..estimation.design import as_series
        from ..expr import evaluate

        index = data.frame.index
        return pd.DataFrame({t.label: as_series(evaluate(t.expr, data.lagged), index) for t in self.equation.terms}, index=index)[self.names]

    def _plain(self) -> bool:
        """The left-hand side is the target series itself (not a transformation such as dlog(c))."""
        return type(self.equation.lhs).__name__ == "Name"

    def _lhs(self, data: Dataset) -> pd.Series:
        from ..estimation.design import as_series
        from ..expr import evaluate

        return as_series(evaluate(self.equation.lhs, data.lagged), data.frame.index)

    def predict(self, data: Dataset, span: Any = None, dynamic: bool = False) -> pd.Series:
        """Predictions over ``span``. Static: of the left-hand side, from the regressors as observed. Dynamic: of the
        target series, period by period, with later periods' regressors and left-hand side using the earlier
        predictions (a transformed left-hand side such as dlog(c) is solved for c)."""
        mask = np.asarray(data.sample_mask(span))
        periods = data.frame.index[mask]
        if not dynamic:
            X = self._features(data).loc[periods]
            ok = X.notna().all(axis=1)
            out = pd.Series(np.nan, index=periods, name="forecast")
            if ok.any():
                out[ok] = self.model.predict(X[ok].to_numpy(float))
            return out
        return self._dynamic(data, periods)[0]

    def _dynamic(self, data: Dataset, periods: pd.Index) -> tuple[pd.Series, pd.Series]:
        """Dynamic predictions of the target, and d(target)/d(left-hand side) at each (to carry a standard error over)."""
        from scipy.optimize import newton

        from ..data import Dataset as _Dataset

        frame = data.frame.copy()
        target = self.dependent
        out = pd.Series(np.nan, index=periods, name="forecast")
        slope = pd.Series(1.0, index=periods)
        for period in periods:
            row = self._features(_Dataset(frame, data.axes, data.variables)).loc[[period]]
            if row.isna().any(axis=None):
                raise SpecificationError(f"a regressor is missing for {period}")
            value = float(self.model.predict(row.to_numpy(float))[0])
            if self._plain():
                frame.loc[period, target] = value
            else:
                def gap(x: float) -> float:
                    frame.loc[period, target] = x
                    return float(self._lhs(_Dataset(frame, data.axes, data.variables)).loc[period]) - value

                previous = frame[target].loc[:period].iloc[:-1].dropna()
                start = float(previous.iloc[-1]) if len(previous) else 1.0
                solved = float(newton(gap, start, tol=1e-12, maxiter=100))
                h = 1e-6 * max(abs(solved), 1.0)
                derivative = (gap(solved + h) - gap(solved - h)) / (2 * h)
                frame.loc[period, target] = solved
                slope[period] = 1 / derivative if derivative else np.nan
            out[period] = float(frame.loc[period, target])
        return out, slope

    @api()
    def forecast(self, data: Dataset, span: tuple[Any, Any], dynamic: bool = False, alpha: float = 0.05) -> Any:
        """Forecasts over ``span`` with a (1 - alpha) interval from the cross-validated RMSE. Static forecasts are of the
        left-hand side; dynamic ones of the target series (as for equations). Dynamic forecasts that feed on themselves
        (lags of the target among the regressors, or in a transformed left-hand side) have the interval one step ahead
        only: later steps compound errors, and their standard error is left missing."""
        from ..data import Dataset as _Dataset

        if dynamic:
            mask = np.asarray(data.sample_mask(span))
            point, slope = self._dynamic(data, data.frame.index[mask])
            se = (self.cv_rmse * slope.abs()).rename("se")
            target = self.dependent
            feeds = (not self._plain()) or any(target in t.expr.variables() for t in self.equation.terms)
            if feeds:
                se.iloc[1:] = np.nan
            actual = data[target].reindex(point.index) if target in data.names else pd.Series(np.nan, index=point.index)
        else:
            point = self.predict(data, span)
            se = pd.Series(self.cv_rmse, index=point.index)
            actual = self._lhs(data).reindex(point.index)
        z = sps.norm.ppf(1 - alpha / 2)
        frame = pd.DataFrame({"forecast": point, "se": se, "lower": point - z * se, "upper": point + z * se,
                              "actual": actual.to_numpy(float)})
        return _Dataset(frame, data.axes)

    def plot(self) -> Any:
        from ..figures import Figure, Line, Panel

        return Figure((Panel((Line(self.y, self.dependent, color="#222222"), Line(self.fitted, "fitted", style="--")),
                             title=f"{self.equation.text}: {self.method.label}"),))

    def __repr__(self) -> str:
        return f"<FittedLearner {self.method.label}: {self.equation.text}, {len(self.y)} observations, CV RMSE {self.cv_rmse:.4g}>"


def _learn(method: Any, eq: Equation, data: Dataset, span: Any, where: Any, weights: Any, missing: str) -> FittedLearner:
    from dataclasses import replace

    from ..estimation.design import provenance

    d, regressors = _design(eq, data, span, where, weights, missing, method.label)
    names = [d.names[j] for j in regressors]
    if any(n not in [t.label for t in eq.terms] for n in names):
        raise SpecificationError(f"{method.label} takes numeric regressors (no categorical terms)")
    X, y = d.X[:, regressors], d.y
    errors, tested = [], []
    with specs.ADAPTER.run(method.adapter_spec, method) as run:
        for train, test in time_series_folds(len(y), method.folds):
            model = method._model(run)
            model.fit(X[train], y[train])
            errors.append(y[test] - model.predict(X[test]))
            tested.append(test)
        cv_resid = pd.Series(np.concatenate(errors), index=d.index[np.concatenate(tested)], name="cv error")
        model = method._model(run)
        model.fit(X, y)
    prov = with_backend(provenance(eq, method, data, d.index, d.dropped), run.record, plugin=f"{PLUGIN.name} {PLUGIN.version}")
    prov = replace(prov, seed=getattr(method, "seed", None))
    return FittedLearner(eq, method, model, names, pd.Series(y, index=d.index, name=eq.require_dependent()),
                         pd.DataFrame(X, index=d.index, columns=names), float(np.sqrt(np.mean(cv_resid.to_numpy() ** 2))), cv_resid, prov)


def _changed(method: Any, settings: dict[str, tuple[str, Any]]) -> dict[str, Any]:
    """The library arguments of the settings that differ from the library's defaults: field -> (argument, default).
    Defaults stay out of the call, so the call and its record are as before for fits that leave them alone."""
    return {arg: getattr(method, name) for name, (arg, default) in settings.items() if getattr(method, name) != default}


def _scaled(model: Any) -> Any:
    pipeline, preprocessing = require("sklearn.pipeline"), require("sklearn.preprocessing")
    return pipeline.make_pipeline(preprocessing.StandardScaler(), model)


@dataclass(frozen=True)
class RandomForest:
    """A random forest: the average of many regression trees, each grown on a bootstrap sample with random subsets of regressors.

    Averaging decorrelated trees captures nonlinearities and interactions with little tuning: more trees only reduce
    noise, while larger leaves and fewer regressors per split smooth the fit. Trees cannot extrapolate: forecasts
    stay within the range of outcomes seen in the sample, which matters for trending series. Flexible learners can
    fit the sample almost perfectly, so judge them by the cross-validated RMSE and R-squared in the summary
    (expanding windows, each tested on later periods), not the in-sample fit; the cross-validated RMSE is also the
    forecast standard error. The settings are not tuned automatically: compare the cross-validated error across
    them. There are no coefficients; ``fit.importance()`` shows how much the fit relies on each regressor, not a
    causal effect.
    """

    maturity: ClassVar[str] = "beta"
    label: ClassVar[str] = "Random forest"
    kind: ClassVar[str] = "learner"
    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)
    references: ClassVar[tuple[str, ...]] = (
        "Breiman, L. (2001). Random forests. Machine Learning 45(1), 5-32. doi:10.1023/a:1010933404324",
    )
    requires: ClassVar[tuple[str, ...]] = ("sklearn",)
    wraps: ClassVar[str] = "sklearn.ensemble.RandomForestRegressor"
    adapter_spec: ClassVar[AdapterSpec] = specs.RANDOM_FOREST

    trees: Annotated[int, Widget("number", min=10, max=5000, help="the number of trees")] = 500
    min_leaf: Annotated[int, Widget("number", min=1, max=100, help="the fewest observations in a leaf (larger: smoother)")] = 5
    max_features: Annotated[float, Widget("number", min=0.1, max=1.0, step=0.1, advanced=True, help="the share of regressors tried at each split")] = 0.5
    folds: Annotated[int, _FOLDS] = 5
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="seed of the random draws: the same seed gives the same fit")] = 0
    max_depth: Annotated[int | None, Widget("number", min=1, max=100, advanced=True, help="the deepest a tree grows; empty = no limit")] = None
    min_split: Annotated[int, Widget("number", min=2, max=200, advanced=True, help="the fewest observations a node needs to be split")] = 2
    criterion: Annotated[Literal["squared_error", "absolute_error", "friedman_mse", "poisson"], Widget(
        "choice", advanced=True, help="the split criterion: squared error, absolute error, Friedman's MSE, Poisson deviance")] = "squared_error"
    bootstrap: Annotated[bool, Widget("checkbox", advanced=True, help="grow each tree on a bootstrap sample (unticked: on all observations)")] = True
    max_samples: Annotated[float | None, Widget("number", min=0.05, max=1.0, step=0.05, advanced=True,
                                                help="the share of observations in each bootstrap sample; empty = all")] = None
    pruning: Annotated[float, Widget("number", min=0, max=1e6, step=0.001, advanced=True,
                                     help="cost-complexity pruning (ccp_alpha); 0 = none")] = 0.0

    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other scikit-learn arguments of the estimator")] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_backend_options(self)

    def _model(self, run: Any) -> Any:
        more = _changed(self, {"max_depth": ("max_depth", None), "min_split": ("min_samples_split", 2),
                               "criterion": ("criterion", "squared_error"), "bootstrap": ("bootstrap", True),
                               "max_samples": ("max_samples", None), "pruning": ("ccp_alpha", 0.0)})
        return run.call("model", require("sklearn.ensemble").RandomForestRegressor, n_estimators=int(self.trees),
                        min_samples_leaf=int(self.min_leaf), max_features=float(self.max_features), random_state=int(self.seed), **more)

    def estimate(self, eq: Equation, data: Dataset, *, span: Any = None, where: Any = None, weights: Any = None, missing: str = "drop") -> FittedLearner:
        return _learn(self, eq, data, span, where, weights, missing)


@dataclass(frozen=True)
class GradientBoosting:
    """Boosted regression trees: small trees fitted one after another to the errors left by the earlier ones, each scaled by the learning rate.

    Unlike a forest, boosting can overfit as rounds are added: the number of trees and the learning rate work together
    (a smaller rate needs more trees), and early stopping can choose the rounds. Depth sets the order of
    interactions. Like any tree method it extrapolates poorly: forecasts flatten outside the range of the sample.
    Flexible learners can fit the sample almost perfectly, so judge them by the cross-validated RMSE and R-squared
    in the summary (expanding windows, each tested on later periods), not the in-sample fit; the cross-validated
    RMSE is also the forecast standard error. The settings are not tuned automatically: compare the cross-validated
    error across them. There are no coefficients; ``fit.importance()`` shows how much the fit relies on each
    regressor, not a causal effect.
    """

    maturity: ClassVar[str] = "beta"
    label: ClassVar[str] = "Boosted trees"
    kind: ClassVar[str] = "learner"
    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)
    references: ClassVar[tuple[str, ...]] = (
        ("Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics 29(5), 1189-1232. doi:10.1214/aos/1013203451"),
    )
    requires: ClassVar[tuple[str, ...]] = ("sklearn",)
    wraps: ClassVar[str] = "sklearn.ensemble.GradientBoostingRegressor"
    adapter_spec: ClassVar[AdapterSpec] = specs.GRADIENT_BOOSTING

    trees: Annotated[int, Widget("number", min=10, max=5000, help="the number of boosting rounds")] = 300
    learning_rate: Annotated[float, Widget("number", min=0.001, max=1.0, step=0.01, help="the shrinkage of each tree (smaller: slower, smoother)")] = 0.05
    depth: Annotated[int, Widget("number", min=1, max=10, help="the depth of each tree (1: additive in the regressors)")] = 2
    subsample: Annotated[float, Widget("number", min=0.1, max=1.0, step=0.1, advanced=True, help="the share of observations each tree sees")] = 0.8
    folds: Annotated[int, _FOLDS] = 5
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="seed of the random draws: the same seed gives the same fit")] = 0
    loss: Annotated[Literal["squared_error", "absolute_error", "huber", "quantile"], Widget(
        "choice", help="the loss each tree reduces: squared error, absolute error, Huber, or a quantile's check loss")] = "squared_error"
    quantile: Annotated[float, Widget("number", min=0.01, max=0.99, step=0.05, advanced=True,
                                      help="the quantile (loss quantile), or Huber's quantile of the errors (loss huber)")] = 0.9
    min_leaf: Annotated[int, Widget("number", min=1, max=100, advanced=True, help="the fewest observations in a leaf")] = 1
    min_split: Annotated[int, Widget("number", min=2, max=200, advanced=True, help="the fewest observations a node needs to be split")] = 2
    max_features: Annotated[float | None, Widget("number", min=0.1, max=1.0, step=0.1, advanced=True,
                                                 help="the share of regressors tried at each split; empty = all")] = None
    stop_after: Annotated[int | None, Widget("number", min=1, max=1000, advanced=True,
                                             help="stop when this many rounds do not improve the held-out score; empty = no early stopping")] = None
    held_out: Annotated[float, Widget("number", min=0.05, max=0.5, step=0.05, advanced=True,
                                      help="the share held out to decide early stopping")] = 0.1
    pruning: Annotated[float, Widget("number", min=0, max=1e6, step=0.001, advanced=True,
                                     help="cost-complexity pruning (ccp_alpha); 0 = none")] = 0.0

    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other scikit-learn arguments of the estimator")] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_backend_options(self)

    def _model(self, run: Any) -> Any:
        more = _changed(self, {"loss": ("loss", "squared_error"), "min_leaf": ("min_samples_leaf", 1), "min_split": ("min_samples_split", 2),
                               "max_features": ("max_features", None), "stop_after": ("n_iter_no_change", None),
                               "held_out": ("validation_fraction", 0.1), "pruning": ("ccp_alpha", 0.0)})
        if self.loss in ("huber", "quantile") and self.quantile != 0.9:
            more["alpha"] = float(self.quantile)
        return run.call("model", require("sklearn.ensemble").GradientBoostingRegressor, n_estimators=int(self.trees),
                        learning_rate=float(self.learning_rate), max_depth=int(self.depth), subsample=float(self.subsample),
                        random_state=int(self.seed), **more)

    def estimate(self, eq: Equation, data: Dataset, *, span: Any = None, where: Any = None, weights: Any = None, missing: str = "drop") -> FittedLearner:
        return _learn(self, eq, data, span, where, weights, missing)


@dataclass(frozen=True)
class NeuralNetwork:
    """A feed-forward neural network (one or more hidden layers): a flexible sieve approximation, on standardized regressors.

    With the few hundred observations of most macroeconomic series a network overfits easily: keep the hidden layers
    small and use the penalty. The fit depends on the random starting weights, so try other seeds and check that the
    cross-validated error is stable. Flexible learners can fit the sample almost perfectly, so judge them by the
    cross-validated RMSE and R-squared in the summary (expanding windows, each tested on later periods), not the
    in-sample fit; the cross-validated RMSE is also the forecast standard error. The settings are not tuned
    automatically: compare the cross-validated error across them. There are no coefficients; ``fit.importance()``
    shows how much the fit relies on each regressor, not a causal effect.
    """

    maturity: ClassVar[str] = "beta"
    label: ClassVar[str] = "Neural network"
    kind: ClassVar[str] = "learner"
    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)
    references: ClassVar[tuple[str, ...]] = (
        ("Hornik, K., Stinchcombe, M. and White, H. (1989). Multilayer feedforward networks are universal approximators. Neural Networks 2(5), 359-366. doi:10.1016/0893-6080(89)90020-8"),
        ("Hastie, T., Tibshirani, R. and Friedman, J. (2009). The Elements of Statistical Learning, 2nd ed., chapter 11, \"Neural networks\". Springer. doi:10.1007/978-0-387-84858-7"),
    )
    requires: ClassVar[tuple[str, ...]] = ("sklearn",)
    wraps: ClassVar[str] = "sklearn.neural_network.MLPRegressor"
    adapter_spec: ClassVar[AdapterSpec] = specs.NEURAL_NETWORK

    hidden: Annotated[tuple[int, ...], Widget("text", help="units in each hidden layer, e.g. 8 or 16, 8")] = (8,)
    penalty: Annotated[float, Widget("number", min=0, max=100, step=0.001, help="weight decay (L2 penalty on the weights)")] = 0.001
    folds: Annotated[int, _FOLDS] = 5
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="seed of the random draws: the same seed gives the same fit")] = 0
    activation: Annotated[Literal["relu", "tanh", "logistic", "identity"], Widget(
        "choice", help="the hidden units' activation: rectified linear, tanh, logistic, or none (identity)")] = "relu"
    solver: Annotated[Literal["adam", "lbfgs", "sgd"], Widget(
        "choice", advanced=True, help="the optimiser: Adam, L-BFGS (often best on small samples), stochastic gradient descent")] = "adam"
    learning_rate: Annotated[float, Widget("number", min=1e-6, max=1.0, step=0.0005, advanced=True,
                                           help="the initial learning rate (adam, sgd)")] = 0.001
    batch: Annotated[int | None, Widget("number", min=1, max=100_000, advanced=True,
                                        help="observations in each minibatch (adam, sgd); empty = automatic")] = None
    max_iter: Annotated[int, Widget("number", min=10, max=1_000_000, advanced=True, help="the most passes (epochs) or iterations")] = 5000
    early_stopping: Annotated[bool, Widget("checkbox", advanced=True,
                                           help="stop when a held-out 10% stops improving (adam, sgd)")] = False

    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other scikit-learn arguments of the estimator")] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_backend_options(self)
        object.__setattr__(self, "hidden", tuple(int(h) for h in (self.hidden if not isinstance(self.hidden, int) else (self.hidden,))))
        if not self.hidden or min(self.hidden) < 1:
            raise SpecificationError("give at least one hidden layer with at least one unit")

    def _model(self, run: Any) -> Any:
        nn, compose = require("sklearn.neural_network"), require("sklearn.compose")
        more = _changed(self, {"activation": ("activation", "relu"), "solver": ("solver", "adam"),
                               "learning_rate": ("learning_rate_init", 0.001), "batch": ("batch_size", None),
                               "max_iter": ("max_iter", 5000), "early_stopping": ("early_stopping", False)})
        mlp = run.call("model", nn.MLPRegressor, hidden_layer_sizes=self.hidden, alpha=float(self.penalty), random_state=int(self.seed), **more)
        return compose.TransformedTargetRegressor(regressor=_scaled(mlp), transformer=require("sklearn.preprocessing").StandardScaler())

    def estimate(self, eq: Equation, data: Dataset, *, span: Any = None, where: Any = None, weights: Any = None, missing: str = "drop") -> FittedLearner:
        return _learn(self, eq, data, span, where, weights, missing)


@dataclass(frozen=True)
class NearestNeighbours:
    """Nearest neighbours: the average of the outcomes in the k most similar periods (a local, kernel-like regression).

    k sets the bias-variance trade-off: few neighbours follow the data closely and are noisy, many are smooth but
    blur local features. With many regressors no period is truly near another (the curse of dimensionality), so keep
    the regressors few. Forecasts are averages of past outcomes and cannot extrapolate. Flexible learners can fit
    the sample almost perfectly, so judge them by the cross-validated RMSE and R-squared in the summary (expanding
    windows, each tested on later periods), not the in-sample fit; the cross-validated RMSE is also the forecast
    standard error. The settings are not tuned automatically: compare the cross-validated error across them. There
    are no coefficients; ``fit.importance()`` shows how much the fit relies on each regressor, not a causal effect.
    """

    maturity: ClassVar[str] = "beta"
    label: ClassVar[str] = "Nearest neighbours"
    kind: ClassVar[str] = "learner"
    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)
    references: ClassVar[tuple[str, ...]] = (
        ("Hastie, T., Tibshirani, R. and Friedman, J. (2009). The Elements of Statistical Learning, 2nd ed., chapter 13, \"Prototype methods and nearest-neighbors\". Springer. doi:10.1007/978-0-387-84858-7"),
    )
    requires: ClassVar[tuple[str, ...]] = ("sklearn",)
    wraps: ClassVar[str] = "sklearn.neighbors.KNeighborsRegressor"
    adapter_spec: ClassVar[AdapterSpec] = specs.NEAREST_NEIGHBOURS

    k: Annotated[int, Widget("number", min=1, max=200, help="the number of neighbours averaged")] = 10
    distance_weighted: Annotated[bool, Widget("checkbox", help="weight closer neighbours more")] = False
    folds: Annotated[int, _FOLDS] = 5
    power: Annotated[float, Widget("number", min=1, max=10, step=1, advanced=True,
                                   help="the Minkowski distance's power: 1 city-block, 2 straight-line (Euclidean)")] = 2.0

    backend_options: Annotated[BackendOptions, Widget("", advanced=True, help="other scikit-learn arguments of the estimator")] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_backend_options(self)

    def _model(self, run: Any) -> Any:
        knn = run.call("model", require("sklearn.neighbors").KNeighborsRegressor, n_neighbors=int(self.k),
                       weights="distance" if self.distance_weighted else "uniform", **_changed(self, {"power": ("p", 2.0)}))
        return _scaled(knn)

    def estimate(self, eq: Equation, data: Dataset, *, span: Any = None, where: Any = None, weights: Any = None, missing: str = "drop") -> FittedLearner:
        return _learn(self, eq, data, span, where, weights, missing)


for _cls in (Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, NeuralNetwork, NearestNeighbours):
    register("methods", _cls, plugin=PLUGIN)
register("methods", Bagging)  # needs only NumPy
