"""SHAP values of a machine-learning fit: how much each regressor moves each prediction away from the average one.

    sv = shap_values(fit)                       # the in-sample periods; fit.shap() is the same
    sv.summary()                                # mean |SHAP| by regressor
    sv.values                                   # one row per period, one column per regressor
    shap_values(fit, data, ("2024Q1", "2025Q4")) # periods outside the sample

A prediction is split into a base value (the average prediction) plus one contribution per regressor, the Shapley
values of the cooperative game whose players are the regressors (Lundberg and Lee 2017): the contributions add up
to the prediction exactly, and a regressor the model ignores gets zero. Tree models (random forests, boosted trees,
LightGBM, XGBoost) use TreeSHAP (Lundberg et al. 2020), which is exact and needs no sampling: the base value is the
average over the training sample, with absent regressors handled by following the tree's own split proportions
("path-dependent"). Other learners (neural networks, nearest neighbours) are explained from the model's predictions
alone: an absent regressor takes the values of a background sample of the training periods ("interventional"),
with every subset of regressors evaluated (exact) when there are at most ten, and otherwise a seeded sample of
orderings (the permutation explainer). SHAP values describe the fitted model, not causal effects: correlated
regressors share credit in ways that depend on how absence is defined.

The work is done by the ``shap`` package (``pip install pymetrix[shap]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from ._api import api
from ._random import generator as make_rng
from .adapters import require
from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["ShapValues", "shap_values"]

TREE_METHODS = ("RandomForest", "GradientBoosting", "LightGBM", "XGBoost")  # TreeSHAP reads their trees
EXACT_UP_TO = 10  # regressors: 2^10 = 1024 subsets for each background row


@dataclass(frozen=True, eq=False)
class ShapValues:
    """SHAP values of a learner's predictions: one contribution per regressor and period, and the base value they add to."""

    values: pd.DataFrame  # periods by regressors
    base: float  # the base (expected) value: the prediction with no regressor known
    features: pd.DataFrame = field(repr=False)  # the regressors' values in the periods explained
    prediction: pd.Series = field(repr=False)  # the model's prediction in those periods
    explainer: str = "tree"  # "tree", "exact" or "permutation"
    title: str = ""
    background: pd.Index | None = field(default=None, repr=False)  # the training periods absent regressors are drawn from
    evaluations: int | None = None  # the permutation explainer's model evaluations per period
    seed: int | None = None  # the seed of the background sample and the orderings
    library: str = ""  # "shap 0.48.0"

    stage = "estimate"

    @property
    def additivity_error(self) -> float:
        """The largest gap between base + contributions and the prediction (rounding only, by construction)."""
        return float(np.max(np.abs(self.values.sum(axis=1).to_numpy() + self.base - self.prediction.to_numpy()))) if len(self.values) else 0.0

    @property
    def mean_abs(self) -> pd.Series:
        """Mean |SHAP| by regressor, largest first: the average size of each regressor's contribution."""
        return self.values.abs().mean().sort_values(ascending=False).rename("mean |SHAP|")

    def _how(self) -> str:
        if self.explainer == "tree":
            return "TreeSHAP (exact, path-dependent: absent regressors follow the trees' split proportions)"
        size = f"{len(self.background)} background periods" if self.background is not None else "a background sample"
        if self.explainer == "exact":
            return f"exact Shapley values over every subset of regressors, absent ones drawn from {size}"
        return (f"the permutation explainer ({self.evaluations} model evaluations per period, seed {self.seed}), "
                f"absent regressors drawn from {size}")

    def summary(self) -> Table:
        order = self.mean_abs.index
        rows = [[float(self.values[c].abs().mean()), float(self.values[c].mean()), float(self.values[c].std(ddof=0))] for c in order]
        n = len(self.values)
        span = f"{self.values.index[0]} to {self.values.index[-1]}" if n else "no periods"
        return Table(["mean |SHAP|", "mean SHAP", "sd"], rows, index=[str(c) for c in order], index_name="regressor",
                     title=f"SHAP values: {self.title}" if self.title else "SHAP values",
                     notes=(f"{n} periods ({span}); base value {self.base:.6g}: each prediction is the base value plus the "
                            "regressors' contributions.",
                            f"Computed by {self._how()}.",
                            "Larger mean |SHAP| means the model's predictions move more with that regressor; it is not a causal effect."))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        """The contributions with the base value and the prediction, one row per period."""
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.values.assign(base=self.base, prediction=self.prediction)

    def plot(self) -> Any:
        from .figures import Bars, Figure, Panel, contribution_layers

        bars = self.mean_abs.iloc[::-1]
        bars = pd.Series(bars.to_numpy(), index=[str(i) for i in bars.index])
        panels = [Panel((Bars(bars, "mean |SHAP|", horizontal=True),), title="Mean |SHAP| by regressor", legend=False)]
        if isinstance(self.values.index, pd.PeriodIndex) and len(self.values) > 1:
            panels.append(Panel(contribution_layers(self.values, self.prediction - self.base, "prediction - base"),
                                title="Contributions by period"))
        return Figure(tuple(panels), title=f"SHAP values: {self.title}" if self.title else "SHAP values")

    def __repr__(self) -> str:
        return f"<ShapValues {self.explainer}: {self.values.shape[0]} periods, {self.values.shape[1]} regressors, base {self.base:.4g}>"


def _rows(fit: Any, data: Dataset | None, span: Any) -> pd.DataFrame:
    """The regressors in the periods to explain: the fit's sample, or ``span`` of ``data`` (rows with a regressor missing left out)."""
    if data is None:
        X = fit.X
        if span is not None:
            start, end = span
            X = X.loc[start:end]
        return X
    mask = np.asarray(data.sample_mask(span))
    X = fit._features(data).loc[data.frame.index[mask]]
    return X[X.notna().all(axis=1)]


@api("pymetrix:shap_values")
def shap_values(fit: Any, data: Dataset | None = None, span: tuple[Any, Any] | None = None,
                explainer: Literal["auto", "tree", "exact", "permutation"] = "auto", background: int = 100, evaluations: int = 0,
                seed: int = 0) -> ShapValues:
    """SHAP values of a machine-learning fit (``FittedLearner``) over its sample, or over ``span`` of ``data``.

    ``explainer``: ``"tree"`` (TreeSHAP, exact, for random forests, boosted trees, LightGBM and XGBoost), ``"exact"``
    (every subset of regressors), ``"permutation"`` (a seeded sample of orderings), or ``"auto"``: tree for tree models,
    else exact up to ten regressors and permutation beyond. The model-agnostic explainers draw absent regressors from
    ``background`` training periods (all when the sample is no larger; otherwise a sample drawn with ``seed``);
    ``evaluations`` is the permutation explainer's budget per period (0: 500, or 10 (2p + 1) for p regressors if larger)."""
    from .ext.sklearn import FittedLearner

    if not isinstance(fit, FittedLearner):
        raise SpecificationError("SHAP values are for machine-learning fits (a random forest, boosted trees, LightGBM, XGBoost, a "
                                 "neural network or nearest neighbours); a linear equation's contribution of x is b (x - mean(x))")
    if explainer not in ("auto", "tree", "exact", "permutation"):
        raise SpecificationError(f"explainer is 'auto', 'tree', 'exact' or 'permutation', not {explainer!r}")
    if background < 1 or evaluations < 0:
        raise SpecificationError("background is at least 1 and evaluations at least 0")
    shap = require("shap")
    X = _rows(fit, data, span)
    if X.empty:
        raise SpecificationError("no periods to explain: the span has no rows with every regressor")
    p = X.shape[1]
    tree = type(fit.method).__name__ in TREE_METHODS
    kind = explainer if explainer != "auto" else ("tree" if tree else "exact" if p <= EXACT_UP_TO else "permutation")
    if kind == "tree" and not tree:
        raise SpecificationError(f"TreeSHAP reads a tree model's trees; {fit.method.label} has none: use 'exact' or 'permutation'")
    Xn = X.to_numpy(float)
    library = f"shap {getattr(shap, '__version__', '')}".strip()
    title = f"{fit.equation.text}, {fit.method.label}"
    if kind == "tree":
        ex = shap.TreeExplainer(fit.model)
        values = np.asarray(ex.shap_values(Xn, check_additivity=True), float)
        base = float(np.ravel(ex.expected_value)[0])
        prediction = fit.model.predict(Xn)
        return ShapValues(pd.DataFrame(values, index=X.index, columns=X.columns), base, X, pd.Series(prediction, index=X.index, name="prediction"),
                          "tree", title, library=library)
    n = len(fit.X)
    rows = np.arange(n) if n <= background else np.sort(make_rng(seed).choice(n, size=int(background), replace=False))
    masker = shap.maskers.Independent(fit.X.to_numpy(float)[rows], max_samples=len(rows))
    state = np.random.get_state()  # shap seeds NumPy's global generator: put it back afterwards
    try:
        if kind == "exact":
            if p > 16:
                raise SpecificationError(f"exact SHAP values evaluate 2^{p} subsets: use the permutation explainer")
            out = shap.explainers.Exact(fit.model.predict, masker)(Xn, max_evals=max(2**p + 1, 100_000), silent=True)
            budget = None
        else:
            budget = int(evaluations) or max(500, 10 * (2 * p + 1))
            if budget < 2 * p + 1:
                raise SpecificationError(f"the permutation explainer needs at least 2p + 1 = {2 * p + 1} evaluations per period")
            out = shap.explainers.Permutation(fit.model.predict, masker, seed=int(seed))(Xn, max_evals=budget, silent=True)
    finally:
        np.random.set_state(state)
    values = np.asarray(out.values, float)
    base = float(np.ravel(out.base_values)[0])
    prediction = fit.model.predict(Xn)
    return ShapValues(pd.DataFrame(values, index=X.index, columns=X.columns), base, X, pd.Series(prediction, index=X.index, name="prediction"),
                      kind, title, fit.X.index[rows], budget, int(seed), library)
