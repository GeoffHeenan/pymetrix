"""Reference plugins: gradient-boosted trees from LightGBM and XGBoost, as equation methods beside scikit-learn's.

    Equation("dgdp ~ dgdp[-1] + spread[-1] + pmi[-1] + ...").fit(data, method=LightGBM())
    Equation(...).fit(data, method=XGBoost(depth=3, l2=5))
    Equation(...).fit(data, method=LightGBM(loss="quantile", quantile=0.9))       # a conditional quantile

Both return a ``FittedLearner`` (``pymetrix.ext.sklearn``), like the random forest and scikit-learn's boosted trees:
in-sample fit, the error of expanding-window cross-validation (also the forecast standard error), permutation
importance, SHAP values (``fit.shap()``), conformal intervals (``fit.conformal(...)``) and recursive forecasts. Each
runs on one thread with its reproducible settings, so the same seed gives the same fit on every run. Double machine
learning takes them as nuisance learners (``learner="lightgbm"`` or ``learner=XGBoost(...)``).

Install with ``pip install pymetrix[lightgbm]`` or ``pip install pymetrix[xgboost]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from .._version import __version__
from ..adapters import lgbm as lgbm_specs
from ..adapters import require
from ..adapters import xgb as xgb_specs
from ..adapters.base import Adapter, AdapterSpec, BackendOptions, check_backend_options
from ..errors import SpecificationError
from ..forms import Widget
from ..plugins import PluginInfo, register
from .sklearn import _FOLDS, FittedLearner, _changed, _learn

if TYPE_CHECKING:
    from ..data import Dataset
    from ..estimation.equation import Equation

LIGHTGBM_PLUGIN = PluginInfo(
    name="pymetrix-lightgbm",
    version=__version__,
    requires={"lightgbm": ">=4.6"},
    license="BSD-3-Clause",
    maintainer="Geoffrey Heenan",
    description="Gradient-boosted trees from LightGBM",
    install="pip install pymetrix[lightgbm]",
)
XGBOOST_PLUGIN = PluginInfo(
    name="pymetrix-xgboost",
    version=__version__,
    requires={"xgboost": ">=2.1.4"},
    license="BSD-3-Clause",
    maintainer="Geoffrey Heenan",
    description="Gradient-boosted trees from XGBoost",
    install="pip install pymetrix[xgboost]",
)

__all__ = ["LightGBM", "XGBoost"]

_SEED = Widget("number", min=0, max=2**31 - 1, advanced=True, help="seed of the random draws: the same seed gives the same fit")
_LOSS = Widget("choice", help="the loss each tree reduces: squared error, absolute error, or a quantile's check loss")
_QUANTILE = Widget("number", min=0.01, max=0.99, step=0.05, advanced=True, help="the quantile, with loss quantile")
_OPTIONS = Widget("", advanced=True, help="other arguments of the library's estimator")


def _fixed(spec: AdapterSpec) -> dict[str, Any]:
    return spec.fixed_for("model")


def _check(method: Any) -> None:
    check_backend_options(method)
    if not 0 < method.quantile < 1:
        raise SpecificationError("the quantile is between 0 and 1")
    if not 0 < method.subsample <= 1:
        raise SpecificationError("subsample is a share: above 0 and at most 1")


@dataclass(frozen=True)
class LightGBM:
    """Gradient-boosted trees from LightGBM: trees grown leaf by leaf on binned regressors, each fitted to the errors left by the earlier ones.

    Where scikit-learn's boosted trees grow each tree level by level to a fixed depth, LightGBM splits whichever
    leaf reduces the loss most, so the number of leaves, not the depth, sets a tree's size; the fewest observations
    in a leaf keeps it from chasing single periods, which matters in short macroeconomic samples. As with any
    boosting, the number of trees and the learning rate work together (a smaller rate needs more trees), and the
    forecasts flatten outside the range of the sample. Judge the fit by the cross-validated RMSE and R-squared in the
    summary (expanding windows, each tested on later periods), not the in-sample fit; the cross-validated RMSE is also
    the forecast standard error. The settings are not tuned automatically. Loss quantile fits a conditional quantile,
    which conformalised quantile regression uses. The fit runs on one thread in LightGBM's deterministic mode, so the
    same seed gives the same trees. There are no coefficients; ``fit.importance()`` and ``fit.shap()`` show how much
    the fit relies on each regressor, not a causal effect.
    """

    maturity: ClassVar[str] = "beta"
    label: ClassVar[str] = "LightGBM"
    kind: ClassVar[str] = "learner"
    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow", "shap-values", "conformal-prediction")
    references: ClassVar[tuple[str, ...]] = (
        ("Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, T.-Y. (2017). LightGBM: a highly efficient "
         "gradient boosting decision tree. Advances in Neural Information Processing Systems 30, 3146-3154."),
        ("Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics 29(5), 1189-1232. "
         "doi:10.1214/aos/1013203451"),
    )
    requires: ClassVar[tuple[str, ...]] = ("lightgbm",)
    wraps: ClassVar[str] = "lightgbm.LGBMRegressor"
    adapter_spec: ClassVar[AdapterSpec] = lgbm_specs.LIGHTGBM
    backend_adapter: ClassVar[Adapter] = lgbm_specs.ADAPTER
    plugin_info: ClassVar[PluginInfo] = LIGHTGBM_PLUGIN

    trees: Annotated[int, Widget("number", min=10, max=10000, help="the number of boosting rounds")] = 300
    learning_rate: Annotated[float, Widget("number", min=0.001, max=1.0, step=0.01, help="the shrinkage of each tree (smaller: slower, smoother)")] = 0.05
    leaves: Annotated[int, Widget("number", min=2, max=1024, help="the most leaves in a tree (its size)")] = 15
    min_leaf: Annotated[int, Widget("number", min=1, max=500, help="the fewest observations in a leaf (larger: smoother)")] = 10
    subsample: Annotated[float, Widget("number", min=0.1, max=1.0, step=0.1, advanced=True, help="the share of observations each tree sees")] = 0.8
    folds: Annotated[int, _FOLDS] = 5
    seed: Annotated[int, _SEED] = 0
    loss: Annotated[Literal["squared_error", "absolute_error", "quantile"], _LOSS] = "squared_error"
    quantile: Annotated[float, _QUANTILE] = 0.9
    max_features: Annotated[float, Widget("number", min=0.1, max=1.0, step=0.1, advanced=True, help="the share of regressors each tree may use")] = 1.0
    depth: Annotated[int | None, Widget("number", min=1, max=64, advanced=True, help="the deepest a tree grows; empty = no limit (leaves set the size)")] = None
    l1: Annotated[float, Widget("number", min=0, max=1e6, step=0.1, advanced=True, help="L1 penalty on the leaf values")] = 0.0
    l2: Annotated[float, Widget("number", min=0, max=1e6, step=0.1, advanced=True, help="L2 penalty on the leaf values")] = 0.0
    min_gain: Annotated[float, Widget("number", min=0, max=1e6, step=0.01, advanced=True, help="the smallest fall in the loss that a split must give")] = 0.0

    backend_options: Annotated[BackendOptions, _OPTIONS] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check(self)

    def _arguments(self) -> dict[str, Any]:
        args: dict[str, Any] = {"n_estimators": int(self.trees), "learning_rate": float(self.learning_rate), "num_leaves": int(self.leaves),
                                "min_child_samples": int(self.min_leaf), "subsample": float(self.subsample), "random_state": int(self.seed)}
        if self.subsample < 1:
            args["subsample_freq"] = 1  # LightGBM samples rows only when told how often
        if self.loss != "squared_error":
            args["objective"] = {"absolute_error": "regression_l1", "quantile": "quantile"}[self.loss]
        if self.loss == "quantile":
            args["alpha"] = float(self.quantile)
        args.update(_changed(self, {"max_features": ("colsample_bytree", 1.0), "depth": ("max_depth", None), "l1": ("reg_alpha", 0.0),
                                    "l2": ("reg_lambda", 0.0), "min_gain": ("min_split_gain", 0.0)}))
        return args

    def _model(self, run: Any) -> Any:
        return run.call("model", require("lightgbm").LGBMRegressor, **self._arguments())

    def estimator(self, classifier: bool = False, seed: int | None = None) -> Any:
        """The library's estimator with these settings (and pymetrix's fixed ones), unfitted: a classifier (for a
        propensity score, without the loss) or a regressor; ``seed`` replaces the seed (double ML's cross-fitting)."""
        args = {**_fixed(self.adapter_spec), **self._arguments(), **dict(self.backend_options)}
        if seed is not None:
            args["random_state"] = int(seed) % (2**31 - 1)
        if classifier:
            args = {k: v for k, v in args.items() if k not in ("objective", "alpha")}
            return require("lightgbm").LGBMClassifier(**args)
        return require("lightgbm").LGBMRegressor(**args)

    def estimate(self, eq: Equation, data: Dataset, *, span: Any = None, where: Any = None, weights: Any = None, missing: str = "drop") -> FittedLearner:
        return _learn(self, eq, data, span, where, weights, missing)


@dataclass(frozen=True)
class XGBoost:
    """Gradient-boosted trees from XGBoost: trees fitted one after another to the errors left by the earlier ones, with penalties on the leaf values.

    XGBoost adds L1 and L2 penalties on the leaves' values and a smallest gain for a split to the boosting of
    scikit-learn's boosted trees, which steadies the fit when the sample is short. Depth sets the order of
    interactions (2: pairs of regressors), and the number of trees and the learning rate work together (a smaller
    rate needs more trees); the forecasts flatten outside the range of the sample. Judge the fit by the
    cross-validated RMSE and R-squared in the summary (expanding windows, each tested on later periods), not the
    in-sample fit; the cross-validated RMSE is also the forecast standard error. The settings are not tuned
    automatically. Loss quantile fits a conditional quantile, which conformalised quantile regression uses. The fit
    runs on one thread with the histogram method, so the same seed gives the same trees. There are no coefficients;
    ``fit.importance()`` and ``fit.shap()`` show how much the fit relies on each regressor, not a causal effect.
    """

    maturity: ClassVar[str] = "beta"
    label: ClassVar[str] = "XGBoost"
    kind: ClassVar[str] = "learner"
    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow", "shap-values", "conformal-prediction")
    references: ClassVar[tuple[str, ...]] = (
        ("Chen, T. and Guestrin, C. (2016). XGBoost: a scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD "
         "International Conference on Knowledge Discovery and Data Mining, 785-794."),
        ("Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics 29(5), 1189-1232. "
         "doi:10.1214/aos/1013203451"),
    )
    requires: ClassVar[tuple[str, ...]] = ("xgboost",)
    wraps: ClassVar[str] = "xgboost.XGBRegressor"
    adapter_spec: ClassVar[AdapterSpec] = xgb_specs.XGBOOST
    backend_adapter: ClassVar[Adapter] = xgb_specs.ADAPTER
    plugin_info: ClassVar[PluginInfo] = XGBOOST_PLUGIN

    trees: Annotated[int, Widget("number", min=10, max=10000, help="the number of boosting rounds")] = 300
    learning_rate: Annotated[float, Widget("number", min=0.001, max=1.0, step=0.01, help="the shrinkage of each tree (smaller: slower, smoother)")] = 0.05
    depth: Annotated[int, Widget("number", min=1, max=20, help="the depth of each tree (1: additive in the regressors)")] = 2
    subsample: Annotated[float, Widget("number", min=0.1, max=1.0, step=0.1, advanced=True, help="the share of observations each tree sees")] = 0.8
    folds: Annotated[int, _FOLDS] = 5
    seed: Annotated[int, _SEED] = 0
    loss: Annotated[Literal["squared_error", "absolute_error", "quantile"], _LOSS] = "squared_error"
    quantile: Annotated[float, _QUANTILE] = 0.9
    max_features: Annotated[float, Widget("number", min=0.1, max=1.0, step=0.1, advanced=True, help="the share of regressors each tree may use")] = 1.0
    min_child_weight: Annotated[float, Widget("number", min=0, max=1e6, step=1, advanced=True,
                                              help="the smallest total weight (with squared error, observations) in a leaf")] = 1.0
    l1: Annotated[float, Widget("number", min=0, max=1e6, step=0.1, advanced=True, help="L1 penalty on the leaf values")] = 0.0
    l2: Annotated[float, Widget("number", min=0, max=1e6, step=0.1, advanced=True, help="L2 penalty on the leaf values")] = 1.0
    min_gain: Annotated[float, Widget("number", min=0, max=1e6, step=0.01, advanced=True, help="the smallest fall in the loss that a split must give")] = 0.0

    backend_options: Annotated[BackendOptions, _OPTIONS] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check(self)

    def _arguments(self) -> dict[str, Any]:
        args: dict[str, Any] = {"n_estimators": int(self.trees), "learning_rate": float(self.learning_rate), "max_depth": int(self.depth),
                                "subsample": float(self.subsample), "random_state": int(self.seed)}
        if self.loss != "squared_error":
            args["objective"] = {"absolute_error": "reg:absoluteerror", "quantile": "reg:quantileerror"}[self.loss]
        if self.loss == "quantile":
            args["quantile_alpha"] = float(self.quantile)
        args.update(_changed(self, {"max_features": ("colsample_bytree", 1.0), "min_child_weight": ("min_child_weight", 1.0),
                                    "l1": ("reg_alpha", 0.0), "l2": ("reg_lambda", 1.0), "min_gain": ("gamma", 0.0)}))
        return args

    def _model(self, run: Any) -> Any:
        return run.call("model", require("xgboost").XGBRegressor, **self._arguments())

    def estimator(self, classifier: bool = False, seed: int | None = None) -> Any:
        """The library's estimator with these settings (and pymetrix's fixed ones), unfitted: a classifier (for a
        propensity score, without the loss) or a regressor; ``seed`` replaces the seed (double ML's cross-fitting)."""
        args = {**_fixed(self.adapter_spec), **self._arguments(), **dict(self.backend_options)}
        if seed is not None:
            args["random_state"] = int(seed) % (2**31 - 1)
        if classifier:
            args = {k: v for k, v in args.items() if k not in ("objective", "quantile_alpha")}
            return require("xgboost").XGBClassifier(**args)
        return require("xgboost").XGBRegressor(**args)

    def estimate(self, eq: Equation, data: Dataset, *, span: Any = None, where: Any = None, weights: Any = None, missing: str = "drop") -> FittedLearner:
        return _learn(self, eq, data, span, where, weights, missing)


register("methods", LightGBM, plugin=LIGHTGBM_PLUGIN)
register("methods", XGBoost, plugin=XGBOOST_PLUGIN)
