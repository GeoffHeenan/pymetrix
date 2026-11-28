"""The XGBoost adapter: regularised gradient-boosted trees (``pymetrix.ext.boosting``).

XGBoost's scikit-learn interface takes its settings in the constructor, so ``backend_options`` go to the ``model``
stage. One thread and the histogram tree method make a fit the same on every run with the same library version.
"""

from __future__ import annotations

from .base import Adapter, AdapterSpec, Option

ADAPTER = Adapter("xgboost")

_UNLESS = "given only when it is not XGBoost's default"
XGBOOST = AdapterSpec(
    "xgboost.XGBRegressor", "xgboost", {"X (regressors, without the constant)": "X", "y": "y"},
    (Option("trees", "n_estimators"), Option("learning_rate", "learning_rate"), Option("depth", "max_depth"),
     Option("subsample", "subsample"), Option("seed", "random_state"),
     Option("loss", "objective", "squared_error → 'reg:squarederror', absolute_error → 'reg:absoluteerror', "
                                 "quantile → 'reg:quantileerror'"),
     Option("quantile", "quantile_alpha", "with loss quantile"), Option("max_features", "colsample_bytree", _UNLESS),
     Option("min_child_weight", "min_child_weight", _UNLESS), Option("l1", "reg_alpha", _UNLESS), Option("l2", "reg_lambda", _UNLESS),
     Option("min_gain", "gamma", _UNLESS), Option("folds", None, "expanding-window cross-validation folds (TimeSeriesSplit)")),
    {"model.n_jobs": (1, "one thread: the same fit on every run"),
     "model.tree_method": ("hist", "XGBoost's default on the CPU, fixed so that a change of default does not change the fit")},
    {"predict()": "fitted, forecasts"}, default_stage="model", stages=("model",))

SPECS = {"XGBoost": XGBOOST}
