"""The LightGBM adapter: gradient-boosted trees grown leaf by leaf on histograms (``pymetrix.ext.boosting``).

LightGBM's scikit-learn interface takes its settings in the constructor, so ``backend_options`` go to the ``model``
stage. One thread, row-wise histograms and LightGBM's ``deterministic`` mode make a fit the same on every run with
the same library version.
"""

from __future__ import annotations

from .base import Adapter, AdapterSpec, Option

ADAPTER = Adapter("lightgbm")

_UNLESS = "given only when it is not LightGBM's default"
LIGHTGBM = AdapterSpec(
    "lightgbm.LGBMRegressor", "lightgbm", {"X (regressors, without the constant)": "X", "y": "y"},
    (Option("trees", "n_estimators"), Option("learning_rate", "learning_rate"), Option("leaves", "num_leaves"),
     Option("min_leaf", "min_child_samples"),
     Option("subsample", "subsample", "with subsample_freq=1 when below 1 (a new sample every round)"),
     Option("seed", "random_state"),
     Option("loss", "objective", "squared_error → 'regression', absolute_error → 'regression_l1', quantile → 'quantile'"),
     Option("quantile", "alpha", "with loss quantile"), Option("max_features", "colsample_bytree", _UNLESS),
     Option("depth", "max_depth", _UNLESS), Option("l1", "reg_alpha", _UNLESS), Option("l2", "reg_lambda", _UNLESS),
     Option("min_gain", "min_split_gain", _UNLESS),
     Option("folds", None, "expanding-window cross-validation folds (TimeSeriesSplit)")),
    {"model.n_jobs": (1, "one thread: the same fit on every run"),
     "model.deterministic": (True, "LightGBM's reproducible mode"),
     "model.force_row_wise": (True, "fixes the histogram layout, which the reproducible mode needs"),
     "model.verbosity": (-1, "LightGBM prints its progress otherwise")},
    {"predict()": "fitted, forecasts"}, default_stage="model", stages=("model",))

SPECS = {"LightGBM": LIGHTGBM}
