"""The scikit-learn adapter: shrinkage and machine-learning methods (``pymetrix.ext.sklearn``).

scikit-learn estimators take their settings in the constructor, so ``backend_options`` go to the ``model`` stage.
Cross-validation refits the same estimator on each window with the same arguments.
"""

from __future__ import annotations

from .base import Adapter, AdapterSpec, Option

ADAPTER = Adapter("sklearn", "scikit-learn")

_INPUTS = {"X (regressors, standardized for shrinkage, without the constant)": "X", "y": "y"}
_FOLDS = Option("folds", None, "expanding-window cross-validation folds (TimeSeriesSplit)")
_PENALTY = Option("alpha", "alpha", "0: chosen by time-series cross-validation over a grid")
_SHRINK_OUT = {"coef_": "params (rescaled to the original regressors)", "intercept_": "params['const']"}
_CONVERGE = {"model.max_iter": (100_000, "coordinate descent stops short on standardized macro data"),
             "model.tol": (1e-10, "tight enough for the coefficients to match a closed form where one exists")}


def _spec(wraps: str, *options: Option, fixed: dict | None = None, outputs: dict | None = None) -> AdapterSpec:
    return AdapterSpec(wraps, "sklearn", _INPUTS, (*options, _FOLDS), fixed or {}, outputs or {"predict()": "fitted, forecasts"},
                       default_stage="model", stages=("model",))


RIDGE = _spec("sklearn.linear_model.Ridge", _PENALTY, outputs=_SHRINK_OUT)
LASSO = _spec("sklearn.linear_model.Lasso", _PENALTY, fixed=_CONVERGE, outputs=_SHRINK_OUT)
ELASTIC_NET = _spec("sklearn.linear_model.ElasticNet", _PENALTY, Option("l1_ratio", "l1_ratio"), fixed=_CONVERGE, outputs=_SHRINK_OUT)
_UNLESS = "given only when it is not scikit-learn's default"
RANDOM_FOREST = _spec("sklearn.ensemble.RandomForestRegressor", Option("trees", "n_estimators"), Option("min_leaf", "min_samples_leaf"),
                      Option("max_features", "max_features"), Option("seed", "random_state"), Option("max_depth", "max_depth", _UNLESS),
                      Option("min_split", "min_samples_split", _UNLESS), Option("criterion", "criterion", _UNLESS),
                      Option("bootstrap", "bootstrap", _UNLESS), Option("max_samples", "max_samples", _UNLESS),
                      Option("pruning", "ccp_alpha", _UNLESS))
GRADIENT_BOOSTING = _spec("sklearn.ensemble.GradientBoostingRegressor", Option("trees", "n_estimators"),
                          Option("learning_rate", "learning_rate"), Option("depth", "max_depth"), Option("subsample", "subsample"),
                          Option("seed", "random_state"), Option("loss", "loss", _UNLESS),
                          Option("quantile", "alpha", "with loss huber or quantile, when not 0.9"),
                          Option("min_leaf", "min_samples_leaf", _UNLESS), Option("min_split", "min_samples_split", _UNLESS),
                          Option("max_features", "max_features", _UNLESS), Option("stop_after", "n_iter_no_change", _UNLESS),
                          Option("held_out", "validation_fraction", _UNLESS), Option("pruning", "ccp_alpha", _UNLESS))
NEURAL_NETWORK = _spec("sklearn.neural_network.MLPRegressor (standardized inputs and target)", Option("hidden", "hidden_layer_sizes"),
                       Option("penalty", "alpha"), Option("seed", "random_state"), Option("activation", "activation", _UNLESS),
                       Option("solver", "solver", _UNLESS), Option("learning_rate", "learning_rate_init", _UNLESS),
                       Option("batch", "batch_size", _UNLESS), Option("max_iter", "max_iter", "5000 unless given"),
                       Option("early_stopping", "early_stopping", _UNLESS),
                       fixed={"model.max_iter": (5000, "scikit-learn's default (200) stops before convergence on small samples")})
NEAREST_NEIGHBOURS = _spec("sklearn.neighbors.KNeighborsRegressor (standardized inputs)", Option("k", "n_neighbors"),
                           Option("distance_weighted", "weights", "True → 'distance', False → 'uniform'"),
                           Option("power", "p", _UNLESS))

SPECS = {"Ridge": RIDGE, "Lasso": LASSO, "ElasticNet": ELASTIC_NET, "RandomForest": RANDOM_FOREST, "GradientBoosting": GRADIENT_BOOSTING,
         "NeuralNetwork": NEURAL_NETWORK, "NearestNeighbours": NEAREST_NEIGHBOURS}
