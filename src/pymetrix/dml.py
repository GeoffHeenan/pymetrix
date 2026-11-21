"""Double/debiased machine learning (Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey and Robins 2018).

    double_ml(data, "y", "d", ["x1", "x2", ...])                          # partially linear regression, OLS learners
    double_ml(data, "y", "d", controls, learner="forest", folds=5, repeats=5)
    double_ml(data, "y", "d", controls, model="pliv", instrument="z")     # partially linear IV
    double_ml(data, "y", "d", controls, model="irm", score="ATTE")        # interactive model: ATE or ATT (doubly robust)

The models and their Neyman-orthogonal scores psi = psi_a theta + psi_b:

- PLR, Y = D theta + g(X) + U, D = m(X) + V: partialling out, psi = (Y - l(X) - theta (D - m(X))) (D - m(X)),
  with l(X) = E[Y|X] and m(X) = E[D|X];
- PLIV, Y = D theta + g(X) + U, Z = m(X) + V, E[U|Z, X] = 0: psi = (Y - l(X) - theta (D - r(X))) (Z - m(X)),
  with r(X) = E[D|X] and m(X) = E[Z|X];
- IRM, Y = g(D, X) + U, D binary with propensity m(X): the doubly robust (AIPW) scores, for the ATE
  psi = g(1,X) - g(0,X) + D (Y - g(1,X)) / m(X) - (1 - D)(Y - g(0,X)) / (1 - m(X)) - theta, and for the ATT
  psi = D (Y - g(0,X)) / p - m(X) (1 - D)(Y - g(0,X)) / (p (1 - m(X))) - theta D / p, p = P(D = 1).

Nuisance functions are learned by K-fold cross-fitting: each fold's predictions come from learners fitted on the
other folds. theta solves the score averaged over all observations (the paper's DML2). Its variance is
mean(psi^2) / (n J^2), J = mean(psi_a); with ``cluster`` the folds keep each cluster whole and the variance sums psi
within clusters (times G / (G - 1)). With ``repeats`` > 1 the sample is split again and the estimates are aggregated
by their median, the standard error by sqrt(median(se_s^2 + (theta_s - theta)^2)) (the paper's section 3.4).

Learners: ``"ols"`` (least squares; a logit for the propensity score) needs nothing more; ``"lasso"``, ``"forest"``
and ``"boosting"`` (and pymetrix's ``Lasso()``, ``RandomForest()`` and ``GradientBoosting()`` with their settings)
need scikit-learn (``pip install pymetrix[ml]``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from . import _micro
from ._api import api
from ._random import generator as make_rng
from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["FittedDoubleML", "double_ml", "dml_learner"]

LEARNERS = ("ols", "lasso", "forest", "boosting")


# -- learners ---------------------------------------------------------------------------------------------------------

def _ols_fit(X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    A = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    return lambda Xn: np.column_stack([np.ones(len(Xn)), Xn]) @ b


def _logit_fit(X: np.ndarray, y: np.ndarray, max_iter: int = 100) -> Callable[[np.ndarray], np.ndarray]:
    """A logit by Newton's method, with a tiny ridge so that separated samples still give finite coefficients."""
    A = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(A.shape[1])
    b[0] = np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))
    ridge = 1e-6 * len(y) * np.eye(len(b))
    ridge[0, 0] = 0.0
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-np.clip(A @ b, -35, 35)))
        g = A.T @ (y - p) - ridge @ b
        H = (A * (p * (1 - p))[:, None]).T @ A + ridge
        step = np.linalg.solve(H, g)
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return lambda Xn: 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones(len(Xn)), Xn]) @ b, -35, 35)))


def dml_learner(spec: Any, classifier: bool, seed: int) -> tuple[str, Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]]]:
    """(label, fit) for a learner given by name (``"ols"``, ``"lasso"``, ``"forest"``, ``"boosting"``) or as one of
    pymetrix's scikit-learn learners; ``fit(X, y)`` returns a predictor (of probabilities, for a classifier)."""
    kind = type(spec).__name__ if not isinstance(spec, str) else spec.lower()
    if kind == "ols":
        return ("logit" if classifier else "least squares"), (_logit_fit if classifier else _ols_fit)
    names = {"lasso": "Lasso", "forest": "RandomForest", "boosting": "GradientBoosting"}
    kind = {v: v for v in names.values()}.get(kind, names.get(kind, kind))
    if kind not in names.values():
        raise SpecificationError(f"a learner is one of {', '.join(map(repr, LEARNERS))} or Lasso(), RandomForest(), GradientBoosting(); "
                                 f"not {spec!r}")
    from .adapters import require

    def opt(name: str, default: Any) -> Any:
        return default if isinstance(spec, str) else getattr(spec, name, default)

    rs = int(opt("seed", 0)) + seed
    if kind == "Lasso":
        lm, pipe, prep = require("sklearn.linear_model"), require("sklearn.pipeline"), require("sklearn.preprocessing")
        alpha = float(opt("alpha", 0.0))
        cv = int(opt("folds", 5))
        if classifier:
            import inspect

            modern = "use_legacy_attributes" in inspect.signature(lm.LogisticRegressionCV).parameters  # scikit-learn 1.8 on
            l1 = {"l1_ratio": 1.0} if modern else {"penalty": "l1"}
            l1s = {"l1_ratios": (1.0,), "scoring": "neg_log_loss", "use_legacy_attributes": False} if modern else {"penalty": "l1"}
            model = (lm.LogisticRegression(C=1 / alpha, solver="liblinear", random_state=rs, **l1) if alpha > 0
                     else lm.LogisticRegressionCV(cv=cv, solver="liblinear", random_state=rs, **l1s))
            label = "l1-penalised logit"
        else:
            model = lm.Lasso(alpha=alpha, max_iter=100_000) if alpha > 0 else lm.LassoCV(cv=cv, max_iter=100_000, random_state=rs)
            label = "lasso"
        label += f" (penalty {alpha:g})" if alpha > 0 else " (penalty by cross-validation)"
        make = lambda: pipe.make_pipeline(prep.StandardScaler(), model)  # noqa: E731
    elif kind == "RandomForest":
        ens = require("sklearn.ensemble")
        args = dict(n_estimators=int(opt("trees", 500)), min_samples_leaf=int(opt("min_leaf", 5)), max_features=float(opt("max_features", 0.5)),
                    max_depth=opt("max_depth", None), random_state=rs)
        model = ens.RandomForestClassifier(**args) if classifier else ens.RandomForestRegressor(**args)
        label = f"random forest ({args['n_estimators']} trees)"
        make = lambda: model  # noqa: E731
    else:
        ens = require("sklearn.ensemble")
        args = dict(n_estimators=int(opt("trees", 300)), learning_rate=float(opt("learning_rate", 0.05)), max_depth=int(opt("depth", 2)),
                    subsample=float(opt("subsample", 0.8)), random_state=rs)
        model = ens.GradientBoostingClassifier(**args) if classifier else ens.GradientBoostingRegressor(**args)
        label = f"boosted trees ({args['n_estimators']} rounds)"
        make = lambda: model  # noqa: E731

    def fit(X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        m = require("sklearn.base").clone(make())
        if classifier:
            m.fit(X, y.astype(int))
            return lambda Xn: m.predict_proba(Xn)[:, 1]
        m.fit(X, y)
        return m.predict

    return label, fit


# -- folds and scores -------------------------------------------------------------------------------------------------

def _folds(n: int, k: int, rng: np.random.Generator, groups: np.ndarray | None) -> np.ndarray:
    """A fold number for each observation (0 .. k-1); clusters are kept whole."""
    if k == 1:
        return np.zeros(n, dtype=int)
    if groups is None:
        out = np.empty(n, dtype=int)
        out[rng.permutation(n)] = np.arange(n) % k
        return out
    labels, inverse = np.unique(groups, return_inverse=True)
    if len(labels) < k:
        raise SpecificationError(f"{len(labels)} clusters: too few for {k} folds")
    which = np.empty(len(labels), dtype=int)
    which[rng.permutation(len(labels))] = np.arange(len(labels)) % k
    return which[inverse]


def _cross_fit(fit: Callable, X: np.ndarray, y: np.ndarray, folds: np.ndarray, subset: np.ndarray | None = None) -> np.ndarray:
    """Out-of-fold predictions of y from X; a learner fitted on the training folds (only rows in ``subset``)."""
    pred = np.empty(len(y))
    for f in np.unique(folds):
        test = folds == f
        train = ~test if len(np.unique(folds)) > 1 else np.ones(len(y), bool)  # one fold: no cross-fitting
        if subset is not None:
            train = train & subset
        if train.sum() < 2:
            raise SpecificationError("a training fold has too few observations for the learner")
        pred[test] = fit(X[train], y[train])(X[test])
    return pred


def _scores(model: str, score: str, y: np.ndarray, d: np.ndarray, z: np.ndarray | None, X: np.ndarray, folds: np.ndarray,
            learners: dict[str, tuple[str, Callable]], trim: float) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """(psi_a, psi_b, out-of-fold RMSE of each nuisance function)."""
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))  # noqa: E731
    if model == "plr":
        l_hat = _cross_fit(learners["l"][1], X, y, folds)
        m_hat = _cross_fit(learners["m"][1], X, d, folds)
        u, v = y - l_hat, d - m_hat
        return -v * v, u * v, {"l(X) = E[Y|X]": rmse(y, l_hat), "m(X) = E[D|X]": rmse(d, m_hat)}
    if model == "pliv":
        assert z is not None
        l_hat = _cross_fit(learners["l"][1], X, y, folds)
        m_hat = _cross_fit(learners["m"][1], X, z, folds)
        r_hat = _cross_fit(learners["r"][1], X, d, folds)
        u, v, w = y - l_hat, z - m_hat, d - r_hat
        return -w * v, u * v, {"l(X) = E[Y|X]": rmse(y, l_hat), "m(X) = E[Z|X]": rmse(z, m_hat), "r(X) = E[D|X]": rmse(d, r_hat)}
    treated = d == 1
    g0 = _cross_fit(learners["g"][1], X, y, folds, subset=~treated)
    g1 = _cross_fit(learners["g"][1], X, y, folds, subset=treated) if score == "ATE" else np.full(len(y), np.nan)
    m = np.clip(_cross_fit(learners["m"][1], X, d, folds), trim, 1 - trim)
    fits = {"g(0,X) = E[Y|D=0,X]": rmse(y[~treated], g0[~treated]), "m(X) = P(D=1|X)": rmse(d, m)}
    if score == "ATE":
        fits["g(1,X) = E[Y|D=1,X]"] = rmse(y[treated], g1[treated])
        psi_b = g1 - g0 + d * (y - g1) / m - (1 - d) * (y - g0) / (1 - m)
        return -np.ones(len(y)), psi_b, fits
    p = d.mean()
    psi_b = d * (y - g0) / p - m * (1 - d) * (y - g0) / (p * (1 - m))
    return -d / p, psi_b, fits


def _solve(psi_a: np.ndarray, psi_b: np.ndarray, groups: np.ndarray | None) -> tuple[float, float]:
    """theta and its standard error from the stacked scores."""
    n = len(psi_a)
    theta = -psi_b.sum() / psi_a.sum()
    psi = psi_a * theta + psi_b
    J = psi_a.mean()
    if groups is None:
        var = np.mean(psi**2) / (n * J**2)
    else:
        sums = pd.Series(psi).groupby(groups).sum().to_numpy()
        G = len(sums)
        var = G / (G - 1) * np.sum(sums**2) / (n**2 * J**2)
    return float(theta), float(np.sqrt(var))


# -- the result -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FittedDoubleML:
    """A double/debiased machine-learning estimate: the causal parameter, its standard error, and each split's estimate."""

    model: str  # "plr", "pliv" or "irm"
    score: str  # "partialling out", "ATE" or "ATTE"
    outcome: str
    treatment: str
    instrument: str | None
    controls: tuple[str, ...]
    theta: float
    se: float
    thetas: np.ndarray = field(repr=False)  # one estimate per repetition of the sample split
    ses: np.ndarray = field(repr=False)
    learners: dict[str, str] = field(repr=False)  # nuisance function -> learner
    nuisance_rmse: pd.DataFrame = field(repr=False)  # out-of-fold RMSE of each nuisance function, by repetition
    folds: int = 5
    repeats: int = 1
    nobs: int = 0
    clusters: int | None = None
    cluster: str | None = None
    splits: np.ndarray = field(default=None, repr=False)  # (n, repeats) fold of each observation in each repetition
    alpha: float = 0.05
    provenance: Any = field(default=None, repr=False)

    stage = "estimate"

    @property
    def params(self) -> Any:
        from .estimation.params import PointEstimate

        return PointEstimate((self.treatment,), np.array([self.theta]), np.array([[self.se**2]]))

    @property
    def stats(self) -> dict[str, float]:
        out: dict[str, float] = {"nobs": float(self.nobs), "folds": float(self.folds), "repeats": float(self.repeats)}
        if self.clusters:
            out["clusters"] = float(self.clusters)
        return out

    def interval(self, alpha: float | None = None) -> tuple[float, float]:
        q = sps.norm.ppf(1 - (self.alpha if alpha is None else alpha) / 2)
        return self.theta - q * self.se, self.theta + q * self.se

    def summary(self) -> Table:
        label = {"plr": "Partially linear regression", "pliv": "Partially linear IV", "irm": "Interactive regression model"}[self.model]
        what = {"partialling out": "the effect", "ATE": "the average effect (ATE)", "ATTE": "the average effect on the treated (ATT)"}[self.score]
        split = (f"{self.folds}-fold cross-fitting" if self.folds > 1 else "no cross-fitting (nuisances fitted on the whole sample)")
        notes = [f"{label}: {what} of {self.treatment} on {self.outcome}"
                 + (f", instrument {self.instrument}" if self.instrument else "") + f"; {len(self.controls)} controls, {self.nobs} observations.",
                 f"{split}, {self.repeats} repetition(s)" + (", median aggregation." if self.repeats > 1 else "."),
                 "Learners: " + "; ".join(f"{k}: {v}" for k, v in self.learners.items()) + ".",
                 (f"Standard errors clustered by {self.cluster} ({self.clusters} clusters)." if self.cluster
                  else "Heteroskedasticity-robust standard errors from the score.")]
        return _micro.coefficient_table([self.treatment], [self.theta], [self.se], f"Double machine learning: {self.outcome}", notes,
                                        index_name="parameter")

    def splits_table(self) -> pd.DataFrame:
        q = sps.norm.ppf(1 - self.alpha / 2)
        frame = pd.DataFrame({"estimate": self.thetas, "std err": self.ses, "lower": self.thetas - q * self.ses,
                              "upper": self.thetas + q * self.ses}, index=pd.Index([f"split {i + 1}" for i in range(len(self.thetas))], name="repetition"))
        return frame

    def plot(self) -> Any:
        from .figures import Estimates, Figure, Panel

        t = self.splits_table()
        lo, hi = self.interval()
        est = pd.concat([t["estimate"], pd.Series([self.theta], index=["aggregate"])])
        lower = pd.concat([t["lower"], pd.Series([lo], index=["aggregate"])])
        upper = pd.concat([t["upper"], pd.Series([hi], index=["aggregate"])])
        return Figure((Panel((Estimates(est, lower, upper, f"{self.treatment}"),), title=f"Double ML: {self.treatment} on {self.outcome}, by sample split"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.summary().to_frame()

    def __repr__(self) -> str:
        return f"<FittedDoubleML {self.model} {self.treatment} on {self.outcome}: {self.theta:.4g} (s.e. {self.se:.3g})>"


@api("pymetrix:double_ml")
def double_ml(data: Dataset, outcome: str, treatment: str, controls: Sequence[str], model: Literal["plr", "pliv", "irm"] = "plr",
              instrument: str | None = None, score: Literal["ATE", "ATTE"] = "ATE", learner: Any = "ols", learner_treatment: Any = None,
              folds: int = 5, repeats: int = 1, cluster: str | None = None, trim: float = 0.01, seed: int = 0) -> FittedDoubleML:
    """Double/debiased machine learning of the effect of ``treatment`` on ``outcome`` given ``controls``: ``model``
    ``"plr"`` (partially linear), ``"pliv"`` (partially linear IV, with ``instrument``) or ``"irm"`` (interactive, a
    0/1 treatment; ``score`` ``"ATE"`` or ``"ATTE"``). ``learner`` learns the outcome's nuisance functions and
    ``learner_treatment`` (default: the same) the treatment's and instrument's; ``folds`` for cross-fitting (1: none),
    ``repeats`` sample splits aggregated by the median; ``cluster`` a series or axis for clustered errors and folds."""
    controls = _micro.names(controls)
    model = str(model).lower()  # type: ignore[assignment]
    if model not in ("plr", "pliv", "irm"):
        raise SpecificationError(f"model is 'plr', 'pliv' or 'irm', not {model!r}")
    if not controls:
        raise SpecificationError("double machine learning needs controls (the X the nuisance functions depend on)")
    if model == "pliv" and not instrument:
        raise SpecificationError("the partially linear IV model needs an instrument")
    if model != "pliv" and instrument:
        raise SpecificationError("an instrument goes with model='pliv'")
    if score not in ("ATE", "ATTE"):
        raise SpecificationError(f"score is 'ATE' or 'ATTE', not {score!r}")
    if folds < 1 or repeats < 1:
        raise SpecificationError("folds and repeats must be at least 1")
    if not 0 <= trim < 0.5:
        raise SpecificationError("trim is between 0 and 0.5")
    from ._capture import capture

    wanted = [outcome, treatment, *([instrument] if instrument else []), *controls, *([cluster] if cluster else [])]
    frame = _micro.columns(data, wanted, "double machine learning")
    numeric = [outcome, treatment, *([instrument] if instrument else []), *controls]
    y, d = _micro.numeric(frame, [outcome], "double ML")[:, 0], _micro.numeric(frame, [treatment], "double ML")[:, 0]
    z = _micro.numeric(frame, [instrument], "double ML")[:, 0] if instrument else None
    X = _micro.numeric(frame, controls, "double ML")
    n = len(y)
    if n < 2 * max(folds, 2) + X.shape[1]:
        raise SpecificationError(f"{n} observations: too few for {folds} folds and {X.shape[1]} controls")
    if model == "irm" and not np.isin(d, (0.0, 1.0)).all():
        raise SpecificationError(f"the interactive model needs a 0/1 treatment; {treatment} takes other values")
    if model == "irm" and (d.min() == d.max()):
        raise SpecificationError(f"{treatment} must have both treated and untreated observations")
    groups = frame[cluster].to_numpy() if cluster else None
    lt = learner if learner_treatment is None else learner_treatment
    rng = make_rng(seed)
    thetas, ses, fits, splits, used = [], [], [], [], {}
    with capture() as caught:
        for r in range(repeats):
            sub = int(rng.integers(0, 2**31 - 1))
            if model == "plr":
                learners = {"l": dml_learner(learner, False, sub), "m": dml_learner(lt, False, sub)}
                labels = {"l(X) = E[Y|X]": learners["l"][0], "m(X) = E[D|X]": learners["m"][0]}
            elif model == "pliv":
                learners = {"l": dml_learner(learner, False, sub), "m": dml_learner(lt, False, sub), "r": dml_learner(lt, False, sub)}
                labels = {"l(X) = E[Y|X]": learners["l"][0], "m(X) = E[Z|X]": learners["m"][0], "r(X) = E[D|X]": learners["r"][0]}
            else:
                learners = {"g": dml_learner(learner, False, sub), "m": dml_learner(lt, True, sub)}
                labels = {"g(D,X) = E[Y|D,X]": learners["g"][0], "m(X) = P(D=1|X)": learners["m"][0]}
            used = labels
            f = _folds(n, folds, rng, groups)
            psi_a, psi_b, rmse = _scores(model, score, y, d, z, X, f, learners, trim)
            if not np.isfinite(psi_a).all() or abs(psi_a.sum()) < 1e-12:
                raise SpecificationError("the score's derivative is zero: the treatment (or instrument) has no variation left given the controls")
            t, s = _solve(psi_a, psi_b, groups)
            thetas.append(t)
            ses.append(s)
            fits.append(rmse)
            splits.append(f)
    thetas_a, ses_a = np.array(thetas), np.array(ses)
    theta = float(np.median(thetas_a))
    se = float(np.sqrt(np.median(ses_a**2 + (thetas_a - theta) ** 2))) if repeats > 1 else float(ses_a[0])
    nuisance = pd.DataFrame(fits, index=pd.Index([f"split {i + 1}" for i in range(repeats)], name="repetition"))
    score_name = "partialling out" if model in ("plr", "pliv") else score
    spec = {"outcome": outcome, "treatment": treatment, "controls": controls, "model": model, "instrument": instrument, "score": score_name,
            "learner": str(learner), "learner_treatment": str(lt), "folds": folds, "repeats": repeats, "cluster": cluster, "trim": trim}
    prov = _micro.provenance(spec, data, frame, numeric, caught, {"type": "double machine learning", "model": model, "score": score_name,
                                                                  "learners": dict(used)}, seed=seed)
    return FittedDoubleML(model, score_name, outcome, treatment, instrument, tuple(controls), theta, se, thetas_a, ses_a, dict(used), nuisance,
                          folds, repeats, n, len(np.unique(groups)) if cluster else None, cluster, np.column_stack(splits), 0.05, prov)
