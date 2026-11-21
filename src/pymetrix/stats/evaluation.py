"""Evaluating and comparing forecasts: efficiency, encompassing, predictive ability, nested models, interval coverage,
and comparisons of many forecasts.

    mincer_zarnowitz(actual, forecast, horizon=4)       # efficiency: actual = a + b forecast, a = 0 and b = 1
    encompassing(actual, f1, f2, horizon=4)              # does f2 add anything to f1? (Harvey, Leybourne, Newbold)
    giacomini_white(loss1, loss2, horizon=1)             # conditional predictive ability
    clark_west(actual, small, large, horizon=1)          # nested models: does the larger one forecast better?
    christoffersen(hits, coverage=0.9)                   # interval forecasts: right coverage, hits independent
    spa(benchmark_losses, {"a": la, "b": lb})            # does any model beat the benchmark? (Hansen; White's Reality Check)
    model_confidence_set({"a": la, "b": lb, "c": lc})    # the set of best models (Hansen, Lunde, Nason)
    stepm(benchmark_losses, {"a": la, "b": lb})          # which models beat the benchmark? (Romano-Wolf)
    linlin_loss(errors, tau=0.7); linex_loss(errors, a=1.0)   # asymmetric losses
    binary_scores(probabilities, outcomes)               # Brier, log score, ROC area, Kuipers
    pesaran_timmermann(actual, forecast)                 # directional accuracy

Series are aligned on their index (missing values dropped); arrays must have the same length. ``horizon`` is the
forecast horizon h: h-step errors are correlated up to lag h - 1 even for an optimal forecast, so standard
errors allow for that (Newey-West, h - 1 lags, unless stated). These are the tests of Elliott and Timmermann
(2016), chapters 2, 12 and 15 to 18; ``Backtest`` gives them for backtests (``efficiency``, ``encompasses``, ``compare``,
``interval_test``), and ``spa`` and ``model_confidence_set`` take backtests too. The last two use the arch
package's bootstrap implementations (stationary bootstrap).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from .._api import api
from ..errors import SpecificationError
from .base import TestResult

__all__ = ["binary_log_score", "binary_scores", "brier_score", "christoffersen", "clark_west", "encompassing", "giacomini_white", "kuipers_score",
           "linex_loss", "linlin_loss", "mincer_zarnowitz", "model_confidence_set", "pesaran_timmermann", "roc_auc", "roc_curve", "spa", "stepm"]


def _aligned(*series: Any) -> list[np.ndarray]:
    """Arrays of the same length: pandas objects aligned on their index, then rows with any missing value dropped."""
    if all(isinstance(s, pd.Series) for s in series):
        frame = pd.concat(list(series), axis=1, join="inner").dropna()
        return [frame.iloc[:, i].to_numpy(float) for i in range(len(series))]
    arrays = [np.asarray(s, float).ravel() for s in series]
    if len({len(a) for a in arrays}) != 1:
        raise SpecificationError("give series of the same length")
    keep = np.all(np.isfinite(np.column_stack(arrays)), axis=1)
    return [a[keep] for a in arrays]


def newey_west(u: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West (Bartlett) long-run covariance of the rows of ``u`` (n x k, or a vector), with ``lags`` lags.
    ``u`` is used as given: centre it first when estimating the variance of a mean."""
    u = np.asarray(u, float)
    u = u[:, None] if u.ndim == 1 else u
    n = len(u)
    s = u.T @ u / n
    for lag in range(1, int(lags) + 1):
        g = u[lag:].T @ u[:-lag] / n
        s += (1 - lag / (lags + 1)) * (g + g.T)
    return s


def mean_differential_test(d: np.ndarray, horizon: int = 1, small_sample: bool = True) -> tuple[float, float | None, float]:
    """The Diebold-Mariano statistic for E[d] = 0, with its degrees of freedom (None: normal) and two-sided p-value.

    The variance of the mean allows for MA(h - 1) correlation (equal weights to lag h - 1, as Diebold and Mariano);
    ``small_sample`` applies the Harvey, Leybourne and Newbold (1997) correction and Student-t p-values."""
    d = np.asarray(d, float)
    n, h = len(d), int(horizon)
    u = d - d.mean()
    lrv = u @ u / n + 2 * sum(u[k:] @ u[:-k] / n for k in range(1, h))
    if lrv <= 0:
        lrv = u @ u / n  # the truncated estimate can be negative in small samples
    if lrv == 0:
        return 0.0, (n - 1 if small_sample else None), 1.0
    stat = float(d.mean() / np.sqrt(lrv / n))
    if small_sample:
        stat *= float(np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n))
        return stat, n - 1, float(2 * sps.t.sf(abs(stat), n - 1))
    return stat, None, float(2 * sps.norm.sf(abs(stat)))


def mincer_zarnowitz(actual: Any, forecast: Any, horizon: int = 1) -> TestResult:
    """Mincer-Zarnowitz efficiency test: regress the outcome on a constant and the forecast, and test a = 0, b = 1
    (Wald, chi-squared with 2 degrees of freedom; Newey-West standard errors with h - 1 lags).

    An efficient forecast under squared loss is unbiased (a = 0) and moves one for one with the outcome (b = 1):
    b < 1 means the forecasts vary too much."""
    y, f = _aligned(actual, forecast)
    n = len(y)
    if n < 5:
        raise SpecificationError("the Mincer-Zarnowitz test needs at least five forecasts")
    X = np.column_stack([np.ones(n), f])
    xtx = X.T @ X / n
    b = np.linalg.solve(xtx, X.T @ y / n)
    e = y - X @ b
    bread = np.linalg.inv(xtx)
    V = bread @ newey_west(X * e[:, None], max(int(horizon) - 1, 0)) @ bread / n
    diff = b - np.array([0.0, 1.0])
    stat = float(diff @ np.linalg.solve(V, diff))
    se = np.sqrt(np.diag(V))
    return TestResult("Mincer-Zarnowitz", "the forecasts are efficient (intercept 0, slope 1)", stat, 2, float(sps.chi2.sf(stat, 2)), "chi2",
                      details={"intercept": float(b[0]), "intercept se": float(se[0]), "slope": float(b[1]), "slope se": float(se[1]),
                               "observations": n, "horizon": int(horizon)})


def encompassing(actual: Any, forecast1: Any, forecast2: Any, horizon: int = 1, small_sample: bool = True) -> TestResult:
    """Forecast encompassing (Harvey, Leybourne and Newbold 1998): does forecast 1 encompass forecast 2, so that
    combining them would not improve on forecast 1?

    The statistic is the Diebold-Mariano test of E[(e1 - e2) e1] = 0, one-sided: a large positive value means
    forecast 2 holds information forecast 1 lacks."""
    y, f1, f2 = _aligned(actual, forecast1, forecast2)
    if len(y) < 5:
        raise SpecificationError("the encompassing test needs at least five forecasts")
    e1, e2 = y - f1, y - f2
    stat, df, _ = mean_differential_test((e1 - e2) * e1, horizon, small_sample)
    p = float(sps.t.sf(stat, df) if df is not None else sps.norm.sf(stat))
    return TestResult("Encompassing (HLN)", "forecast 1 encompasses forecast 2", stat, df, p, "t" if df is not None else "normal",
                      details={"observations": len(y), "horizon": int(horizon), "one-sided": "yes",
                               "combination weight on forecast 2": float(np.sum((e1 - e2) * e1) / np.sum((e1 - e2) ** 2)) if np.any(e1 != e2) else 0.0})


def giacomini_white(loss1: Any, loss2: Any, horizon: int = 1, conditional: bool = True) -> TestResult:
    """Giacomini-White (2006) test of equal conditional predictive ability, from the losses of two forecasts.

    With ``conditional`` the instruments are a constant and the loss differential when the forecast was made
    (h periods before), so the test asks whether past relative performance predicts future relative performance;
    otherwise only the constant (an unconditional test, like Diebold-Mariano). The statistic is
    n Zbar' Omega^-1 Zbar, chi-squared with as many degrees of freedom as instruments; Omega is the outer product
    of Z for one-step forecasts and Newey-West with h - 1 lags beyond. The sign says which forecast loses less:
    see ``details``."""
    l1, l2 = _aligned(loss1, loss2)
    d = l1 - l2
    h = int(horizon)
    if conditional:
        if len(d) <= h + 4:
            raise SpecificationError("the Giacomini-White test needs more forecasts than the horizon plus four")
        Z = np.column_stack([np.ones(len(d) - h), d[:-h]])
        target = d[h:]
    else:
        Z, target = np.ones((len(d), 1)), d
    zd = Z * target[:, None]
    n, q = zd.shape
    zbar = zd.mean(axis=0)
    omega = zd.T @ zd / n if h == 1 else newey_west(zd, h - 1)
    stat = float(n * zbar @ np.linalg.solve(omega, zbar))
    return TestResult("Giacomini-White" + (" (conditional)" if conditional else " (unconditional)"),
                      "the two forecasts have equal predictive ability" + (" given the information when they were made" if conditional else ""),
                      stat, q, float(sps.chi2.sf(stat, q)), "chi2",
                      details={"mean loss differential (1 - 2)": float(d.mean()), "better forecast": "1" if d.mean() < 0 else "2" if d.mean() > 0 else "neither",
                               "observations": n, "horizon": h})


def clark_west(actual: Any, small: Any, large: Any, horizon: int = 1) -> TestResult:
    """Clark-West (2007) test for nested models: does the larger model forecast better than the smaller one it nests?

    The MSPE difference is adjusted for the noise the larger model's extra parameters add under the null:
    f = e_small^2 - (e_large^2 - (small - large)^2); t = mean(f) / se (Newey-West, h - 1 lags), one-sided against
    the larger model forecasting better, with normal p-values. Diebold-Mariano is undersized for nested models."""
    y, fs, fl = _aligned(actual, small, large)
    n = len(y)
    if n < 5:
        raise SpecificationError("the Clark-West test needs at least five forecasts")
    adj = (y - fs) ** 2 - ((y - fl) ** 2 - (fs - fl) ** 2)
    var = float(newey_west(adj - adj.mean(), max(int(horizon) - 1, 0))[0, 0])
    stat = float(adj.mean() / np.sqrt(var / n)) if var > 0 else 0.0
    return TestResult("Clark-West", "the larger model forecasts no better than the smaller model it nests", stat, None, float(sps.norm.sf(stat)), "normal",
                      details={"MSPE small": float(np.mean((y - fs) ** 2)), "MSPE large": float(np.mean((y - fl) ** 2)),
                               "adjusted difference": float(adj.mean()), "observations": n, "horizon": int(horizon), "one-sided": "yes"})


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * np.log(y)


def christoffersen(hits: Any, coverage: float = 0.9) -> TestResult:
    """Christoffersen (1998) test of interval forecasts, from the hits (1 when the outcome fell outside the interval).

    Conditional coverage (chi-squared, 2 degrees of freedom) is the sum of unconditional coverage (Kupiec: the
    share of misses is 1 - coverage) and independence (a miss today does not make a miss tomorrow more likely,
    a first-order Markov chain); both parts are in ``details``. Independence is for one-step forecasts."""
    if not 0 < coverage < 1:
        raise SpecificationError("coverage must be between 0 and 1")
    (h,) = _aligned(hits)
    h = h.astype(int)
    if not set(np.unique(h)) <= {0, 1}:
        raise SpecificationError("hits must be 0 (inside the interval) or 1 (outside)")
    n = len(h)
    if n < 6:
        raise SpecificationError("the Christoffersen test needs at least six forecasts")
    p, n1 = 1 - coverage, int(h.sum())
    n0, pi = n - n1, n1 / n
    lr_uc = -2 * (_xlogy(n0, 1 - p) + _xlogy(n1, p) - _xlogy(n0, 1 - pi) - _xlogy(n1, pi))
    prev, cur = h[:-1], h[1:]
    n00, n01 = int(np.sum((prev == 0) & (cur == 0))), int(np.sum((prev == 0) & (cur == 1)))
    n10, n11 = int(np.sum((prev == 1) & (cur == 0))), int(np.sum((prev == 1) & (cur == 1)))
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi2 = (n01 + n11) / (n - 1)
    lr_ind = -2 * (_xlogy(n00 + n10, 1 - pi2) + _xlogy(n01 + n11, pi2)
                   - _xlogy(n00, 1 - pi01) - _xlogy(n01, pi01) - _xlogy(n10, 1 - pi11) - _xlogy(n11, pi11))
    lr_uc, lr_ind = max(float(lr_uc), 0.0), max(float(lr_ind), 0.0)
    stat = lr_uc + lr_ind
    return TestResult("Christoffersen", f"the {coverage:.0%} intervals have the right coverage and their misses are independent", stat, 2,
                      float(sps.chi2.sf(stat, 2)), "chi2",
                      details={"misses": n1, "observations": n, "miss rate": pi, "expected miss rate": p,
                               "coverage (Kupiec) statistic": lr_uc, "coverage p-value": float(sps.chi2.sf(lr_uc, 1)),
                               "independence statistic": lr_ind, "independence p-value": float(sps.chi2.sf(lr_ind, 1))})


def _loss_frame(losses: Mapping[str, Any]) -> pd.DataFrame:
    names = list(losses)
    columns = _aligned(*[losses[k] for k in names])
    return pd.DataFrame(dict(zip(names, columns)))


def _bootstrap_provenance(test: str, losses: pd.DataFrame, settings: Mapping[str, Any], seed: int, record: Any) -> Any:
    """Where a bootstrap comparison of losses comes from: the loss series, the test's settings and seed, and what arch
    was given and said."""
    from .._capture import fingerprint
    from .._version import __version__
    from ..adapters.base import with_backend
    from ..metadata import Provenance

    index = losses.index
    sample = (str(index[0]), str(index[-1])) if len(index) and not isinstance(index, pd.RangeIndex) else None
    prov = Provenance(data_vintages={}, spec={"losses": [str(c) for c in losses.columns]}, method={"test": test, "settings": dict(settings)},
                      sample=sample, nobs=len(losses), dropped=0, seed=seed, package_version=__version__,
                      data_fingerprint=fingerprint(losses))
    return with_backend(prov, record)


def _computed_by(record: Any) -> str:
    return f"Computed by {record.label} ({record.wraps})."


def _block(n: int, block: int | None, horizon: int) -> int:
    return int(block) if block else max(int(horizon), int(round(n ** (1 / 3))))


def spa(benchmark: Any, models: Mapping[str, Any], horizon: int = 1, reps: int = 1000, block: int | None = None, seed: int = 0) -> TestResult:
    """Hansen's (2005) test for superior predictive ability: does any of ``models`` have lower expected loss than
    the ``benchmark``? From the losses (one series per model), with the stationary bootstrap (arch's ``SPA``).

    ``details`` give the consistent p-value (the test), the lower and upper bounds (the upper is White's 2000
    Reality Check), and the model with the lowest average loss. ``block`` is the mean block length (default the
    larger of the horizon and n^(1/3))."""
    from ..adapters import require

    SPA = require("arch.bootstrap").SPA
    if not models:
        raise SpecificationError("give at least one model to compare with the benchmark")
    frame = _loss_frame({"__benchmark__": benchmark, **dict(models)})
    bench, rest = frame.pop("__benchmark__"), frame
    n = len(rest)
    size = _block(n, block, horizon)
    from ..adapters import arch as arch_adapter

    with arch_adapter.ADAPTER.run(arch_adapter.SPA) as run:
        test = run.call("model", SPA, bench.to_numpy(), rest.to_numpy(), block_size=size, reps=reps, seed=seed)
        run.call("fit", test.compute)
        pv = test.pvalues
    prov = _bootstrap_provenance("spa", frame.assign(benchmark=bench), {"horizon": horizon, "reps": reps, "block": size}, seed, run.record)
    d = bench.to_numpy()[:, None] - rest.to_numpy()  # positive: the model loses less than the benchmark
    sd = np.sqrt(np.array([newey_west(d[:, k] - d[:, k].mean(), size - 1)[0, 0] for k in range(d.shape[1])]) / n)
    tstats = np.where(sd > 0, d.mean(axis=0) / np.where(sd > 0, sd, 1), 0.0)
    best = rest.columns[int(np.argmin(rest.mean().to_numpy()))]
    return TestResult("Superior predictive ability (Hansen)", "no model is better than the benchmark", float(max(tstats.max(), 0.0)), None,
                      float(pv["consistent"]), "bootstrap",
                      details={"p-value, lower": float(pv["lower"]), "p-value, upper (Reality Check)": float(pv["upper"]),
                               "models": len(rest.columns), "lowest average loss": str(best), "observations": n,
                               "block length": size, "replications": reps}, provenance=prov)


def model_confidence_set(losses: Mapping[str, Any], alpha: float = 0.1, horizon: int = 1, reps: int = 1000, block: int | None = None,
                         seed: int = 0) -> Any:
    """The model confidence set of Hansen, Lunde and Nason (2011): the models that contain the best one with
    probability 1 - ``alpha``, from their losses (one series per model), with the stationary bootstrap (arch's
    ``MCS``, range statistic). A table: each model's average loss, its MCS p-value, and whether it is in the set."""
    from ..adapters import require
    from ..output import Table

    MCS = require("arch.bootstrap").MCS
    frame = _loss_frame(dict(losses))
    if frame.shape[1] < 2:
        raise SpecificationError("give at least two models")
    n = len(frame)
    size = _block(n, block, horizon)
    from ..adapters import arch as arch_adapter

    with arch_adapter.ADAPTER.run(arch_adapter.MCS) as run:
        mcs = run.call("model", MCS, frame, size=alpha, reps=reps, block_size=size, method="R", seed=seed)
        run.call("fit", mcs.compute)
        pv = mcs.pvalues.iloc[:, 0]
        included = set(mcs.included)
    order = frame.mean().sort_values().index
    rows = [[float(frame[m].mean()), float(pv[m]), "yes" if m in included else "no"] for m in order]
    return Table(["average loss", "MCS p-value", f"in the {1 - alpha:.0%} set"], rows, index=[str(m) for m in order],
                 title="Model confidence set (Hansen, Lunde and Nason)",
                 notes=(f"{len(included)} of {frame.shape[1]} models in the set; {n} forecasts, stationary bootstrap with mean block "
                        f"length {size} and {reps} replications.", _computed_by(run.record)))


def stepm(benchmark: Any, models: Mapping[str, Any], alpha: float = 0.05, horizon: int = 1, reps: int = 1000, block: int | None = None,
          seed: int = 0) -> Any:
    """Romano and Wolf's (2005) stepwise multiple test (StepM): which of ``models`` have lower expected loss than the
    ``benchmark``, controlling the familywise error rate at ``alpha``? From the losses, with the stationary bootstrap
    (arch's ``StepM``). A table: each model's average loss less the benchmark's, and whether it is superior."""
    from ..adapters import require
    from ..output import Table

    StepM = require("arch.bootstrap").StepM
    if not models:
        raise SpecificationError("give at least one model to compare with the benchmark")
    frame = _loss_frame({"__benchmark__": benchmark, **dict(models)})
    bench = frame.pop("__benchmark__")
    n = len(frame)
    size = _block(n, block, horizon)
    from ..adapters import arch as arch_adapter

    with arch_adapter.ADAPTER.run(arch_adapter.STEPM) as run:
        test = run.call("model", StepM, bench.to_numpy(), frame, size=alpha, block_size=size, reps=reps, seed=seed)
        run.call("fit", test.compute)
        superior = {str(m) for m in test.superior_models}
    diff = frame.sub(bench, axis=0).mean().sort_values()
    rows = [[float(diff[m]), "yes" if str(m) in superior else "no"] for m in diff.index]
    return Table(["average loss less the benchmark's", "superior"], rows, index=[str(m) for m in diff.index],
                 title="Superior models (Romano-Wolf StepM)",
                 notes=(f"{len(superior)} of {frame.shape[1]} models beat the benchmark, familywise error rate {alpha:.0%}; {n} forecasts, "
                        f"stationary bootstrap with mean block length {size} and {reps} replications.", _computed_by(run.record)))


# ---------------------------------------------------------------------------------------------
# asymmetric losses (Elliott and Timmermann, chapter 2)


def linlin_loss(errors: Any, tau: float = 0.5) -> np.ndarray:
    """Lin-lin (check, or pinball) loss of errors e = outcome - forecast: tau e when e >= 0, (tau - 1) e otherwise.
    tau > 0.5 makes under-forecasts costlier; the optimal forecast is the tau quantile. tau = 0.5 is half the absolute error."""
    if not 0 < tau < 1:
        raise SpecificationError("tau must be between 0 and 1")
    e = np.asarray(errors, float)
    return np.where(e >= 0, tau * e, (tau - 1) * e)


def linex_loss(errors: Any, a: float = 1.0) -> np.ndarray:
    """Linex loss (Varian 1975) of errors e = outcome - forecast: exp(a e) - a e - 1. With a > 0 under-forecasts
    (positive errors) cost exponentially and over-forecasts about linearly; a < 0 the reverse; near a = 0 it is
    a^2 e^2 / 2. For a normal forecast the optimal forecast is the mean plus a sd^2 / 2."""
    if a == 0:
        raise SpecificationError("the linex parameter a must not be zero (use squared loss)")
    e = np.asarray(errors, float)
    return np.exp(a * e) - a * e - 1


# ---------------------------------------------------------------------------------------------
# binary and directional forecasts (Elliott and Timmermann, chapter 12)


def _binary(probabilities: Any, outcomes: Any) -> tuple[np.ndarray, np.ndarray]:
    p, y = _aligned(probabilities, outcomes)
    if np.any((p < 0) | (p > 1)):
        raise SpecificationError("probability forecasts must be between 0 and 1")
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise SpecificationError("outcomes must be 0 or 1")
    return p, y


def brier_score(probabilities: Any, outcomes: Any) -> float:
    """The Brier score: the mean squared difference between the forecast probability and the 0/1 outcome (lower is better)."""
    p, y = _binary(probabilities, outcomes)
    return float(np.mean((p - y) ** 2))


def binary_log_score(probabilities: Any, outcomes: Any) -> float:
    """The average log probability given to what happened (higher is better; minus infinity for a certain miss)."""
    p, y = _binary(probabilities, outcomes)
    with np.errstate(divide="ignore"):
        return float(np.mean(np.log(np.where(y == 1, p, 1 - p))))


def roc_auc(probabilities: Any, outcomes: Any) -> float:
    """The area under the ROC curve: the probability that an event got a higher forecast than a non-event (ties
    count half); 0.5 is no skill, 1 perfect discrimination. The Mann-Whitney statistic over n1 n0."""
    p, y = _binary(probabilities, outcomes)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise SpecificationError("the ROC curve needs both events and non-events")
    ranks = sps.rankdata(np.concatenate([pos, neg]))
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def roc_curve(probabilities: Any, outcomes: Any) -> pd.DataFrame:
    """The ROC curve: for each threshold (the distinct forecasts, highest first), the false and true positive rates
    of forecasting an event when the probability is at or above it."""
    p, y = _binary(probabilities, outcomes)
    if not np.any(y == 1) or not np.any(y == 0):
        raise SpecificationError("the ROC curve needs both events and non-events")
    thresholds = np.unique(p)[::-1]
    tpr = [float(np.mean(p[y == 1] >= t)) for t in thresholds]
    fpr = [float(np.mean(p[y == 0] >= t)) for t in thresholds]
    return pd.DataFrame({"threshold": np.concatenate([[np.inf], thresholds]), "false positive rate": [0.0, *fpr],
                         "true positive rate": [0.0, *tpr]})


def kuipers_score(forecast_events: Any, outcomes: Any) -> float:
    """The Kuipers (Hanssen-Kuipers) score of 0/1 event forecasts: the hit rate less the false-alarm rate (0 is no skill, 1 perfect)."""
    f, y = _binary(forecast_events, outcomes)
    if not set(np.unique(f)) <= {0.0, 1.0}:
        raise SpecificationError("event forecasts must be 0 or 1 (threshold the probabilities first)")
    if not np.any(y == 1) or not np.any(y == 0):
        raise SpecificationError("the Kuipers score needs both events and non-events")
    return float(np.mean(f[y == 1]) - np.mean(f[y == 0]))


@api("pymetrix:binary_scores")
def binary_scores(probabilities: Any, outcomes: Any, threshold: float = 0.5, data: Any = None) -> Any:
    """Probability forecasts of a 0/1 outcome at a glance: Brier score (and the base rate's), log score, ROC area,
    and the hit rate, false-alarm rate and Kuipers score of forecasting an event when the probability is at least
    ``threshold``. With ``data`` (a dataset), ``probabilities`` and ``outcomes`` name its series, over its active sample."""
    from ..output import Table

    if data is not None:
        frame = data.in_sample()
        missing = [n for n in (probabilities, outcomes) if not isinstance(n, str) or n not in frame.columns]
        if missing:
            raise SpecificationError(f"with data, name its series: {', '.join(map(str, missing))} not found")
        probabilities, outcomes = frame[probabilities], frame[outcomes]
    p, y = _binary(probabilities, outcomes)
    events = (p >= threshold).astype(float)
    both = bool(np.any(y == 1) and np.any(y == 0))
    nan = float("nan")
    rows = [["observations", len(y)], ["share of events", float(y.mean())], ["Brier score", brier_score(p, y)],
            ["Brier score of the base rate", float(np.mean((y.mean() - y) ** 2))], ["log score", binary_log_score(p, y)],
            ["ROC area", roc_auc(p, y) if both else nan],
            ["hit rate", float(np.mean(events[y == 1])) if np.any(y == 1) else nan],
            ["false-alarm rate", float(np.mean(events[y == 0])) if np.any(y == 0) else nan],
            ["Kuipers score", kuipers_score(events, y) if both else nan]]
    return Table(["", "value"], rows, title="Probability forecast scores",
                 notes=(f"Events are forecast when the probability is at least {threshold:g}. Brier: lower is better (compare it with the "
                        "base rate's); log score and ROC area: higher is better (0.5 is no skill for the ROC area).",))


def pesaran_timmermann(actual: Any, forecast: Any) -> TestResult:
    """Pesaran and Timmermann's (1992) test of directional accuracy: do the forecasts get the sign of the outcome
    right more often than independent guesses with the same frequencies would? Standard normal, one-sided."""
    y, f = _aligned(actual, forecast)
    n = len(y)
    if n < 10:
        raise SpecificationError("the Pesaran-Timmermann test needs at least ten forecasts")
    up_y, up_f = (y > 0).astype(float), (f > 0).astype(float)
    p_hat = float(np.mean(up_y == up_f))
    py, pf = float(up_y.mean()), float(up_f.mean())
    p_star = py * pf + (1 - py) * (1 - pf)
    v_hat = p_star * (1 - p_star) / n
    v_star = (2 * py - 1) ** 2 * pf * (1 - pf) / n + (2 * pf - 1) ** 2 * py * (1 - py) / n + 4 * py * pf * (1 - py) * (1 - pf) / n**2
    if v_hat - v_star <= 0:
        raise SpecificationError("the outcomes or the forecasts all have the same sign: the test is not defined")
    stat = float((p_hat - p_star) / np.sqrt(v_hat - v_star))
    return TestResult("Pesaran-Timmermann", "the forecasts' signs are independent of the outcomes' (no directional skill)", stat, None,
                      float(sps.norm.sf(stat)), "normal",
                      details={"share of signs right": p_hat, "expected under independence": p_star, "observations": n, "one-sided": "yes"})
