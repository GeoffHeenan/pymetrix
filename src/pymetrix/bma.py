"""Bayesian model averaging over linear regressions (Elliott and Timmermann 2016, section 14.6).

    bma = BMA("dlog(gdp)", ["dlog(gdp)[-1]", "spread[-1]", "pmi[-1]", "dlog(credit)[-1]"]).fit(data)
    bma.summary()                     # posterior inclusion probabilities and averaged coefficients
    bma.models()                      # the most probable models
    bma.forecast(data, ("2025Q1", "2025Q4"))     # model-averaged forecasts (regressors known in the span)
    backtest(bma, data, ("2015Q1", "2022Q4"), horizon=1)

Every subset of the candidate regressors is a model; each has the intercept and the regressors in ``always``.
Priors: Zellner's g-prior on the candidates' coefficients, flat priors on the intercept and the ``always``
regressors, p(sigma) proportional to 1/sigma, and equal prior probability for every model. The g-prior gives
the marginal likelihood in closed form (Liang, Paulo, Molina, Clyde and Berger 2008):

    p(y | M) proportional to (1 + g)^((n - q - k) / 2) (1 + g (1 - R^2))^(-(n - q) / 2)

with k the model's candidates, q the always-included terms (with the intercept), and R^2 that of the candidates
after the always-included terms are partialled out. The default g is the benchmark of Fernandez, Ley and Steel
(2001), max(n, K^2) for K candidates. Within a model, the candidates' coefficients are the least-squares ones
shrunk by g / (1 + g), with Student-t posteriors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from pymetrix._lazy import stats as sps

from ._api import api, renamed_arguments
from .errors import SpecificationError
from .forms import Widget
from .output import Table
from .plugins import register
from .results import deprecated_alias

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["BMA", "FittedBMA", "log_marginal_likelihood"]

MAX_CANDIDATES = 15  # 2^15 = 32,768 models, each a small least-squares fit


def log_marginal_likelihood(r2: Any, n: int, k: Any, g: float, q: int = 1) -> np.ndarray:
    """log p(y | M) up to a constant common to all models: ((n - q - k) / 2) log(1 + g) - ((n - q) / 2) log(1 + g (1 - R^2))."""
    r2, k = np.asarray(r2, float), np.asarray(k, float)
    return 0.5 * (n - q - k) * np.log1p(g) - 0.5 * (n - q) * np.log1p(g * (1 - r2))


@dataclass(frozen=True)
class BMA:
    """Bayesian model averaging over every subset of candidate regressors, with Zellner's g-prior.

    Each model has an intercept, the ``always`` regressors and a subset of the ``candidates``; all models are equally
    likely a priori. ``g`` 0 means the benchmark prior, max(n, K^2)."""

    label: ClassVar[str] = "Bayesian model averaging"
    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)

    outcome: Annotated[str, Widget("text", help="the series to explain or forecast, e.g. dlog(gdp)")]  # was target
    candidates: Annotated[tuple[str, ...], Widget("text", help="regressors that may or may not belong, e.g. spread[-1], pmi[-1]")]
    always: Annotated[tuple[str, ...], Widget("text", advanced=True, help="regressors in every model (besides the intercept)")] = ()
    g: Annotated[float, Widget("number", min=0, max=1e9, advanced=True, help="the g-prior's g; 0 = the benchmark max(n, K^2)")] = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "always", tuple(self.always))
        if not self.candidates:
            raise SpecificationError("give at least one candidate regressor")
        if len(self.candidates) > MAX_CANDIDATES:
            raise SpecificationError(f"at most {MAX_CANDIDATES} candidates: every subset is a model ({2 ** MAX_CANDIDATES:,} models)")
        if len(set(self.candidates) | set(self.always)) < len(self.candidates) + len(self.always):
            raise SpecificationError("a regressor is listed twice")
        if self.g < 0:
            raise SpecificationError("g must be positive (or 0 for the benchmark)")

    def _frame(self, data: Dataset) -> pd.DataFrame:
        columns = {"__y__": data.eval(self.outcome)}
        for name in (*self.always, *self.candidates):
            columns[name] = data.eval(name)
        return pd.DataFrame(columns)

    @api("pymetrix:bma.fit")
    def fit(self, data: Dataset, span: Any = None) -> FittedBMA:
        """Fit every model on the periods of ``span`` (first, last) and weight each by its posterior probability."""
        from ._capture import capture, provenance_for

        if data.kind != "time_series" and span is not None:
            raise SpecificationError("a sample span needs time-series data")
        frame = self._frame(data)[np.asarray(data.sample_mask(span))].dropna()
        n, K, a = len(frame), len(self.candidates), len(self.always)
        q = 1 + a
        if n < q + K + 3:
            raise SpecificationError(f"{n} complete observations: too few for {K} candidates")
        g = float(self.g) or float(max(n, K**2))
        y = frame["__y__"].to_numpy(float)
        Z = np.column_stack([np.ones(n), frame[list(self.always)].to_numpy(float)]) if a else np.ones((n, 1))
        X = frame[list(self.candidates)].to_numpy(float)
        with capture() as caught:
            # partial out the intercept and the always-included regressors (Frisch-Waugh)
            Zp = np.linalg.pinv(Z)
            y_r = y - Z @ (Zp @ y)
            X_r = X - Z @ (Zp @ X)
            sst = float(y_r @ y_r)
            models, rows = [], []
            for size in range(K + 1):
                for subset in combinations(range(K), size):
                    cols = list(subset)
                    if cols:
                        Xs = X_r[:, cols]
                        b, *_ = np.linalg.lstsq(Xs, y_r, rcond=None)
                        r2 = 1 - float(np.sum((y_r - Xs @ b) ** 2)) / sst
                    else:
                        b, r2 = np.zeros(0), 0.0
                    models.append(cols)
                    rows.append((r2, len(cols), b))
        r2 = np.array([r[0] for r in rows])
        k = np.array([r[1] for r in rows])
        logml = log_marginal_likelihood(r2, n, k, g, q)
        post = np.exp(logml - logsumexp(logml))
        shrink = g / (1 + g)
        coef = np.zeros((len(models), K))
        var = np.zeros((len(models), K))
        df = n - q
        for m, (cols, (r2_m, _, b)) in enumerate(zip(models, rows)):
            if not cols:
                continue
            Xs = X_r[:, cols]
            scale = shrink * sst * (1 - shrink * r2_m) / df  # Student-t scale (Liang et al. 2008)
            cov = scale * np.linalg.inv(Xs.T @ Xs) * df / (df - 2)
            coef[m, cols] = shrink * b
            var[m, cols] = np.diag(cov)
        mean = post @ coef
        sd = np.sqrt(np.maximum(post @ (var + coef**2) - mean**2, 0))
        pip = np.array([post[[m for m, cols in enumerate(models) if j in cols]].sum() for j in range(K)])
        prov = provenance_for(self, data, [], frame.index, frame, caught,
                              {"type": "Bayesian model averaging", "g": g, "models": len(models), "prior": "g-prior, equal model probabilities"},
                              span=span)
        return FittedBMA(self, models, post, coef, var, mean, sd, pip, g, n, frame, prov)


@dataclass(frozen=True, eq=False)
class FittedBMA:
    """A Bayesian model average: every model's posterior probability and coefficients, and the averages."""

    spec: BMA
    models: list[list[int]] = field(repr=False)  # the candidates in each model (indices)
    posterior: np.ndarray = field(repr=False)  # each model's posterior probability
    coefs: np.ndarray = field(repr=False)  # each model's posterior mean of the candidates' coefficients (0 when left out)
    variances: np.ndarray = field(repr=False)
    mean: np.ndarray = field(repr=False)  # the model-averaged posterior means
    sd: np.ndarray = field(repr=False)
    inclusion: np.ndarray = field(repr=False)  # posterior inclusion probabilities
    g: float = 0.0
    nobs: int = 0
    frame: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    provenance: Any = None

    @property
    def dependent(self) -> str:
        return self.spec.outcome

    target = deprecated_alias("dependent", "target", why="the dependent variable; target is its old name")

    def summary(self) -> Table:
        """Posterior inclusion probabilities and the model-averaged coefficients (means and sds), most probable first."""
        order = np.argsort(-self.inclusion, kind="stable")
        rows = [[float(self.inclusion[j]), float(self.mean[j]), float(self.sd[j]),
                 float(self.mean[j] / self.inclusion[j]) if self.inclusion[j] > 0 else float("nan")] for j in order]
        best = int(np.argmax(self.posterior))
        notes = (f"{len(self.models):,} models, {self.nobs} observations; g = {self.g:g}; equal prior probability for every model.",
                 "Coefficient means and sds average over all models (0 where a regressor is left out); the last column is the "
                 "mean in the models that include it.",
                 f"Most probable model: {', '.join(self.spec.candidates[j] for j in self.models[best]) or 'no candidates'} "
                 f"(probability {self.posterior[best]:.3f}).")
        if self.spec.always:
            notes += (f"In every model: the intercept and {', '.join(self.spec.always)}.",)
        return Table(["inclusion probability", "mean", "sd", "mean if included"], rows,
                     index=[self.spec.candidates[j] for j in order], index_name="regressor",
                     title=f"Bayesian model averaging: {self.dependent}", notes=notes)

    def models_table(self, top: int = 10) -> Table:
        """The ``top`` most probable models: their posterior probabilities and regressors."""
        order = np.argsort(-self.posterior, kind="stable")[:top]
        K = len(self.spec.candidates)
        rows = [[float(self.posterior[m]), len(self.models[m])] + ["x" if j in self.models[m] else "" for j in range(K)] for m in order]
        return Table(["probability", "regressors", *self.spec.candidates], rows, index=[str(i + 1) for i in range(len(order))],
                     index_name="rank", title=f"The {len(order)} most probable models",
                     notes=(f"Cumulative probability of these models: {float(self.posterior[order].sum()):.3f}.",))

    def _parts(self, data: Dataset, span: Any) -> tuple[pd.Index, np.ndarray, np.ndarray]:
        """Each period's regressors in the span: the always-included block (with the intercept) and the candidates."""
        frame = self.spec._frame(data)
        frame = frame[np.asarray(data.sample_mask(span))]
        keep = frame.drop(columns="__y__").notna().all(axis=1)
        frame = frame[keep]
        Z = np.column_stack([np.ones(len(frame)), frame[list(self.spec.always)].to_numpy(float)]) if self.spec.always else np.ones((len(frame), 1))
        return frame.index, Z, frame[list(self.spec.candidates)].to_numpy(float)

    @api()
    def forecast(self, data: Dataset, span: tuple[Any, Any], alpha: float = 0.05, dynamic: bool = False) -> Any:
        """Model-averaged forecasts for the periods of ``span`` whose regressors are known (like an equation's static
        forecasts): each model's predictive Student-t, averaged with the posterior probabilities. Columns: forecast, se
        (of the mixture), lower and upper (a normal approximation), and actual where known."""
        from .data import Dataset as _Dataset

        est = self.frame
        y = est["__y__"].to_numpy(float)
        Zf = np.column_stack([np.ones(len(est)), est[list(self.spec.always)].to_numpy(float)]) if self.spec.always else np.ones((len(est), 1))
        Xf = est[list(self.spec.candidates)].to_numpy(float)
        Zp = np.linalg.pinv(Zf)
        y_r, X_r = y - Zf @ (Zp @ y), Xf - Zf @ (Zp @ Xf)
        sst = float(y_r @ y_r)
        n, q = len(est), Zf.shape[1]
        df = n - q
        shrink = self.g / (1 + self.g)
        index, Z, X = self._parts(data, span)
        if not len(index):
            raise SpecificationError("no period of the span has all the regressors")
        means = np.zeros((len(self.models), len(index)))
        variances = np.zeros_like(means)
        ZtZinv = np.linalg.inv(Zf.T @ Zf)
        for m, cols in enumerate(self.models):
            beta = self.coefs[m, cols]
            gamma = Zp @ (y - Xf[:, cols] @ beta) if cols else Zp @ y  # the always-included terms given the candidates
            means[m] = Z @ gamma + (X[:, cols] @ beta if cols else 0.0)
            r2 = 1 - float(np.sum((y_r - X_r[:, cols] @ np.linalg.lstsq(X_r[:, cols], y_r, rcond=None)[0]) ** 2)) / sst if cols else 0.0
            s2 = sst * (1 - shrink * r2) / df
            lev = np.einsum("ij,jk,ik->i", Z, ZtZinv, Z)
            if cols:
                Xc = X[:, cols] - Z @ (Zp @ Xf[:, cols])  # the forecast period's candidates, net of the always-included block
                lev = lev + shrink * np.einsum("ij,jk,ik->i", Xc, np.linalg.inv(X_r[:, cols].T @ X_r[:, cols]), Xc)
            variances[m] = s2 * (1 + lev) * df / (df - 2)
        mean = self.posterior @ means
        se = np.sqrt(np.maximum(self.posterior @ (variances + means**2) - mean**2, 0))
        z = sps.norm.ppf(1 - alpha / 2)
        frame = pd.DataFrame({"forecast": mean, "se": se, "lower": mean - z * se, "upper": mean + z * se}, index=index)
        actual = data.eval(self.dependent).reindex(index)
        frame["actual"] = actual.to_numpy(float)
        return _Dataset(frame, data.axes)

    def plot(self) -> Any:
        """Posterior inclusion probabilities."""
        from .figures import Bars, Figure, Panel, Rule

        heights = pd.Series(self.inclusion, index=list(self.spec.candidates))
        return Figure((Panel((Bars(heights, "inclusion probability"), Rule(0.5, "--")), title=f"Posterior inclusion probabilities: {self.dependent}",
                             legend=False),))

    def __repr__(self) -> str:
        return f"<FittedBMA {self.dependent}: {len(self.spec.candidates)} candidates, {len(self.models):,} models, {self.nobs} observations>"


BMA.__init__ = renamed_arguments(target="outcome")(BMA.__init__)  # type: ignore[method-assign]
register("models", BMA)
