"""Smooth-transition autoregressions: two AR regimes blended by a smooth function of a lag of the series
(Teräsvirta 1994; van Dijk, Teräsvirta and Franses 2002).

    star = SmoothTransitionAR("dgdp", lags=2, delay=1).fit(data)              # logistic (LSTAR)
    SmoothTransitionAR("dgdp", lags=2, transition="exponential").fit(data)    # exponential (ESTAR)
    star.summary(); star.transition_path(); star.forecast(data, ("2025Q1", "2025Q4")); star.plot()

y_t = phi_1' x_t (1 - G(s_t)) + phi_2' x_t G(s_t) + e_t, with x_t = (1, y_(t-1), ..., y_(t-p)) and s_t = y_(t-d). The
transition G goes from 0 to 1: logistic, G = 1 / (1 + exp(-gamma (s - c) / sd(s))), for a change between low and high
values of s; exponential, G = 1 - exp(-gamma ((s - c) / sd(s))^2), for values near c against those far from it. gamma
says how fast the change is (a large gamma makes LSTAR a threshold model at c); it is scaled by the standard deviation
of s, so it does not depend on the series' units.

**Estimation.** For given (gamma, c) the model is linear in the phis, so they are least squares and the sum of squared
residuals is a function of (gamma, c) alone. It is minimised over a grid (gamma from 0.5 to 100, c between the 15% and
85% quantiles of s), then refined by Nelder-Mead. Standard errors are Gauss-Newton's, sigma^2 (J'J)^-1, with J the
derivatives of the fitted values with respect to every parameter (gamma and c by central differences); gamma's is
unreliable when it is large, as the likelihood flattens. ``terasvirta`` tests linearity against this model first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
from scipy import optimize

from ._api import api
from .errors import SpecificationError
from .forms import Widget
from .output import Table
from .results import deprecated_alias
from ._random import generator as make_rng

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["FittedSmoothTransitionAR", "SmoothTransitionAR", "transition"]


def transition(s: Any, gamma: float, c: float, scale: float, kind: str = "logistic") -> np.ndarray:
    """The transition function G(s) in [0, 1]."""
    z = (np.asarray(s, float) - c) / scale
    if kind == "logistic":
        return 0.5 * (1 + np.tanh(gamma * z / 2))  # = 1 / (1 + exp(-gamma z)), without overflow
    return 1 - np.exp(-gamma * z**2)


def _concentrated(y: np.ndarray, X: np.ndarray, s: np.ndarray, gamma: float, c: float, scale: float, kind: str) -> tuple[float, np.ndarray]:
    G = transition(s, gamma, c, scale, kind)[:, None]
    Z = np.hstack([X * (1 - G), X * G])
    b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e = y - Z @ b
    return float(e @ e), b


@dataclass(frozen=True)
class SmoothTransitionAR:
    """A two-regime smooth-transition autoregression (LSTAR or ESTAR), by nonlinear least squares."""

    code_positional: ClassVar[int] = 1  # generated code gives the series by position: SmoothTransitionAR('y')
    label: ClassVar[str] = "Smooth-transition autoregression (STAR)"
    topics: ClassVar[tuple[str, ...]] = ("smooth-transition",)

    series: Annotated[str, Widget("text", help="the series, e.g. GDP growth")]
    lags: Annotated[int, Widget("number", min=1, max=12, help="the autoregressive order in each regime")] = 1
    delay: Annotated[int, Widget("number", min=0, max=12, help="the lag of the series that drives the transition; 0 = choose (1 to lags)")] = 1
    transition: Annotated[Literal["logistic", "exponential"], Widget("choice", help="logistic (LSTAR) or exponential (ESTAR)")] = "logistic"

    def __post_init__(self) -> None:
        if self.lags < 1:
            raise SpecificationError("lags must be 1 or more")
        if self.delay < 0 or self.delay > 12:
            raise SpecificationError("the delay must be 0 (choose) or a lag up to 12")
        if self.transition not in ("logistic", "exponential"):
            raise SpecificationError("transition is 'logistic' or 'exponential'")

    def _delays(self) -> list[int]:
        return [self.delay] if self.delay else list(range(1, self.lags + 1))

    def _arrays(self, data: Dataset, span: Any) -> tuple[pd.Index, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
        from .nonlinear import ThresholdAR

        index, y, X, q, _ = ThresholdAR(self.series, self.lags, self.delay)._arrays(data, span)  # the same sample rules
        return index, y, X, q

    @api("pymetrix:smooth_transition_ar.fit")
    def fit(self, data: Dataset, span: Any = None) -> FittedSmoothTransitionAR:
        """Estimate by least squares concentrated on (gamma, c), over a grid and then by Nelder-Mead (and over the
        delays, when the delay is 0)."""
        index, y, X, q = self._arrays(data, span)
        k = X.shape[1]
        if len(y) < 2 * k + 20:
            raise SpecificationError(f"{len(y)} observations: too few for two AR({self.lags}) regimes")
        best: tuple[float, int, float, float] | None = None
        for d, s in q.items():
            scale = float(np.std(s, ddof=1)) or 1.0
            cs = np.quantile(s, np.linspace(0.15, 0.85, 15)) if self.transition == "logistic" else np.quantile(s, np.linspace(0.2, 0.8, 13))
            for g in np.geomspace(0.5, 100, 20):
                for c in cs:
                    ssr = _concentrated(y, X, s, g, c, scale, self.transition)[0]
                    if best is None or ssr < best[0]:
                        best = (ssr, d, float(np.log(g)), float(c))
        _, d, lg0, c0 = best
        s = q[d]
        scale = float(np.std(s, ddof=1)) or 1.0
        lo, hi = float(np.quantile(s, 0.05)), float(np.quantile(s, 0.95))

        def objective(p: np.ndarray) -> float:
            lg, c = p
            if not (np.log(0.01) < lg < np.log(1000.0)) or not lo <= c <= hi:
                return 1e300
            return _concentrated(y, X, s, np.exp(lg), c, scale, self.transition)[0]

        found = optimize.minimize(objective, [lg0, c0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 4000})
        gamma, c = float(np.exp(found.x[0])), float(found.x[1])
        ssr, b = _concentrated(y, X, s, gamma, c, scale, self.transition)
        n, m = len(y), 2 * k + 2
        sigma2 = ssr / (n - m)

        def fitted(theta: np.ndarray) -> np.ndarray:
            G = transition(s, theta[-2], theta[-1], scale, self.transition)
            return (X @ theta[:k]) * (1 - G) + (X @ theta[k:2 * k]) * G

        theta = np.concatenate([b, [gamma, c]])
        G = transition(s, gamma, c, scale, self.transition)[:, None]
        J = np.hstack([X * (1 - G), X * G, np.zeros((n, 2))])
        for j in (2 * k, 2 * k + 1):
            h = 1e-6 * max(abs(theta[j]), 1.0)
            up, dn = theta.copy(), theta.copy()
            up[j] += h
            dn[j] -= h
            J[:, j] = (fitted(up) - fitted(dn)) / (2 * h)
        cov = sigma2 * np.linalg.pinv(J.T @ J)
        ssr_linear = float(np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2))
        return FittedSmoothTransitionAR(self, data[self.series], d, gamma, c, scale, b[:k], b[k:], cov, float(sigma2), float(ssr),
                                        ssr_linear, index, s, span, bool(found.success))


@dataclass(frozen=True, eq=False)
class FittedSmoothTransitionAR:
    """An estimated smooth-transition autoregression."""

    spec: SmoothTransitionAR
    data: pd.Series = field(repr=False)
    delay: int = 1
    gamma: float = 1.0
    c: float = 0.0
    scale: float = 1.0
    phi1: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))  # the regime where G = 0
    phi2: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))  # where G = 1
    cov: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))  # phi1, phi2, gamma, c
    sigma2: float = 0.0
    ssr: float = 0.0
    ssr_linear: float = 0.0
    index: Any = field(default=None, repr=False)
    switch: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))  # s_t over the sample
    span: Any = None
    converged: bool = True

    stage = "estimate"

    @property
    def dependent(self) -> str:
        return self.spec.series

    target = deprecated_alias("dependent", "target", why="the dependent variable; target is its old name")

    @property
    def variables(self) -> list[str]:
        return [self.spec.series]

    def transition_path(self) -> pd.Series:
        """G(s_t) over the sample: the weight on the second regime in each period."""
        return pd.Series(transition(self.switch, self.gamma, self.c, self.scale, self.spec.transition), index=self.index, name="transition")

    def summary(self) -> Table:
        k = len(self.phi1)
        names = ["constant"] + [f"{self.dependent}[-{j}]" for j in range(1, self.spec.lags + 1)]
        se = np.sqrt(np.diag(self.cov))
        rows = [[float(self.phi1[j]), float(se[j]), float(self.phi2[j]), float(se[k + j])] for j in range(k)]
        rows += [[self.gamma, float(se[2 * k]), np.nan, np.nan], [self.c, float(se[2 * k + 1]), np.nan, np.nan]]
        kind = "logistic (LSTAR)" if self.spec.transition == "logistic" else "exponential (ESTAR)"
        regimes = ("G = 0", "G = 1")
        notes = (f"{kind} transition in {self.dependent}[-{self.delay}]; gamma is scaled by the switching variable's sd ({self.scale:.4g}).",
                 f"Error sd {np.sqrt(self.sigma2):.6g}; sum of squared residuals {self.ssr:.6g} against {self.ssr_linear:.6g} for the linear AR.",
                 "Gauss-Newton standard errors; gamma's is unreliable when gamma is large." + ("" if self.converged else
                                                                                            " The optimiser did not converge."))
        return Table([f"regime {regimes[0]}", "std err", f"regime {regimes[1]}", "std err "], rows, index=names + ["gamma", "c"],
                     index_name="coefficient", title=f"Smooth-transition autoregression for {self.dependent}", notes=notes)

    def predictive_draws(self, data: Dataset, span: tuple[Any, Any], draws: int | None = None, seed: int = 0) -> tuple[pd.PeriodIndex, np.ndarray]:
        """Simulated forecast paths (draws, periods, 1): each draw moves through the transition with its own shocks."""
        from .data.sample import as_span

        freq = self.data.index.freq
        span = as_span(span)
        start, end = pd.Period(str(span[0]), freq=freq), pd.Period(str(span[1]), freq=freq)
        periods = pd.period_range(start, end, freq=freq, name=self.data.index.name or "time")
        p, d = self.spec.lags, self.delay
        m = max(p, d)
        history = data[self.dependent].reindex(pd.period_range(start - m, start - 1, freq=freq)).to_numpy(float)
        if not np.all(np.isfinite(history)):
            raise SpecificationError(f"data for the {m} periods before {start} are needed to start the forecast")
        n = int(draws or 2000)
        rng = make_rng(seed)
        hist = np.tile(history, (n, 1))
        paths = np.empty((n, len(periods)))
        sd = np.sqrt(self.sigma2)
        for i in range(len(periods)):
            x = np.column_stack([np.ones(n)] + [hist[:, -j] for j in range(1, p + 1)])
            G = transition(hist[:, -d], self.gamma, self.c, self.scale, self.spec.transition)
            new = (x @ self.phi1) * (1 - G) + (x @ self.phi2) * G + sd * rng.standard_normal(n)
            paths[:, i] = new
            hist = np.column_stack([hist, new])
        return periods, paths[:, :, None]

    @api()
    def forecast(self, data: Dataset, span: tuple[Any, Any], alpha: float = 0.05, dynamic: bool = True, draws: int = 2000, seed: int = 0) -> Any:
        """Forecasts by simulation: mean, sd and the 1 - alpha interval from the simulated quantiles."""
        from .data import Dataset as _Dataset

        if not dynamic:
            raise SpecificationError("smooth-transition AR forecasts are dynamic")
        periods, paths = self.predictive_draws(data, span, draws, seed)
        paths = paths[:, :, 0]
        lo, hi = np.quantile(paths, [alpha / 2, 1 - alpha / 2], axis=0)
        frame = pd.DataFrame({"forecast": paths.mean(axis=0), "se": paths.std(axis=0, ddof=1), "lower": lo, "upper": hi}, index=periods)
        frame["actual"] = data[self.dependent].reindex(periods).to_numpy(float)
        return _Dataset(frame, data.axes)

    def plot(self) -> Any:
        from .figures import Figure, Line, Panel, Scatter

        G = self.transition_path()
        order = np.argsort(self.switch)
        curve = pd.Series(G.to_numpy()[order], index=pd.Index(self.switch[order], name=f"{self.dependent}[-{self.delay}]"), name="G")
        return Figure((Panel((Line(self.data, self.dependent, color="#222222"),), title=self.dependent),
                       Panel((Line(G, "weight on the second regime"),), title="The transition over time"),
                       Panel((Line(curve, "G"), Scatter(pd.Series(self.switch), pd.Series(G.to_numpy()), "periods")),
                             title=f"G against {self.dependent}[-{self.delay}]")))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.summary().to_frame()

    def __repr__(self) -> str:
        return f"<FittedSmoothTransitionAR {self.dependent}: {self.spec.transition}, gamma {self.gamma:.4g}, c {self.c:.4g}>"
