"""Factor models and portfolios.

    fm = factor_model(data, ["a", "b", "c"], ["mkt", "smb", "hml"], rf="rf")   # alphas, betas, the GRS test
    fm.rolling_betas(window=60)
    port = optimize_portfolio(data, ["a", "b", "c"], objective="max_sharpe", rf=0.0001)
    port.frontier(30); port.plot()

**Factor models.** Each asset's excess return is regressed on the factors (themselves excess returns, as for
Fama-French factors), with Newey-West standard errors. The Gibbons, Ross and Shanken (1989) test asks whether every
alpha is zero: W = (T - N - L) / N * a'S^-1 a / (1 + m'O^-1 m) ~ F(N, T - N - L), with S the residuals' covariance,
m the factors' means and O their covariance (both covariances divided by T). With one asset it is the square of
alpha's t statistic with classical standard errors.

**Portfolios.** Weights summing to one that minimise variance, maximise the Sharpe ratio over ``rf``, or minimise
variance at a target mean return, by scipy's SLSQP: long only by default, or within ``bounds``. The mean and
covariance are the sample's, or given (``cov`` may be a matrix, or a fitted multivariate GARCH, whose forecast for
the next period is used). Returns and the risk-free rate are per period, as in the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats as sps

from pymetrix.extension import api, call_migration, renamed_arguments
from pymetrix.errors import SpecificationError
from pymetrix.output import Table
from pymetrix.extension import plugin_context
from . import PLUGIN

if TYPE_CHECKING:
    from pymetrix.data import Dataset
    from pymetrix.figures import Figure


def grs(alpha: np.ndarray, resid: np.ndarray, factors: np.ndarray) -> tuple[float, float, int, int]:
    """The Gibbons-Ross-Shanken F statistic, its p-value and degrees of freedom."""
    T, N = resid.shape
    L = factors.shape[1]
    S = resid.T @ resid / T  # maximum likelihood, as the exact F distribution needs
    m = factors.mean(axis=0)
    O = np.atleast_2d(np.cov(factors.T, ddof=0))
    W = (T - N - L) / N * float(alpha @ np.linalg.solve(S, alpha)) / (1 + float(m @ np.linalg.solve(O, m)))
    return W, float(sps.f.sf(W, N, T - N - L)), N, T - N - L


@dataclass(frozen=True, eq=False)
class FactorModel:
    """Time-series factor regressions of several assets, with the GRS test of their alphas."""

    assets: tuple[str, ...]
    factors: tuple[str, ...]
    fits: dict[str, Any] = field(repr=False)  # asset -> FittedEquation
    excess: pd.DataFrame = field(repr=False)  # the assets' excess returns
    factor_values: pd.DataFrame = field(repr=False)
    test: tuple[float, float, int, int] = (np.nan, np.nan, 0, 0)

    stage = "finance"

    def coefficients_table(self) -> Table:
        cols, rows = [], []
        for name in ("const", *self.factors):
            cols += [f"{'alpha' if name == 'const' else name}", "t"]
        for a in self.assets:
            f = self.fits[a]
            p = dict(zip(f.params.names, zip(f.params.values, f.params.se)))
            row = []
            for name in ("const", *self.factors):
                v, s = p[name]
                row += [float(v), float(v / s) if s > 0 else np.nan]
            rows.append(row + [float(f.stats.get("r2", np.nan))])
        return Table(cols + ["R2"], rows, index=list(self.assets), index_name="asset", title="Factor model",
                     notes=("Excess returns on the factors; t statistics from Newey-West standard errors.",
                            f"GRS test that every alpha is zero: F({self.test[2]}, {self.test[3]}) = {self.test[0]:.4f}, "
                            f"p = {self.test[1]:.4f}."))

    def summary(self) -> Table:
        return self.coefficients_table()

    def rolling_betas(self, window: int = 60) -> pd.DataFrame:
        """Each asset's betas by least squares on the last ``window`` periods, through the sample."""
        F = np.column_stack([np.ones(len(self.factor_values)), self.factor_values.to_numpy(float)])
        out = {}
        for a in self.assets:
            y = self.excess[a].to_numpy(float)
            betas = np.full((len(y), len(self.factors)), np.nan)
            for t in range(window, len(y) + 1):
                betas[t - 1] = np.linalg.lstsq(F[t - window:t], y[t - window:t], rcond=None)[0][1:]
            for j, f in enumerate(self.factors):
                out[f"{a} on {f}"] = betas[:, j]
        return pd.DataFrame(out, index=self.excess.index)

    def plot(self) -> Figure:
        from pymetrix.figures import Figure, Line, Panel

        window = max(20, min(60, len(self.excess) // 4))
        rb = self.rolling_betas(window)
        return Figure((Panel(tuple(Line(rb[c], c) for c in rb.columns), title=f"Rolling betas ({window} periods)"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return self.coefficients_table().to_frame()

    def __repr__(self) -> str:
        return f"<FactorModel of {', '.join(self.assets)} on {', '.join(self.factors)}: GRS p = {self.test[1]:.4f}>"


def _solve(objective: str, mu: np.ndarray, cov: np.ndarray, rf: float, target_return: float | None, bounds: list[tuple[float, float]]) -> np.ndarray:
    n = len(mu)
    cons: list[dict[str, Any]] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1, "jac": lambda w: np.ones(n)}]
    if objective == "target_return":
        if target_return is None:
            raise SpecificationError("give the target mean return (per period), target_return=, for objective='target_return'")
        cons.append({"type": "eq", "fun": lambda w: w @ mu - target_return, "jac": lambda w: mu})

    if objective == "max_sharpe":
        def f(w: np.ndarray) -> float:
            sd = np.sqrt(w @ cov @ w)
            return -(w @ mu - rf) / sd if sd > 0 else 1e12

        def jac(w: np.ndarray) -> np.ndarray:  # the Sharpe ratio's gradient, exactly
            Sw = cov @ w
            sd = np.sqrt(w @ Sw)
            return -(mu * sd - (w @ mu - rf) * Sw / sd) / sd**2
    else:
        def f(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        def jac(w: np.ndarray) -> np.ndarray:  # type: ignore[misc]
            return 2 * cov @ w
    start = np.full(n, 1 / n)
    found = optimize.minimize(f, start, jac=jac, method="SLSQP", bounds=bounds, constraints=cons,
                              options={"maxiter": 2000, "ftol": 1e-14})
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    feasible = (abs(found.x.sum() - 1) < 1e-8 and np.all(found.x >= lo - 1e-9) and np.all(found.x <= hi + 1e-9)
                and all(abs(c["fun"](found.x)) < 1e-8 for c in cons))
    if not found.success and not (feasible and found.status == 8):  # 8: the line search hit numerical precision at a solution
        raise SpecificationError(f"the portfolio could not be found: {found.message} (is the target reachable within the bounds?)")
    return np.where(np.abs(found.x) < 1e-10, 0.0, found.x)  # the optimiser's rounding, not a holding


@dataclass(frozen=True, eq=False)
class Portfolio:
    """Optimal portfolio weights, with their mean, volatility and Sharpe ratio, and the efficient frontier."""

    assets: tuple[str, ...]
    objective: str
    weights: np.ndarray
    mu: np.ndarray = field(repr=False)
    cov: np.ndarray = field(repr=False)
    rf: float = 0.0
    bounds: tuple[tuple[float, float], ...] = ()
    source: str = "sample"

    stage = "finance"

    @property
    def mean(self) -> float:
        return float(self.weights @ self.mu)

    @property
    def volatility(self) -> float:
        return float(np.sqrt(self.weights @ self.cov @ self.weights))

    @property
    def sharpe(self) -> float:
        return (self.mean - self.rf) / self.volatility if self.volatility > 0 else float("nan")

    def weights_table(self) -> Table:
        rows = [[float(w)] for w in self.weights]
        return Table(["weight"], rows, index=list(self.assets), index_name="asset",
                     title=f"Portfolio: {self.objective.replace('_', ' ')}",
                     notes=(f"Mean {self.mean:.6g}, volatility {self.volatility:.6g}, Sharpe ratio {self.sharpe:.4f} per period "
                            f"(covariance: {self.source}).",))

    def summary(self) -> Table:
        return self.weights_table()

    def frontier(self, points: int = 25) -> pd.DataFrame:
        """The efficient frontier within the same bounds: for target means from the minimum-variance portfolio's to the
        highest reachable, the least volatility and its weights."""
        bounds = list(self.bounds)
        w_min = _solve("min_variance", self.mu, self.cov, self.rf, None, bounds)
        low = float(w_min @ self.mu)
        caps = np.array([b[1] for b in bounds])
        order = np.argsort(-self.mu)  # the highest mean: fill the best assets to their caps
        w, left = np.zeros(len(self.mu)), 1.0
        for i in order:
            w[i] = min(caps[i], left)
            left -= w[i]
        high = float(w @ self.mu)
        rows = []
        for target in np.linspace(low, high, points):
            try:
                wt = _solve("target_return", self.mu, self.cov, self.rf, float(target), bounds)
            except SpecificationError:
                continue
            rows.append({"mean": float(wt @ self.mu), "volatility": float(np.sqrt(wt @ self.cov @ wt)),
                         **{a: float(x) for a, x in zip(self.assets, wt)}})
        return pd.DataFrame(rows)

    def plot(self) -> Figure:
        from pymetrix.figures import Figure, Line, Panel, Scatter

        f = self.frontier(25)
        line = Line(pd.Series(f["mean"].to_numpy(), index=pd.Index(f["volatility"].to_numpy(), name="volatility")), "efficient frontier")
        sd = np.sqrt(np.diag(self.cov))
        assets = Scatter(pd.Series(sd), pd.Series(self.mu), "assets")
        chosen = Scatter(pd.Series([self.volatility]), pd.Series([self.mean]), self.objective.replace("_", " "))
        return Figure((Panel((line, assets, chosen), title="Mean and volatility", xlabel="volatility"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return pd.DataFrame({"weight": self.weights}, index=list(self.assets))

    def __repr__(self) -> str:
        return f"<Portfolio {self.objective}: {', '.join(f'{a} {w:.3f}' for a, w in zip(self.assets, self.weights))}>"


with plugin_context(PLUGIN):
    @api("finance:factor_model")
    def factor_model(data: Dataset, assets: Sequence[str], factors: Sequence[str], rf: str | None = None) -> FactorModel:
        """Each asset's excess return (over ``rf``, if given) on the factors, with Newey-West errors, and the GRS test
        that every alpha is zero."""
        from pymetrix.estimation import OLS, Equation

        assets = [assets] if isinstance(assets, str) else list(assets)
        factors = [factors] if isinstance(factors, str) else list(factors)
        needed = [*assets, *factors, *([rf] if rf else [])]
        missing = [s for s in needed if s not in data.names]
        if missing:
            raise SpecificationError(f"no series {', '.join(map(repr, missing))} in the dataset")
        if data.time_axis is None or data.kind != "time_series":
            raise SpecificationError("a factor model needs returns over time")
        frame = data.frame[list(dict.fromkeys(needed))].dropna()
        T, N, L = len(frame), len(assets), len(factors)
        if T <= N + L + 1:
            raise SpecificationError(f"{T} periods are too few for {N} assets and {L} factors")
        excess = pd.DataFrame({a: frame[a] - (frame[rf] if rf else 0.0) for a in assets})
        f_values = frame[factors]
        from pymetrix.data import Dataset as _D

        work = _D.from_frame(pd.concat([excess.add_prefix("ex_"), f_values], axis=1))
        fits = {a: Equation(f"ex_{a} ~ {' + '.join(factors)}").fit(work, method=OLS(cov="hac")) for a in assets}
        alpha = np.array([fits[a].params["const"] for a in assets])
        resid = np.column_stack([fits[a].resid.to_numpy(float) for a in assets])
        return FactorModel(tuple(assets), tuple(factors), fits, excess, f_values, grs(alpha, resid, f_values.to_numpy(float)))

    @renamed_arguments(target="target_return")
    @api("finance:optimize_portfolio", version=2)
    def optimize_portfolio(data: Dataset, assets: Sequence[str], objective: Literal["min_variance", "max_sharpe", "target_return"] = "min_variance",
                           rf: float = 0.0, target_return: float | None = None, long_only: bool = True,
                           bounds: Sequence[tuple[float, float]] | None = None, cov: Any = None, mu: Any = None) -> Portfolio:
        """Portfolio weights that minimise variance, maximise the Sharpe ratio over ``rf`` or reach the mean return
        ``target_return`` with the least variance (``target_return`` was called ``target``, which still works, with a
        DeprecationWarning). ``bounds`` per asset (default: 0 to 1 when ``long_only``, else -1 to 2); ``cov``: a
        covariance matrix, or a fitted multivariate GARCH (its next-period forecast); ``mu``: mean returns to use
        instead of the sample's (Black-Litterman's posterior, say)."""
        assets = [assets] if isinstance(assets, str) else list(assets)
        missing = [a for a in assets if a not in data.names]
        if missing:
            raise SpecificationError(f"no series {', '.join(map(repr, missing))} in the dataset")
        if objective not in ("min_variance", "max_sharpe", "target_return"):
            raise SpecificationError("objective is 'min_variance', 'max_sharpe' or 'target_return'")
        r = data.frame[assets].dropna().to_numpy(float)
        if mu is None:
            mu = r.mean(axis=0)
        else:
            mu = np.asarray(getattr(mu, "values", mu), float)
            if mu.shape != (len(assets),):
                raise SpecificationError(f"give one mean return for each of the {len(assets)} assets")
        source = "sample"
        if cov is None:
            S = np.cov(r.T, ddof=1)
        elif hasattr(cov, "forecast") and hasattr(cov, "variables"):  # a fitted multivariate GARCH
            if list(cov.variables) != assets:
                raise SpecificationError(f"the multivariate GARCH is of {', '.join(cov.variables)}; the assets must match, in order")
            S, source = cov.forecast(1)[0], f"{cov.model} forecast for the next period"
        else:
            S = np.asarray(getattr(cov, "values", cov), float)
            source = "given"
        if S.shape != (len(assets), len(assets)):
            raise SpecificationError(f"the covariance must be {len(assets)} x {len(assets)}")
        default = (0.0, 1.0) if long_only else (-1.0, 2.0)
        b = [tuple(map(float, x)) for x in bounds] if bounds else [default] * len(assets)
        if len(b) != len(assets) or sum(lo for lo, _ in b) > 1 or sum(hi for _, hi in b) < 1:
            raise SpecificationError("the bounds must allow weights that sum to one, one pair per asset")
        w = _solve(objective, mu, S, float(rf), target_return, b)
        return Portfolio(tuple(assets), objective, w, mu, S, float(rf), tuple(b), source)


@call_migration("finance:optimize_portfolio", 1)
def _optimize_portfolio_target_return(arguments: dict[str, Any]) -> dict[str, Any]:  # version 2: target became target_return
    arguments["target_return"] = arguments.pop("target", None)
    return arguments
