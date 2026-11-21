"""Portfolio strategies over time: rolling rebalancing, risk parity, Black-Litterman, costs and turnover.

    bt = backtest_strategy(data, ["a", "b", "c"], ["equal_weight", "min_variance", "risk_parity"], window=60, cost=0.001)
    bt.performance(); bt.plot(); bt.weights["risk_parity"]
    risk_parity_weights(cov); black_litterman(cov, market_weights, views=[...], q=[...])

**Backtests.** Every ``rebalance`` periods, each strategy sets weights from the ``window`` periods before (never
the period it is applied to): equal weights, minimum variance, maximum Sharpe (over ``rf``), risk parity, or
inverse volatility. Between rebalances the weights drift with the assets' returns. A rebalance trades the
difference between the drifted and the new weights; its turnover, sum |w_new - w_drifted|, costs ``cost`` per unit
(0.001 for 10 basis points), taken from that period's return. Returns are simple (``kind="log"`` converts them).

**Risk parity** gives every asset the same share of the portfolio's variance: long-only weights by Spinu's (2013)
convex problem, minimise x'Sx / 2 - sum b_i log x_i, then w = x / sum x, with budgets b (equal by default).

**Black-Litterman** (1992; He and Litterman 1999): the returns implied by market weights, pi = delta S w_m, are
combined with views P mu = q, held with uncertainty Omega (by default diag(P tau S P')), into the posterior mean
mu = pi + tau S P' (P tau S P' + Omega)^-1 (q - P pi), and covariance S + tau S - tau S P' (P tau S P' + Omega)^-1 P tau S.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import optimize

from pymetrix.extension import api
from pymetrix.errors import SpecificationError
from pymetrix.output import Table
from pymetrix.extension import plugin_context
from . import PLUGIN
from .blacklitterman import BlackLittermanPosterior, view_label
from .portfolio import _solve

if TYPE_CHECKING:
    from pymetrix.data import Dataset
    from pymetrix.figures import Figure

STRATEGIES = ("equal_weight", "inverse_volatility", "min_variance", "max_sharpe", "risk_parity")


def risk_parity_weights(cov: Any, budgets: Any = None) -> np.ndarray:
    """Long-only weights whose shares of the portfolio's variance are ``budgets`` (equal by default)."""
    S = np.asarray(cov, float)
    n = len(S)
    b = np.full(n, 1 / n) if budgets is None else np.asarray(budgets, float) / np.sum(budgets)
    scale = np.sqrt(np.diag(S))

    def f(x: np.ndarray) -> float:
        return 0.5 * x @ S @ x - b @ np.log(x)

    def g(x: np.ndarray) -> np.ndarray:
        return S @ x - b / x

    found = optimize.minimize(f, 1 / scale, jac=g, method="L-BFGS-B", bounds=[(1e-12, None)] * n,
                              options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 5000})
    return found.x / found.x.sum()


def black_litterman(cov: Any, market_weights: Any, views: Any = None, q: Any = None, omega: Any = None, delta: float = 2.5,
                    tau: float = 0.05) -> BlackLittermanPosterior:
    """The Black-Litterman posterior mean and covariance of returns, from market weights and views ``views @ mu = q``.
    ``cov`` is the returns' covariance: an array, a DataFrame or a pymetrix table (``data.cov([...])``), whose names
    label the assets. A ``BlackLittermanPosterior`` (prior, posterior, views, optimal weights), which unpacks as
    ``mu, sigma = black_litterman(...)``."""
    if isinstance(cov, Table):  # e.g. data.cov(names)
        cov = cov.to_frame()
    S = np.asarray(cov, float)
    w = np.asarray(market_weights, float)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] != len(w):
        raise SpecificationError(f"the covariance must be square with a row for each of the {len(w)} market weights")
    if isinstance(cov, pd.DataFrame):
        assets = [str(c) for c in cov.columns]
    elif isinstance(market_weights, pd.Series):
        assets = [str(c) for c in market_weights.index]
    else:
        assets = [f"asset {i + 1}" for i in range(len(w))]
    pi = delta * S @ w
    tS = tau * S
    P_frame = q_series = Om_frame = None
    if views is None:
        mu, sigma = pi, S + tS
    else:
        P = np.atleast_2d(np.asarray(views, float))
        q = np.atleast_1d(np.asarray(q, float))
        if P.shape != (len(q), len(w)):
            raise SpecificationError(f"give one row of {len(w)} view weights for each of the {len(q)} views")
        Om = np.diag(np.diag(P @ tS @ P.T)) if omega is None else np.atleast_2d(np.asarray(omega, float))
        K = tS @ P.T @ np.linalg.inv(P @ tS @ P.T + Om)
        mu, sigma = pi + K @ (q - P @ pi), S + tS - K @ P @ tS
        labels = _unique([view_label(row, assets) for row in P])
        P_frame = pd.DataFrame(P, index=pd.Index(labels, name="view"), columns=pd.Index(assets, name="asset"))
        q_series = pd.Series(q, index=P_frame.index, name="q")
        Om_frame = pd.DataFrame(Om, index=P_frame.index, columns=P_frame.index)
    idx = pd.Index(assets, name="asset")
    return BlackLittermanPosterior(tuple(assets), pd.Series(pi, index=idx, name="prior"), pd.Series(mu, index=idx, name="posterior"),
                                   pd.DataFrame(sigma, index=idx, columns=idx), pd.DataFrame(S, index=idx, columns=idx),
                                   pd.Series(w, index=idx, name="market weight"), P_frame, q_series, Om_frame, omega is None,
                                   float(delta), float(tau), _provenance(S, w, assets, views, q, omega, delta, tau))


def _unique(labels: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for lab in labels:
        seen[lab] = seen.get(lab, 0) + 1
        out.append(lab if seen[lab] == 1 else f"{lab} ({seen[lab]})")
    return out


def _provenance(S: np.ndarray, w: np.ndarray, assets: list[str], views: Any, q: Any, omega: Any, delta: float, tau: float) -> Any:
    """What the posterior was made from: the inputs (the covariance by its fingerprint) and the method."""
    import hashlib

    import pymetrix
    from pymetrix.metadata import Provenance

    spec = {"assets": list(assets), "market_weights": w.tolist(), "views": None if views is None else np.atleast_2d(np.asarray(views, float)).tolist(),
            "q": None if q is None else np.atleast_1d(np.asarray(q, float)).tolist(),
            "omega": None if omega is None else np.atleast_2d(np.asarray(omega, float)).tolist(), "delta": float(delta), "tau": float(tau)}
    method = {"type": "Black-Litterman", "estimation": "closed-form posterior (He and Litterman 1999)",
              "omega": "diag(P tau S P')" if omega is None else "given"}
    digest = hashlib.sha256(np.ascontiguousarray(S, float).tobytes() + np.ascontiguousarray(w, float).tobytes()).hexdigest()[:16]
    return Provenance(data_vintages={}, spec=spec, method=method, sample=None, nobs=0, dropped=0, package_version=pymetrix.__version__,
                      data_fingerprint=digest)


def _weights(strategy: str, R: np.ndarray, rf: float, long_only: bool) -> np.ndarray:
    n = R.shape[1]
    if strategy == "equal_weight":
        return np.full(n, 1 / n)
    S = np.cov(R.T, ddof=1)
    if strategy == "inverse_volatility":
        v = 1 / np.sqrt(np.diag(S))
        return v / v.sum()
    if strategy == "risk_parity":
        return risk_parity_weights(S)
    bounds = [(0.0, 1.0) if long_only else (-1.0, 2.0)] * n
    return _solve(strategy, R.mean(axis=0), S, rf, None, bounds)


def run_backtest(R: np.ndarray, weights_at: Any, window: int, rebalance: int, cost: float) -> dict[str, np.ndarray]:
    """Gross and net returns, weights (at the start of each period) and turnover of a rebalanced portfolio.
    ``weights_at(t)`` gives the target weights from the returns before period t."""
    T, n = R.shape
    gross, net, turnover = np.full(T, np.nan), np.full(T, np.nan), np.zeros(T)
    W = np.full((T, n), np.nan)
    w = None
    for t in range(window, T):
        if w is None or (t - window) % rebalance == 0:
            target = weights_at(t)
            turnover[t] = np.abs(target - w).sum() if w is not None else np.abs(target).sum()
            w = target
        W[t] = w
        gross[t] = float(w @ R[t])
        net[t] = gross[t] - cost * turnover[t]
        grown = w * (1 + R[t])
        w = grown / grown.sum() if grown.sum() != 0 else w  # the weights drift with the returns
    return {"gross": gross, "net": net, "weights": W, "turnover": turnover}


@dataclass(frozen=True, eq=False)
class StrategyBacktest:
    """Rebalanced portfolio strategies' returns, weights and turnover, out of sample."""

    assets: tuple[str, ...]
    strategies: tuple[str, ...]
    returns: pd.DataFrame  # net returns, one column a strategy
    gross: pd.DataFrame = field(repr=False)
    weights: dict[str, pd.DataFrame] = field(repr=False)
    turnover: pd.DataFrame = field(repr=False)
    window: int = 60
    rebalance: int = 1
    cost: float = 0.0
    per_year: int = 12
    rf: Any = None  # a series of the risk-free rate, aligned with the returns, or None

    stage = "finance"

    def performance(self) -> Table:
        from .returns import measures

        order = ["annual return", "annual volatility", "Sharpe ratio", "Sharpe std err", "Sortino ratio", "maximum drawdown",
                 "Calmar ratio", "hit rate"]
        rf = None if self.rf is None else self.rf.to_numpy(float)
        cols = {s: measures(self.returns[s].to_numpy(float), self.per_year, "simple", rf) for s in self.strategies}
        rows = [[float(cols[s][m]) for s in self.strategies] for m in order]
        rows.append([float(self.turnover[s].sum() / (len(self.returns) / self.per_year)) for s in self.strategies])
        return Table(list(self.strategies), rows, index=order + ["annual turnover"], index_name="measure", title="Strategy backtest",
                     notes=(f"{len(self.returns)} periods out of sample; weights from the {self.window} periods before, reset every "
                            f"{self.rebalance} period(s); cost {self.cost:g} per unit of turnover, net of which the returns are.",))

    def summary(self) -> Table:
        return self.performance()

    def cumulative(self) -> pd.DataFrame:
        return (1 + self.returns).cumprod()

    def plot(self) -> Figure:
        from pymetrix.figures import Figure, Line, Panel

        c = self.cumulative()
        first = self.weights[self.strategies[-1]]
        return Figure((Panel(tuple(Line(c[s], s.replace("_", " ")) for s in c.columns), title="Growth of 1, net of costs"),
                       Panel(tuple(Line(first[a], a) for a in first.columns), title=f"Weights: {self.strategies[-1].replace('_', ' ')}")))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return self.returns

    def __repr__(self) -> str:
        return f"<StrategyBacktest of {', '.join(self.strategies)} on {len(self.assets)} assets>"


with plugin_context(PLUGIN):
    @api("finance:backtest_strategy")
    def backtest_strategy(data: Dataset, assets: Sequence[str], strategies: Sequence[str] = ("equal_weight", "min_variance", "risk_parity"),
                          window: int = 60, rebalance: int = 1, cost: float = 0.0, rf: str | float = 0.0, long_only: bool = True,
                          kind: Literal["simple", "log"] = "simple", periods_per_year: int | None = None) -> StrategyBacktest:
        """Out-of-sample returns of rebalanced portfolio strategies (equal weight, inverse volatility, minimum variance,
        maximum Sharpe, risk parity), each re-estimated every ``rebalance`` periods on the ``window`` before, net of
        ``cost`` per unit of turnover."""
        from .returns import annual_periods

        assets = [assets] if isinstance(assets, str) else list(assets)
        strategies = [strategies] if isinstance(strategies, str) else list(strategies)
        unknown = [s for s in strategies if s not in STRATEGIES]
        if unknown:
            raise SpecificationError(f"unknown strategies {', '.join(map(repr, unknown))}: choose from {', '.join(STRATEGIES)}")
        rf_name = rf if isinstance(rf, str) else None
        needed = [*assets, *([rf_name] if rf_name else [])]
        missing = [s for s in needed if s not in data.names]
        if missing:
            raise SpecificationError(f"no series {', '.join(map(repr, missing))} in the dataset")
        if data.time_axis is None or data.kind != "time_series":
            raise SpecificationError("a backtest needs returns over time")
        frame = data.frame[needed].dropna()
        R = frame[assets].to_numpy(float)
        if kind == "log":
            R = np.expm1(R)
        if window < len(assets) + 2 or window >= len(R) - 1:
            raise SpecificationError(f"the window must be at least {len(assets) + 2} periods and leave some to test ({len(R)} in all)")
        if rebalance < 1 or cost < 0:
            raise SpecificationError("rebalance is at least 1 period, and cost is not negative")
        rf_values = frame[rf_name].to_numpy(float) if rf_name else np.full(len(R), float(rf))
        out, W, gross, turnover = {}, {}, {}, {}
        for s in strategies:
            run = run_backtest(R, lambda t, s=s: _weights(s, R[t - window:t], float(np.mean(rf_values[t - window:t])), long_only),
                               window, rebalance, cost)
            idx = frame.index[window:]
            out[s], gross[s], turnover[s] = run["net"][window:], run["gross"][window:], run["turnover"][window:]
            W[s] = pd.DataFrame(run["weights"][window:], index=idx, columns=assets)
        idx = frame.index[window:]
        q = int(periods_per_year) if periods_per_year else annual_periods(frame.index)
        rf_series = pd.Series(rf_values[window:], index=idx) if rf_name else None
        return StrategyBacktest(tuple(assets), tuple(strategies), pd.DataFrame(out, index=idx), pd.DataFrame(gross, index=idx), W,
                                pd.DataFrame(turnover, index=idx), int(window), int(rebalance), float(cost), q, rf_series)


with plugin_context(PLUGIN):  # a recorded call: the Black-Litterman dialog (app.py) builds it
    black_litterman = api("finance:black_litterman")(black_litterman)
