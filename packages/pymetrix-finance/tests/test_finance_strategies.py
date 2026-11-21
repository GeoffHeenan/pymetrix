"""Portfolio strategies (the finance plugin), checked independently: equal weights rebalanced every period earn the
assets' average return; a portfolio never rebalanced grows as its buy-and-hold value by hand; costs are the cost
times turnover counted by hand; each rebalance uses only the past (the public optimiser on that window gives the same
weights); risk parity equalises risk shares and, with two assets, is inverse volatility; Black-Litterman's two
textbook forms agree, give the implied returns without views, and hold a certain view exactly."""

import importlib

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, backtest_strategy, optimize_portfolio
from pymetrix.errors import SpecificationError

st = importlib.import_module("pymetrix_finance.strategies")


def _monthly(T=200, n=4, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.02, (n, n))
    R = rng.normal(0.008, 0.01, (T, n)) + rng.normal(0, 1, (T, n)) @ A
    names = [f"a{i}" for i in range(n)]
    return Dataset.from_frame(pd.DataFrame(R, columns=names, index=pd.period_range("2000-01", periods=T, freq="M", name="time"))), names, R


def test_equal_weights_and_buy_and_hold_by_hand():
    d, names, R = _monthly()
    bt = backtest_strategy(d, names, ["equal_weight"], window=24)
    assert bt.returns["equal_weight"].to_numpy() == pytest.approx(R[24:].mean(axis=1), abs=1e-14)
    hold = backtest_strategy(d, names, ["equal_weight"], window=24, rebalance=10_000)
    value = (np.full(4, 0.25) * np.cumprod(1 + R[24:], axis=0)).sum(axis=1)  # each asset's quarter grown, by hand
    assert np.cumprod(1 + hold.returns["equal_weight"].to_numpy()) == pytest.approx(value, rel=1e-12)
    assert hold.turnover["equal_weight"].iloc[1:].eq(0).all()


def test_costs_are_charged_on_turnover():
    d, names, R = _monthly(n=2)
    bt = backtest_strategy(d, names, ["equal_weight"], window=12, rebalance=3, cost=0.01)
    t = bt.turnover["equal_weight"].to_numpy()
    assert bt.returns["equal_weight"].to_numpy() == pytest.approx(bt.gross["equal_weight"].to_numpy() - 0.01 * t, abs=1e-15)
    w = np.array([0.5, 0.5])
    for k in range(3):  # three periods of drift after the first rebalance, then back to halves
        w = w * (1 + R[12 + k]) / (w @ (1 + R[12 + k]))
    assert t[3] == pytest.approx(np.abs(0.5 - w).sum(), rel=1e-12) and t[1] == 0 and t[0] == 1  # the first buys in


def test_no_look_ahead():
    d, names, R = _monthly(seed=3)
    bt = backtest_strategy(d, names, ["min_variance", "max_sharpe"], window=36, rebalance=6, rf=0.001)
    for t in (36, 60, 102):
        past = Dataset.from_frame(d.frame.iloc[t - 36:t])
        row = d.frame.index[t]
        assert bt.weights["min_variance"].loc[row].to_numpy() == pytest.approx(optimize_portfolio(past, names).weights, abs=1e-6)
        sharpe = optimize_portfolio(past, names, objective="max_sharpe", rf=0.001).weights
        assert bt.weights["max_sharpe"].loc[row].to_numpy() == pytest.approx(sharpe, abs=1e-5)
    with pytest.raises(SpecificationError, match="unknown strategies"):
        backtest_strategy(d, names, ["momentum"])


def test_risk_parity():
    _, _, R = _monthly(n=5, seed=4)
    S = np.cov(R.T)
    w = st.risk_parity_weights(S)
    share = w * (S @ w) / (w @ S @ w)
    assert share == pytest.approx(np.full(5, 0.2), abs=1e-8) and w.sum() == pytest.approx(1)
    b = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    wb = st.risk_parity_weights(S, b)
    assert wb * (S @ wb) / (wb @ S @ wb) == pytest.approx(b, abs=1e-8)
    two = np.array([[0.04, 0.012], [0.012, 0.01]])  # with two assets, risk parity is inverse volatility, whatever the correlation
    assert st.risk_parity_weights(two) == pytest.approx(np.array([1 / 0.2, 1 / 0.1]) / 15, abs=1e-8)


def test_black_litterman():
    _, _, R = _monthly(n=3, seed=5)
    S = np.cov(R.T)
    wm = np.array([0.5, 0.3, 0.2])
    pi, C = st.black_litterman(S, wm, delta=3.0, tau=0.05)
    assert pi == pytest.approx(3.0 * S @ wm) and np.linalg.solve(3.0 * S, pi) == pytest.approx(wm)  # implied returns give back the market
    P, q = np.array([[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]]), np.array([0.002, 0.01])
    Om = np.diag([1e-5, 2e-5])
    mu, _ = st.black_litterman(S, wm, P, q, Om, delta=3.0, tau=0.05)
    tS = 0.05 * S  # the precision-weighted form (Theil's mixed estimation)
    precision = np.linalg.inv(np.linalg.inv(tS) + P.T @ np.linalg.inv(Om) @ P)
    assert mu == pytest.approx(precision @ (np.linalg.solve(tS, pi) + P.T @ np.linalg.solve(Om, q)), rel=1e-9)
    sure, _ = st.black_litterman(S, wm, P, q, np.diag([1e-14, 1e-14]))
    assert P @ sure == pytest.approx(q, abs=1e-9)
    d, names, _ = _monthly(n=3, seed=5)  # the posterior means drive the optimiser
    port = optimize_portfolio(d, names, objective="max_sharpe", mu=mu, bounds=[(-10.0, 10.0)] * 3)
    z = np.linalg.solve(np.cov(d.frame[names].to_numpy().T), mu)
    assert port.weights == pytest.approx(z / z.sum(), abs=1e-5)


def test_the_dialog():
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS
    from pymetrix.views import sections

    d, names, _ = _monthly()
    s = Session()
    s.objects["data"] = d
    s.run(COMMANDS["BacktestStrategy"].cls(", ".join(names), "equal_weight, risk_parity", window=24, cost=0.001))
    bt = s.objects["strategy_results"]
    assert bt.strategies == ("equal_weight", "risk_parity") and [t for t, _ in sections(bt)][0] == "Performance"
