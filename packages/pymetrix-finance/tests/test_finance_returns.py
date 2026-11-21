"""Returns, performance, factor models and portfolios (the finance plugin), checked independently: returns against
numpy's differences; each measure from its definition by hand (the drawdown by a plain loop, Lo's standard error,
beta by numpy's polyfit); the GRS test against alpha's squared t statistic with one asset and against the
maximum-Sharpe-ratio identity with several; rolling betas against a direct least-squares fit; portfolio weights
against their closed forms, and long-only weights against a grid search."""

import importlib

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, factor_model, optimize_portfolio, performance, returns
from pymetrix.errors import SpecificationError

ret = importlib.import_module("pymetrix_finance.returns")  # the module (the package exports a function of that name)
port = importlib.import_module("pymetrix_finance.portfolio")


def _monthly(T=300, seed=0):
    """Three assets driven by two factors, with a risk-free rate, monthly."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0.005, 0.04, (T, 2))
    rf = rng.uniform(0.001, 0.003, T)
    a = 0.001 + f @ np.array([[1.0, 0.5, 0.2], [0.3, -0.2, 0.8]]) + rng.normal(0, 0.02, (T, 3)) + rf[:, None]
    idx = pd.period_range("1990-01", periods=T, freq="M", name="time")
    return pd.DataFrame(np.column_stack([a, f, rf]), columns=["a", "b", "c", "mkt", "smb", "rf"], index=idx)


def test_returns_from_prices():
    p = pd.Series([100.0, 102.0, 99.0, 105.0, 104.0])
    d = Dataset.from_frame(pd.DataFrame({"p": p.to_numpy()}, index=pd.period_range("2000", periods=5, freq="Y", name="time")))
    out = returns(d, "p")
    assert np.allclose(out["p_ret"].to_numpy()[1:], np.diff(np.log(p))) and np.isnan(out["p_ret"].iloc[0])
    two = returns(d, ["p"], kind="simple", periods=2, suffix="_r2")
    assert np.allclose(two["p_r2"].to_numpy()[2:], p.to_numpy()[2:] / p.to_numpy()[:-2] - 1)
    bad = Dataset.from_frame(pd.DataFrame({"p": [1.0, 0.0, 2.0]}, index=pd.period_range("2000", periods=3, freq="Y", name="time")))
    with pytest.raises(SpecificationError, match="positive prices"):
        returns(bad, "p")


def test_annual_periods_and_drawdown():
    for freq, q in (("D", 252), ("W", 52), ("M", 12), ("Q", 4), ("Y", 1)):
        assert ret.annual_periods(pd.period_range("2000-01-03", periods=3, freq=freq)) == q
    r = pd.Series([0.1, -0.2, 0.05, 0.3, -0.1], name="r")
    wealth, peak, by_hand = 1.0, 1.0, []
    for x in r:  # the fall from the highest wealth so far, by a plain loop
        wealth *= 1 + x
        peak = max(peak, wealth)
        by_hand.append(wealth / peak - 1)
    assert np.allclose(ret.drawdown(r, "simple"), by_hand)
    assert np.allclose(ret.drawdown(np.log1p(r), "log"), by_hand)


def test_performance_measures_by_hand():
    df = _monthly()
    perf = performance(Dataset.from_frame(df), ["a", "b"], rf="rf", benchmark="mkt", kind="simple")
    v = perf.values["a"]
    r, rf, m, q, T = df["a"].to_numpy(), df["rf"].to_numpy(), df["mkt"].to_numpy(), 12, len(df)
    ex = r - rf
    sr = ex.mean() / r.std(ddof=1)
    assert v["Sharpe ratio"] == pytest.approx(sr * np.sqrt(q), rel=1e-12)
    assert v["Sharpe std err"] == pytest.approx(np.sqrt((1 + sr**2 / 2) / T) * np.sqrt(q), rel=1e-12)
    assert v["annual return"] == pytest.approx(np.prod(1 + r) ** (12 / T) - 1, rel=1e-12)
    assert v["annual volatility"] == pytest.approx(r.std(ddof=1) * np.sqrt(12), rel=1e-12)
    assert v["Sortino ratio"] == pytest.approx(ex.mean() / np.sqrt(np.mean(np.minimum(ex, 0) ** 2)) * np.sqrt(12), rel=1e-12)
    wealth = np.cumprod(1 + r)
    mdd = np.min(wealth / np.maximum.accumulate(wealth) - 1)
    assert v["maximum drawdown"] == pytest.approx(mdd) and v["Calmar ratio"] == pytest.approx(v["annual return"] / -mdd)
    beta, alpha = np.polyfit(m - rf, ex, 1)
    assert v["beta"] == pytest.approx(beta, rel=1e-10) and v["alpha (annual)"] == pytest.approx(alpha * 12, rel=1e-8)
    active = r - m
    assert v["information ratio"] == pytest.approx(active.mean() / active.std(ddof=1) * np.sqrt(12), rel=1e-12)
    assert v["hit rate"] == pytest.approx(np.mean(r > 0)) and v["periods"] == T
    assert perf.cumulative()["a"].iloc[-1] == pytest.approx(wealth[-1])
    assert performance(Dataset.from_frame(df), "a", periods_per_year=4).per_year == 4


def test_the_grs_test_and_the_factor_regressions():
    df = _monthly()
    d = Dataset.from_frame(df)
    one = factor_model(d, ["a"], ["mkt", "smb"], rf="rf")
    from pymetrix import OLS, Equation

    # one asset: GRS is the squared t statistic of alpha with classical standard errors, and F(1, T - 1 - L)
    work = Dataset.from_frame(df.assign(ex=df["a"] - df["rf"]))
    classic = Equation("ex ~ mkt + smb").fit(work, method=OLS())
    t = classic.params["const"] / dict(zip(classic.params.names, classic.params.se))["const"]
    assert one.test[0] == pytest.approx(t**2, rel=1e-8) and one.test[2:] == (1, len(df) - 3)

    # several: (T - N - L) / N times the gain in the squared maximum Sharpe ratio, over 1 plus the factors' own
    fm = factor_model(d, ["a", "b", "c"], ["mkt", "smb"], rf="rf")
    ex = (df[["a", "b", "c"]].to_numpy() - df[["rf"]].to_numpy())
    f = df[["mkt", "smb"]].to_numpy()

    def sr2(x):
        mu = x.mean(axis=0)
        return float(mu @ np.linalg.solve(np.cov(x.T, ddof=0), mu))

    T, N, L = len(df), 3, 2
    all_, fac = sr2(np.column_stack([f, ex])), sr2(f)
    assert fm.test[0] == pytest.approx((T - N - L) / N * (all_ - fac) / (1 + fac), rel=1e-8)
    from scipy import stats

    assert fm.test[1] == pytest.approx(stats.f.sf(fm.test[0], N, T - N - L))

    X = np.column_stack([np.ones(T), f])
    coef = np.linalg.lstsq(X, ex, rcond=None)[0]
    assert np.allclose([fm.fits[a].params["mkt"] for a in "abc"], coef[1])
    rb = fm.rolling_betas(60)
    direct = np.linalg.lstsq(X[100:160], ex[100:160, 1], rcond=None)[0]
    assert rb["b on smb"].iloc[159] == pytest.approx(direct[2]) and rb.iloc[:59].isna().all().all()
    assert "GRS" in fm.summary().notes[-1]


def test_portfolios_against_their_closed_forms():
    df = _monthly(seed=4)
    d = Dataset.from_frame(df)
    names = ["a", "b", "c"]
    mu, S = df[names].mean().to_numpy(), np.cov(df[names].to_numpy().T, ddof=1)
    one, wide = np.ones(3), [(-10.0, 10.0)] * 3
    w_mv = np.linalg.solve(S, one) / (one @ np.linalg.solve(S, one))
    assert optimize_portfolio(d, names, bounds=wide).weights == pytest.approx(w_mv, abs=1e-6)
    rf = 0.001
    z = np.linalg.solve(S, mu - rf)
    tangency = optimize_portfolio(d, names, objective="max_sharpe", rf=rf, bounds=wide)
    assert tangency.weights == pytest.approx(z / z.sum(), abs=1e-5)
    target = 0.006  # two-fund: the least variance at a target mean, by its Lagrangian
    A, B, C = one @ np.linalg.solve(S, one), one @ np.linalg.solve(S, mu), mu @ np.linalg.solve(S, mu)
    lam, gam = (C - target * B) / (A * C - B**2), (target * A - B) / (A * C - B**2)
    w_t = lam * np.linalg.solve(S, one) + gam * np.linalg.solve(S, mu)
    got = optimize_portfolio(d, names, objective="target_return", target_return=target, bounds=wide)
    assert got.weights == pytest.approx(w_t, abs=1e-5) and got.mean == pytest.approx(target)

    long = optimize_portfolio(d, names)  # long only: the grid's best
    grid = [(x, y, 1 - x - y) for x in np.linspace(0, 1, 201) for y in np.linspace(0, 1, 201) if x + y <= 1 + 1e-12]
    best = min(grid, key=lambda w: np.array(w) @ S @ np.array(w))
    assert np.all(long.weights >= -1e-10) and long.volatility**2 <= np.array(best) @ S @ np.array(best) + 1e-12
    frontier = long.frontier(8)
    assert frontier["volatility"].is_monotonic_increasing and frontier["mean"].iloc[0] == pytest.approx(long.mean, abs=1e-8)
    with pytest.raises(SpecificationError, match="could not be found"):
        optimize_portfolio(d, names, objective="target_return", target_return=1.0)


def test_a_garch_covariance_and_the_dialogs():
    from pymetrix import mgarch
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS
    from pymetrix.views import sections

    df = _monthly(T=400)
    df.index = pd.period_range("2001-01-01", periods=400, freq="D", name="time")
    df["pa"] = 100 * np.exp(df["a"].cumsum())
    d = Dataset.from_frame(df)
    fit = mgarch(d, ["a", "b"], model="ccc")
    by_garch = optimize_portfolio(d, ["a", "b"], cov=fit)
    assert by_garch.weights == pytest.approx(optimize_portfolio(d, ["a", "b"], cov=fit.forecast(1)[0]).weights)
    assert "forecast" in by_garch.source
    with pytest.raises(SpecificationError, match="must match"):
        optimize_portfolio(d, ["b", "a"], cov=fit)

    s = Session()
    s.objects["data"] = d
    s.run(COMMANDS["MeasurePerformance"].cls("a, b", rf="rf"))
    assert s.history[-1].code == "perf = performance(data, ['a', 'b'], rf='rf')"
    s.run(COMMANDS["FactorRegression"].cls("a, b", "mkt"))
    s.run(COMMANDS["OptimizePortfolio"].cls("a, b, c", objective="max_sharpe"))
    s.run(COMMANDS["Returns"].cls("pa", kind="simple"))
    assert "pa_ret" in s.objects["data"].names
    assert [t for t, _ in sections(s.objects["perf"])][:1] == ["Measures"]
    assert [t for t, _ in sections(s.objects["portfolio"])][0] == "Weights"


def test_target_return_was_target():
    import warnings

    from pymetrix.calls import Call
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS, from_json

    d = Dataset.from_frame(_monthly())
    names = ["a", "b", "c"]
    new = optimize_portfolio(d, names, objective="target_return", target_return=0.006, bounds=[(-10.0, 10.0)] * 3)
    with pytest.warns(DeprecationWarning, match="'target' of optimize_portfolio\\(\\) is now called 'target_return'"):
        old = optimize_portfolio(d, names, objective="target_return", target=0.006, bounds=[(-10.0, 10.0)] * 3)
    assert old.weights == pytest.approx(new.weights, rel=1e-12)

    s = Session()
    s.objects["data"] = d
    s.run(COMMANDS["OptimizePortfolio"].cls("a, b, c", objective="target_return", target_return=0.006))
    call = s.history[-1].command.to_call()
    assert "target_return=0.006" in s.history[-1].code and call.version == 2
    saved = dict(call.arguments)  # the same call saved at version 1, with target
    saved["target"] = saved.pop("target_return")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s.run(Call("finance:optimize_portfolio", 1, None, saved, output="older"))
    assert s.objects["older"].weights == pytest.approx(s.objects["portfolio"].weights, rel=1e-12)
    step = from_json({"kind": "command", "id": "finance:optimize_portfolio", "version": 1,
                      "fields": {"assets": "a, b, c", "objective": "target_return", "target": 0.006}})
    assert step.target_return == 0.006
