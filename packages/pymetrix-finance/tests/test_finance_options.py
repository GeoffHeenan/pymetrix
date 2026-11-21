"""Options and yield curves (the finance plugin), checked independently: Black-Scholes prices against Hull's worked
examples and put-call parity; the binomial tree converging to Black-Scholes, matching Hull's American put, and an
American call without dividends worth its European price; the Greeks against finite differences; implied volatility
recovering the volatility that made each price; Nelson-Siegel and Svensson recovering known factors and decays
exactly from noise-free curves; the loadings' limits; the AR(1) forecasts against statsmodels."""

import importlib

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, implied_volatility, option_price, yield_curve
from pymetrix.errors import SpecificationError

opt = importlib.import_module("pymetrix_finance.options")
yc_mod = importlib.import_module("pymetrix_finance.yields")


def test_black_scholes_against_hull_and_parity():
    # Hull, Options, Futures, and Other Derivatives, Example 15.6: S 42, K 40, r 10%, sigma 20%, six months
    assert opt.black_scholes(42, 40, 0.5, 0.1, 0.2) == pytest.approx(4.76, abs=0.005)
    assert opt.black_scholes(42, 40, 0.5, 0.1, 0.2, kind="put") == pytest.approx(0.81, abs=0.005)
    S, K, T, r, q = 100.0, np.array([80, 100, 125.0]), 1.5, 0.04, 0.02
    c, p = opt.black_scholes(S, K, T, r, 0.3, q), opt.black_scholes(S, K, T, r, 0.3, q, "put")
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


def test_the_binomial_tree():
    assert opt.binomial(42, 40, 0.5, 0.1, 0.2, steps=4000) == pytest.approx(opt.black_scholes(42, 40, 0.5, 0.1, 0.2), abs=2e-3)
    # Hull, Example 21.1 (Figure 21.3): an American put, S = K = 50, r 10%, sigma 40%, five months: 4.49 with five steps
    assert opt.binomial(50, 50, 5 / 12, 0.1, 0.4, kind="put", american=True, steps=5) == pytest.approx(4.49, abs=0.005)
    european = opt.binomial(50, 50, 5 / 12, 0.1, 0.4, kind="put", steps=500)
    american = opt.binomial(50, 50, 5 / 12, 0.1, 0.4, kind="put", american=True, steps=500)
    assert american > european + 0.1
    # without dividends an American call is never exercised early
    assert opt.binomial(50, 45, 1, 0.05, 0.3, american=True) == pytest.approx(opt.binomial(50, 45, 1, 0.05, 0.3), abs=1e-10)


def test_the_greeks_against_finite_differences():
    S, K, T, r, s, q = 100.0, 95.0, 0.75, 0.03, 0.25, 0.01
    for kind in ("call", "put"):
        g = opt.option_price(S, K, T, r, s, q, kind=kind).values

        def bs(S_=S, T_=T, r_=r, s_=s):
            return opt.black_scholes(S_, K, T_, r_, s_, q, kind)

        h = 1e-4
        assert g["delta"] == pytest.approx((bs(S_=S + h) - bs(S_=S - h)) / (2 * h), rel=1e-6)
        assert g["gamma"] == pytest.approx((bs(S_=S + 0.01) - 2 * bs() + bs(S_=S - 0.01)) / 1e-4, rel=1e-4)
        assert g["vega"] == pytest.approx((bs(s_=s + h) - bs(s_=s - h)) / (2 * h), rel=1e-6)
        assert g["theta"] == pytest.approx(-(bs(T_=T + h) - bs(T_=T - h)) / (2 * h), rel=1e-6)
        assert g["rho"] == pytest.approx((bs(r_=r + h) - bs(r_=r - h)) / (2 * h), rel=1e-6)
    # an American call without dividends is a European one: the tree's Greeks match the closed forms
    tree = opt.option_price(S, K, T, r, s, kind="call", american=True, steps=2000).values
    exact = opt.greeks(S, K, T, r, s, 0.0, "call")
    assert tree["delta"] == pytest.approx(exact["delta"], abs=2e-3) and tree["gamma"] == pytest.approx(exact["gamma"], rel=0.02)
    assert tree["vega"] == pytest.approx(exact["vega"], rel=0.01) and tree["theta"] == pytest.approx(exact["theta"], rel=0.02)
    assert tree["rho"] == pytest.approx(exact["rho"], rel=0.01)
    with pytest.raises(SpecificationError, match="positive"):
        option_price(100, 100, 0, 0.03, 0.2)


def test_implied_volatility_recovers_the_volatility():
    K = np.tile([70.0, 85, 100, 115, 130], 3)
    T = np.repeat([0.1, 0.5, 2.0], 5)
    vol = 0.18 + 0.15 * (K / 100 - 1) ** 2 + 0.02 * T
    flag = np.where(K >= 100, 1.0, -1.0)  # out-of-the-money options: puts below the spot, calls above
    price = np.where(flag == 1, opt.black_scholes(100, K, T, 0.03, vol, 0.01), opt.black_scholes(100, K, T, 0.03, vol, 0.01, "put"))
    frame = pd.DataFrame({"quote": [f"q{i:02d}" for i in range(15)], "price": price, "strike": K, "expiry": T, "cp": flag})
    smile = implied_volatility(Dataset.from_frame(frame, entity="quote"), "price", "strike", "expiry", spot=100, rate=0.03,
                               dividend=0.01, kind="cp")
    assert smile.frame["implied vol"].to_numpy() == pytest.approx(vol, abs=1e-8)
    assert smile.surface().shape == (5, 3) and list(smile.frame["kind"][:2]) == ["put", "put"]
    assert np.isnan(opt.implied_vol(150.0, 100, 100, 1, 0.03))  # above the spot: no volatility gives it
    assert np.isnan(opt.implied_vol(1.0, 100, 50, 1, 0.03))  # below its intrinsic value


def _curves(F, m, lam, lam2=None, T=None):
    idx = pd.period_range("2000-01", periods=len(F), freq="M", name="time")
    cols = [f"y{i}" for i in range(len(m))]
    return Dataset.from_frame(pd.DataFrame(F @ yc_mod.loadings(m, lam, lam2).T, columns=cols, index=idx)), cols


def test_nelson_siegel_and_svensson_recover_their_factors():
    m = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    L = yc_mod.loadings([1e-12, 1e4], 0.7)
    assert L[0] == pytest.approx([1, 1, 0]) and L[1] == pytest.approx([1, 0, 0], abs=1e-3)  # short end b0 + b1; long end b0
    rng = np.random.default_rng(0)
    F = np.column_stack([5 + np.cumsum(rng.normal(0, 0.1, 150)), -1 + np.cumsum(rng.normal(0, 0.1, 150)), rng.normal(0, 1, 150)])
    data, cols = _curves(F, m, 0.6)
    ns = yield_curve(data, cols, m)
    assert ns.lam == pytest.approx(0.6, rel=1e-6) and ns.estimated
    assert ns.factors.to_numpy() == pytest.approx(F, abs=1e-6)
    fixed = yield_curve(data, cols, m, lam=0.6)
    assert fixed.factors.to_numpy() == pytest.approx(F, abs=1e-10) and not fixed.estimated
    assert ns.curve(maturities=[1e-9]).iloc[0] == pytest.approx(F[-1, 0] + F[-1, 1], abs=1e-6)

    F4 = np.column_stack([F, rng.normal(0, 1, 150)])
    sdata, _ = _curves(F4, m, 0.8, 0.15)
    sv = yield_curve(sdata, cols, m, model="svensson")
    assert (sv.lam, sv.lam2) == pytest.approx((0.8, 0.15), rel=1e-5)
    assert sv.factors.to_numpy() == pytest.approx(F4, abs=1e-5)
    nested = yield_curve(data, cols, m, model="svensson", lam=0.6, lam2=0.1)  # Nelson-Siegel curves: no second hump
    assert nested.factors["curvature 2"].abs().max() < 1e-8 and nested.factors.iloc[:, :3].to_numpy() == pytest.approx(F, abs=1e-8)
    with pytest.raises(SpecificationError, match="one positive maturity"):
        yield_curve(data, cols, m[:-1])


def test_the_diebold_li_forecasts():
    from statsmodels.tsa.ar_model import AutoReg

    m = np.array([0.25, 1, 2, 5, 10, 30])
    rng = np.random.default_rng(3)
    F = np.zeros((200, 3))
    F[0] = (5, -1, 0)
    for t in range(1, 200):
        F[t] = (0.2, -0.05, 0.0) + np.array([0.96, 0.9, 0.7]) * F[t - 1] + rng.normal(0, 0.1, 3)
    data, cols = _curves(F, m, 0.7)
    yc = yield_curve(data, cols, m, lam=0.7)
    ar = yc.ar1()
    for j, c in enumerate(yc.factors.columns):
        sm = AutoReg(yc.factors[c].to_numpy(), lags=1, trend="c").fit()
        assert ar.loc[c].to_numpy() == pytest.approx(sm.params, rel=1e-8)
    f = F[-1].copy()
    for _ in range(3):  # three steps by hand
        f = ar["const"].to_numpy() + ar["phi"].to_numpy() * f
    fc = yc.forecast(3)
    assert fc.iloc[-1].to_numpy() == pytest.approx(yc_mod.loadings(m, 0.7) @ f, abs=1e-8)
    assert str(fc.index[0]) == str(data.frame.index[-1] + 1)


def test_the_dialogs():
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS
    from pymetrix.views import sections

    s = Session()
    s.run(COMMANDS["PriceOption"].cls(spot=42, strike=40, expiry=0.5, rate=0.1, vol=0.2))
    assert s.objects["option"].price == pytest.approx(4.76, abs=0.005)
    assert s.history[-1].code.startswith("option = option_price(")
    m = [0.25, 1, 2, 5, 10]
    rng = np.random.default_rng(1)
    data, cols = _curves(np.column_stack([5 + rng.normal(0, 0.1, 40), -1 + rng.normal(0, 0.1, 40), rng.normal(0, 1, 40)]), np.array(m), 0.5)
    s.objects["data"] = data
    s.run(COMMANDS["FitYieldCurve"].cls(", ".join(cols), "0.25, 1, 2, 5, 10", lam=0.5))
    assert s.objects["curve"].lam == 0.5
    assert [t for t, _ in sections(s.objects["curve"])][0] == "Fit"
