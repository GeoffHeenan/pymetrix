"""Volatility surfaces (the finance plugin), checked independently: Heston prices against QuantLib's analytic engine
and against Black-Scholes in its constant-volatility limit; put-call parity of Heston prices; SABR's implied
volatility against QuantLib's ``sabrVolatility``; SVI's butterfly function against a numerical second derivative of
call prices (the density); each model recovering parameters that made its own quotes; and the SVI arbitrage checks
flagging a smile built to have arbitrage."""

import importlib

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, vol_surface
from pymetrix_finance.options import black_scholes

sf = importlib.import_module("pymetrix_finance.surface")
HESTON = (0.04, 1.5, 0.05, 0.6, -0.7)


def test_heston_against_quantlib():
    ql = pytest.importorskip("QuantLib")
    today = ql.Date(1, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    S, r, q = 100.0, 0.03, 0.01
    for days in (91, 274, 730):
        mat = today + days
        T = dc.yearFraction(today, mat)
        rts, dts = (ql.YieldTermStructureHandle(ql.FlatForward(today, x, dc)) for x in (r, q))
        engine = ql.AnalyticHestonEngine(ql.HestonModel(ql.HestonProcess(rts, dts, ql.QuoteHandle(ql.SimpleQuote(S)), *HESTON)))
        strikes = [70.0, 90.0, 100.0, 115.0, 140.0]
        ref = []
        for K in strikes:
            o = ql.EuropeanOption(ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(mat))
            o.setPricingEngine(engine)
            ref.append(o.NPV())
        assert sf.heston_price(S, strikes, T, r, q, *HESTON) == pytest.approx(ref, abs=1e-7)


def test_heston_limits_and_parity():
    K = np.array([80.0, 100.0, 125.0])
    flat = sf.heston_price(100, K, 1.0, 0.03, 0.01, 0.04, 1.0, 0.04, 1e-5, 0.0)  # no volatility of variance: Black-Scholes at 20%
    assert flat == pytest.approx(black_scholes(100, K, 1.0, 0.03, 0.2, 0.01), abs=1e-5)  # sigma^2 divides: rounding, not the model
    c, p = sf.heston_price(100, K, 0.5, 0.03, 0.01, *HESTON), sf.heston_price(100, K, 0.5, 0.03, 0.01, *HESTON, kind="put")
    assert c - p == pytest.approx(100 * np.exp(-0.01 * 0.5) - K * np.exp(-0.03 * 0.5))


def test_sabr_against_quantlib():
    ql = pytest.importorskip("QuantLib")
    for beta in (1.0, 0.5):
        for K in (70.0, 95.0, 100.0, 110.0, 150.0):
            ours = sf.sabr_vol([K], 100.0, 1.5, 0.25 if beta == 1 else 2.5, beta, -0.35, 0.7)[0]
            assert ours == pytest.approx(ql.sabrVolatility(K, 100.0, 1.5, 0.25 if beta == 1 else 2.5, beta, 0.7, -0.35), rel=1e-10)


def test_svi_butterfly_is_the_density():
    # g(k) / sqrt(2 pi w) exp(-d2^2 / 2) is the risk-neutral density in k (Gatheral): compare with d^2 C / d K^2
    a, b, rho, m, s, T = 0.02, 0.12, -0.4, 0.05, 0.25, 1.0
    F = 100.0
    K = np.linspace(70, 140, 15)
    k = np.log(K / F)
    w = sf.svi_total_variance(k, a, b, rho, m, s)
    g = sf.svi_g(k, a, b, rho, m, s)
    d2 = -k / np.sqrt(w) - np.sqrt(w) / 2
    density_k = g / np.sqrt(2 * np.pi * w) * np.exp(-d2**2 / 2)

    def call(x):
        kk = np.log(x / F)
        return black_scholes(F, x, T, 0.0, np.sqrt(sf.svi_total_variance(kk, a, b, rho, m, s) / T))

    h = 1e-3
    density_K = (call(K + h) - 2 * call(K) + call(K - h)) / h**2
    assert density_K * K == pytest.approx(density_k, rel=1e-4)  # the density of K times K is that of k


def _quotes(prices, K, T):
    frame = pd.DataFrame({"quote": [f"q{i:03d}" for i in range(len(K))], "price": prices, "strike": K, "expiry": T})
    return Dataset.from_frame(frame, entity="quote")


def test_each_model_recovers_its_own_quotes():
    K = np.tile(np.arange(70.0, 136.0, 5.0), 4)
    T = np.repeat([0.25, 0.5, 1.0, 2.0], 14)
    prices = np.concatenate([sf.heston_price(100, K[T == t], t, 0.03, 0.01, *HESTON) for t in np.unique(T)])
    h = vol_surface(_quotes(prices, K, T), "price", "strike", "expiry", spot=100, rate=0.03, dividend=0.01, model="heston")
    assert h.params.iloc[0].to_numpy() == pytest.approx(HESTON, rel=1e-4) and not h.at_bounds
    F = 100 * np.exp(0.02 * T)
    vols = np.concatenate([sf.sabr_vol(K[T == t], F[T == t][0], t, 0.22, 1.0, -0.5, 0.8) for t in np.unique(T)])
    sabr = vol_surface(_quotes(black_scholes(100, K, T, 0.03, vols, 0.01), K, T), "price", "strike", "expiry", spot=100, rate=0.03,
                       dividend=0.01, model="sabr")
    assert sabr.params.to_numpy() == pytest.approx(np.tile([0.22, -0.5, 0.8], (4, 1)), rel=1e-5)
    k = np.log(K / F)
    w = sf.svi_total_variance(k, 0.01, 0.1, -0.5, 0.02, 0.2) * np.sqrt(T)  # rises with expiry
    svi = vol_surface(_quotes(black_scholes(100, K, T, 0.03, np.sqrt(w / T), 0.01), K, T), "price", "strike", "expiry", spot=100,
                      rate=0.03, dividend=0.01, model="svi")
    assert (svi.quotes["fitted vol"] - svi.quotes["market vol"]).abs().max() < 1e-4
    assert svi.vol([100.0], 1.0)[0] == pytest.approx(np.sqrt(w[(T == 1.0) & (K == 100)][0]), rel=1e-3)
    assert (svi.arbitrage().to_frame()["free of arbitrage"] == "yes").all()


def test_the_arbitrage_checks_catch_arbitrage():
    K = np.tile(np.arange(70.0, 136.0, 5.0), 2)
    T = np.repeat([0.5, 1.0], 14)
    F = 100 * np.exp(0.02 * T)
    vol = np.where(T == 0.5, 0.3, 0.15)  # total variance falls from six months to a year: calendar arbitrage
    surf = vol_surface(_quotes(black_scholes(100, K, T, 0.03, vol, 0.01), K, T), "price", "strike", "expiry", spot=100, rate=0.03,
                       dividend=0.01, model="svi")
    assert list(surf.arbitrage().to_frame()["free of arbitrage"]) == ["yes", "no"]


def test_the_fitted_surface_in_3d():
    """The Surface tab: the model's volatility over a grid of strikes (or log moneyness) and expiries, against Hagan's
    SABR formula at the parameters that made the quotes, drawn as a 3-D surface."""
    import json

    from pymetrix.figures import Surface
    from pymetrix.saved import _decode_view, _encode_view
    from pymetrix.views import view

    K = np.tile(np.arange(70.0, 136.0, 5.0), 3)
    T = np.repeat([0.25, 1.0, 2.0], 14)
    F = 100 * np.exp(0.02 * T)
    vols = np.concatenate([sf.sabr_vol(K[T == t], F[T == t][0], t, 0.22, 1.0, -0.5, 0.8) for t in np.unique(T)])
    sabr = vol_surface(_quotes(black_scholes(100, K, T, 0.03, vols, 0.01), K, T), "price", "strike", "expiry", spot=100, rate=0.03,
                       dividend=0.01, model="sabr")
    grid = sabr.grid(strikes=8, expiries=5)
    assert grid.shape == (5, 8) and list(grid.index) == pytest.approx([0.25, 0.6875, 1.125, 1.5625, 2.0])
    strikes = grid.columns.to_numpy()

    def hagan(t):  # at a quoted expiry: Hagan's formula, at the parameters that made the quotes
        return sf.sabr_vol(strikes, 100 * np.exp(0.02 * t), t, 0.22, 1.0, -0.5, 0.8)

    for t in (0.25, 2.0):
        assert grid.loc[t].to_numpy() == pytest.approx(hagan(t), rel=1e-5)
    w1, w2 = hagan(1.0) ** 2 * 1.0, hagan(2.0) ** 2 * 2.0  # between: total variance interpolated linearly in expiry
    assert grid.loc[1.125].to_numpy() == pytest.approx(np.sqrt((w1 + 0.125 * (w2 - w1)) / 1.125), rel=1e-5)
    k = sabr.grid("moneyness", strikes=6, expiries=3)
    assert k.columns.name == "log moneyness" and k.columns[0] == pytest.approx(sabr.quotes["k"].min())
    t = k.index[1]
    strikes = 100 * np.exp(0.02 * t) * np.exp(k.columns.to_numpy())  # K = F e^k at each expiry
    assert k.loc[t].to_numpy() == pytest.approx(sabr.vol(strikes, t))
    fig = view(sabr, "Surface")
    (layer,) = fig.panels[0].layers
    assert isinstance(layer, Surface) and fig.panels[0].projection == "3d"
    drawn = fig.draw()
    ax = next(a for a in drawn.axes if a.name == "3d")
    Z = layer.values.to_numpy()
    corners = [Z[r:r + 2, c:c + 2].mean() for r in range(Z.shape[0] - 1) for c in range(Z.shape[1] - 1)]
    assert np.asarray(ax.collections[0].get_array()) == pytest.approx(corners)
    assert ax.get_zlabel() == "implied volatility" and ax.get_ylabel() == "expiry (years)"
    assert _decode_view(json.loads(json.dumps(_encode_view(fig)))).png() == fig.png()
    one = vol_surface(_quotes(black_scholes(100, K[T == 1.0], 1.0, 0.03, vols[T == 1.0], 0.01), K[T == 1.0], np.ones(14)), "price",
                      "strike", "expiry", spot=100, rate=0.03, dividend=0.01, model="sabr")
    with pytest.raises(ValueError, match="two or more expiries"):
        one.plot_surface()
    assert "Surface" not in [title for title, _ in __import__("pymetrix.views", fromlist=["view_sections"]).view_sections(one)]
