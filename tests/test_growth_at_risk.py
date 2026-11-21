"""Growth at risk, checked independently: the skewed t integrates to one, is Student's t without skew and near the
skew normal with many degrees of freedom; its quantiles and shortfall by numerical integration; known parameters
recovered from their quantiles; the quantile regressions against statsmodels' QuantReg; the slopes of a
location-scale model recovered at each quantile; quantiles never crossing."""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, optimize, stats

import importlib

from pymetrix import Dataset, Equation, growth_at_risk
from pymetrix.estimation.quantile import Quantile

g = importlib.import_module("pymetrix.growth_at_risk")  # the module (pymetrix exports its function under the same name)


def test_the_skewed_t():
    assert integrate.quad(lambda x: g.skewt_pdf(x, 0.3, 1.2, 2.0, 5.0), -np.inf, np.inf)[0] == pytest.approx(1, abs=1e-8)
    x = np.linspace(-4, 4, 9)
    assert g.skewt_pdf(x, 0.5, 2.0, 0.0, 7.0) == pytest.approx(stats.t.pdf(x, 7, 0.5, 2.0), rel=1e-12)
    assert g.skewt_pdf(x, 0.0, 1.0, 3.0, 1e6) == pytest.approx(stats.skewnorm.pdf(x, 3.0), abs=1e-5)
    p = (1.0, 2.0, -3.0, 6.0)
    cdf = lambda v: integrate.quad(lambda u: g.skewt_pdf(u, *p), -np.inf, v)[0]  # noqa: E731
    for q in (0.05, 0.5, 0.9):
        assert cdf(g.skewt_quantiles([q], *p)[0]) == pytest.approx(q, abs=1e-4)
    q05 = optimize.brentq(lambda v: cdf(v) - 0.05, -30, 10)
    tail = integrate.quad(lambda u: u * g.skewt_pdf(u, *p), -np.inf, q05)[0] / 0.05
    assert g.skewt_shortfall(0.05, *p) == pytest.approx(tail, rel=2e-3)


def test_parameters_from_their_quantiles():
    for p in ((1.0, 2.0, -3.0, 6.0), (0.0, 0.5, 1.5, 20.0), (-2.0, 1.0, 0.0, 4.0)):
        qs = g.skewt_quantiles(g.MATCHED, *p)
        assert g.fit_skewt(g.MATCHED, qs) == pytest.approx(p, rel=1e-4, abs=1e-4)


def _data(T=160, seed=0):
    rng = np.random.default_rng(seed)
    x = np.maximum(rng.normal(0, 0.5, T), -1.6)  # the scale 1 + 0.6 x stays positive (clipped beyond 3 s.d.)
    y = np.zeros(T)
    for t in range(1, T):  # location and scale both move with x: the q quantile's slope is -0.5 + 0.6 F^-1(q)
        y[t] = 2 - 0.5 * x[t - 1] + (1 + 0.6 * x[t - 1]) * rng.normal()
    return Dataset.from_frame(pd.DataFrame({"g": y, "nfci": x}, index=pd.period_range("1980Q1", periods=T, freq="Q", name="time")))


def test_the_projections_against_quantreg_and_by_construction():
    from statsmodels.regression.quantile_regression import QuantReg

    data = _data()
    gar = growth_at_risk(data, "g", ["nfci"], horizons=(1,))
    f = data.frame
    future, x = f["g"].shift(-1), f["nfci"]
    ok = future.notna() & x.notna()
    X = np.column_stack([np.ones(ok.sum()), x[ok]])
    for q in (0.05, 0.5, 0.95):
        ref = QuantReg(future[ok].to_numpy(), X).fit(q=q)
        assert gar.fits[(1, q)].params.values == pytest.approx(ref.params, rel=1e-6, abs=1e-8)
    # in a large sample, each quantile's slope is the location-scale model's (the regressions alone, without the skewed t)
    d2 = _data(T=4000, seed=1).frame
    work = Dataset.from_frame(pd.DataFrame({"future": d2["g"].shift(-1), "x": d2["nfci"]}).dropna())
    for q in (0.1, 0.5, 0.9):
        slope = Equation("future ~ x").fit(work, method=Quantile(q=q)).params["x"]
        assert slope == pytest.approx(-0.5 + 0.6 * stats.norm.ppf(q), abs=0.08)
    P = gar.quantiles(1).to_numpy()
    assert np.all(np.diff(P, axis=1) >= 0)  # rearranged: never crossing
    risk = gar.at_risk(1)
    assert (risk["expected shortfall"] <= risk["growth at risk"] + 1e-9).all()
    assert np.corrcoef(risk["growth at risk"], data.frame["nfci"].loc[risk.index])[0, 1] < -0.5  # tighter conditions, more risk


def test_the_dialog():
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS
    from pymetrix.views import sections

    s = Session()
    s.objects["data"] = _data(T=90)
    s.run(COMMANDS["FitGrowthAtRisk"].cls("g", "nfci", horizons="1"))
    assert s.history[-1].code == "gar = growth_at_risk(data, 'g', ['nfci'], horizons=[1])"
    assert [t for t, _ in sections(s.objects["gar"])][0] == "Summary"
