"""Value at risk and expected shortfall (the finance plugin), checked independently: ES's closed forms against numerical
integration; historical VaR against the order statistics; Kupiec and Christoffersen against counts made by hand; the
dynamic quantile test against statsmodels' regression; the Basel zones against the published table; and coverage on
simulated returns."""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy import stats as sps

from pymetrix import Dataset
from pymetrix_finance import RiskForecast, value_at_risk
from pymetrix_finance import risk


def _returns(r, name="r"):
    return Dataset.from_frame(pd.DataFrame({name: r}, index=pd.period_range("2001-01-01", periods=len(r), freq="D", name="time")))


def test_expected_shortfall_closed_forms_match_integration():
    p = 0.01
    v, e = risk.normal(0.001, 0.02, p)
    q = 0.001 + 0.02 * sps.norm.ppf(p)
    tail = integrate.quad(lambda x: x * sps.norm.pdf(x, 0.001, 0.02), -np.inf, q)[0] / p
    assert v == pytest.approx(-q, rel=1e-12) and e == pytest.approx(-tail, rel=1e-7)
    for nu in (3.5, 6.0, 30.0):
        v, e = risk.student_t(nu, 0.0005, 0.015, p)
        q = sps.t.ppf(p, nu, 0.0005, 0.015)
        tail = integrate.quad(lambda x: x * sps.t.pdf(x, nu, 0.0005, 0.015), -np.inf, q)[0] / p
        assert v == pytest.approx(-q, rel=1e-10) and e == pytest.approx(-tail, rel=1e-6)
    cf_v, cf_e = risk.cornish_fisher(0.0, 1.0, 0.0, 0.0, p)  # no skewness or excess kurtosis: the normal
    n_v, n_e = risk.normal(0.0, 1.0, p)
    assert cf_v == pytest.approx(n_v, rel=1e-12) and cf_e == pytest.approx(n_e, rel=2e-3)
    fat_v, _ = risk.cornish_fisher(0.0, 1.0, -0.5, 3.0, p)
    assert fat_v > n_v  # left skew and fat tails: a larger loss


def test_historical_var_is_the_order_statistic():
    rng = np.random.default_rng(1)
    x = rng.standard_t(4, 250)
    v, e = risk.historical(x, 0.01)
    s = np.sort(x)
    h = (len(x) - 1) * 0.01  # numpy's linear rule: between the order statistics floor(h) and floor(h) + 1
    q = s[int(h)] + (h - int(h)) * (s[int(h) + 1] - s[int(h)])
    assert v == pytest.approx(-q, rel=1e-12) and e == pytest.approx(-s[s <= q].mean(), rel=1e-12)


def test_the_backtests_by_hand():
    rng = np.random.default_rng(4)
    hits = rng.random(1000) < 0.02  # too many violations for 1% VaR
    p = 0.01
    n, x = len(hits), int(hits.sum())
    lr, pval = risk.kupiec(hits, p)
    by_hand = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - x / n) - x * np.log(x / n))
    assert lr == pytest.approx(by_hand, rel=1e-12) and pval < 0.05
    ch = risk.christoffersen(hits, p)
    counts = {(a, b): 0 for a in (0, 1) for b in (0, 1)}
    for a, b in zip(hits[:-1].astype(int), hits[1:].astype(int)):
        counts[a, b] += 1
    assert (ch["n00"], ch["n01"], ch["n10"], ch["n11"]) == (counts[0, 0], counts[0, 1], counts[1, 0], counts[1, 1])
    pi01, pi11 = counts[0, 1] / (counts[0, 0] + counts[0, 1]), counts[1, 1] / (counts[1, 0] + counts[1, 1]) if counts[1, 0] + counts[1, 1] else 0
    pi = (counts[0, 1] + counts[1, 1]) / (n - 1)

    def xl(a, b):
        return a * np.log(b) if a > 0 else 0.0

    lr_ind = -2 * (xl(counts[0, 0] + counts[1, 0], 1 - pi) + xl(counts[0, 1] + counts[1, 1], pi)
                   - xl(counts[0, 0], 1 - pi01) - xl(counts[0, 1], pi01) - xl(counts[1, 0], 1 - pi11) - xl(counts[1, 1], pi11))
    assert ch["LR_ind"] == pytest.approx(lr_ind, rel=1e-10)
    var = np.abs(rng.normal(2, 0.3, 1000))
    dq, p_dq, k = risk.dynamic_quantile(hits, var, p)
    import statsmodels.api as sm

    hit = hits.astype(float) - p
    X = np.column_stack([np.ones(996)] + [hit[4 - j:1000 - j] for j in range(1, 5)] + [var[4:]])
    b = sm.OLS(hit[4:], X).fit().params
    assert dq == pytest.approx(b @ X.T @ X @ b / (p * (1 - p)), rel=1e-8) and k == 6


def test_the_basel_zones():
    # 250 days of 99% VaR: green for 0-4 violations, yellow for 5-9, red for 10 or more (Basel Committee, 1996)
    assert [risk.traffic_light(x, 250, 0.01) for x in (0, 4, 5, 9, 10, 15)] == ["green", "green", "yellow", "yellow", "red", "red"]


def test_coverage_on_simulated_returns():
    rng = np.random.default_rng(7)
    iid = _returns(0.01 * rng.standard_normal(2250))
    fc = value_at_risk(iid, "r", level=0.99, method="normal", window=250)
    assert isinstance(fc, RiskForecast) and len(fc.frame) == 2000
    rate = fc.frame["violation"].mean()
    assert 0.005 < rate < 0.016
    table = fc.backtests()
    kupiec_p = table.rows[table.index.index("Kupiec: proportion of failures")][1]
    assert kupiec_p > 0.01
    n = 3000  # GARCH(1,1) returns: volatility clusters
    omega, alpha, beta = 1e-6, 0.08, 0.9
    r, s2 = np.zeros(n), omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    clustered = _returns(r)
    garch = value_at_risk(clustered, "r", method="garch", window=1000, refit=250)
    assert 0.004 < garch.frame["violation"].mean() < 0.02
    fhs = value_at_risk(clustered, "r", method="fhs", window=1000, refit=250)
    assert np.all(fhs.frame["es"] >= fhs.frame["var"])  # the shortfall is beyond the VaR
    for method in ("historical", "t", "cornish_fisher"):
        f = value_at_risk(iid, "r", method=method, window=250, start="2007-01-01")
        assert f.frame.index[0] == pd.Period("2007-01-01", "D") and np.all(f.frame["es"] >= f.frame["var"] - 1e-12)


def test_refusals_and_the_dialog():
    from pymetrix.errors import SpecificationError

    data = _returns(np.random.default_rng(0).normal(0, 0.01, 400))
    with pytest.raises(SpecificationError, match="method is one of"):
        value_at_risk(data, "r", method="magic")
    with pytest.raises(SpecificationError, match="more than the window"):
        value_at_risk(data, "r", window=500)
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS

    s = Session()
    s.objects["data"] = data
    s.run(COMMANDS["ValueAtRisk"].cls("r", method="normal", window=200))
    assert s.history[-1].code == "risk = value_at_risk(data, 'r', method='normal', window=200)"
    from pymetrix.views import sections

    assert [title for title, _ in sections(s.objects["risk"])][:2] == ["Backtests", "Graph"] or "Backtests" in [t for t, _ in sections(s.objects["risk"])]
    from pymetrix.plugins import plugin_of_command

    assert plugin_of_command(COMMANDS["ValueAtRisk"].cls).name == "pymetrix-finance"


def test_the_garch_forecast_matches_arch():
    arch = pytest.importorskip("arch")
    rng = np.random.default_rng(11)
    n, omega, alpha, beta = 2000, 1e-6, 0.08, 0.9
    r, s2 = np.zeros(n), omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    ours = risk._fit_garch(pd.Series(r, index=pd.period_range("2001-01-01", periods=n, freq="D")))
    ref = arch.arch_model(r * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
    their_next = float(ref.forecast(horizon=1).variance.iloc[-1, 0]) / 100**2
    assert np.sqrt(ours.var_next) == pytest.approx(np.sqrt(their_next), rel=0.02)  # two optimisers, one model
    assert ours.alpha == pytest.approx(ref.params["alpha[1]"], abs=0.02) and ours.beta == pytest.approx(ref.params["beta[1]"], abs=0.03)
