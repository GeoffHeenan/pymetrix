"""Risk contributions and event studies (the finance plugin), checked independently: components adding up to the
portfolio's VaR and ES (Euler); marginal VaR and ES against finite differences of the portfolio's normal VaR; the
multivariate GARCH case against its own portfolio VaR; the market model against numpy's polyfit; Patell's
standardisation against its textbook formula by a plain loop; a known abnormal return recovered; and every test's
size under the null near 5% by simulation."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, abnormal_returns, risk_contributions
from pymetrix.errors import SpecificationError
from pymetrix_finance.risk import normal


def _returns(T=1000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.normal(0.0003, 0.01, T)
    R = np.column_stack([0.0001 + b * m + rng.normal(0, s, T) for b, s in ((1.2, 0.01), (0.8, 0.015), (0.5, 0.02))])
    df = pd.DataFrame(R, columns=list("abc"), index=pd.period_range("2010-01-01", periods=T, freq="D", name="time"))
    df["mkt"] = m
    return df


def test_normal_contributions():
    df = _returns()
    d = Dataset.from_frame(df)
    w = np.array([0.5, 0.3, 0.2])
    rc = risk_contributions(d, list("abc"), w, level=0.975)
    R = df[list("abc")].to_numpy()
    mu, S = R.mean(axis=0), np.cov(R.T, ddof=1)

    def var_es(weights):
        return normal(float(weights @ mu), float(np.sqrt(weights @ S @ weights)), 0.025)

    assert (rc.var, rc.es) == pytest.approx(var_es(w), rel=1e-12)
    f = rc.frame()
    assert f["component VaR"].sum() == pytest.approx(rc.var, rel=1e-12) and f["component ES"].sum() == pytest.approx(rc.es, rel=1e-12)
    h = 1e-6
    for i in range(3):
        e = np.eye(3)[i] * h
        assert f["marginal VaR"].iloc[i] == pytest.approx((var_es(w + e)[0] - var_es(w - e)[0]) / (2 * h), rel=1e-6)
        assert f["marginal ES"].iloc[i] == pytest.approx((var_es(w + e)[1] - var_es(w - e)[1]) / (2 * h), rel=1e-6)
    assert f["share of VaR"].sum() == pytest.approx(1.0)


def test_historical_and_garch_contributions():
    df = _returns()
    d = Dataset.from_frame(df)
    w = np.array([0.5, 0.3, 0.2])
    rh = risk_contributions(d, list("abc"), w, method="historical")
    R = df[list("abc")].to_numpy()
    rp = R @ w
    q = np.quantile(rp, 0.01)
    assert rh.var == pytest.approx(-q) and rh.es == pytest.approx(-rp[rp <= q].mean())
    f = rh.frame()
    assert f["component ES"].to_numpy() == pytest.approx(-w * R[rp <= q].mean(axis=0))  # the tail periods' average, by hand
    assert f["component VaR"].sum() == pytest.approx(rh.var) and f["component ES"].sum() == pytest.approx(rh.es)

    from pymetrix import mgarch

    fit = mgarch(d, ["a", "b"], model="ccc")
    rg = risk_contributions(d, ["a", "b"], [0.6, 0.4], cov=fit)
    out = fit.portfolio_var([0.6, 0.4], level=0.99)
    assert (rg.var, rg.es) == pytest.approx((out["var"], out["es"])) and "forecast" in rg.source
    with pytest.raises(SpecificationError, match="one weight"):
        risk_contributions(d, list("abc"), [0.5, 0.5])


def _events(df, assets_dates, shock=0.0):
    out = df.copy()
    for a, dt in assets_dates:
        out.loc[pd.Period(dt, "D"), a] += shock
    return Dataset.from_frame(out)


def test_the_market_model_patell_and_a_known_effect():
    df = _returns()
    idx = df.index
    pairs = [("a", str(idx[400])), ("b", str(idx[500])), ("c", str(idx[650])), ("a", str(idx[800]))]
    ev = abnormal_returns(_events(df, pairs, 0.04), pairs, market="mkt", window=(-5, 5), car_window=(0, 0))
    assert ev.car().mean() == pytest.approx(0.04, abs=0.015) and ev.statistics()["Patell Z"][1] < 1e-4
    # event 1 by hand: the market model over periods -250 to -11, its abnormal returns and Patell's standardisation
    asset, pos = "b", 500
    est = df.iloc[pos - 250:pos - 10]
    b, a = np.polyfit(est["mkt"], est[asset], 1)
    resid = est[asset] - a - b * est["mkt"]
    s2 = (resid**2).sum() / (len(est) - 2)
    mbar, sxx = est["mkt"].mean(), ((est["mkt"] - est["mkt"].mean()) ** 2).sum()
    label = f"b {idx[500]}"
    for rel in range(-5, 6):
        t = pos + rel
        ar = df[asset].iloc[t] + (0.04 if rel == 0 else 0.0) - a - b * df["mkt"].iloc[t]
        assert ev.ar.loc[rel, label] == pytest.approx(ar, abs=1e-12)
        sd = np.sqrt(s2 * (1 + 1 / len(est) + (df["mkt"].iloc[t] - mbar) ** 2 / sxx))
        assert ev.sar.loc[rel, label] == pytest.approx(ar / sd, rel=1e-9)
    mean_model = abnormal_returns(Dataset.from_frame(df), pairs[:1], window=(-5, 5))
    est = df.iloc[400 - 250:400 - 10]["a"]
    assert mean_model.ar.iloc[:, 0].to_numpy() == pytest.approx(df["a"].iloc[395:406].to_numpy() - est.mean())
    with pytest.raises(SpecificationError, match="not in the data"):
        abnormal_returns(Dataset.from_frame(df), {"a": "1990-01-01"}, market="mkt")


def test_the_tests_have_their_size_under_the_null():
    rng = np.random.default_rng(11)
    reject = {k: 0 for k in ("cross-sectional t", "Patell Z", "BMP t", "Corrado rank")}
    reps = 200
    for r in range(reps):
        df = _returns(T=420, seed=int(rng.integers(1 << 31)))
        pairs = [(a, str(df.index[300 + 3 * j])) for j, a in enumerate("abc" * 7)]  # 21 events
        stats = abnormal_returns(Dataset.from_frame(df), pairs, market="mkt", window=(-3, 3), car_window=(0, 1)).statistics()
        for k, (_, p) in stats.items():
            reject[k] += p < 0.05
    for k, n in reject.items():
        assert 0.015 <= n / reps <= 0.095, (k, n / reps)


def test_the_dialogs():
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS
    from pymetrix.views import sections

    df = _returns()
    s = Session()
    s.objects["data"] = Dataset.from_frame(df)
    s.run(COMMANDS["DecomposeRisk"].cls("a, b, c", "0.5, 0.3, 0.2"))
    assert s.history[-1].code == "risk_parts = risk_contributions(data, ['a', 'b', 'c'], [0.5, 0.3, 0.2])"
    s.run(COMMANDS["StudyEvents"].cls(f"a: {df.index[400]}; b: {df.index[500]}, {df.index[700]}", market="mkt"))
    ev = s.objects["events"]
    assert len(ev.events) == 3 and [t for t, _ in sections(ev)][0] == "Tests"
