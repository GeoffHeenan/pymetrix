"""Moments of series, checked independently.

- ``Dataset.describe``: count, mean, std, min, percentiles and max against pandas' own describe; skewness and excess
  kurtosis against scipy.stats.skew and kurtosis with bias=False (the adjusted Fisher-Pearson statistics G1 and G2,
  as pandas' skew and kurt and Excel's SKEW and KURT), to 1e-12.
- ``corr`` (Pearson, Spearman, Kendall tau-b) against scipy.stats.pearsonr, spearmanr and kendalltau on each pair's
  complete periods; ``cov`` against numpy.cov; ``partial_corr`` against the correlation of OLS residuals computed by
  hand (numpy.linalg.lstsq), and for one series held constant against the textbook formula
  (r_xy - r_xz r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2)).
- ``crosscorrelogram`` against a plain loop over numpy (sums over T, full-sample moments), its lag 0 against the Pearson
  correlation, and its band 1.96/sqrt(T); ``autocov`` against the same loop and statsmodels' acovf.
- Moving-window functions (movavg, movsum, movsd, movvar, movmin, movmax, movcov, movcorr, movbeta) against pandas'
  rolling functions (beta as rolling cov / rolling var), missing until a full window; per entity in a panel (each
  entity against pandas on its own rows); gaps in the dates respected; derivatives against finite differences.
- The dialogs: the code they record reproduces their results.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pymetrix import Dataset
from pymetrix.expr import Name, diff, evaluate, parse_expression


def _data(n=120, seed=7):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, 3))
    x = np.cumsum(e[:, 0]) * 0.1 + e[:, 0]
    y = 0.6 * np.roll(x, 2) + e[:, 1] ** 2
    z = 0.3 * x - 0.4 * y + rng.standard_t(4, n)
    frame = pd.DataFrame({"x": x, "y": y, "z": z}, index=pd.period_range("1990Q1", periods=n, freq="Q"))
    frame.iloc[5, 1] = np.nan
    return Dataset.from_frame(frame)


def test_describe_against_pandas_and_scipy():
    data = _data()
    frame = data.frame[["x", "y", "z"]].astype(float)
    table = data.describe().to_frame()
    assert list(table.columns) == ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "skewness", "kurtosis"]
    want = frame.describe().T
    np.testing.assert_allclose(table[want.columns].to_numpy(float), want.to_numpy(float), rtol=1e-12)
    for name in ["x", "y", "z"]:
        v = frame[name].dropna().to_numpy()
        assert table.loc[name, "skewness"] == pytest.approx(stats.skew(v, bias=False), rel=1e-12)
        assert table.loc[name, "kurtosis"] == pytest.approx(stats.kurtosis(v, fisher=True, bias=False), rel=1e-12)
    some = data.describe(["x", "z"], percentiles=(0.05, 0.95)).to_frame()
    assert list(some.index) == ["x", "z"] and list(some.columns[4:7]) == ["5%", "50%", "95%"]
    assert some.loc["z", "95%"] == pytest.approx(np.percentile(frame["z"], 95), rel=1e-12)
    assert any("excess kurtosis" in n for n in data.describe().notes)


def test_describe_uses_the_active_sample():
    data = _data().with_sample("1995Q1:2010Q4")
    table = data.describe(["x"]).to_frame()
    assert table.loc["x", "count"] == 64 and any(n.startswith("Sample:") for n in data.describe().notes)


@pytest.mark.parametrize("method, reference", [("pearson", stats.pearsonr), ("spearman", stats.spearmanr), ("kendall", stats.kendalltau)])
def test_correlations_against_scipy(method, reference):
    data = _data()
    table = data.corr(["x", "y", "z"], method=method).to_frame()
    frame = data.frame
    for a in "xyz":
        for b in "xyz":
            both = frame[[a, b]].dropna()
            want = 1.0 if a == b else reference(both[a], both[b])[0]
            assert table.loc[a, b] == pytest.approx(want, rel=1e-12)


def test_covariances_against_numpy():
    data = _data()
    table = data.cov(["x", "z"]).to_frame()
    np.testing.assert_allclose(table.to_numpy(float), np.cov(data.frame[["x", "z"]].to_numpy().T), rtol=1e-12)
    assert data.cov(["x", "y"]).to_frame().loc["x", "y"] == pytest.approx(np.cov(data.frame[["x", "y"]].dropna().to_numpy().T)[0, 1], rel=1e-12)


def test_partial_correlations_against_ols_residuals_by_hand():
    data = _data().with_series(w="x^2")
    table = data.partial_corr(["x", "y", "z"], given=["w"]).to_frame()
    both = data.frame[["x", "y", "z", "w"]].dropna()
    Z = np.column_stack([np.ones(len(both)), both["w"]])
    resid = {}
    for name in "xyz":
        beta = np.linalg.lstsq(Z, both[name].to_numpy(), rcond=None)[0]
        resid[name] = both[name].to_numpy() - Z @ beta
    for a in "xyz":
        for b in "xyz":
            assert table.loc[a, b] == pytest.approx(np.corrcoef(resid[a], resid[b])[0, 1], rel=1e-10)
    # one series held constant: the textbook formula from the simple correlations
    r = both[["x", "y", "w"]].corr()
    want = (r.loc["x", "y"] - r.loc["x", "w"] * r.loc["y", "w"]) / np.sqrt((1 - r.loc["x", "w"] ** 2) * (1 - r.loc["y", "w"] ** 2))
    assert data.partial_corr(["x", "y"], given="w").to_frame().loc["x", "y"] == pytest.approx(want, rel=1e-10)
    with pytest.raises(Exception, match="both correlated and held constant"):
        data.partial_corr(["x", "w"], given="w")


def test_cross_correlogram_against_numpy_by_hand():
    data = _data()
    lags = 6
    table = data.crosscorrelogram("x", "y", lags=lags).to_frame()
    both = data.frame[["x", "y"]].dropna()
    x, y = both["x"].to_numpy(), both["y"].to_numpy()
    n = len(x)
    for k in range(-lags, lags + 1):
        total = sum((x[t] - x.mean()) * (y[t + k] - y.mean()) for t in range(n) if 0 <= t + k < n)
        want = total / n / (x.std() * y.std())
        assert table.loc[table["lag"] == k, "correlation"].item() == pytest.approx(want, rel=1e-12)
    assert table.loc[table["lag"] == 0, "correlation"].item() == pytest.approx(stats.pearsonr(x, y)[0], rel=1e-12)
    assert f"±{1.96 / np.sqrt(n):.3f}" in data.crosscorrelogram("x", "y", lags=lags).notes[0]
    # y = x shifted two periods ahead peaks at k = 2 when y is x two periods later
    shifted = data.with_series(ahead="x[-2]")
    peak = shifted.crosscorrelogram("x", "ahead", lags=4).to_frame()
    assert peak.loc[peak["correlation"].idxmax(), "lag"] == 2


def test_autocovariances_by_hand_and_against_statsmodels():
    from statsmodels.tsa.stattools import acovf

    data = _data()
    table = data.autocov("z", lags=8).to_frame()
    z = data.frame["z"].to_numpy()
    np.testing.assert_allclose(table["autocovariance"].to_numpy(), acovf(z, nlag=8, fft=False), rtol=1e-12)
    n = len(z)
    for k in range(9):
        assert table["autocovariance"].iloc[k] == pytest.approx(np.dot(z[: n - k] - z.mean(), z[k:] - z.mean()) / n, rel=1e-12)
    assert table["autocorrelation"].iloc[0] == 1.0


ROLLING = [
    ("movavg(x, 5)", lambda f: f["x"].rolling(5).mean()),
    ("movsum(x, 4)", lambda f: f["x"].rolling(4).sum()),
    ("movsd(x, 6)", lambda f: f["x"].rolling(6).std()),
    ("movvar(x, 6)", lambda f: f["x"].rolling(6).var()),
    ("movmin(x, 3)", lambda f: f["x"].rolling(3).min()),
    ("movmax(x, 3)", lambda f: f["x"].rolling(3).max()),
    ("movcov(x, z, 8)", lambda f: f["x"].rolling(8).cov(f["z"])),
    ("movcorr(x, z, 8)", lambda f: f["x"].rolling(8).corr(f["z"])),
    ("movbeta(z, x, 10)", lambda f: f["z"].rolling(10).cov(f["x"]) / f["x"].rolling(10).var()),
    ("movavg(dlog(exp(x)), 3)", lambda f: (f["x"] - f["x"].shift(1)).rolling(3).mean()),
    ("movavg(y, 4)", lambda f: f["y"].rolling(4).mean()),  # y has a missing value: the windows over it are missing
]


@pytest.mark.parametrize("text, reference", ROLLING, ids=[r[0] for r in ROLLING])
def test_moving_functions_against_pandas(text, reference):
    data = _data()
    got = data.eval(text)
    want = reference(data.frame)
    np.testing.assert_allclose(got.to_numpy(), want.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)
    assert got.isna().sum() == want.isna().sum()


def test_moving_functions_per_entity_in_a_panel_and_across_gaps():
    rng = np.random.default_rng(3)
    rows = []
    for country, n in [("AUS", 30), ("NZL", 25)]:
        for t, period in enumerate(pd.period_range("2000Q1", periods=n, freq="Q")):
            rows.append({"country": country, "time": period, "x": rng.standard_normal(), "z": rng.standard_normal()})
    frame = pd.DataFrame(rows).set_index(["country", "time"])
    panel = Dataset.from_frame(frame.reset_index(), time="time", entity="country")
    for text, reference in ROLLING[:9]:
        got = panel.eval(text)
        for country in ["AUS", "NZL"]:
            want = reference(frame.xs(country, level="country"))
            np.testing.assert_allclose(got.xs(country, level="country").to_numpy(), want.to_numpy(), rtol=1e-9, atol=1e-12,
                                       equal_nan=True, err_msg=f"{text} {country}")
    # a missing period: the windows that would span it are missing, not computed over the wrong periods
    gappy = Dataset.from_frame(frame.xs("AUS", level="country").drop(pd.Period("2001Q2", "Q")))
    avg = gappy.eval("movavg(x, 3)")
    assert avg.loc[pd.Period("2001Q3", "Q"):pd.Period("2001Q4", "Q")].isna().all()
    assert not np.isnan(avg.loc[pd.Period("2002Q1", "Q")])


def test_moving_functions_in_expressions_print_differentiate_and_compile():
    from pymetrix.expr.compile import compile_exprs

    e = parse_expression("movcorr(x, z, 4) + movbeta(z, x[-1], 3) + movsd(x, 3)")
    assert str(e) == "movcorr(x, z, 4) + movbeta(z, x[-1], 3) + movsd(x, 3)" and parse_expression(str(e)) == e
    assert {n.lag for n in e.refs() if n.id == "x"} == {0, -1, -2, -3}
    rng = np.random.default_rng(1)
    point = {(n.id, n.lag): rng.standard_normal() for n in e.refs()}
    f = lambda p: float(evaluate(e, lambda name, lag: p[(name, lag)]))
    for ref in e.refs():
        h = 1e-6
        up, down = dict(point), dict(point)
        up[(ref.id, ref.lag)] += h
        down[(ref.id, ref.lag)] -= h
        exact = float(evaluate(diff(e, Name(ref.id, ref.lag)), lambda name, lag: point[(name, lag)]))
        assert exact == pytest.approx((f(up) - f(down)) / (2 * h), rel=1e-5, abs=1e-7), str(ref)
    compiled = compile_exprs([e])
    assert float(compiled([point[r] for r in compiled.refs])[0]) == pytest.approx(f(point), rel=1e-12)
    with pytest.raises(Exception, match="cannot be differentiated"):
        diff(parse_expression("movmax(x, 3)"), Name("x", -1))
    with pytest.raises(Exception, match="window length must be an integer of at least 2"):
        parse_expression("movsd(x, 1)")


def test_the_dialogs_record_code_that_reproduces_their_results():
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import Correlations, CrossCorrelogram, DescriptiveStatistics

    s = Session()
    s.run_code("data = Dataset.example('sample_macro').derive(dly='dlog(y)', dlc='dlog(c)', dli='dlog(i)')")
    data = s["data"]
    steps = [
        (DescriptiveStatistics(series="dly, dlc", percentiles="5, 50, 95", name="stats"), data.describe(["dly", "dlc"], percentiles=(0.05, 0.5, 0.95))),
        (Correlations(series="dly, dlc, dli", method="kendall", name="kendall"), data.corr(["dly", "dlc", "dli"], method="kendall")),
        (Correlations(series="dly, dlc", show="covariance", name="cov"), data.cov(["dly", "dlc"])),
        (Correlations(series="dlc, dli", show="partial", given="dly", name="partial"), data.partial_corr(["dlc", "dli"], given="dly")),
        (CrossCorrelogram(x="dly", y="dlc", lags=4, name="ccf"), data.crosscorrelogram("dly", "dlc", lags=4)),
        (CrossCorrelogram(x="dly", lags=4, name="acov"), data.autocov("dly", lags=4)),
    ]
    for cmd, want in steps:
        assert cmd.validate(s) == []
        s.run(cmd)
        pd.testing.assert_frame_equal(s[cmd.name].to_frame(), want.to_frame())
    script = s.script()
    assert "stats = data.describe(('dly', 'dlc'), percentiles=(0.05, 0.5, 0.95))" in script
    assert "partial = data.partial_corr(('dlc', 'dli'), given=('dly',))" in script
    replay = Session()
    replay.run_code(script.replace("from pymetrix import *", ""))
    for cmd, want in steps:
        pd.testing.assert_frame_equal(replay[cmd.name].to_frame(), want.to_frame())
    assert Correlations(series="dly", show="partial").validate(s) == ["choose two or more series for partial correlations"]
