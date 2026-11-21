"""Cross-checks of hand-written pieces that lacked one, against statsmodels and SciPy (library audit, October 2026).

Each test computes the same thing two ways: with pymetrix, and directly with the reference library.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sm = pytest.importorskip("statsmodels.api")

from pymetrix import OLS, Cluster, Dataset, Equation, LocalProjection  # noqa: E402
from pymetrix.filters import HP, MovingAverage  # noqa: E402

DATA = Path(__file__).parents[1] / "examples" / "sample_macro.parquet"


@pytest.fixture(scope="module")
def macro():
    return Dataset.read(str(DATA)).with_series(ly="100*log(y)", lc="100*log(c)")


def test_two_way_cluster_covariance():
    rng = np.random.default_rng(3)
    rows = [{"country": c, "year": y, "x": rng.normal(), "e": rng.normal()} for c in range(25) for y in range(2000, 2016)]
    f = pd.DataFrame(rows)
    f["y"] = 1 + 0.5 * f.x + f.e + f.country.map(dict(enumerate(rng.normal(size=25)))) + f.year.map(dict(zip(range(2000, 2016), rng.normal(size=16))))
    data = Dataset.from_frame(f, time="year", entity="country", freq="Y")
    ours = Equation("y ~ x").fit(data, method=OLS(cov=Cluster("country", "year")))
    frame = data.frame.reset_index()
    groups = np.column_stack([pd.factorize(frame.country)[0], pd.factorize(frame.year.astype(str))[0]])
    ref = sm.OLS(frame.y, sm.add_constant(frame.x)).fit(cov_type="cluster", cov_kwds={"groups": groups})
    assert np.allclose(ours.params.values, ref.params.values)
    assert np.allclose(ours.params.se, ref.bse.values, rtol=1e-8)


def test_nonlinear_least_squares_against_scipy():
    from scipy.optimize import curve_fit

    rng = np.random.default_rng(4)
    x = rng.uniform(0, 3, 150)
    y = 2.0 + 1.5 * np.exp(0.6 * x) + rng.normal(scale=0.3, size=150)
    data = Dataset.from_frame(pd.DataFrame({"i": range(150), "x": x, "y": y}), entity="i")
    ours = Equation("y ~ a + b*exp(c*x)", params=["a", "b", "c"]).fit(data, method=OLS(start={"a": 1, "b": 1, "c": 0.5}))
    popt, pcov = curve_fit(lambda x, a, b, c: a + b * np.exp(c * x), x, y, p0=[1, 1, 0.5])
    assert np.allclose(ours.params.values, popt, rtol=1e-6)
    assert np.allclose(ours.params.se, np.sqrt(np.diag(pcov)), rtol=1e-4)


def test_likelihood_ratio_test(macro):
    from pymetrix.stats.compare import compare

    small = Equation("lc ~ ly").fit(macro, span=("2000Q4", None))
    big = Equation("lc ~ ly + lc[-1] + ly[-1]").fit(macro, span=("2000Q4", None))
    ours = compare(small, big, test="lr")
    f = macro.frame
    rows = f.index >= pd.Period("2000Q4", "Q")
    Xb = sm.add_constant(pd.DataFrame({"ly": f.ly, "lc1": f.lc.shift(), "ly1": f.ly.shift()})[rows])
    ref_big = sm.OLS(f.lc[rows], Xb).fit()
    ref_small = sm.OLS(f.lc[rows], sm.add_constant(f.ly[rows])).fit()
    stat, p, df = ref_big.compare_lr_test(ref_small)
    assert ours.statistic == pytest.approx(stat, rel=1e-10) and ours.df == df and ours.p_value == pytest.approx(p, rel=1e-8)


def test_local_projections_are_horizon_by_horizon_ols(macro):
    lp = LocalProjection("lc", "ly", horizons=4, lags=2).fit(macro)
    f = macro.frame
    for h in (0, 2, 4):
        X = pd.DataFrame({"ly": f.ly, "lc1": f.lc.shift(1), "lc2": f.lc.shift(2), "ly1": f.ly.shift(1), "ly2": f.ly.shift(2)})
        y = f.lc.shift(-h)
        ok = X.notna().all(axis=1) & y.notna()
        ref = sm.OLS(y[ok], sm.add_constant(X[ok])).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1, "use_correction": True})
        row = lp.irf.loc[h]
        assert row["response"] == pytest.approx(ref.params["ly"], rel=1e-10)
        assert row["se"] == pytest.approx(ref.bse["ly"], rel=1e-8)
        assert row["nobs"] == ok.sum()


def test_rolling_regressions_against_rolling_ols(macro):
    from statsmodels.regression.rolling import RollingOLS

    rolling = Equation("lc ~ ly").fit_rolling(macro, 20)
    f = macro.frame
    ref = RollingOLS(f.lc, sm.add_constant(f.ly), window=20).fit()
    ours = rolling.params
    ours = ours.frame if hasattr(ours, "frame") else ours
    ref_params = ref.params.dropna()
    ref_params.index = pd.PeriodIndex(ref_params.index, freq="Q")
    common = ours.index.intersection(ref_params.index)
    assert len(common) == len(ref_params) == len(ours)
    assert np.allclose(ours.loc[common, ["const", "ly"]].to_numpy(), ref_params.loc[common, ["const", "ly"]].to_numpy())


def test_one_sided_hp_is_the_end_point_of_expanding_two_sided_filters(macro):
    from statsmodels.tsa.filters.hp_filter import hpfilter

    y = macro["ly"].to_numpy()
    ours = macro.filter("ly", HP(one_sided=True), derived=False)["ly_trend"].to_numpy()
    ref = np.array([hpfilter(y[: t + 1], 1600)[1][-1] if t >= 2 else y[t] for t in range(len(y))])
    assert np.allclose(ours, ref, atol=1e-8)


def test_moving_average_is_pandas_rolling(macro):
    for centered in (True, False):
        ours = macro.filter("ly", MovingAverage(window=5, centered=centered), derived=False)["ly_trend"]
        ref = macro["ly"].rolling(5, center=centered).mean()
        assert np.allclose(ours, ref, equal_nan=True)
