"""Cointegrating regressions and the Gregory-Hansen test, checked independently: the long-run covariance and its
automatic bandwidth against arch's Bartlett estimator; FMOLS, CCR and DOLS (with arch's BIC choice of leads and lags)
against arch.unitroot.cointegration, estimates and standard errors, with a constant and with a trend (arch 9 with its
own automatic bandwidth; arch 7 and 8 with a whole-number bandwidth handed over; tests/test_lrv_bandwidth.py checks the
rule against arch 9's numbers whatever arch is installed); the
Gregory-Hansen statistic at its break as statsmodels' ADF t on that regression's residuals; its simulated 5% critical
values against Gregory and Hansen's table; a simulated break date recovered; a span and a row condition (where) against
arch on just those rows, and the dialog's Start, End and Where against the API call and its replayed script."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, cointegrating_regression, gregory_hansen
from pymetrix.cointegrating_regression import _gh_statistic, long_run_covariance, newey_west_bandwidth
from pymetrix.errors import SpecificationError

arch_coint = pytest.importorskip("arch.unitroot.cointegration")

# arch 9.0 changed its automatic Bartlett bandwidth to Newey and West's (1994) rule as R's sandwich has it (the pilot
# lag's integer part, S - 1 as the bandwidth b of the weights 1 - j/(b+1)) and, for a fractional b, includes lag
# floor(b) + 1. pymetrix's rule is arch 9's (pymetrix 0.1.0a18's, arch's before 9.0, was removed after 0.1.0a20).
# arch 9 is compared with no bandwidth handed over. arch 7 and 8 (the oldest-versions job installs arch 7) choose
# another automatic bandwidth, and leave out lag floor(b) + 1 of a fractional one, so they are given pymetrix's
# bandwidth rounded up to a whole number, with which the two use the same lags 1..b; pymetrix is given it too.
ARCH_BEFORE_9 = int(pytest.importorskip("arch").__version__.split(".")[0]) < 9


def _whole(b: float) -> float | None:
    """The bandwidth to hand arch: None (its own automatic choice) with arch 9; b rounded up before 9."""
    return float(np.ceil(b)) if ARCH_BEFORE_9 else None


@pytest.fixture(scope="module")
def system():
    rng = np.random.default_rng(0)
    T = 400
    v = rng.normal(size=(T, 2))
    x = np.cumsum(v, axis=0)
    u = np.zeros(T)
    e = 0.5 * v[:, 0] + rng.normal(size=T)  # endogenous regressors, serially correlated errors
    for t in range(1, T):
        u[t] = 0.5 * u[t - 1] + e[t]
    y = 1 + 2 * x[:, 0] - x[:, 1] + u
    data = Dataset.from_frame(pd.DataFrame({"y": y, "a": x[:, 0], "b": x[:, 1]}, index=pd.period_range("1900Q1", periods=T, freq="Q")))
    return data, pd.Series(y, name="y"), pd.DataFrame(x, columns=["a", "b"])


def test_the_long_run_covariance_against_arch():
    from arch.covariance.kernel import Bartlett

    e = np.random.default_rng(1).normal(size=(300, 3)).cumsum(0) * 0.01 + np.random.default_rng(2).normal(size=(300, 3))
    for force_int in (True, False):
        omega, lam, sigma, b = long_run_covariance(e, force_int=force_int)
        if ARCH_BEFORE_9:
            if b != np.ceil(b):
                continue  # arch 7 and 8 leave out lag floor(b) + 1 of a fractional bandwidth
            auto = Bartlett(e, bandwidth=b, center=False, force_int=force_int)
        else:
            auto = Bartlett(e, center=False, force_int=force_int)  # arch's own automatic bandwidth
        assert b == pytest.approx(auto.bandwidth, rel=1e-12)
        assert omega == pytest.approx(np.asarray(auto.cov.long_run), abs=1e-12)
        assert lam == pytest.approx(np.asarray(auto.cov.one_sided), abs=1e-12)
    # with 100 rows the pilot lag 4 (T/100)^(2/9) is exactly 4 under either rounding, so Newey and West's S agrees
    # with every arch version's estimate
    e100 = e[:100]
    assert newey_west_bandwidth(e100) == pytest.approx(Bartlett(e100, center=False).opt_bandwidth, rel=1e-12)


@pytest.mark.parametrize("trend", ["c", "ct"])
@pytest.mark.parametrize("method", ["fmols", "ccr", "dols"])
def test_against_arch(system, method, trend):
    data, Y, X = system
    ours = cointegrating_regression(data, "y", ["a", "b"], method=method, trend=trend)
    cls = {"fmols": arch_coint.FullyModifiedOLS, "ccr": arch_coint.CanonicalCointegratingReg, "dols": arch_coint.DynamicOLS}[method]
    given = _whole(ours.bandwidth)
    if given is not None:
        ours = cointegrating_regression(data, "y", ["a", "b"], method=method, trend=trend, bandwidth=given)
    ref = cls(Y, X, trend=trend).fit(bandwidth=given)
    m = len(ours.params)
    assert ours.bandwidth == pytest.approx(ref.bandwidth, rel=1e-10)
    assert ours.params == pytest.approx(ref.params.to_numpy()[:m], abs=1e-10)
    assert ours.se == pytest.approx(ref.std_errors.to_numpy()[:m], abs=1e-10)
    if method == "dols":
        assert (ours.leads, ours.lags) == (ref.leads, ref.lags)
    # near the true 2; not within a multiple of its standard error, since with a trend Newey and West's automatic
    # bandwidth (arch 9's and sandwich's) is 0 for FMOLS and CCR here, which ignores the errors' serial correlation
    assert ours.params[0] == pytest.approx(2, abs=0.1) and ours.names[:2] == ("a", "b")


def test_dols_with_given_leads_and_lags(system):
    data, Y, X = system
    ours = cointegrating_regression(data, "y", ["a", "b"], method="dols", leads=2, lags=3)
    given = _whole(ours.bandwidth)
    if given is not None:
        ours = cointegrating_regression(data, "y", ["a", "b"], method="dols", leads=2, lags=3, bandwidth=given)
    ref = arch_coint.DynamicOLS(Y, X, trend="c", leads=2, lags=3).fit(bandwidth=given)
    assert ours.params == pytest.approx(ref.params.to_numpy()[:3], abs=1e-10)
    assert ours.se == pytest.approx(ref.std_errors.to_numpy()[:3], abs=1e-10)
    b = 5.0 if ARCH_BEFORE_9 else 4.5  # a fractional bandwidth given (arch 9: lag 5 used as here)
    given = cointegrating_regression(data, "y", ["a", "b"], method="dols", leads=2, lags=3, bandwidth=b)
    ref = arch_coint.DynamicOLS(Y, X, trend="c", leads=2, lags=3).fit(bandwidth=b)
    assert given.se == pytest.approx(ref.std_errors.to_numpy()[:3], abs=1e-10)


def test_gregory_hansen():
    from statsmodels.tsa.stattools import adfuller

    rng = np.random.default_rng(3)
    T = 200
    x = np.cumsum(rng.normal(size=T))
    y = 1 + x + np.where(np.arange(T) > 120, 3.0, 0.0) + rng.normal(size=T) * 0.5  # cointegrated, the level shifts at 121
    data = Dataset.from_frame(pd.DataFrame({"y": y, "x": x}, index=pd.period_range("1950Q1", periods=T, freq="Q")))
    res = gregory_hansen(data, "y", ["x"], model="level", reps=200)
    stat, tb = _gh_statistic(y, x[:, None], "level", 1, 0.15)
    Z = np.column_stack([np.ones(T), (np.arange(T) > tb).astype(float), x])
    u = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    assert stat == pytest.approx(adfuller(u, maxlag=1, regression="n", autolag=None)[0], rel=1e-10)
    assert abs(tb - 120) <= 2 and res.p_value < 0.01 and res.statistic == stat


@pytest.mark.parametrize("model, table", [("level", -4.61), ("regime", -4.95)])
def test_simulated_critical_values_against_the_table(model, table):
    rng = np.random.default_rng(4)
    walk = lambda: np.cumsum(rng.standard_normal(300))  # noqa: E731
    data = Dataset.from_frame(pd.DataFrame({"y": walk(), "x": walk()}, index=pd.period_range("1900Q1", periods=300, freq="Q")))
    res = gregory_hansen(data, "y", ["x"], model=model, reps=400, seed=5)
    assert res.critical_values["5%"] == pytest.approx(table, abs=0.25)  # Gregory and Hansen's asymptotic value, m = 1
    with pytest.raises(SpecificationError, match="model"):
        gregory_hansen(data, "y", ["x"], model="bad")


def test_the_dialog(system):
    from pymetrix.commands import COMMANDS, Session

    data, _, _ = system
    s = Session()
    s.objects["data"] = data
    s.run(COMMANDS["FitCointegratingRegression"].cls("y", "a, b", method="dols"))
    assert s.history[-1].code.startswith("coint = cointegrating_regression(data, 'y', ['a', 'b'], method='dols'")
    assert s.objects["coint"].method == "dols"


def test_span_and_where_choose_the_rows(system):
    """Start, End and a row condition: the estimates equal arch's on just those rows (taken as consecutive periods)."""
    data, Y, X = system
    flagged = Dataset.from_frame(data.frame.assign(keep=np.where(np.arange(len(Y)) % 50 < 45, 1.0, 0.0)))  # 5 rows in 50 out
    span = ("1910Q1", "1980Q4")
    rows = (np.asarray(data.frame.index >= pd.Period("1910Q1")) & np.asarray(data.frame.index <= pd.Period("1980Q4"))
            & (flagged["keep"].to_numpy() == 1))
    ours = cointegrating_regression(flagged, "y", ["a", "b"], method="fmols", span=span, where="keep == 1")
    ref = arch_coint.FullyModifiedOLS(Y[rows].reset_index(drop=True), X[rows].reset_index(drop=True),
                                      trend="c").fit(bandwidth=_whole(ours.bandwidth))  # FMOLS's b is a whole number
    assert len(ours.resid) == rows.sum() < 284
    assert ours.params == pytest.approx(ref.params.to_numpy()[:3], abs=1e-10) and ours.se == pytest.approx(ref.std_errors.to_numpy()[:3], abs=1e-10)
    whole = cointegrating_regression(data, "y", ["a", "b"])
    assert cointegrating_regression(data, "y", ["a", "b"], where=None).params == pytest.approx(whole.params, abs=0)  # defaults: as before
    with pytest.raises(SpecificationError, match="too few"):
        cointegrating_regression(flagged, "y", ["a", "b"], where="keep == 2")


def test_the_dialog_with_start_end_and_where(system):
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import FitCointegratingRegression

    data, _, _ = system
    flagged = Dataset.from_frame(data.frame.assign(keep=np.where(np.arange(400) % 50 < 45, 1.0, 0.0)))
    s = Session()
    s.objects["data"] = flagged
    s.run(FitCointegratingRegression("y", "a, b", method="ccr", start="1910Q1", end="1980Q4", where="keep == 1", name="c1"))
    s.run(FitCointegratingRegression("y", "a, b", method="dols", end="1990Q4", name="c2"))
    by_api = cointegrating_regression(flagged, "y", ["a", "b"], method="ccr", span=("1910Q1", "1980Q4"), where="keep == 1")
    np.testing.assert_array_equal(s["c1"].params, by_api.params)
    np.testing.assert_array_equal(s["c1"].cov, by_api.cov)
    np.testing.assert_array_equal(s["c2"].params, cointegrating_regression(flagged, "y", ["a", "b"], method="dols", span=(None, "1990Q4")).params)
    script = s.script()
    assert "span=('1910Q1', '1980Q4'), where='keep == 1'" in script and "span=(None, '1990Q4')" in script
    ns = dict(s.objects)
    exec(script, ns)
    np.testing.assert_array_equal(ns["c1"].params, by_api.params)
    np.testing.assert_array_equal(ns["c1"].cov, by_api.cov)
    assert "span" not in FitCointegratingRegression("y", "a").to_call().to_code()  # empty boxes: the whole sample, as before
