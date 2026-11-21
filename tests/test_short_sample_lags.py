"""ADF and Engle-Granger in short samples: the default largest lag keeps residual degrees of freedom, and a degenerate
regression is refused, never reported as a statistic of 0.0.

The case found: 16 annual observations, "Maximum lags" left empty. statsmodels' default largest lag,
min(ceil(12 (n/100)^(1/4)), n//2 - terms - 1) = 7, leaves the residuals' ADF regression (8 rows, 8 regressors) no
residual degrees of freedom: its sum of squares is zero, AIC picks that lag, and the t-ratio is 0.0, with simulated
critical values of 0 too (every draw degenerate in the same way).

Checked against: statsmodels' coint and adfuller at the capped largest lag (the statistic and lag reproduce them);
arch's engle_granger and ADF at the lag chosen (the same t-ratio, an independent implementation); statsmodels' own
default, which is shown to give the zero this guards against.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")
from statsmodels.tsa.stattools import adfuller, coint  # noqa: E402

from pymetrix import Dataset  # noqa: E402
from pymetrix.commands import RunTest, Session  # noqa: E402
from pymetrix.errors import PyMetrixError, SpecificationError  # noqa: E402
from pymetrix.stats import adf_arrays as A  # noqa: E402
from pymetrix.stats import run_test  # noqa: E402


def _quiet(f, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*args, **kw)


@pytest.fixture(scope="module")
def annual16() -> Dataset:
    """16 annual observations of two I(1) series (random walks; y loads on x)."""
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(16))
    y = 0.8 * x + 0.5 * np.cumsum(rng.standard_normal(16))
    return Dataset.from_frame(pd.DataFrame({"y": y, "x": x}, index=pd.period_range("2005", periods=16, freq="Y")))


def test_statsmodels_default_is_degenerate_here(annual16):
    """The cause: statsmodels' default largest lag for the residuals is 7, which leaves no degrees of freedom."""
    y, x = annual16.frame["y"].to_numpy(), annual16.frame[["x"]].to_numpy()
    assert _quiet(coint, y, x)[0] == 0.0
    res = _quiet(adfuller, y - x[:, 0] * np.linalg.lstsq(np.column_stack([x, np.ones(16)]), y, rcond=None)[0][0], regression="n")
    assert res[2] == 7 and res[3] == 8  # 7 lags on 8 rows: 8 regressors, no residual degrees of freedom


def test_default_max_lag_rule():
    """Schwert's ceil(12 (n/100)^(1/4)), capped at statsmodels' n//2 - terms - 1 and to keep MIN_DF = 4 residual
    degrees of freedom after the parameters already used; statsmodels' rule unchanged from 26 observations on."""
    schwert = lambda n: int(np.ceil(12 * (n / 100) ** 0.25))  # noqa: E731
    assert A.MIN_DF == 4
    assert A.default_max_lag(16, 0, 2) == 4 and A.default_max_lag(16, 1) == 4 and A.default_max_lag(16, 0) == 5
    for n in range(8, 400):
        for ntrend in (0, 1, 2, 3):
            for used in (0, 2, 3):
                try:
                    lag = A.default_max_lag(n, ntrend, used)
                except SpecificationError:
                    assert n - 2 - ntrend - used < 1  # only when even lag 0 leaves nothing
                    continue
                assert 0 <= lag <= min(schwert(n), n // 2 - ntrend - 1) or lag == 0
                df = n - 2 - 2 * lag - ntrend - used
                assert df >= A.MIN_DF or (lag == 0 and df >= 1)
                if n >= 26 + ntrend + used:
                    assert lag == min(schwert(n), n // 2 - ntrend - 1)  # statsmodels' rule


@pytest.mark.parametrize("critical_values", ["mackinnon", "simulated"])
def test_engle_granger_16_annual_default_lags(annual16, critical_values):
    r = annual16.test("engle_granger", "y, x", critical_values=critical_values, draws=500)
    y, x = annual16.frame["y"].to_numpy(), annual16.frame[["x"]].to_numpy()
    top = A.default_max_lag(16, 0, 2)  # x and the constant
    stat, _, crit = _quiet(coint, y, x, maxlag=top, autolag="aic")
    assert np.isfinite(r.statistic) and r.statistic != 0.0
    assert r.statistic == pytest.approx(stat, rel=1e-9)
    assert len(set(r.critical_values.values())) == 3 and all(v < -1 for v in r.critical_values.values())
    if critical_values == "mackinnon":
        assert list(r.critical_values.values()) == list(map(float, crit))
    else:
        assert r.details["lags used"] <= top and "warning" not in r.details
        arch = pytest.importorskip("arch.unitroot")  # an independent implementation, at the lag chosen
        ref = _quiet(arch.engle_granger, y, x[:, 0], trend="c", lags=r.details["lags used"]).stat
        assert r.statistic == pytest.approx(ref, rel=1e-9)


def test_the_simulated_null_is_not_degenerate():
    sims = A.simulate_engle_granger(16, 2, "c", draws=400, seed=3)
    assert np.isfinite(sims).all() and not np.any(sims == 0)
    sims = A.simulate_adf(16, "n", draws=400, seed=3)
    assert np.isfinite(sims).all() and not np.any(sims == 0)


@pytest.mark.parametrize("trend", ["n", "c", "ct"])
def test_adf_16_default_lags(annual16, trend):
    x = annual16.frame["y"].to_numpy()
    ntrend = 0 if trend == "n" else len(trend)
    ref = _quiet(adfuller, x, maxlag=A.default_max_lag(16, ntrend), regression=trend, autolag="AIC")
    for cv in ("mackinnon", "simulated"):
        r = annual16.test("adf", "y", trend=trend, critical_values=cv, draws=200)
        assert r.statistic != 0.0 and r.statistic == pytest.approx(ref[0], rel=1e-9)
        assert (r.details["lags used"], r.details["observations"]) == (ref[2], ref[3])


@pytest.mark.parametrize("critical_values", ["mackinnon", "simulated"])
def test_a_lag_with_no_degrees_of_freedom_is_refused(annual16, critical_values):
    for test, kw in (("engle_granger", {"variables": "y, x", "max_lags": 7}),
                     ("adf", {"variable": "y", "trend": "n", "max_lags": 7}),
                     ("adf", {"variable": "y", "trend": "n", "lags": 7})):
        with pytest.raises(SpecificationError, match="no residual degrees of freedom") as err:
            run_test(annual16, test, critical_values=critical_values, draws=100, **kw)
        assert isinstance(err.value, PyMetrixError)
    assert annual16.test("engle_granger", "y, x", max_lags=6, critical_values=critical_values, draws=100).statistic != 0


def test_a_degenerate_statistic_is_refused():
    for bad in (0.0, -0.0, float("nan")):
        with pytest.raises(SpecificationError, match="degenerate"):
            A.degenerate_statistic(bad, "ADF (y)")
    A.degenerate_statistic(-2.5, "ADF (y)")
    A.degenerate_statistic(-np.inf, "Engle-Granger")  # perfect collinearity keeps its own message


def test_panel_tests_on_short_series_use_the_cap():
    """Fisher-ADF and Pedroni search each entity's lags up to the same capped rule; with trend n and 16 periods
    statsmodels' default left every entity a zero statistic (and the same p-value)."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(5):
        x = np.cumsum(rng.standard_normal(16))
        y = x + rng.standard_normal(16)
        rows += [(f"e{i}", 2000 + t, y[t], x[t]) for t in range(16)]
    panel = Dataset.from_frame(pd.DataFrame(rows, columns=["id", "year", "y", "x"]), entity="id", time="year", freq="Y")
    fisher = panel.test("fisher_adf", "y", trend="n")
    p = []
    for _, g in panel.frame.groupby(level=0):
        p.append(_quiet(adfuller, g["y"].to_numpy(), maxlag=A.default_max_lag(16, 0), regression="n", autolag="AIC")[1])
    assert fisher.statistic == pytest.approx(-2 * np.sum(np.log(p)), rel=1e-12)
    pedroni = panel.test("pedroni", "y, x", replications=500)
    lags = pedroni.details["lags"]
    top = int(str(lags).split()[-1])
    assert top <= A.default_max_lag(16, 0, 2) and np.isfinite(pedroni.statistic)


def test_dialog_with_maximum_lags_empty_equals_the_api_and_replays(annual16):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from pymetrix.ui.qt import command_dialog

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s = Session()
    s.objects["data"] = annual16
    dialog = command_dialog(RunTest, s)
    dialog.editors["source"].widget.setCurrentText("data")
    dialog.editors["test"].widget.setCurrentText("engle_granger")
    sub = dialog.editors["options"]
    sub.sub["variables"].set("y, x")
    sub.sub["critical_values"].set("simulated")
    sub.sub["draws"].set(300)
    assert sub.sub["max_lags"].widget.text() == ""  # Maximum lags left empty
    s.run(RunTest(**{k: dialog.editors[k].get() for k in ("source", "test", "options")}, name="eg"))
    got = s["eg"]
    by_api = run_test(annual16, "engle_granger", variables="y, x", critical_values="simulated", draws=300)
    assert got.statistic != 0.0 and 0.0 not in got.critical_values.values()
    assert (got.statistic, got.p_value, got.critical_values) == (by_api.statistic, by_api.p_value, by_api.critical_values)
    ns = dict(s.objects)
    exec(s.script(), ns)
    assert (ns["eg"].statistic, ns["eg"].critical_values) == (got.statistic, got.critical_values)
