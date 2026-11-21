"""Small-sample critical values for ADF and Engle-Granger, simulated for the sample at hand.

Checked against: statsmodels' adfuller and coint (the array-level statistic reproduces them, lag for lag); a direct,
independent simulation written here (statsmodels' statistic on each random walk, or least squares with the dummies by
numpy and statsmodels' ADF on the residuals), which the critical values and p-value equal at a fixed seed; MacKinnon's
response surfaces, which the simulated values approach at n = 500 within Monte Carlo error; and today's results,
which the default reproduces exactly."""

import time
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")  # the reference statistic
from statsmodels.tsa.adfvalues import mackinnoncrit  # noqa: E402
from statsmodels.tsa.stattools import adfuller, coint  # noqa: E402

from pymetrix import Dataset, random_generator  # noqa: E402
from pymetrix.commands import RunTest, Session  # noqa: E402
from pymetrix.errors import SpecificationError  # noqa: E402
from pymetrix.stats import adf_arrays as A  # noqa: E402
from pymetrix.stats import run_test  # noqa: E402

AUTO = {"aic": "AIC", "bic": "BIC", "t-stat": "t-stat", None: None}


def _adfuller(x, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return adfuller(x, **kw)


def _coint(y, X, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return coint(y, X, **kw)


@pytest.fixture(scope="module")
def short() -> Dataset:
    """25 quarters: two cointegrated series, an unrelated one, a level-shift dummy and an impulse dummy."""
    rng = np.random.default_rng(11)
    T = 25
    common = np.cumsum(rng.normal(size=T))
    shift = (np.arange(T) >= 15).astype(float)
    frame = pd.DataFrame({"y": common + 0.8 * shift + rng.normal(scale=0.4, size=T), "x": common + rng.normal(scale=0.4, size=T),
                          "z": np.cumsum(rng.normal(size=T)), "shift": shift, "pulse": (np.arange(T) == 8).astype(float)},
                         index=pd.period_range("2015Q1", periods=T, freq="Q"))
    return Dataset.from_frame(frame)


def _quantiles(sims):
    return {f"{q:.0%}": float(np.quantile(sims, q)) for q in (0.01, 0.05, 0.10)}


def _same_values(result, sims, statistic):
    """The result's critical values and p-value are those of ``sims`` (computed independently in the test)."""
    for k, v in _quantiles(sims).items():
        assert result.critical_values[k] == pytest.approx(v, rel=1e-9, abs=1e-12)
    p = float(np.mean(np.asarray(sims) <= statistic))
    assert result.p_value == p
    assert result.details["p-value MCSE"] == pytest.approx(np.sqrt(p * (1 - p) / len(sims)), rel=1e-12, abs=1e-15)
    assert result.details["draws"] == len(sims)


# -- the array-level statistic is statsmodels' --------------------------------------------------------------------------


@pytest.mark.parametrize("trend", ["c", "ct", "ctt", "n"])
@pytest.mark.parametrize("autolag", ["aic", "bic", "t-stat", None])
def test_array_adf_reproduces_adfuller(trend, autolag):
    rng = np.random.default_rng(3)
    for n in (16, 20, 30, 90):
        for rep in range(4):
            x = np.cumsum(rng.normal(size=n)) + (0.2 * np.arange(n) if rep % 2 else 0.0)
            top = None if autolag and rep < 2 else min(3, n // 2 - (0 if trend == "n" else len(trend)) - 1)
            ntrend = 0 if trend == "n" else len(trend)  # the default largest lag is statsmodels', capped in short samples
            ref = _adfuller(x, maxlag=A.default_max_lag(n, ntrend) if top is None else top, regression=trend, autolag=AUTO[autolag])
            ours = A.adf(x, trend, lags=top, autolag=autolag, max_lags=top)
            assert ours.lags[0] == ref[2] and ours.nobs[0] == ref[3]  # the same lag chosen, on the same sample
            assert ours.statistic[0] == pytest.approx(ref[0], rel=1e-9)
    many = np.cumsum(rng.normal(size=(40, 30)), axis=1)  # a stack at once: each row as on its own
    stacked = A.adf(many, trend, lags=2 if autolag is None else None, autolag=autolag)
    alone = [A.adf(row, trend, lags=2 if autolag is None else None, autolag=autolag).statistic[0] for row in many]
    np.testing.assert_allclose(stacked.statistic, alone, rtol=1e-12)


@pytest.mark.parametrize("trend", ["c", "ct", "ctt", "n"])
@pytest.mark.parametrize("autolag", ["aic", "bic", "t-stat"])
def test_array_engle_granger_reproduces_coint(trend, autolag):
    rng = np.random.default_rng(4)
    for n in (20, 30, 80):
        for k in (2, 3):
            for rep in range(3):
                Y = np.cumsum(rng.normal(size=(n, k)), axis=0)
                if rep == 1:
                    Y[:, 0] = Y[:, 1] + rng.normal(size=n)  # cointegrated
                top = None if rep < 2 else 2
                used = k - 1 + (0 if trend == "n" else len(trend))  # the levels regression's columns
                ref = _coint(Y[:, 0], Y[:, 1:], trend=trend, maxlag=A.default_max_lag(n, 0, used) if top is None else top,
                             autolag=autolag)[0]
                ours = A.engle_granger(Y[:, 0], Y[:, 1:], trend, autolag, top).statistic[0]
                assert ours != 0 and ours == pytest.approx(ref, rel=1e-9)
    x = np.cumsum(rng.normal(size=40))
    assert A.engle_granger(2 * x + 1, x[:, None], trend if trend != "n" else "c").statistic[0] == -np.inf  # collinear, as coint


def test_the_data_statistic_is_the_one_reported_by_default(short):
    """With simulated critical values the statistic, lags and observations are those MacKinnon's path reports."""
    for kw in ({}, {"trend": "ct", "max_lags": 3}, {"autolag": "bic", "max_lags": 4}):
        base = short.test("engle_granger", "y, x", **kw)
        sim = short.test("engle_granger", "y, x", critical_values="simulated", draws=50, **kw)
        assert sim.statistic == pytest.approx(base.statistic, rel=1e-9)
    for kw in ({"lags": 2}, {"trend": "ct", "lags": "aic", "max_lags": 4}, {"lags": "t-stat", "max_lags": 3}):
        base = short.test("adf", "z", **kw)
        sim = short.test("adf", "z", critical_values="simulated", draws=50, **kw)
        assert sim.statistic == pytest.approx(base.statistic, rel=1e-9)
        assert (sim.details["lags used"], sim.details["observations"]) == (base.details["lags used"], base.details["observations"])


# -- the default is today's -------------------------------------------------------------------------------------------


def test_default_gives_todays_results_exactly(short):
    frame = short.frame
    for trend in ("c", "ct"):
        eg = short.test("engle_granger", "y, x", trend=trend, max_lags=3)
        explicit = short.test("engle_granger", "y, x", trend=trend, max_lags=3, critical_values="mackinnon", draws=99, seed=5)
        stat, p, crit = _coint(frame["y"].to_numpy(), frame[["x"]].to_numpy(), trend=trend, maxlag=3, autolag="aic")
        assert (eg.statistic, eg.p_value) == (float(stat), float(p)) and list(eg.critical_values.values()) == list(map(float, crit))
        assert (explicit.statistic, explicit.p_value, explicit.critical_values, explicit.details) == (eg.statistic, eg.p_value, eg.critical_values, eg.details)
        assert eg.distribution == "MacKinnon" and eg.details == {} and eg.provenance.seed is None
        adf = short.test("adf", "z", trend=trend)
        ref = _adfuller(frame["z"].to_numpy(), regression=trend, autolag="AIC")
        assert (adf.statistic, adf.p_value, adf.critical_values) == (float(ref[0]), float(ref[1]), {k: float(v) for k, v in ref[4].items()})
        assert adf.details == {"lags used": ref[2], "observations": ref[3]}
        assert adf.provenance.backend["wraps"] == "statsmodels.tsa.stattools.adfuller"


# -- the simulation equals a direct one ---------------------------------------------------------------------------------


@pytest.mark.parametrize("trend", ["c", "ct", "ctt", "n"])
def test_engle_granger_equals_a_direct_simulation(short, trend):
    """At a fixed seed: 300 draws of three independent random walks of the sample's length (numpy's default
    generator, as pymetrix seeds it), each tested by statsmodels' coint with the same trend and lag rule."""
    draws, seed = 300, 21
    result = short.test("engle_granger", "y, x, z", trend=trend, max_lags=2, critical_values="simulated", draws=draws, seed=seed)
    walks = np.cumsum(np.random.default_rng(seed).standard_normal((draws, 25, 3)), axis=1)
    sims = [_coint(w[:, 0], w[:, 1:], trend=trend, maxlag=2, autolag="aic")[0] for w in walks]
    _same_values(result, sims, result.statistic)
    assert result.distribution == "simulated for this sample"
    assert "simulated for this sample" in result.details["note"] and "3 independent Gaussian random walks of 25" in result.details["note"]


@pytest.mark.parametrize("trend,lags", [("c", "aic"), ("ct", 1), ("ctt", "bic"), ("n", "t-stat")])
def test_adf_simulates_a_single_random_walk(short, trend, lags):
    """ADF draws one random walk of the series' length per draw (standard_normal((draws, n))), tested by statsmodels'
    adfuller with the same trend and lag rule."""
    draws, seed = 300, 8
    result = short.test("adf", "z", trend=trend, lags=lags, max_lags=3, critical_values="simulated", draws=draws, seed=seed)
    walks = np.cumsum(np.random.default_rng(seed).standard_normal((draws, 25)), axis=1)
    rule = {"maxlag": lags, "autolag": None} if isinstance(lags, int) else {"maxlag": 3, "autolag": AUTO[lags]}
    sims = [_adfuller(w, regression=trend, **rule)[0] for w in walks]
    _same_values(result, sims, result.statistic)
    assert "300 Gaussian random walks of 25 observations" in result.details["note"]


def test_seed_and_draws_in_the_provenance_and_the_generator_setting(short):
    a = short.test("engle_granger", "y, x", critical_values="simulated", draws=200, seed=3, max_lags=2)
    b = short.test("engle_granger", "y, x", critical_values="simulated", draws=200, seed=3, max_lags=2)
    c = short.test("engle_granger", "y, x", critical_values="simulated", draws=200, seed=4, max_lags=2)
    assert a.critical_values == b.critical_values and a.critical_values != c.critical_values
    prov = a.provenance
    assert prov.seed == 3 and prov.random_generator == "pcg64"
    assert prov.method["settings"]["draws"] == 200 and prov.method["settings"]["critical_values"] == "simulated"
    assert prov.method["simulation"]["draws"] == 200 and prov.method["simulation"]["seed"] == 3
    assert prov.backend["name"] == "pymetrix" and prov.resolved == {"lags used": a.details["lags used"]}
    with random_generator("mt19937"):  # the package's generator setting drives the draws, and is recorded
        m = short.test("engle_granger", "y, x", critical_values="simulated", draws=200, seed=3, max_lags=2)
    assert m.provenance.random_generator == "mt19937" and m.critical_values != a.critical_values
    walks = np.cumsum(np.random.Generator(np.random.MT19937(3)).standard_normal((200, 25, 2)), axis=1)
    _same_values(m, [_coint(w[:, 0], w[:, 1:], maxlag=2, autolag="aic")[0] for w in walks], m.statistic)


# -- exog -----------------------------------------------------------------------------------------------------------


def _eg_with_dummies(y, X, D, trend, top):
    """Engle-Granger by hand: least squares (numpy) of y on X, the trend terms and the dummies; statsmodels' ADF
    without deterministic terms on the residuals."""
    t = np.arange(1, len(y) + 1, dtype=float)
    terms = {"c": [t ** 0], "ct": [t ** 0, t], "n": []}[trend]
    Z = np.column_stack([X, *terms, D])
    resid = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return _adfuller(resid, maxlag=top, regression="n", autolag="AIC")[0]


@pytest.mark.parametrize("trend", ["c", "ct", "n"])
def test_exog_dummies_enter_every_levels_regression(short, trend):
    draws, seed = 250, 13
    result = short.test("engle_granger", "y, x", trend=trend, max_lags=2, exog="shift, pulse", critical_values="simulated",
                        draws=draws, seed=seed)
    frame = short.frame
    D = frame[["shift", "pulse"]].to_numpy()
    assert result.statistic == pytest.approx(_eg_with_dummies(frame["y"].to_numpy(), frame[["x"]].to_numpy(), D, trend, 2), rel=1e-9)
    walks = np.cumsum(np.random.default_rng(seed).standard_normal((draws, 25, 2)), axis=1)  # the dummies are not drawn
    _same_values(result, [_eg_with_dummies(w[:, 0], w[:, 1:], D, trend, 2) for w in walks], result.statistic)
    assert result.details["fixed regressors"] == "shift, pulse" and "shift, pulse fixed in the levels regression" in result.details["note"]
    plain = short.test("engle_granger", "y, x", trend=trend, max_lags=2, critical_values="simulated", draws=draws, seed=seed)
    assert result.critical_values["5%"] < plain.critical_values["5%"]  # more regressors push the null distribution out


def test_exog_and_simulation_settings_are_checked(short):
    with pytest.raises(SpecificationError, match="exog needs critical_values='simulated'"):
        short.test("engle_granger", "y, x", exog="shift")
    with pytest.raises(SpecificationError, match="collinear"):  # a constant dummy repeats the constant
        Dataset.from_frame(short.frame.assign(one=1.0)).test("engle_granger", "y, x", exog="one", critical_values="simulated")
    with pytest.raises(SpecificationError, match="not in the data"):
        short.test("engle_granger", "y, x", exog="nothing", critical_values="simulated")
    with pytest.raises(SpecificationError, match="'mackinnon' or 'simulated'"):
        short.test("adf", "z", critical_values="exact")
    with pytest.raises(SpecificationError, match="at least 10"):
        short.test("adf", "z", critical_values="simulated", draws=5)
    with pytest.raises(SpecificationError, match="backend_options"):
        short.test("adf", "z", critical_values="simulated", backend_options={"store": False})


def test_too_few_degrees_of_freedom_is_refused(short):
    """With 20 observations statsmodels' default largest lag (9) leaves the residuals' ADF regression no degrees of
    freedom and a zero statistic; pymetrix's default is capped (6, keeping 4 after the levels regression's 2), and a
    largest lag that leaves none is refused rather than reported as 0.0."""
    head = Dataset.from_frame(short.frame.iloc[:20])
    y, x = head.frame["y"].to_numpy(), head.frame[["x"]].to_numpy()
    assert _coint(y, x)[0] == 0  # statsmodels' default: the degenerate zero
    assert A.default_max_lag(20, 0, 2) == 6
    base = head.test("engle_granger", "y, x")
    sim = head.test("engle_granger", "y, x", critical_values="simulated", draws=100)
    assert base.statistic == pytest.approx(_coint(y, x, maxlag=6)[0], rel=1e-12) and base.statistic != 0
    assert sim.statistic == pytest.approx(base.statistic, rel=1e-9) and "warning" not in sim.details
    assert len(set(sim.critical_values.values())) == 3 and 0 not in sim.critical_values.values()
    for cv in ("mackinnon", "simulated"):
        with pytest.raises(SpecificationError, match="no residual degrees of freedom"):
            head.test("engle_granger", "y, x", max_lags=9, critical_values=cv, draws=100)


# -- towards MacKinnon -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("which,trend,k", [("adf", "c", 1), ("adf", "ct", 1), ("eg", "c", 2), ("eg", "ct", 3)])
def test_large_samples_approach_mackinnon(which, trend, k):
    """At n = 500 with a fixed lag, the share of 2,000 simulated statistics below MacKinnon's 1, 5 and 10% critical
    values (for this n) is each level within 4 binomial standard errors; at n = 15 with the AIC lag search, the
    simulated 5% value is far from MacKinnon's, which is why the option exists."""
    draws, n = 2000, 500
    sims = (A.simulate_adf(n, trend, lags=0, autolag=None, draws=draws, seed=2) if which == "adf"
            else A.simulate_engle_granger(n, k, trend, autolag=None, max_lags=0, draws=draws, seed=2))
    for level, value in zip((0.01, 0.05, 0.10), mackinnoncrit(k, trend, n - 1)):
        assert abs(np.mean(sims <= value) - level) < 4 * np.sqrt(level * (1 - level) / draws)
    small = (A.simulate_adf(15, trend, draws=draws, seed=2) if which == "adf"
             else A.simulate_engle_granger(15, k, trend, max_lags=3, draws=draws, seed=2))
    assert abs(np.quantile(small, 0.05) - mackinnoncrit(k, trend, 14)[1]) > 0.2  # 0.3 to 1.5 in these four cases


@pytest.mark.timing
def test_two_thousand_draws_at_twenty_observations_well_under_a_second():
    """2000 draws at n = 20 should take well under a second. Asserted under 5 s, the best of three runs (slack for a
    loaded machine); the times measured are printed (about 0.02-0.04 s each on a 4-core Linux machine)."""
    A.simulate_engle_granger(20, 2, max_lags=3, draws=20)  # warm up
    times = {}
    for name, run in (("adf", lambda: A.simulate_adf(20, "c", draws=2000, seed=0)),
                      ("engle_granger", lambda: A.simulate_engle_granger(20, 2, "c", max_lags=3, draws=2000, seed=0)),
                      ("engle_granger_default_lags", lambda: A.simulate_engle_granger(20, 2, "c", draws=2000, seed=0))):
        best = np.inf
        for _ in range(3):
            start = time.perf_counter()
            sims = run()
            best = min(best, time.perf_counter() - start)
        times[name] = best
        assert sims.shape == (2000,) and np.isfinite(sims).all()
    print("2000 draws at n = 20:", {k: f"{v:.3f} s" for k, v in times.items()})
    assert max(times.values()) < 5.0


# -- the dialog --------------------------------------------------------------------------------------------------------


def test_dialog_equals_the_api_call_and_the_script_replays(short):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from pymetrix.ui.qt import command_dialog

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s = Session()
    s.objects["data"] = short
    dialog = command_dialog(RunTest, s)
    dialog.editors["source"].widget.setCurrentText("data")
    dialog.editors["test"].widget.setCurrentText("engle_granger")
    sub = dialog.editors["options"]
    assert {"critical_values", "draws", "seed", "exog"} <= set(sub.sub)
    assert not dialog.in_advanced("options") and sub.get() == ""  # the defaults: nothing written
    sub.sub["variables"].set("y, x")
    sub.sub["critical_values"].set("simulated")
    sub.sub["draws"].set(400)
    sub.sub["seed"].set(9)
    sub.sub["max_lags"].widget.setText("2")
    sub.sub["exog"].set("shift")
    options = sub.get()
    assert "critical_values='simulated'" in options and "draws=400" in options and "exog='shift'" in options
    s.run(RunTest(**{k: dialog.editors[k].get() for k in ("source", "test", "options")}, name="eg"))
    by_api = run_test(short, "engle_granger", variables="y, x", max_lags=2, critical_values="simulated", draws=400, seed=9, exog="shift")
    got = s["eg"]
    assert (got.statistic, got.p_value, got.critical_values) == (by_api.statistic, by_api.p_value, by_api.critical_values)
    ns = dict(s.objects)
    exec(s.script(), ns)
    assert (ns["eg"].statistic, ns["eg"].p_value, ns["eg"].critical_values) == (got.statistic, got.p_value, got.critical_values)
    s.run(RunTest(source="data", test="adf", options="z, trend='ct', critical_values='simulated', draws=300, seed=2", name="adf"))
    assert "data.test('adf', 'z', trend='ct', critical_values='simulated', draws=300, seed=2)" in s.script()
    ref = short.test("adf", "z", trend="ct", critical_values="simulated", draws=300, seed=2)
    assert s["adf"].critical_values == ref.critical_values and s["adf"].p_value == ref.p_value
