"""Panel unit-root (Fisher-type, Im-Pesaran-Shin, Levin-Lin-Chu, CIPS against plm, Breitung by hand and under the null) and
stationarity (Hadri) tests."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

pytest.importorskip("statsmodels")

from statsmodels.tsa.stattools import adfuller, kpss  # noqa: E402

from pymetrix import Dataset  # noqa: E402
from pymetrix.stats import run_test  # noqa: E402


def panel(kind, N=20, T=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N):
        e = rng.normal(size=T)
        y = np.cumsum(e) if kind == "random walk" else e.copy() if kind == "white noise" else np.zeros(T)
        if kind == "stationary":  # AR(0.5)
            for t in range(T):
                y[t] = (0.5 * y[t - 1] if t else 0.0) + e[t]
        rows += [{"id": f"c{i:02d}", "t": 2000 + t, "y": float(y[t])} for t in range(T)]
    return Dataset.from_frame(pd.DataFrame(rows), entity="id", time="t", freq="Y")


def test_fisher_combines_the_entity_adf_p_values():
    data = panel("stationary")
    result = run_test(data, "fisher_adf", "y", lags=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = np.array([adfuller(g.to_numpy(), maxlag=1, regression="c", autolag=None)[1] for _, g in data["y"].groupby(level="id")])
    assert result.statistic == pytest.approx(-2 * np.sum(np.log(p))) and result.df == 2 * len(p)
    assert result.details["Choi Z"] == pytest.approx(np.sum(stats.norm.ppf(p)) / np.sqrt(len(p)))
    assert result.p_value < 0.001 and result.details["Choi p-value"] < 0.001  # stationary: the unit-root null is rejected
    assert result.provenance.backend["name"] == "statsmodels"  # each entity's ADF went through the adapter
    assert result.provenance.options["stages"]["test"] == {"maxlag": 1, "regression": "c", "autolag": None}
    walks = run_test(panel("random walk"), "fisher_adf", "y")
    assert walks.p_value > 0.05


def test_hadri_is_the_average_kpss_statistic():
    data = panel("white noise")
    result = run_test(data, "hadri", "y")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        each = [kpss(g.to_numpy(), regression="c", nlags=0)[0] for _, g in data["y"].groupby(level="id")]
    assert result.details["LM"] == pytest.approx(np.mean(each))
    assert result.statistic == pytest.approx(np.sqrt(20) * (np.mean(each) - 1 / 6) / np.sqrt(1 / 45))
    assert result.p_value > 0.01  # stationary: not rejected
    persistent = panel("stationary")  # AR(0.5): the original test over-rejects; a long-run variance helps
    assert run_test(persistent, "hadri", "y", lags=8).statistic < run_test(persistent, "hadri", "y").statistic
    assert run_test(persistent, "hadri", "y", lags="auto").details["Bartlett lags"] == 3
    walks = run_test(panel("random walk"), "hadri", "y", trend="ct")
    assert walks.p_value < 0.001  # unit roots: stationarity rejected
    pooled = run_test(data, "hadri", "y", heteroskedastic=False)
    assert pooled.details["variance"] == "pooled"


def test_requirements():
    with pytest.raises(ValueError, match="needs a panel"):
        run_test(Dataset.example("sample_macro"), "hadri", "c")
    frame = panel("stationary").frame.reset_index()
    unbalanced = Dataset.from_frame(frame.iloc[5:], entity="id", time="t", freq="Y")
    with pytest.raises(ValueError, match="balanced panel"):
        run_test(unbalanced, "hadri", "y")
    assert run_test(unbalanced, "fisher_adf", "y").details["entities"] == 20  # Fisher-type tests allow unbalanced panels


def _gretl_panel(seed, N, T, ar=None):
    """The panels the gretl references were computed on (gretl 2026b, levinlin and adf)."""
    rng = np.random.default_rng(seed)
    if ar is None:
        rows = [{"id": i + 1, "t": 2000 + t, "y": v} for i in range(N) for t, v in enumerate(np.cumsum(rng.normal(size=T)))]
    else:
        rows = []
        for i in range(N):
            e, y = rng.normal(size=T), np.zeros(T)
            for t in range(T):
                y[t] = (ar * y[t - 1] if t else 0) + e[t] + 0.3 * i
            y += 0.05 * i * np.arange(T)
            rows += [{"id": i + 1, "t": 2000 + t, "y": y[t]} for t in range(T)]
    return Dataset.from_frame(pd.DataFrame(rows), entity="id", time="t", freq="Y")


@pytest.mark.parametrize("trend, lags, gretl", [("c", 0, -4.91531), ("c", 2, -3.15184), ("ct", 1, -1.81779), ("n", 1, 2.17904)])
def test_levin_lin_chu_matches_gretl(trend, lags, gretl):
    result = run_test(_gretl_panel(5, 8, 40, ar=0.8), "levin_lin_chu", "y", lags=lags, trend=trend)
    assert result.statistic == pytest.approx(gretl, abs=1e-5)
    assert result.p_value == pytest.approx(stats.norm.cdf(gretl), abs=1e-5)


@pytest.mark.parametrize("T, trend, lags, gretl", [(27, "ct", 3, 0.342809), (27, "c", 1, -0.49008), (55, "ct", 3, -0.739169),
                                                   (55, "c", 1, -1.48035), (120, "ct", 3, -0.174095), (120, "c", 1, -0.411928)])
def test_levin_lin_chu_table_rows_match_gretl(T, trend, lags, gretl):
    assert run_test(_gretl_panel(1, 4, T), "levin_lin_chu", "y", lags=lags, trend=trend).statistic == pytest.approx(gretl, abs=1e-5)


def test_im_pesaran_shin_matches_gretl_within_simulation_error():
    data = _gretl_panel(5, 8, 40, ar=0.8)
    assert run_test(data, "im_pesaran_shin", "y", lags=0).details["t-bar"] == pytest.approx(-2.12286, abs=1e-5)  # exact
    for trend, lags, gretl in (("c", 2, -0.704936), ("ct", 1, -0.654778)):
        result = run_test(data, "im_pesaran_shin", "y", lags=lags, trend=trend)
        se = result.details["simulation s.e. of W"]
        assert abs(result.statistic - gretl) < 3 * se + 0.01  # gretl reads IPS's table at T = 40; we simulate
        assert result.p_value == pytest.approx(stats.norm.cdf(result.statistic))


def test_im_pesaran_shin_moments_and_power():
    from pymetrix.stats.panel_tests import _adf_t, _ips_moments

    rng = np.random.default_rng(0)
    Y = np.cumsum(rng.normal(size=(5, 50)), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for trend, p in (("c", 2), ("ct", 1), ("n", 0)):
            ref = [adfuller(y, maxlag=p, regression=trend, autolag=None)[0] for y in Y]
            assert np.allclose(_adf_t(Y, p, trend), ref)  # the simulated statistic is statsmodels' statistic
    mean, var = _ips_moments(1000, 0, "c", 20_000, 0)
    assert mean == pytest.approx(-1.53, abs=0.03) and var == pytest.approx(0.71, abs=0.04)  # the Dickey-Fuller limit
    assert run_test(panel("stationary"), "im_pesaran_shin", "y", lags=1).p_value < 0.001
    assert run_test(panel("random walk"), "im_pesaran_shin", "y").p_value > 0.05
    assert run_test(panel("stationary"), "levin_lin_chu", "y", lags=1).p_value < 0.001
    unbalanced = Dataset.from_frame(panel("stationary").frame.reset_index().iloc[5:], entity="id", time="t", freq="Y")
    assert run_test(unbalanced, "im_pesaran_shin", "y", lags=1).details["entities"] == 20
    with pytest.raises(ValueError, match="balanced"):
        run_test(unbalanced, "levin_lin_chu", "y", lags=1)


def test_fisher_reports_choi_logit():
    result = run_test(panel("stationary"), "fisher_adf", "y", lags=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = np.array([adfuller(g.to_numpy(), maxlag=1, regression="c", autolag=None)[1] for _, g in panel("stationary")["y"].groupby(level="id")])
    n = len(p)
    logit = np.sum(np.log(p / (1 - p))) * np.sqrt(3 * (5 * n + 4) / (np.pi**2 * n * (5 * n + 2)))
    assert result.details["Choi logit"] == pytest.approx(logit) and result.details["Choi logit p-value"] < 0.001


def _factor_panel(rho, N=20, T=60, seed=0):
    """y_it = rho y_i,t-1 + gamma_i f_t + e_it, with one strong common factor f_t (a random walk's shocks)."""
    rng = np.random.default_rng(seed)
    f, gamma = rng.normal(size=T), rng.uniform(1, 3, size=N)
    rows = []
    for i in range(N):
        y = np.zeros(T)
        for t in range(T):
            y[t] = (rho * y[t - 1] if t else 0.0) + gamma[i] * f[t] + rng.normal()
        rows += [{"id": f"c{i:02d}", "t": 1950 + t, "y": float(y[t])} for t in range(T)]
    return Dataset.from_frame(pd.DataFrame(rows), entity="id", time="t", freq="Y")


def test_cadf_is_the_augmented_regression():
    sm = pytest.importorskip("statsmodels.api")
    from pymetrix.stats.panel_tests import _cadf_t

    Y = np.cumsum(np.random.default_rng(4).normal(size=(6, 40)), axis=1)
    p, i = 1, 2
    ybar, y = Y.mean(axis=0), Y[i]
    dy, dbar = np.diff(y), np.diff(ybar)
    X = pd.DataFrame({"y_lag": y[1:-1], "ybar_lag": ybar[1:-1], "dbar": dbar[1:], "dbar_lag": dbar[:-1], "dy_lag": dy[:-1],
                      "const": 1.0, "trend": np.arange(1, 39.0)})
    ref = sm.OLS(dy[1:], X).fit().tvalues["y_lag"]
    assert _cadf_t(Y[None], p, "ct")[0, i] == pytest.approx(ref, rel=1e-10)


@pytest.mark.parametrize("trend, N, T, table", [("c", 10, 30, (-2.57, -2.33, -2.21)), ("c", 20, 20, (-2.40, -2.21, -2.10)),
                                                ("ct", 10, 30, (-3.10, -2.86, -2.73))])
def test_cips_critical_values_reproduce_pesaran_table_ii(trend, N, T, table):
    from pymetrix.stats.panel_tests import _cips_null

    null = _cips_null(N, T, 0, trend, False, 20_000, 0)
    assert np.allclose(np.quantile(null, [0.01, 0.05, 0.10]), table, atol=0.03)  # Pesaran (2007), Tables II(b) and II(c)


def test_cips_with_a_common_factor():
    unit_roots = _factor_panel(1.0)
    result = run_test(unit_roots, "cips", "y", replications=2000)
    assert result.p_value > 0.05 and result.details["entities"] == 20 and set(result.critical_values) == {"1%", "5%", "10%"}
    assert result.critical_values["1%"] < result.critical_values["5%"] < result.critical_values["10%"]
    stationary = run_test(_factor_panel(0.5), "cips", "y", replications=2000)
    assert stationary.p_value < 0.01 and stationary.statistic < stationary.critical_values["1%"]
    starred = run_test(unit_roots, "cips", "y", truncated=True, replications=2000)
    assert starred.name.startswith("Pesaran CIPS*") and starred.statistic >= result.statistic
    with_lags = run_test(unit_roots, "cips", "y", lags=2, trend="ct", replications=2000)
    assert with_lags.details["lags"] == 2
    unbalanced = Dataset.from_frame(unit_roots.frame.reset_index().iloc[3:], entity="id", time="t", freq="Y")
    short = run_test(unbalanced, "cips", "y", replications=500)  # the null simulated with the same periods missing
    assert short.details["periods"] == "57 to 60" and 0 < short.p_value < 1


def _grunfeld(unbalanced):
    f = Dataset.example("grunfeld").frame.reset_index()
    year = f["year"].dt.year
    if unbalanced:  # a late start, an early end and a gap
        f = f[~(((f["firm"] == "Chrysler") & (year < 1940)) | ((f["firm"] == "US Steel") & (year > 1950)) | ((f["firm"] == "IBM") & (year == 1945)))]
    return Dataset.from_frame(f, entity="firm", time="year")


# plm 2.6.3, cipstest(pdata.frame(grunfeld)$invest, type=, lags=) on pymetrix's Grunfeld data (11 firms), and on the
# unbalanced panel of _grunfeld(True); plm's means at each period average the entities whose regression row is complete
@pytest.mark.parametrize("unbalanced, trend, lags, plm", [
    (False, "c", 1, -2.2592433728), (False, "c", 2, -1.5289757248), (False, "ct", 1, -2.2183069714),
    (False, "ct", 2, -1.4465252548), (False, "n", 1, -1.6473537701), (False, "n", 2, -1.1241762903),
    (True, "c", 1, -1.4194063894), (True, "c", 2, -0.8255277488), (True, "ct", 1, -1.8165773042),
    (True, "ct", 2, -1.2427436547), (True, "n", 1, -1.5123737445), (True, "n", 2, -0.9218750299)])
def test_cips_matches_plm(unbalanced, trend, lags, plm):
    assert run_test(_grunfeld(unbalanced), "cips", "invest", lags=lags, trend=trend, replications=200).statistic == pytest.approx(plm, abs=1e-9)


def test_the_unbalanced_cadf_is_the_balanced_one_on_a_balanced_panel():
    from pymetrix.stats.panel_tests import _cadf_t, _cadf_t_unbalanced

    Y = np.cumsum(np.random.default_rng(1).normal(size=(3, 6, 30)), axis=2)
    for trend in ("n", "c", "ct"):
        for p in (0, 2):
            assert np.allclose(_cadf_t(Y, p, trend), _cadf_t_unbalanced(Y, p, trend), rtol=1e-10)


def _breitung_by_hand(Y, trend):
    """Breitung (2000) without prewhitening, written out: each series over its residual standard deviation; with a
    constant, dy_t against y_t-1 - y_0; with a trend, the forward-orthogonalised dy_t against the detrended y_t-1."""
    num = den = 0.0
    terms = []
    for y in Y:
        y = y / (np.std(np.diff(y), ddof=1) if trend == "ct" else np.sqrt(np.mean(np.diff(y) ** 2)))  # less the drift with a trend
        dy = np.diff(y)
        T = len(dy)
        for t in range(1, T + (trend == "c")):
            if trend == "c":
                e, x = dy[t - 1], y[t - 1] - y[0]
            else:
                e = np.sqrt((T - t) / (T - t + 1)) * (dy[t - 1] - dy[t:].mean())
                x = y[t - 1] - y[0] - (t - 1) / T * (y[T] - y[0])
            terms.append((e, x))
            num += e * x
            den += x * x
    a = num / den
    s2 = sum((e - a * x) ** 2 for e, x in terms) / (len(terms) - 1)
    return a / np.sqrt(s2 / den)


@pytest.mark.parametrize("trend", ["c", "ct"])
def test_breitung_by_hand(trend):
    data = panel("random walk", N=6, T=25)
    Y = np.stack([g.to_numpy() for _, g in data["y"].groupby(level="id")])
    result = run_test(data, "breitung", "y", trend=trend)
    assert result.statistic == pytest.approx(_breitung_by_hand(Y, trend), rel=1e-10)
    assert result.details["periods used"] == 24 - (trend == "ct")


def test_breitung_is_standard_normal_under_the_null_and_robust_to_a_common_factor():
    from pymetrix.stats.panel_tests import _breitung_terms

    rng = np.random.default_rng(0)

    def lam(E, X):
        a = np.sum(E * X) / np.sum(X * X)
        return a / np.sqrt(np.sum((E - a * X) ** 2) / (E.size - 1) / np.sum(X * X))

    for trend in ("c", "ct"):  # random walks with their own levels (and drifts): the statistic is N(0, 1)
        draws = np.array([lam(*_breitung_terms(np.cumsum(rng.normal(size=(20, 101)), axis=1) + rng.normal(size=(20, 1)) * 3
                                               + np.arange(101) * rng.normal(size=(20, 1)) * (trend == "ct"), 0, trend))
                          for _ in range(1000)])
        assert abs(draws.mean()) < 0.15 and abs(draws.std() - 1) < 0.15 and 0.03 < np.mean(draws < -1.645) < 0.08
    reject = {False: 0, True: 0}
    for seed in range(150):  # a strong common factor: the standard test over-rejects, Breitung and Das's does not
        data = _factor_panel(1.0, N=10, T=120, seed=seed)  # Breitung and Das need T large against N
        for robust in reject:
            reject[robust] += run_test(data, "breitung", "y", robust=robust).p_value < 0.05
    assert reject[False] / 150 > 0.3 and reject[True] / 150 < 0.13  # N(0, 1) as N and T grow; about 8-11% with N = 10
    assert run_test(panel("stationary"), "breitung", "y", lags=1).p_value < 0.001
    with pytest.raises(ValueError, match="balanced"):
        run_test(Dataset.from_frame(panel("random walk").frame.reset_index().iloc[3:], entity="id", time="t", freq="Y"), "breitung", "y")
