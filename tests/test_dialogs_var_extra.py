"""Dialogs for more VAR identification and VAR types, VECM tests, spectra and long memory, VARMA, threshold VARs and
bilinear models.

Each dialog's run equals the API call it stands for, its recorded script replays to the same result, and the menus
and Next steps reach it."""

import os

import numpy as np
import pandas as pd
import pytest

from pymetrix import (BVARSV, MSVAR, VAR, VECM, Dataset, MaxShare, QuantileVAR, arfima, bilinear, cross_spectrum, gph,
                      local_whittle, periodogram, spectrum, threshold_var, varma)
from pymetrix.commands import HistoricalDecomposition, ImpulseResponse, Session, VarianceDecomposition
from pymetrix.commands.var_extra import (FitBilinear, FitBVARSV, FitMSVAR, FitQuantileVAR, FitThresholdVAR, FitVARMA, LongMemory,
                                         SpectralAnalysis, ThresholdLinearityTest, VECMRestrictionTest)

pytest.importorskip("statsmodels")


def _macro():
    return Dataset.example("sample_macro").derive(dy="400*dlog(y)", dc="400*dlog(c)", di="400*dlog(i)")


def _levels(T=300, seed=0):
    """b and e are random walks; a - b and c - 0.5 b - 0.8 e are stationary."""
    rng = np.random.default_rng(seed)
    b, e = np.cumsum(rng.normal(size=T)), np.cumsum(rng.normal(size=T))
    a, c = np.zeros(T), np.zeros(T)
    for t in range(1, T):
        a[t] = a[t - 1] - 0.3 * (a[t - 1] - b[t - 1]) + (b[t] - b[t - 1]) + 0.5 * rng.normal()
        c[t] = c[t - 1] - 0.4 * (c[t - 1] - 0.5 * b[t - 1] - 0.8 * e[t - 1]) + 0.5 * rng.normal()
    return Dataset.from_frame(pd.DataFrame({"a": a, "b": b, "c": c, "e": e}, index=pd.period_range("1950Q1", periods=T, freq="Q", name="time")))


@pytest.fixture(scope="module")
def session():
    s = Session()
    s.objects["data"] = _macro()
    s.objects["levels"] = _levels()
    s.objects["var"] = VAR(["dy", "dc", "di"], lags=2).fit(s.objects["data"])
    s.objects["vecm"] = VECM(["a", "b", "c", "e"], lags=1, rank=2, deterministic="co").fit(s.objects["levels"])
    return s


def _replay(s, name):
    ns = {k: v for k, v in s.objects.items() if k in ("data", "levels", "var", "vecm")}
    exec(s.script(), ns)
    return ns[name]


def _same_irf(a, b):
    np.testing.assert_allclose(a.values, b.values, rtol=1e-10, atol=1e-12)
    for x, y in ((a.lower, b.lower), (a.upper, b.upper)):
        assert (x is None) == (y is None)
        if x is not None:
            np.testing.assert_allclose(x, y, rtol=1e-10, atol=1e-12)


# identification in the impulse-response and decomposition dialogs


def test_max_share_and_news_shocks(session):
    var = session["var"]
    session.run(ImpulseResponse("var", scheme="MaxShare", share_variable="dy", share_horizon=24, draws=30, seed=2, name="ms_irf"))
    _same_irf(session["ms_irf"], var.identify(MaxShare("dy", horizon=24)).irf(draws=30, seed=2))
    session.run(ImpulseResponse("var", scheme="News", share_variable="dy", draws=0, name="news_irf"))
    news = var.identify(MaxShare("dy", zero_impact=True)).irf(draws=0)
    _same_irf(session["news_irf"], news)
    assert abs(session["news_irf"].values[0, 0, 0]) < 1e-10  # no effect on dy on impact
    session.run(VarianceDecomposition("var", scheme="News", share_variable="dy", share_start=1, name="news_fevd"))
    np.testing.assert_allclose(session["news_fevd"].values, var.identify(MaxShare("dy", start=1, zero_impact=True)).fevd().values)
    session.run(HistoricalDecomposition("var", scheme="MaxShare", share_variable="di", name="ms_hd"))
    ref = var.identify(MaxShare("di")).historical_decomposition()
    np.testing.assert_allclose(session["ms_hd"].contributions["di"].to_numpy(), ref.contributions["di"].to_numpy())
    script = session.script()
    assert "var.identify(MaxShare('dy', horizon=24)).irf(draws=30, seed=2)" in script
    assert "MaxShare('dy', start=1, zero_impact=True)" in script
    _same_irf(_replay(session, "ms_irf"), session["ms_irf"])
    _same_irf(_replay(session, "news_irf"), news)
    np.testing.assert_allclose(_replay(session, "news_fevd").values, session["news_fevd"].values)


def test_identification_through_heteroskedasticity(session):
    var = session["var"]
    session.run(ImpulseResponse("var", scheme="Heteroskedasticity", breaks="2012Q1", periods=12, cumulative=True, name="h_irf"))
    _same_irf(session["h_irf"], var.heteroskedasticity_svar(["2012Q1"]).irf(12, cumulative=True))
    session.run(VarianceDecomposition("var", scheme="Heteroskedasticity", breaks="2008Q1, 2016Q1", periods=12, name="h_fevd"))
    hsvar = var.heteroskedasticity_svar(["2008Q1", "2016Q1"])
    fevd = hsvar.fevd(12)
    np.testing.assert_allclose(session["h_fevd"].values, fevd.values)
    np.testing.assert_allclose(fevd.values.sum(axis=2), 1.0)  # shares of each variable's variance add to one
    theta = np.cumsum(hsvar.irf(11).values ** 2, axis=0)  # the same shares from the responses
    np.testing.assert_allclose(fevd.values, theta / theta.sum(axis=2, keepdims=True))
    assert "var.heteroskedasticity_svar(['2012Q1']).irf(periods=12, cumulative=True)" in session.script()
    _same_irf(_replay(session, "h_irf"), session["h_irf"])
    np.testing.assert_allclose(_replay(session, "h_fevd").values, fevd.values)
    assert ImpulseResponse("var", scheme="Heteroskedasticity").validate(session)
    assert VarianceDecomposition("vecm", scheme="Heteroskedasticity", breaks="1990Q1").validate(session)  # VARs only


def test_generalised_impulse_responses(session):
    session.run(ImpulseResponse("var", scheme="Generalised", periods=10, draws=25, seed=4, orthogonal=False, name="g_irf"))
    _same_irf(session["g_irf"], session["var"].girf(10, draws=25, seed=4))
    assert "var.girf(periods=10, draws=25, seed=4)" in session.script()
    _same_irf(_replay(session, "g_irf"), session["g_irf"])


def test_scheme_problems(session):
    assert ImpulseResponse("var", scheme="MaxShare").validate(session)
    assert "not a variable" in ImpulseResponse("var", scheme="MaxShare", share_variable="zz").validate(session)[0]
    assert ImpulseResponse("vecm", scheme="Generalised").validate(session)
    assert ImpulseResponse("var", scheme="MaxShare", share_variable="dy", share_start=50, share_horizon=40).validate(session)


# more VAR types


def test_bvar_with_stochastic_volatility_and_its_responses(session):
    session.run(FitBVARSV("dy, dc", lags=1, draws=60, burn=40, tightness=0.5, seed=3, end="2023Q4", name="sv"))
    ref = BVARSV(["dy", "dc"], lags=1).fit(session["data"], draws=60, burn=40, tightness=0.5, seed=3, span=(None, "2023Q4"))
    np.testing.assert_allclose(session["sv"].B, ref.B)
    np.testing.assert_allclose(session["sv"].h, ref.h)
    session.run(ImpulseResponse("sv", periods=8, period="2010Q1", coverage=0.68, name="sv_irf"))
    _same_irf(session["sv_irf"], ref.irf(8, period="2010Q1", coverage=0.68))
    np.testing.assert_allclose(_replay(session, "sv").h, ref.h)
    _same_irf(_replay(session, "sv_irf"), session["sv_irf"])
    assert ImpulseResponse("sv", scheme="LongRun").validate(session)
    assert VarianceDecomposition("sv", scheme="LongRun").validate(session)  # recursive only
    assert not VarianceDecomposition("sv").validate(session)


def test_markov_switching_var_and_its_responses(session):
    session.run(FitMSVAR("dy, di", lags=1, regimes=2, switching="intercept", starts=3, name="msv"))
    ref = MSVAR(["dy", "di"], lags=1, regimes=2, switching="intercept").fit(session["data"], starts=3)
    assert session["msv"].loglike == pytest.approx(ref.loglike, rel=1e-12)
    np.testing.assert_allclose(session["msv"].smoothed.to_numpy(), ref.smoothed.to_numpy())
    session.run(ImpulseResponse("msv", periods=6, regime=1, name="msv_irf"))
    _same_irf(session["msv_irf"], ref.irf(regime=1, periods=6, draws=500, coverage=0.95))  # the dialog's Draws and Coverage
    assert session["msv_irf"].lower is not None
    assert "msv = MSVAR(" in session.script() and "switching='intercept'" in session.script()
    assert _replay(session, "msv").loglike == pytest.approx(ref.loglike, rel=1e-12)
    _same_irf(_replay(session, "msv_irf"), session["msv_irf"])


def test_quantile_var_and_its_responses(session):
    session.run(FitQuantileVAR("dy, dc", lags=1, quantile=0.2, name="qv"))
    ref = QuantileVAR(["dy", "dc"], lags=1, quantile=0.2).fit(session["data"])
    np.testing.assert_allclose(session["qv"].A, ref.A)
    session.run(ImpulseResponse("qv", periods=8, name="qv_irf"))
    _same_irf(session["qv_irf"], ref.irf(8))
    np.testing.assert_allclose(_replay(session, "qv").A, ref.A)
    _same_irf(_replay(session, "qv_irf"), session["qv_irf"])


def test_varma_threshold_var_and_bilinear(session):
    session.run(FitVARMA("dy, dc", p=1, q=1, name="vm"))
    ref = varma(session["data"], ["dy", "dc"], p=1, q=1)
    np.testing.assert_allclose(session["vm"].values, ref.values)
    session.run(FitThresholdVAR("dy, dc", lags=1, threshold="di", delay=2, name="tv"))
    tv = threshold_var(session["data"], ["dy", "dc"], lags=1, threshold="di", delay=2)
    assert session["tv"].gamma == tv.gamma and session["tv"].LR == pytest.approx(tv.LR, rel=1e-12)
    session.run(ThresholdLinearityTest("tv", draws=19, seed=5, name="tsay"))
    assert session["tsay"].p_value == tv.test(draws=19, seed=5).p_value
    session.run(FitBilinear("dy", end="2020Q4", name="bl"))
    bl = bilinear(session["data"], "dy", span=(None, "2020Q4"))
    np.testing.assert_allclose(session["bl"].values, bl.values)
    assert "tsay = tv.test(draws=19, seed=5)" in session.script()
    np.testing.assert_allclose(_replay(session, "vm").values, ref.values)
    assert _replay(session, "tsay").p_value == session["tsay"].p_value
    np.testing.assert_allclose(_replay(session, "bl").values, bl.values)


# VECM tests


def test_vecm_restriction_tests(session):
    vecm = session["vecm"]
    session.run(VECMRestrictionTest("vecm", h_matrix="1 0; -1 -0.5\n0 1; 0 -0.8", name="beta_test"))
    ref = vecm.test_beta([[1, 0], [-1, -0.5], [0, 1], [0, -0.8]])
    assert session["beta_test"].statistic == pytest.approx(ref.statistic, rel=1e-12) and session["beta_test"].df == ref.df
    session.run(VECMRestrictionTest("vecm", test="weak exogeneity", variables="b, e", name="wx"))
    assert session["wx"].statistic == pytest.approx(vecm.test_weak_exogeneity(["b", "e"]).statistic, rel=1e-12)
    session.run(VECMRestrictionTest("vecm", test="eigenvalue fluctuation", draws=19, seed=2, name="fl"))
    assert session["fl"].p_value == vecm.fluctuation_test(draws=19, seed=2).p_value
    script = session.script()
    assert "wx = vecm.test_weak_exogeneity(['b', 'e'])" in script or "test_weak_exogeneity(variables=['b', 'e'])" in script
    assert _replay(session, "beta_test").statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert _replay(session, "fl").p_value == session["fl"].p_value
    assert VECMRestrictionTest("vecm").validate(session)  # no H
    assert VECMRestrictionTest("vecm", h_matrix="1 *; 0 1").validate(session)
    assert VECMRestrictionTest("vecm", test="weak exogeneity", variables="zz").validate(session)


# spectra and long memory


def test_spectra(session):
    data = session["data"]
    session.run(SpectralAnalysis("dy", kind="periodogram", name="pg"))
    np.testing.assert_allclose(session["pg"].density, periodogram(data, "dy").density)
    session.run(SpectralAnalysis("dy", bandwidth=3, start="2002Q1", name="sp"))
    np.testing.assert_allclose(session["sp"].density, spectrum(data, "dy", bandwidth=3, span=("2002Q1", None)).density)
    session.run(SpectralAnalysis("dy", kind="cross-spectrum", other="dc", bandwidth=4, name="cs"))
    cs = cross_spectrum(data, "dy", "dc", bandwidth=4)
    np.testing.assert_allclose(session["cs"].coherence, cs.coherence)
    np.testing.assert_allclose(session["cs"].gain, cs.gain)
    np.testing.assert_allclose(_replay(session, "cs").phase, cs.phase)
    np.testing.assert_allclose(_replay(session, "sp").density, session["sp"].density)
    assert SpectralAnalysis("dy", kind="cross-spectrum").validate(session)


def test_long_memory(session):
    data = session["data"]
    session.run(LongMemory("dy", method="GPH", gph_power=0.6, name="d_gph"))
    assert session["d_gph"].statistic == pytest.approx(gph(data, "dy", power=0.6).statistic)
    session.run(LongMemory("dy", name="d_lw"))
    assert session["d_lw"].statistic == pytest.approx(local_whittle(data, "dy").statistic)
    session.run(LongMemory("dy", method="ARFIMA", p=1, name="af"))
    np.testing.assert_allclose(session["af"].params, arfima(data, "dy", p=1).params)
    np.testing.assert_allclose(_replay(session, "af").params, session["af"].params)
    assert _replay(session, "d_gph").statistic == pytest.approx(session["d_gph"].statistic)


# menus, next steps and forms


def test_menus_and_next_steps(session):
    from pymetrix.commands import COMMANDS
    from pymetrix.commands.menus import menu_order
    from pymetrix.ui.next_steps import steps_for

    places = {info.cls.__name__: where for info, where in menu_order([i for i in COMMANDS.values() if i.menu])}
    assert places["FitBVARSV"] == "Estimate → VARs and local projections → BVAR with stochastic volatility…"
    assert places["SpectralAnalysis"] == "Test → Spectrum…"
    assert places["LongMemory"] == "Estimate → Time series → Long memory…"
    assert places["FitThresholdVAR"] == "Estimate → Nonlinear → Threshold VAR…"
    assert places["VECMRestrictionTest"] == "Test → VECM restrictions and stability…"
    decompositions = {"sv": ["VarianceDecomposition", "HistoricalDecomposition"], "msv": ["VarianceDecomposition", "HistoricalDecomposition"],
                      "qv": ["VarianceDecomposition"]}
    for name in ("sv", "msv", "qv"):
        if name in session.objects:
            assert [s[1].__name__ for s in steps_for(session[name])] == ["ImpulseResponse", *decompositions[name]]
    assert "VECMRestrictionTest" in [s[1].__name__ for s in steps_for(session["vecm"])]


def test_the_new_fields_in_the_qt_forms(session):
    pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.forms import form_spec
    from pymetrix.ui.qt import command_dialog

    form = command_dialog(ImpulseResponse, session)
    assert not form.row_visible("share_variable") and not form.row_visible("breaks")
    form.editors["scheme"].set("News")
    assert form.row_visible("share_variable") and form.row_visible("share_horizon") and form.on_advanced_tab("share_start")
    form.editors["scheme"].set("Heteroskedasticity")
    assert form.row_visible("breaks") and not form.row_visible("share_variable")
    assert form.on_advanced_tab("period") and form.on_advanced_tab("regime")
    assert "Generalised" in {f.name: f for f in form_spec(ImpulseResponse)}["scheme"].choices
    assert "Generalised" not in {f.name: f for f in form_spec(VarianceDecomposition)}["scheme"].choices
    assert "Heteroskedasticity" not in {f.name: f for f in form_spec(HistoricalDecomposition)}["scheme"].choices
    vecm = command_dialog(VECMRestrictionTest, session)
    assert vecm.row_visible("h_matrix") and not vecm.row_visible("variables")
    vecm.editors["test"].set("weak exogeneity")
    assert vecm.row_visible("variables") and not vecm.row_visible("h_matrix")
    memory = command_dialog(LongMemory, session)
    assert not memory.row_visible("p")
    memory.editors["method"].set("ARFIMA")
    assert memory.row_visible("p") and memory.row_visible("q")
    for cls in (FitBVARSV, FitMSVAR, FitQuantileVAR, FitVARMA, FitThresholdVAR, ThresholdLinearityTest, FitBilinear, SpectralAnalysis):
        assert command_dialog(cls, session) is not None
