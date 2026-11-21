"""The Variance decomposition and Historical decomposition dialogs on BVARs with stochastic volatility, Markov-switching
VARs and quantile VARs: each run equals the API call, its recorded script replays to the same result, the views show
the bands, and Next steps reach the dialogs."""

import numpy as np
import pytest

from pymetrix import BVARSV, MSVAR, Dataset, QuantileVAR
from pymetrix.commands import HistoricalDecomposition, Session, VarianceDecomposition
from pymetrix.commands.var_extra import FitBVARSV, FitMSVAR, FitQuantileVAR

pytest.importorskip("statsmodels")


@pytest.fixture(scope="module")
def session():
    s = Session()
    s.objects["data"] = Dataset.example("sample_macro").derive(dy="400*dlog(y)", dc="400*dlog(c)", di="400*dlog(i)")
    s.run(FitBVARSV("dy, dc", lags=1, draws=30, burn=20, seed=3, name="sv"))
    s.run(FitMSVAR("dy, di", lags=1, regimes=2, switching="intercept", starts=2, name="msv"))
    s.run(FitQuantileVAR("dy, dc", lags=1, quantile=0.5, name="qv"))
    return s


def _replay(s, name):
    ns = {"data": s.objects["data"]}
    exec(s.script(), ns)
    return ns[name]


def _same_fevd(a, b):
    np.testing.assert_allclose(a.values, b.values, rtol=1e-10, atol=1e-12)
    for x, y in ((a.lower, b.lower), (a.upper, b.upper)):
        assert (x is None) == (y is None)
        if x is not None:
            np.testing.assert_allclose(x, y, rtol=1e-10, atol=1e-12)


def _same_hd(a, b):
    for v in a.contributions:
        np.testing.assert_allclose(a.contributions[v].to_numpy(), b.contributions[v].to_numpy(), rtol=1e-10, atol=1e-10)
        if a.has_bands:
            np.testing.assert_allclose(a.upper[v].to_numpy(), b.upper[v].to_numpy(), rtol=1e-10, atol=1e-10)
    assert a.has_bands == b.has_bands


def test_bvar_sv_decompositions(session):
    ref = BVARSV(["dy", "dc"], lags=1).fit(session["data"], draws=30, burn=20, seed=3)
    session.run(VarianceDecomposition("sv", periods=8, period="2010Q1", coverage=0.9, name="sv_fevd"))
    _same_fevd(session["sv_fevd"], ref.fevd(8, period="2010Q1", coverage=0.9))
    assert session["sv_fevd"].has_bands and "sv.fevd(" in session.script()
    session.run(HistoricalDecomposition("sv", draws=10, coverage=0.8, name="sv_hd"))
    _same_hd(session["sv_hd"], ref.historical_decomposition(coverage=0.8, draws=10))
    _same_fevd(_replay(session, "sv_fevd"), session["sv_fevd"])
    _same_hd(_replay(session, "sv_hd"), session["sv_hd"])
    from pymetrix.views import view_sections as views

    assert "Bands: dy" in [name for name, _ in views(session["sv_fevd"])]
    assert "Bands graph" in [name for name, _ in views(session["sv_hd"])]


def test_ms_var_decompositions(session):
    ref = MSVAR(["dy", "di"], lags=1, regimes=2, switching="intercept").fit(session["data"], starts=2)
    session.run(VarianceDecomposition("msv", periods=6, regime=1, name="msv_fevd"))
    _same_fevd(session["msv_fevd"], ref.fevd(regime=1, periods=6))
    session.run(HistoricalDecomposition("msv", name="msv_hd"))
    _same_hd(session["msv_hd"], ref.historical_decomposition())
    assert float(session["msv_hd"].residual.abs().to_numpy().max()) < 1e-10
    _same_fevd(_replay(session, "msv_fevd"), session["msv_fevd"])
    _same_hd(_replay(session, "msv_hd"), session["msv_hd"])
    # bands from draws of the parameters (Draws, Coverage, Seed)
    session.run(VarianceDecomposition("msv", periods=6, regime=0, draws=60, coverage=0.9, seed=4, name="msv_fevd_b"))
    _same_fevd(session["msv_fevd_b"], ref.fevd(regime=0, periods=6, draws=60, coverage=0.9, seed=4))
    assert session["msv_fevd_b"].has_bands
    session.run(HistoricalDecomposition("msv", draws=40, coverage=0.8, name="msv_hd_b"))
    _same_hd(session["msv_hd_b"], ref.historical_decomposition(draws=40, coverage=0.8))
    assert session["msv_hd_b"].has_bands and not HistoricalDecomposition("msv", draws=100).validate(session)
    _same_fevd(_replay(session, "msv_fevd_b"), session["msv_fevd_b"])
    _same_hd(_replay(session, "msv_hd_b"), session["msv_hd_b"])
    assert VarianceDecomposition("msv", scheme="LongRun").validate(session)


def test_quantile_var_decomposition(session):
    session.run(VarianceDecomposition("qv", periods=8, name="qv_fevd"))
    _same_fevd(session["qv_fevd"], QuantileVAR(["dy", "dc"], lags=1, quantile=0.5).fit(session["data"]).fevd(8))
    _same_fevd(_replay(session, "qv_fevd"), session["qv_fevd"])
    assert "no historical decomposition" in HistoricalDecomposition("qv").validate(session)[0]


def test_next_steps_and_forms(session):
    from pymetrix.forms import form_spec
    from pymetrix.ui.next_steps import steps_for

    assert [s[1].__name__ for s in steps_for(session["sv"])] == ["ImpulseResponse", "VarianceDecomposition", "HistoricalDecomposition"]
    assert [s[1].__name__ for s in steps_for(session["qv"])] == ["ImpulseResponse", "VarianceDecomposition"]
    fields = {f.name: f for f in form_spec(VarianceDecomposition)}
    assert {"period", "coverage", "regime", "draws", "seed"} <= set(fields)
