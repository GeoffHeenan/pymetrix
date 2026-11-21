"""Dialog fields for soft holds and least-norm plans (Model > Solve model, Model > Forecast from estimated model), HAC without
the small-sample factor (Estimate equation) and identification in elasticities (Model > Identification).

Each dialog's run equals the API call it stands for, and its recorded script runs to the same result."""

import os

import numpy as np
import pandas as pd
import pytest

from pymetrix import OLS, Dataset, Equation, Scenario, templates
from pymetrix.commands import FitEquation, Session
from pymetrix.commands.builtin import CheckIdentification, EstimateModel, ForecastModel
from pymetrix.estimation.registry import method_settings


@pytest.fixture(scope="module")
def session():
    nk = templates.new_keynesian()
    truth = nk.solve().simulate(120, nk.shock_sd, seed=11)
    truth.index = pd.period_range("1990Q1", periods=120, freq="Q", name="time")
    rng = np.random.default_rng(5)
    s = Session()
    s.objects["nk"] = nk
    s.objects["data"] = nk.measurement.apply(Dataset.from_frame(truth))
    s.objects["eqdata"] = Dataset.from_frame(pd.DataFrame({"x": rng.normal(size=120), "y": rng.normal(size=120)},
                                                          index=pd.period_range("1990Q1", periods=120, freq="Q", name="time")))
    s.run(EstimateModel("nk", estimate="kappa, rho"))
    return s


def _replay(s, name):
    ns = dict(s.objects)
    exec(s.script(), ns)
    return ns[name]


def test_hac_correction_on_the_estimate_form(session):
    session.run(FitEquation("y ~ x", data="eqdata", cov="hac", hac_lags=4, hac_correction=False, name="plain"))
    session.run(FitEquation("y ~ x", data="eqdata", cov="hac", hac_lags=4, name="corrected"))
    by_api = Equation("y ~ x").fit(session["eqdata"], method=OLS(cov="hac", hac_lags=4, hac_correction=False))
    np.testing.assert_allclose(session["plain"].params.cov, by_api.params.cov, rtol=1e-12)
    assert not np.allclose(session["plain"].params.cov, session["corrected"].params.cov)
    script = session.script()
    assert "hac_correction=False" in script and script.count("hac_correction") == 1  # left out at its default
    np.testing.assert_allclose(_replay(session, "plain").params.cov, by_api.params.cov, rtol=1e-12)
    assert "hac_correction" not in [f.name for f in method_settings("OLS")]  # on the main form, not in Method settings


def test_soft_holds_and_least_norm_in_the_forecast_dialog(session):
    fit = session["estimated"]
    span = ("2020Q1", "2021Q4")
    session.run(ForecastModel("estimated", "2020Q1", "2021Q4", hold="i[2020Q1:2020Q2]=0.5, y[2020Q4]=-1", free="ei, ey, epi",
                              hold_sd="y[2020Q4]=0.25", free_sd="ei=0.2, ey=0.5, epi=0.3", name="plan"))
    scenario = Scenario(hold={"i": {"2020Q1": 0.5, "2020Q2": 0.5}, "y": {"2020Q4": -1.0}}, free=["ei", "ey", "epi"],
                        hold_sd={"y": {"2020Q4": 0.25}}, free_sd={"ei": 0.2, "ey": 0.5, "epi": 0.3})
    by_api = fit.forecast(span, scenario)
    columns = ["y", "pi", "i", "ei", "ey", "epi"]
    np.testing.assert_allclose(session["plan"].frame.loc["2020Q1":"2021Q4", columns], by_api.frame.loc["2020Q1":"2021Q4", columns], atol=1e-10)
    assert session["plan"]["i"].loc["2020Q1"] == pytest.approx(0.5, abs=1e-10)  # exact
    assert 0 < abs(session["plan"]["y"].loc["2020Q4"] + 1.0) < 1.0  # soft: near, not on
    assert "hold_sd" in session.script() and "free_sd" in session.script()
    np.testing.assert_allclose(_replay(session, "plan").frame.loc["2020Q1":"2021Q4", columns],
                               by_api.frame.loc["2020Q1":"2021Q4", columns], atol=1e-10)
    bad = ForecastModel("estimated", "2020Q1", "2021Q4", hold="i=0.5", free="ei", free_sd="ei")
    assert bad.validate(session) and "scenario" in bad.validate(session)[0]


def test_identification_in_elasticities(session):
    session.run(CheckIdentification("estimated", lags=3, elasticity=True, name="ident"))
    by_api = session["estimated"].identification(lags=3, elasticity=True)
    np.testing.assert_allclose(session["ident"].pairwise, by_api.pairwise, rtol=1e-12)
    assert session["ident"].elasticity
    assert "ident = estimated.identification(lags=3, elasticity=True)" in session.script()


def test_the_new_rows_in_the_qt_forms(session):
    pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import command_dialog

    form = command_dialog(FitEquation, session)
    assert form.on_advanced_tab("hac_correction") and not form.row_visible("hac_correction")
    form.editors["cov"].set("hac")
    assert form.row_visible("hac_correction")
    plan = command_dialog(ForecastModel, session)
    assert plan.on_advanced_tab("hold_sd") and plan.on_advanced_tab("free_sd")
    assert command_dialog(CheckIdentification, session).on_advanced_tab("elasticity")
