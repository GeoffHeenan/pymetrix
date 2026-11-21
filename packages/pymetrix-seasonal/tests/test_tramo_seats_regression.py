"""TRAMO-SEATS with interventions (outliers known in advance) and user regressors (holidays, strikes, policy changes).

Checked independently:
- the regression effects equal generalised least squares by hand on the differenced series, given the ARIMA parameters
  TRAMO estimated (exact up to rounding);
- a regressor's effect goes where it is assigned: the adjusted series with the effect in the irregular differs from the
  one with it in the calendar component by exactly the effect;
- planted effects are recovered.

Skipped without JPype, Java 21 or the JDemetra+ jars (``JDPLUS_PATH`` or ``pymetrix_seasonal.jdemetra.install()``)."""

import numpy as np
import pandas as pd
import pytest
from pymetrix_seasonal import TramoSeats, jdemetra
from scipy.linalg import toeplitz

from pymetrix import Dataset
from pymetrix.errors import DataError, SpecificationError
from pymetrix.seasonal import seasonal_components


@pytest.fixture(scope="module")
def J():
    try:
        return jdemetra.classes()
    except SpecificationError as exc:
        pytest.skip(str(exc))


N = 80
T = np.arange(N)
HOLIDAY = (T % 7 == 3).astype(float)
SHIFT = np.where(T >= 40, 3.0, 0.0)  # a level shift in 2015Q1


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    season = np.tile([1.0, -0.5, 0.3, -0.8], N // 4)
    noise = np.cumsum(rng.normal(0, 0.3, N)) + rng.normal(0, 0.2, N)
    y = 100 + noise + season + SHIFT + 2.0 * HOLIDAY
    idx = pd.period_range("2005Q1", periods=N, freq="Q", name="time")
    return Dataset.from_frame(pd.DataFrame({"y": y, "holiday": HOLIDAY}, index=idx))


def _sd4(v):
    d = np.diff(v)
    return d[4:] - d[:-4]  # (1 - B)(1 - B^4)


def _tramo(J, data, method):
    """JDemetra+'s own result, for the coefficients and ARIMA parameters at full precision."""
    import jpype

    idx = data.frame.index
    y = data.frame["y"].to_numpy()
    spec = method.specification(J, 4, idx)
    R = "jdplus.toolkit.base.api.timeseries.regression."
    ctx = J(R + "ModellingContext")()
    sup = J(R + "TsDataSuppliers")()
    of = J("jdplus.toolkit.base.r.timeseries.TsUtility").of
    sup.set("holiday", J("jdplus.toolkit.base.api.timeseries.StaticTsDataSupplier")(of(4, 2005, 1, jpype.JArray(jpype.JDouble)(HOLIDAY))))
    ctx.getTsVariableManagers().set("r", sup)
    res = J("jdplus.tramoseats.base.r.TramoSeats").process(of(4, 2005, 1, jpype.JArray(jpype.JDouble)(y)), spec, ctx)

    def get(k):
        return res.getData(k, J("java.lang.Object"))

    names = [str(v) for v in get("regression.details.description")]
    coefs = [float(c) for c in get("regression.details.coefficients")]
    return dict(zip(names, coefs)), float(get("arima.theta(1)").getCoefficient()), float(get("arima.btheta(1)").getCoefficient())


def test_regression_effects_are_gls_given_tramos_arima(J, data):
    method = TramoSeats(spec="rsa0", regressors="holiday", interventions="ls 2015Q1")
    coefs, th, bth = _tramo(J, data, method)
    assert set(coefs) == {"const", "holiday", "LS (2015-01)"}
    y = data.frame["y"].to_numpy()
    ls = np.where(T >= 40, 0.0, -1.0)  # JDemetra+'s level shift: zero from its date on
    X = np.column_stack([np.ones(N - 5), _sd4(HOLIDAY), _sd4(ls)])
    ma = np.zeros(6)
    ma[[0, 1, 4, 5]] = 1.0, th, bth, th * bth  # (1 + th B)(1 + bth B^4)
    col = np.zeros(N - 5)
    col[:6] = [np.sum(ma[k:] * ma[:6 - k]) for k in range(6)]
    Vi = np.linalg.inv(toeplitz(col))
    beta = np.linalg.solve(X.T @ Vi @ X, X.T @ Vi @ _sd4(y))
    np.testing.assert_allclose([coefs["const"], coefs["holiday"], coefs["LS (2015-01)"]], beta, rtol=1e-6, atol=1e-8)
    assert coefs["holiday"] == pytest.approx(2.0, abs=0.2) and coefs["LS (2015-01)"] == pytest.approx(3.0, abs=0.3)


def test_effects_go_to_their_component(J, data):
    notes = {}
    calendar, _ = seasonal_components(data, "y", TramoSeats(spec="rsa0", regressors="holiday", interventions="ls 2015Q1"), notes)
    irregular, _ = seasonal_components(data, "y", TramoSeats(spec="rsa0", regressors="holiday=irregular", interventions="ls 2015Q1"))
    coefs, _, _ = _tramo(J, data, TramoSeats(spec="rsa0", regressors="holiday", interventions="ls 2015Q1"))
    np.testing.assert_allclose(irregular.to_numpy() - calendar.to_numpy(), coefs["holiday"] * HOLIDAY, atol=1e-8)  # additive here
    assert notes["tramo_regression"].startswith("holiday 1.9") and "LS (2015-01) 3.0" in notes["tramo_regression"]
    assert "LS (2015-01)" not in notes["tramo_outliers"]  # given, not found


def test_interventions_alone_and_with_outlier_detection(J, data):
    notes = {}
    seasonal_components(data, "y", TramoSeats(spec="rsa3", interventions="ls 2015Q1, ao 2010Q3"), notes)
    assert "LS (2015-01)" in notes["tramo_regression"] and "AO (2010-07)" in notes["tramo_regression"]
    assert "LS (2015-01)" not in notes["tramo_outliers"]


def test_through_the_dataset_and_derived_inputs(J, data):
    out = data.seasonal_adjust("y", TramoSeats(spec="rsa0", regressors="holiday"))
    assert np.all(np.isfinite(out.frame["y_sa"].to_numpy()))
    recipe = out.variables["y_sa"].recipe
    assert recipe.inputs(out) == frozenset({"y", "holiday"})  # a changed regressor recomputes the adjusted series


def test_refusals(J, data):
    with pytest.raises(SpecificationError, match="is not a type"):
        TramoSeats(interventions="xx 2015Q1")
    with pytest.raises(SpecificationError, match="not a series name"):
        TramoSeats(regressors="holiday=weekly")
    with pytest.raises(SpecificationError, match="outside the series"):
        seasonal_components(data, "y", TramoSeats(spec="rsa0", interventions="ls 2030Q1"))
    with pytest.raises(SpecificationError, match="not series of the dataset"):
        seasonal_components(data, "y", TramoSeats(spec="rsa0", regressors="strike"))
    gappy = Dataset.from_frame(data.frame.assign(holiday=data.frame["holiday"].where(np.arange(N) != 10)))
    with pytest.raises(DataError, match="missing values"):
        seasonal_components(gappy, "y", TramoSeats(spec="rsa0", regressors="holiday"))
    assert TramoSeats(regressors="holiday, strike=irregular").user_regressors() == [("holiday", "calendar"), ("strike", "irregular")]
    assert TramoSeats(interventions="LS 2015Q1, ao 2010Q3").fixed_outliers() == [("ls", "2015Q1"), ("ao", "2010Q3")]


def test_the_dialog_and_its_script(J, data):
    from pymetrix.commands import Session
    from pymetrix.commands.seasonal import SeasonalAdjust

    s = Session()
    s.objects["data"] = data
    s.run(SeasonalAdjust("y", data="data", method="TRAMO-SEATS",
                         settings="spec='rsa0', regressors='holiday=irregular', interventions='ls 2015Q1'", name="adjusted"))
    direct = data.seasonal_adjust("y", TramoSeats(spec="rsa0", regressors="holiday=irregular", interventions="ls 2015Q1"))
    np.testing.assert_allclose(s["adjusted"].frame["y_sa"].to_numpy(), direct.frame["y_sa"].to_numpy(), atol=1e-10)
    ns = {"data": data}
    exec(s.script(), ns)
    np.testing.assert_allclose(ns["adjusted"].frame["y_sa"].to_numpy(), direct.frame["y_sa"].to_numpy(), atol=1e-10)
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.forms import build
    from pymetrix.plugins import registry
    from pymetrix.ui.qt import command_dialog

    dialog = command_dialog(SeasonalAdjust, s)
    dialog.editors["method"].widget.setCurrentText("TRAMO-SEATS")
    sub = dialog.editors["settings"].sub
    sub["regressors"].widget.setText("holiday, strike=irregular")  # commas and = inside a setting: quoted when written
    sub["interventions"].widget.setText("ls 2015Q1, ao 2010Q3")
    method = build(registry("seasonal").get("TRAMO-SEATS"), dialog.editors["settings"].get())
    assert (method.regressors, method.interventions) == ("holiday, strike=irregular", "ls 2015Q1, ao 2010Q3")
