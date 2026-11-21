"""Choosing an equation's lags by information criteria (Equation.select_lags).

- An AR(3) with a constant: AIC, BIC and HQ differences between lag lengths against statsmodels' ar_select_order
  (AutoReg fitted on a common sample, holding back max_lags), to 1e-8, and the same lags chosen.
- Every candidate on the same sample; distributed lags of several variables; lags of a transformed left-hand side;
  lags already in the equation not added again; ARMA errors (maximum likelihood); refusals; the dialog."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation
from pymetrix.errors import SpecificationError
from pymetrix.estimation.timeseries import ARMA


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    n = 400
    y, x = np.zeros(n), rng.normal(size=n)
    for t in range(3, n):
        y[t] = 0.5 * y[t - 1] - 0.3 * y[t - 2] + 0.2 * y[t - 3] + 0.4 * x[t] + rng.normal()
    return Dataset.from_frame(pd.DataFrame({"y": y, "x": x, "level": np.exp(np.cumsum(0.01 * y))},
                                           index=pd.period_range("1900Q1", periods=n, freq="Q", name="time")))


def test_an_autoregression_against_statsmodels(data):
    from statsmodels.tsa.ar_model import ar_select_order

    search = Equation("y ~ 1").select_lags(data, max_lags=8)
    frame = search.table.to_frame()
    y = data.frame.y.to_numpy()
    for ours, theirs in (("AIC", "aic"), ("BIC", "bic"), ("HQ", "hqic")):
        ref = ar_select_order(y, 8, ic=theirs, trend="c")
        values = getattr(ref, theirs)
        theirs_by_p = np.array([values[tuple(range(1, p + 1)) if p else 0] for p in range(9)])
        assert np.allclose(frame[ours].to_numpy() - frame[ours].iloc[0], theirs_by_p - theirs_by_p[0], atol=1e-8)
        assert search.best[ours.lower()] == len(ref.ar_lags)
    assert search.best == {"aic": 3, "bic": 3, "hq": 3} and search.equation("bic").text == "y ~ 1 + y[-1] + y[-2] + y[-3]"
    assert set(frame["nobs"]) == {392} and search.sample == ("1902Q1", "1999Q4")


def test_distributed_transformed_and_existing_lags(data):
    both = Equation("y ~ x").select_lags(data, max_lags=3, of=["y", "x"])
    assert both.equations[2].text == "y ~ x + y[-1] + x[-1] + y[-2] + x[-2]"
    logs = Equation("dlog(level) ~ x").select_lags(data, max_lags=2)
    assert logs.equations[2].text == "dlog(level) ~ x + dlog(level)[-1] + dlog(level)[-2]"
    existing = Equation("y ~ y[-1] + x").select_lags(data, max_lags=3)
    assert existing.equations[1].text == "y ~ y[-1] + x" and existing.equations[3].text == "y ~ y[-1] + x + y[-2] + y[-3]"
    arma = Equation("y ~ x", errors=ARMA(1, 0)).select_lags(data, max_lags=2)
    assert arma.equations[2].errors == ARMA(1, 0) and np.isfinite(arma.table.to_frame()["HQ"]).all()


def test_refusals_and_the_dialog(data):
    from pymetrix.commands import EquationLagSearch, Session

    with pytest.raises(SpecificationError, match="at least 1"):
        Equation("y ~ x").select_lags(data, max_lags=0)
    with pytest.raises(SpecificationError, match="identit|linear"):
        Equation("y = x + level").select_lags(data)
    with pytest.raises(SpecificationError, match="criterion is one of"):
        Equation("y ~ x").select_lags(data, max_lags=2).equation("fpe")
    session = Session()
    session.objects["data"] = data
    session.run(EquationLagSearch("y ~ x", of="y, x", max_lags=2))
    script = session.script()
    assert "lag_search = Equation('y ~ x').select_lags(data, max_lags=2, of=['y', 'x'])" in script
    ns = {"data": data}
    exec(script, ns)
    assert ns["lag_search"].best == session["lag_search"].best
