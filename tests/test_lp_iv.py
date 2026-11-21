"""LP-IV (local projections with external instruments), checked independently: the shock as its own instrument gives
the least-squares projection exactly; each horizon's coefficient against linearmodels' IV2SLS on the same regressors;
the true responses recovered where least squares is biased by an endogenous shock."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, LocalProjection
from pymetrix.errors import SpecificationError

THETA = np.array([1.0, 0.8, 0.5, 0.2, 0.0, 0.0])  # the response of y to a unit structural shock, by horizon


def _data(T=3000, seed=0):
    rng = np.random.default_rng(seed)
    e, u = rng.normal(size=T), rng.normal(size=T)
    x = e + u  # the policy variable: the structural shock plus an endogenous part
    y = np.convolve(e, THETA)[:T] + 0.8 * u + rng.normal(0, 0.5, T)  # u moves y too: least squares on x is biased
    z = e + rng.normal(0, 1.0, T)  # an external instrument: correlated with e only
    idx = pd.period_range("1800-01", periods=T, freq="M", name="time")
    return Dataset.from_frame(pd.DataFrame({"y": y, "x": x, "z": z}, index=idx))


def test_the_shock_as_its_own_instrument_is_least_squares():
    data = _data(T=600)
    ols = LocalProjection("y", "x", horizons=4, lags=2).fit(data)
    iv = LocalProjection("y", "x", horizons=4, lags=2, instruments=["x"]).fit(data)
    assert iv.irf["response"].to_numpy() == pytest.approx(ols.irf["response"].to_numpy(), rel=1e-10)
    assert iv.irf["se"].to_numpy() == pytest.approx(ols.irf["se"].to_numpy(), rel=1e-8)


def test_each_horizon_against_iv2sls():
    lm = pytest.importorskip("linearmodels")
    data = _data(T=800, seed=1)
    lp = LocalProjection("y", "x", horizons=3, lags=1, instruments=["z"]).fit(data)
    f = data.frame
    for h in (0, 3):
        frame = pd.DataFrame({"lhs": f["y"].shift(-h), "x": f["x"], "y1": f["y"].shift(1), "x1": f["x"].shift(1), "z": f["z"]}).dropna()
        frame = frame.reset_index(drop=True)
        ref = lm.IV2SLS(frame["lhs"], frame[["y1", "x1"]].assign(const=1.0), frame[["x"]], frame[["z"]]).fit(cov_type="unadjusted")
        assert lp.irf.loc[h, "response"] == pytest.approx(ref.params["x"], rel=1e-10)


def test_lp_iv_recovers_the_responses_where_least_squares_is_biased():
    data = _data()
    ols = LocalProjection("y", "x", horizons=5, lags=0).fit(data)
    iv = LocalProjection("y", "x", horizons=5, lags=0, instruments=["z"]).fit(data)
    resp, se = iv.irf["response"].to_numpy(), iv.irf["se"].to_numpy()
    assert np.all(np.abs(resp - THETA) < 4 * se)
    assert ols.irf.loc[0, "response"] > THETA[0] - 0.2 and abs(ols.irf.loc[0, "response"] - (THETA[0] + 0.8) / 2) < 0.1  # biased
    assert (iv.irf["first-stage F"] > 100).all()
    with pytest.raises(SpecificationError, match="not both"):
        LocalProjection("y", "x", instruments=["z"], state="z")


def test_the_dialog():
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS

    s = Session()
    s.objects["data"] = _data(T=400)
    s.run(COMMANDS["FitLocalProjection"].cls("y", "x", horizons=2, lags=1, instruments="z"))
    assert "instruments=['z']" in s.history[-1].code and "first-stage F" in s.objects[s.history[-1].outputs[0]].irf
