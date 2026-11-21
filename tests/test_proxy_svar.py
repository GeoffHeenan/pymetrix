"""Proxy SVARs, checked independently: the impact column by its covariance formula by hand; the true column recovered
from a simulated non-recursive SVAR (where Cholesky would be wrong); the relative impact equal to LP-IV's horizon-0
coefficient with the same lags (an identity, since the VAR's residuals are orthogonal to its lags); the responses as
the VAR's moving-average coefficients times the impact; the first stage by least squares."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pymetrix import VAR, Dataset, LocalProjection
from pymetrix.errors import SpecificationError

B = np.array([[0.8, 0.24, 0.0], [0.4, 0.8, 0.0], [-0.24, 0.32, 0.8]])  # not lower triangular
A1 = np.array([[0.5, 0.1, 0.0], [0.2, 0.4, 0.1], [0.0, 0.2, 0.3]])


def _data(T=1200, seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=(T, 3))
    y = np.zeros((T, 3))
    for t in range(1, T):
        y[t] = A1 @ y[t - 1] + B @ eps[t]
    z = eps[:, 0] + rng.normal(0, noise, T)  # the proxy: the first shock, measured with error
    frame = pd.DataFrame({"r": y[:, 0], "x": y[:, 1], "p": y[:, 2], "z": z}, index=pd.period_range("1900-01", periods=T, freq="M", name="time"))
    return Dataset.from_frame(frame)


def test_the_impact_by_hand_and_recovered():
    data = _data()
    var = VAR(["r", "x", "p"], lags=1).fit(data)
    ps = var.proxy_svar(data, "z", draws=0)
    U = var.estimate.U
    z = data["z"].reindex(var.arrays.index).to_numpy()
    c = np.array([np.cov(U[:, j], z, ddof=0)[0, 1] for j in range(3)])
    b = c / c[0]
    b = b / np.sqrt(b @ np.linalg.solve(var.estimate.sigma, b))
    assert ps.impact.to_numpy() == pytest.approx(b, rel=1e-10)
    assert ps.impact.to_numpy() == pytest.approx(B[:, 0], abs=0.06)  # the true column, where Cholesky's first column is not
    chol = np.linalg.cholesky(var.estimate.sigma)[:, 0]
    assert np.abs(chol - B[:, 0]).max() > 3 * np.abs(ps.impact.to_numpy() - B[:, 0]).max()


def test_the_lp_iv_identity_and_the_responses():
    data = _data(T=600, seed=1)
    var = VAR(["r", "x", "p"], lags=2).fit(data)
    unit = var.proxy_svar(data, "z", scale="unit", periods=6, draws=0)
    for v in ("x", "p"):
        lp = LocalProjection(v, "r", horizons=0, lags=2, controls=[w for w in ("r", "x", "p") if w != "r"], instruments=["z"]).fit(data)
        assert lp.irf.loc[0, "response"] == pytest.approx(unit.impact[v], rel=1e-10)
    from pymetrix.var import _ma

    assert unit.irf.values[:, :, 0] == pytest.approx((_ma(var.coefs, 6) @ unit.impact.to_numpy()[:, None])[:, :, 0])
    u, z = var.estimate.U[:, 0], data["z"].reindex(var.arrays.index).to_numpy()
    slope, _, r, _, _ = stats.linregress(z, u)
    F = r**2 / (1 - r**2) * (len(u) - 2)
    assert unit.first_stage[0] == pytest.approx(F, rel=1e-8)


def test_bands_and_refusals():
    data = _data(T=500, seed=2)
    var = VAR(["r", "x", "p"], lags=1).fit(data)
    ps = var.proxy_svar(data, "z", periods=4, draws=200, seed=3)
    assert np.all(ps.irf.lower <= ps.irf.values + 1e-12) and np.all(ps.irf.values <= ps.irf.upper + 1e-12)
    assert ps.irf.upper[0, 0, 0] - ps.irf.lower[0, 0, 0] > 0
    with pytest.raises(SpecificationError, match="not one of"):
        var.proxy_svar(data, "z", normalise_on="gdp")
    assert ps.summary().title.startswith("Proxy SVAR")


def test_the_dialog_with_a_variable_to_normalise_on():  # the call builder once took target= as its own argument
    from pymetrix.commands import COMMANDS, Session

    data = _data(T=300, seed=4)
    s = Session()
    s.objects["data"] = data
    s.objects["var"] = VAR(["r", "x", "p"], lags=1).fit(data)
    s.run(COMMANDS["ProxySVARCommand"].cls("z", normalise_on="x", draws=0))
    assert "normalise_on='x'" in s.history[-1].code and s.objects["proxy"].summary().title.startswith("Proxy SVAR")
    assert s.objects["proxy"].normalise_on == "x"


def test_the_old_name_target_still_works():
    data = _data(T=300, seed=5)
    var = VAR(["r", "x", "p"], lags=1).fit(data)
    new = var.proxy_svar(data, "z", normalise_on="x", scale="unit", draws=0)
    with pytest.warns(DeprecationWarning, match="'target' .* is now called 'normalise_on'"):
        old = var.proxy_svar(data, "z", target="x", scale="unit", draws=0)
    assert old.impact.to_numpy() == pytest.approx(new.impact.to_numpy(), rel=1e-12) and old.normalise_on == "x"
    with pytest.raises(TypeError, match="both"):
        var.proxy_svar(data, "z", target="x", normalise_on="x")


def test_old_scripts_calls_and_dialog_steps_replay():
    import warnings

    from pymetrix.calls import Call
    from pymetrix.codec import encode
    from pymetrix.commands import Session
    from pymetrix.commands.registry import from_json

    data = _data(T=300, seed=6)
    s = Session()
    s.objects["data"] = data
    s.objects["var"] = VAR(["r", "x", "p"], lags=1).fit(data)
    with warnings.catch_warnings():  # a recorded script typed with the old keyword: recorded under the new name
        warnings.simplefilter("ignore", DeprecationWarning)
        s.run_code('ps = var.proxy_svar(data, "z", target="x", draws=0)')
    arguments = s.history[-1].command.arguments
    assert arguments["normalise_on"] == "x" and "target" not in arguments and s.history[-1].command.version == 2
    assert "normalise_on='x'" in s.history[-1].code
    # a call saved at version 1, with target among its arguments
    saved = Call("pymetrix:fitted_var.proxy_svar", 1, {"$ref": "var"},
                 {"data": {"$ref": "data"}, "instrument": "z", "target": "x", "periods": 20, "scale": "sd", "draws": 0,
                  "block": None, "alpha": 0.05, "seed": 0}, output="old")
    assert "normalise_on='x'" in saved.to_code()
    s.run(saved)
    assert s.objects["old"].normalise_on == "x"
    # a dialog step saved at schema 1, with its field target
    step = from_json({"kind": "command", "id": "pymetrix:proxy_svar", "version": 1,
                      "fields": {"instrument": "z", "target": "x", "draws": 0, "name": "older"}})
    assert step.normalise_on == "x"
    s.run(step)
    assert s.objects["older"].normalise_on == "x" and encode(step.normalise_on) == "x"
