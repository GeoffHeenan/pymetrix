"""The worked wrapper example (examples/custom_wrapper.py, explained in docs/wrappers.md): each wrapper reproduces
statsmodels called directly, passes the plugin conformance checks, and works like a built-in."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sm = pytest.importorskip("statsmodels.api")

from pymetrix import Dataset, Equation, backtest  # noqa: E402
from pymetrix.plugins import _REGISTRIES, _PLUGINS, registry  # noqa: E402
from pymetrix.testing import check_plugin  # noqa: E402

EXAMPLES = Path(__file__).parents[1] / "examples"


@pytest.fixture
def wrappers(monkeypatch):
    """The example imported as a user's module would be; its registrations (methods, models, command) undone afterwards."""
    from pymetrix.commands.registry import COMMAND_IDS, COMMANDS

    monkeypatch.syspath_prepend(str(EXAMPLES))
    saved = {g: dict(r._entries) for g, r in _REGISTRIES.items()}
    plugins, ids, commands = dict(_PLUGINS), dict(COMMAND_IDS), dict(COMMANDS)
    spec = importlib.util.spec_from_file_location("custom_wrapper", EXAMPLES / "custom_wrapper.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_wrapper"] = module
    spec.loader.exec_module(module)
    yield module
    for g, entries in saved.items():
        _REGISTRIES[g]._entries = entries
    for target, before in ((_PLUGINS, plugins), (COMMAND_IDS, ids), (COMMANDS, commands)):
        target.clear()
        target.update(before)
    sys.modules.pop("custom_wrapper", None)


@pytest.fixture
def data():
    """Consumption and income with a few large outliers, where Huber and least squares differ."""
    rng = np.random.default_rng(3)
    n = 120
    y = 100 + np.cumsum(rng.normal(0.5, 1, n))
    c = 5 + 0.8 * y + rng.normal(0, 1, n)
    c[[20, 55, 90]] += [15, -12, 18]
    frame = pd.DataFrame({"c": c, "y": y}, index=pd.period_range("1990Q1", periods=n, freq="Q"))
    return Dataset.from_frame(frame)


def test_huber_reproduces_statsmodels(wrappers, data):
    fit = Equation("c ~ y").fit(data, method=wrappers.Huber())
    direct = sm.RLM(data["c"].to_numpy(), sm.add_constant(data["y"].to_numpy()), M=sm.robust.norms.HuberT(t=1.345)).fit()
    assert np.allclose(fit.params.values, direct.params) and np.allclose(fit.params.cov, direct.cov_params())
    assert fit.extra["weights"].min() < 0.5  # the outliers are downweighted
    ols = Equation("c ~ y").fit(data)
    assert abs(fit.params["y"] - 0.8) < abs(ols.params["y"] - 0.8) or abs(fit.params["const"] - 5) < abs(ols.params["const"] - 5)
    assert fit.provenance.method["backend"].startswith("statsmodels ")


def test_theta_reproduces_statsmodels(wrappers, data):
    from statsmodels.tsa.forecasting.theta import ThetaModel

    fit = wrappers.Theta("y").fit(data, span=(None, "2015Q4"))
    fc = fit.forecast(data, ("2016Q1", "2016Q4")).frame
    sample = data["y"].loc[: pd.Period("2015Q4", freq="Q")].to_numpy()
    direct = ThetaModel(sample, period=4, deseasonalize=False).fit()
    assert np.allclose(fc["forecast"], direct.forecast(4, theta=2.0))
    band = direct.prediction_intervals(4, theta=2.0)
    assert np.allclose(fc["lower"], band["lower"]) and np.allclose(fc["upper"], band["upper"])


def test_the_wrappers_work_like_builtins(wrappers, data):
    check_plugin(wrappers.PLUGIN)  # forms, help (wraps resolves), settings round trip, for every registration
    assert registry("methods").get("Huber robust regression").obj is wrappers.Huber
    from pymetrix.commands import FitEquation, Session

    s = Session()
    s.objects["data"] = data
    s.run(FitEquation("c ~ y", method="Huber robust regression", method_settings="t=2.0", name="robust"))
    s.run(wrappers.FitTheta("y", name="theta"))
    script = s.script()
    assert "method=Huber(t=2.0)" in script and "theta = Theta('y', theta=2.0).fit(data)" in script
    assert wrappers.FitTheta.menu == "Estimate/Theta method..." and wrappers.FitTheta.command_id == "example_wrappers:fit_theta"
    bt = backtest(s["theta"], data, ("2010Q1", "2018Q4"), horizon=4, label="Theta")  # re-estimated with spec.fit at each origin
    assert len(bt.errors) == 36 * 4 and bt.density_scores(["y"])["n"].sum() == 36 * 4  # with densities, from the se column
    hub = backtest(s["robust"], data, ("2010Q1", "2018Q4"), horizon=1)  # a method's fit is re-estimated like least squares
    assert len(hub.errors) == 36
