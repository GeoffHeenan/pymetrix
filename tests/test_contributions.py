"""Data contributions: the Kalman-smoothed estimates split by observed series. Checked by the parts adding up exactly
to the smoothed path and the forecast, and by a series with no news contributing nothing; and a direct
check of linearity: doubling a series' news doubles its contribution; and against the smoothed-state weights
calculated directly, E[a | y] = Cov(a, y) Var(y)^-1 (y - d) over the whole sample at once (no Kalman recursion).

With no measurement error, a variable observed as it is (pi through inflation, i through the policy rate) is known
exactly, so its smoothed estimate comes from its own series alone: other series contribute nothing to it in sample.
Only with measurement errors do the series share the work."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, templates


@pytest.fixture(scope="module")
def nk_fit():
    nk = templates.new_keynesian()
    truth = nk.solve().simulate(80, nk.shock_sd, seed=4)
    truth.index = pd.period_range("2000Q1", periods=80, freq="Q", name="time")
    data = nk.measurement.apply(Dataset.from_frame(truth))
    ragged = data.with_series(infl=data["infl"].where(data.frame.index < pd.Period("2019Q2"), np.nan))
    return nk, nk.estimate(ragged, estimate=["rho"])


def test_the_parts_add_up_to_the_smoothed_path_and_the_forecast(nk_fit):
    nk, fit = nk_fit
    c = fit.contributions(horizon=4)
    steady = fit.solution.model.steady.values
    smoothed = fit.smoothed.frame
    for v in ("y", "pi", "i"):
        assert np.allclose(c.contributions[v].iloc[:80].sum(axis=1), smoothed[v] - steady[v], atol=1e-12), v
    forecast = fit.forecast(("2020Q1", "2020Q4")).solved().frame
    assert np.allclose(c.deviations["y"].iloc[80:], forecast["y"] - steady["y"], atol=1e-12)
    assert list(c.contributions["y"].columns) == ["infl", "rate", "growth", "start"]
    assert c.title == "Data contributions" and "Data contributions: y" in c.table("y").title


def test_contributions_are_linear_in_each_series(nk_fit):
    """Doubling one series' deviations from what the model expects doubles its contribution and leaves the others."""
    _, fit = nk_fit
    system, data = fit.system, fit.data
    j = list(system.observables).index("rate")
    expected = system.d[j]
    doubled = data.with_series(rate=expected + 2 * (data["rate"] - expected))
    a, b = system.contributions(data), system.contributions(doubled)
    for v in ("y", "pi", "i"):
        assert np.allclose(b.contributions[v]["rate"], 2 * a.contributions[v]["rate"], atol=1e-12)
        assert np.allclose(b.contributions[v]["growth"], a.contributions[v]["growth"], atol=1e-12)
    flat = data.with_series(rate=pd.Series(expected, index=data.frame.index))
    assert np.allclose(system.contributions(flat).contributions["y"]["rate"], 0.0, atol=1e-12)  # no news, no contribution


def _direct(system, data):
    """Each observed series' contribution to each state, in sample, from the weights of the conditional expectation
    of every state given every observation (the system starting from its stationary distribution)."""
    y, _ = system.observations(data)
    n, k = len(y), len(system.states)
    a0, P0, D = system.start("auto")
    assert not D.any() and not np.any(a0) and not np.any(system.c)
    powers = [np.eye(k)]
    for _ in range(n):
        powers.append(system.T @ powers[-1])
    cov = np.zeros((n * k, n * k))  # Cov(a(t), a(s)) = T^(t-s) P0 for t >= s: stationary
    for t in range(n):
        for u in range(t + 1):
            block = powers[t - u] @ P0
            cov[t * k:(t + 1) * k, u * k:(u + 1) * k] = block
            cov[u * k:(u + 1) * k, t * k:(t + 1) * k] = block.T
    Z = np.kron(np.eye(n), system.Z)
    seen = ~np.isnan(y.ravel())
    Zs = Z[seen]
    Syy = Zs @ cov @ Zs.T + np.kron(np.eye(n), system.H)[np.ix_(seen, seen)]
    W = cov @ Zs.T @ np.linalg.inv(Syy)  # states (stacked) x observations seen
    news = (y - system.d).ravel()[seen]
    which = np.tile(np.arange(len(system.observables)), n)[seen]
    return {o: (W[:, which == j] @ news[which == j]).reshape(n, k) for j, o in enumerate(system.observables)}


@pytest.mark.parametrize("errors", [None, {"infl": 0.3, "rate": 0.2, "growth": 0.5}])
def test_the_parts_equal_the_smoothed_state_weights(nk_fit, errors):
    nk, fit = nk_fit
    system = fit.system if errors is None else fit.solution.system(nk.measurement.with_errors(errors), fit.shock_sd)
    c = system.contributions(fit.data)
    direct = _direct(system, fit.data)
    for o, part in direct.items():
        for i, v in enumerate(system.variables):
            assert np.allclose(c.contributions[v][o].to_numpy(), part[:, i], atol=1e-8), (o, v)
    assert np.allclose(c.contributions["y"]["start"], 0.0, atol=1e-12)

    def others(v, own):  # the largest part of v from a series other than its own
        return max(c.contributions[v][o].abs().max() for o in system.observables if o != own)

    if errors is None:  # pi and i observed without error: nothing but their own series moves them
        seen = ~np.isnan(fit.data["infl"].to_numpy())
        assert others("i", "rate") < 1e-10 and np.abs(c.contributions["pi"].loc[seen, ["rate", "growth"]].to_numpy()).max() < 1e-10
        assert c.contributions["y"]["rate"].abs().max() > 1e-3  # y's level is inferred from the other series too
    else:  # with measurement errors every series informs every variable
        assert others("pi", "infl") > 0.01 and others("i", "rate") > 0.01


def test_the_dialog(nk_fit):
    from pymetrix.commands import DataContributions, Session
    from pymetrix.views import view_sections

    _, fit = nk_fit
    s = Session()
    s.objects["fit"] = fit
    s.run(DataContributions("fit", horizon=4))
    assert "contributions = fit.contributions(horizon=4)" in s.script()
    titles = [t for t, _ in view_sections(s["contributions"])]
    assert "Table: y" in titles and "Shocks" not in titles
