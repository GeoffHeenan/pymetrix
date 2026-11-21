"""Measurement maps, the state-space system as a result, and template models."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, MeasurementMap, StateSpaceSystem, templates
from pymetrix import statespace as ss
from pymetrix.models import ModelError


def _dataset(frame):
    frame = frame.copy()
    frame.index = pd.period_range("1990Q1", periods=len(frame), freq="Q", name="time")
    return Dataset.from_frame(frame)


@pytest.fixture(scope="module")
def nk():
    t = templates.new_keynesian()
    return t, t.solve()


@pytest.fixture(scope="module")
def simulated(nk):
    """Model variables simulated from the NK model, and the data they imply through its measurement equations."""
    t, solution = nk
    truth = _dataset(solution.simulate(200, t.shock_sd, seed=4))
    observed = t.measurement.apply(truth)
    return truth, observed


def test_measurement_equations_linearised_by_hand(nk):
    t, solution = nk
    system = t.system()
    assert isinstance(system, StateSpaceSystem) and system.observables == ("infl", "rate", "growth")
    assert system.states == ("y", "pi", "i", "y[-1]", "pi[-1]", "i[-1]")  # growth uses y[-1]: lagged states added
    Z = pd.DataFrame(system.Z, index=system.observables, columns=system.states)
    assert Z.loc["infl", "pi"] == 4 and Z.loc["rate", "i"] == 4  # infl = pi_ss + 4*pi; rate = r_ss + pi_ss + 4*i
    assert Z.loc["growth", "y"] == 4 and Z.loc["growth", "y[-1]"] == -4 and Z.loc["growth", "pi"] == 0
    assert np.allclose(system.d, [2.0, 3.0, 2.0])  # the steady state in the units of the data
    T = pd.DataFrame(system.T, index=system.states, columns=system.states)
    assert T.loc["y[-1]", "y"] == 1 and np.allclose(T.iloc[:3, :3], solution.P)


def test_the_filter_recovers_the_model_from_data_in_data_units(nk, simulated):
    t, _ = nk
    truth, observed = simulated
    estimated = t.system().filter(observed)
    for v in ("pi", "i"):  # observed through annualised inflation and the policy rate: exact
        assert np.allclose(estimated[v].to_numpy()[5:], truth[v].to_numpy()[5:], atol=1e-6)
    offset = estimated["y"].to_numpy() - truth["y"].to_numpy()
    assert np.std(offset) < 1e-6  # GDP growth pins down changes in the gap, not its level: right up to a constant
    assert np.allclose(estimated["growth_fitted"].to_numpy()[5:], observed["growth"].to_numpy()[5:], atol=1e-6)


def test_likelihood_matches_statsmodels_kalman_filter(nk, simulated, monkeypatch):
    t, _ = nk
    _, observed = simulated
    system = t.system(measurement_sd={"growth": 0.5, "infl": 0.2})
    monkeypatch.setattr(ss, "ENGINE", "native")
    native = system.loglike(observed)
    monkeypatch.setattr(ss, "ENGINE", "statsmodels")
    pytest.importorskip("statsmodels")
    assert native == pytest.approx(system.loglike(observed), rel=1e-9)
    assert np.isfinite(native)


def test_apply_is_exact_and_nonlinear():
    t = templates.rbc()
    levels = _dataset(t.solve().simulate(40, t.shock_sd, seed=1))
    observed = t.measurement.apply(levels)
    assert np.allclose(observed["log_y"], np.log(levels["y"])) and np.allclose(observed["log_c"], np.log(levels["c"]))


def test_an_unobserved_output_gap_from_inflation_and_the_policy_rate():
    t = templates.gap_model()
    solution = t.solve()
    truth = _dataset(solution.simulate(240, t.shock_sd, seed=7))
    rng = np.random.default_rng(0)
    observed = t.measurement.apply(truth).frame.assign(infl=lambda f: f["infl"] + rng.normal(0, 0.1, len(f)))
    gap = t.system().filter(Dataset.from_frame(observed))["y_gap"]
    assert np.corrcoef(gap, truth["y_gap"])[0, 1] > 0.9  # the gap is never observed
    assert t.check().ok


def test_measurement_errors_and_mistakes(nk):
    t, solution = nk
    system = t.system(measurement_sd={"growth": 0.5})
    assert system.H[2, 2] == pytest.approx(0.25) and system.H[0, 0] == 0
    with pytest.raises(ModelError, match="cannot depend on the future"):
        MeasurementMap(["x = pi[+1]"])
    with pytest.raises(ModelError, match="more than one"):
        MeasurementMap(["x = pi", "x = i"])
    with pytest.raises(ModelError, match="not variables of the model"):
        solution.system(MeasurementMap(["x = output"]), t.shock_sd)
    with pytest.raises(ModelError, match="observed = expression"):
        MeasurementMap(["x ~ pi"])
    with pytest.raises(ValueError, match="no infl"):
        system.filter(_dataset(pd.DataFrame({"rate": np.zeros(10)})))


def test_templates_and_shocks_from_roles():
    for make in (templates.new_keynesian, templates.rbc, templates.gap_model):
        t = make()
        assert t.check().ok, t.check().errors()
        assert np.allclose(t.model.linearize(t.steady_state()).solve().P, t.solve().P)  # shocks default to Role.SHOCK
    passive = templates.new_keynesian(phi_pi=0.8).check()
    assert not passive.ok and "not unique" in passive.errors()[0]


def test_dialogs_views_and_projects(simulated, tmp_path):
    from pymetrix.commands import KalmanFilter, LoadTemplate, Session, StateSpaceForm, load_project, save_project
    from pymetrix.views import view_sections

    _, observed = simulated
    s = Session()
    s.objects["data"] = observed
    s.run(LoadTemplate("new_keynesian", name="nk"))
    s.run(StateSpaceForm("nk", errors="growth=0.5"))
    s.run(StateSpaceForm("nk", measurement="infl = pi_ss + 4*pi\nrate = 3 + 4*i", measurement_params="pi_ss=2", shock_sd="ei=0.3",
                         name="custom"))
    s.run(KalmanFilter("system", name="filtered"))
    script = s.script()
    for line in ("nk = templates.new_keynesian()", "system = nk.system(measurement_sd={'growth': 0.5})",
                 "custom = nk.system(MeasurementMap(['infl = pi_ss + 4*pi', 'rate = 3 + 4*i'], params={'pi_ss': 2.0}), shock_sd={'ei': 0.3})",
                 "filtered = system.filter(data)"):
        assert line in script
    namespace = {"data": observed}
    exec(script, namespace)  # the generated code runs as plain Python and gives the same result
    assert np.allclose(namespace["filtered"]["y"], s["filtered"]["y"])
    reopened = load_project(save_project(s, tmp_path / "nk.pymx"))
    assert np.allclose(reopened["filtered"]["pi"], s["filtered"]["pi"])
    assert [t for t, _ in view_sections(s["system"])] == ["Measurement", "Measurement matrix", "Transition", "Shocks", "Shock sizes"]
    assert [t for t, _ in view_sections(s["nk"])] == ["Equations", "Parameters", "Measurement", "Shocks"]
    assert StateSpaceForm("missing", measurement="").validate(s) == ["write measurement equations (a solution has none of its own)"]
