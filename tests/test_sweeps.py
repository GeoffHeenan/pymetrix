"""Parameter sweeps and scenario lists: each run checked against the same model solved on its own."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Scenario, compare_solutions, scenarios, sweep, templates
from pymetrix.commands import CompareScenarios, LoadTemplate, ParameterSweep, Session
from pymetrix.errors import NotFoundError, SpecificationError
from pymetrix.commands.sweeps import _values


@pytest.fixture(scope="module")
def nk():
    return templates.new_keynesian()


def test_each_run_equals_the_model_solved_with_those_values(nk):
    s = sweep(nk, {"phi_pi": [1.2, 2.0], "kappa": [0.05, 0.1]}, "irf", periods=8)
    assert len(s) == 4 and s.points[1] == (1.2, 0.1) and all(e is None for e in s.errors)
    alone = templates.new_keynesian(phi_pi=2.0, kappa=0.05).solve().irf(8, shock_sd=nk.shock_sd)
    assert np.allclose(s[{"phi_pi": 2.0, "kappa": 0.05}].values, alone.values)
    table = s.table("y", "ei", horizon=2)
    assert table.index[2] == "phi_pi=2, kappa=0.05" and table.rows[2][0] == pytest.approx(alone["y", "ei"].iloc[2])
    assert len(s.plot("y", "ei").panels[0].layers) == 4
    paired = sweep(nk, {"phi_pi": [1.2, 2.0], "kappa": [0.05, 0.1]}, "steady_state", grid="zip")
    assert paired.points == ((1.2, 0.05), (2.0, 0.1))
    sizes = sweep(nk, {"sd_ei": [0.1, 0.2]}, "moments")
    assert sizes.table("i").rows[1][0] > sizes.table("i").rows[0][0]  # larger policy shocks, more volatile rates
    custom = sweep(nk, {"beta": [0.98, 0.99]}, lambda m: m.solve().irf(4)["pi", "ey"].iloc[1])
    assert custom.values(float).shape == (2, 1)


def _by_hand(phi_pi, sigma=1.0, beta=0.99, kappa=0.1, rho=0.8, phi_y=0.5, periods=12):
    """Responses of (y, pi, i) to a unit demand shock in the three-equation NK model, by undetermined coefficients:
    the shocks are iid, so the lagged rate is the only state; x(t) = p i(t-1) + q e(t) and E x(t+1) = p i(t)."""
    from scipy.optimize import brentq

    def p_of(g):  # y and pi on the lagged rate, given the rate's own persistence g
        py = -g / (sigma * (1 - g) - g * kappa / (1 - beta * g))
        return py, kappa * py / (1 - beta * g)

    g = brentq(lambda g: rho + (1 - rho) * (phi_pi * p_of(g)[1] + phi_y * p_of(g)[0]) - g, 1e-9, 0.6)  # the stable root
    py, ppi = p_of(g)
    a = py - (1 - ppi) / sigma  # impact: y = a i + 1, pi = beta ppi i + kappa y, i = (1 - rho)(phi_pi pi + phi_y y)
    A = np.array([[1, 0, -a], [-kappa, 1, -beta * ppi], [-(1 - rho) * phi_y, -(1 - rho) * phi_pi, 1]])
    q = np.linalg.solve(A, [1.0, 0, 0])
    out = [q]
    for h in range(1, periods):
        out.append(np.array([py, ppi, g]) * g ** (h - 1) * q[2])
    return np.array(out)


def test_the_swept_parameter_reaches_each_solution(nk):
    """Each run of a sweep over phi_pi equals the hand calculation at its own phi_pi. The output response to a demand
    shock hardly moves (an iid shock, a smoothed rule and a flat Phillips curve leave policy little to do), so that
    panel looks the same for every run; output's response to a cost-push shock shows the difference (it doubles)."""
    values = [1.2, 1.5, 2.0, 3.0]
    s = sweep(nk, {"phi_pi": values}, "irf", periods=12)
    for v, run in zip(values, s.results):
        hand = 0.5 * _by_hand(v, periods=13)  # the template's demand shock (sd 0.5), horizons 0 to 12
        for i, name in enumerate(("y", "pi", "i")):
            assert np.allclose(run[name, "ey"].to_numpy(), hand[:, i], atol=1e-10), (v, name)
    y_ey = s.table("y", "ey", horizon=0).to_frame().iloc[:, 0]
    y_epi = s.table("y", "epi", horizon=0).to_frame().iloc[:, 0]
    assert y_ey.max() - y_ey.min() < 0.01 * y_ey.abs().max()  # under 1%: the four lines overlap
    assert y_epi.abs().max() > 1.9 * y_epi.abs().min()


def test_the_graph_shows_the_pair_chosen(nk):
    from pymetrix.views import view_sections

    s = sweep(nk, {"phi_pi": [1.2, 3.0]}, "irf", periods=6, response="y", shock="epi")
    assert s.plot().panels[0].title == "y, shock to epi" and s.plot("pi", "ei").panels[0].title == "pi, shock to ei"
    assert sweep(nk, {"phi_pi": [1.2]}, periods=6).plot().panels[0].title == "y, shock to ey"  # the first pair
    assert [t for t, _ in view_sections(s)][:3] == ["Runs", "Graph", "y, shock to epi"]
    with pytest.raises(NotFoundError, match="no shock 'ez'"):
        sweep(nk, {"phi_pi": [1.2]}, response="y", shock="ez")
    with pytest.raises(NotFoundError, match="no response 'q'"):
        sweep(nk, {"phi_pi": [1.2]}, response="q")
    with pytest.raises(SpecificationError, match="what='irf'"):
        sweep(nk, {"phi_pi": [1.2]}, "moments", response="y")
    session = Session()
    session.run(LoadTemplate("new_keynesian", name="nk"))
    session.run(ParameterSweep("nk", "phi_pi", "1.2, 3", periods=6, response="y", shock="epi"))
    assert "runs = sweep(nk, {'phi_pi': [1.2, 3.0]}, response='y', shock='epi', periods=6)" in session.script()
    assert session["runs"].plot().panels[0].title == "y, shock to epi"


def test_the_demo_opens_on_the_cost_push_shock():
    from pymetrix import demo

    runs = demo("policy_and_nonlinear")["runs"]
    assert runs.plot().panels[0].title == "y, shock to epi"


def test_failed_runs_are_kept_and_names_checked(nk):
    s = sweep(nk, {"phi_pi": [0.5, 1.5]}, "irf")  # 0.5: no unique stable solution (the Taylor principle fails)
    assert s.errors[0] is not None and s.errors[1] is None and s.results[0] is None
    assert "failed" in s.summary().notes[0] and np.isnan(s.table("y", "ei", horizon=0).rows[0][0])
    with pytest.raises(ValueError, match="failed"):
        s[{"phi_pi": 0.5}]
    with pytest.raises(KeyError, match="not parameters or shock sizes"):
        sweep(nk, {"phi": [1.0]})
    with pytest.raises(ValueError, match="same number of values"):
        sweep(nk, {"phi_pi": [1.2, 2.0], "kappa": [0.1]}, grid="zip")
    with pytest.raises(ValueError, match="no shocks"):
        sweep(nk.model, {"kappa": [0.1]}, "irf")


def test_runs_in_parallel_give_the_same_results(nk):
    one = sweep(nk, {"phi_pi": [1.2, 1.5, 2.0]}, "irf", periods=6)
    many = sweep(nk, {"phi_pi": [1.2, 1.5, 2.0]}, "irf", periods=6, workers=2)
    assert all(np.allclose(a.values, b.values) for a, b in zip(one.results, many.results))
    with pytest.raises(ValueError, match="other processes"):
        sweep(nk, {"phi_pi": [1.2, 1.5]}, lambda m: 1.0, workers=2)


def _fit(nk):
    truth = nk.solve().simulate(80, nk.shock_sd, seed=1)
    truth.index = pd.period_range("2000Q1", periods=80, freq="Q", name="time")
    return nk.estimate(nk.measurement.apply(Dataset.from_frame(truth)), estimate=["rho"])


def test_scenario_lists_equal_each_scenario_on_its_own(nk):
    fit = _fit(nk)
    span = ("2020Q1", "2021Q4")
    tight = Scenario(shift={"ei": {"2020Q1": 1.0}})
    c = scenarios(fit, {"baseline": Scenario(), "tight": tight}, span)
    assert np.allclose(c.frame("i")["tight"], fit.forecast(span, tight).solved().frame["i"])
    diff = c.difference("tight", "baseline", ["y", "i"]).to_frame()
    assert diff["i"].iloc[0] > 0 and diff["y"].iloc[0] < 0
    assert len(c.plot(["y", "i"]).panels) == 2
    parallel = scenarios(fit, [Scenario(name="a"), Scenario(shift={"ei": {"2020Q1": 1.0}}, name="b")], span, workers=2)
    assert parallel.names == ["a", "b"] and np.allclose(parallel.frame("i")["b"], c.frame("i")["tight"])
    both = compare_solutions(base=c["baseline"], tight=c["tight"])
    assert np.allclose(both.frame("y")["tight"], c.frame("y")["tight"])
    with pytest.raises(ValueError, match="its own name"):
        scenarios(fit, [Scenario(), Scenario()], span)


def test_the_dialogs(nk):
    from pymetrix.views import view_sections

    assert _values("1:2:0.25") == [1.0, 1.25, 1.5, 1.75, 2.0] and _values("1.2, 1.5; 2") == [1.2, 1.5, 2.0]
    s = Session()
    s.objects["nk"] = nk
    s.run(ParameterSweep("nk", "phi_pi", "1.2, 2", parameter2="kappa", values2="0.05:0.1:0.05", periods=6))
    assert "runs = sweep(nk, {'phi_pi': [1.2, 2.0], 'kappa': [0.05, 0.1]}, periods=6)" in s.script()
    assert len(s["runs"]) == 4 and [t for t, _ in view_sections(s["runs"])][0] == "Runs"
    assert ParameterSweep("nk", "phi_pi", "1:0:1").validate(s) == ["give each parameter some values"]
    assert ParameterSweep("nk", "phi_pi", "1, 2", parameter2="phi_pi", values2="1").validate(s) == ["sweep two different parameters"]
    fit = _fit(nk)
    s.objects["fit"] = fit
    s.objects["tight"] = Scenario(shift={"ei": {"2020Q1": 1.0}})
    s.objects["calm"] = Scenario()
    s.run(CompareScenarios(source="fit", scenarios="calm, tight", start="2020Q1", end="2021Q4"))
    assert "comparison = scenarios(fit, {'calm': calm, 'tight': tight}, ('2020Q1', '2021Q4'))" in s.script()
    assert "tight less calm" in [t for t, _ in view_sections(s["comparison"])]
    s.objects["a"], s.objects["b"] = s["comparison"]["calm"], s["comparison"]["tight"]
    s.run(CompareScenarios(solutions="a, b", name="ab"))
    assert "ab = compare_solutions(a=a, b=b)" in s.script()
    assert CompareScenarios(solutions="a").validate(s) == ["tick two or more solutions"]
    assert CompareScenarios(source="fit", scenarios="calm").validate(s) == ["give the periods to solve (start and end)"]


def test_a_scenario_set_compares_each_scenario_with_one_baseline(nk):
    from pymetrix import Report
    from pymetrix.views import view_sections

    fit = _fit(nk)
    span = ("2020Q1", "2021Q4")
    calm, tight, loose = Scenario(), Scenario(shift={"ei": {"2020Q1": 1.0}}), Scenario(shift={"ei": {"2020Q1": -0.5}})
    c = scenarios(fit, {"tight": tight, "baseline": calm, "loose": loose}, span)
    assert c.baseline == "baseline" and c.alternatives == ["tight", "loose"]  # one named baseline, wherever it is
    dev = c.deviations("i")
    assert list(dev.columns) == ["tight", "loose"]
    assert np.allclose(dev["tight"], c.frame("i")["tight"] - c.frame("i")["baseline"])  # by hand
    impact = c.impact(["y", "i"]).to_frame()
    assert list(impact.index) == ["tight", "loose"] and list(impact.columns) == ["y", "i"]
    assert impact.loc["tight", "i"] == pytest.approx(dev["tight"].mean())
    assert impact.loc["tight", "i"] > 0 > impact.loc["loose", "i"]  # a tighter rate path above the baseline, a looser below
    end = c.impact("i", at="end").to_frame()
    assert end.loc["tight", "i"] == pytest.approx(dev["tight"].iloc[-1])
    peak = c.impact("i", at="peak").to_frame()
    assert abs(peak.loc["tight", "i"]) == pytest.approx(dev["tight"].abs().max())
    assert c.impact("i", at="2020Q3").to_frame().loc["loose", "i"] == pytest.approx(dev.loc[pd.Period("2020Q3"), "loose"])
    with pytest.raises(LookupError, match="no period"):
        c.impact("i", at="2030Q1")
    assert [p.title for p in c.plot(["y", "i"], deviations=True).panels] == ["y less baseline", "i less baseline"]
    assert c.plot("i").panels[0].layers[0].label == "baseline"  # the baseline drawn first
    other = c.with_baseline("tight")
    assert other.alternatives == ["baseline", "loose"] and np.allclose(other.deviations("i")["baseline"], -dev["tight"])
    with pytest.raises(LookupError, match="baseline"):
        c.with_baseline("nope")
    first = scenarios(fit, [Scenario(name="a"), Scenario(shift={"ei": {"2020Q1": 1.0}}, name="b")], span)
    assert first.baseline == "a"  # no scenario named baseline: the first
    assert scenarios(fit, {"a": calm, "b": tight}, span, baseline="b").baseline == "b"
    titles = [t for t, _ in view_sections(c)]
    assert titles[:4] == ["Scenarios", "Impact", "Graph", "Less the baseline"]
    assert "tight less baseline" in titles and "loose less baseline" in titles
    brief = Report.from_template("policy_brief", {"Policy options": c})
    shown = [b.caption or getattr(b.value, "title", "") for b in brief.blocks if b.kind in ("table", "figure")]
    assert len(shown) == 4 and "Policy options" in [b.value for b in brief.blocks if b.kind == "heading"]


def test_the_compare_scenarios_dialog_records_the_baseline(nk):
    s = Session()
    fit = _fit(nk)
    s.objects["fit"] = fit
    s.objects["calm"], s.objects["tight"] = Scenario(), Scenario(shift={"ei": {"2020Q1": 1.0}})
    s.run(CompareScenarios(source="fit", scenarios="calm, tight", start="2020Q1", end="2021Q4", baseline="tight"))
    assert "comparison = scenarios(fit, {'calm': calm, 'tight': tight}, ('2020Q1', '2021Q4'), baseline='tight')" in s.script()
    assert s["comparison"].baseline == "tight"
    assert CompareScenarios(source="fit", scenarios="calm, tight", start="2020Q1", end="2021Q4", baseline="x").validate(s) == [
        "the baseline 'x' is not one of the scenarios ticked"]
    s.objects["a"], s.objects["b"] = s["comparison"]["calm"], s["comparison"]["tight"]
    s.run(CompareScenarios(solutions="a, b", baseline="b", name="ab"))
    assert "ab = compare_solutions(b=b, a=a)" in s.script() and s["ab"].baseline == "b"


def test_a_scenario_sets_loss_ranks_the_scenarios(nk):
    from pymetrix import Report
    from pymetrix.views import view_sections

    fit = _fit(nk)
    span = ("2020Q1", "2021Q4")
    c = scenarios(fit, {"baseline": Scenario(), "tight": Scenario(shift={"ei": {"2020Q1": 1.0}})}, span)
    with pytest.raises(SpecificationError, match="set the loss first"):
        c.losses()
    ranked = c.with_loss({"pi": 1, "y": 0.5}, desired={"pi": 0.5}, discount=0.99)
    table = ranked.losses().to_frame()
    for name in ("baseline", "tight"):  # by hand: the sum over the periods of 0.99^t (w (x - desired)^2)
        paths = c.frame("pi")[name].to_numpy(), c.frame("y")[name].to_numpy()
        beta = 0.99 ** np.arange(len(paths[0]))
        pi_part, y_part = np.sum(beta * (paths[0] - 0.5) ** 2), 0.5 * np.sum(beta * paths[1] ** 2)
        assert table.loc[name, "pi"] == pytest.approx(pi_part) and table.loc[name, "y"] == pytest.approx(y_part)
        assert table.loc[name, "loss"] == pytest.approx(pi_part + y_part)
    assert table.loc["baseline", "less baseline"] == 0
    assert table.loc["tight", "less baseline"] == pytest.approx(table.loc["tight", "loss"] - table.loc["baseline", "loss"])
    assert list(ranked.summary().to_frame()["loss"]) == pytest.approx(list(table["loss"]))
    assert c.loss == {} and ranked.names == c.names  # a new set: the old one unchanged
    titles = [t for t, _ in view_sections(ranked)]
    assert titles[:3] == ["Scenarios", "Impact", "Loss"] and "Loss" not in [t for t, _ in view_sections(c)]
    brief = Report.from_template("policy_brief", {"Options": ranked})
    assert any(b.kind == "table" and "Loss" in getattr(b.value, "title", "") for b in brief.blocks)
    for bad, match in (({}, "weights"), ({"pi": -1}, "zero or more"), ({"nope": 1}, "no variable")):
        with pytest.raises(LookupError if match == "no variable" else SpecificationError, match=match):
            c.with_loss(bad)
    with pytest.raises(SpecificationError, match="not in the loss"):
        c.with_loss({"pi": 1}, desired={"y": 1})
    with pytest.raises(SpecificationError, match="discount"):
        c.with_loss({"pi": 1}, discount=1.5)


def test_the_compare_scenarios_dialog_sets_the_loss(nk):
    s = Session()
    s.objects["fit"] = _fit(nk)
    s.objects["calm"], s.objects["tight"] = Scenario(), Scenario(shift={"ei": {"2020Q1": 1.0}})
    s.run(CompareScenarios(source="fit", scenarios="calm, tight", start="2020Q1", end="2021Q4", loss="pi=1, y=0.5",
                           desired="pi=0.5", discount=0.99))
    assert ("comparison = scenarios(fit, {'calm': calm, 'tight': tight}, ('2020Q1', '2021Q4'))"
            ".with_loss({'pi': 1.0, 'y': 0.5}, desired={'pi': 0.5}, discount=0.99)") in s.script()
    assert s["comparison"].loss == {"pi": 1.0, "y": 0.5} and s["comparison"].discount == 0.99
    s.objects["a"], s.objects["b"] = s["comparison"]["calm"], s["comparison"]["tight"]
    s.run(CompareScenarios(solutions="a, b", loss="pi=1", name="ab"))
    assert "ab = compare_solutions(a=a, b=b).with_loss({'pi': 1.0})" in s.script()
    assert CompareScenarios(source="fit", scenarios="calm, tight", start="2020Q1", end="2021Q4",
                            desired="pi=2").validate(s) == ["give the loss's weights (e.g. pi=1, ygap=0.5) for the desired values"]
    assert CompareScenarios(source="fit", scenarios="calm, tight", start="2020Q1", end="2021Q4", loss="pi=1",
                            desired="y=2").validate(s) == ["desired values for y, which are not in the loss"]
