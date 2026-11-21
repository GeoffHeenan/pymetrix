"""Measurement maps from raw data and back (``transforms``), and series reported but not observed (``reported``).

Checked by hand:
- forward: estimating from raw levels (GDP, CPI) gives the same likelihood and estimates as from the growth rates
  worked out beforehand;
- reverse: the levels rebuilt beyond the data equal the last level compounded by the forecast growth rates, in the
  forecast and in every draw; a four-quarter change, x(t) = x(t-4) (1 + obs/100); a per-capita transform with the
  population given, gdp(t) = pop(t) (gdp/pop)(t-1) exp(obs/400);
- reported: the output gap with the smoothed history equals 100 times the smoothed y, and with forecasts and draws
  100 times their y; the deterministic trends of observed series reach the draws (they did not before);
- the model text: the from clause and the reported section, their problems, writing back.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, ModelSource, templates
from pymetrix.models import MeasurementMap, ModelError
from pymetrix.models.stochastic import Simulations, observables

CUT = 360  # the data end here; forecasts beyond


def _quarters(n, start="1925Q1"):
    return pd.period_range(start, periods=n, freq="Q", name="time")


@pytest.fixture(scope="module")
def setup():
    nk = templates.new_keynesian()
    truth = nk.solve().simulate(400, nk.shock_sd, seed=11)
    truth.index = _quarters(400)
    observed = nk.measurement.apply(Dataset.from_frame(truth)).frame
    growth = observed["growth"].fillna(observed["growth"].iloc[1]).to_numpy()
    raw = pd.DataFrame({"gdp": 100 * np.exp(np.cumsum(growth) / 400), "cpi": 100 * np.exp(np.cumsum(observed["infl"].to_numpy()) / 400),
                        "rate": observed["rate"].to_numpy()}, index=truth.index)
    m = nk.measurement
    raw_map = MeasurementMap(m.equations, params=dict(m.params), labels={**m.labels, "gap": "output gap, percent"},
                             transforms={"growth": "400*dlog(gdp)", "infl": "400*dlog(cpi)"}, reported=["gap = 100*y"])
    return nk, replace(nk, measurement=raw_map), Dataset.from_frame(observed.iloc[:CUT]), Dataset.from_frame(raw.iloc[:CUT]), raw


def test_estimating_from_raw_data_is_estimating_from_growth_rates(setup):
    nk, raw_nk, observed, raw, _ = setup
    a = templates.new_keynesian(rho=0.6).estimate(observed, estimate=["rho", "sd_ey"], span=("1925Q2", None))
    b = replace(templates.new_keynesian(rho=0.6), measurement=raw_nk.measurement).estimate(raw, estimate=["rho", "sd_ey"], span=("1925Q2", None))
    assert b.stats["loglike"] == pytest.approx(a.stats["loglike"], abs=1e-6)
    assert np.allclose(b.params.values, a.params.values, atol=1e-5)
    made = raw_nk.measurement.observed(raw)
    assert np.allclose(made["growth"].to_numpy()[1:], observed.frame["growth"].to_numpy()[1:], atol=1e-9)
    assert raw_nk.measurement.data_series() == ("cpi", "rate", "gdp")  # in the order of the measurement equations


def test_forecasts_report_raw_levels_and_the_output_gap(setup):
    _, raw_nk, _, raw, _ = setup
    fit = raw_nk.estimate(raw, estimate=["rho"], span=("1925Q2", None))
    span = ("2015Q1", "2017Q4")
    fc = fit.forecast(span)
    f = fc.frame
    periods = pd.period_range(*span, freq="Q")
    last = raw.frame["gdp"].iloc[-1]
    expected = last * np.exp(np.cumsum(f.loc[periods, "growth"].to_numpy()) / 400)  # compounded by hand
    assert np.allclose(f.loc[periods, "gdp"].to_numpy(), expected, rtol=1e-12)
    assert np.allclose(f.loc[periods, "gap"].to_numpy(), 100 * f.loc[periods, "y"].to_numpy(), atol=1e-12)
    # the smoothed history carries the gap too
    smoothed = fit.smoothed.frame
    assert np.allclose(smoothed["gap"].to_numpy(), 100 * smoothed["y"].to_numpy(), atol=1e-12)
    # every draw: its own growth compounded into its own level
    drawn = fit.forecast(span, draws=40, seed=3).simulation
    g, lv, gap, y = (drawn.values[:, :, drawn.names.index(n)] for n in ("growth", "gdp", "gap", "y"))
    assert np.allclose(lv, last * np.exp(np.cumsum(g, axis=1) / 400), rtol=1e-12)
    assert np.allclose(gap, 100 * y, atol=1e-12) and np.std(lv[:, -1]) > 0


def test_other_transforms_rebuilt_by_hand():
    idx = _quarters(12, "2000Q1")
    rng = np.random.default_rng(2)
    x = 100 + np.cumsum(rng.normal(1, 1, 12))
    pop = np.linspace(50, 52, 12)
    data = Dataset.from_frame(pd.DataFrame({"x": x[:8], "pop": pop[:8]}, index=idx[:8]))
    later = pd.DataFrame({"yoy": rng.normal(2, 1, 4)}, index=idx[8:])
    four = MeasurementMap(["yoy = 4*q"], transforms={"yoy": "100*(x/x[-4] - 1)"})
    back = four.reverse(later, data)["x"].to_numpy()
    hand = list(x[:8])
    for v in later["yoy"]:
        hand.append(hand[-4] * (1 + v / 100))
    assert np.allclose(back, hand[8:], rtol=1e-12)
    per_head = MeasurementMap(["g = 4*q"], transforms={"g": "400*dlog(gdp/pop)"}, sources={"g": "gdp"})
    with_pop = Dataset.from_frame(pd.DataFrame({"gdp": np.r_[x[:8], [np.nan] * 4], "pop": pop}, index=idx))
    obs = pd.DataFrame({"g": [1.0, 2.0, -1.0, 0.5]}, index=idx[8:])
    rebuilt = per_head.reverse(obs, with_pop)["gdp"].to_numpy()
    level, hand = x[7] / pop[7], []
    for t, v in zip(range(8, 12), obs["g"]):
        level *= np.exp(v / 400)
        hand.append(level * pop[t])
    assert np.allclose(rebuilt, hand, rtol=1e-12)


def test_trends_reach_the_draws():
    m = MeasurementMap(["infl = 4*pi"], trends={"infl": "0.5"})
    idx = _quarters(6, "2000Q1")
    path = Dataset.from_frame(pd.DataFrame({"pi": np.linspace(0, 1, 6)}, index=idx))
    sims = Simulations(path.frame.loc[idx[2:], ["pi"]].to_numpy(float)[None], ("pi",), idx[2:])
    drawn = observables(m, path, sims)
    assert np.allclose(drawn.values[0, :, drawn.names.index("infl")], m.apply(path).frame["infl"].to_numpy()[2:], atol=1e-12)


def test_what_is_refused():
    with pytest.raises(ModelError, match="name the one it is rebuilt into"):
        MeasurementMap(["g = 4*q"], transforms={"g": "400*dlog(gdp/pop)"})
    with pytest.raises(ModelError, match="does not move with pop"):
        MeasurementMap(["g = 4*q"], transforms={"g": "gdp + 0*pop[-1] + pop[-1]"}, sources={"g": "pop"})
    with pytest.raises(ModelError, match="present and past"):
        MeasurementMap(["g = 4*q"], transforms={"g": "x[+1] - x"})
    with pytest.raises(ModelError, match="names of their own"):
        MeasurementMap(["g = 4*q"], reported=["g = q"])
    with pytest.raises(ModelError, match="not y\\[1\\]|not y\\[\\+1\\]"):
        MeasurementMap(["g = 4*q"], reported=["gap = y[+1]"])
    with pytest.raises(ModelError, match="errors given for \\['gap'\\]"):
        MeasurementMap(["g = 4*q"], reported=["gap = q"], errors={"gap": 1.0})
    with pytest.raises(ModelError, match="transforms given for"):
        MeasurementMap(["g = 4*q"], transforms={"h": "dlog(x)"})
    m = MeasurementMap(["g = 4*q"], reported=["gap = 100*q"], labels={"gap": "gap"}, transforms={"g": "400*dlog(x)"})
    assert "reported gap" in repr(m) and m.table().index == ["g", "gap"]


TEXT = '''parameters
    rho = 0.8
    g = 2
shocks
    e = 1
equations
    y = rho*y[-1] + e
measurement
    growth = g + 4*(y - y[-1])  from 400*dlog(gdp)  error 0.5  "GDP growth"
reported
    gap = 100*y  "output gap"
'''


def test_the_model_text_and_the_dialog(tmp_path):
    tpl = ModelSource(TEXT).build()
    m = tpl.measurement
    assert m.transforms == {"growth": "400*dlog(gdp)"} and m.reported == ("gap = 100*y",) and m.labels["gap"] == "output gap"
    assert ModelSource(TEXT).export("pymetrix").text.split("\n", 1)[1] == TEXT
    check = ModelSource(TEXT).check()
    assert check.ok and ("gap", "reported") in {(s.name, s.role) for s in check.symbols}
    bad = ModelSource(TEXT.replace("gap = 100*y", "gap = 100*z")).check()
    assert bad.errors[0].line == 11 and "z in a reported series" in bad.errors[0].message
    lead = ModelSource(TEXT.replace("gap = 100*y", "gap = y[+1]")).check()
    assert "not y[1]" in lead.errors[0].message or "not y[+1]" in lead.errors[0].message
    dyn = ModelSource(TEXT).export("dynare")
    assert any("transforms" in n for n in dyn.notes) and any("reported series (gap)" in n for n in dyn.notes)

    from pymetrix.commands import Session, StateSpaceForm
    from pymetrix.testing import check_project

    s = Session()
    s.run_code("from pymetrix import templates\nnk = templates.new_keynesian()")
    command = StateSpaceForm("nk", measurement="growth = 2 + 4*(y - y[-1])\nrate = 3 + 4*i", transforms="growth = 400*dlog(gdp)",
                             reported="gap = 100*y\nreal = 4*(i - pi)")
    assert command.validate(s) == []
    assert StateSpaceForm("nk", reported="gap = y").validate(s) == ["reported series and transforms go with measurement equations: write them too"]
    assert "transforms={'growth': '400*dlog(gdp)'}" in command.to_code() and "reported=['gap = 100*y', 'real = 4*(i - pi)']" in command.to_code()
    check_project(s, command, tmp_path)
    system = s["system"]
    assert system.measurement.transforms == {"growth": "400*dlog(gdp)"} and len(system.measurement.reports) == 2
