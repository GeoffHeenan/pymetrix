"""Multiplicative growth (variables in levels written in logs) and perfect foresight with a growth terminal condition.

Checked against the Solow model's balanced growth path (growth log(1 + g), capital-output ratio
s / (1 - (1 - delta) / (1 + g))), and a gap model on a growing trend whose path after a shock is known in closed form."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model
from pymetrix.models import ModelError, PerfectForesight

S, DELTA, ALPHA, G = 0.2, 0.05, 0.3, 0.02


@pytest.fixture(scope="module")
def solow():
    return Model([Equation("A ~ (1 + g)*A[-1]", params=["g"], name="technology"),
                  Equation("y ~ A^(1-alpha) * k[-1]^alpha", params=["alpha"], name="production"),
                  Equation("k ~ s*y + (1 - delta)*k[-1]", params=["s", "delta"], name="capital")],
                 params={"g": G, "alpha": ALPHA, "s": S, "delta": DELTA})


def test_levels_growing_by_a_factor_are_logs_growing_by_its_log(solow):
    logs = solow.in_logs()
    assert "exp(k[-1])" in str(logs.equations[2]) and logs.equations[2].name == "capital"
    ss = logs.steady_state(growth=True, guess={"A": 0.0, "y": 0.5, "k": 1.5})
    for v in ("A", "y", "k"):
        assert ss.growth[v] == pytest.approx(np.log(1 + G), abs=1e-12), v
    assert np.exp(ss["k"] - ss["y"]) == pytest.approx(S / (1 - (1 - DELTA) / (1 + G)), rel=1e-10)
    pinned = logs.steady_state(growth=True, fix={"A": 0.0}, guess={"y": 0.5, "k": 1.5})
    assert pinned["A"] == 0.0 and np.exp(pinned["k"] - pinned["y"]) == pytest.approx(S / (1 - (1 - DELTA) / (1 + G)), rel=1e-12)
    only_k = solow.in_logs("k")
    assert "exp(k" in str(only_k.equations[2]) and "exp(y" not in str(only_k.equations[1])
    with pytest.raises(ModelError, match="not endogenous"):
        solow.in_logs("q")


def test_perfect_foresight_ends_on_the_balanced_growth_path():
    g, a, b = 0.5, 0.5, 0.3  # gap: z = a z[+1] + b z[-1] + e; trend grows by g
    model = Model([Equation("ystar ~ ystar[-1] + g", params=["g"], name="trend"),
                   Equation("y ~ ystar + a*(y[+1] - ystar - g) + b*(y[-1] - ystar[-1]) + e", params=["a", "b", "g"], name="gap")],
                  params={"g": g, "a": a, "b": b})
    n = 60
    idx = pd.period_range("2000Q1", periods=n + 1, freq="Q", name="time")
    trend = 10.0 + g * np.arange(n + 1)
    e = np.zeros(n + 1)
    e[1] = 1.0
    data = Dataset.from_frame(pd.DataFrame({"ystar": trend, "y": trend, "e": e}, index=idx))
    span = (idx[1], idx[-1])
    path = model.solve(data, span, solver=PerfectForesight(terminal="growth")).frame
    stable = (1 - np.sqrt(1 - 4 * a * b)) / (2 * a)  # the stable root of a L^-1 - 1 + b L
    z1 = 1.0 / (1 - a * stable)
    expected_gap = z1 * stable ** np.arange(n)
    gap = (path["y"] - path["ystar"]).to_numpy()[1:]
    assert np.allclose(path["ystar"].to_numpy(), trend, atol=1e-9)  # the trend keeps growing to the end
    assert np.allclose(gap, expected_gap, atol=1e-9)
    held = model.solve(data, span, solver=PerfectForesight(terminal="constant")).frame
    assert abs((held["y"] - held["ystar"]).to_numpy()[-1]) > 1e-3  # holding levels constant bends the end of the path
