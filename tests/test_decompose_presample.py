"""Shock decompositions from a pre-sample state (LinearSolution.decompose(levels, start="presample")).

On a path that follows the model: the parts add up to the path in both modes; with start="presample" the first
decomposed period's shocks are the shocks' (the initial condition is P x(-1)); the two modes' parts converge as P^t,
since they differ only by where period 0's shocks are counted."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import templates
from pymetrix.errors import SpecificationError


@pytest.fixture(scope="module")
def setup():
    nk = templates.new_keynesian()
    sol = nk.solve()
    path = sol.simulate(80, nk.shock_sd, seed=4)
    path.index = pd.period_range("2000Q1", periods=80, freq="Q", name="time")
    return sol, path


def test_both_modes_add_up(setup):
    sol, path = setup
    for start in ("first", "presample"):
        d = sol.decompose(path, start=start)
        assert np.allclose(d.residual.to_numpy(), 0, atol=1e-10), start
    first, pre = sol.decompose(path), sol.decompose(path, start="presample")
    assert pre.deviations.index[0] == first.deviations.index[1]  # the presample period is not decomposed
    k = len(sol.endogenous)
    x_pre = (path.iloc[0][list(sol.endogenous)] - pd.Series(dict(sol.model.steady.values))[list(sol.endogenous)]).to_numpy()
    for i, v in enumerate(sol.endogenous):  # the initial condition one period on is P x(-1); the shocks get the rest
        assert pre.contributions[v]["initial conditions"].iloc[0] == pytest.approx((sol.P[:k, :k] @ x_pre)[i], abs=1e-10)
    assert np.abs(pre.shocks.iloc[0]).max() > 0 and np.all(first.shocks.iloc[0] == 0)


def test_the_modes_converge(setup):
    sol, path = setup
    first, pre = sol.decompose(path), sol.decompose(path, start="presample")
    gaps = []
    for v in sol.endogenous:
        a = first.contributions[v].loc[pre.deviations.index]
        gaps.append(np.abs(a.to_numpy() - pre.contributions[v].to_numpy()).max(axis=1))
    gap = np.max(gaps, axis=0)
    rho = float(np.max(np.abs(np.linalg.eigvals(sol.P))))
    assert gap[-1] < 1e-6 * max(gap[0], 1e-12) + 1e-12 and gap[-1] <= gap[0]
    assert np.all(gap[10:] <= gap[0] * (rho + 1e-3) ** np.arange(10, len(gap)) * 10 + 1e-12)  # geometric, as P^t


def test_refusals(setup):
    sol, path = setup
    with pytest.raises(SpecificationError, match="start is 'first'"):
        sol.decompose(path, start="previous")
    with pytest.raises(SpecificationError, match="pre-sample period and at least one more"):
        sol.decompose(path.iloc[:1], start="presample")
