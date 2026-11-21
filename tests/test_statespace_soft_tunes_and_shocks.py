"""Soft filter tunes and smoothed shocks on a model's state-space form (StateSpaceSystem).

Soft tunes: a tune with a standard deviation is an observation with that error, seen in its own period only. A tiny
standard deviation pins the smoothed value; a large one barely moves it; None or 0 gives exactly today's (exact)
result; the pair form and tune_sd= agree; contributions still add up to the smoothed path.

Smoothed shocks: filter(shocks=True) adds shocks that satisfy x(t) - P x(t-1) = Q e(t) to rounding error on the smoothed path, on
both engines, and equal the smoothed states of the same model with the shocks carried as states (the augmented
system), e(0) included."""

import numpy as np
import pandas as pd
import pytest

import pymetrix.statespace as ss
from pymetrix import Dataset, templates
from pymetrix.errors import SpecificationError


@pytest.fixture(scope="module")
def setup():
    nk = templates.new_keynesian()
    sol = nk.solve()
    truth = sol.simulate(120, nk.shock_sd, seed=3)
    truth.index = pd.period_range("1990Q1", periods=120, freq="Q", name="time")
    data = nk.measurement.apply(Dataset.from_frame(truth))
    system = sol.system(nk.measurement, nk.shock_sd, measurement_sd={o: 0.3 for o in nk.measurement.observables})
    return sol, system, data


def test_soft_tunes(setup):
    _, system, data = setup
    mid = str(data.frame.index[60])
    plain = system.filter(data)
    target = plain["y"].loc[mid] + 2.0
    exact = system.filter(data, tunes={"y": {mid: target}})
    assert exact["y"].loc[mid] == pytest.approx(target, abs=1e-10)
    for sd in (None, 0, 0.0):  # exact, as today, whichever way it is written
        np.testing.assert_array_equal(system.filter(data, tunes={"y": {mid: (target, sd)}}).frame.to_numpy(), exact.frame.to_numpy())
    np.testing.assert_array_equal(system.filter(data, tunes={"y": {mid: target}}, tune_sd={"y": None}).frame.to_numpy(), exact.frame.to_numpy())
    tiny = system.filter(data, tunes={"y": {mid: (target, 1e-6)}})
    assert tiny["y"].loc[mid] == pytest.approx(target, abs=1e-8)
    big = system.filter(data, tunes={"y": {mid: (target, 1e3)}})
    assert abs(big["y"].loc[mid] - plain["y"].loc[mid]) < 1e-5  # a 2.0 push with an sd of 1000 barely registers
    moves = [abs(system.filter(data, tunes={"y": {mid: target}}, tune_sd={"y": {mid: sd}})["y"].loc[mid] - plain["y"].loc[mid])
             for sd in (2.0, 0.5, 0.1)]
    assert moves[0] < moves[1] < moves[2] < 2.0  # partly met, more so as the sd shrinks
    same = system.filter(data, tunes={"y": {mid: target}}, tune_sd={"y": 0.5})
    np.testing.assert_allclose(same.frame.to_numpy(), system.filter(data, tunes={"y": {mid: (target, 0.5)}}).frame.to_numpy(), atol=1e-12)


def test_soft_tunes_in_contributions_and_mixed(setup):
    _, system, data = setup
    p1, p2 = str(data.frame.index[50]), str(data.frame.index[90])
    tunes = {"y": {p1: (1.0, 0.2), p2: 0.5}, "pi": {p2: (0.3, 0.1)}}  # soft, exact and soft
    filtered = system.filter(data, tunes=tunes)
    decomposition = system.contributions(data, tunes=tunes)
    steady = system.levels["y"]
    np.testing.assert_allclose(decomposition.deviations["y"].to_numpy(), filtered["y"].to_numpy() - steady, atol=1e-8)
    assert "y (tune)" in decomposition.contributions["y"].columns and "pi (tune)" in decomposition.contributions["y"].columns
    assert filtered["y"].loc[p2] == pytest.approx(0.5, abs=1e-8)  # the exact one among soft ones still holds
    with pytest.raises(SpecificationError, match="zero or positive"):
        system.filter(data, tunes={"y": {p1: (1.0, -1.0)}})
    with pytest.raises(SpecificationError, match="have no tunes"):
        system.filter(data, tunes={"y": {p1: 1.0}}, tune_sd={"pi": 0.1})
    with pytest.raises(SpecificationError, match="have no tune"):
        system.filter(data, tunes={"y": {p1: 1.0}}, tune_sd={"y": {p2: 0.1}})


@pytest.mark.parametrize("engine", ["statsmodels", "native"])
def test_smoothed_shocks(setup, engine, monkeypatch):
    sol, system, data = setup
    monkeypatch.setattr(ss, "ENGINE", engine)
    plain = system.filter(data)
    smoothed = system.filter(data, shocks=True)
    shocks = list(sol.shocks)
    assert list(smoothed.frame.columns) == [*plain.frame.columns, *shocks]
    np.testing.assert_array_equal(smoothed.frame[list(plain.frame.columns)].to_numpy(), plain.frame.to_numpy())  # nothing else moves
    k = len(sol.variables)
    x = smoothed.frame[list(sol.variables)].to_numpy() - np.array([system.levels.get(v, 0.0) for v in sol.variables])
    e = smoothed.frame[shocks].to_numpy()
    assert len(e) == len(data.frame)
    np.testing.assert_allclose(x[1:] - x[:-1] @ sol.P.T, e[1:] @ sol.Q.T, atol=1e-12)  # x(t) - P x(t-1) = Q e(t)
    # the same model with the shocks as states: s(t) = [x(t); e(t)]
    n, m = len(system.states), len(sol.shocks)
    T = np.block([[system.T, np.zeros((n, m))], [np.zeros((m, n + m))]])
    R = np.vstack([system.R, np.eye(m)])
    Z = np.hstack([system.Z, np.zeros((len(system.observables), m))])
    augmented = ss.SystemMatrices((*system.states, *system.shocks), system.observables, system.shocks, T, np.zeros(n + m), R,
                                  system.Q, Z, system.d, system.H, tuple(True for _ in range(n + m)))
    y, _ = system.observations(data)
    out = ss.run(y, augmented, "auto")
    assert out.engine == engine
    np.testing.assert_allclose(e, out.a_smooth[:, n:], atol=1e-9)  # e(0) from the start, the rest from the smoother
    np.testing.assert_allclose(x, out.a_smooth[:, :k], atol=1e-9)


def test_smoothed_shocks_with_tunes(setup):
    """With tunes, the shocks are those of the tuned path: the identity holds on it, exact and soft tunes alike."""
    sol, system, data = setup
    mid = str(data.frame.index[60])
    for tune in (1.0, (1.0, 0.2)):
        smoothed = system.filter(data, tunes={"y": {mid: tune}}, shocks=True)
        x = smoothed.frame[list(sol.variables)].to_numpy() - np.array([system.levels.get(v, 0.0) for v in sol.variables])
        e = smoothed.frame[list(sol.shocks)].to_numpy()
        np.testing.assert_allclose(x[1:] - x[:-1] @ sol.P.T, e[1:] @ sol.Q.T, atol=1e-10)
        np.testing.assert_array_equal(smoothed.frame[list(sol.variables)].to_numpy(),
                                      system.filter(data, tunes={"y": {mid: tune}}).frame[list(sol.variables)].to_numpy())


def test_native_disturbance_smoother_identity(setup, monkeypatch):
    """The disturbance smoother itself, R eta(t|n) = a(t+1|n) - c - T a(t|n), wherever R has fewer columns than rows."""
    _, system, data = setup
    monkeypatch.setattr(ss, "ENGINE", "native")
    y, _ = system.observations(data)
    out = ss.run(y, system, "auto")
    gap = out.a_smooth[1:] - system.c - out.a_smooth[:-1] @ system.T.T
    np.testing.assert_allclose(out.eta_smooth[:-1] @ system.R.T, gap, atol=1e-12)
    assert np.all(out.eta_smooth[-1] == 0)


def test_list_pairs(setup):
    """A saved call gives a (value, sd) pair back as a list: it means the same."""
    _, system, data = setup
    mid = str(data.frame.index[60])
    as_tuple = system.filter(data, tunes={"y": {mid: (1.0, 0.2)}})
    as_list = system.filter(data, tunes={"y": {mid: [1.0, 0.2]}})
    np.testing.assert_array_equal(as_tuple.frame.to_numpy(), as_list.frame.to_numpy())
    with pytest.raises(SpecificationError, match="pair"):
        system.filter(data, tunes={"y": {mid: (1.0, 0.2, 3.0)}})
