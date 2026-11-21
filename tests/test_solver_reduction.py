"""The first-order solver works on Dynare's reduced pencil (static variables taken out by a QR, the pencil on the
predetermined and forward-looking variables only). Reference: QZ on the full companion pencil of every variable,
written out here: the same decision rules, shock impacts and finite roots."""

import numpy as np
import pytest
from scipy import linalg

from pymetrix import Equation, Model, templates
from pymetrix.metadata import Role
from pymetrix.models.linear import _UNIT, SteadyState, _solve, _warm_solve
from pymetrix.models.solvers import ModelError


def _full_pencil(lin):
    """QZ on [x(t-1); x(t)] for every variable: the textbook companion form, size 2n."""
    n = len(lin.variables)
    F = np.block([[np.eye(n), np.zeros((n, n))], [np.zeros((n, n)), lin.A0]])
    G = np.block([[np.zeros((n, n)), np.eye(n)], [-lin.A2, -lin.A1]])
    AA, BB, alpha, beta, Q, Z = linalg.ordqz(G, F, sort=lambda a, b: np.abs(a) < np.abs(b) * (1 + _UNIT), output="real")
    Z11, Z21 = Z[:n, :n], Z[n:, :n]
    # stable roots: one per variable that is lagged; the rest of the n are zero roots of variables that are not
    P = Z21 @ np.linalg.pinv(Z11)
    Qs = -np.linalg.solve(lin.A0 @ P + lin.A1, lin.B)
    with np.errstate(divide="ignore", invalid="ignore"):
        roots = np.where(np.abs(beta) > 1e-12, alpha / beta, np.inf)
    return P, Qs, roots


def _random_model(blocks: int, seed: int) -> Model:
    """Blocks of a forward-looking gap, a backward trend, a policy rule with a two-period lead, and identities:
    some static, one led only. Random coefficients, so no roots repeat."""
    rng = np.random.default_rng(seed)
    eqs, params, roles = [], {}, {}
    for i in range(blocks):
        a, b, c, d = rng.uniform(0.2, 0.45, 4)
        prev = f"x{i - 1}" if i else "x0"
        eqs += [Equation(f"x{i} ~ a{i}*x{i}[-1] + b{i}*x{i}[+1] + 0.1*{prev} - 0.2*r{i} + ex{i}", params=[f"a{i}", f"b{i}"]),
                Equation(f"g{i} ~ c{i}*g{i}[-1] + 0.05*x{i} + eg{i}", params=[f"c{i}"]),
                Equation(f"r{i} ~ d{i}*r{i}[-1] + (1 - d{i})*(1.5*x{i}[+2] + 0.5*y{i}) + er{i}", params=[f"d{i}"]),
                Equation(f"y{i} = x{i} + 0.5*g{i}"),
                Equation(f"w{i} = y{i} - 0.3*r{i} + 0.1*y{i}[+1]"),
                Equation(f"q{i} = 2*w{i} - y{i}")]
        params.update({f"a{i}": a, f"b{i}": b, f"c{i}": c + 0.5, f"d{i}": d + 0.4})
        roles.update({f"ex{i}": Role.SHOCK, f"eg{i}": Role.SHOCK, f"er{i}": Role.SHOCK})
    return Model(eqs, params=params, roles=roles)


def _finite(roots):
    r = np.abs(roots[np.isfinite(roots)])
    return np.sort(r[r > 1e-9])  # the full pencil has zero roots for variables that are not lagged


@pytest.mark.parametrize("make", [lambda: templates.new_keynesian(), lambda: templates.rbc(), lambda: templates.gap_model(),
                                  lambda: _random_model(3, 1), lambda: _random_model(8, 2)])
def test_the_reduced_pencil_gives_the_full_pencil_solution(make):
    t = make()
    model = getattr(t, "model", t)
    lin = model.linearize(model.steady_state(), shocks=getattr(t, "shocks", None))
    sol = _solve(lin)
    P, Q, roots = _full_pencil(lin)
    np.testing.assert_allclose(sol.P, P, atol=1e-10)
    np.testing.assert_allclose(sol.Q, Q, atol=1e-10)
    np.testing.assert_allclose(_finite(sol.eigenvalues), _finite(roots), rtol=1e-8)


def test_static_variables_must_be_determined():
    model = Model([Equation("x = 0.5*x[-1] + 0.3*x[+1] + e"), Equation("y = x + z"), Equation("z = y - x")],
                  roles={"e": Role.SHOCK})  # the second and third say the same thing: y and z are not determined
    at = SteadyState({"x": 0.0, "y": 0.0, "z": 0.0, "e": 0.0}, model=model)
    with pytest.raises(ModelError, match="not determined"):
        model.linearize(at, shocks=["e"]).solve()


def test_the_likelihood_on_the_states_it_needs_equals_the_full_one():
    import pandas as pd

    from pymetrix import Dataset
    from pymetrix.models import MeasurementMap
    from pymetrix.statespace import loglike

    model = _random_model(6, 3)
    sd = {s: 0.1 for s in model.compile().shocks}
    sol = model.linearize(model.steady_state()).solve()
    paths = sol.simulate(80, sd, seed=2)
    paths.index = pd.period_range("1990Q1", periods=80, freq="Q", name="time")
    m = MeasurementMap(["ox = x0", "oy = y3", "oq = q5 + q5[-1]", "og = g2"], errors={"ox": 0.05})
    data = m.apply(Dataset.from_frame(paths))
    system = sol.system(m, sd)
    assert len(system.needed().states) < len(system.states)  # static and led-only variables are not needed
    y, index = system.observations(data)
    assert system.loglike(data) == pytest.approx(loglike(y, system, "auto"), abs=1e-7)
    with pytest.raises(ModelError, match="not observed series"):
        m.with_errors({"nothing": 1.0})


def test_a_warm_start_gives_the_schur_solution_or_hands_over():
    """From a nearby solution, the fixed-point iteration agrees with QZ; where QZ finds no unique stable solution (here
    a Taylor rule too weak, or a Phillips curve so steep that the model explodes) the warm start hands over, and the
    same error is raised."""
    nk = templates.new_keynesian()
    start = nk.model.linearize(nk.model.steady_state(), shocks=nk.shocks).solve()
    model = _random_model(4, 7)
    random_start = model.linearize(model.steady_state()).solve()
    rng = np.random.default_rng(1)
    cases = [(nk.model, {"phi_pi": v, "kappa": k}, nk.shocks, start) for v in np.linspace(0.5, 3, 15) for k in (0.02, 0.5)]
    cases += [(model, {"a0": rng.uniform(0.1, 0.9), "b0": rng.uniform(0.1, 1.2), "d1": rng.uniform(0.3, 0.99)}, None, random_start)
              for _ in range(20)]
    warm_used = 0
    for base, values, shocks, begin in cases:
        m = Model(base.equations, params=base.parameters.with_values(values), roles=base.roles)
        lin = m.linearize(m.steady_state(), shocks=shocks)
        try:
            reference = lin.solve()
        except ModelError as exc:
            with pytest.raises(ModelError, match=str(exc)[:30]):
                lin.solve(start=begin)
            continue
        warm = _warm_solve(lin, begin)
        warm_used += warm is not None
        got = lin.solve(start=begin)
        np.testing.assert_allclose(got.P, reference.P, atol=1e-10)
        np.testing.assert_allclose(got.Q, reference.Q, atol=1e-10)
    assert warm_used >= 5  # the iteration did the work in a good share of the cases


def test_chains_give_the_same_draws_in_this_process_or_others():
    import pandas as pd

    from pymetrix import Dataset
    from pymetrix.estimation.priors import Beta, Gamma

    nk = templates.new_keynesian()
    truth = nk.solve().simulate(80, nk.shock_sd, seed=3)
    truth.index = pd.period_range("2000Q1", periods=80, freq="Q", name="time")
    fit = nk.estimate(nk.measurement.apply(Dataset.from_frame(truth)), estimate=["kappa", "rho"],
                      priors={"kappa": Gamma(0.15, 0.05), "rho": Beta(0.7, 0.1)})
    here = fit.mcmc(draws=60, chains=2, seed=4, workers=1)
    there = fit.mcmc(draws=60, chains=2, seed=4, workers=2)
    np.testing.assert_array_equal(here.draws, there.draws)
    np.testing.assert_array_equal(here.acceptance, there.acceptance)
    assert not np.array_equal(here.draws[0], here.draws[1])  # each chain its own random stream
