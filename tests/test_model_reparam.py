"""Estimation evaluates the model at each trial value through ``Model.with_parameters``, a copy that shares the
compiled equations and the linearisation's structure. Reference: the model built again from its equations at
those values, as estimation did before; and SciPy for the closed-form prior densities."""

import pickle

import numpy as np
import pytest
from scipy import stats

from pymetrix import Model, templates
from pymetrix.estimation.priors import Beta, Gamma, InvGamma, Normal
from pymetrix.models.fast import functions
from pymetrix.models.solvers import ModelError

from test_solver_reduction import _random_model


def _rebuilt(model, values):
    return Model(model.equations, params={**model.params, **values}, roles=model.roles)


def _solve(model, shocks=None):
    ss = model.steady_state()
    return ss, model.linearize(ss, shocks=shocks).solve()


@pytest.mark.parametrize("make", [lambda: templates.new_keynesian(), lambda: templates.rbc(), lambda: _random_model(3, 4)])
def test_a_copy_at_other_values_equals_the_model_built_again(make):
    t = make()
    model, shocks = getattr(t, "model", t), getattr(t, "shocks", None)
    _solve(model, shocks)  # the structure is worked out here, and shared by the copies
    rng = np.random.default_rng(0)
    for _ in range(4):
        values = {n: v * rng.uniform(0.9, 1.1) for n, v in model.params.items()}
        copy, again = model.with_parameters(values), _rebuilt(model, values)
        try:
            ss2, s2 = _solve(again, shocks)
        except ModelError:  # values with no stable solution: the copy must say so too
            with pytest.raises(ModelError):
                _solve(copy, shocks)
            continue
        ss1, s1 = _solve(copy, shocks)
        assert copy.endogenous == again.endogenous and copy.exogenous == again.exogenous
        for v in again.endogenous:
            assert ss1[v] == pytest.approx(ss2[v], abs=1e-12)
        np.testing.assert_allclose(s1.P, s2.P, atol=1e-12)
        np.testing.assert_allclose(s1.Q, s2.Q, atol=1e-12)
        assert [str(e.residual) for e in copy.model_equations] == [str(e.residual) for e in again.model_equations]


def test_copies_share_the_compiled_equations_and_leave_the_model_alone():
    model = _random_model(2, 5)
    before = dict(model.params)
    copy = model.with_parameters({"a0": 0.3})
    assert copy.params == {**before, "a0": 0.3} and model.params == before
    assert copy._system.parameters["a0"].value == 0.3 and model._system.parameters["a0"].value == before["a0"]
    if functions(model) is not None:
        assert functions(copy) is functions(model)


def test_the_closed_form_prior_densities_equal_scipy():
    rng = np.random.default_rng(1)
    for _ in range(500):
        m, s, x = rng.uniform(0.2, 0.8), rng.uniform(0.02, 0.2), rng.uniform(0.01, 0.99)
        c = m * (1 - m) / s**2 - 1
        assert Beta(m, s).logpdf(x) == pytest.approx(stats.beta.logpdf(x, m * c, (1 - m) * c), abs=1e-10)
        g, gs, y = rng.uniform(0.1, 3), rng.uniform(0.05, 2), rng.uniform(0.01, 5)
        assert Gamma(g, gs).logpdf(y) == pytest.approx(stats.gamma.logpdf(y, (g / gs) ** 2, scale=gs**2 / g), abs=1e-10)
        a = (g / gs) ** 2 + 2
        assert InvGamma(g, gs).logpdf(y) == pytest.approx(stats.invgamma.logpdf(y, a, scale=g * (a - 1)), abs=1e-10)
        assert Normal(g, gs).logpdf(y) == pytest.approx(stats.norm.logpdf(y, g, gs), abs=1e-10)
    for x in (0.0, 1.0, -0.5, 1.5):  # outside the support: SciPy's values
        assert Beta(0.5, 0.1).logpdf(x) == stats.beta.logpdf(x, *_beta_ab(0.5, 0.1))
    assert Gamma(1, 0.5).logpdf(-1) == -np.inf and InvGamma(1, 0.5).logpdf(0) == -np.inf


def _beta_ab(m, s):
    c = m * (1 - m) / s**2 - 1
    return m * c, (1 - m) * c


def test_a_model_and_its_copies_pickle_after_solving():
    """Sweeps and parallel chains send models to other processes: the shared compiled code stays behind."""
    model = _random_model(2, 6)
    ss, s1 = _solve(model)
    copy = model.with_parameters({"a0": 0.3})
    _solve(copy)
    back = pickle.loads(pickle.dumps(copy))
    assert "_shared" not in back.__dict__ and back.params["a0"] == 0.3
    np.testing.assert_allclose(_solve(back)[1].P, _solve(copy)[1].P, atol=1e-12)
    np.testing.assert_allclose(_solve(pickle.loads(pickle.dumps(model)))[1].P, s1.P, atol=1e-12)


def test_one_newton_step_solves_a_linear_steady_state(monkeypatch):
    """A linear model's steady state comes from one exact Newton step, without the root search; a nonlinear one still
    goes to the search. Both equal the search's answer."""
    import pymetrix.models.linear as linear
    from pymetrix import templates

    calls = []
    real_root = linear.optimize.root
    monkeypatch.setattr(linear.optimize, "root", lambda *a, **k: calls.append(1) or real_root(*a, **k))
    nk = templates.new_keynesian().model  # linear
    ss = nk.steady_state()
    assert calls == []
    rbc = templates.rbc().model  # nonlinear
    ss_rbc = rbc.steady_state()
    assert calls  # one step from the starting values does not solve it
    monkeypatch.setattr(linear, "_newton_step", lambda *a: None)
    for model, found in ((nk, ss), (rbc, ss_rbc)):
        searched = model.steady_state()
        for v in model.endogenous:
            assert found[v] == pytest.approx(searched[v], abs=1e-10)
