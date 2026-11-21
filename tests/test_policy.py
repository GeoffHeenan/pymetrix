"""Optimal policy: checked against the closed forms for the New Keynesian model with a persistent cost-push shock
and loss pi^2 + lambda x^2 (Gali 2015, ch. 5): discretion, x = -(kappa/lambda) pi and
pi = lambda / (kappa^2 + lambda (1 - beta rho)) u; commitment, the targeting rule pi = -(lambda/kappa)(x - x[-1])
and x = delta x[-1] - kappa delta / (lambda (1 - delta beta rho)) u. Optimal simple rules against a grid search."""

import numpy as np
import pytest

from pymetrix import Equation, Model, optimal_policy, osr, templates
from pymetrix.metadata import Role
from pymetrix.models import ModelError

B, K, S, RHO, LAM = 0.99, 0.1, 1.0, 0.8, 0.25
LOSS = {"pi": 1.0, "x": LAM}


@pytest.fixture(scope="module")
def gali():
    return Model([Equation("pi ~ bet*pi[+1] + kap*x + u", params=["bet", "kap"], name="Phillips"),
                  Equation("x ~ x[+1] - sig*(i - pi[+1])", params=["sig"], name="IS"),
                  Equation("u ~ rho*u[-1] + eu", params=["rho"], name="cost push"),
                  Equation("i ~ phi*pi", params=["phi"], name="rule")],
                 params=dict(bet=B, kap=K, sig=S, rho=RHO, phi=1.5), roles={"eu": Role.SHOCK})


def test_discretion_equals_the_closed_form(gali):
    dis = optimal_policy(gali, "i", LOSS, kind="discretion", beta=B)
    irf = dis.irf(12, {"eu": 1.0})
    pi, x, u = irf["pi", "eu"].to_numpy(), irf["x", "eu"].to_numpy(), RHO ** np.arange(13)
    assert np.allclose(pi, LAM / (K**2 + LAM * (1 - B * RHO)) * u, rtol=1e-9)
    assert np.allclose(x, -K / LAM * pi, rtol=1e-9)
    assert dis.dropped == "rule"


def test_commitment_equals_the_closed_form(gali):
    com = optimal_policy(gali, "i", LOSS, kind="commitment", beta=B)
    irf = com.irf(12, {"eu": 1.0})
    pi, x, u = irf["pi", "eu"].to_numpy(), irf["x", "eu"].to_numpy(), RHO ** np.arange(13)
    x_lag = np.r_[0.0, x[:-1]]
    assert np.allclose(pi, -LAM / K * (x - x_lag), atol=1e-12)  # the targeting rule
    a = LAM / (LAM * (1 + B) + K**2)
    delta = (1 - np.sqrt(1 - 4 * B * a**2)) / (2 * a * B)
    assert np.allclose(x, delta * x_lag - K * delta / (LAM * (1 - delta * B * RHO)) * u, atol=1e-12)
    assert any(v.startswith("mult_") for v in com.solution.variables)  # history through the multipliers
    dis = optimal_policy(gali, "i", LOSS, kind="discretion", beta=B)
    assert com.loss() < dis.loss()  # the gain from commitment


def test_optimal_simple_rules_match_a_grid_search(gali):
    from pymetrix.models.policy import _rule_loss

    best = osr(gali, ["phi"], LOSS, bounds={"phi": (1.01, 50.0)})
    grid = np.linspace(1.01, 3.0, 801)
    losses = [_rule_loss(gali, {"phi": g}, LOSS) for g in grid]
    assert best.values["phi"] == pytest.approx(grid[int(np.argmin(losses))], abs=3e-3)
    assert best.loss == pytest.approx(min(losses), rel=1e-6) and best.loss <= min(losses) + 1e-9
    assert optimal_policy(gali, "i", LOSS, kind="commitment", beta=B).loss() < best.loss
    nk = templates.new_keynesian()
    two = osr(nk, ["phi_pi", "phi_y"], {"pi": 1.0, "y": 0.25, "i": 0.1}, bounds={"phi_pi": (1.01, 10), "phi_y": (0, 5)})
    assert two.template.model.parameters["phi_pi"].value == pytest.approx(two.values["phi_pi"])


def test_errors_and_the_dialogs(gali):
    with pytest.raises(ModelError, match="not an endogenous variable"):
        optimal_policy(gali, "r", LOSS)
    with pytest.raises(ModelError, match="not variables of the model"):
        optimal_policy(gali, "i", {"inflation": 1.0})
    with pytest.raises(ModelError, match="no equation named"):
        optimal_policy(gali, "i", LOSS, rule="taylor")
    with pytest.raises(ValueError, match="commitment"):
        optimal_policy(gali, "i", LOSS, kind="ramsey")

    from pymetrix.commands import LoadTemplate, OptimalSimpleRule, Session, SolveOptimalPolicy
    from pymetrix.views import view_sections

    s = Session()
    s.run(LoadTemplate("new_keynesian", name="nk"))
    s.run(SolveOptimalPolicy("nk", "i", "pi=1, y=0.25", kind="discretion"))
    assert "optimal = optimal_policy(nk, 'i', {'pi': 1.0, 'y': 0.25}, kind='discretion')" in s.script()
    assert [t for t, _ in view_sections(s["optimal"])][0] == "Summary"
    s.run(OptimalSimpleRule("nk", "phi_pi", "pi=1, y=0.25", lower="phi_pi=1.01", upper="phi_pi=10"))
    assert "best_rule = osr(nk, ['phi_pi'], {'pi': 1.0, 'y': 0.25}, bounds={'phi_pi': (1.01, 10.0)})" in s.script()
    assert SolveOptimalPolicy("nk", "i", "").validate(s) == ["give the loss weights, e.g. pi=1, y=0.25"]
