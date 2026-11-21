"""The Blanchard-Kahn counts on a solution: n_stable (unit roots included, as the solver counts them) and n_predetermined
(variables that appear lagged), equal for every solution the solver returns, and stated in the roots table."""

import numpy as np
import pytest

from pymetrix import Equation, Model, templates
from pymetrix.metadata import Role
from pymetrix.models import ModelError
from pymetrix.models.linear import SteadyState


def test_new_keynesian():
    sol = templates.new_keynesian().solve()
    assert sol.n_stable == sol.n_predetermined == 1
    assert sol.n_stable == int(np.sum(np.abs(sol.eigenvalues) < 1))
    table = str(sol.roots())
    assert "stable roots 1 (unit roots included), predetermined variables 1" in table


def test_unit_roots_count_as_stable():
    model = Model([Equation("x ~ a*x[-1] + e", params=["a"]), Equation("y ~ b*y[-1] + x", params=["b"])],
                 params={"a": 1.0, "b": 0.5}, roles={"e": Role.SHOCK})
    walk = model.linearize(at=SteadyState({"x": 0.0, "y": 0.0, "e": 0.0}, model=model)).solve()  # x's level is not pinned down
    assert walk.n_predetermined == 2 and walk.n_stable == 2
    assert "unit root" in str(walk.roots())


def test_more_predetermined_variables_than_stable_roots_is_refused():
    explosive = Model([Equation("x ~ a*x[-1] + e", params=["a"])], params={"a": 1.5}, roles={"e": Role.SHOCK})
    with pytest.raises(ModelError, match="0 stable roots for 1 predetermined"):
        explosive.linearize().solve()
