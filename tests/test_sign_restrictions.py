"""Sign-restricted VARs: every accepted impact matrix factors the error covariance and has the signs asked for; the
identified set matches its bounds found independently over a grid of rotation angles (two variables); a simulated
structural VAR's true responses lie in the set; impossible restrictions are refused; the dialogs record the code."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import VAR, Dataset, SignRestrictions
from pymetrix.errors import SpecificationError

B0 = np.array([[1.0, 0.5], [-0.6, 0.8]])  # columns: supply (output up, inflation down), demand (both up)
A1 = np.array([[0.5, 0.1], [0.0, 0.4]])
SIGNS = {"supply": {"gy": "+", "pi": "-"}, "demand": {"gy": "+", "pi": "+"}}


def _simulate(B, A, T, seed=0, names=("gy", "pi")):
    rng = np.random.default_rng(seed)
    y = np.zeros((T, len(names)))
    for t in range(1, T):
        y[t] = A @ y[t - 1] + B @ rng.standard_normal(len(names))
    idx = pd.period_range("1000Q1", periods=T, freq="Q", name="time")
    return Dataset.from_frame(pd.DataFrame(y, columns=list(names), index=idx))


@pytest.fixture(scope="module")
def var():
    return VAR(["gy", "pi"], lags=1).fit(_simulate(B0, A1, 3000))


def test_every_accepted_impact_matrix_factors_sigma_and_has_the_signs(var):
    sr = var.identify(SignRestrictions(SIGNS, horizon=2, rotations=300))
    sigma = np.asarray(var.sigma, float)
    from pymetrix.var import _ma

    phi = _ma(var.coefs, 2)
    for B in sr.sign_draws():
        assert B @ B.T == pytest.approx(sigma, rel=1e-10)  # an orthogonal rotation of a factor of sigma
        R = phi @ B
        assert np.all(R[:, 0, 0] > 0) and np.all(R[:, 1, 0] < 0)  # supply: gy+, pi- on impact and two periods after
        assert np.all(R[:, 0, 1] > 0) and np.all(R[:, 1, 1] > 0)  # demand: gy+, pi+
    assert sr.shocks == ["supply", "demand"]


def test_the_identified_set_matches_a_grid_over_rotation_angles(var):
    # with two variables every orthogonal Q is a rotation or reflection by an angle, so the impact responses to the
    # supply shock are P0 u for unit u whose orthogonal partner (either sign) carries the demand signs
    sigma = np.asarray(var.sigma, float)
    P0 = np.linalg.cholesky(sigma)
    values = []
    for theta in np.linspace(0, 2 * np.pi, 200_001):
        u = np.array([np.cos(theta), np.sin(theta)])
        b, c = P0 @ u, P0 @ np.array([-u[1], u[0]])
        if b[0] > 0 and b[1] < 0 and ((c[0] > 0 and c[1] > 0) or (c[0] < 0 and c[1] < 0)):
            values.append(b[0])
    low, high = min(values), max(values)
    draws = var.identify(SignRestrictions(SIGNS, rotations=4000, seed=1)).sign_draws()
    supply_gy = np.array([B[0, 0] for B in draws])
    assert supply_gy.min() >= low - 1e-9 and supply_gy.max() <= high + 1e-9  # never outside the set
    assert supply_gy.min() == pytest.approx(low, abs=0.02 * (high - low)) and supply_gy.max() == pytest.approx(high, abs=0.02 * (high - low))


def test_the_true_responses_lie_in_the_set_and_the_median_target_is_a_member(var):
    sr = var.identify(SignRestrictions(SIGNS, seed=2))
    irf = sr.irf(8, draws=0)
    from pymetrix.var import _ma

    truth = _ma(A1[None], 8) @ B0
    assert np.all(truth >= irf.lower - 0.05) and np.all(truth <= irf.upper + 0.05)  # sampling error in the VAR is small
    P = sr.impact()
    assert any(np.allclose(P, B) for B in sr.sign_draws())  # the median target is one coherent accepted model
    fevd = sr.fevd(8)
    assert list(fevd.shocks) == ["supply", "demand"] and np.allclose(fevd.values.sum(axis=2), 1.0)
    assert "Sign restrictions (supply: gy+, pi-; demand: gy+, pi+" in irf.identification
    bands = sr.irf(4, draws=40)  # with the bootstrap: a rotation for each draw of the VAR
    assert bands.lower.shape == (5, 2, 2) and "draws of the VAR" in bands.identification


def test_a_partly_identified_var_and_the_bayesian_var():
    data = _simulate(np.array([[1.0, 0.3, 0.0], [-0.5, 0.9, 0.2], [0.2, 0.1, 0.7]]), np.diag([0.5, 0.4, 0.3]), 800, seed=4,
                     names=("gy", "pi", "r"))
    var = VAR(["gy", "pi", "r"], lags=1).fit(data)
    sr = var.identify(SignRestrictions({"supply": {"gy": "+", "pi": "-"}}, rotations=200))
    assert sr.shocks == ["supply", "unidentified 1", "unidentified 2"]
    assert sr.irf(4, draws=0).values.shape == (5, 3, 3)
    from pymetrix import Bayes, Minnesota

    bvar = VAR(["gy", "pi", "r"], lags=1).fit(data, method=Bayes(prior=Minnesota(), draws=60))  # one accepted rotation per posterior draw
    irf = bvar.identify(SignRestrictions({"supply": {"gy": "+", "pi": "-"}}, rotations=100)).irf(4, draws=50)
    assert "posterior" in irf.identification and irf.lower is not None and np.all(irf.lower[0, 0, 0] > 0)


def test_restrictions_that_cannot_hold_are_refused():
    oscillating = VAR(["gy", "pi"], lags=1).fit(_simulate(B0, -0.8 * np.eye(2), 600, seed=3))
    with pytest.raises(SpecificationError, match="no rotation satisfies"):
        oscillating.identify(SignRestrictions(SIGNS, horizon=1, rotations=5, max_tries=2000))  # responses flip each period
    v = VAR(["gy", "pi"], lags=1).fit(_simulate(B0, A1, 300))
    with pytest.raises(SpecificationError, match="does not have"):
        v.identify(SignRestrictions({"supply": {"gdp": "+"}}))
    with pytest.raises(SpecificationError, match="use '\\+' or '-'"):
        SignRestrictions({"supply": {"gy": "up"}})
    with pytest.raises(SpecificationError, match="apart from the variables"):
        v.identify(SignRestrictions({"gy": {"gy": "+"}}))
    with pytest.raises(SpecificationError, match="identify the VAR by sign restrictions"):
        v.sign_draws()


def test_the_dialogs_record_the_code():
    from pymetrix.commands import FitVAR, ImpulseResponse, Session, VarianceDecomposition

    s = Session()
    s.objects["data"] = _simulate(B0, A1, 400)
    s.run(FitVAR("gy, pi", lags="1", name="var"))
    signs = "supply: gy+, pi-; demand: gy+, pi+"
    s.run(ImpulseResponse("var", periods=8, scheme="SignRestrictions", signs=signs, sign_horizon=1, draws=0))
    s.run(VarianceDecomposition("var", periods=8, scheme="SignRestrictions", signs=signs))
    assert s.history[1].code == ("irf = var.identify(SignRestrictions({'supply': {'gy': 1, 'pi': -1}, "  # signs as stored
                                 "'demand': {'gy': 1, 'pi': 1}}, horizon=1)).irf(periods=8, draws=0)")
    assert list(s.objects["fevd"].shocks) == ["supply", "demand"]
    with pytest.raises(Exception, match="end each variable with"):
        s.run(ImpulseResponse("var", scheme="SignRestrictions", signs="supply: gy, pi-"))
    namespace = {"data": s.objects["data"]}
    exec(s.script(), namespace)  # noqa: S102 - the script identifies the same way
    assert np.allclose(namespace["irf"].values, s.objects["irf"].values)
