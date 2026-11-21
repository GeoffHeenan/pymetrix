"""Identification without data, pairwise collinearity, elasticities and the public moments vector.

Model.identification and Template.identification test a calibrated model's parameters before any data exist. They
give the same Identification as fitting and testing at the same point. A product a*b of two parameters is found
unidentified, one short of full rank. Each parameter's pairwise collinearity is the |cosine| of its scaled Jacobian
column with its closest parameter's, checked by hand, in levels and in elasticities. The moments vector is
checked against an AR(1)'s analytic moments."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model, templates
from pymetrix.metadata import Role
from pymetrix.models import ModelError
from pymetrix.models.bayes import Identification, _moments, moments


@pytest.fixture(scope="module")
def nk():
    return templates.new_keynesian()


@pytest.fixture(scope="module")
def nk_data(nk):
    truth = nk.solve().simulate(120, nk.shock_sd, seed=4)
    truth.index = pd.period_range("1990Q1", periods=120, freq="Q", name="time")
    return nk.measurement.apply(Dataset.from_frame(truth))


def _same(a, b):
    assert a.names == b.names and a.rank == b.rank and a.closest == b.closest
    assert a.at == pytest.approx(b.at, abs=0)
    for field in ("jacobian", "singular_values", "sensitivity", "collinearity", "pairwise"):
        np.testing.assert_allclose(getattr(a, field), getattr(b, field), rtol=1e-12, atol=1e-14)


def test_template_without_data_equals_fitting_and_testing(nk, nk_data):
    names = ["kappa", "rho", "sd_ey"]
    fit = nk.estimate(nk_data, names)
    calibration = {"kappa": nk.model.parameters["kappa"].value, "rho": nk.model.parameters["rho"].value, "sd_ey": nk.shock_sd["ey"]}
    for elasticity in (False, True):
        _same(nk.identification(names, lags=3, elasticity=elasticity), fit.identification(lags=3, at=calibration, elasticity=elasticity))
    # at= moves the point in both
    other = {"kappa": 0.2, "rho": 0.7, "sd_ey": 0.4}
    _same(nk.identification(names, at=other), fit.identification(at=other))
    # the default parameters are estimate's: every calibrated parameter and shock size
    assert set(nk.identification().names) == {*nk.model.parameters.by_status("calibrated"), *(f"sd_{s}" for s in nk.shocks)}


def test_a_product_of_two_parameters_is_one_short(nk_data):
    product = Model([Equation("x ~ a*b*x[-1] + e", params=["a", "b"])], params={"a": 0.8, "b": 0.9}, roles={"e": Role.SHOCK})
    bad = product.identification(["x"], ["a", "b", "sd_e"], shock_sd={"e": 0.5})
    assert bad.rank == 2 and len(bad.names) == 3 and not bad.identified and set(bad.problems) == {"a", "b"}
    assert bad.closest[:2] == ("b", "a") and bad.pairwise[0] == pytest.approx(1.0, abs=1e-8)
    good = product.identification(["x"], ["a", "sd_e"], shock_sd={"e": 0.5})
    assert good.identified
    with pytest.raises(ModelError, match="not among the parameters tested"):
        product.identification(["x"], ["a"], at={"b": 0.5})


def _pairwise_by_hand(idn, base=None):
    theta = np.array([idn.at[n] for n in idn.names])
    scaled = idn.jacobian * np.abs(theta)
    if base is not None:
        keep = np.abs(base) > 1e-6 * np.max(np.abs(base))
        scaled = scaled[keep] / np.abs(base[keep])[:, None]
    unit = scaled / np.linalg.norm(scaled, axis=0)
    cos = np.abs(unit.T @ unit)
    np.fill_diagonal(cos, -1)
    return cos.max(axis=1), [idn.names[k] for k in cos.argmax(axis=1)]


def test_pairwise_collinearity_in_levels_and_elasticities(nk):
    levels = nk.identification()
    pairwise, closest = _pairwise_by_hand(levels)
    np.testing.assert_allclose(levels.pairwise, pairwise, rtol=1e-12)
    assert list(levels.closest) == closest
    assert np.all(levels.pairwise <= levels.collinearity + 1e-12)  # one parameter explains no more than all the others
    elastic = nk.identification(elasticity=True)
    system = nk.solve().system(nk.measurement, nk.shock_sd)
    pairwise, closest = _pairwise_by_hand(elastic, moments(system, 4))
    np.testing.assert_allclose(elastic.pairwise, pairwise, rtol=1e-10)
    assert list(elastic.closest) == closest and elastic.elasticity and not levels.elasticity
    np.testing.assert_array_equal(elastic.jacobian, levels.jacobian)  # the Jacobian itself is the same
    assert "pairwise" in str(levels.summary()) and "In elasticities" in str(elastic.summary())


def test_moments_are_public_and_match_an_ar1():
    rho, s = 0.7, 0.5
    model = Model([Equation("x ~ rho*x[-1] + e", params=["rho"])], params={"rho": rho}, roles={"e": Role.SHOCK})
    system = model.linearize().solve().system(["x"], {"e": s})
    g0 = s**2 / (1 - rho**2)
    np.testing.assert_allclose(moments(system, 3), [0.0, g0, rho * g0, rho**2 * g0, rho**3 * g0], rtol=1e-12)
    np.testing.assert_array_equal(_moments(system, 3), moments(system, 3))  # the old private name still works


def test_existing_fields_come_first_unchanged():
    names = [f.name for f in dataclasses.fields(Identification)]
    assert names[:9] == ["names", "at", "jacobian", "rank", "singular_values", "sensitivity", "collinearity", "closest", "lags"]
    assert names[9:] == ["pairwise", "elasticity"]
