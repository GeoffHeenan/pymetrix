"""Maximum likelihood for explicit-form equations: the right-hand side as the family's index.

- A linear index written explicitly (y ~ a + b*x) against the linear-form fit (itself checked against statsmodels),
  for Logit, Probit and Poisson: estimates to 1e-8, standard errors to 1e-9; white standard errors too.
- Genuinely nonlinear indexes against statsmodels' GenericLikelihoodModel (numerical optimisation and Hessian):
  a logit with eta = a + b x^c and a Poisson with an identity link, mean = a + b x: estimates to 1e-6, standard errors
  to 1e-5 relatively; bounds kept; refusals."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from pymetrix import Dataset, Equation
from pymetrix.errors import SpecificationError
from pymetrix.estimation.families import Logit, Poisson, Probit
from pymetrix.estimation.ml import ML
from pymetrix.metadata import Param


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.uniform(0.5, 3, n)
    frame = pd.DataFrame({"yb": (rng.uniform(size=n) < expit(-1 + 0.8 * x**1.5)).astype(float),
                          "yc": rng.poisson(1 + 2 * x).astype(float), "x": x})
    return Dataset.from_frame(frame.rename_axis("id").reset_index(), entity="id")


@pytest.mark.parametrize("family, y", [(Logit(), "yb"), (Probit(), "yb"), (Poisson(), "yc")], ids=["logit", "probit", "poisson"])
def test_a_linear_index_written_out(data, family, y):
    for cov in ("classic", "white"):
        linear = Equation(f"{y} ~ x", family=family).fit(data, method=ML(cov=cov))
        explicit = Equation(f"{y} ~ a + b*x", params=["a", "b"], family=family).fit(data, method=ML(cov=cov))
        assert np.allclose(explicit.params.values, linear.params.values, atol=1e-8)
        assert np.allclose(explicit.params.se, linear.params.se, atol=1e-9)
        assert explicit.stats["loglike"] == pytest.approx(linear.stats["loglike"], rel=1e-12)


def test_nonlinear_indexes_against_statsmodels(data):
    from statsmodels.base.model import GenericLikelihoodModel

    f = data.frame
    x, yb, yc = f.x.to_numpy(), f.yb.to_numpy(), f.yc.to_numpy()

    class PowerLogit(GenericLikelihoodModel):
        def loglike(self, p):
            e = p[0] + p[1] * x ** p[2]
            return np.sum(yb * e - np.logaddexp(0, e))

    class IdentityPoisson(GenericLikelihoodModel):
        def loglike(self, p):
            m = p[0] + p[1] * x
            return np.sum(yc * np.log(m) - m) if np.all(m > 0) else -1e300

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref1 = PowerLogit(yb, np.ones((len(x), 1))).fit(start_params=[0, 1, 1], method="bfgs", disp=0, maxiter=2000, gtol=1e-10)
        ref2 = IdentityPoisson(yc, np.ones((len(x), 1))).fit(start_params=[1, 1], method="bfgs", disp=0, maxiter=2000, gtol=1e-10)
    ours1 = Equation("yb ~ a + b*x^c", params=["a", "b", "c"], family=Logit()).fit(data, method=ML(start={"a": 0, "b": 1, "c": 1}))
    ours2 = Equation("yc ~ log(a + b*x)", params=["a", "b"], family=Poisson()).fit(data, method=ML(start={"a": 1, "b": 1}))
    from scipy.special import gammaln

    for ours, ref, constant in ((ours1, ref1, 0.0), (ours2, ref2, -np.sum(gammaln(yc + 1)))):  # the reference leaves out -ln y!
        assert np.allclose(ours.params.values, ref.params, atol=1e-6)
        assert np.allclose(ours.params.se, ref.bse, rtol=1e-5)
        assert ours.stats["loglike"] == pytest.approx(ref.llf + constant, rel=1e-10) and ours.provenance.converged


def test_bounds_and_refusals(data):
    bounded = Equation("yb ~ a + b*x^c", params=["a", "b", Param("c", bounds=(0.5, 1.2), start=1.0)], family=Logit()).fit(
        data, method=ML(start={"a": 0, "b": 1}))
    assert bounded.params.as_dict()["c"] <= 1.2 + 1e-12
    with pytest.raises(SpecificationError, match="starting values"):
        Equation("yc ~ log(a + b*x)", params=["a", "b"], family=Poisson()).fit(data, method=ML(start={"a": -5, "b": 0}))
    with pytest.raises(SpecificationError, match="0/1"):
        Equation("yc ~ a + b*x", params=["a", "b"], family=Logit()).fit(data, method=ML())
