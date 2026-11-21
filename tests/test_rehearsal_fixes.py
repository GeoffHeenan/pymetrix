"""Found rehearsing the reviewer demo on 0.1.0a5: the Hausman dialog's defaults compared a fit with itself and
reported a statistic of 0 on 0 degrees of freedom; the posterior summary showed effective draws with decimals."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model
from pymetrix.errors import SpecificationError
from pymetrix.estimation.priors import Normal
from pymetrix.metadata import Role
from pymetrix.stats import compare


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=80)
    y = 1 + 0.5 * x + rng.normal(size=80)
    return Dataset.from_frame(pd.DataFrame({"y": y, "x": x}, index=pd.period_range("2000Q1", periods=80, freq="Q", name="time")))


def test_hausman_refuses_a_fit_compared_with_itself(data):
    fit = Equation("y ~ x").fit(data)
    with pytest.raises(SpecificationError, match="two different fits"):
        compare(fit, fit, test="hausman")
    again = Equation("y ~ x").fit(data)  # the same estimate in another object: nothing to test either
    with pytest.raises(SpecificationError, match="nothing to test"):
        compare(fit, again, test="hausman", sigma="separate")


def test_the_posterior_summary_counts_effective_draws_in_whole_draws():
    rng = np.random.default_rng(1)
    x = np.zeros(100)
    for t in range(1, 100):
        x[t] = 0.7 * x[t - 1] + 0.5 * rng.normal()
    d = Dataset.from_frame(pd.DataFrame({"x": x}, index=pd.period_range("1950Q1", periods=100, freq="Q", name="time")))
    model = Model([Equation("x ~ rho*x[-1] + e", params=["rho"])], params={"rho": 0.3}, roles={"e": Role.SHOCK})
    fit = model.estimate(d, ["x"], estimate=["rho"], shock_sd={"e": 0.5}, priors={"rho": Normal(0.5, 0.2)})
    row = fit.mcmc(draws=300, chains=2, seed=1, workers=1).summary().rows[0]
    assert isinstance(row[8], int) and row[8] > 0
