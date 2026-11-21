"""Estimation from several starting points, draws from the priors, and jobs in other processes."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, templates
from pymetrix._processes import ProcessError, run_in_processes
from pymetrix.estimation.priors import Beta, Gamma, InvGamma, InvGamma1, Normal, Uniform
from pymetrix.models.estimation import _starts_note


@pytest.mark.parametrize("prior, sd", [(Normal(1, 0.5), 0.5), (Beta(0.6, 0.1), 0.1), (Gamma(0.3, 0.1), 0.1),
                                       (InvGamma(0.2, 0.05), 0.05), (InvGamma1(0.2, 0.1), 0.1), (Uniform(0, 2), 2 / np.sqrt(12))])
def test_draws_from_each_prior_have_its_mean_and_standard_deviation(prior, sd):
    rng = np.random.default_rng(0)
    draws = np.array([prior.rvs(rng) for _ in range(40000)])
    mean = (prior.low + prior.high) / 2 if isinstance(prior, Uniform) else prior.mean
    assert draws.mean() == pytest.approx(mean, abs=4 * sd / np.sqrt(len(draws)) + 1e-3)
    assert draws.std() == pytest.approx(sd, rel=0.05)


def test_several_starts_find_the_mode_and_say_so():
    nk = templates.new_keynesian()
    truth = nk.solve().simulate(100, nk.shock_sd, seed=6)
    truth.index = pd.period_range("2000Q1", periods=100, freq="Q", name="time")
    data = nk.measurement.apply(Dataset.from_frame(truth))
    priors = {"kappa": Gamma(0.15, 0.05), "rho": Beta(0.7, 0.1)}
    one = nk.estimate(data, estimate=["kappa", "rho"], priors=priors)
    three = nk.estimate(data, estimate=["kappa", "rho"], priors=priors, starts=3, seed=1)
    starts = three.provenance.method["starts"]
    assert len(starts) == 3 and starts[0]["from"] == pytest.approx(dict(zip(["kappa", "rho"], [0.1, 0.8])))
    assert len({tuple(round(v, 6) for v in s["from"].values()) for s in starts}) == 3  # different starting points
    np.testing.assert_allclose(three.params.values, one.params.values, atol=1e-4)
    assert three.stats["logpost"] == pytest.approx(max(s["log_posterior"] for s in starts))
    assert "3 starts: 3 reached the best log posterior" in "\n".join(three.summary().notes)


def test_the_note_names_other_modes():
    starts = [{"log_posterior": -1484.28}, {"log_posterior": -1484.82}, {"log_posterior": -1484.2801}]
    assert _starts_note(starts) == ("3 starts: 2 reached the best log posterior (-1484.2800); others stopped at -1484.82 "
                                    "(other modes, or not converged)")
    assert _starts_note(starts[:1]) == ""


def _square(x):  # defined here, so another process cannot import it
    return x * x


def test_jobs_in_other_processes():
    seen = []
    assert run_in_processes([(pow, (i, 2)) for i in range(3)], 2, lambda j, r: seen.append(j)) == [0, 1, 4]
    assert sorted(seen) == [0, 1, 2]
    with pytest.raises(ProcessError, match="invalid literal"):
        run_in_processes([(int, ("x",))], 1)
    assert run_in_processes([(lambda x: x, (1,))], 1) is None  # cannot be pickled: the caller runs it here
    assert run_in_processes([(_square, (3,))], 1) is None  # cannot be imported there: likewise
