"""Monte Carlo experiments, checked against what is known exactly: OLS's sampling mean and standard deviation under a
fixed design, a true null's rejection rate within its binomial band, the same draws whatever the number of workers,
failures counted without stopping the run."""

import operator

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, MonteCarlo

X = np.linspace(-2, 2, 40)  # a fixed design: the slope's sampling s.d. is sigma / sqrt(sum (x - xbar)^2)


def _dgp(rng):
    return Dataset.from_frame(pd.DataFrame({"y": 1 + 0.5 * X + rng.normal(scale=2.0, size=len(X)), "x": X},
                                           index=pd.period_range("2000Q1", periods=len(X), freq="Q")))


def test_ols_sampling_distribution_and_a_true_nulls_size():
    mc = MonteCarlo(_dgp, lambda d: Equation("y ~ x").fit(d), reps=2000, truth={"x": 0.5, "const": 1.0}).run()
    m = mc.moments()
    sd = 2.0 / np.sqrt(np.sum((X - X.mean()) ** 2))
    assert m.loc["x", "mean"] == pytest.approx(0.5, abs=3 * sd / np.sqrt(2000))  # unbiased
    assert m.loc["x", "s.d."] == pytest.approx(sd, rel=0.05)
    assert m.loc["x", "RMSE"] == pytest.approx(sd, rel=0.05) and abs(m.loc["x", "bias"]) < 3 * sd / np.sqrt(2000)
    assert m.loc["x", "replications"] == 2000 and not mc.failures
    tests = MonteCarlo(_dgp, lambda d: Equation("y ~ x").fit(d).test("wald", "x = 0.5"), reps=2000).run()
    r = tests.rejection()
    assert abs(r["5%"] - 0.05) < 3 * r["5% s.e."] and abs(r["10%"] - 0.10) < 3 * r["10% s.e."]  # exact F under normality
    assert "Rejection rates" in str(tests.summary())


def test_the_same_draws_whatever_the_number_of_workers():
    dgp = operator.methodcaller("normal", size=30)  # picklable, so the replications can go to other processes
    one = MonteCarlo(dgp, np.mean, reps=40, seed=7).run()
    two = MonteCarlo(dgp, np.mean, reps=40, seed=7, workers=2).run()
    pd.testing.assert_frame_equal(one.draws, two.draws)
    assert not two.notes  # it did run in other processes
    assert one.moments().loc["value", "s.d."] == pytest.approx(1 / np.sqrt(30), rel=0.3)
    here = MonteCarlo(lambda rng: rng.normal(size=30), np.mean, reps=40, seed=7, workers=2).run()  # a lambda: run here
    pd.testing.assert_frame_equal(one.draws, here.draws)
    assert "could not be sent" in here.notes[0]


def test_failures_are_counted_and_kept():
    def estimator(x):
        if x[0] > 1:
            raise ValueError("too big")
        return {"mean": float(np.mean(x)), "p_value": float(np.abs(x[0]) > 1.96)}

    mc = MonteCarlo(lambda rng: rng.normal(size=5), estimator, reps=300).run()
    assert 0 < len(mc.failures) < 100 and "too big" in next(iter(mc.failures.values()))
    assert mc.draws["mean"].isna().sum() == len(mc.failures)
    assert "failed" in str(mc.summary())
    with pytest.raises(ValueError, match="every replication failed"):
        MonteCarlo(lambda rng: 1, lambda d: 1 / 0, reps=3).run()
    with pytest.raises(ValueError, match="function of a numpy Generator"):
        MonteCarlo("not a function", np.mean).run()
