"""Monte Carlo standard errors of posterior means (s.d. / sqrt(ESS)) in a Bayes fit's posterior table, summary and stats,
beside the ESS: checked against the AR(1) closed form for the variance of a chain's mean, batch means computed by hand,
independent draws (ESS = N), and ArviZ's mcse when it is installed."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation
from pymetrix.estimation import Bayes
from pymetrix.estimation.priors import Normal
from pymetrix.models.bayes import ess, hpd, mcse, rhat


def _ar1(phi, chains=4, n=20_000, seed=0):
    """Chains of a stationary AR(1) with unit innovations: x_t = phi x_{t-1} + e_t."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(chains, n))
    x = np.empty((chains, n))
    x[:, 0] = e[:, 0] / np.sqrt(1 - phi**2)  # started in the stationary distribution
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9])
def test_against_the_ar1_closed_form(phi):
    x = _ar1(phi)
    N = x.size
    var_x = 1 / (1 - phi**2)
    exact = np.sqrt(var_x * (1 + phi) / (1 - phi) / N)  # the long-run variance of an AR(1) mean, over N
    assert mcse(x) == pytest.approx(exact, rel=0.1)  # the estimate of a long-run variance is itself noisy
    assert mcse(x) == pytest.approx(x.std(ddof=1) / np.sqrt(ess(x)), rel=1e-12)  # the definition


def test_against_batch_means_by_hand():
    x = _ar1(0.7, seed=3)
    size = 400  # batches much longer than the chain's memory (1 / (1 - 0.7) draws)
    means = x.reshape(x.shape[0], -1, size).mean(axis=2).ravel()
    batch = np.sqrt(means.var(ddof=1) * size / x.size)  # the batch-means estimate of the mean's Monte Carlo error
    assert mcse(x) == pytest.approx(batch, rel=0.1)


def test_independent_draws_have_the_textbook_error():
    rng = np.random.default_rng(5)
    x = rng.normal(2.0, 3.0, size=(2, 50_000))
    assert mcse(x) == pytest.approx(3.0 / np.sqrt(x.size), rel=0.03)
    assert np.isnan(mcse(np.array([[1.0, 2.0, 3.0]])))  # too short for an ESS


@pytest.fixture(scope="module")
def fit():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    y = 1.0 + 0.6 * x + rng.normal(scale=0.8, size=60)
    data = Dataset.from_frame(pd.DataFrame({"y": y, "x": x}, index=pd.period_range("1960Q1", periods=60, freq="Q")))
    return Equation("y ~ x").fit(data, method=Bayes(prior={"x": Normal(0.5, 0.2)}, draws=2000, chains=2, burn=200, sampler="rwmh"))


def test_in_the_fits_posterior_summary_and_stats_beside_the_ess(fit):
    post = fit.posterior()
    columns = list(post.columns)
    assert columns[columns.index("ESS") + 1] == "MCSE"
    d = fit.draws
    for name in ("const", "x", "sigma"):
        per = np.stack([d.loc[d["chain"] == c, name].to_numpy() for c in (1, 2)])
        by_hand = d[name].std(ddof=1) / np.sqrt(ess(per))
        assert post.loc[name, "MCSE"] == pytest.approx(by_hand, rel=1e-12)
        assert fit.stats[f"mcse_{name}"] == pytest.approx(by_hand, rel=1e-12)
        assert fit.stats[f"ess_{name}"] == pytest.approx(ess(per), rel=1e-12)
        assert round(fit.stats[f"ess_{name}"]) == post.loc[name, "ESS"]
        assert fit.stats[f"mcse_{name}"] < post.loc[name, "s.d."]
    table = fit.summary()
    assert table.columns[table.columns.index("ESS") + 1] == "MCSE"
    statistics = fit.statistics()
    assert {"mcse_x", "ess_x"} <= set(statistics["statistic"])


def test_the_columns_there_before_are_unchanged(fit):
    """The posterior table as before the MCSE column (computed here as it was), to the last digit."""
    d, post = fit.draws, fit.posterior()
    for name in ("const", "x", "sigma"):
        x = d[name].to_numpy()
        per = np.stack([x[d["chain"].to_numpy() == c] for c in (1, 2)])
        low, high = hpd(x, 0.9)
        before = {"mean": x.mean(), "s.d.": x.std(ddof=1), "median": float(np.median(x)), "90% HPD low": low,
                  "90% HPD high": high, "R-hat": rhat(per), "ESS": int(round(ess(per)))}
        assert {k: post.loc[name, k] for k in before} == before
    assert list(post.columns[:-1]) == ["mean", "s.d.", "median", "90% HPD low", "90% HPD high", "R-hat", "ESS"]


@pytest.mark.parametrize("phi, seed", [(0.0, 1), (0.6, 11), (0.9, 2)])
def test_against_arviz(phi, seed):
    """ArviZ 0.22 agreed within 0.3% on these chains (it splits each chain in two for its ESS; pymetrix does not)."""
    az = pytest.importorskip("arviz")
    x = _ar1(phi, chains=4, n=5000, seed=seed)
    theirs = float(az.mcse(az.from_dict(posterior={"theta": x}), method="mean")["theta"])
    assert mcse(x) == pytest.approx(theirs, rel=0.01)
