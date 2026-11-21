"""Metropolis-Hastings with jumps between modes. References: a two-component normal mixture with known weights, and a
model whose likelihood depends on rho squared (modes at plus and minus rho), its posterior computed exactly on a
grid. Without jumps, chains stay in the mode they start in."""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from pymetrix import Dataset, Equation, Model
from pymetrix.errors import SpecificationError
from pymetrix.estimation.priors import Normal
from pymetrix.metadata import Role
from pymetrix.models.bayes import _chain, ess


@pytest.mark.parametrize("narrow", [1.0, 0.4])  # the second mode the same shape as the first, or much narrower
def test_the_draws_keep_each_modes_share_of_a_mixture(narrow):
    m1, m2 = np.zeros(3), np.array([8.0, -6.0, 4.0])
    s1, s2 = np.array([1.0, 0.7, 1.4]), narrow * np.array([1.0, 0.7, 1.4])
    w2 = 0.3

    def logpost(x):
        a = np.log(1 - w2) + stats.norm.logpdf(x, m1, s1).sum()
        b = np.log(w2) + stats.norm.logpdf(x, m2, s2).sum()
        return float(np.logaddexp(a, b))

    factor, c0 = np.diag(s1), 2.38 / np.sqrt(3)
    modes = np.array([m1, m2])
    runs = [_chain(logpost, m1, factor, c0, None, 2000, 20000, s, True, modes, 0.1, j)
            for j, s in enumerate(np.random.SeedSequence(4).spawn(4))]
    draws = np.stack([r[0] for r in runs])
    in_second = (draws[:, :, 0] > 4).astype(float)
    share = in_second.mean()
    se = in_second.std() / np.sqrt(ess(in_second))
    assert abs(share - w2) < 4 * se and se < 0.03
    exact_mean = (1 - w2) * m1 + w2 * m2
    for i in range(3):
        assert abs(draws[:, :, i].mean() - exact_mean[i]) < 4 * draws[:, :, i].std() / np.sqrt(ess(draws[:, :, i]))
    assert all(r[4][1] > 50 for r in runs)  # every chain crossed many times

    alone = _chain(logpost, m1, factor, c0, None, 2000, 20000, np.random.SeedSequence(4), True, None, 0.1)
    assert (alone[0][:, 0] > 4).mean() == 0  # without jumps the chain never leaves the first mode


def _quarters(n):
    return pd.period_range("1950Q1", periods=n, freq="Q", name="time")


@pytest.fixture(scope="module")
def squared():
    """x = rho^2 x[-1] + e: the likelihood is the same at rho and -rho; a prior centred at 0.2 weighs them unequally."""
    rng = np.random.default_rng(2)
    x = np.zeros(150)
    for t in range(1, 150):
        x[t] = 0.6 * x[t - 1] + 0.5 * rng.normal()
    data = Dataset.from_frame(pd.DataFrame({"x": x}, index=_quarters(150)))
    model = Model([Equation("x ~ rho*rho*x[-1] + e", params=["rho"])], params={"rho": 0.7}, roles={"e": Role.SHOCK})
    fit = model.estimate(data, ["x"], estimate=["rho"], shock_sd={"e": 0.5}, priors={"rho": Normal(0.2, 0.5)})
    return model, data, fit


def _exact_share_below_zero(fit):
    # 1201 points (0 on the grid): the share moves by under 1e-10 from 4801 points (and from 601), against a
    # tolerance of 4 Monte Carlo standard errors (hundredths); each point solves the model and runs a Kalman filter
    g = np.linspace(-1.2, 1.2, 1201)
    lp = np.array([fit.log_posterior([v]) for v in g])
    w = np.exp(np.where(np.isfinite(lp), lp - lp[np.isfinite(lp)].max(), -np.inf))
    return integrate.trapezoid(w * (g < 0), g) / integrate.trapezoid(w, g)


def test_a_bimodal_posterior_matches_the_exact_one(squared):
    _, _, fit = squared
    exact = _exact_share_below_zero(fit)
    assert 0.2 < exact < 0.45  # two modes, unequal weights
    other = {"rho": -fit.params.values[0]}
    post = fit.mcmc(draws=6000, chains=2, seed=3, modes=[other], workers=1)
    below = (post.draws[:, :, 0] < 0).astype(float)
    se = below.std() / np.sqrt(ess(below))
    assert abs(below.mean() - exact) < 4 * se and se < 0.04
    assert post.modes.shape == (2, 1) and post.jumps.shape == (2, 2) and np.all(post.jumps[:, 1] > 20)
    notes = " ".join(post.summary().notes)
    assert "Mode jumps between 2 modes" in notes
    assert post.provenance.method["jump"] == 0.05 and len(post.provenance.method["modes"]) == 2


def test_the_modes_come_from_the_estimates_starts(squared):
    model, data, _ = squared
    fit = model.estimate(data, ["x"], estimate=["rho"], shock_sd={"e": 0.5}, priors={"rho": Normal(0.2, 0.5)}, starts=6, seed=1,
                         workers=1)
    ends = sorted(r["to"]["rho"] for r in fit.provenance.method["starts"])
    assert ends[0] < 0 < ends[-1]  # the starts found both modes
    post = fit.mcmc(draws=3000, chains=2, seed=3, workers=1)
    assert post.modes is not None and len(post.modes) == 2
    assert post.modes[:, 0].min() < 0 < post.modes[:, 0].max()  # near plus and minus rho (the prior moves them a little)
    plain = fit.mcmc(draws=3000, chains=2, seed=3, workers=1, modes=False)
    assert plain.modes is None
    for c in range(2):  # without jumps each chain keeps to one mode
        assert np.all(plain.draws[c, :, 0] > 0) or np.all(plain.draws[c, :, 0] < 0)


def test_one_mode_leaves_the_draws_unchanged(squared):
    _, _, fit = squared
    a = fit.mcmc(draws=500, chains=1, seed=9, workers=1)
    b = fit.mcmc(draws=500, chains=1, seed=9, workers=1, modes=False)
    assert a.modes is None and np.array_equal(a.draws, b.draws)


def test_mode_jump_options_are_checked(squared):
    _, _, fit = squared
    with pytest.raises(SpecificationError, match="jump"):
        fit.mcmc(draws=100, jump=1.0)
    with pytest.raises(SpecificationError, match="every estimated parameter"):
        fit.mcmc(draws=100, modes=[{"sigma": 1.0}])
