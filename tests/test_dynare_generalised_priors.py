"""Dynare's generalised priors (the third and fourth prior parameters of estimated_params) and ``prior_trunc``.

A beta on (low, high), a gamma or inverse gamma shifted to start at low, the mean and standard deviation those of the
generalised distribution (Dynare's manual); a normal's third and fourth parameters are bounds. Checked against Dynare 6.0's
log prior densities and prior bounds at prior_trunc=0.01 (tests/fixtures/dynare-me-corr-priors.json), SciPy's located and
scaled distributions, and the means and standard deviations by numerical integration.

One difference from Dynare 6.0: its inverse gamma of type 2 with a third parameter takes its scale from the mean before
the shift (``s = 2 mu (1 + (mu - low)^2 / sd^2)``), so its mean is mean + low, not the mean given; pymetrix keeps the mean
given, as the manual states and as Dynare does for the other shapes. The test shows Dynare's number is that formula.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats

from pymetrix import read_mod
from pymetrix.estimation.priors import (Beta, Gamma, GeneralisedBeta, InvGamma1, Normal, ShiftedGamma, ShiftedInvGamma1,
                                        ShiftedInvGamma2, quantile)
from pymetrix.models import ModelError

FIXTURES = Path(__file__).parent / "fixtures"
MOD = FIXTURES / "dynare" / "me_corr_priors.mod"
RECORDED = json.loads((FIXTURES / "dynare-me-corr-priors.json").read_text(encoding="utf-8"))


def _moments(prior):
    lo, hi = prior.support
    density = lambda x: math.exp(prior.logpdf(x))  # noqa: E731
    mass = integrate.quad(density, lo, hi, limit=200)[0]
    mean = integrate.quad(lambda x: x * density(x), lo, hi, limit=200)[0]
    second = integrate.quad(lambda x: x * x * density(x), lo, hi, limit=200)[0]
    return mass, mean, math.sqrt(second - mean**2)


def test_the_file_reads_generalised_priors():
    dyn = read_mod(MOD)
    assert dyn.priors == {"rho": GeneralisedBeta(0.6, 0.1, -0.5, 1.0), "alpha": ShiftedGamma(0.6, 0.1, 0.2),
                          "sd_e": ShiftedInvGamma1(0.6, 0.3, 0.1), "sd_y": ShiftedInvGamma2(0.4, 0.2, 0.05),
                          "corr_y_c": Normal(0.3, 0.1)}
    # the defaults (0 and 1 for a beta, 0 for the others) give the usual priors
    plain = read_mod(MOD.read_text(encoding="utf-8").replace("-0.5, 1;", "0, 1;").replace("0.1, 0.2;", "0.1, 0;"))
    assert plain.priors["rho"] == Beta(0.6, 0.1) and plain.priors["alpha"] == Gamma(0.6, 0.1)


def test_log_densities_match_dynare():
    dyn = read_mod(MOD)
    for name, value in RECORDED["log_prior"].items():
        if name != "sd_y":
            assert dyn.priors[name].logpdf(dyn.start[name]) == pytest.approx(value, abs=1e-10), name
    # Dynare 6.0's shifted inverse gamma 2: nu = 2 (2 + (mu - low)^2 / sd^2), s = 2 mu (1 + (mu - low)^2 / sd^2) (mu, not
    # mu - low); 1/(x - low) is gamma(nu/2, scale 2/s)
    mu, sd, low, x = 0.4, 0.2, 0.05, 0.35
    m2 = (mu - low) ** 2
    nu, s = 2 * (2 + m2 / sd**2), 2 * mu * (1 + m2 / sd**2)
    assert stats.invgamma.logpdf(x - low, nu / 2, scale=s / 2) == pytest.approx(RECORDED["log_prior"]["sd_y"], abs=1e-10)
    assert _moments(dyn.priors["sd_y"])[1] == pytest.approx(0.4, abs=1e-6)  # pymetrix: the mean given


def test_hand_formulas_and_moments():
    for prior in (GeneralisedBeta(0.6, 0.1, -0.5, 1.0), GeneralisedBeta(3.5, 0.2, 3, 4), ShiftedGamma(0.6, 0.1, 0.2),
                  ShiftedInvGamma1(0.6, 0.3, 0.1), ShiftedInvGamma2(0.4, 0.2, 0.05)):
        mass, mean, sd = _moments(prior)
        assert mass == pytest.approx(1, abs=1e-7) and mean == pytest.approx(prior.mean, abs=1e-6), prior
        assert sd == pytest.approx(prior.sd, rel=1e-5), prior
    g = GeneralisedBeta(3.5, 0.2, 3, 4)  # the beta on (3, 4) as SciPy's loc and scale
    a = 0.5 * (0.5 * 0.5 / 0.2**2 - 1)
    assert g.logpdf(3.3) == pytest.approx(stats.beta.logpdf(3.3, a, a, loc=3, scale=1), abs=1e-12)
    sg = ShiftedGamma(0.6, 0.1, 0.2)
    assert sg.logpdf(0.5) == pytest.approx(stats.gamma.logpdf(0.5, 16, loc=0.2, scale=0.025), abs=1e-12)
    assert ShiftedInvGamma1(0.6, 0.3, 0.1).logpdf(0.4) == pytest.approx(InvGamma1(0.5, 0.3).logpdf(0.3), abs=1e-14)
    assert g.support == (3, 4) and sg.support == (0.2, math.inf) and g.logpdf(4.5) == -math.inf
    rng = np.random.default_rng(1)
    draws = np.array([g.rvs(rng) for _ in range(20000)])
    assert draws.min() > 3 and draws.max() < 4 and abs(draws.mean() - 3.5) < 4 * 0.2 / math.sqrt(20000)


def test_prior_trunc_bounds_match_dynare():
    text = MOD.read_text(encoding="utf-8").replace("lik_init=1)", "lik_init=1, prior_trunc=0.01)")
    dyn = read_mod(text)
    for name, (low, high) in RECORDED["bounds_prior_trunc_0.01"].items():
        if name != "sd_y":  # Dynare's shifted inverse gamma 2 differs (above)
            assert dyn.bounds[name] == pytest.approx((low, high), abs=1e-9), name
    assert quantile(dyn.priors["sd_y"], 0.5) == pytest.approx(stats.invgamma.ppf(0.5, 5.0625, scale=0.35 * 4.0625, loc=0.05), rel=1e-9)
    support = read_mod(text.replace("prior_trunc=0.01", "prior_trunc=0"))
    assert support.bounds["rho"] == (-0.5, 1.0) and support.bounds["alpha"] == (0.2, math.inf)
    assert read_mod(MOD).bounds == {"corr_y_c": (-1.0, 1.0)}  # without prior_trunc, no bounds from the priors (as before)


def test_written_and_read_back():
    from pymetrix import ModelSource
    from pymetrix.models.writers import export_model

    dyn = read_mod(MOD)
    dynare = export_model(dyn, "dynare").text
    assert "rho, 0.65, , , beta_pdf, 0.6, 0.1, -0.5, 1;" in dynare and "gamma_pdf, 0.6, 0.1, 0.2;" in dynare
    written = export_model(dyn, "pymetrix").text
    assert "prior beta(0.6, 0.1, -0.5, 1)" in written
    specs = ModelSource(written).build().model.parameters
    assert specs["rho"].prior == dyn.priors["rho"] and specs["alpha"].prior == dyn.priors["alpha"]


def test_normal_bounds_and_refusals():
    dyn = read_mod(MOD.read_text(encoding="utf-8").replace("normal_pdf, 0.3, 0.1;", "normal_pdf, 0.3, 0.1, 0, 0.8;"))
    assert dyn.priors["corr_y_c"] == Normal(0.3, 0.1) and dyn.bounds["corr_y_c"] == (0.0, 0.8)
    with pytest.raises(ModelError, match="must be above low"):
        read_mod(MOD.read_text(encoding="utf-8").replace("0.1, 0.2;", "0.1, 0.7;"))


def test_estimation_keeps_parameters_inside_the_generalised_support():
    dyn = read_mod(MOD)
    fit = dyn.estimate()
    est = fit.params.as_dict()
    assert -0.5 < est["rho"] < 1 and est["alpha"] > 0.2 and est["sd_e"] > 0.1 and est["sd_y"] > 0.05
    assert fit.stats["logpost"] >= dyn.log_posterior() - 1e-6  # the mode is at least as good as the initial values


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
