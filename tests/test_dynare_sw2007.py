"""Smets and Wouters (2007) read from its Dynare file, against Dynare 6.0's own output.

The reference (tests/fixtures/dynare-sw2007.json) comes from Dynare itself, run by the dynare-reference workflow (case
sw2007): numbers only. The model, data and mode files are Johannes Pfeifer's DSGE_mod replication (GPL), so they are
not kept here: they are downloaded at the reference's pinned commit when the test runs, and the test is skipped
without a network. Checked: the steady state and first-order decision rules at the published mode, and the log
posterior (likelihood with presample=4 and lik_init=2, plus 36 priors) there and at Dynare's own mode. pymetrix's own
posterior mode, standard errors and Laplace density against Dynare's, and its posterior with both modes against Dynare's
Metropolis-Hastings, only when PYMETRIX_SLOW=1 (about ten minutes and half an hour). The analytic gradient of the
likelihood there against differences."""

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from pymetrix import Model, read_mod
from pymetrix.models.bayes import rhat

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "dynare-sw2007.json").read_text(encoding="utf-8"))
FILES = ("Smets_Wouters_2007.mod", "usmodel_data.mat", "usmodel_mode.mat")


@pytest.fixture(scope="module")
def sw(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sw2007")
    base = f"https://raw.githubusercontent.com/JohannesPfeifer/DSGE_mod/{REFERENCE['source']['commit']}/Smets_Wouters_2007"
    try:
        for name in FILES:
            with urllib.request.urlopen(f"{base}/{name}", timeout=30) as response:  # noqa: S310 - a fixed https URL
                (folder / name).write_bytes(response.read())
    except OSError as exc:
        pytest.skip(f"the Smets-Wouters files could not be downloaded: {exc}")
    return read_mod(folder / "Smets_Wouters_2007.mod")


def test_the_decision_rules_at_the_published_mode(sw):
    at = REFERENCE["at_published_mode"]
    values = dict(zip(at["param_names"], at["params"]))
    model = Model(sw.model.equations, params=sw.model.parameters.with_values({n: values[n] for n in sw.model.parameters.names}),
                  roles=sw.model.roles)
    steady = model.steady_state(guess=sw.guess or None)
    solution = model.linearize(steady, shocks=list(sw.exogenous)).solve()
    endo = at["endo_names"]
    rows = [solution.variables.index(endo[i - 1]) for i in at["order_var"]]
    cols = [solution.variables.index(endo[i - 1]) for i in at["state_var"]]
    shocks = [list(solution.shocks).index(e) for e in at["exo_names"]]
    np.testing.assert_allclose(solution.P[np.ix_(rows, cols)], at["ghx"], atol=1e-10)
    np.testing.assert_allclose(solution.Q[np.ix_(rows, shocks)], at["ghu"], atol=1e-10)
    np.testing.assert_allclose([steady.values[v] for v in endo], at["steady_state"], atol=1e-12)


def test_the_log_posterior_at_the_published_mode_and_at_dynares_mode(sw):
    at = REFERENCE["at_published_mode"]
    published = sw.mode_file()
    assert len(sw.priors) == len(sw.estimated) == 36  # every prior read (BETA_PDF in capitals, as Dynare allows)
    assert published["crhoa"] == pytest.approx(dict(zip(at["param_names"], at["params"]))["crhoa"])
    assert sw.log_posterior(published) == pytest.approx(at["log_posterior_at_mode"], abs=6e-4)  # Dynare prints 3 decimals
    mode = REFERENCE["mode"]
    assert sw.log_posterior(mode["posterior_mode"]) == pytest.approx(mode["log_posterior_at_mode"], abs=1e-6)


def test_the_analytic_gradient_at_dynares_mode(sw):
    """The Kalman-filter score (pymetrix.models.kalman_score: 36 parameters, 27 states, 230 periods from the scaled
    start) against five-point differences of the same likelihood, for a dozen parameters of every kind. The forward
    recursion it replaced overflowed here (1e30 and more after a hundred periods)."""
    from pymetrix.models.kalman_score import loglike_score

    fit = sw.estimate(max_iter=0, cov="none")
    lik = fit.log_posterior.__self__
    point = {n: float(REFERENCE["mode"]["posterior_mode"][n]) for n in lik.free}
    ll, g = loglike_score(lik, point)
    assert ll == pytest.approx(lik.loglike(point), rel=1e-10)
    names = ["sd_ea", "sd_epinf", "crhob", "crhoms", "cmaw", "csigma", "chabb", "cprobw", "crpi", "constebeta", "ctrend", "calfa"]
    for name in names:
        h = 1e-4 * abs(point[name])  # the likelihood bends sharply near the unit roots (crhow 0.98, cmaw 0.93)
        f = [loglike_score(lik, {**point, name: point[name] + k * h})[0] for k in (-2, -1, 1, 2)]
        ref = (f[0] - 8 * f[1] + 8 * f[2] - f[3]) / (12 * h)
        assert g[lik.free.index(name)] == pytest.approx(ref, rel=1e-6, abs=1e-6 * np.max(np.abs(g))), name


@pytest.mark.skipif(not os.environ.get("PYMETRIX_SLOW"), reason="about ten minutes: set PYMETRIX_SLOW=1")
def test_the_posterior_mode_from_the_files_starting_values(sw):
    """Dynare's mode from the file's own starting values (a limited-memory optimiser reached a second, lower mode),
    with Dynare's standard errors and Laplace density. Measured: the mode within 0.002 standard errors, standard
    errors within 0.5%, the Laplace density 0.013 apart (numerical Hessians)."""
    mode = REFERENCE["mode"]
    fit = sw.estimate()
    assert fit.stats["logpost"] == pytest.approx(mode["log_posterior_at_mode"], abs=1e-4)
    sd = mode["posterior_std_at_mode"]
    se = dict(zip(fit.params.names, fit.params.se))
    for name in fit.params.names:
        assert abs(fit.params[name] - mode["posterior_mode"][name]) < 0.05 * sd[name], name
        assert se[name] == pytest.approx(sd[name], rel=0.05), name
    assert fit.laplace() == pytest.approx(mode["marginal_density"]["laplace"], abs=0.05)


@pytest.mark.skipif(not os.environ.get("PYMETRIX_SLOW"), reason="about half an hour: set PYMETRIX_SLOW=1")
def test_the_posterior_with_both_modes_matches_dynares(sw):
    """The posterior has two modes of almost equal height (risk-premium persistence about 0.14 and 0.82), which
    Dynare's 2 x 200,000 draws also visit. Eight starts find both; Metropolis-Hastings jumps between them. Measured:
    every posterior mean within 0.25 of Dynare's posterior standard deviation (median 0.09), the persistence's 90%
    interval 0.07-0.80 against Dynare's HPD 0.08-0.82, modified harmonic mean 0.3 apart. The tolerances allow for
    Monte Carlo error on both sides (Dynare's chains cross between the modes rarely).

    R-hat is only checked for gross failure (below 1.3). Measured 1.03 to 1.22 at these lengths. The worst is the
    price-markup shock's persistence (crhopinf): a thin tail towards 0.87, with ctrend and cmap lower too, which a
    chain enters rarely and leaves slowly. Mode jumps cannot help, as there is no peak there. Dynare's HPD for crhopinf
    starts at 0.955, so how much weight that tail carries is open (docs/validation.md)."""
    dynare = REFERENCE["mcmc"]
    fit = sw.estimate(starts=8, seed=1)
    post = fit.mcmc(draws=40000, chains=4, burn=20000, seed=11)
    assert post.modes is not None and len(post.modes) >= 2  # the starts found the second mode
    persistence = post.modes[:, list(fit.params.names).index("crhob")]
    assert persistence.min() < 0.3 and persistence.max() > 0.6
    for name in fit.params.names:
        assert abs(post[name].mean() - dynare["posterior_mean"][name]) < 0.4 * dynare["posterior_std"][name], name
    low, high = np.percentile(post["crhob"], [5, 95])
    assert low == pytest.approx(dynare["posterior_hpdinf"]["crhob"], abs=0.06)
    assert high == pytest.approx(dynare["posterior_hpdsup"]["crhob"], abs=0.06)
    assert max(rhat(post.draws[:, :, i]) for i in range(len(fit.params.names))) < 1.3
    assert post.marginal_likelihood() == pytest.approx(dynare["marginal_density"]["modified_harmonic_mean"], abs=0.6)


def test_the_log_posterior_away_from_the_modes(sw):
    """Dynare's own log posterior (evaluate_posterior_kernel) at sixteen points: the published mode, Dynare's mode, and
    draws from pymetrix's long run, ten of them in the tail of crhopinf where pymetrix's chains put more weight than
    Dynare's run did. Measured: within 2e-8 everywhere, so the two compute the same posterior and the difference
    is in the sampling."""
    points = json.loads((Path(__file__).parent / "fixtures" / "sw2007-points.json").read_text(encoding="utf-8"))["points"]
    dynare = json.loads((Path(__file__).parent / "fixtures" / "dynare-sw2007-points.json").read_text(encoding="utf-8"))
    assert dynare["labels"] == [p["label"] for p in points]
    for point, expected in zip(points, dynare["log_posterior"]):
        assert sw.log_posterior(point["values"]) == pytest.approx(expected, abs=1e-6), point["label"]
