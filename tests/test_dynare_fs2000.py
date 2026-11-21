"""Dynare's fs2000 example estimated by pymetrix, against Dynare's own output.

The reference (tests/fixtures/dynare-fs2000.json) comes from Dynare itself, run by the dynare-reference workflow
(.github/workflows/dynare-reference.yml): numbers only. The model and data files are Dynare's (GPL), so they are not
kept here: they are downloaded from git.dynare.org at the reference's Dynare version when the test runs, and the
test is skipped without a network. The posterior mode, the log posterior there, standard errors and the Laplace
marginal density are checked on every run; the Metropolis-Hastings comparison (several minutes) only when
PYMETRIX_SLOW=1."""

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from pymetrix import read_mod

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "dynare-fs2000.json").read_text(encoding="utf-8"))
FILES = ("fs2000.mod", "fs2000_data.m")


@pytest.fixture(scope="module")
def fs2000(tmp_path_factory):
    folder = tmp_path_factory.mktemp("fs2000")
    base = f"https://git.dynare.org/Dynare/dynare/-/raw/{REFERENCE['source']['tag']}/examples"
    try:
        for name in FILES:
            with urllib.request.urlopen(f"{base}/{name}", timeout=30) as response:  # noqa: S310 - a fixed https URL
                (folder / name).write_bytes(response.read())
    except OSError as exc:
        pytest.skip(f"Dynare's fs2000 files could not be downloaded: {exc}")
    dyn = read_mod(folder / "fs2000.mod")
    return dyn, dyn.estimate()


def _reference(field: str, names: tuple[str, ...]) -> np.ndarray:
    return np.array([REFERENCE[field][n] for n in names], dtype=float)


def test_the_file_is_read_as_dynare_reads_it(fs2000):
    dyn, _ = fs2000
    assert dyn.loglinear and dyn.varobs == ("gp_obs", "gy_obs")
    assert set(dyn.estimated) == set(REFERENCE["posterior_mode"])
    assert len(dyn.data().frame) == REFERENCE["nobs"]


def test_the_posterior_mode_matches_dynares(fs2000):
    _, fit = fs2000
    names = fit.params.names
    assert fit.stats["logpost"] == pytest.approx(REFERENCE["log_posterior_at_mode"], abs=1e-3)
    mode, se = _reference("posterior_mode", names), _reference("posterior_std_at_mode", names)
    assert np.all(np.abs(fit.params.values - mode) < 0.05 * se), dict(zip(names, (fit.params.values - mode) / se))
    assert np.allclose(fit.params.se, se, rtol=0.1), dict(zip(names, fit.params.se / se))


def test_the_laplace_marginal_density_matches_dynares(fs2000):
    _, fit = fs2000
    assert fit.laplace() == pytest.approx(REFERENCE["marginal_density"]["laplace"], abs=0.05)


@pytest.mark.skipif(not os.environ.get("PYMETRIX_SLOW"), reason="several minutes: set PYMETRIX_SLOW=1")
def test_the_posterior_by_metropolis_hastings_matches_dynares(fs2000):
    _, fit = fs2000
    kept = int(REFERENCE["mh_replic"] * (1 - REFERENCE["mh_drop"]))
    post = fit.mcmc(draws=kept, chains=int(REFERENCE["mh_nblocks"]), seed=1)
    names = post.names
    sd = _reference("posterior_std", names)
    means = np.array([post[n].mean() for n in names])
    assert np.all(np.abs(means - _reference("posterior_mean", names)) < 0.15 * sd), dict(zip(names, (means - _reference("posterior_mean", names)) / sd))
    stds = np.array([post[n].std() for n in names])
    assert np.allclose(stds, sd, rtol=0.1), dict(zip(names, stds / sd))
    assert post.marginal_likelihood() == pytest.approx(REFERENCE["marginal_density"]["modified_harmonic_mean"], abs=0.3)
