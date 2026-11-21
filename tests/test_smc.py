"""Sequential Monte Carlo (Herbst and Schorfheide 2014, as Dynare 6's): FittedModel.smc, a Posterior.

Checked independently: an AR(1) coefficient's posterior and the log marginal data density against the exact ones on
a grid; a posterior with two modes (a coefficient that enters squared) found with both, where a Metropolis-Hastings
chain stays in one; the New Keynesian template's posterior and marginal density against Metropolis-Hastings's and the
Laplace approximation; the same draws in one process or several."""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from pymetrix import Dataset, Equation, Model, templates
from pymetrix.errors import SpecificationError
from pymetrix.estimation.priors import Beta, Gamma, Normal
from pymetrix.metadata import Role
from pymetrix.models import ModelError, Posterior


def _ar1(coefficient, seed, n, form="rho*x[-1]", prior=Normal(0.5, 0.2), start=0.3):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = coefficient * x[t - 1] + 0.5 * rng.normal()
    data = Dataset.from_frame(pd.DataFrame({"x": x}, index=pd.period_range("1950Q1", periods=n, freq="Q", name="time")))
    name = "rho" if "rho" in form else "a"
    model = Model([Equation(f"x ~ {form} + e", params=[name])], params={name: start}, roles={"e": Role.SHOCK})
    return model.estimate(data, ["x"], estimate=[name], shock_sd={"e": 0.5}, priors={name: prior})


@pytest.fixture(scope="module")
def ar1():
    fit = _ar1(0.7, 1, 120)
    return fit, fit.smc(particles=1000, seed=1)


def test_the_posterior_and_marginal_density_match_the_exact_ones(ar1):
    fit, post = ar1
    # 2001 points: the moments and the log density move by under 1e-8 from 8001 points (and from 1001), against
    # tolerances of 0.01 and more below; each point is a Kalman filter, so the grid was a third of the test's time
    g = np.linspace(-1.5, 2.5, 2001)
    lp = np.array([fit.log_posterior([v]) for v in g])
    top = lp[np.isfinite(lp)].max()
    w = np.exp(np.where(np.isfinite(lp), lp - top, -np.inf))
    z = integrate.trapezoid(w, g)
    mean = integrate.trapezoid(w * g, g) / z
    sd = np.sqrt(integrate.trapezoid(w * (g - mean) ** 2, g) / z)
    draws = post["rho"]
    assert abs(draws.mean() - mean) < 4 * sd / np.sqrt(post.smc["ess"][-1])
    assert draws.std() == pytest.approx(sd, rel=0.08)
    # the share of prior draws giving an explosive model counts: the exact density integrates over the same region
    assert post.smc["prior_share"] < 1
    assert post.marginal_likelihood() == pytest.approx(top + np.log(z), abs=0.08)
    assert post.marginal_likelihood(method="mhm") == pytest.approx(top + np.log(z), abs=0.15)


def test_two_modes_are_both_found():
    fit = _ar1(0.8, 2, 150, form="a*a*x[-1]", prior=Normal(0.0, 1.0), start=0.7)  # a and -a fit the same
    post = fit.smc(particles=1000, seed=1)
    a = post["a"]
    assert 0.3 < (a > 0).mean() < 0.7
    assert a[a > 0].mean() == pytest.approx(-a[a < 0].mean(), abs=0.03)
    chain = fit.mcmc(draws=1000, chains=1, seed=1, modes=False)
    assert (chain["a"] > 0).mean() in (0.0, 1.0)  # a Metropolis-Hastings chain from the mode stays there


def test_the_new_keynesian_posterior_agrees_with_metropolis_hastings():
    nk = templates.new_keynesian()
    truth = nk.solve().simulate(120, nk.shock_sd, seed=11)
    truth.index = pd.period_range("1990Q1", periods=120, freq="Q", name="time")
    fit = nk.estimate(nk.measurement.apply(Dataset.from_frame(truth)), ["kappa", "rho"],
                      priors={"kappa": Gamma(0.3, 0.1), "rho": Beta(0.7, 0.1)})
    smc = fit.smc(particles=600, seed=2)
    mh = fit.mcmc(draws=2500, chains=2, seed=2)
    for name in ("kappa", "rho"):
        assert smc[name].mean() == pytest.approx(mh[name].mean(), abs=0.5 * mh[name].std())
        assert smc[name].std() == pytest.approx(mh[name].std(), rel=0.25)
    assert smc.marginal_likelihood() == pytest.approx(fit.laplace() if hasattr(fit, "laplace") else smc.laplace(), abs=0.3)
    assert len(smc.smc["phi"]) > 2 and smc.smc["phi"][-1] == 1.0 and np.all(np.diff(smc.smc["phi"]) > 0)


def test_reproducible_in_any_number_of_processes(ar1):
    fit, _ = ar1
    here = fit.smc(particles=200, seed=7, workers=1)
    again = fit.smc(particles=200, seed=7, workers=1)
    elsewhere = fit.smc(particles=200, seed=7, workers=2)
    np.testing.assert_array_equal(here.draws, again.draws)
    np.testing.assert_array_equal(here.draws, elsewhere.draws)


def test_results_and_refusals(ar1):
    fit, post = ar1
    assert isinstance(post, Posterior) and post.sampler == "smc" and post.draws.shape == (1, 1000, 1)
    table = str(post.summary())
    assert "Sequential Monte Carlo" in table and "(sequential Monte Carlo)" in table
    assert post.provenance.method["sampler"].startswith("sequential Monte Carlo")
    irf = post.irf(6, draws=40)
    assert np.all(irf.lower <= irf.upper + 1e-12)
    with pytest.raises(SpecificationError, match="particles"):
        fit.smc(particles=10)
    mh = fit.mcmc(draws=200, chains=1, seed=1)
    with pytest.raises(SpecificationError, match="fit.smc"):
        mh.marginal_likelihood(method="smc")
    plain = _ar1(0.7, 1, 60, prior=Normal(0.5, 0.2))
    from dataclasses import replace

    with pytest.raises(ModelError, match="give every estimated parameter a prior"):
        replace(plain, priors={}).smc(particles=100)


def test_the_dialog(ar1):
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import RunSMC

    fit, _ = ar1
    s = Session()
    s.objects["estimated"] = fit
    s.run(RunSMC("estimated", particles=300, seed=3, name="post"))
    assert s["post"].sampler == "smc" and "post = estimated.smc(particles=300, seed=3)" in s.script()


def test_ess_fraction_was_target(ar1):
    """``ess_fraction`` is the new name of ``target``: the same draws for the same seed; the old keyword warns, both
    at once are refused; a higher share takes more, smaller tempering steps."""
    import warnings

    from pymetrix.models.bayes import smc

    fit, _ = ar1
    new = fit.smc(particles=200, seed=5, ess_fraction=0.8, workers=1)
    with pytest.warns(DeprecationWarning, match=r"'target' of FittedModel\.smc\(\) is now called 'ess_fraction'"):
        old = fit.smc(particles=200, seed=5, target=0.8, workers=1)
    np.testing.assert_array_equal(new.draws, old.draws)
    assert new.smc["phi"] == old.smc["phi"] and new.marginal_likelihood() == old.marginal_likelihood()
    with pytest.warns(DeprecationWarning, match=r"'target' of smc\(\) is now called 'ess_fraction'"):
        np.testing.assert_array_equal(smc(fit, 200, seed=5, target=0.8, workers=1).draws, new.draws)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_array_equal(fit.smc(200, 2, 0.8, 5, workers=1).draws, new.draws)  # by position, as before
    with pytest.raises(SpecificationError, match="both 'target' and 'ess_fraction'"):
        fit.smc(particles=200, target=0.5, ess_fraction=0.5)
    with pytest.raises(SpecificationError, match="both 'target' and 'ess_fraction'"):
        smc(fit, 200, target=0.5, ess_fraction=0.5)
    with pytest.raises(SpecificationError, match="ess_fraction a share between 0 and 1"):
        fit.smc(particles=200, ess_fraction=1.5)
    fine = fit.smc(particles=200, seed=5, ess_fraction=0.95, workers=1)
    coarse = fit.smc(particles=200, seed=5, ess_fraction=0.2, workers=1)
    assert len(fine.smc["phi"]) > len(coarse.smc["phi"])


def test_steps_and_calls_saved_with_target_replay(ar1):
    """A schema-1 Sequential Monte Carlo step (and a version-1 recorded smc call) saved with ``target`` opens with
    ``ess_fraction`` and replays to the same draws as the API, as does its generated script."""
    import warnings

    from pymetrix.calls import Call
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import RunSMC
    from pymetrix.commands.registry import from_json, to_json

    fit, _ = ar1
    direct = fit.smc(particles=200, seed=4, ess_fraction=0.7, workers=1)
    step = from_json({"kind": "command", "id": "pymetrix:run_smc", "version": 1,
                      "fields": {"source": "estimated", "particles": 200, "seed": 4, "name": "post", "target": 0.7,
                                 "workers": 1}})
    assert isinstance(step, RunSMC) and step.ess_fraction == 0.7 and not hasattr(step, "target")
    assert to_json(step)["version"] == 2 and "target" not in to_json(step)["fields"]
    s = Session()
    s.objects["estimated"] = fit
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        s.run(step)
        code = "post = estimated.smc(particles=200, ess_fraction=0.7, seed=4, workers=1)"
        assert code in s.script()
        np.testing.assert_array_equal(s["post"].draws, direct.draws)
        call = step.to_call(s)
        assert call.version == 2 and call.arguments["ess_fraction"] == 0.7 and "target" not in call.arguments
        saved = dict(call.arguments)  # the same call saved at version 1, with target
        saved["target"] = saved.pop("ess_fraction")
        s.run(Call("pymetrix:fitted_model.smc", 1, call.receiver, saved, output="older"))
        np.testing.assert_array_equal(s["older"].draws, direct.draws)
        replay = Session()
        replay.objects["estimated"] = fit
        replay.run_code(code)
        np.testing.assert_array_equal(replay["post"].draws, direct.draws)


def test_the_dialog_shows_effective_sample_fraction(ar1):
    import os

    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import RunSMC
    from pymetrix.forms import form_spec
    from pymetrix.ui.qt import command_dialog

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert "target" not in {f.name for f in form_spec(RunSMC)}
    fit, _ = ar1
    s = Session()
    s.objects["estimated"] = fit
    chosen = RunSMC("estimated", particles=200, seed=4, name="post", ess_fraction=0.7, workers=1)
    dialog = command_dialog(RunSMC, s, initial=chosen)
    try:
        assert "target" not in dialog.editors and dialog.editors["ess_fraction"].get() == pytest.approx(0.7)
        assert dialog.on_advanced_tab("ess_fraction")
        assert "Effective sample fraction" in dialog.row_label("ess_fraction").text().replace("&", "")
        dialog.findChild(QtWidgets.QDialogButtonBox).accepted.emit()
        assert dialog.command == chosen
        np.testing.assert_array_equal(s["post"].draws, fit.smc(particles=200, seed=4, ess_fraction=0.7, workers=1).draws)
    finally:
        dialog.close()
        dialog.deleteLater()
