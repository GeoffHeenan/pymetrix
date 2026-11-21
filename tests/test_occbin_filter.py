"""The OccBin filter and estimation with occasionally binding constraints (Dynare 6's piecewise Kalman filter).

Checked independently:
- a constraint that never binds gives the linear Kalman filter: the same likelihood and smoothed states;
- given the regimes it chose, its likelihood and filtered states are statsmodels' Kalman filter with the same
  time-varying matrices (statsmodels.tsa.statespace, time-varying transition, intercept and selection);
- on data simulated by OccBin and observed without error, the filtered and smoothed states, the binding periods and
  the shocks (those not muted at the bound) are the true ones;
- estimation with a constraint that never binds is the linear estimation; on data where the floor binds, the OccBin
  likelihood fits better than the linear one at the true parameters."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, occbin, occbin_filter, templates
from pymetrix.models import MeasurementMap, ModelError
from pymetrix.models.occbin import _Piecewise
from pymetrix.models.occbin_filter import piecewise_filter

DIRECT = MeasurementMap.direct(["y", "pi", "i"])
NOISY = MeasurementMap.direct(["y", "pi", "i"], {"y": 0.02, "pi": 0.02, "i": 0.02})


def _simulate(template, n, seed, extra=None, noise=0.0):
    rng = np.random.default_rng(seed)
    shocks = {s: {t + 1: float(v) for t, v in enumerate(rng.normal(0, template.shock_sd[s], n))} for s in template.shocks}
    for name, values in (extra or {}).items():
        shocks[name].update(values)
    path = occbin(template, shocks, periods=n)
    frame = path.frame[["y", "pi", "i"]].copy()
    frame.index = pd.period_range("1980Q1", periods=n, freq="Q", name="time")
    if noise:
        frame += rng.normal(0, noise, frame.shape)
    return path, Dataset.from_frame(frame)


@pytest.fixture(scope="module")
def exact():
    zlb = templates.new_keynesian(lower_bound=0.0)
    path, data = _simulate(zlb, 80, 1, {"ey": {t: -10.0 for t in range(30, 36)}, "epi": {50: -6.0}})
    return zlb, path, data


@pytest.fixture(scope="module")
def noisy():
    zlb = templates.new_keynesian(lower_bound=2.0)  # the floor binds now and then under the model's own shocks
    path, data = _simulate(zlb, 120, 1, noise=0.02)
    return zlb, path, data


def test_a_constraint_that_never_binds_is_the_linear_filter(exact):
    _, _, data = exact
    far = templates.new_keynesian(lower_bound=-100.0)
    result = occbin_filter(far, data, DIRECT)
    system = far.solve().system(DIRECT, far.shock_sd)
    assert result.loglike == pytest.approx(system.loglike(data), abs=1e-9)
    np.testing.assert_allclose(result.smoothed.frame[["y", "pi", "i"]].to_numpy(), system.filter(data).frame[["y", "pi", "i"]].to_numpy(),
                               atol=1e-10)
    assert not result.binding.to_numpy().any()


def test_observed_without_error_it_recovers_the_truth(exact):
    zlb, path, data = exact
    result = occbin_filter(zlb, data, DIRECT)
    truth = path.frame[["y", "pi", "i"]].to_numpy()
    np.testing.assert_allclose(result.filtered[["y", "pi", "i"]].to_numpy(), truth, atol=1e-10)
    np.testing.assert_allclose(result.smoothed.frame[["y", "pi", "i"]].to_numpy(), truth, atol=1e-10)
    binds = path.binding.to_numpy()[:, 0]
    assert binds.sum() == 7
    np.testing.assert_array_equal(result.binding.to_numpy()[:, 0], binds)
    np.testing.assert_array_equal(result.ex_post.to_numpy()[:, 0], binds)
    shocks = result.smoothed.frame[["ey", "epi", "ei"]].to_numpy()[1:]  # the first: the pre-sample state is estimated
    true = path.shocks[["ey", "epi", "ei"]].to_numpy()[1:]
    np.testing.assert_allclose(shocks[:, :2], true[:, :2], atol=1e-9)
    slack = ~binds[1:]
    np.testing.assert_allclose(shocks[slack, 2], true[slack, 2], atol=1e-9)  # the policy shock does nothing at the floor


def test_the_likelihood_is_statsmodels_kalman_filter_given_the_regimes(noisy):
    from statsmodels.tsa.statespace.kalman_filter import KalmanFilter

    zlb, _, data = noisy
    pw = _Piecewise(zlb.model, tuple(zlb.shocks), {}, 40)
    pw.tie = True
    system = pw.solution.system(NOISY, zlb.shock_sd)
    y, _ = system.observations(data)
    run = piecewise_filter(pw, system, y)
    assert run["binds"].any()
    # statsmodels: alpha(t+1) = c(t) + T(t) alpha(t) + R(t) eta(t); the filter's x(t) = c(t) + T(t) x(t-1) + R(t) e(t)
    kf = KalmanFilter(k_endog=3, k_states=system.T.shape[0], k_posdef=3)
    kf.bind(np.asarray(y - system.d, dtype=float))
    T, c, R = np.moveaxis(run["T"], 0, -1), run["c"].T, np.moveaxis(run["R"], 0, -1)
    kf["design"], kf["obs_cov"], kf["state_cov"] = system.Z, system.H, system.Q
    kf["transition"] = np.concatenate([T[:, :, 1:], T[:, :, -1:]], axis=2)
    kf["state_intercept"] = np.concatenate([c[:, 1:], c[:, -1:]], axis=1)
    kf["selection"] = np.concatenate([R[:, :, 1:], R[:, :, -1:]], axis=2)
    kf.initialize_known(run["c"][0] + run["T"][0] @ run["a0"],
                        run["T"][0] @ run["P0"] @ run["T"][0].T + run["R"][0] @ system.Q @ run["R"][0].T)
    assert run["loglike"] == pytest.approx(kf.loglike(), abs=1e-9)
    np.testing.assert_allclose(kf.filter().filtered_state.T, run["a_filt"], atol=1e-10)


def test_the_floor_fits_data_that_hit_it_better_than_the_linear_model(noisy):
    zlb, path, data = noisy
    assert path.binding.to_numpy().sum() >= 5
    with_floor = occbin_filter(zlb, data, NOISY).loglike
    linear = zlb.solve().system(NOISY, zlb.shock_sd).loglike(data)
    assert with_floor > linear + 5


def test_estimation(noisy):
    zlb, _, data = noisy
    far = templates.new_keynesian(lower_bound=-100.0)
    plain = far.estimate(data, ["rho"], measurement=NOISY)
    via_filter = far.estimate(data, ["rho"], measurement=NOISY, occbin=True)  # never binds: the same estimate
    assert via_filter.params.as_dict()["rho"] == pytest.approx(plain.params.as_dict()["rho"], abs=1e-5)
    assert via_filter.stats["loglike"] == pytest.approx(plain.stats["loglike"], abs=1e-6)
    fit = zlb.estimate(data, ["rho"], measurement=NOISY, occbin=True)
    assert "OccBin" in fit.method and fit.provenance.method["filter"].startswith("OccBin")
    assert abs(fit.params.as_dict()["rho"] - 0.8) < 3 * fit.params.se[0] + 0.02
    assert fit.stats["loglike"] >= occbin_filter(zlb, data, NOISY).loglike - 1e-6  # the maximum is at least the truth's
    assert np.isfinite(fit.log_posterior(np.array([0.8])))  # what Metropolis-Hastings samples
    from pymetrix.estimation.priors import Beta

    bayes = zlb.estimate(data, ["rho"], priors={"rho": Beta(0.7, 0.1)}, measurement=NOISY, occbin=True)
    post = bayes.mcmc(draws=20, chains=1, seed=3)
    assert post.draws.shape == (1, 20, 1) and np.all(np.isfinite(post.draws))


def test_errors(exact):
    zlb, path, data = exact
    from pymetrix import Equation, Model
    from pymetrix.metadata import Role

    plain = Model([Equation("x ~ a*x[-1] + e", params=["a"])], params={"a": 0.5}, roles={"e": Role.SHOCK})
    with pytest.raises(ModelError, match="no max or min"):
        occbin_filter(plain, Dataset.from_frame(pd.DataFrame({"x": np.zeros(10)}, index=pd.period_range("2000Q1", periods=10, freq="Q"))),
                      ["x"])
    # a rate below its floor, observed without error: only the slack regime (with a large policy shock) can give it, and
    # the smoothed path then breaks the constraint, which the summary reports as a period differing ex post
    below = Dataset.from_frame(data.frame.assign(i=data.frame["i"].where(data.frame.index != data.frame.index[60], -2.0)))
    result = occbin_filter(zlb, below, DIRECT)
    assert not result.binding.iloc[60, 0] and result.ex_post.iloc[60, 0]
    assert "periods differing ex post" in str(result.summary())
    from pymetrix.models.occbin_filter import _pinv_logdet

    F = np.diag([1.0, 0.0])  # the second observation is fixed by the model
    Fi, logdet, rank = _pinv_logdet(F, np.array([0.5, 0.0]), 0)
    assert rank == 1 and logdet == pytest.approx(0.0) and Fi[0, 0] == pytest.approx(1.0)
    with pytest.raises(ModelError, match="give that series a measurement error"):
        _pinv_logdet(F, np.array([0.5, 0.3]), 0)  # but it is not where the model fixes it


def test_dialog_view_and_script(exact):
    from pymetrix.commands import Session
    from pymetrix.commands.occbin import OccBinFilterCommand
    from pymetrix.views import sections

    zlb, path, _ = exact
    levels = path.frame.copy()
    levels.index = pd.period_range("1980Q1", periods=len(levels), freq="Q", name="time")
    data = zlb.measurement.apply(Dataset.from_frame(levels))  # the template's own observables: inflation, rate, growth
    s = Session()
    s.objects["zlb"], s.objects["data"] = zlb, data
    s.run(OccBinFilterCommand("zlb", "data", name="ob"))
    assert s["ob"].binding.to_numpy().sum() >= 5
    assert "ob = occbin_filter(zlb, data)" in s.script()
    assert [t for t, _ in sections(s["ob"])][:1] == ["Summary"] and "Binding" in [t for t, _ in sections(s["ob"])]
    assert "OccBinFilter" in repr(s["ob"]) and "periods binding" in str(s["ob"].summary())
