"""OccBin beyond the filter: the smoother iterated on the smoothed path, posterior forecasts, targets held by free
shocks and shock decompositions, all respecting the model's occasionally binding constraints.

Checked independently:
- the iterated smoother (``occbin_filter(..., smoother="iterated")``, as Dynare 6's OccBin smoother): when nothing
  binds it is the linear Kalman smoother (1e-10 on exact data, 1e-8 with measurement errors); on data simulated by
  OccBin, observed without error, it gives the true path and regimes; with measurement errors, over several samples
  its regimes agree with the true ones at least as often as the filter's, and its smoothed path satisfies the model's
  own policy equation with its max (evaluated in levels, period by period) to 1e-8, which the filter's path breaks;
- posterior forecasts with OccBin: each draw's path is OccBin's own simulation (``occbin``, anticipated shocks) from
  the state the OccBin filter gives at that draw's parameters (recomputed independently for each draw, 1e-10), and
  the floor holds in every draw, where the linear posterior forecast breaks it;
- targets held by free shocks under OccBin: the nonlinear perfect-foresight solution (stacked Newton on the model
  with its max) with the same holds (1e-8); with holds in some periods only, the perfect-foresight solution with the
  shocks found reproduces the path and the targets (1e-8); a target the free shock cannot reach is refused;
- the OccBin shock decomposition: the parts add up to the smoothed path (1e-10); with a constraint that never binds
  it is the linear decomposition (1e-10) and the binding part is zero."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, PerfectForesight, Scenario, occbin, occbin_filter, templates
from pymetrix.expr.evaluate import evaluate
from pymetrix.models import MeasurementMap, ModelError
from pymetrix.models.occbin_filter import BINDING, occbin_forecast

DIRECT = MeasurementMap.direct(["y", "pi", "i"])
NOISY = MeasurementMap.direct(["y", "pi", "i"], {"y": 0.02, "pi": 0.02, "i": 0.02})
SPAN = ("2025Q1", "2027Q4")
RECESSION = Scenario(set={"ey": {"2025Q1": -3.0, "2025Q2": -3.0}})


def _simulate(template, n, seed, noise=0.0, extra=None, start="1980Q1"):
    rng = np.random.default_rng(seed)
    shocks = {s: {t + 1: float(v) for t, v in enumerate(rng.normal(0, template.shock_sd[s], n))} for s in template.shocks}
    for name, values in (extra or {}).items():
        shocks[name].update(values)
    path = occbin(template, shocks, periods=n)
    frame = path.frame[["y", "pi", "i"]].copy()
    frame.index = pd.period_range(start, periods=n, freq="Q", name="time")
    if noise:
        frame += rng.normal(0, noise, frame.shape)
    return path, Dataset.from_frame(frame)


def _policy_residual(template, frame):
    """The model's own policy equation, i = max(rule, floor), evaluated in levels on a path with its shocks."""
    eq = next(e for e in template.model.model_equations if e.name == "policy")
    return np.array([float(evaluate(eq.residual, lambda name, lag, t=t: frame[name].iloc[t + lag])) for t in range(1, len(frame))])


# the iterated smoother

def test_the_iterated_smoother_with_nothing_binding_is_the_linear_smoother():
    far = templates.new_keynesian(lower_bound=-100.0)
    system = far.solve().system(DIRECT, far.shock_sd)
    _, exact = _simulate(templates.new_keynesian(lower_bound=0.0), 80, 1, extra={"ey": {t: -10.0 for t in range(30, 36)}})
    result = occbin_filter(far, exact, DIRECT, smoother="iterated")
    np.testing.assert_allclose(result.smoothed.frame[["y", "pi", "i"]].to_numpy(), system.filter(exact).frame[["y", "pi", "i"]].to_numpy(),
                               atol=1e-10)
    assert result.smoother_iterations == 1 and not result.regimes.to_numpy().any()
    _, noisy = _simulate(far, 100, 2, noise=0.02)
    result = occbin_filter(far, noisy, NOISY, smoother="iterated")
    linear = far.solve().system(NOISY, far.shock_sd).filter(noisy)
    np.testing.assert_allclose(result.smoothed.frame[["y", "pi", "i"]].to_numpy(), linear.frame[["y", "pi", "i"]].to_numpy(), atol=1e-8)


def test_observed_without_error_the_iterated_smoother_recovers_the_truth():
    zlb = templates.new_keynesian(lower_bound=0.0)
    path, data = _simulate(zlb, 80, 1, extra={"ey": {t: -10.0 for t in range(30, 36)}, "epi": {50: -6.0}})
    result = occbin_filter(zlb, data, DIRECT, smoother="iterated")
    np.testing.assert_allclose(result.smoothed.frame[["y", "pi", "i"]].to_numpy(), path.frame[["y", "pi", "i"]].to_numpy(), atol=1e-10)
    np.testing.assert_array_equal(result.regimes.to_numpy(), path.binding.to_numpy())
    np.testing.assert_array_equal(result.ex_post.to_numpy(), path.binding.to_numpy())


def test_the_iterated_regimes_respect_the_constraint_on_the_smoothed_path():
    zlb = templates.new_keynesian(lower_bound=2.0)  # the floor binds now and then under the model's own shocks
    agree_filter = agree_iterated = 0
    broken_by_filter = 0
    for seed in range(1, 9):
        noise = (0.02, 0.1, 0.2, 0.3)[seed % 4]
        path, data = _simulate(zlb, 120, seed, noise)
        measurement = MeasurementMap.direct(["y", "pi", "i"], {v: noise for v in ("y", "pi", "i")})
        filtered = occbin_filter(zlb, data, measurement)
        iterated = occbin_filter(zlb, data, measurement, smoother="iterated")
        truth = path.binding.to_numpy()[:, 0]
        agree_filter += int((filtered.binding.to_numpy()[:, 0] == truth).sum())
        agree_iterated += int((iterated.regimes.to_numpy()[:, 0] == truth).sum())
        # the smoothed path satisfies i = max(rule, floor), with its smoothed shocks, in every period
        assert np.max(np.abs(_policy_residual(zlb, iterated.smoothed.frame))) < 1e-8
        np.testing.assert_array_equal(iterated.ex_post.to_numpy(), iterated.regimes.to_numpy())
        assert not iterated.smoother_cycled and iterated.smoother_kept == 0
        assert iterated.loglike == filtered.loglike  # the likelihood is the filter's
        np.testing.assert_array_equal(iterated.binding.to_numpy(), filtered.binding.to_numpy())
        broken_by_filter += int(np.max(np.abs(_policy_residual(zlb, filtered.smoothed.frame))) > 1e-3)
    assert agree_iterated >= agree_filter
    assert broken_by_filter >= 4  # the filter's regimes, kept by its smoother, are contradicted by the smoothed path


def test_the_iterated_smoother_reports_and_refuses():
    zlb = templates.new_keynesian(lower_bound=0.0)
    _, data = _simulate(zlb, 80, 1, extra={"ey": {t: -10.0 for t in range(30, 36)}, "epi": {50: -6.0}})
    # a rate below its floor observed without error: binding there makes the data impossible, so that period keeps
    # the filter's regime and differs ex post
    below = Dataset.from_frame(data.frame.assign(i=data.frame["i"].where(data.frame.index != data.frame.index[60], -2.0)))
    result = occbin_filter(zlb, below, DIRECT, smoother="iterated")
    assert result.smoother_kept == 1 and not result.regimes.iloc[60, 0] and result.ex_post.iloc[60, 0]
    text = str(result.summary())
    assert "periods binding, smoothed" in text and "the filter's were kept" in text
    with pytest.raises(ModelError, match="'filter'.*'iterated'"):
        occbin_filter(zlb, data, DIRECT, smoother="smoothed")


def test_the_dialog_iterates_the_smoother():
    from pymetrix.commands import Session
    from pymetrix.commands.occbin import OccBinFilterCommand
    from pymetrix.views import sections

    zlb = templates.new_keynesian(lower_bound=2.0)
    path, _ = _simulate(zlb, 100, 3)
    levels = path.frame.copy()
    levels.index = pd.period_range("1980Q1", periods=len(levels), freq="Q", name="time")
    s = Session()
    s.objects["zlb"], s.objects["data"] = zlb, zlb.measurement.apply(Dataset.from_frame(levels))
    s.run(OccBinFilterCommand("zlb", "data", name="ob", iterated=True))
    assert s["ob"].smoother == "iterated"
    assert "ob = occbin_filter(zlb, data, smoother='iterated')" in s.script()
    binding = dict(sections(s["ob"]))["Binding"]
    assert "(smoothed)" in str(binding)


# posterior forecasts

def _data(template, seed=1, n=100):
    _, data = _simulate(template, n, seed, 0.02, start="2000Q1")
    return data


@pytest.fixture(scope="module")
def posterior():
    from pymetrix.estimation.priors import Beta

    zlb = templates.new_keynesian(lower_bound=2.0)
    data = _data(zlb)
    fit = zlb.estimate(data, ["rho"], priors={"rho": Beta(0.7, 0.1)}, measurement=NOISY, occbin=True)
    return zlb, data, fit, fit.mcmc(draws=40, chains=1, seed=3)


def test_each_posterior_draw_is_occbins_forecast_at_its_parameters(posterior):
    zlb, data, fit, post = posterior
    result = post.forecast(SPAN, RECESSION, draws=4, shocks=False)
    assert "OccBin" in result.simulation.info["method"]
    sims = result.simulation
    for values, drawn in zip(post._subsample(4), sims.values):
        model = zlb.model.with_parameters(values)
        smoothed = occbin_filter(model, data, NOISY, zlb.shock_sd, shocks=list(zlb.shocks)).smoothed.frame
        start = {v: float(smoothed[v].iloc[-1]) for v in ("y", "pi", "i")}
        own = occbin(model, {"ey": {1: -3.0, 2: -3.0}}, periods=12, anticipated=True, initial=start, shock_names=list(zlb.shocks))
        np.testing.assert_allclose(drawn[:, [sims.names.index(v) for v in ("y", "pi", "i")]], own.frame[["y", "pi", "i"]].to_numpy(),
                                   atol=1e-10)


def test_the_floor_holds_in_every_posterior_draw(posterior):
    zlb, _, fit, post = posterior
    floor = fit.model.parameters["i_floor"].value
    with_shocks = post.forecast(SPAN, RECESSION, draws=10, seed=5)
    i = with_shocks.simulation.names.index("i")
    assert with_shocks.simulation.info["failed"] == 0 and with_shocks.simulation.values.shape[0] == 10
    assert np.min(with_shocks.simulation.values[:, :, i]) >= floor - 1e-10
    assert np.min(with_shocks.simulation.values[:, :, i]) <= floor + 1e-10  # the floor binds in some
    linear = post.forecast(SPAN, RECESSION, draws=10, seed=5, occbin=False)
    assert np.min(linear.simulation.values[:, :, i]) < floor - 0.05
    assert "OccBin" not in linear.simulation.info["method"]


def test_the_posterior_dialog_takes_occbin(posterior):
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import ForecastModel

    _, _, _, post = posterior
    s = Session()
    s.objects["post"] = post
    command = ForecastModel("post", "2025Q1", "2027Q4", set_to="ey[2025Q1:2025Q2]=-3", draws=3, shocks=False, occbin=True, name="fc")
    assert command.validate(s) == []
    s.run(command)
    assert "occbin=True" in s.script() and "OccBin" in s["fc"].simulation.info["method"]


# targets held by free shocks

@pytest.fixture(scope="module")
def fitted():
    zlb = templates.new_keynesian(lower_bound=2.0)
    return zlb.estimate(_data(zlb), ["rho"], measurement=NOISY, occbin=True)


def _shells(fit, last):
    history = fit.smoothed.frame[["y", "pi", "i"]]
    full = pd.period_range(history.index[0], last, freq="Q", name="time")
    return Dataset.from_frame(history.reindex(full).assign(ey=0.0, epi=0.0, ei=0.0))


def test_holds_over_the_span_are_the_perfect_foresight_solution(fitted):
    periods = pd.period_range("2025Q1", "2049Q4", freq="Q")
    hold = Scenario(hold={"y": {str(p): (-2.0 if k < 6 else 0.0) for k, p in enumerate(periods)}}, free=["ey"])
    paths, binding, _, shocks = occbin_forecast(fitted.model, list(fitted.shock_sd), {}, fitted.smoothed, ("2025Q1", "2049Q4"), hold,
                                                horizon=200)
    assert binding.to_numpy().sum() >= 3  # the floor binds while output is held down
    exact = fitted.model.solve(_shells(fitted, "2049Q4"), ("2025Q1", "2049Q4"), hold, solver=PerfectForesight(terminal="steady_state"))
    np.testing.assert_allclose(paths[["y", "pi", "i"]].to_numpy(), exact.frame.loc["2025Q1":, ["y", "pi", "i"]].to_numpy(), atol=1e-8)
    np.testing.assert_allclose(shocks["ey"].to_numpy(), exact.frame.loc["2025Q1":, "ey"].to_numpy(), atol=1e-8)


def test_holds_in_some_periods(fitted):
    floor = fitted.model.parameters["i_floor"].value
    hold = Scenario(hold={"y": {"2025Q1": -2.0, "2025Q2": -2.0, "2025Q3": -1.0}}, free=["ey"])
    forecast = fitted.forecast(SPAN, hold)
    assert forecast.provenance.method["type"] == "piecewise-linear (OccBin)" and "targets" in forecast.provenance.method
    assert forecast.provenance.method["binding"]["policy (max)"]
    np.testing.assert_allclose(forecast["y"].loc["2025Q1":"2025Q3"].to_numpy(), [-2.0, -2.0, -1.0], atol=1e-10)
    assert forecast["i"].loc["2025Q1":].min() >= floor - 1e-10
    found = {"ey": {str(p): float(v) for p, v in forecast.frame.loc["2025Q1":"2027Q4", "ey"].items()}}
    exact = fitted.model.solve(_shells(fitted, "2040Q4"), ("2025Q1", "2040Q4"), Scenario(set=found),
                               solver=PerfectForesight(terminal="steady_state"))
    np.testing.assert_allclose(exact.frame.loc["2025Q1":"2027Q4", ["y", "pi", "i"]].to_numpy(),
                               forecast.frame.loc["2025Q1":"2027Q4", ["y", "pi", "i"]].to_numpy(), atol=1e-8)
    linear = fitted.forecast(SPAN, hold, occbin=False)
    assert linear["i"].loc["2025Q1":].min() < floor - 0.05


def test_holds_refused(fitted):
    with pytest.raises(ModelError, match="cannot hold the targets"):  # below the floor, which the policy shock cannot move
        fitted.forecast(SPAN, Scenario(hold={"i": {"2025Q1": -1.0}}, free=["ei"]))
    with pytest.raises(ModelError, match="anticipated shocks only"):
        fitted.forecast(SPAN, Scenario(hold={"y": {"2025Q1": -1.0}}, free=["ey"], unanticipated=["ey"]))
    with pytest.raises(ModelError, match="no draws"):
        fitted.forecast(SPAN, Scenario(hold={"y": {"2025Q1": -1.0}}, free=["ey"]), draws=5)
    with pytest.raises(ModelError, match="pair each hold"):
        fitted.forecast(SPAN, Scenario(hold={"y": {"2025Q1": -1.0}}))


# shock decompositions

def test_the_decomposition_adds_up_and_is_linear_when_nothing_binds():
    zlb = templates.new_keynesian(lower_bound=2.0)
    _, data = _simulate(zlb, 120, 1, 0.02)
    for smoother in ("filter", "iterated"):
        result = occbin_filter(zlb, data, NOISY, smoother=smoother)
        d = result.decompose()
        assert np.max(np.abs(d.residual.to_numpy())) < 1e-10
        np.testing.assert_allclose(d.deviations.to_numpy(), result.smoothed.frame[["y", "pi", "i"]].to_numpy(), atol=1e-10)  # steady state 0
        binds = result.regimes.to_numpy()[:, 0]
        assert np.abs(d.contributions["i"][BINDING].to_numpy()[binds]).max() > 0.1
        assert "regimes held" in d.title
    far = templates.new_keynesian(lower_bound=-100.0)
    result = occbin_filter(far, data, NOISY, smoother="iterated")
    d, linear = result.decompose(), far.solve().decompose(result.smoothed)
    for v in ("y", "pi", "i"):
        parts = d.contributions[v]
        assert np.max(np.abs(parts[BINDING].to_numpy())) == 0.0
        np.testing.assert_allclose(parts[list(linear.contributions[v].columns)].to_numpy(), linear.contributions[v].to_numpy(), atol=1e-10)


def test_a_fitted_models_decomposition_follows_its_filter(fitted):
    d = fitted.decompose()
    assert "OccBin" in d.title and BINDING in d.contributions["i"].columns
    assert np.max(np.abs(d.residual.to_numpy())) < 1e-10
    assert BINDING not in fitted.decompose(occbin=False).contributions["i"].columns
