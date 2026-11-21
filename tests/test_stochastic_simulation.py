"""Stochastic simulation of models, checked against exact distributions and other implementations.

- A fitted AR(1): the h-step forecast's mean and standard deviation, and its quantiles, against the exact normal.
- Two fitted equations with correlated errors (a VAR(1)): the h-step covariance against statsmodels' VAR forecast MSE.
- A nonlinear equation (a random walk in logs): the draws' quantiles against the exact lognormal ones.
- Bootstrapped errors: every one-step deviation is one of the residuals; their quantiles match the residuals'.
- Coefficient draws: the one-step variance against s^2 + x'Vx.
- A first-order solution: the h-step variance against the decision rules' sum, and the long-run standard deviations
  against the theoretical moments (the Lyapunov equation).
- An estimated model's forecast: each draw's observed series are the measurement equations applied to its paths.

Tolerances are four Monte Carlo standard errors."""

import pickle

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model, Scenario, templates
from pymetrix.commands import Session
from pymetrix.commands.builtin import ForecastModel, SolveModel
from pymetrix.errors import SpecificationError
from pymetrix.models.solvers import FirstOrder, PerfectForesight

D = 20_000


def _data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        e = rng.multivariate_normal([0, 0], [[0.25, 0.1], [0.1, 0.36]])
        y[t] = 1 + 0.6 * y[t - 1] + 0.1 * x[t - 1] + e[0]
        x[t] = 0.5 + 0.2 * y[t - 1] + 0.5 * x[t - 1] + e[1]
    idx = pd.period_range("1990Q1", periods=n, freq="Q", name="time")
    return Dataset.from_frame(pd.DataFrame({"y": y, "x": x, "level": 100 * np.exp(np.cumsum(rng.normal(0.005, 0.01, n)))}, index=idx))


SPAN = ("2020Q1", "2022Q4")  # the twelve quarters after the data


def test_ar1_matches_the_exact_forecast_distribution():
    data = _data()
    fit = Equation("y ~ y[-1]").fit(data)
    sol = Model([fit]).solve(data, SPAN, draws=D, seed=1)
    c, b = fit.params.values
    s, last = fit.stats["se_regression"], data.frame["y"].iloc[-1]
    h = np.arange(1, 13)
    mean = np.array([c * sum(b ** i for i in range(k)) + b ** k * last for k in h])
    sd = s * np.sqrt([sum(b ** (2 * i) for i in range(k)) for k in h])
    assert np.allclose(sol.mean()["y"].to_numpy(), mean, atol=4 * sd / np.sqrt(D))
    assert np.allclose(sol.sd()["y"].to_numpy(), sd, rtol=4 / np.sqrt(2 * D))
    q = sol.quantiles([0.05, 0.95])
    tail = 4 * np.sqrt(0.05 * 0.95) / (0.1031 / sd * np.sqrt(D))  # four standard errors of a 5% or 95% quantile
    assert np.allclose(q["y_q5"].to_numpy(), mean - 1.6449 * sd, atol=tail) and np.allclose(q["y_q95"].to_numpy(), mean + 1.6449 * sd, atol=tail)
    assert np.allclose(sol.solved()["y"].to_numpy(), mean)  # the deterministic path: a linear model's mean
    assert sol.provenance.method["stochastic"]["draws"] == D and sol.simulation.info["failed"] == 0


def test_correlated_equations_match_statsmodels_var_mse():
    from statsmodels.tsa.api import VAR

    data = _data()
    fy, fx = Equation("y ~ y[-1] + x[-1]").fit(data), Equation("x ~ y[-1] + x[-1]").fit(data)
    sol = Model([fy, fx]).solve(data, SPAN, draws=D, seed=2)
    var = VAR(data.frame[["y", "x"]].to_numpy()).fit(1)
    mse = var.mse(12)  # sigma_u with n - k, as each equation's standard error here
    draws = sol.simulation.values  # draws x periods x (y, x)
    for h in (0, 3, 11):
        cov = np.cov(draws[:, h, :].T)
        assert np.allclose(cov, mse[h], rtol=0.05, atol=0.01), h
    assert np.allclose(sol.solved().frame[["y", "x"]].to_numpy(), var.forecast(data.frame[["y", "x"]].to_numpy()[-1:], 12), atol=1e-8)


def test_a_nonlinear_equation_matches_the_lognormal():
    data = _data()
    fit = Equation("dlog(level) ~ 1").fit(data)
    sol = Model([fit]).solve(data, SPAN, draws=D, seed=3)
    mu, s, last = fit.params.values[0], fit.stats["se_regression"], data.frame["level"].iloc[-1]
    h = np.arange(1, 13)
    for q, z in ((0.05, -1.6449), (0.5, 0.0), (0.95, 1.6449)):
        exact = last * np.exp(h * mu + z * s * np.sqrt(h))
        assert np.allclose(sol.quantiles([q])[f"level_{Solution_label(q)}"].to_numpy(), exact, rtol=0.003), q


def Solution_label(q):  # noqa: N802 - the column suffix quantiles uses
    from pymetrix.models.solution import Solution

    return Solution._label(q)


def test_bootstrap_draws_only_the_residuals():
    data = _data()
    fit = Equation("y ~ y[-1]").fit(data)
    sol = Model([fit]).solve(data, SPAN, draws=5000, seed=4, errors="bootstrap")
    first = sol.simulation.values[:, 0, 0] - sol.solved()["y"].to_numpy()[0]
    resid = fit.resid.dropna().to_numpy()
    assert np.allclose(np.min(np.abs(first[:, None] - resid[None, :]), axis=1), 0, atol=1e-9)  # each draw a residual
    assert np.allclose(np.quantile(first, [0.1, 0.5, 0.9]), np.quantile(resid, [0.1, 0.5, 0.9]), atol=0.06)
    assert sol.simulation.info["errors"] == "bootstrap" and sol.simulation.info["periods resampled"] == len(resid)


def test_coefficient_draws_add_parameter_uncertainty():
    data = _data()
    fit = Equation("y ~ y[-1]").fit(data)
    sol = Model([fit]).solve(data, SPAN, draws=D, seed=5, coefficients=True)
    x = np.array([1.0, data.frame["y"].iloc[-1]])
    exact = fit.stats["se_regression"] ** 2 + x @ np.asarray(fit.params.cov) @ x
    assert np.isclose(np.var(sol.simulation.values[:, 0, 0], ddof=1), exact, rtol=4 * np.sqrt(2 / D))
    assert "coefficients" in sol.simulation.info


def test_first_order_matches_the_decision_rules_and_moments():
    nk = templates.new_keynesian()
    sol = nk.solve()
    out = sol.solve(None, ("2025Q1", "2049Q4"), draws=D, shock_sd=nk.shock_sd, seed=6)
    S = np.diag([nk.shock_sd[s] ** 2 for s in sol.shocks])
    k = len(sol.model.observed)
    V = np.zeros_like(sol.P)
    for h in range(8):  # the h-step variance: sum of P^i Q S Q' P^i'
        V = sol.P @ V @ sol.P.T + sol.Q @ S @ sol.Q.T
        sims = out.simulation.values[:, h, :]
        assert np.allclose(np.var(sims, axis=0, ddof=1), np.diag(V)[:k], rtol=4 * np.sqrt(2 / D)), h
    longrun = out.sd().frame.iloc[-1]
    assert np.allclose(longrun[list(sol.model.observed)].to_numpy(), sol.moments(nk.shock_sd).std.to_numpy(), rtol=0.02)
    model = Model([Equation("y = 0.9*y[-1] + e")])  # Model.solve with the first-order solver draws the same way
    model_sol = model.solve(Dataset.from_frame(pd.DataFrame({"y": [0.0], "e": [0.0]}, index=pd.period_range("2024Q4", periods=1, freq="Q", name="time"))),
                            ("2025Q1", "2026Q4"), solver=FirstOrder(), draws=D, shock_sd={"e": 1.0}, seed=7)
    assert np.allclose(model_sol.sd()["y"].to_numpy(), np.sqrt(np.cumsum(0.81 ** np.arange(8))), rtol=4 * np.sqrt(2 / D))


def test_estimated_model_forecasts_draw_the_estimated_shocks():
    session = pytest.importorskip("pymetrix").demo("nk_estimation")
    fit = next(v for v in session.objects.values() if type(v).__name__ == "FittedModel")
    out = fit.forecast(("2025Q1", "2026Q4"), draws=400, seed=8)
    sims = out.simulation
    observed = [m.observable for m in fit.measurement.measurements]
    assert set(observed) <= set(sims.names)
    i = 17  # one draw, its model paths put in the solution, through the measurement equations
    frame = out.frame.copy()
    for v in fit.solution.model.observed:
        frame.loc[sims.periods, v] = sims.values[i, :, sims.names.index(v)]
    applied = fit.measurement.apply(Dataset.from_frame(frame)).frame.loc[sims.periods]
    for v in observed:
        assert np.allclose(sims.values[i, :, sims.names.index(v)], applied[v].to_numpy())
    assert out.simulation.info["shock_sd"] == {k: v for k, v in fit.shock_sd.items() if v > 0}
    assert len(out.fan(observed[:2]).panels) == 2


def test_fixed_variables_and_targets_hold_in_every_draw():
    data = _data()
    fy, fx = Equation("y ~ y[-1] + x[-1]").fit(data), Equation("x ~ y[-1] + x[-1]").fit(data)
    sol = Model([fy, fx]).solve(data, SPAN, Scenario(fix={"x": 2.0}), draws=500, seed=9)
    assert np.allclose(sol.simulation.values[:, :, sol.simulation.names.index("x")], 2.0)
    level = Equation("level ~ level[-1] + g").fit(Dataset.from_frame(data.frame.assign(g=np.random.default_rng(1).normal(size=len(data.frame)))))
    model = Model([level])
    g = data.frame.assign(g=0.0)
    sol = model.solve(Dataset.from_frame(g), SPAN, Scenario(hold={"level": 150.0}, free=["g"]), draws=300, seed=10)
    assert np.allclose(sol.simulation.values[:, :, sol.simulation.names.index("level")], 150.0)
    assert np.std(sol.simulation.values[:, :, sol.simulation.names.index("g")]) > 0  # the free variable absorbs the errors


def test_draws_that_cannot_be_solved_are_dropped_and_counted():
    data = _data()
    fit = Equation("y ~ y[-1]").fit(data)
    model = Model([fit, Equation("z = log(y)")])
    sol = model.solve(data.with_series(z=np.log(np.abs(data.frame["y"]) + 1)), SPAN, draws=2000, seed=11, shock_sd={"y": 3.0})
    failed = sol.simulation.info["failed"]
    assert 0 < failed < 2000 and np.isnan(sol.simulation.values).any(axis=(1, 2)).sum() == failed
    assert np.isfinite(sol.quantiles([0.5])["z_q50"].to_numpy()).all()


def test_same_seed_same_draws_and_refusals():
    data = _data()
    model = Model([Equation("y ~ y[-1]").fit(data)])
    a, b = (model.solve(data, SPAN, draws=50, seed=12) for _ in range(2))
    assert np.array_equal(a.simulation.values, b.simulation.values)
    assert pickle.loads(pickle.dumps(a)).simulation.values.shape == (50, 12, 1)
    for kwargs, message in (({"draws": 1}, "at least 2"), ({"draws": 10, "errors": "t"}, "normal' or 'bootstrap"),
                            ({"draws": 10, "shock_sd": {"nope": 1.0}}, "not stochastic equations")):
        with pytest.raises(SpecificationError, match=message):
            model.solve(data, SPAN, **kwargs)
    with pytest.raises(SpecificationError, match="needs the shocks' standard deviations"):
        templates.new_keynesian().solve().solve(None, ("2025Q1", "2025Q4"), draws=10)
    with pytest.raises(SpecificationError, match="has no draws"):
        model.solve(data, SPAN).quantiles()
    with pytest.raises(SpecificationError, match="first-order solution"):
        Model([Equation("y = 0.5*y[+1] + e")]).solve(Dataset.from_frame(data.frame.assign(e=0.0)), SPAN, solver=PerfectForesight(), draws=10)


def test_dialogs_record_the_simulation():
    data = _data()
    session = Session()
    session.objects["data"] = data
    session.objects["eq"] = Equation("y ~ y[-1]").fit(data)
    sol = session.run(SolveModel("eq", "2020Q1", "2022Q4", draws=200, seed=3, errors="bootstrap"))
    assert sol.simulation.values.shape == (200, 12, 1)
    assert "draws=200, seed=3, errors='bootstrap'" in session.script()
    demo = pytest.importorskip("pymetrix").demo("nk_estimation")
    name = next(k for k, v in demo.objects.items() if type(v).__name__ == "FittedModel")
    out = demo.run(ForecastModel(name, "2025Q1", "2025Q4", draws=100, seed=1))
    assert out.simulation.values.shape[0] == 100 and "draws=100, seed=1" in demo.script()


@pytest.fixture(scope="module")
def ar1_posterior():
    """An AR(1) model, every variable observed, estimated with priors and sampled by Metropolis-Hastings."""
    from pymetrix import MeasurementMap, priors
    from pymetrix.metadata import Role

    rng = np.random.default_rng(0)
    y = np.zeros(80)
    for t in range(1, 80):
        y[t] = 0.7 * y[t - 1] + rng.normal(0, 1)
    data = Dataset.from_frame(pd.DataFrame({"y": y}, index=pd.period_range("2005Q1", periods=80, freq="Q", name="time")))
    model = Model([Equation("y ~ rho*y[-1] + e", params=["rho"])], params={"rho": 0.5}, roles={"e": Role.SHOCK})
    fit = model.estimate(data, MeasurementMap.direct(["y"]), shocks=["e"], shock_sd={"e": 1.0}, estimate=["rho", "sd_e"],
                         priors={"rho": priors.Normal(0.5, 0.3), "sd_e": priors.InvGamma1(1.0, 1.0)})
    return fit.mcmc(draws=1500, chains=2, seed=1), y[-1]


def test_posterior_forecasts_solve_the_model_at_each_draw(ar1_posterior):
    post, last = ar1_posterior
    n = 1000
    rho = np.array([v["rho"] for v in post._subsample(n)])
    sd = np.array([v["sd_e"] for v in post._subsample(n)])
    alone = post.forecast(("2025Q1", "2026Q4"), draws=n, shocks=False)  # each draw: rho^h y_T, exactly
    assert np.allclose(alone.simulation.values[:, :, 0], rho[:, None] ** np.arange(1, 9)[None, :] * last, atol=1e-12)
    assert np.allclose(alone.solved()["y"].to_numpy(), post.fit.params.as_dict()["rho"] ** np.arange(1, 9) * last)  # at the mode
    full = post.forecast(("2025Q1", "2026Q4"), draws=n, seed=2)
    one = full.simulation.values[:, 0, 0]  # the one-step predictive: mean E[rho] y_T, variance E[sd^2] + Var(rho y_T)
    variance = np.mean(sd ** 2) + np.var(rho * last)
    assert abs(one.mean() - np.mean(rho * last)) < 4 * np.sqrt(variance / n)
    assert abs(np.var(one, ddof=1) / variance - 1) < 4 * np.sqrt(2 / n)
    info = full.simulation.info
    assert (info["draws"], info["parameter draws"], info["failed"], info["shocks"]) == (n, n, 0, True)
    assert np.array_equal(full.simulation.values, post.forecast(("2025Q1", "2026Q4"), draws=n, seed=2).simulation.values)


def test_posterior_forecast_refusals_and_dialog(ar1_posterior):
    post, _ = ar1_posterior
    with pytest.raises(SpecificationError, match="at least 2"):
        post.forecast(("2025Q1", "2025Q4"), draws=1)
    with pytest.raises(SpecificationError, match="shocks=False"):
        post.forecast(("2025Q1", "2025Q4"), Scenario(hold={"y": 0.0}, free=["e"]), draws=10)
    session = Session()
    session.objects.update(post=post, fit=post.fit)
    assert ForecastModel("post", "2025Q1", "2025Q4").validate(session) == ["give the number of posterior draws to forecast at (draws), e.g. 500"]
    assert ForecastModel("fit", "2025Q1", "2025Q4", shocks=False).validate(session)
    out = session.run(ForecastModel("post", "2025Q1", "2025Q4", draws=50, seed=3, shocks=False))
    assert out.simulation.values.shape == (50, 4, 1) and "post.forecast(('2025Q1', '2025Q4'), draws=50, shocks=False, seed=3)" in session.script()
    assert ForecastModel("post", "2025Q1", "2025Q4", draws=50, seed=3, shocks=False).to_code() == session.script().splitlines()[-1]


def test_the_forecast_dialog_holds_variables_and_reads_its_old_field_name():
    from pymetrix.commands.registry import from_json

    dialog = ForecastModel("fit", "2025Q1", "2025Q4", hold="y[2025Q1:2025Q2]=0.5", free="e")
    scenario = dialog._scenario()
    assert set(scenario.hold) == {"y"} and tuple(scenario.free) == ("e",)
    # a step saved at schema 1, when the field was called target
    old = from_json({"kind": "command", "id": "pymetrix:forecast_model", "version": 1,
                     "fields": {"source": "fit", "start": "2025Q1", "end": "2025Q4", "target": "y[2025Q1:2025Q2]=0.5", "free": "e"}})
    assert old == dialog
