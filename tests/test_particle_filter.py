"""The particle filter (pymetrix.models.particle), checked independently:

- on a first-order linear Gaussian model its log likelihood converges to the Kalman filter's (the mean over seeds
  within a few standard errors, the spread shrinking about as 1/sqrt(N)) and its filtered means match Kalman's;
- with a nonlinear measurement equation, against the exact likelihood of a grid (point-mass) filter;
- for pruned second-order dynamics, against a two-dimensional grid filter of the same pruned state, and its
  propagation against ``SecondOrderSolution.simulate`` path for path;
- estimation on data simulated from a second-order model recovers the parameters (mode and particle marginal
  Metropolis-Hastings).
No Dynare particle-filter output is recorded in the repository's fixtures, so there is no comparison with Dynare's.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pymetrix import Dataset, Equation, Model, priors, templates
from pymetrix.errors import SpecificationError
from pymetrix.metadata import Role
from pymetrix.models.measurement import MeasurementMap
from pymetrix.models.perturbation import second_order
from pymetrix.models.solvers import ModelError


def _periods(n: int) -> pd.PeriodIndex:
    return pd.period_range("2000Q1", periods=n, freq="Q", name="time")


@pytest.fixture(scope="module")
def nk():
    nk = templates.new_keynesian()
    sol = nk.solve()
    m = nk.measurement.with_errors({o: 1.0 for o in nk.measurement.observables})
    path = sol.simulate(60, nk.shock_sd, seed=11)
    path.index = _periods(60)
    frame = m.apply(Dataset.from_frame(path)).frame
    frame = frame + np.random.default_rng(0).standard_normal(frame.shape)
    return nk, sol, m, Dataset.from_frame(frame)


def test_first_order_converges_to_the_kalman_likelihood(nk):
    template, sol, m, data = nk
    kalman = sol.system(m, template.shock_sd).loglike(data)
    spread = {}
    for n in (250, 4000):
        ll = np.array([sol.particle_filter(data, m, template.shock_sd, particles=n, seed=s, quantiles=()).loglike for s in range(12)])
        spread[n] = ll.std(ddof=1)
        # the likelihood is unbiased, its log biased down by about half its variance: allow for both
        assert abs(ll.mean() + 0.5 * spread[n] ** 2 - kalman) < 4 * spread[n] / np.sqrt(len(ll)) + 0.05, (n, ll.mean(), kalman)
    ratio = spread[4000] / spread[250]
    assert 0.12 < ratio < 0.45  # about 1/sqrt(16) = 0.25
    guided = np.array([sol.particle_filter(data, m, template.shock_sd, particles=1000, seed=s, proposal="guided",
                                           quantiles=()).loglike for s in range(6)])
    assert abs(guided.mean() - kalman) < 4 * guided.std(ddof=1) / np.sqrt(len(guided)) + 0.05
    assert guided.std(ddof=1) < spread[4000]  # the guided proposal with 1000 particles beats the bootstrap with 4000 here


def test_first_order_filtered_means_match_kalman(nk):
    from pymetrix.statespace import kalman_filter

    template, sol, m, data = nk
    system = sol.system(m, template.shock_sd)
    y, _ = system.observations(data)
    a0, P0, _ = system.initial("stationary")
    kf = kalman_filter(y, system, a0, P0)
    pf = sol.particle_filter(data, m, template.shock_sd, particles=20000, seed=3, proposal="guided")
    for v in ("y", "pi", "i"):
        j = system.states.index(v)
        kalman = kf.a_filt[:, j] + system.levels[v]
        sd = np.sqrt(kf.P_filt[:, j, j])
        assert np.max(np.abs(pf.filtered[v].to_numpy() - kalman) / sd) < 0.15, v
    # the quantiles bracket the means; observed series fitted; the ESS and resampling recorded
    q = pf.quantiles
    assert np.all(q[0.05]["pi"] <= pf.filtered["pi"].to_numpy() + 1e-12) and np.all(q[0.95]["pi"] >= pf.filtered["pi"].to_numpy() - 1e-12)
    assert "infl_fitted" in pf.filtered.names and len(pf.ess) == 60 and pf.ess.max() <= 20000
    assert pf.provenance.seed == 3 and pf.provenance.random_generator == "pcg64" and pf.provenance.data_fingerprint
    assert "Particle filter" in str(pf.summary())


def _grid_loglike_ar1(y, rho, sd, err, measure, grid=2001):
    """The exact likelihood of x(t) = rho x(t-1) + sd e(t), obs = measure(x) + err u(t), by a point-mass filter."""
    s = sd / np.sqrt(1 - rho**2)
    x = np.linspace(-7 * s, 7 * s, grid)
    dx = x[1] - x[0]
    p = stats.norm.pdf(x, scale=s) * dx  # x(0), stationary
    trans = stats.norm.pdf(x[None, :], loc=rho * x[:, None], scale=sd) * dx  # from row to column
    ll = 0.0
    for obs in y:
        p = p @ trans
        like = stats.norm.pdf(obs, loc=measure(x), scale=err)
        c = float(p @ like)
        ll += np.log(c)
        p = p * like / c
    return ll


def test_nonlinear_measurement_against_a_grid_filter():
    rho, sd, err = 0.8, 0.5, 0.3
    model = Model([Equation("z ~ rho*z[-1] + e", params=["rho"])], params={"rho": rho}, roles={"e": Role.SHOCK})
    sol = model.linearize(model.steady_state(guess={"z": 0.0}), shocks=["e"]).solve()
    m = MeasurementMap(["obs = exp(z) + 0.5*z^2"], errors={"obs": err})
    path = np.zeros(40)
    rng = np.random.default_rng(5)
    for t in range(40):
        path[t] = rho * (path[t - 1] if t else rng.standard_normal() * sd / np.sqrt(1 - rho**2)) + sd * rng.standard_normal()
    y = np.exp(path) + 0.5 * path**2 + err * rng.standard_normal(40)
    data = Dataset.from_frame(pd.DataFrame({"obs": y}, index=_periods(40)))
    exact = _grid_loglike_ar1(y, rho, sd, err, lambda x: np.exp(x) + 0.5 * x**2)
    ll = np.array([sol.particle_filter(data, m, {"e": sd}, particles=3000, seed=s, quantiles=()).loglike for s in range(10)])
    assert abs(ll.mean() - exact) < 4 * ll.std(ddof=1) / np.sqrt(len(ll)) + 0.03, (ll.mean(), exact)
    assert sol.particle_filter(data, m, {"e": sd}, particles=50, quantiles=()).provenance.method["measurement"] == "exact"


def _engine_scalar(a, b, g, h, q, s, err):
    from pymetrix.models.particle import _Engine, _Rules

    rules = _Rules(("x",), ("x",), np.array([0]), np.array([[a]]), np.array([[b]]), np.array([[g]]), np.array([[h]]),
                   np.array([[q]]), np.array([s]), np.array([[1.0]]), {"x": 0.0}, 2)
    return _Engine(rules, MeasurementMap(["obs = x"], errors={"obs": err}))


def _grid_loglike_pruned(y, a, b, g, h, q, s, err, burn, nf=161, ns=161):
    """The exact likelihood (up to the grid) of the pruned second-order scalar process
    xf' = a xf + b u, xs' = a xs + g/2 xf^2 + h xf u + q/2 u^2 + s/2, obs = xf' + xs' + err e, on a two-dimensional
    grid of (xf, xs): mass moved to each xf' cell (u = (xf' - a xf)/b) and split between the two xs cells around xs'."""
    V = b**2 / (1 - a**2)
    Es = 0.5 * (g * V + q + s) / (1 - a)
    f = np.linspace(-6 * np.sqrt(V), 6 * np.sqrt(V), nf)
    df = f[1] - f[0]
    sim_f, sim_s, lo, hi = 0.0, Es, Es, Es  # the range of xs: from a long simulation, widened
    for u in np.random.default_rng(0).standard_normal(20000):
        sim_f, sim_s = a * sim_f + b * u, a * sim_s + 0.5 * g * sim_f**2 + h * sim_f * u + 0.5 * q * u**2 + 0.5 * s
        lo, hi = min(lo, sim_s), max(hi, sim_s)
    xs = np.linspace(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo), ns)
    ds = xs[1] - xs[0]
    P = np.zeros((nf, ns))
    j0 = np.searchsorted(xs, Es) - 1
    w = (Es - xs[j0]) / ds
    pf = stats.norm.pdf(f, scale=np.sqrt(V)) * df
    P[:, j0] += pf * (1 - w)
    P[:, j0 + 1] += pf * w
    u = (f[None, :] - a * f[:, None]) / b  # [from xf, to xf']
    pu = stats.norm.pdf(u) * df / b
    base = 0.5 * g * f[:, None] ** 2 + h * f[:, None] * u + 0.5 * q * u**2 + 0.5 * s  # [from xf, to xf']

    def step(P):
        new_s = a * xs[None, None, :] + base[:, :, None]  # [from xf, to xf', from xs]
        pos = np.clip((new_s - xs[0]) / ds, 0, ns - 1.000001)
        lo = np.floor(pos).astype(int)
        frac = pos - lo
        mass = P[:, None, :] * pu[:, :, None]  # [from xf, to xf', from xs]
        out = np.zeros((nf, ns))
        to = np.broadcast_to(np.arange(nf)[None, :, None], mass.shape)
        np.add.at(out, (to.ravel(), lo.ravel()), (mass * (1 - frac)).ravel())
        np.add.at(out, (to.ravel(), lo.ravel() + 1), (mass * frac).ravel())
        return out

    for _ in range(burn):
        P = step(P)
    ll = 0.0
    x = f[:, None] + xs[None, :]
    for obs in y:
        P = step(P)
        like = stats.norm.pdf(obs, loc=x, scale=err)
        c = float(np.sum(P * like))
        ll += np.log(c)
        P = P * like / c
    return ll


def test_pruned_second_order_against_a_grid_filter():
    from pymetrix.models.particle import run

    a, b, g, h, q, s, err = 0.7, 0.5, 0.6, 0.4, 0.5, 0.3, 0.4
    engine = _engine_scalar(a, b, g, h, q, s, err)
    rng = np.random.default_rng(2)
    xf = xs_ = 0.0
    xs_ = 0.5 * (g * b**2 / (1 - a**2) + q + s) / (1 - a)
    y = []
    for _ in range(25):
        u = rng.standard_normal()
        xf, xs_ = a * xf + b * u, a * xs_ + 0.5 * g * xf**2 + h * xf * u + 0.5 * q * u**2 + 0.5 * s
        y.append(xf + xs_ + err * rng.standard_normal())
    y = np.array(y)
    exact = _grid_loglike_pruned(y, a, b, g, h, q, s, err, burn=5)
    ll = np.array([run(engine, y[:, None], np.array([err]), np.zeros((25, 1)), 4000, seed, burn=5)["loglike"] for seed in range(10)])
    assert abs(ll.mean() - exact) < 4 * ll.std(ddof=1) / np.sqrt(len(ll)) + 0.05, (ll.mean(), exact)


@pytest.fixture(scope="module")
def rbc():
    """A growth model with persistent technology, solved to second order; data simulated from it (pruned)."""
    m = Model([Equation("c ~ c[+1] / (beta*(1 + alpha*exp(z[+1])*k^(alpha - 1) - delta))", params=["alpha", "beta", "delta"]),
               Equation("y ~ exp(z)*k[-1]^alpha", params=["alpha"]),
               Equation("k ~ y + (1 - delta)*k[-1] - c", params=["delta"]),
               Equation("z ~ rho*z[-1] + e", params=["rho"])],
              params={"alpha": 0.33, "beta": 0.99, "delta": 0.025, "rho": 0.9}, roles={"e": Role.SHOCK})
    k = ((1 / 0.99 - 1 + 0.025) / 0.33) ** (1 / (0.33 - 1))
    guess = {"k": k, "y": k**0.33, "c": k**0.33 - 0.025 * k, "z": 0.0}
    s2 = second_order(m, ["e"], {"e": 0.02}, guess=guess)
    sim = s2.simulate(100, seed=4, burn=200)
    sim.index = _periods(100)
    meas = MeasurementMap(["dy = 100*log(y)", "dc = 100*log(c)"], errors={"dy": 0.5, "dc": 0.5}, reported=["gap = 100*z"])
    frame = meas.apply(Dataset.from_frame(sim)).frame
    frame = frame + 0.5 * np.random.default_rng(1).standard_normal(frame.shape)
    return m, guess, s2, meas, Dataset.from_frame(frame), sim


def test_pruned_propagation_is_the_second_order_simulation(rbc):
    from pymetrix.models.particle import _Engine, rules_of

    _, _, s2, meas, _, _ = rbc
    engine = _Engine(rules_of(s2), meas)
    shocks = np.random.default_rng(3).standard_normal((30, 1)) * 0.02
    path = s2.simulate(30, shocks=shocks)
    st = engine.r.st
    sf, ss = np.zeros((1, len(st))), np.zeros((1, len(st)))
    for t in range(30):
        xf, xs = engine.step(sf, ss, shocks[t][None, :])
        x = xf + xs
        sf, ss = xf[:, st], xs[:, st]
        for v in ("c", "k", "y"):
            assert x[0, s2.variables.index(v)] + s2.steady.values[v] == pytest.approx(path[v].iloc[t], rel=1e-12)
    # the second-order filter follows the data: the filtered technology tracks the simulated one; reported series too
    pf = s2.particle_filter(rbc[4], meas, particles=2000, seed=1, proposal="guided")
    assert np.corrcoef(pf.filtered["z"].to_numpy(), rbc[5]["z"].to_numpy())[0, 1] > 0.95
    assert np.allclose(pf.filtered["gap"].to_numpy(), 100 * pf.filtered["z"].to_numpy(), atol=1e-9)


def test_errors(rbc, nk):
    m, guess, s2, meas, data, _ = rbc
    with pytest.raises(ModelError, match="measurement error"):
        s2.particle_filter(data, meas.with_errors({"dy": 0.0}), particles=50)
    with pytest.raises(SpecificationError, match="shock sizes built in"):
        from pymetrix.models.particle import particle_filter

        particle_filter(s2, data, meas, {"e": 0.1})
    with pytest.raises(SpecificationError, match="proposal"):
        s2.particle_filter(data, meas, particles=50, proposal="auxiliary")
    with pytest.raises(SpecificationError, match="filter='particle'"):
        m.estimate(data, meas, shocks=["e"], estimate=["rho"], guess=guess, order=2, filter="kalman")
    with pytest.raises(SpecificationError, match="order is 1"):
        m.estimate(data, meas, shocks=["e"], estimate=["rho"], guess=guess, order=3)
    with pytest.raises(ModelError, match="measurement error for every observed series"):
        m.estimate(data, meas.with_errors({"dc": 0.0}), shocks=["e"], estimate=["rho"], guess=guess, order=2)
    with pytest.raises(SpecificationError, match="OccBin"):
        m.estimate(data, meas, shocks=["e"], estimate=["rho"], guess=guess, order=2, occbin=True)


def test_estimation_recovers_the_parameters(rbc):
    m, guess, _, meas, data, _ = rbc
    fit = m.estimate(data, meas, shocks=["e"], shock_sd={"e": 0.01}, estimate=["rho", "sd_e"], guess=guess, order=2,
                     particles=300, cov="none", bounds={"rho": (0.5, 0.99)})
    rho, sd = fit.params.values
    assert abs(rho - 0.9) < 0.05 and abs(sd - 0.02) < 0.004
    method = fit.provenance.method
    assert method["solution"].startswith("second-order") and method["filter"].startswith("particle") and method["particles"] == 300
    assert fit.provenance.seed == 0 and fit.provenance.random_generator
    assert fit.nonlinear is not None and fit.particle.loglike == pytest.approx(fit.stats["loglike"])
    assert "particle filter on dy, dc" in str(fit.summary())
    # the same seed, the same likelihood (common random numbers)
    lik = fit.log_posterior.__self__
    assert lik.loglike({"rho": 0.85, "sd_e": 0.02}) == lik.loglike({"rho": 0.85, "sd_e": 0.02})


def test_pmmh_samples_around_the_truth(rbc):
    m, guess, _, meas, data, _ = rbc
    fit = m.estimate(data, meas, shocks=["e"], shock_sd={"e": 0.015}, estimate=["rho", "sd_e"], guess=guess, order=2,
                     filter="guided", particles=100, priors={"rho": priors.Beta(0.7, 0.2), "sd_e": priors.InvGamma(0.02, 0.05)})
    assert np.all(np.isfinite(fit.params.se))
    post = fit.mcmc(draws=100, chains=1, burn=100, seed=1, workers=1)  # short: a check that it runs pseudo-marginally
    assert post.sampler == "pmmh" and post.provenance.method["sampler"].startswith("particle marginal")
    mean = post.mean
    assert abs(mean["rho"] - 0.9) < 0.05 and abs(mean["sd_e"] - 0.02) < 0.004
    again = fit.mcmc(draws=10, chains=1, burn=10, seed=2, workers=1)
    assert np.array_equal(again.draws, fit.mcmc(draws=10, chains=1, burn=10, seed=2, workers=1).draws)  # reproducible
    with pytest.raises(ModelError, match="SMC squared"):
        fit.smc(particles=100)
    with pytest.raises(ModelError, match="NUTS needs the gradient of an exact likelihood"):
        fit.mcmc(draws=10, chains=1, sampler="nuts")
