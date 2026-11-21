"""The particle smoother (backward simulation in the shocks, pymetrix.models.particle._smooth), checked:

- on a linear Gaussian model (the New Keynesian template, first order, with measurement errors and a lagged
  measurement equation) against the Kalman smoother: means and quantiles within Monte Carlo error, the smoothed shocks
  against the disturbance smoother, and much closer to the smoother than the filtered means are;
- with a nonlinear measurement equation on a scalar autoregression, against a grid (point-mass) forward-backward
  smoother;
- for pruned second-order dynamics, whose transition is degenerate (the second-order part is deterministic given the
  shocks), against a forward-backward smoother on a two-dimensional grid of the pruned state;
- estimation with the particle filter keeps the smoothed values in ``fit.smoothed``; the dialog and refusals.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pymetrix import Dataset, Equation, Model, templates
from pymetrix.errors import SpecificationError
from pymetrix.metadata import Role
from pymetrix.models.measurement import MeasurementMap
from pymetrix.statespace import run as kalman_run


def _periods(n):
    return pd.period_range("2000Q1", periods=n, freq="Q", name="time")


@pytest.fixture(scope="module")
def nk():
    nk = templates.new_keynesian()
    sol = nk.solve()
    m = nk.measurement.with_errors({o: 1.0 for o in nk.measurement.observables})
    path = sol.simulate(30, nk.shock_sd, seed=11)
    path.index = _periods(30)
    frame = m.apply(Dataset.from_frame(path)).frame + np.random.default_rng(0).standard_normal((30, 3))
    return nk, sol, m, Dataset.from_frame(frame)


def test_a_linear_gaussian_model_against_the_kalman_smoother(nk):
    template, sol, m, data = nk
    system = sol.system(m, template.shock_sd)
    y, _ = system.observations(data)
    kr = kalman_run(y, system, "auto")
    kalman = system.filter(data, shocks=True).frame
    paths = 300
    pf = sol.particle_filter(data, m, template.shock_sd, particles=3000, seed=4, smooth_paths=paths, candidates=300)
    assert pf.smooth_paths == paths and "300 paths" in pf.provenance.method["smoother"]
    assert list(pf.smoothed.frame.columns) == list(pf.filtered.frame.columns)
    for v in ("y", "pi", "i"):
        j = system.states.index(v)
        sd = np.sqrt(kr.P_smooth[:, j, j])
        error = np.abs(pf.smoothed[v].to_numpy() - kalman[v].to_numpy()) / sd
        # Monte Carlo error: about 1/sqrt(paths) = 0.06 of a standard deviation from the paths, and the particles'
        assert error.mean() < 0.12 and error.max() < 0.5, (v, error.mean(), error.max())
        lower = pf.smoothed_quantiles[(0.05, v)].to_numpy()
        assert np.mean(np.abs(lower - (kalman[v].to_numpy() - 1.6449 * sd)) / sd) < 0.2
    # smoothing matters for the output gap, which later growth data inform: the filtered means are far off
    filtered_error = np.abs(pf.filtered["y"].to_numpy() - kalman["y"].to_numpy()).mean()
    assert np.abs(pf.smoothed["y"].to_numpy() - kalman["y"].to_numpy()).mean() < 0.3 * filtered_error
    for s in template.shocks:  # the smoothed shocks against the disturbance smoother
        scale = template.shock_sd[s]
        assert np.mean(np.abs(pf.smoothed_shocks[s].to_numpy() - kalman[s].to_numpy())) < 0.15 * scale, s
    # the same seed, the same smoother
    again = sol.particle_filter(data, m, template.shock_sd, particles=300, seed=4, smooth_paths=20, candidates=50)
    twice = sol.particle_filter(data, m, template.shock_sd, particles=300, seed=4, smooth_paths=20, candidates=50)
    assert np.array_equal(again.smoothed.frame.to_numpy(), twice.smoothed.frame.to_numpy())
    assert again.loglike == sol.particle_filter(data, m, template.shock_sd, particles=300, seed=4).loglike  # smoothing after
    assert again.smoothed_quantiles is not None and sol.particle_filter(data, m, template.shock_sd, particles=50).smoothed is None


def _grid_smoother_ar1(y, rho, sd, err, measure, grid=1201):
    """E[x(t) | all data] for x(t) = rho x(t-1) + sd e(t), obs = measure(x) + err u(t): forward-backward on a grid."""
    s = sd / np.sqrt(1 - rho**2)
    x = np.linspace(-7 * s, 7 * s, grid)
    dx = x[1] - x[0]
    p = stats.norm.pdf(x, scale=s) * dx
    trans = stats.norm.pdf(x[None, :], loc=rho * x[:, None], scale=sd) * dx
    likes = [stats.norm.pdf(obs, loc=measure(x), scale=err) for obs in y]
    alphas = []
    for like in likes:
        p = p @ trans
        p = p * like
        p /= p.sum()
        alphas.append(p)
    beta = np.ones(grid)
    means = np.zeros(len(y))
    for t in range(len(y) - 1, -1, -1):
        post = alphas[t] * beta
        means[t] = (post @ x) / post.sum()
        beta = trans @ (likes[t] * beta)
        beta /= beta.max()
    return means


def test_a_nonlinear_measurement_against_a_grid_smoother():
    rho, sd, err = 0.8, 0.5, 0.3
    model = Model([Equation("z ~ rho*z[-1] + e", params=["rho"])], params={"rho": rho}, roles={"e": Role.SHOCK})
    sol = model.linearize(model.steady_state(guess={"z": 0.0}), shocks=["e"]).solve()
    m = MeasurementMap(["obs = exp(z) + 0.5*z^2"], errors={"obs": err})
    rng = np.random.default_rng(5)
    path = np.zeros(30)
    for t in range(30):
        path[t] = rho * (path[t - 1] if t else rng.standard_normal() * sd / np.sqrt(1 - rho**2)) + sd * rng.standard_normal()
    y = np.exp(path) + 0.5 * path**2 + err * rng.standard_normal(30)
    data = Dataset.from_frame(pd.DataFrame({"obs": y}, index=_periods(30)))
    exact = _grid_smoother_ar1(y, rho, sd, err, lambda x: np.exp(x) + 0.5 * x**2)
    pf = sol.particle_filter(data, m, {"e": sd}, particles=6000, seed=1, smooth_paths=300, candidates=300, quantiles=())
    smoothed = pf.smoothed["z"].to_numpy()
    filtered = pf.filtered["z"].to_numpy()
    scale = sd / np.sqrt(1 - rho**2)
    # the measurement is not monotonic (exp(z) + z^2/2 turns at z = -0.57): some periods' smoothed distributions have
    # two modes, where the particles' error is largest (seeds 1 to 4: worst period 0.10 to 0.22, mean 0.034 to 0.041)
    assert np.mean(np.abs(smoothed - exact)) < 0.06 * scale and np.max(np.abs(smoothed - exact)) < 0.3 * scale
    assert np.mean(np.abs(smoothed - exact)) < 0.55 * np.mean(np.abs(filtered - exact))


def _grid_smoother_pruned(y, a, b, g, h, q, s, err, burn, n=101):
    """E[xf + xs | all data] for the pruned scalar process of test_particle_filter's grid filter, by forward-backward on a
    grid of (xf, xs): each cell's mass moves to every xf' cell (u = (xf' - a xf)/b) and splits between the two xs cells
    around the xs' it implies."""
    V = b**2 / (1 - a**2)
    Es = 0.5 * (g * V + q + s) / (1 - a)
    f = np.linspace(-6 * np.sqrt(V), 6 * np.sqrt(V), n)
    df = f[1] - f[0]
    sim_f, sim_s, lo, hi = 0.0, Es, Es, Es
    for u in np.random.default_rng(0).standard_normal(20000):
        sim_f, sim_s = a * sim_f + b * u, a * sim_s + 0.5 * g * sim_f**2 + h * sim_f * u + 0.5 * q * u**2 + 0.5 * s
        lo, hi = min(lo, sim_s), max(hi, sim_s)
    xs = np.linspace(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo), n)
    ds = xs[1] - xs[0]
    P = np.zeros((n, n))
    j0 = np.searchsorted(xs, Es) - 1
    w = (Es - xs[j0]) / ds
    pf = stats.norm.pdf(f, scale=np.sqrt(V)) * df
    P[:, j0] += pf * (1 - w)
    P[:, j0 + 1] += pf * w
    u = (f[None, :] - a * f[:, None]) / b  # [from xf, to xf']
    pu = stats.norm.pdf(u) * df / b
    base = 0.5 * g * f[:, None] ** 2 + h * f[:, None] * u + 0.5 * q * u**2 + 0.5 * s
    new_s = a * xs[None, None, :] + base[:, :, None]  # [from xf, to xf', from xs]
    pos = np.clip((new_s - xs[0]) / ds, 0, n - 1.000001)
    low = np.floor(pos).astype(int)
    frac = pos - low
    to = np.broadcast_to(np.arange(n)[None, :, None], low.shape)
    flat_lo, flat_hi = (to * n + low).ravel(), (to * n + low + 1).ravel()

    def forward(P):
        mass = P[:, None, :] * pu[:, :, None]
        out = np.bincount(flat_lo, (mass * (1 - frac)).ravel(), n * n) + np.bincount(flat_hi, (mass * frac).ravel(), n * n)
        return out.reshape(n, n)

    def backward(B):  # sum over the cells each cell's mass goes to, of B there: [from xf, from xs]
        gathered = (1 - frac) * B[to, low] + frac * B[to, low + 1]
        return np.einsum("il,ilj->ij", pu, gathered)

    for _ in range(burn):
        P = forward(P)
    x = f[:, None] + xs[None, :]
    likes, alphas = [], []
    for obs in y:
        P = forward(P)
        like = stats.norm.pdf(obs, loc=x, scale=err)
        P = P * like
        P /= P.sum()
        likes.append(like)
        alphas.append(P)
    B = np.ones((n, n))
    means = np.zeros(len(y))
    for t in range(len(y) - 1, -1, -1):
        post = alphas[t] * B
        means[t] = np.sum(post * x) / post.sum()
        B = backward(likes[t] * B)
        B /= B.max()
    return means


def test_pruned_second_order_against_a_grid_smoother():
    from test_particle_filter import _engine_scalar

    from pymetrix.models.particle import run

    a, b, g, h, q, s, err = 0.7, 0.5, 0.6, 0.4, 0.5, 0.3, 0.4
    engine = _engine_scalar(a, b, g, h, q, s, err)
    rng = np.random.default_rng(2)
    xf = 0.0
    xs_ = 0.5 * (g * b**2 / (1 - a**2) + q + s) / (1 - a)
    y = []
    for _ in range(20):
        u = rng.standard_normal()
        xf, xs_ = a * xf + b * u, a * xs_ + 0.5 * g * xf**2 + h * xf * u + 0.5 * q * u**2 + 0.5 * s
        y.append(xf + xs_ + err * rng.standard_normal())
    y = np.array(y)
    exact = _grid_smoother_pruned(y, a, b, g, h, q, s, err, burn=5)
    out = run(engine, y[:, None], np.array([err]), np.zeros((20, 1)), 3000, 7, burn=5, keep=(), smooth_paths=400, candidates=300)
    smoothed = out["smoothed_means"][:, 0]
    filtered = out["means"][:, 0]
    sd = np.sqrt(b**2 / (1 - a**2))
    assert np.max(np.abs(smoothed - exact)) < 0.15 * sd, np.abs(smoothed - exact).max()
    assert np.mean(np.abs(smoothed - exact)) < 0.5 * np.mean(np.abs(filtered - exact))


def test_estimation_keeps_the_smoothed_values(nk):
    template, _, m, data = nk
    fit = template.estimate(data, ["rho"], measurement=m, filter="particle", particles=300, seed=3, max_iter=20, cov="none",
                            smooth_paths=30)
    assert fit.particle.smooth_paths == 30 and fit.smoothed is fit.particle.smoothed
    assert not np.allclose(fit.smoothed.frame.to_numpy(), fit.particle.filtered.frame.to_numpy())
    plain = template.estimate(data, ["rho"], measurement=m, filter="particle", particles=300, seed=3, max_iter=20, cov="none",
                              smooth_paths=0)
    assert plain.particle.smoothed is None and plain.smoothed is plain.particle.filtered
    assert "smoothed" in str(fit.particle.summary())
    with pytest.raises(SpecificationError, match="smooth_paths"):
        template.estimate(data, ["rho"], measurement=m, filter="particle", particles=300, smooth_paths=-1)
    with pytest.raises(SpecificationError, match="smooth_paths"):
        fit.solution.particle_filter(data, m, template.shock_sd, particles=50, smooth_paths=-2)


def test_the_dialog(nk):
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import EstimateModel

    from dataclasses import replace

    _, _, m, data = nk
    raw = Dataset.from_frame(data.frame)
    model = replace(templates.new_keynesian(), measurement=m)  # measurement errors: the particle filter needs them
    s = Session()
    s.objects.update({"nk": model, "data": raw})
    s.run(EstimateModel("nk", "data", estimate="rho", filter="particle", particles=200, max_iter=10, smooth_paths=20, name="fit"))
    code = s.history[-1].code
    assert "smooth_paths=20" in code and "filter='particle'" in code
    api = model.estimate(raw, ["rho"], filter="particle", particles=200, seed=1, max_iter=10, smooth_paths=20)
    assert np.allclose(s["fit"].smoothed.frame.to_numpy(), api.smoothed.frame.to_numpy())
    namespace = dict(s.namespace) | {"nk": model, "data": raw}
    exec(code, namespace)  # noqa: S102 - the session's own script
    assert np.allclose(namespace["fit"].smoothed.frame.to_numpy(), s["fit"].smoothed.frame.to_numpy())
