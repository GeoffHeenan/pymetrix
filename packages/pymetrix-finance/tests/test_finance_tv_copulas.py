"""Time-varying copulas (Patton 2006), checked independently:

- the filtered path of every family against a loop written here from the recursion (the fixed-point start by plain
  iteration, the lag window by hand), to 1e-12, and the log likelihood against the constant copulas' densities
  evaluated period by period (1e-10);
- with alpha = beta = 0 the model is the constant copula: its log likelihood equals the Gaussian copula's at
  rho = L(omega) to 1e-12, and the restricted fit's likelihood equals the constant Gaussian copula's maximum
  likelihood (a direct one-dimensional search, 1e-6);
- data simulated here (a loop drawing each period's pair from the bivariate normal or t at that period's
  correlation) from known (omega, alpha, beta) are recovered within three standard errors (Gaussian, 3,000 periods;
  t, 1,500);
- the forecast is the recursion's next value, the first simulated period has it and Kendall's tau of its draws
  is 2/pi arcsin(rho) within three standard errors; portfolio VaR, the summary, the errors and the dialog's
  recorded code.
"""

import importlib
import math

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

from pymetrix import Dataset, tv_copula
from pymetrix.errors import SpecificationError

tv = importlib.import_module("pymetrix_finance.tvcopulas")
cp = importlib.import_module("pymetrix_finance.copulas")


def _hand_path(u, v, family, x, m):
    """The recursion as written in Patton (2006), period by period."""
    nu = x[3] if family == "t" else None
    if family == "gaussian":
        f = [stats.norm.ppf(a) * stats.norm.ppf(b) for a, b in zip(u, v)]
    elif family == "t":
        f = [stats.t.ppf(a, nu) * stats.t.ppf(b, nu) for a, b in zip(u, v)]
    else:
        f = [abs(a - b) for a, b in zip(u, v)]
    fbar = sum(f) / len(f)
    blocks = [(x[0], x[1], x[2])] if family != "sjc" else [(x[0], x[1], x[2]), (x[3], x[4], x[5])]
    out = []
    for om, al, be in blocks:
        if family in ("gaussian", "t"):
            link, lo, hi = (lambda z: (1 - math.exp(-z)) / (1 + math.exp(-z))), -0.9999, 0.9999
        else:
            link, lo, hi = (lambda z: 1 / (1 + math.exp(-z))), 1e-6, 0.99
        p = 0.0 if family in ("gaussian", "t") else 0.5
        for _ in range(5000):
            p = link(om + be * p + al * fbar)
        path = []
        for t in range(len(u) + 1):
            window = [f[t - j] if t - j >= 0 else fbar for j in range(1, m + 1)]
            p = min(max(link(om + be * p + al * sum(window) / m), lo), hi)
            path.append(p)
        out.append(np.array(path))
    return out


@pytest.mark.parametrize("family, x", [("gaussian", [0.2, 0.4, 1.2]), ("t", [0.1, 0.3, 1.5, 7.0]), ("clayton", [-0.5, -2.0, 2.0]),
                                       ("gumbel", [0.3, -3.0, 1.5]), ("sjc", [-0.5, -2.0, 2.0, 0.3, -3.0, 1.5])])
def test_path_against_a_hand_written_loop(family, x):
    U = cp.sample_copula("clayton", {"theta": 1.5}, 300, 2, np.random.default_rng(5))
    u, v = U[:, 0], U[:, 1]
    got = tv.tv_filter(u, v, family, x, 7)
    hand = _hand_path(u, v, family, x, 7)
    keys = {"gaussian": ["rho"], "t": ["rho"], "clayton": ["lower"], "gumbel": ["upper"], "sjc": ["upper", "lower"]}[family]
    for k, h in zip(keys, hand):
        assert got[k] == pytest.approx(h, abs=1e-12)
    # the likelihood: the constant copula's density at each period's parameter
    ll = 0.0
    for t in range(len(u)):
        if family in ("gaussian", "t"):
            R = np.array([[1, got["rho"][t]], [got["rho"][t], 1]])
            ll += float(cp.gaussian_logpdf(U[t:t + 1], R)[0] if family == "gaussian" else cp.t_logpdf(U[t:t + 1], R, x[3])[0])
        elif family == "clayton":
            ll += float(cp.clayton_logpdf(U[t:t + 1], -1 / math.log2(got["lower"][t]))[0])
        elif family == "gumbel":
            ll += float(cp.gumbel_logpdf(U[t:t + 1], 1 / math.log2(2 - got["upper"][t]))[0])
        else:
            lu, lo = got["upper"][t], got["lower"][t]
            a = cp.ARCH["bb7"].logpdf(U[t:t + 1], (1 / math.log2(2 - lu), -1 / math.log2(lo)))[0]
            b = cp.ARCH["bb7"].logpdf(1 - U[t:t + 1], (1 / math.log2(2 - lo), -1 / math.log2(lu)))[0]
            ll += math.log(0.5 * (math.exp(a) + math.exp(b)))
    assert tv.tv_loglike(u, v, family, x, 7) == pytest.approx(ll, abs=1e-10)


def test_constant_when_alpha_and_beta_are_zero():
    U = cp.sample_copula("gaussian", {"R": np.array([[1, 0.5], [0.5, 1]])}, 500, 2, np.random.default_rng(6))
    r = 0.47
    R = np.array([[1, r], [r, 1]])
    assert tv.tv_loglike(U[:, 0], U[:, 1], "gaussian", [2 * np.arctanh(r), 0.0, 0.0]) == pytest.approx(
        float(np.sum(cp.gaussian_logpdf(U, R))), abs=1e-12)
    data = _data(stats.norm.ppf(U), ["a", "b"])
    fit = tv_copula(data, ["a", "b"])
    P = cp.pseudo_observations(U)
    best = optimize.minimize_scalar(lambda q: -float(np.sum(cp.gaussian_logpdf(P, np.array([[1, q], [q, 1]])))),
                                    bounds=(-0.99, 0.99), method="bounded", options={"xatol": 1e-10})
    assert fit.constant_loglike == pytest.approx(-best.fun, abs=1e-6)
    assert fit.loglike >= fit.constant_loglike - 1e-9


def _simulate(family, x, T, seed, m=10, burn=300):
    """Each period's pair from the bivariate normal (or t) at that period's correlation: a loop written here."""
    rng = np.random.default_rng(seed)
    nu = x[3] if family == "t" else None
    q = (lambda a: stats.norm.ppf(a)) if family == "gaussian" else (lambda a: stats.t.ppf(a, nu))
    f, out, rho = [], [], 0.0
    for t in range(T + burn):
        fb = np.mean(f[-m:]) if len(f) >= m else 0.0
        rho = np.tanh((x[0] + x[2] * rho + x[1] * fb) / 2)
        z1, z2 = rng.standard_normal(2)
        a, b = z1, rho * z1 + np.sqrt(1 - rho**2) * z2
        if family == "gaussian":
            ua, ub = stats.norm.cdf(a), stats.norm.cdf(b)
        else:
            s = np.sqrt(rng.chisquare(nu) / nu)
            ua, ub = stats.t.cdf(a / s, nu), stats.t.cdf(b / s, nu)
        f.append(q(ua) * q(ub))
        out.append((ua, ub))
    return np.array(out[burn:])


@pytest.mark.parametrize("family, x, T", [("gaussian", [0.1, 0.3, 1.5], 3000), ("t", [0.1, 0.3, 1.5, 6.0], 1500)])
def test_recovers_simulated_parameters(family, x, T):
    U = _simulate(family, x, T, 8)
    fit = tv.fit_tv_copula(U[:, 0], U[:, 1], family)
    for name, value in zip(tv.param_names(family), x):
        assert abs(fit["params"][name] - value) < 3 * fit["se"][name], (name, fit["params"][name], fit["se"][name])


def _data(X, names):
    idx = pd.period_range("2000-01-01", periods=len(X), freq="D", name="time")
    return Dataset.from_frame(pd.DataFrame(X, columns=names, index=idx))


def test_forecast_simulation_and_api():
    U = _simulate("gaussian", [0.1, 0.3, 1.5], 800, 9)
    data = _data(stats.t.ppf(U, 6) * 0.01, ["a", "b"])
    fit = tv_copula(data, ["a", "b"], margins="t")
    full = tv.tv_filter(fit.u[:, 0], fit.u[:, 1], "gaussian", [fit.params[k] for k in ("omega", "alpha", "beta")])
    assert fit.forecast()["rho"] == pytest.approx(full["rho"][-1], abs=1e-15)
    assert list(fit.path.index) == list(data.frame.index) and fit.path["rho"].to_numpy() == pytest.approx(full["rho"][:-1])
    sim = fit.simulate(horizon=5, n=20_000, seed=3)
    assert sim["u"].shape == (20_000, 5, 2) and np.all(sim["rho"][:, 0] == fit.forecast()["rho"])
    assert sim["rho"][:, 1].std() > 0  # the later periods' correlations move with the draws
    tau = stats.kendalltau(sim["u"][:, 0, 0], sim["u"][:, 0, 1])[0]
    se = np.std([stats.kendalltau(a[:, 0], a[:, 1])[0] for a in np.array_split(sim["u"][:, 0, :], 20)], ddof=1) / np.sqrt(20)
    assert abs(tau - 2 / np.pi * np.arcsin(fit.forecast()["rho"])) < 3 * se
    var = fit.portfolio_var([0.5, 0.5], level=0.99, n=50_000)
    assert 0 < var["var"] < var["es"]
    assert "Patton" in str(fit.summary()) and "rho: next period" in str(fit.summary())
    t = tv_copula(data, ["a", "b"], family="t", lags=5)
    assert "tail" in t.path and t.lags == 5 and t.params["nu"] > 2
    for family in ("clayton", "gumbel", "sjc"):
        g = tv_copula(data, ["a", "b"], family=family)
        assert np.isfinite(g.loglike) and g.loglike >= g.constant_loglike - 1e-6
    with pytest.raises(SpecificationError, match="two series"):
        tv_copula(_data(np.column_stack([U, U[:, :1]]), ["a", "b", "c"]), ["a", "b", "c"])
    with pytest.raises(SpecificationError, match="family"):
        tv_copula(data, ["a", "b"], family="frank")


def test_the_dialog_replays():
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS

    U = _simulate("gaussian", [0.1, 0.3, 1.5], 400, 10)
    data = _data(stats.norm.ppf(U) * 0.01, ["a", "b"])
    s = Session()
    s.objects["data"] = data
    s.run(COMMANDS["FitTVCopula"].cls("a, b", family="gumbel", lags=5, margins="normal"))
    code = s.history[-1].code
    assert "family='gumbel'" in code and "lags=5" in code and "margins='normal'" in code
    namespace = dict(s.namespace) | {"data": data}
    exec("from pymetrix_finance import *\n" + code, namespace)
    assert namespace["tv_copula_fit"].params == pytest.approx(s.objects["tv_copula_fit"].params)
