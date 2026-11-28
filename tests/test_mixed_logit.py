"""The mixed logit by simulated maximum likelihood: against xlogit 0.2.7's recorded estimates on the same Halton draws
(the simulated log likelihood at xlogit's estimates equal to xlogit's to 1e-8, with and without a panel), the Halton
draws against the radical inverse coded by hand, the analytic gradient against central differences (uncorrelated
and correlated), recovery of true means, standard deviations and a correlation in a simulation, the conditional
means and the marginal effects recomputed by hand from the draws."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, SpecificationError, mixed_logit
from pymetrix.mixed_logit import halton_draws, simulated_loglike


def _panel_choices(people=400, tasks=5, seed=7):
    """Repeated choices among three products: a lognormal coefficient on minus the price, a normal one on quality."""
    rng = np.random.default_rng(seed)
    n = people * tasks
    b_price = np.repeat(np.exp(-0.5 + 0.5 * rng.standard_normal(people)), tasks)
    b_quality = np.repeat(1.0 + 0.8 * rng.standard_normal(people), tasks)
    price, quality = rng.uniform(0, 4, (n, 3)), rng.normal(0, 1, (n, 3))
    u = np.array([0.0, 0.4, -0.3]) - b_price[:, None] * price + b_quality[:, None] * quality + rng.gumbel(size=(n, 3))
    chosen = (u == u.max(axis=1, keepdims=True)).astype(float)
    return pd.DataFrame({"person": np.repeat(np.arange(people), 3 * tasks), "task": np.repeat(np.arange(n), 3),
                         "product": np.tile(["p1", "p2", "p3"], n), "mprice": -price.ravel(), "quality": quality.ravel(),
                         "chosen": chosen.ravel()})


@pytest.fixture(scope="module")
def shop():
    return Dataset.from_frame(_panel_choices(), entity=["task", "product"])


# xlogit 0.2.7, MixedLogit.fit(randvars={"mprice": "ln", "quality": "n"}, fit_intercept=True, n_draws=100 (Halton),
# num_hess=True, optim_method="L-BFGS-B"; its default BFGS failed on these data): _intercept.p2, _intercept.p3, mprice,
# quality, sd.mprice, sd.quality; standard errors; log likelihood
XLOGIT = {
    "panel": (np.array([0.353557921563, -0.368677395446, -0.581431099711, 0.873698846657, 0.506957519122, 0.630730115306]),
              np.array([0.064692363483, 0.072614071705, 0.07161143185, 0.057034233554, 0.105066053878, 0.069152933839]),
              -1700.231191830502),
    "cases": (np.array([0.322708518155, -0.354503088497, -0.608333066878, 0.79096864236, 0.388683813055, 0.398844766039]),
              np.array([0.062887700302, 0.070554173309, 0.081013504247, 0.059570291348, 0.274283757582, 0.145327937802]),
              -1731.4520449064003),
}


def _ours(x):
    """xlogit's order (constants first) in ours (regressors first)."""
    return np.r_[x[2:4], x[0:2], x[4:]]


@pytest.mark.parametrize("panel", ["panel", "cases"])
def test_against_xlogit(shop, panel):
    fit = mixed_logit(shop, "chosen", ["mprice", "quality"], random={"mprice": "lognormal", "quality": "normal"},
                      panel="person" if panel == "panel" else None, alternative_constants=True, draws=100)
    b, se, ll = XLOGIT[panel]
    assert list(fit.names) == ["mprice", "quality", "p2", "p3", "sd.mprice", "sd.quality"]
    assert simulated_loglike(_ours(b), fit.design["sim"], gradient=False)[0] == pytest.approx(ll, abs=1e-8)  # the same simulated likelihood
    assert np.allclose(fit.values, _ours(b), atol=1e-5) and np.allclose(fit.se, _ours(se), rtol=1e-4)
    assert fit.loglike >= ll - 1e-9 and fit.people == (400 if panel == "panel" else 2000) and fit.cases == 2000
    assert fit.provenance.resolved["draws"] == 100 and fit.provenance.resolved["halton_drop"] == 100
    assert fit.provenance.resolved["halton_primes"] == [2, 3]


def test_halton_draws_by_hand():
    def radical_inverse(i, base):
        out, f = 0.0, 1.0 / base
        while i:
            out += f * (i % base)
            i //= base
            f /= base
        return out

    from scipy.stats import norm

    draws = halton_draws(people=3, draws=4, dimensions=3, drop=100)
    for p in range(3):
        for r in range(4):
            for k, base in enumerate((2, 3, 5)):
                assert draws[p, r, k] == pytest.approx(norm.ppf(radical_inverse(100 + p * 4 + r, base)), abs=1e-12)


@pytest.mark.parametrize("correlated", [False, True])
def test_gradient_against_central_differences(shop, correlated):
    small = Dataset.from_frame(_panel_choices(people=60, tasks=4, seed=1), entity=["task", "product"])
    fit = mixed_logit(small, "chosen", ["mprice", "quality"], random={"mprice": "lognormal", "quality": "normal"}, panel="person",
                      alternative_constants=True, draws=40, correlated=correlated)
    sim = fit.design["sim"]
    theta = np.asarray(fit.values) + 0.05
    _, grad, scores, _ = simulated_loglike(theta, sim)
    h = 1e-6
    for i in range(len(theta)):
        e = np.zeros(len(theta))
        e[i] = h
        num = (simulated_loglike(theta + e, sim, False)[0] - simulated_loglike(theta - e, sim, False)[0]) / (2 * h)
        assert grad[i] == pytest.approx(num, rel=1e-6, abs=1e-6)
    assert np.allclose(scores.sum(axis=0), grad)
    if correlated:
        assert fit.names[-3:] == ("chol.mprice.mprice", "chol.quality.mprice", "chol.quality.quality")


def _big(people=1000, tasks=6, seed=0):
    """Correlated tastes: log price coefficient N(-0.5, 0.5^2); time N(-1, 1) with correlation 0.6 with it."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((people, 2))
    b_price = np.repeat(np.exp(-0.5 + 0.5 * z[:, 0]), tasks)
    b_time = np.repeat(-1.0 + 0.6 * z[:, 0] + 0.8 * z[:, 1], tasks)
    n = people * tasks
    price, time = rng.uniform(0, 4, (n, 3)), rng.normal(0, 1, (n, 3))
    u = np.array([0.0, 0.3, -0.2]) - b_price[:, None] * price + b_time[:, None] * time + rng.gumbel(size=(n, 3))
    chosen = (u == u.max(axis=1, keepdims=True)).astype(float)
    frame = pd.DataFrame({"person": np.repeat(np.arange(people), 3 * tasks), "task": np.repeat(np.arange(n), 3),
                          "alt": np.tile(["a", "b", "c"], n), "mprice": -price.ravel(), "time": time.ravel(), "chosen": chosen.ravel()})
    return Dataset.from_frame(frame, entity=["task", "alt"])


def test_recovers_true_values():
    """1000 people, 6 choices each: the means, the Cholesky factor (chol.time.mprice 0.6 = correlation 0.6 times s.d.
    1) and the constants within 3 standard errors, with 200 Halton draws."""
    fit = mixed_logit(_big(), "chosen", ["mprice", "time"], random={"mprice": "lognormal", "time": "normal"}, panel="person",
                      alternative_constants=True, correlated=True, draws=200)
    truth = {"mprice": -0.5, "time": -1.0, "b": 0.3, "c": -0.2, "chol.mprice.mprice": 0.5, "chol.time.mprice": 0.6, "chol.time.time": 0.8}
    got = dict(zip(fit.names, zip(fit.values, fit.se)))
    for name, value in truth.items():
        assert abs(got[name][0] - value) < 3 * got[name][1], name
    moments = fit.random_coefficients().to_frame()
    assert abs(moments.loc["correlation: time, mprice", "estimate"] - 0.6) < 3 * moments.loc["correlation: time, mprice", "std err"]
    assert moments.loc["median: mprice", "estimate"] == pytest.approx(np.exp(got["mprice"][0]))
    assert fit.loglike > fit.loglike_logit + 100  # heterogeneity the conditional logit misses


def test_conditional_means_and_effects_by_hand(shop):
    fit = mixed_logit(shop, "chosen", ["mprice", "quality"], random={"quality": "normal"}, panel="person", alternative_constants=True, draws=60)
    sim = fit.design["sim"]
    theta = np.asarray(fit.values)
    beta_q = theta[1] + theta[-1] * sim.eta[:, :, 0]                         # (people, draws)
    f = shop.frame.reset_index().sort_values(["person", "task", "product"])
    X = np.stack([f["mprice"].to_numpy().reshape(-1, 3), f["quality"].to_numpy().reshape(-1, 3)])
    asc = np.array([0.0, theta[2], theta[3]])
    person = f["person"].to_numpy()[::3]
    chosen = f["chosen"].to_numpy().reshape(-1, 3).argmax(axis=1)
    V = theta[0] * X[0][:, :, None] + beta_q[person][:, None, :] * X[1][:, :, None] + asc[None, :, None]
    P = np.exp(V) / np.exp(V).sum(axis=1, keepdims=True)                     # (cases, 3, draws)
    like = pd.DataFrame(np.log(P[np.arange(len(chosen)), chosen])).groupby(person).sum().to_numpy()
    w = np.exp(like - like.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    assert np.allclose(fit.individual["quality"].to_numpy(), (w * beta_q).sum(axis=1), atol=1e-12)
    # average own and cross effects of quality: E[beta L_j (delta_jk - L_k)] over the draws, averaged over the cases
    D = np.einsum("cr,cjr,jk->jk", beta_q[person], P, np.eye(3)) - np.einsum("cr,cjr,ckr->jk", beta_q[person], P, P)
    me = fit.marginal_effects()
    assert np.allclose(me.table("quality").to_numpy(), D / P.shape[2] / len(chosen), atol=1e-12)
    assert np.allclose(fit.probabilities.groupby(level="task").sum(), 1.0)
    with pytest.raises(SpecificationError, match="averaged over the cases"):
        fit.marginal_effects(at="mean")


def test_pseudo_random_draws_follow_the_seed(shop):
    kw = {"random": {"quality": "normal"}, "panel": "person", "draws": 30, "draw_type": "random"}
    a = mixed_logit(shop, "chosen", ["mprice", "quality"], seed=1, **kw)
    b = mixed_logit(shop, "chosen", ["mprice", "quality"], seed=1, **kw)
    c = mixed_logit(shop, "chosen", ["mprice", "quality"], seed=2, **kw)
    assert np.array_equal(a.values, b.values) and not np.allclose(a.values, c.values)
    assert a.provenance.seed == 1 and a.provenance.resolved["draw_type"] == "random" and "halton_drop" not in a.provenance.resolved


def test_errors(shop):
    with pytest.raises(SpecificationError, match="give random"):
        mixed_logit(shop, "chosen", ["mprice"])
    with pytest.raises(SpecificationError, match="not among them"):
        mixed_logit(shop, "chosen", ["mprice"], random={"quality": "normal"})
    with pytest.raises(SpecificationError, match="normal' or 'lognormal"):
        mixed_logit(shop, "chosen", ["mprice"], random={"mprice": "triangular"})
    with pytest.raises(SpecificationError, match="same for every alternative"):
        mixed_logit(shop, "chosen", ["mprice"], random={"mprice": "normal"}, panel="quality", draws=5)


def test_the_dialog_records_the_call_and_the_script_replays(shop):
    from pymetrix.commands import COMMANDS, Session

    s = Session()
    s.objects["shop"] = shop
    cmd = COMMANDS["FitMixedLogit"].cls("chosen", "mprice, quality", "quality: normal", data="shop", panel="person", draws=20)
    assert cmd.validate(s) == [] and COMMANDS["FitMixedLogit"].cls("chosen", "mprice", "mprice: uniform").validate(s)
    s.run(cmd)
    assert s.history[-1].code == ("mixed = mixed_logit(shop, 'chosen', regressors=['mprice', 'quality'], random={'quality': 'normal'}, "
                                  "panel='person', draws=20)")
    namespace = {"shop": shop}
    exec(s.script(), namespace)  # noqa: S102 - the generated script reproduces the estimate
    assert np.array_equal(namespace["mixed"].values, s.objects["mixed"].values)
