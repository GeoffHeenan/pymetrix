"""The nested logit (FIML, RU2) against Biogeme 3.3.5's recorded estimates on Greene's travel-mode data (bundled with
statsmodels), against the conditional logit when every lambda is 1 (to 1e-8), its analytic gradient against central
differences, the probabilities against McFadden's GEV formula coded by hand, and recovery of true values in a
simulation."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, SpecificationError, choice_logit, nested_logit
from pymetrix.nested_logit import nested_loglike


@pytest.fixture(scope="module")
def modes():
    """Greene and Hensher's 210 travellers choosing air, train, bus or car (statsmodels' modechoice, public domain)."""
    from statsmodels.datasets import modechoice

    df = modechoice.load_pandas().data
    df["mode"] = df["mode"].map({1: "air", 2: "train", 3: "bus", 4: "car"})
    df["individual"] = df["individual"].astype(int)
    df["group"] = df["mode"].map({"air": "other", "car": "other", "train": "public", "bus": "public"})
    return Dataset.from_frame(df, entity=["individual", "mode"])


SPEC = {"regressors": ["gc", "ttme"], "alternative_constants": True, "case_variables": ["hinc"]}

# Biogeme 3.3.5 (biogeme.models.lognested, tolerance 1e-10), same utilities: air the base; its nest parameters are
# mu = 1 / lambda. Coefficients, then Rao-Cramer and robust standard errors, then the log likelihood.
BIOGEME_TWO = {  # nests public = {train, bus}, other = {air, car}
    "gc": (-0.017853764580510263, 0.006567808907885667, 0.008118355931312805),
    "ttme": (-0.10312591942313303, 0.016843005140413456, 0.01958251057629304),
    "train": (0.1287753056117245, 0.7574738928634699, 0.9041183249002824),
    "hinc:train": (-0.04909251013294968, 0.016584369847670727, 0.01695482160541987),
    "bus": (-1.3559396564798982, 0.8589487515580305, 0.8779221405148653),
    "hinc:bus": (-0.02203058480790441, 0.017766308576141636, 0.015537426892318929),
    "car": (-6.210646245329127, 1.1800678876867707, 1.2636485353996434),
    "hinc:car": (0.005548897644516375, 0.01684517260433507, 0.014211381608767023),
    "mu:public": (1.1328493232028938, 0.2832424932868136, 0.2805020617123888),
    "mu:other": (0.6104233630653567, 0.1610910437540087, 0.18739638701299438),
}
BIOGEME_TWO_LL = -187.03246737735256
BIOGEME_GREENE = {  # Greene's nests: fly = {air} (lambda fixed at 1), ground = {train, bus, car}
    "gc": (-0.01230874170628534, 0.0037474795067005684), "ttme": (-0.07099825683730597, 0.015043426063161874),
    "train": (0.17440464104763984, 0.5547064216342443), "hinc:train": (-0.03700467161657253, 0.013233890424081535),
    "bus": (-0.8386668880351741, 0.7103711681196669), "hinc:bus": (-0.018563085978103935, 0.012945063024391188),
    "car": (-3.884528838365354, 1.196297300125323), "hinc:car": (-0.002350310879097934, 0.010871981072110756),
    "mu:ground": (1.5707577301562028, 0.3798452791628672),
}
BIOGEME_GREENE_LL = -187.68245715741173
BIOGEME_LOGIT_LL = -189.5251525799339


def _as_lambda(name, value, se):
    """Biogeme's mu as lambda = 1 / mu, its standard error by the delta method."""
    return (f"lambda:{name[3:]}", 1 / value, se / value**2) if name.startswith("mu:") else (name, value, se)


def test_against_biogeme(modes):
    fit = nested_logit(modes, "choice", nests="group", **SPEC)
    robust = nested_logit(modes, "choice", nests="group", cov="cluster", **SPEC)
    got = dict(zip(fit.names, zip(fit.values, fit.se, robust.se)))
    G = fit.cases
    for name, (value, se, rob) in BIOGEME_TWO.items():
        key, v, s = _as_lambda(name, value, se)
        _, _, r = _as_lambda(name, value, rob)
        assert got[key][0] == pytest.approx(v, rel=1e-4, abs=1e-6), key  # Biogeme stops at a gradient of about 4e-6
        assert got[key][1] == pytest.approx(s, rel=1e-3), key
        assert got[key][2] * np.sqrt((G - 1) / G) == pytest.approx(r, rel=1e-3), key  # Biogeme's sandwich has no G / (G - 1)
    assert fit.loglike == pytest.approx(BIOGEME_TWO_LL, abs=1e-7)
    assert fit.loglike_logit == pytest.approx(BIOGEME_LOGIT_LL, abs=1e-7)
    assert fit.iia_test.df == 2 and fit.iia_test.statistic == pytest.approx(2 * (BIOGEME_TWO_LL - BIOGEME_LOGIT_LL), abs=1e-6)
    assert fit.lambdas["public"] == pytest.approx(1 / 1.1328493232028938, rel=1e-4)


def test_greene_nests_with_a_degenerate_nest(modes):
    fit = nested_logit(modes, "choice", nests={"fly": ["air"], "ground": ["train", "bus", "car"]}, **SPEC)
    got = dict(zip(fit.names, zip(fit.values, fit.se)))
    assert "lambda:fly" not in got and fit.fixed_nests == ("fly",) and fit.lambdas["fly"] == 1.0
    for name, (value, se) in BIOGEME_GREENE.items():
        key, v, s = _as_lambda(name, value, se)
        assert got[key][0] == pytest.approx(v, rel=2e-4, abs=1e-6), key  # Biogeme stopped short here (gradient 2.6e-5)
        assert got[key][1] == pytest.approx(s, rel=2e-3), key
    assert fit.loglike == pytest.approx(BIOGEME_GREENE_LL, abs=1e-7)
    assert "fixed at 1" in " ".join(fit.summary().notes)


def test_every_lambda_one_is_the_conditional_logit(modes):
    cl = choice_logit(modes, "choice", **SPEC)
    one = nested_logit(modes, "choice", nests="group", dissimilarity="one", **SPEC)
    assert list(one.names) == list(cl.names) and one.iia_test is None
    assert np.allclose(one.values, cl.values, atol=1e-8, rtol=0) and np.allclose(one.se, cl.se, atol=1e-8, rtol=0)
    assert one.loglike == pytest.approx(cl.loglike, abs=1e-8)
    assert np.allclose(one.probabilities.to_numpy(), cl.probabilities.reindex(one.probabilities.index).to_numpy(), atol=1e-10)
    # the likelihood itself: lambdas of 1 give the conditional logit's at any coefficients
    free = nested_logit(modes, "choice", nests="group", **SPEC)
    d = free.design
    choices = __import__("pymetrix._choice", fromlist=["x"]).read_choices(modes, "choice", SPEC["regressors"], None, None, True, ["hinc"], None, "t")
    beta = np.random.default_rng(0).normal(0, 0.01, len(cl.values))
    ll, _, _ = nested_loglike(np.r_[beta, 1.0, 1.0], choices.X, choices.available, choices.chosen, d["nest_of"], d["lam_param"])
    ll_logit = __import__("pymetrix._choice", fromlist=["x"]).logit_loglike(beta, choices.X, choices.available, choices.chosen)[0]
    assert ll == pytest.approx(ll_logit, abs=1e-10)


def _gev_probabilities(V, available, nest_of, lam):
    """McFadden's GEV form, coded apart from the module: P_j = y_j G_j / G with y = exp(V) and
    G = sum_m (sum_{k in m} y_k^(1/lambda_m))^lambda_m."""
    y = np.where(available, np.exp(V), 0.0)
    G = np.zeros(len(V))
    inner = {}
    for m in np.unique(nest_of):
        inner[m] = (y[:, nest_of == m] ** (1 / lam[m])).sum(axis=1)
        G += inner[m] ** lam[m]
    P = np.zeros_like(y)
    for j in range(V.shape[1]):
        m = nest_of[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            Gj = np.where(available[:, j], inner[m] ** (lam[m] - 1) * y[:, j] ** (1 / lam[m] - 1), 0.0)
        P[:, j] = y[:, j] * Gj / G
    return P


def _nested_trips(n=600, seed=3):
    """Trips by car, bus, train or tram; bus, train and tram share a nest; the tram is not always available."""
    rng = np.random.default_rng(seed)
    alts = ["car", "bus", "train", "tram"]
    rows = []
    for i in range(n):
        for a in alts:
            if a == "tram" and rng.random() < 0.4:
                continue
            rows.append({"trip": i, "mode": a, "cost": rng.normal(5, 2), "time": rng.normal(3, 1), "income": 0.0,
                         "group": "private" if a == "car" else "public"})
    frame = pd.DataFrame(rows)
    frame["income"] = frame.groupby("trip")["cost"].transform(lambda s: rng.normal(3, 1))
    frame["chosen"] = 0.0
    return frame


def _draw_choices(frame, theta_fn, seed):
    """Choices drawn from the model's probabilities (GEV formula by hand)."""
    rng = np.random.default_rng(seed)
    alts = ["bus", "car", "train", "tram"]
    nest_of = np.array([1, 0, 1, 1])
    out = frame.copy()
    for i, g in out.groupby("trip"):
        avail = np.isin(alts, g["mode"].to_numpy())
        V = np.full(4, 0.0)
        for _, r in g.iterrows():
            V[alts.index(r["mode"])] = theta_fn(r)
        P = _gev_probabilities(V[None], avail[None], nest_of, np.array([1.0, 0.5]))[0]
        k = rng.choice(4, p=P / P.sum())
        out.loc[g.index[g["mode"].to_numpy() == alts[k]], "chosen"] = 1.0
    return out


def test_probabilities_and_gradient_by_hand():
    frame = _nested_trips(n=150, seed=1)
    frame = _draw_choices(frame, lambda r: -0.4 * r["cost"] - 0.8 * r["time"], seed=2)
    data = Dataset.from_frame(frame, entity=["trip", "mode"])
    fit = nested_logit(data, "chosen", ["cost", "time"], nests="group")
    d = fit.design
    choices = __import__("pymetrix._choice", fromlist=["x"]).read_choices(data, "chosen", ["cost", "time"], None, None, False, [], None, "t")
    lam = fit.lambdas.reindex(d["nest_names"]).to_numpy()
    P = _gev_probabilities(choices.X @ fit.values[:2], choices.available, d["nest_of"], lam)
    assert np.allclose(fit.probabilities.to_numpy(), P[choices.available][np.argsort(choices.used)], atol=1e-12)
    assert np.allclose(fit.probabilities.groupby(level="trip").sum(), 1.0)
    # the analytic gradient against central differences, away from the optimum, with varying choice sets
    theta = fit.values + np.array([0.05, -0.03, 0.1])
    _, grad, scores = nested_loglike(theta, choices.X, choices.available, choices.chosen, d["nest_of"], d["lam_param"])
    h = 1e-6
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        up = nested_loglike(theta + e, choices.X, choices.available, choices.chosen, d["nest_of"], d["lam_param"])[0]
        down = nested_loglike(theta - e, choices.X, choices.available, choices.chosen, d["nest_of"], d["lam_param"])[0]
        assert grad[i] == pytest.approx((up - down) / (2 * h), rel=1e-6, abs=1e-6)
    assert np.allclose(scores.sum(axis=0), grad)


def test_recovers_true_values():
    """5000 trips drawn from a nested logit with lambda 0.5 for the public modes: the estimates within 3.5 standard
    errors of the truth, and the LR test rejects IIA."""
    frame = _nested_trips(n=5000, seed=5)
    truth = {"cost": -0.4, "time": -0.8, "bus": -0.5, "train": 0.2, "tram": -0.3}

    def utility(r):
        return truth["cost"] * r["cost"] + truth["time"] * r["time"] + truth.get(r["mode"], 0.0)

    data = Dataset.from_frame(_draw_choices(frame, utility, seed=6), entity=["trip", "mode"])
    fit = nested_logit(data, "chosen", ["cost", "time"], nests="group", alternative_constants=True, base="car")
    got = dict(zip(fit.names, zip(fit.values, fit.se)))
    for name, value in {**truth, "lambda:public": 0.5}.items():
        assert abs(got[name][0] - value) < 3.5 * got[name][1], name
    assert got["lambda:public"][1] < 0.06 and fit.iia_test.p_value < 1e-6
    equal = nested_logit(data, "chosen", ["cost", "time"], nests="group", alternative_constants=True, base="car", dissimilarity="equal")
    assert equal.loglike == pytest.approx(fit.loglike, abs=1e-8)  # one nest with a lambda: "equal" is "free"


def test_marginal_effects_against_finite_differences(modes):
    fit = nested_logit(modes, "choice", nests="group", **SPEC)
    d = fit.design
    me = fit.marginal_effects()
    assert "nested logit" in me.summary().notes[0] and "differ" in me.summary().notes[1]
    choices = __import__("pymetrix._choice", fromlist=["x"]).read_choices(modes, "choice", SPEC["regressors"], None, None, True, ["hinc"], None, "t")
    lam = fit.lambdas.reindex(d["nest_names"]).to_numpy()
    p = len(choices.names)

    def probs(X):
        return _gev_probabilities(X @ fit.values[:p], choices.available, d["nest_of"], lam)

    base = probs(choices.X)
    h = 1e-6
    alts = [str(a) for a in choices.alternatives]
    for k, a in enumerate(alts):
        X = choices.X.copy()
        X[:, k, 0] += h  # gc of alternative k
        D = (probs(X) - base) / h
        assert np.allclose(me.table("gc")[a].reindex(alts).to_numpy(), D.mean(axis=0), atol=1e-6)
    table = me.table("gc").to_numpy()
    assert np.allclose(table.sum(axis=0), 0, atol=1e-12) and not np.isclose(table[0, 1], table[3, 1])  # cross effects differ
    at_mean = fit.marginal_effects(at="mean")
    assert at_mean.at == "mean" and np.all(np.isfinite(at_mean.frame["std err"]))


def test_errors(modes):
    with pytest.raises(SpecificationError, match="give nests"):
        nested_logit(modes, "choice", ["gc"])
    with pytest.raises(SpecificationError, match="every alternative needs a nest"):
        nested_logit(modes, "choice", ["gc"], nests={"a": ["air", "car"], "b": ["bus"]})
    with pytest.raises(SpecificationError, match="at least two nests"):
        nested_logit(modes, "choice", ["gc"], nests={"all": ["air", "car", "bus", "train"]})
    with pytest.raises(SpecificationError, match="in two nests"):
        nested_logit(modes, "choice", ["gc"], nests={"a": ["air", "car"], "b": ["bus", "train", "car"]})
    with pytest.raises(SpecificationError, match="dissimilarity"):
        nested_logit(modes, "choice", ["gc"], nests="group", dissimilarity="fixed")


def test_the_dialog_records_the_call_and_the_script_replays(modes):
    from pymetrix.commands import COMMANDS, Session

    s = Session()
    s.objects["modes"] = modes
    s.run(COMMANDS["FitNestedLogit"].cls("choice", "gc, ttme", "group", data="modes", alternative_constants=True, case_variables="hinc"))
    assert s.history[-1].code == ("nested = nested_logit(modes, 'choice', regressors=['gc', 'ttme'], nests='group', alternative_constants=True, "
                                  "case_variables=['hinc'])")
    namespace = {"modes": modes}
    exec(s.script(), namespace)  # noqa: S102 - the generated script reproduces the estimate
    assert np.array_equal(namespace["nested"].values, s.objects["nested"].values)
    assert s.objects["nested"].provenance.resolved["nests"]["air"] == "other"
