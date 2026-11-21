"""Marginal effects and elasticities after the conditional logits. The choice probabilities are recomputed here from
the data and the coefficients, independently of the package: the derivatives against finite differences of them (to
1e-6), cross effects summing to zero, delta-method standard errors against draws of the coefficients, and the
textbook multinomial logit of party identification on the 1996 ANES (statsmodels' MNLogit and its marginal effects,
with the same model written as a conditional logit in long form)."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, SpecificationError, choice_logit, clogit_effects, conditional_logit

sys.path.insert(0, str(Path(__file__).parent))
from test_conditional_logit import _trips  # noqa: E402 - choice sets of three or four modes, air not always available


@pytest.fixture(scope="module")
def trips():
    return _trips(n=300, seed=4)


@pytest.fixture(scope="module")
def fit(trips):
    return choice_logit(trips, "chosen", ["cost", "time"], alternative_constants=True, case_variables=["income"])


ALTS = ["air", "bus", "car", "train"]


def _probabilities(theta, f):
    """Each row's probability from the long data frame f (trip, mode, cost, time, income), air the base."""
    b_cost, b_time = theta[0], theta[1]
    alpha = dict(zip(ALTS, [0.0, *theta[2:5]]))
    gamma = dict(zip(ALTS, [0.0, *theta[5:8]]))
    v = b_cost * f["cost"] + b_time * f["time"] + f["mode"].map(alpha) + f["mode"].map(gamma) * f["income"]
    e = np.exp(v - v.groupby(f["trip"]).transform("max"))
    return e / e.groupby(f["trip"]).transform("sum")


def _by_case(p, f):
    """Probabilities as cases by alternatives, 0 where not available."""
    return pd.DataFrame({"trip": f["trip"], "mode": f["mode"], "p": p}).pivot(index="trip", columns="mode", values="p").reindex(columns=ALTS).fillna(0.0)


def test_derivatives_and_elasticities_against_finite_differences(trips, fit):
    me = fit.marginal_effects()
    f = trips.frame.reset_index()
    theta = np.asarray(fit.values)
    P = _by_case(_probabilities(theta, f), f)
    avail = _by_case(np.ones(len(f)), f).to_numpy() > 0
    h = 1e-5
    for var in ("cost", "time"):
        x = f.pivot(index="trip", columns="mode", values=var).reindex(columns=ALTS).to_numpy()
        D, E = me.table(var), me.table(var, "elasticity")
        for k in ALTS:
            up, down = f.copy(), f.copy()
            up.loc[f["mode"] == k, var] += h
            down.loc[f["mode"] == k, var] -= h
            dP = (_by_case(_probabilities(theta, up), up) - _by_case(_probabilities(theta, down), down)) / (2 * h)
            assert np.allclose(dP.mean().to_numpy(), D[k].to_numpy(), atol=1e-6), (var, k)  # averaged over every case
            kk = ALTS.index(k)
            for j in ALTS:
                both = avail[:, ALTS.index(j)] & avail[:, kk]
                with np.errstate(invalid="ignore"):  # 0/0 where j is not available, left out
                    el = (dP[j].to_numpy() * x[:, kk] / P[j].to_numpy())[both].mean()
                assert el == pytest.approx(E.loc[j, k], abs=1e-6), (var, j, k)
    up, down = f.assign(income=f["income"] + h), f.assign(income=f["income"] - h)
    dP = (_by_case(_probabilities(theta, up), up) - _by_case(_probabilities(theta, down), down)) / (2 * h)
    assert np.allclose(dP.mean().to_numpy(), me.table("income")["income"].to_numpy(), atol=1e-6)
    z = f.groupby("trip")["income"].first().to_numpy()
    with np.errstate(invalid="ignore"):
        el = [(dP[j].to_numpy() * z / P[j].to_numpy())[avail[:, i]].mean() for i, j in enumerate(ALTS)]
    assert np.allclose(el, me.table("income", "elasticity")["income"].to_numpy(), atol=1e-6)


def test_effects_at_the_means_by_the_formulas_and_finite_differences(trips, fit):
    me = fit.marginal_effects(at="mean")
    f = trips.frame.reset_index()
    theta = np.asarray(fit.values)
    means = f.groupby("mode")[["cost", "time"]].mean().reindex(ALTS)
    rep = pd.DataFrame({"trip": 0, "mode": ALTS, "cost": means["cost"].to_numpy(), "time": means["time"].to_numpy(),
                        "income": f.groupby("trip")["income"].first().mean()})
    p = _probabilities(theta, rep).to_numpy()
    for m, var in enumerate(("cost", "time")):
        formula = theta[m] * p[:, None] * (np.eye(4) - p[None, :])  # beta P_j (delta_jk - P_k)
        assert np.allclose(me.table(var).to_numpy(), formula, atol=1e-12)
        assert np.allclose(me.table(var, "elasticity").to_numpy(), formula * rep[var].to_numpy()[None, :] / p[:, None], atol=1e-10)
        for k in range(4):
            up, down = rep.copy(), rep.copy()
            up.loc[k, var] += 1e-5
            down.loc[k, var] -= 1e-5
            fd = (_probabilities(theta, up).to_numpy() - _probabilities(theta, down).to_numpy()) / 2e-5
            assert np.allclose(fd, formula[:, k], atol=1e-6)
        E = me.table(var, "elasticity").to_numpy()
        off = ~np.eye(4, dtype=bool)
        assert np.allclose(np.where(off, E, 0).max(0), np.where(off, E, np.inf).min(0))  # IIA: a column's cross elasticities are equal
    gamma = np.array([0.0, *theta[5:8]])
    assert np.allclose(me.table("income")["income"].to_numpy(), p * (gamma - p @ gamma), atol=1e-12)


def test_cross_effects_sum_to_zero(fit):
    for at in ("average", "mean"):
        me = fit.marginal_effects(at=at)
        for var in ("cost", "time"):
            assert np.abs(me.table(var).sum(axis=0)).max() < 1e-12  # each column: the probabilities still sum to one
        assert abs(me.table("income")["income"].sum()) < 1e-12


def test_delta_method_against_draws_of_the_coefficients(trips, fit):
    """A parametric bootstrap: the effects at 2000 draws from the coefficients' normal approximation."""
    me = fit.marginal_effects()
    f = trips.frame.reset_index()
    rng = np.random.default_rng(11)
    draws = rng.multivariate_normal(fit.values, fit.cov, size=2000)
    wide = {v: f.pivot(index="trip", columns="mode", values=v).reindex(columns=ALTS).to_numpy() for v in ("cost", "time", "income")}
    avail = ~np.isnan(wide["cost"])
    sims = []
    for t in draws:  # the probabilities in numpy, cases by alternatives
        v = t[0] * wide["cost"] + t[1] * wide["time"] + np.array([0.0, *t[2:5]]) + np.array([0.0, *t[5:8]]) * wide["income"]
        e = np.where(avail, np.exp(np.where(avail, v, -np.inf) - np.nanmax(v, axis=1, keepdims=True)), 0.0)
        P = e / e.sum(axis=1, keepdims=True)
        own = t[0] * P * (1 - P)  # dP_j/dcost_j
        cross = -t[0] * P[:, 2] * P[:, 3]  # dP(car)/dcost(train)
        gamma = np.array([0.0, *t[5:8]])
        inc = P * (gamma[None, :] - (P @ gamma)[:, None])
        sims.append(np.concatenate([own.mean(0), [cross.mean()], inc.mean(0)]))
    sd = np.std(sims, axis=0, ddof=1)
    se = np.concatenate([np.diag(me.table("cost", values="std err").to_numpy()), [me.table("cost", values="std err").loc["car", "train"]],
                         me.table("income", values="std err")["income"].to_numpy()])
    assert np.allclose(sd, se, rtol=0.08), np.c_[sd, se]


def test_anes96_party_identification_against_statsmodels_mnlogit():
    """Party identification (seven categories) on the respondents' characteristics, statsmodels' documented MNLogit
    example: the same model as a conditional logit with alternative constants and case variables."""
    sm = pytest.importorskip("statsmodels.api")
    anes = Dataset.example("anes96").frame.reset_index()
    xs = ["logpopul", "selfLR", "age", "educ", "income"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sm.MNLogit(anes["PID"], sm.add_constant(anes[xs])).fit(method="newton", maxiter=100, disp=0)
    pids = sorted(anes["PID"].unique())
    long = anes.loc[anes.index.repeat(len(pids)), ["respondent", "PID", *xs]].reset_index(drop=True)
    long["party"] = [f"p{int(v)}" for v in np.tile(pids, len(anes))]
    long["chosen"] = (long["party"] == [f"p{int(v)}" for v in long["PID"]]).astype(float)
    data = Dataset.from_frame(long.drop(columns="PID"), entity=["respondent", "party"])
    fit = choice_logit(data, "chosen", case_variables=xs, alternative_constants=True)
    assert fit.loglike == pytest.approx(ref.llf, abs=1e-6)
    for at in ("overall", "mean"):
        mine = fit.marginal_effects(at="average" if at == "overall" else "mean")
        theirs = ref.get_margeff(at=at)
        eff = np.asarray(theirs.margeff)  # variables by outcomes
        se = np.asarray(theirs.margeff_se)
        for i, v in enumerate(xs):
            assert np.allclose(mine.table(v)[v].to_numpy(), eff[i], atol=1e-8), (at, v)
            assert np.allclose(mine.table(v, values="std err")[v].to_numpy(), se[i], rtol=1e-4, atol=1e-7), (at, v)


def test_panel_logit_gives_odds_ratios_and_labelled_effects_at_a_fixed_effect():
    wages = Dataset.example("wage_panel")
    fit = conditional_logit(wages, "union", ["married", "exper"])
    plain = clogit_effects(fit)
    odds = plain.frame.xs("odds ratio", level="measure")
    b, se = np.asarray(fit.values), fit.se
    assert np.allclose(odds["estimate"], np.exp(b)) and np.allclose(odds["std err"], np.exp(b) * se)
    assert np.allclose(odds["lower"], fit.odds_ratios()["lower"]) and len(plain.frame) == 2
    assert "not identified" in " ".join(plain.summary().notes)
    with pytest.raises(SpecificationError, match="for the choice logit"):
        plain.table("married")
    me = clogit_effects(fit, fixed_effect=0.0)
    X = fit.design["x"]
    p = 1 / (1 + np.exp(-(X @ b)))
    eff = me.frame.xs("effect at fixed effect 0", level="measure")["estimate"].to_numpy()
    assert np.allclose(eff, b * (p * (1 - p)).mean())
    h = 1e-6  # the derivative of the average probability, shifting each regressor
    for m in range(2):
        e = np.zeros(2)
        e[m] = h
        fd = (1 / (1 + np.exp(-((X + e) @ b))) - 1 / (1 + np.exp(-((X - e) @ b)))).mean() / (2 * h)
        assert fd == pytest.approx(eff[m], abs=1e-6)
    at_mean = clogit_effects(fit, at="mean", fixed_effect=-1.0)
    pm = 1 / (1 + np.exp(-(-1.0 + X.mean(0) @ b)))
    assert np.allclose(at_mean.frame.xs("effect at fixed effect -1", level="measure")["estimate"], b * pm * (1 - pm))
    assert any("fixed effect -1" in n for n in at_mean.summary().notes)


def test_errors_and_the_result(fit):
    with pytest.raises(SpecificationError, match="panel logit"):
        clogit_effects(fit, fixed_effect=0.0)
    with pytest.raises(SpecificationError, match="at is"):
        clogit_effects(fit, at="median")
    with pytest.raises(SpecificationError, match="takes a conditional_logit"):
        clogit_effects("fit")
    me = fit.marginal_effects()
    with pytest.raises(SpecificationError, match="no variable"):
        me.table("price")
    assert me.provenance.data_fingerprint == fit.provenance.data_fingerprint and me.provenance.spec["marginal_effects"]["at"] == "average"
    from pymetrix.views import titles

    assert titles(me)[:3] == ["Summary", "Graph", "Effects: cost"] and "All" in titles(me) and "Marginal effects" in titles(fit)
    assert len(me.summary().rows) == 2 * 4 + 4 and me.plot().panels


def test_the_dialog_records_the_call_and_the_script_replays(trips):
    from pymetrix.commands import COMMANDS, Session
    from pymetrix.ui.next_steps import steps_for

    s = Session()
    s.objects["data"] = Dataset.example("wage_panel")
    s.objects["trips"] = trips
    s.run(COMMANDS["FitChoiceLogit"].cls("chosen", "cost, time", data="trips", alternative_constants=True, case_variables="income"))
    assert [label for label, *_ in steps_for(s.objects["choice"])] == ["Marginal effects…"]
    s.run(COMMANDS["ClogitMarginalEffects"].cls("choice", at="mean"))
    assert s.history[-1].code == "effects = clogit_effects(choice, at='mean')"
    s.run(COMMANDS["FitConditionalLogit"].cls("union", "married, exper"))
    s.run(COMMANDS["ClogitMarginalEffects"].cls("clogit", fixed_effect="0", name="odds"))
    assert s.history[-1].code == "odds = clogit_effects(clogit, fixed_effect=0.0)"
    direct = clogit_effects(choice_logit(trips, "chosen", ["cost", "time"], alternative_constants=True, case_variables=["income"]), at="mean")
    pd.testing.assert_frame_equal(s.objects["effects"].frame, direct.frame)
    assert COMMANDS["ClogitMarginalEffects"].cls("clogit", fixed_effect="zero").validate(s)
    namespace = {"data": s.objects["data"], "trips": trips}
    exec(s.script(), namespace)  # noqa: S102 - the generated script reproduces the effects
    pd.testing.assert_frame_equal(namespace["effects"].frame, direct.frame)
    pd.testing.assert_frame_equal(namespace["odds"].frame, s.objects["odds"].frame)
