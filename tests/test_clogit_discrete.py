"""Discrete changes after the conditional logits: the change in each alternative's probability when a 0/1 variable
goes from 0 to 1, or a variable from its value to the value plus a chosen amount. Checks: the probabilities
recomputed here from the data and the coefficients (exact), changes summing to zero over the alternatives, statsmodels'
MNLogit get_margeff(dummy=True) (and its predictions for a change by 2) on the 1996 ANES with a 0/1 regressor (vote), delta-method standard
errors against draws of the coefficients, the panel logit's at a fixed effect, and the dialog."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, SpecificationError, choice_logit, clogit_effects, conditional_logit

sys.path.insert(0, str(Path(__file__).parent))
from test_conditional_logit import _trips  # noqa: E402 - choice sets of three or four modes, air not always available

ALTS = ["air", "bus", "car", "train"]


@pytest.fixture(scope="module")
def trips():
    f = _trips(n=300, seed=4).frame.reset_index()
    f["fast"] = (f["time"] < 3).astype(float)  # a 0/1 attribute
    f["rich"] = (f["income"] > 3).astype(float)  # a 0/1 characteristic of the case
    return Dataset.from_frame(f, entity=["trip", "mode"])


@pytest.fixture(scope="module")
def fit(trips):
    return choice_logit(trips, "chosen", ["cost", "fast"], alternative_constants=True, case_variables=["income", "rich"])


def _probabilities(theta, f):
    """Cases by alternatives (0 where not available), from the long frame f; air the base."""
    alpha = dict(zip(ALTS, [0.0, *theta[2:5]]))
    g_inc, g_rich = dict(zip(ALTS, [0.0, *theta[5:8]])), dict(zip(ALTS, [0.0, *theta[8:11]]))
    v = theta[0] * f["cost"] + theta[1] * f["fast"] + f["mode"].map(alpha) + f["mode"].map(g_inc) * f["income"] + f["mode"].map(g_rich) * f["rich"]
    e = np.exp(v - v.groupby(f["trip"]).transform("max"))
    p = e / e.groupby(f["trip"]).transform("sum")
    return pd.DataFrame({"trip": f["trip"], "mode": f["mode"], "p": p}).pivot(index="trip", columns="mode", values="p").reindex(columns=ALTS).fillna(0.0)


def _at_means(f):
    """One case with every alternative: attributes at their means by alternative, characteristics at their means."""
    means = f.groupby("mode")[["cost", "fast"]].mean().reindex(ALTS)
    case = f.groupby("trip")[["income", "rich"]].first().mean()
    return pd.DataFrame({"trip": 0, "mode": ALTS, "cost": means["cost"].to_numpy(), "fast": means["fast"].to_numpy(),
                         "income": case["income"], "rich": case["rich"]})


def test_discrete_changes_by_direct_recomputation(trips, fit):
    theta = np.asarray(fit.values)
    for at in ("average", "mean"):
        me = fit.marginal_effects(at=at, dummy=True, changes={"cost": 1.0, "income": -0.5})
        assert me.discrete == {"cost": "x to x + 1", "income": "x to x - 0.5", "fast": "0 to 1", "rich": "0 to 1"}
        f = trips.frame.reset_index() if at == "average" else _at_means(trips.frame.reset_index())
        for var, how in (("fast", "dummy"), ("cost", 1.0)):
            got = me.table(var, "discrete change")
            for k in ALTS:
                lo, hi = f.copy(), f.copy()
                rows = f["mode"] == k  # alternative k's attribute alone
                if how == "dummy":
                    lo.loc[rows, var], hi.loc[rows, var] = 0.0, 1.0
                else:
                    hi.loc[rows, var] += how
                change = (_probabilities(theta, hi) - _probabilities(theta, lo)).mean()
                np.testing.assert_allclose(got[k].to_numpy(), change.to_numpy(), atol=1e-13, err_msg=f"{at} {var} {k}")
        for var, lo_v, hi_v in (("rich", 0.0, 1.0), ("income", 0.0, -0.5)):
            lo = f.assign(**{var: lo_v}) if var == "rich" else f
            hi = f.assign(**{var: hi_v}) if var == "rich" else f.assign(income=f["income"] - 0.5)
            change = (_probabilities(theta, hi) - _probabilities(theta, lo)).mean()
            np.testing.assert_allclose(me.table(var, "discrete change")[var].to_numpy(), change.to_numpy(), atol=1e-13)
        # every column of changes sums to zero over the alternatives: the probabilities still sum to one
        for var in ("fast", "cost"):
            assert np.abs(me.table(var, "discrete change").sum(axis=0)).max() < 1e-13
        for var in ("rich", "income"):
            assert abs(me.table(var, "discrete change")[var].sum()) < 1e-13
    # a small change, divided by its size, is the derivative
    small = fit.marginal_effects(changes={"cost": 1e-6})
    np.testing.assert_allclose(small.table("cost", "discrete change").to_numpy() / 1e-6, small.table("cost").to_numpy(), atol=1e-6)


def test_dummies_found_as_statsmodels_does(fit):
    me = fit.marginal_effects(dummy=True)
    assert set(me.discrete) == {"fast", "rich"}  # cost and income are not 0/1
    plain = fit.marginal_effects()
    assert plain.discrete == {} and "discrete change" not in plain.frame.index.get_level_values("measure")
    pd.testing.assert_frame_equal(me.frame.xs("derivative", level="measure", drop_level=False),
                                  plain.frame.xs("derivative", level="measure", drop_level=False))  # derivatives unchanged
    rows = me.summary().index
    assert "ΔP(car)/Δfast(car)" in rows and "dP(car)/dcost(car)" in rows and "ΔP(bus)/Δrich" in rows
    assert any("Discrete changes" in n for n in me.summary().notes)
    from pymetrix.views import titles

    assert "Discrete changes: fast" in titles(me) and "Discrete changes: cost" not in titles(me)
    with pytest.raises(SpecificationError, match="no discrete change"):
        me.table("cost", "discrete change")
    with pytest.raises(SpecificationError, match="no variable 'price'"):
        fit.marginal_effects(changes={"price": 1})
    with pytest.raises(SpecificationError, match="nonzero"):
        fit.marginal_effects(changes={"cost": 0})
    with pytest.raises(SpecificationError, match="a number"):
        fit.marginal_effects(changes={"cost": "one"})


def test_standard_errors_against_draws_of_the_coefficients(trips, fit):
    me = fit.marginal_effects(dummy=True, changes={"cost": 1.0})
    f = trips.frame.reset_index()
    wide = {v: f.pivot(index="trip", columns="mode", values=v).reindex(columns=ALTS).to_numpy() for v in ("cost", "fast", "income", "rich")}
    avail = ~np.isnan(wide["cost"])

    def probs(t, cost, fast, rich):
        v = t[0] * cost + t[1] * fast + np.array([0.0, *t[2:5]]) + np.array([0.0, *t[5:8]]) * wide["income"] + np.array([0.0, *t[8:11]]) * rich
        e = np.where(avail, np.exp(np.where(avail, v, -np.inf) - np.nanmax(v, axis=1, keepdims=True)), 0.0)
        return e / e.sum(axis=1, keepdims=True)

    car = ALTS.index("car")
    on, off, up = wide["fast"].copy(), wide["fast"].copy(), wide["cost"].copy()
    on[:, car], off[:, car], up[:, car] = 1.0, 0.0, up[:, car] + 1
    draws = np.random.default_rng(5).multivariate_normal(fit.values, fit.cov, size=2000)
    sims = []
    for t in draws:
        fast_car = (probs(t, wide["cost"], on, wide["rich"]) - probs(t, wide["cost"], off, wide["rich"])).mean(0)  # fast of car
        cost_car = (probs(t, up, wide["fast"], wide["rich"]) - probs(t, wide["cost"], wide["fast"], wide["rich"])).mean(0)
        rich = (probs(t, wide["cost"], wide["fast"], np.ones_like(wide["rich"])) - probs(t, wide["cost"], wide["fast"], np.zeros_like(wide["rich"]))).mean(0)
        sims.append(np.concatenate([fast_car, cost_car, rich]))
    sd = np.std(sims, axis=0, ddof=1)
    se = np.concatenate([me.table("fast", "discrete change", "std err")["car"], me.table("cost", "discrete change", "std err")["car"],
                         me.table("rich", "discrete change", "std err")["rich"]])
    assert np.allclose(sd, se, rtol=0.08), np.c_[sd, se]


@pytest.fixture(scope="module")
def anes():
    sm = pytest.importorskip("statsmodels.api")
    frame = Dataset.example("anes96").frame.reset_index()
    xs = ["logpopul", "selfLR", "age", "educ", "income", "vote"]  # vote: 0/1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sm.MNLogit(frame["PID"], sm.add_constant(frame[xs])).fit(method="newton", maxiter=100, disp=0)
    pids = sorted(frame["PID"].unique())
    long = frame.loc[frame.index.repeat(len(pids)), ["respondent", "PID", *xs]].reset_index(drop=True)
    long["party"] = [f"p{int(v)}" for v in np.tile(pids, len(frame))]
    long["chosen"] = (long["party"] == [f"p{int(v)}" for v in long["PID"]]).astype(float)
    fit = choice_logit(Dataset.from_frame(long.drop(columns="PID"), entity=["respondent", "party"]), "chosen", case_variables=xs,
                       alternative_constants=True)
    return ref, fit, xs


def _nan_report(fit, at):
    """Where a NaN in the discrete changes starts: each step recomputed, with what is not finite in it (all-NaN changes
    were seen twice on Windows CI and never reproduced, so the failure itself has to say which step gave them)."""
    from pymetrix import clogit_effects as ce_module  # the function's module, for its steps
    ce = sys.modules[ce_module.__module__]

    def bad(a):
        a = np.asarray(a, float)
        return f"shape {a.shape}, {int((~np.isfinite(a)).sum())} not finite"

    theta = np.asarray(fit.values, float)
    lines = [f"at={at}", f"theta: {bad(theta)}", f"cov: {bad(fit.cov)}"]
    try:
        c = ce._cases(fit.design)
        base = c if at == "overall" else ce._at_means(c)
        beta, alpha, gamma = ce._split(theta, fit.design)
        lines += [f"x: {bad(np.nan_to_num(c.x))}", f"z: {bad(c.z)}", f"available: {int(c.available.sum())} of {c.available.size}",
                  f"beta: {bad(beta)}", f"alpha: {bad(alpha)}", f"gamma: {bad(gamma)}",
                  f"einsum: {bad(np.einsum('r,rna->na', beta, np.nan_to_num(base.x)))}", f"z @ gamma: {bad(base.z @ gamma)}",
                  f"P: {bad(ce.choice_probabilities(theta, fit.design, base))}"]
        plan = ce._discrete_plan(fit.design, list(fit.design["regressors"]) + list(fit.design["case_variables"]),
                                 [*c.x, *c.z.T], True, None)
        lines += [f"plan: {plan}"] + [f"{k}: {bad(v)}" for k, v in ce._choice_discrete(theta, fit.design, base, plan).items()]
        lines += [f"again, the whole result: {bad(fit.marginal_effects(at='average' if at == 'overall' else 'mean', dummy=True).table('vote', 'discrete change')['vote'])}"]
    except Exception as exc:  # noqa: BLE001 - the report must not hide the failure it reports
        lines.append(f"report stopped: {exc!r}")
    return "discrete changes not finite; " + "; ".join(lines)


def test_anes96_dummy_and_count_changes_against_statsmodels(anes):
    ref, fit, xs = anes
    assert fit.loglike == pytest.approx(ref.llf, abs=1e-6)
    for at in ("overall", "mean"):
        mine = fit.marginal_effects(at="average" if at == "overall" else "mean", dummy=True)
        assert set(mine.discrete) == {"vote"}
        theirs = ref.get_margeff(at=at, dummy=True)
        eff, se = np.asarray(theirs.margeff), np.asarray(theirs.margeff_se)
        got = mine.table("vote", "discrete change")["vote"].to_numpy()
        assert np.isfinite(got).all(), _nan_report(fit, at)  # a rare NaN on Windows CI only: say where it starts
        np.testing.assert_allclose(got, eff[-1], atol=1e-8)
        np.testing.assert_allclose(mine.table("vote", "discrete change", "std err")["vote"].to_numpy(), se[-1], rtol=1e-4, atol=1e-7)
        for i, v in enumerate(xs[:-1]):  # the others stay derivatives
            np.testing.assert_allclose(mine.table(v)[v].to_numpy(), eff[i], atol=1e-8)
        # a change of educ by 2 against statsmodels' own predictions (its get_margeff(count=True) fails for MNLogit)
        mine = fit.marginal_effects(at="average" if at == "overall" else "mean", changes={"educ": 2}).table("educ", "discrete change")["educ"]
        X = np.asarray(ref.model.exog, float)
        X = X if at == "overall" else X.mean(axis=0, keepdims=True)
        up = X.copy()
        up[:, 1 + xs.index("educ")] += 2
        theirs = (ref.model.predict(ref.params, up) - ref.model.predict(ref.params, X)).mean(axis=0)
        np.testing.assert_allclose(mine.to_numpy(), theirs, atol=1e-8)


def test_panel_logit_discrete_changes_at_a_fixed_effect():
    wages = Dataset.example("wage_panel")
    fit = conditional_logit(wages, "union", ["married", "exper"])
    with pytest.raises(SpecificationError, match="need a fixed_effect"):
        clogit_effects(fit, dummy=True)
    me = clogit_effects(fit, fixed_effect=0.5, dummy=True, changes={"exper": 2})
    assert me.discrete == {"exper": "x to x + 2", "married": "0 to 1"}
    X, b = fit.design["x"], np.asarray(fit.values)

    def lam(x):
        return 1 / (1 + np.exp(-(0.5 + x @ b)))

    got = me.frame.xs("discrete change at fixed effect 0.5", level="measure")["estimate"]
    on, off, up = X.copy(), X.copy(), X.copy()
    on[:, 0], off[:, 0], up[:, 1] = 1.0, 0.0, up[:, 1] + 2
    assert got.loc["married"].item() == pytest.approx((lam(on) - lam(off)).mean(), abs=1e-14)
    assert got.loc["exper"].item() == pytest.approx((lam(up) - lam(X)).mean(), abs=1e-14)
    assert any("Discrete changes at that fixed effect" in n for n in me.summary().notes)
    m = X.mean(axis=0, keepdims=True)
    at_mean = clogit_effects(fit, at="mean", fixed_effect=0.5, dummy=True).frame.xs("discrete change at fixed effect 0.5", level="measure")
    on, off = m.copy(), m.copy()
    on[:, 0], off[:, 0] = 1.0, 0.0
    assert at_mean["estimate"].loc["married"].item() == pytest.approx((lam(on) - lam(off)).item(), abs=1e-14)


def test_the_dialog_records_the_call_and_the_script_replays(trips):
    from pymetrix.commands import COMMANDS, Session
    from pymetrix.forms import form_spec

    cls = COMMANDS["ClogitMarginalEffects"].cls
    assert {"dummy", "changes"} <= {f.name for f in form_spec(cls)}
    s = Session()
    s.objects["trips"] = trips
    s.run(COMMANDS["FitChoiceLogit"].cls("chosen", "cost, fast", data="trips", alternative_constants=True, case_variables="income, rich"))
    s.run(cls("choice", at="mean", dummy=True, changes="cost: 1, income: -0.5", name="dc"))
    assert s.history[-1].code == "dc = clogit_effects(choice, at='mean', dummy=True, changes={'cost': 1.0, 'income': -0.5})"
    direct = clogit_effects(choice_logit(trips, "chosen", ["cost", "fast"], alternative_constants=True, case_variables=["income", "rich"]),
                            at="mean", dummy=True, changes={"cost": 1.0, "income": -0.5})
    pd.testing.assert_frame_equal(s.objects["dc"].frame, direct.frame)
    assert cls("choice", changes="cost 1").validate(s) and cls("choice", changes="cost: x").validate(s)
    assert cls("choice", changes="cost: 0").validate(s) and not cls("choice", changes="cost: 2; fast: 1").validate(s)
    namespace = {"trips": trips}
    exec(s.script(), namespace)  # noqa: S102 - the generated script reproduces the effects
    pd.testing.assert_frame_equal(namespace["dc"].frame, direct.frame)
