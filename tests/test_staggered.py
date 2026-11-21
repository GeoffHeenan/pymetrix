"""Staggered difference-in-differences, checked independently: an ATT(g, t) by hand from group means; the true effects
recovered when they grow with time since adoption (and no effect before it); Sun and Abraham equal to Callaway and
Sant'Anna with never-treated controls in a balanced panel; Goodman-Bacon's weights summing to one and their weighted
estimates to the two-way fixed effects coefficient, which matches linearmodels' PanelOLS with unit and period
effects; the two-way estimate's bias in this setting."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, bacon_decomposition, staggered_did
from pymetrix.errors import SpecificationError


def _panel(N=300, T=12, seed=0, never=True):
    rng = np.random.default_rng(seed)
    choices, p = ([4, 7, 9, np.inf], [0.25] * 4) if never else ([4, 7, 9], [1 / 3] * 3)
    cohort = rng.choice(choices, N, p=p)
    alpha, lam = rng.normal(0, 1, N), np.cumsum(rng.normal(0, 0.3, T))
    rows = []
    for i in range(N):
        for t in range(T):
            e = t - cohort[i]
            effect = 1.0 + 0.5 * e if e >= 0 else 0.0  # grows with time since adoption
            rows.append({"unit": i, "year": 2000 + t, "d": float(e >= 0), "y": alpha[i] + lam[t] + effect + rng.normal(0, 0.5)})
    return Dataset.from_frame(pd.DataFrame(rows), entity="unit", time="year", freq="Y"), cohort


def test_att_by_hand_and_the_effects_recovered():
    data, cohort = _panel()
    cs = staggered_did(data, "y", "d", draws=99)
    Y = data.frame["y"].unstack("year").to_numpy()
    g, t = 7, 9  # the 2007 cohort in 2009, against the never treated, from 2006
    by_hand = (Y[cohort == g, t] - Y[cohort == g, g - 1]).mean() - (Y[np.isinf(cohort), t] - Y[np.isinf(cohort), g - 1]).mean()
    row = cs.att[(cs.att["event"] == t - g) & (cs.att["cohort"].astype(str) == "2007")]
    assert row["ATT"].iloc[0] == pytest.approx(by_hand, rel=1e-12)
    e = cs.event
    post = e.loc[0:5]
    assert np.all(np.abs(post["estimate"] - (1 + 0.5 * post.index.to_numpy())) < 4 * post["se"])
    pre = e.loc[:-2]
    assert np.all(np.abs(pre["estimate"]) < 4 * pre["se"])  # parallel trends hold here
    notyet = staggered_did(data, "y", "d", control="notyet", draws=49)
    assert notyet.overall()["ATT"] == pytest.approx(cs.overall()["ATT"], abs=0.15)


def test_sun_abraham_equals_callaway_santanna():
    data, _ = _panel(N=200, seed=1)
    cs = staggered_did(data, "y", "d", draws=0)
    sa = staggered_did(data, "y", "d", method="sun_abraham")
    assert sa.event["estimate"].to_numpy() == pytest.approx(cs.event["estimate"].reindex(sa.event.index).to_numpy(), abs=1e-10)
    assert (sa.event["se"] > 0).all()


def test_the_bacon_decomposition():
    lm = pytest.importorskip("linearmodels")
    data, cohort = _panel(N=200, seed=2)
    b = bacon_decomposition(data, "y", "d")
    c = b.comparisons
    assert c["weight"].sum() == pytest.approx(1.0, abs=1e-12)
    assert (c["weight"] * c["estimate"]).sum() == pytest.approx(b.twfe, rel=1e-12)
    f = data.frame.copy()
    f.index = pd.MultiIndex.from_arrays([f.index.get_level_values(0), f.index.get_level_values(1).year])
    ref = lm.PanelOLS(f["y"], f[["d"]], entity_effects=True, time_effects=True).fit()
    assert b.twfe == pytest.approx(ref.params["d"], rel=1e-10)
    cs = staggered_did(data, "y", "d", draws=0)
    assert b.twfe < cs.overall()["ATT"] - 0.3  # two-way fixed effects understates growing effects
    assert b.by_type().loc["later vs earlier treated", "average estimate"] < b.by_type().loc["treated vs never treated", "average estimate"]


def test_refusals_and_the_dialogs():
    data, _ = _panel(N=60, seed=3, never=False)
    with pytest.raises(SpecificationError, match="never treated"):
        staggered_did(data, "y", "d")
    assert staggered_did(data, "y", "d", control="notyet", draws=0).units == 60
    bad = data.frame.copy()
    bad.iloc[11, bad.columns.get_loc("d")] = 0.0  # unit 0, treated by its last year, stops being treated
    with pytest.raises(SpecificationError, match="stay 1"):
        staggered_did(Dataset.from_frame(bad.reset_index(), entity="unit", time="year", freq="Y"), "y", "d", control="notyet")
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS
    from pymetrix.views import sections

    full, _ = _panel(N=80, seed=4)
    s = Session()
    s.objects["data"] = full
    s.run(COMMANDS["FitStaggeredDiD"].cls("y", "d", draws=20))
    assert s.history[-1].code == "staggered = staggered_did(data, 'y', 'd', draws=20)"
    s.run(COMMANDS["BaconDecompose"].cls("y", "d"))
    assert [t for t, _ in sections(s.objects["bacon"])][0] == "Summary"


def _conditional_panel(N=1500, T=8, seed=0):
    """Treated units have high x, and trends rise with x: parallel trends hold only given x. The true effect is 1."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N)
    cohort = np.where(rng.random(N) < 1 / (1 + np.exp(-x)), rng.choice([3, 5], N), np.inf)
    rows = [{"unit": i, "year": 2000 + t, "d": float(t >= cohort[i]), "x": x[i], "zero": 0.0,
             "y": x[i] + 0.5 * x[i] * t + (1.0 if t >= cohort[i] else 0.0) + rng.normal(0, 0.5)} for i in range(N) for t in range(T)]
    return Dataset.from_frame(pd.DataFrame(rows), entity="unit", time="year", freq="Y"), x, cohort


def test_covariates_give_conditional_parallel_trends():
    data, x, cohort = _conditional_panel()
    plain = staggered_did(data, "y", "d", draws=0)
    assert plain.overall()["ATT"] > 1.5  # biased: the treated trend faster
    for estimator in ("dr", "or", "ipw"):
        fit = staggered_did(data, "y", "d", covariates=["x"], estimator=estimator, draws=0)
        assert fit.overall()["ATT"] == pytest.approx(1.0, abs=0.1)
        assert abs(fit.event.loc[-2, "estimate"]) < 0.1  # no pre-trend once x is held fixed
    Y = data.frame["y"].unstack("year").to_numpy()  # outcome regression by hand for the 2003 cohort in 2005
    treated, ctrl = cohort == 3, np.isinf(cohort)
    dy_t, dy_c = Y[treated, 5] - Y[treated, 2], Y[ctrl, 5] - Y[ctrl, 2]
    m = np.polyfit(x[ctrl], dy_c, 1)
    by_hand = np.mean(dy_t - np.polyval(m, x[treated]))
    orr = staggered_did(data, "y", "d", covariates=["x"], estimator="or", draws=0).att
    assert orr[(orr["cohort"].astype(str) == "2003") & (orr["event"] == 2)]["ATT"].iloc[0] == pytest.approx(by_hand, rel=1e-10)


def test_uninformative_covariates_change_nothing():
    data, _, _ = _conditional_panel(N=400, seed=1)
    plain = staggered_did(data, "y", "d", draws=0).event["estimate"]
    for estimator in ("dr", "or", "ipw"):  # a covariate that is zero for everyone: the unconditional estimate exactly
        fit = staggered_did(data, "y", "d", covariates=["zero"], estimator=estimator, draws=0).event["estimate"]
        assert fit.to_numpy() == pytest.approx(plain.to_numpy(), abs=1e-10)
    with pytest.raises(SpecificationError, match="Callaway"):
        staggered_did(data, "y", "d", method="sun_abraham", covariates=["x"])
