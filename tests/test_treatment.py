"""Treatment effects, checked independently: two-way fixed effects against the difference in differences of means in a
balanced two-by-two design; the event study's leads and lags against the simulated truth; the RD jump against weighted
least squares done by hand, and its cross-validated bandwidth against the criterion computed by brute force;
propensity weighting against subclassification (exact with one binary covariate); synthetic-control weights recovered
where the treated unit is an exact mix of donors."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, did, event_study, rdd, synthetic_control, treatment_effect
from pymetrix.errors import SpecificationError
from pymetrix.treatment import cv_bandwidth


def _panel(frame, unit="unit", time="year"):
    return Dataset.from_frame(frame, entity=unit, time=time, freq="Y")


def test_twfe_is_the_difference_in_differences_of_means():
    rng = np.random.default_rng(1)
    rows = []
    for u in range(40):
        treated = u < 20
        for t in (2000, 2001):
            rows.append({"unit": u, "year": t, "d": int(treated and t == 2001), "y": rng.normal() + 3 * (treated and t == 2001)})
    frame = pd.DataFrame(rows)
    result = did(_panel(frame), "y", "d")
    g = frame.assign(treated=frame["unit"] < 20).groupby(["treated", "year"])["y"].mean()
    by_hand = (g[True, 2001] - g[True, 2000]) - (g[False, 2001] - g[False, 2000])
    assert result.estimate == pytest.approx(by_hand, rel=1e-10)
    assert result.means.loc["treated", "after"] == pytest.approx(g[True, 2001])


def test_the_policy_panel_and_its_event_study():
    panel = Dataset.example("policy_panel")
    result = did(panel, "y", "d", controls=["x"])
    assert abs(result.estimate - 2.0) < 3 * result.se
    es = event_study(panel, "y", "d", window=(-4, 4), controls=["x"])
    c = es.coefficients()
    assert c.loc[-1, "estimate"] == 0.0 and list(c.index) == list(range(-4, 5))
    leads, lags = c.loc[[-4, -3, -2]], c.loc[[0, 1, 2, 3, 4]]
    assert np.all(np.abs(leads["estimate"]) < 3 * leads["std err"])  # no effect before adoption
    assert np.all(np.abs(lags["estimate"] - 2.0) < 3 * lags["std err"])  # the true effect after
    with pytest.raises(SpecificationError, match="window"):
        event_study(panel, "y", "d", window=(1, 4))


def test_rd_is_weighted_least_squares_and_the_bandwidth_minimises_cv():
    data = Dataset.example("rd_sample")
    rd = rdd(data, "y", "score", bandwidth=0.4)
    x, y = data.frame["score"].to_numpy(), data.frame["y"].to_numpy()
    w = np.clip(1 - np.abs(x) / 0.4, 0, None)
    keep = w > 0
    X = np.column_stack([np.ones(keep.sum()), x[keep], (x[keep] >= 0), (x[keep] >= 0) * x[keep]])
    b = np.linalg.lstsq(X * np.sqrt(w[keep])[:, None], y[keep] * np.sqrt(w[keep]), rcond=None)[0]
    assert rd.estimate == pytest.approx(b[2], rel=1e-10)
    assert abs(rd.estimate - 1.5) < 3 * rd.se
    grid = np.array([0.2, 0.4, 0.6])
    h, scores = cv_bandwidth(x, y, 0.0, grid)
    errors = []  # the criterion at h = 0.4, by brute force: each point near the cutoff predicted from its far side
    for i in range(len(x)):
        left = x[i] < 0
        side = x < 0 if left else x >= 0
        if not (x[i] >= np.quantile(x[side], 0.5) if left else x[i] <= np.quantile(x[side], 0.5)):
            continue
        pick = side & ((x < x[i]) & (x >= x[i] - 0.4) if left else (x > x[i]) & (x <= x[i] + 0.4))
        if pick.sum() < 3:
            continue
        intercept = np.polyfit(x[pick] - x[i], y[pick], 1)[1]
        errors.append(y[i] - intercept)
    assert scores[1] == pytest.approx(np.mean(np.square(errors)), rel=1e-10)
    assert scores[list(grid).index(h)] == scores.min()
    chosen = rdd(data, "y", "score")
    assert "cross-validation" in chosen.summary().notes[0] and abs(chosen.estimate - 1.5) < 3 * chosen.se


def test_ipw_is_subclassification_with_one_binary_covariate():
    rng = np.random.default_rng(4)
    n = 4000
    z = rng.integers(0, 2, n)
    d = (rng.random(n) < np.where(z == 1, 0.7, 0.3)).astype(float)
    y = 1 + 2 * z + 1.5 * d + rng.normal(size=n)
    data = Dataset.from_frame(pd.DataFrame({"i": range(n), "y": y, "d": d, "z": z.astype(float)}), entity="i")
    ate = treatment_effect(data, "y", "d", ["z"], reps=0)
    cells = pd.DataFrame({"y": y, "d": d, "z": z}).groupby("z")
    by_cell = sum((g["y"][g["d"] == 1].mean() - g["y"][g["d"] == 0].mean()) * len(g) / n for _, g in cells)
    assert ate.estimate == pytest.approx(by_cell, rel=1e-8)  # saturated logit weights: exactly the stratified difference
    att = treatment_effect(data, "y", "d", ["z"], estimand="ATT", reps=0)
    treated_share = cells.apply(lambda g: g["d"].sum()) / d.sum()
    by_cell_att = sum((g["y"][g["d"] == 1].mean() - g["y"][g["d"] == 0].mean()) * treated_share[k] for k, g in cells)
    assert att.estimate == pytest.approx(by_cell_att, rel=1e-8)
    aipw = treatment_effect(data, "y", "d", ["z"], method="aipw", reps=30)
    assert abs(aipw.estimate - 1.5) < 4 * aipw.se and aipw.reps == 30
    with pytest.raises(SpecificationError, match="0 or 1"):
        treatment_effect(data, "d", "y", ["z"])  # y as the treatment: not 0/1


def test_synthetic_control_recovers_exact_weights():
    rng = np.random.default_rng(2)
    years = np.arange(1990, 2010)
    donors = {k: np.cumsum(rng.normal(size=len(years))) + 10 * k for k in range(2, 8)}
    treated = 0.6 * donors[3] + 0.4 * donors[6]  # an exact mix of two donors
    treated = treated + np.where(years >= 2003, -2.0, 0.0)
    rows = [{"unit": 1, "year": int(t), "y": float(v)} for t, v in zip(years, treated)]
    rows += [{"unit": k, "year": int(t), "y": float(v)} for k, path in donors.items() for t, v in zip(years, path)]
    sc = synthetic_control(_panel(pd.DataFrame(rows)), "y", treated=1, start=2003)
    assert sc.weights[3] == pytest.approx(0.6, abs=1e-4) and sc.weights[6] == pytest.approx(0.4, abs=1e-4)
    assert sc.effect == pytest.approx(-2.0, abs=1e-3)
    example = synthetic_control(Dataset.example("synth_panel"), "y", treated=1, start=2005)
    assert -4.0 < example.effect < -2.4 and example.p_value <= 0.1


def test_the_dialogs_record_the_calls():
    from pymetrix.commands import DifferenceInDifferences, FitEventStudy, FitSyntheticControl, RegressionDiscontinuity, Session

    s = Session()
    s.objects["panel"] = Dataset.example("policy_panel")
    s.objects["rd"] = Dataset.example("rd_sample")
    s.objects["sp"] = Dataset.example("synth_panel")
    s.run(DifferenceInDifferences("y", "d", data="panel"))
    s.run(FitEventStudy("y", "d", data="panel", first=-3, last=3))
    s.run(RegressionDiscontinuity("y", "score", data="rd", bandwidth=0.4))
    s.run(FitSyntheticControl("y", "1", "2005", data="sp"))
    assert s.history[0].code == "did_result = did(panel, 'y', 'd')"
    assert s.history[1].code.startswith("event = event_study(panel, 'y', 'd', window=")
    assert s.history[2].code == "rd = rdd(rd, 'y', 'score', bandwidth=0.4)"
    assert type(s.objects["synth"]).__name__ == "SyntheticControl"
