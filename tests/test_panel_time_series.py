"""Mean group, CCE and pooled mean group, checked independently: the mean group as the average of each unit's
statsmodels OLS; in a common-factor panel where the factor loadings of y and x are correlated, MG biased and CCEMG and
CCEP near the true mean slope, with standard errors against the spread over repeated samples; CCEMG as each unit's OLS
on x and the cross-section averages by hand; the pooled mean group recovering a common long run, equal for one unit to
that unit's own ARDL long run (-beta / phi), its concentrated likelihood by hand, and Hausman's size for a true common long run."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, cce, mean_group, pooled_mean_group

sm = pytest.importorskip("statsmodels.api")


def _factor_panel(N=30, T=60, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=T).cumsum() * 0.3
    rows, slopes = [], []
    for i in range(N):
        g, gx = rng.normal(1, 0.5), rng.normal(1, 0.5)
        x = gx * f + rng.normal(size=T)
        b = 1.0 + rng.normal() * 0.2
        slopes.append(b)
        y = 0.5 + b * x + g * f + rng.normal(size=T)
        rows += [(i, 2000 + t, y[t], x[t]) for t in range(T)]
    data = Dataset.from_frame(pd.DataFrame(rows, columns=["u", "t", "y", "x"]), entity="u", time="t", freq="Y")
    return data, np.mean(slopes)


def test_mean_group_and_cce_by_hand():
    data, true = _factor_panel()
    fr = data.frame.reset_index()
    mg = mean_group(data, "y", ["x"])
    assert mg.params == pytest.approx(np.mean([sm.OLS(g["y"], sm.add_constant(g["x"])).fit().params for _, g in fr.groupby("u")], axis=0), abs=1e-12)
    avg = fr.groupby("t")[["y", "x"]].mean().rename(columns={"y": "ybar", "x": "xbar"})
    merged = fr.merge(avg, left_on="t", right_index=True)
    by_hand = [sm.OLS(g["y"], sm.add_constant(g[["x", "ybar", "xbar"]])).fit().params["x"] for _, g in merged.groupby("u")]
    ccemg = cce(data, "y", ["x"])
    assert ccemg.params[0] == pytest.approx(np.mean(by_hand), abs=1e-10)
    assert abs(mg.params[1] - true) > 0.3  # the common factor biases the mean group
    assert ccemg.params[0] == pytest.approx(true, abs=0.1) and cce(data, "y", ["x"], pooled=True).params[0] == pytest.approx(true, abs=0.1)


def test_cce_standard_errors_against_the_spread():
    mg_est, mg_se, p_est, p_se = [], [], [], []
    for seed in range(30):
        data, _ = _factor_panel(N=25, T=40, seed=100 + seed)
        a, b = cce(data, "y", ["x"]), cce(data, "y", ["x"], pooled=True)
        mg_est.append(a.params[0]); mg_se.append(a.se[0]); p_est.append(b.params[0]); p_se.append(b.se[0])
    assert np.mean(mg_se) == pytest.approx(np.std(mg_est, ddof=1), rel=0.4)
    assert np.mean(p_se) == pytest.approx(np.std(p_est, ddof=1), rel=0.4)


def _ardl_panel(N=20, T=80, seed=1, theta=2.0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N):
        phi = -rng.uniform(0.2, 0.6)
        x = rng.normal(size=T).cumsum()
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = y[t - 1] + phi * (y[t - 1] - theta * x[t - 1]) + 0.3 * (x[t] - x[t - 1]) + rng.normal() * 0.5
        rows += [(i, 1900 + t, y[t], x[t]) for t in range(T)]
    return pd.DataFrame(rows, columns=["u", "t", "y", "x"])


def test_the_pooled_mean_group():
    frame = _ardl_panel()
    data = Dataset.from_frame(frame, entity="u", time="t", freq="Y")
    pmg = pooled_mean_group(data, "y", ["x"])
    assert pmg.theta[0] == pytest.approx(2.0, abs=3 * np.sqrt(pmg.theta_cov[0, 0]))
    assert pmg.phi.mean() == pytest.approx(-0.4, abs=0.1)  # the adjustment speeds are drawn from U(0.2, 0.6)
    ll = 0.0
    for _, g in frame.groupby("u"):  # the concentrated likelihood by hand, unit by unit
        y, x = g["y"].to_numpy(), g["x"].to_numpy()
        dy, dx = np.diff(y), np.diff(x)
        Z = np.column_stack([y[1:-1] - pmg.theta[0] * x[1:-1], np.ones(len(dy) - 1), dx[1:]])
        e = dy[1:] - Z @ np.linalg.lstsq(Z, dy[1:], rcond=None)[0]
        ll += -len(e) / 2 * (np.log(2 * np.pi) + np.log(e @ e / len(e)) + 1)
    assert pmg.loglike == pytest.approx(ll, rel=1e-10)
    one = pooled_mean_group(Dataset.from_frame(frame[frame["u"] == 0], entity="u", time="t", freq="Y"), "y", ["x"])
    g = frame[frame["u"] == 0]
    y, x = g["y"].to_numpy(), g["x"].to_numpy()
    dy, dx = np.diff(y), np.diff(x)
    b = sm.OLS(dy[1:], np.column_stack([y[1:-1], x[1:-1], np.ones(len(dy) - 1), dx[1:]])).fit().params
    assert one.theta[0] == pytest.approx(-b[1] / b[0], rel=1e-6)  # one unit: its own long run


def test_hausman_keeps_its_size():
    p = [pooled_mean_group(Dataset.from_frame(_ardl_panel(seed=200 + k), entity="u", time="t", freq="Y"), "y", ["x"]).hausman().p_value
         for k in range(60)]
    assert np.mean(np.array(p) < 0.05) == pytest.approx(0.05, abs=0.06)
