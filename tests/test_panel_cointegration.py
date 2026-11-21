"""Panel cointegration tests, checked independently: Pedroni's moments simulated at large T against his Table 2; the
group ADF term, Kao's pooled Dickey-Fuller regression and Westerlund's error-correction t-statistics against
statsmodels' regressions; and each test's size under the null and power under cointegration in repeated samples."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset
from pymetrix.stats import run_test

sm = pytest.importorskip("statsmodels.api")

PEDRONI = ("panel v", "panel rho", "panel PP t", "panel ADF t", "group rho", "group PP t", "group ADF t")


def _panel(N, T, cointegrated, m=1, seed=0):
    """y = 1 + sum x + u, the x independent random walks; u an AR(0.5) when cointegrated, else a random walk."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N):
        X = np.cumsum(rng.normal(size=(T, m)), axis=0)
        e = rng.normal(size=T)
        u = np.zeros(T)
        for t in range(T):
            u[t] = (0.5 * u[t - 1] if t else 0.0) + e[t]
        y = 1 + X.sum(axis=1) + (u if cointegrated else np.cumsum(rng.normal(size=T)))
        rows += [{"id": f"c{i:02d}", "t": 1900 + t, "y": y[t], **{f"x{j}": X[t, j] for j in range(m)}} for t in range(T)]
    return Dataset.from_frame(pd.DataFrame(rows), entity="id", time="t", freq="Y")


def _p(result, name):
    return float(result.details[name].split("p = ")[1].rstrip(")"))


def test_pedroni_moments_reproduce_his_table_2():
    from pymetrix.stats import panel_cointegration as pc

    mean_v, C, group = pc._pedroni_null(1000, 1, "c", 0, 0, 20_000, 0)
    keys = sorted({k for ks in pc._PANEL_KEYS.values() for k in ks})
    mean = dict(zip(keys, mean_v))
    cov = {(i, j): C[a, b] for a, i in enumerate(keys) for b, j in enumerate(keys)}
    # Pedroni (1999), Table 2, one regressor with a constant: (mean, variance)
    table = {"panel v": (8.62, 60.75), "panel rho": (-6.02, 31.27), "panel PP t": (-1.73, 0.93), "panel ADF t": (-1.73, 0.93),
             "group rho": (-9.05, 35.98), "group PP t": (-2.03, 0.66), "group ADF t": (-2.03, 0.66)}
    for name, (mu, var) in table.items():
        got = pc._delta_moments(name, mean, cov) if name in pc._PANEL_KEYS else group[name]
        assert got[0] == pytest.approx(mu, abs=0.1) and got[1] == pytest.approx(var, rel=0.06), name


def test_pedronis_group_adf_term_is_the_residuals_adf_t():
    from pymetrix.stats.panel_cointegration import _pedroni_parts, _statistics

    data = _panel(1, 80, False, seed=3)
    f = data.frame.reset_index()
    y, x = f["y"].to_numpy(), f[["x0"]].to_numpy()
    p = 2
    e = sm.OLS(y, sm.add_constant(x)).fit().resid
    de = np.diff(e)
    Z = np.column_stack([e[p:-1]] + [de[p - j:len(de) - j] for j in range(1, p + 1)])
    ref = sm.OLS(de[p:], Z).fit()
    ns = len(de) - p
    term = float(_statistics(_pedroni_parts(y[None], x[None], "c", p, 3))["group ADF t"][0])
    assert term == pytest.approx(ref.tvalues[0] * np.sqrt(ns / ref.df_resid), rel=1e-10)  # s*^2 over T, not T - k


def test_pedroni_size_and_power():
    reject = np.zeros(7)
    for seed in range(80):
        r = run_test(_panel(15, 60, False, seed=seed), "pedroni", "y, x0", lags=1, replications=5000)
        reject += [_p(r, name) < 0.05 for name in PEDRONI]
    size = dict(zip(PEDRONI, reject / 80))
    assert all(v < 0.16 for v in size.values()), size  # panel v and rho over-reject a little in short panels
    assert size["group PP t"] < 0.1 and size["group ADF t"] < 0.1
    r = run_test(_panel(15, 60, True, m=2, seed=1), "pedroni", "y, x0, x1", replications=5000)
    assert all(_p(r, name) < 0.01 for name in PEDRONI) and r.details["regressors"] == 2
    assert r.statistic == pytest.approx(float(r.details["group ADF t"].split(" ")[0]), abs=1e-3)  # the headline: group ADF t
    with pytest.raises(ValueError, match="two or more"):
        run_test(_panel(3, 30, True), "pedroni", "y")


def test_kao_pooled_dickey_fuller_by_hand():
    data = _panel(6, 40, False, seed=2)
    r = run_test(data, "kao", "y, x0", lags=0)
    f = data.frame.reset_index()
    dummies = pd.get_dummies(f["id"], dtype=float)  # the within regression as dummies, by statsmodels
    e = sm.OLS(f["y"].to_numpy(), pd.concat([f[["x0"]], dummies], axis=1)).fit().resid.to_numpy().reshape(6, 40)
    df = sm.OLS(e[:, 1:].reshape(-1), e[:, :-1].reshape(-1)).fit()
    assert r.details["pooled rho"] == pytest.approx(df.params[0], rel=1e-10)
    N, T = 6, 40
    t_rho = (df.params[0] - 1) / df.bse[0]
    assert float(r.details["DF t"].split(" ")[0]) == pytest.approx(np.sqrt(1.25) * t_rho + np.sqrt(1.875 * N), abs=1e-3)
    rho_stat = (np.sqrt(N) * T * (df.params[0] - 1) + 3 * np.sqrt(N)) / np.sqrt(10.2)
    assert float(r.details["DF rho"].split(" ")[0]) == pytest.approx(rho_stat, abs=1e-3)


def test_kao_size_and_power():
    names = ("ADF", "DF rho*", "DF t*", "DF rho", "DF t")
    reject = np.zeros(5)
    for seed in range(150):
        r = run_test(_panel(15, 60, False, seed=seed), "kao", "y, x0")
        reject += [_p(r, name) < 0.05 for name in names]
    size = dict(zip(names, reject / 150))
    assert size["ADF"] < 0.1 and size["DF t*"] < 0.1 and size["DF rho*"] < 0.12, size
    assert run_test(_panel(15, 60, True, seed=1), "kao", "y, x0").p_value < 0.001


def test_westerlund_gt_is_the_average_error_correction_t():
    data = _panel(4, 50, True, seed=4)
    p, q = 1, 1
    r = run_test(data, "westerlund", "y, x0", lags=p, leads=q, replications=2000)
    ts = []
    for _, g in data.frame.groupby(level="id"):
        y, x = g["y"].to_numpy(), g["x0"].to_numpy()
        dy, dx = np.diff(y), np.diff(x)
        t = np.arange(p + 1, len(y) - q)  # the periods with dy_t-p and dx_t+q
        X = pd.DataFrame({"y_lag": y[t - 1], "x_lag": x[t - 1], "dy_lag1": dy[t - 2], "dx_lead1": dx[t], "dx": dx[t - 1],
                          "dx_lag1": dx[t - 2], "const": 1.0})
        ts.append(sm.OLS(dy[t - 1], X).fit().tvalues["y_lag"])
    assert r.details["Gt unstandardised"] == pytest.approx(np.mean(ts), rel=1e-10)


def test_westerlund_size_and_power():
    names = ("Gt", "Ga", "Pt", "Pa")
    reject = np.zeros(4)
    for seed in range(100):
        r = run_test(_panel(15, 60, False, seed=seed), "westerlund", "y, x0", replications=5000)
        reject += [_p(r, name) < 0.05 for name in names]
    size = dict(zip(names, reject / 100))
    assert all(v < 0.13 for v in size.values()), size
    r = run_test(_panel(15, 60, True, seed=1), "westerlund", "y, x0", trend="ct", replications=5000)
    assert all(_p(r, name) < 0.01 for name in names)
