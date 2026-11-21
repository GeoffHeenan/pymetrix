"""Forecast evaluation (Elliott and Timmermann 2016, chapters 14-18): scoring rules, PIT and interval tests, efficiency,
encompassing, Giacomini-White, Clark-West, SPA and the model confidence set, backtests' densities and rolling windows,
and forecast combination. Each is checked against an independent calculation: numerical integration, brute force,
statsmodels regressions, hand counts, or simulation."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats as sps

from pymetrix import VAR, Bayes, Dataset, Equation, Minnesota, backtest, combine_forecasts
from pymetrix.errors import SpecificationError
from pymetrix.stats import (berkowitz, christoffersen, clark_west, crps_mixture, crps_normal, crps_sample, encompassing, giacomini_white,
                            log_score_mixture, log_score_normal, log_score_sample, mincer_zarnowitz, pit_mixture, pit_normal, pit_sample)

sm = pytest.importorskip("statsmodels.api")


def _crps_by_integration(cdf, y, lo, hi):
    """The CRPS from its definition: the integral of (F(x) - 1{x >= y})^2."""
    left = integrate.quad(lambda x: cdf(x) ** 2, lo, y, limit=200)[0]
    right = integrate.quad(lambda x: (1 - cdf(x)) ** 2, y, hi, limit=200)[0]
    return left + right


# ---------------------------------------------------------------------------------------------
# scoring rules


@pytest.mark.parametrize("y, m, s", [(0.3, 0.0, 1.0), (-2.5, 1.0, 0.5), (10.0, 9.0, 3.0)])
def test_normal_scores_match_their_definitions(y, m, s):
    assert crps_normal(y, m, s) == pytest.approx(_crps_by_integration(lambda x: sps.norm.cdf(x, m, s), y, m - 40 * s, m + 40 * s), abs=1e-8)
    assert log_score_normal(y, m, s) == pytest.approx(sps.norm.logpdf(y, m, s), abs=1e-12)
    assert pit_normal(y, m, s) == pytest.approx(sps.norm.cdf(y, m, s), abs=1e-14)


def test_sample_scores():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 2.0, 400)
    brute = np.mean(np.abs(x - 0.5)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))  # all n^2 pairs
    assert crps_sample(0.5, x) == pytest.approx(brute, abs=1e-12)
    big = rng.normal(1.0, 2.0, 200_000)
    assert crps_sample(0.5, big) == pytest.approx(float(crps_normal(0.5, 1.0, 2.0)), abs=0.01)  # converges to the normal's
    assert pit_sample(0.5, x) == pytest.approx(np.mean(x < 0.5)) and pit_sample(1.0, [0.0, 1.0, 2.0, 3.0]) == pytest.approx(0.375)
    assert log_score_sample(0.5, big[:20000]) == pytest.approx(sps.norm.logpdf(0.5, 1.0, 2.0), abs=0.02)
    with pytest.raises(SpecificationError):
        crps_sample(0.0, [1.0])


def test_mixture_scores_match_their_definitions():
    w, m, s = np.array([0.2, 0.5, 0.3]), np.array([-1.0, 0.5, 2.0]), np.array([0.5, 1.0, 2.0])
    cdf = lambda x: float(np.sum(w * sps.norm.cdf(x, m, s)))  # noqa: E731
    for y in (-2.0, 0.3, 4.0):
        assert crps_mixture(y, w, m, s) == pytest.approx(_crps_by_integration(cdf, y, -40, 40), abs=1e-7)
        assert log_score_mixture(y, w, m, s) == pytest.approx(np.log(np.sum(w * sps.norm.pdf(y, m, s))), abs=1e-12)
        assert pit_mixture(y, w, m, s) == pytest.approx(cdf(y), abs=1e-14)
    assert crps_mixture(0.3, [1.0], [0.0], [1.0]) == pytest.approx(float(crps_normal(0.3, 0.0, 1.0)), abs=1e-12)  # one component
    with pytest.raises(SpecificationError, match="non-negative"):
        crps_mixture(0.0, [1.5, -0.5], [0, 1], [1, 1])


def test_berkowitz_is_the_likelihood_ratio_of_an_ar1_and_holds_its_size():
    rng = np.random.default_rng(1)
    u = sps.norm.cdf(rng.normal(0.6, 1.6, 80))  # PITs of forecasts that are biased and too narrow
    result = berkowitz(u)
    z = sps.norm.ppf(u)
    ols = sm.OLS(z[1:], sm.add_constant(z[:-1])).fit()
    ll0 = float(np.sum(sps.norm.logpdf(z[1:])))
    assert result.statistic == pytest.approx(2 * (ols.llf - ll0), rel=1e-10) and result.df == 3 and result.p_value < 0.01
    rejections = np.mean([berkowitz(rng.uniform(size=100)).p_value < 0.05 for _ in range(400)])
    assert 0.02 < rejections < 0.09  # the right size for independent uniform PITs


def test_christoffersen_counts():
    hits = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    n, n1, p = 20, 6, 0.1
    pi = n1 / n
    lr_uc = -2 * ((n - n1) * np.log(1 - p) + n1 * np.log(p) - (n - n1) * np.log(1 - pi) - n1 * np.log(pi))
    # transitions, by hand: 00 = 10, 01 = 3, 10 = 3, 11 = 3
    n00, n01, n10, n11 = 10, 3, 3, 3
    p01, p11, p2 = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / 19
    lr_ind = -2 * ((n00 + n10) * np.log(1 - p2) + (n01 + n11) * np.log(p2) - n00 * np.log(1 - p01) - n01 * np.log(p01)
                   - n10 * np.log(1 - p11) - n11 * np.log(p11))
    result = christoffersen(hits, 0.9)
    assert result.details["coverage (Kupiec) statistic"] == pytest.approx(lr_uc, abs=1e-10)
    assert result.details["independence statistic"] == pytest.approx(lr_ind, abs=1e-10)
    assert result.statistic == pytest.approx(lr_uc + lr_ind, abs=1e-10) and result.p_value == pytest.approx(sps.chi2.sf(lr_uc + lr_ind, 2))


# ---------------------------------------------------------------------------------------------
# tests on point forecasts, against statsmodels


@pytest.fixture(scope="module")
def forecasts():
    rng = np.random.default_rng(7)
    n = 120
    x = rng.normal(size=n)
    y = 0.8 * x + rng.normal(scale=0.6, size=n)
    good = pd.Series(0.8 * x, name="good")
    biased = pd.Series(0.3 + 1.3 * x, name="biased")
    return pd.Series(y, name="y"), good, biased, x


def test_mincer_zarnowitz_is_a_hac_wald_test(forecasts):
    y, _, biased, _ = forecasts
    for h in (1, 4):
        result = mincer_zarnowitz(y, biased, horizon=h)
        ols = sm.OLS(y.to_numpy(), sm.add_constant(biased.to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": h - 1, "use_correction": False})
        wald = ols.wald_test((np.eye(2), np.array([0.0, 1.0])), use_f=False, scalar=True)
        assert result.statistic == pytest.approx(float(wald.statistic), rel=1e-9) and result.details["slope"] == pytest.approx(ols.params[1])
    assert result.p_value < 0.01 and mincer_zarnowitz(y, forecasts[1]).p_value > 0.05


def test_clark_west_and_encompassing_are_hac_t_tests(forecasts):
    y, good, biased, _ = forecasts
    small = pd.Series(np.zeros(len(y)))  # the nested model forecasts zero
    for h in (1, 3):
        result = clark_west(y, small, good, horizon=h)
        adj = (y - small) ** 2 - ((y - good) ** 2 - (small - good) ** 2)
        ols = sm.OLS(adj.to_numpy(), np.ones(len(adj))).fit(cov_type="HAC", cov_kwds={"maxlags": h - 1, "use_correction": False})
        assert result.statistic == pytest.approx(float(ols.tvalues[0]), rel=1e-9)
    assert result.p_value < 0.01
    enc = encompassing(y, biased, good, horizon=1)  # the biased forecast does not encompass the good one
    d = ((y - biased) - (y - good)) * (y - biased)
    n = len(d)
    t = d.mean() / np.sqrt(np.var(d) / n) * np.sqrt((n + 1 - 2 + 0) / n)  # DM with the HLN factor at h = 1
    assert enc.statistic == pytest.approx(t, rel=1e-9) and enc.p_value == pytest.approx(sps.t.sf(t, n - 1)) and enc.p_value < 0.01
    assert encompassing(y, good, biased).p_value > 0.05


def test_giacomini_white_is_n_r_squared(forecasts):
    y, good, biased, _ = forecasts
    l1, l2 = (y - good) ** 2, (y - biased) ** 2
    result = giacomini_white(l1, l2, horizon=1)
    d = (l1 - l2).to_numpy()
    Z = np.column_stack([np.ones(len(d) - 1), d[:-1]])
    target = Z * d[1:, None]
    r2 = sm.OLS(np.ones(len(target)), target).fit().rsquared  # uncentred: no constant
    assert result.statistic == pytest.approx(len(target) * r2, rel=1e-9) and result.df == 2 and result.details["better forecast"] == "1"
    unconditional = giacomini_white(l1, l2, conditional=False)
    assert unconditional.df == 1 and unconditional.p_value < 0.05


def test_spa_and_the_model_confidence_set_find_the_best_forecast(forecasts):
    pytest.importorskip("arch")
    from pymetrix.stats import model_confidence_set, spa

    y, good, biased, x = forecasts
    rng = np.random.default_rng(3)
    noisy = good + rng.normal(scale=0.8, size=len(y))
    losses = {"good": (y - good) ** 2, "biased": (y - biased) ** 2, "noisy": (y - noisy) ** 2}
    beaten = spa(losses["biased"], {"good": losses["good"], "noisy": losses["noisy"]}, reps=500, seed=1)
    assert beaten.p_value < 0.05 and beaten.details["lowest average loss"] == "good"
    assert beaten.details["p-value, lower"] <= beaten.p_value <= beaten.details["p-value, upper (Reality Check)"]
    # the best forecast as the benchmark is not beaten: its p-value is far from 10% (seeds 1 to 10 give 0.48 to 0.55
    # with arch 8 and 0.37 to 0.43 with arch 9, whose SPA p-values moved, so a bound of 0.5 is not the test's point)
    assert spa(losses["good"], {"biased": losses["biased"]}, reps=500, seed=1).p_value > 0.2
    table = model_confidence_set(losses, alpha=0.1, reps=500, seed=1)
    inside = dict(zip(table.index, [r[2] for r in table.rows]))
    assert inside["good"] == "yes" and inside["biased"] == "no" and table.index[0] == "good"
    # through the arch adapter: SPA's provenance names arch and its arguments; the tables say what computed them
    assert beaten.provenance.backend["wraps"] == "arch.bootstrap.SPA" and beaten.provenance.seed == 1
    assert beaten.provenance.options["stages"]["model"]["reps"] == 500 and beaten.provenance.nobs == len(y)
    assert table.notes[-1].startswith("Computed by arch ") and table.notes[-1].endswith("(arch.bootstrap.MCS).")


# ---------------------------------------------------------------------------------------------
# backtests


@pytest.fixture(scope="module")
def macro():
    return Dataset.example("sample_macro")


@pytest.fixture(scope="module")
def backtests(macro):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq = backtest(Equation("c ~ y").fit(macro), macro, ("2016Q1", "2022Q4"), horizon=2, label="ols")
        var = backtest(VAR(["c", "y"], lags=2).fit(macro), macro, ("2016Q1", "2022Q4"), horizon=2, label="var")
        bvar = backtest(VAR(["c", "y"], lags=2).fit(macro, method=Bayes(prior=Minnesota(), draws=300)), macro, ("2016Q1", "2022Q4"),
                        horizon=2, label="bvar")
    return eq, var, bvar


def test_backtest_densities_are_scored_from_the_forecasts(macro, backtests):
    eq, var, bvar = backtests
    row = var.errors[(var.errors["variable"] == "c") & (var.errors["horizon"] == 2)].iloc[3]
    origin = row["origin"]
    fc = VAR(["c", "y"], lags=2).fit(macro, span=(None, origin)).forecast(macro, (str(row["target"] - 1), str(row["target"])))
    sd = fc.frame["c_se"].iloc[-1]
    assert row["sd"] == pytest.approx(sd) and row["density"] == "normal"
    assert row["crps"] == pytest.approx(float(crps_normal(row["actual"], row["forecast"], sd)), abs=1e-12)
    first = eq.errors.iloc[0]
    static = Equation("c ~ y").fit(macro, span=(None, first["origin"])).forecast(macro, (str(first["target"]), str(first["target"])))
    assert first["density"] == "normal" and first["sd"] == pytest.approx(static.frame["se"].iloc[0])  # no lag of c: the static standard error
    b = bvar.errors.iloc[0]
    fit = VAR(["c", "y"], lags=2).fit(macro, span=(None, b["origin"]), method=Bayes(prior=Minnesota(), draws=300))
    periods, paths = fit.predictive_draws(macro, (str(b["target"]), str(b["target"] + 1)))
    draws = paths[:, 0, list(fit.variables).index(b["variable"])]
    assert b["density"] == "draws" and b["crps"] == pytest.approx(crps_sample(b["actual"], draws), abs=1e-10)
    assert b["forecast"] == pytest.approx(draws.mean()) and b["pit"] == pytest.approx(pit_sample(b["actual"], draws))
    scores = var.density_scores()
    sub = var.errors[(var.errors["variable"] == "y") & (var.errors["horizon"] == 1)]
    assert scores.loc[("y", 1), "CRPS"] == pytest.approx(sub["crps"].mean()) and scores.loc[("y", 1), "n"] == len(sub)


def test_dynamic_equations_without_standard_errors_have_no_density(macro):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dyn = backtest(Equation("c ~ c[-1] + y").fit(macro), macro, ("2020Q1", "2021Q4"), horizon=2, label="dyn")
    assert dyn.errors["density"].eq("").all() and dyn.density_scores().empty
    with pytest.raises(SpecificationError, match="no densities"):
        dyn.losses("c", 1, "crps")
    assert "Density scores" not in [t for t, _ in __import__("pymetrix.views", fromlist=["x"]).view_sections(dyn)]


def test_backtest_tests_and_comparisons(backtests):
    eq, var, bvar = backtests
    e = var.errors[(var.errors["variable"] == "c") & (var.errors["horizon"] == 1)]
    assert var.efficiency("c", 1).statistic == pytest.approx(mincer_zarnowitz(e["actual"], e["forecast"]).statistic)
    assert var.pit_test("c", 1).statistic == pytest.approx(berkowitz(e["pit"]).statistic)
    hits = ((e["pit"] < 0.05) | (e["pit"] > 0.95)).astype(int)
    assert var.interval_test("c", 1).statistic == pytest.approx(christoffersen(hits).statistic)
    dm = eq.compare(var, "c", 1, loss="crps")
    a, b = eq.losses("c", 1, "crps"), var.losses("c", 1, "crps")
    from pymetrix.stats.evaluation import mean_differential_test

    assert dm.statistic == pytest.approx(mean_differential_test((a - b).to_numpy(), 1)[0]) and dm.details["loss"] == "crps"
    assert eq.compare(var, "c", 1, loss="log score").details["mean loss differential"] == pytest.approx(
        float((-eq.errors.query("variable == 'c' and horizon == 1")["log score"].to_numpy()
               + var.errors.query("variable == 'c' and horizon == 1")["log score"].to_numpy()).mean()))
    assert eq.compare(var, "c", 1, test="giacomini-white").df == 2
    assert eq.compare(var, "c", 1, test="clark-west").name == "Clark-West"
    assert eq.encompasses(var, "c", 1).name == "Encompassing (HLN)"
    table = var.evaluation()
    assert table.columns[1] == "efficiency (MZ)" and len(table.rows) == 4 and "PIT (Berkowitz)" in table.columns
    assert var.plot_pit(horizon=1).panels[0].title.startswith("c, h=1")


def test_the_old_diebold_mariano_numbers_are_unchanged():
    from pymetrix import diebold_mariano

    rng = np.random.default_rng(11)
    e1, e2 = rng.normal(size=40), rng.normal(scale=1.3, size=40)
    d = e1**2 - e2**2
    n, h = 40, 3
    u = d - d.mean()
    lrv = u @ u / n + 2 * sum(u[k:] @ u[:-k] / n for k in range(1, h))
    stat = d.mean() / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    result = diebold_mariano(e1, e2, horizon=h)
    assert result.statistic == pytest.approx(stat, rel=1e-12) and result.p_value == pytest.approx(2 * sps.t.sf(abs(stat), n - 1))


def test_rolling_windows_estimate_on_the_last_periods(macro):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        roll = backtest(Equation("c ~ y").fit(macro), macro, ("2018Q1", "2018Q4"), horizon=1, label="rolling", window=30)
    row = roll.errors.iloc[2]
    origin = pd.Period(row["origin"], freq="Q")
    own = Equation("c ~ y").fit(macro, span=(str(origin - 29), str(origin)))
    assert own.provenance.nobs == 30
    assert row["forecast"] == pytest.approx(float(own.forecast(macro, (str(origin + 1), str(origin + 1))).frame["forecast"].iloc[0]))
    assert roll.provenance.spec["window"] == "rolling, 30 periods"
    with pytest.raises(SpecificationError, match="at least two"):
        backtest(Equation("c ~ y").fit(macro), macro, ("2018Q1", "2018Q4"), horizon=1, window=1)


# ---------------------------------------------------------------------------------------------
# combination


def test_combination_weights_by_hand(backtests):
    eq, var, bvar = backtests
    pair = {"ols": eq, "var": var}
    for method in ("equal", "inverse_mse", "regression"):
        cb = combine_forecasts(pair, method=method, min_history=4)
        w = cb.weights.pivot_table(index=["origin", "variable", "horizon"], columns="member", values="weight")
        a = eq.errors.set_index(["origin", "variable", "horizon"])
        b = var.errors.set_index(["origin", "variable", "horizon"])
        key = ("2021Q2", "c", 1)
        target = a.loc[key, "target"]
        past_a = eq.errors[(eq.errors["variable"] == "c") & (eq.errors["horizon"] == 1) & (eq.errors["target"] <= pd.Period("2021Q2", "Q"))]
        past_b = var.errors[(var.errors["variable"] == "c") & (var.errors["horizon"] == 1)].set_index("target").loc[past_a["target"]]
        fa, fb, y = past_a["forecast"].to_numpy(), past_b["forecast"].to_numpy(), past_a["actual"].to_numpy()
        if method == "equal":
            expected = np.array([0.5, 0.5])
        elif method == "inverse_mse":
            inv = 1 / np.array([np.mean((y - fa) ** 2), np.mean((y - fb) ** 2)])
            expected = inv / inv.sum()
        else:  # weights summing to one by substitution: y - fb = wa (fa - fb)
            wa = float(np.linalg.lstsq((fa - fb)[:, None], y - fb, rcond=None)[0][0])
            expected = np.array([wa, 1 - wa])
        assert w.loc[key, ["ols", "var"]].to_numpy() == pytest.approx(expected, abs=1e-9), method
        row = cb.errors[(cb.errors["origin"] == "2021Q2") & (cb.errors["variable"] == "c") & (cb.errors["horizon"] == 1)].iloc[0]
        assert row["forecast"] == pytest.approx(expected @ np.array([a.loc[key, "forecast"], b.loc[key, "forecast"]]), abs=1e-9)
        assert row["target"] == target


def test_combinations_pool_densities_and_start_with_equal_weights(backtests):
    eq, var, bvar = backtests
    cb = combine_forecasts([eq, var, bvar], method="inverse_mse", min_history=4)
    first = cb.weights[cb.weights["origin"] == "2016Q1"]
    assert np.allclose(first["weight"], 1 / 3) and set(first["history"]) == {0}
    row = cb.errors.iloc[-1]
    members = [bt.errors.set_index(["origin", "variable", "horizon"]).loc[(row["origin"], row["variable"], row["horizon"])] for bt in (eq, var, bvar)]
    w = cb.weights[(cb.weights["origin"] == row["origin"]) & (cb.weights["variable"] == row["variable"])
                   & (cb.weights["horizon"] == row["horizon"])]["weight"].to_numpy()
    m, s = np.array([x["forecast"] for x in members]), np.array([x["sd"] for x in members])
    assert row["density"] == "pool" and row["crps"] == pytest.approx(crps_mixture(row["actual"], w, m, s), abs=1e-12)
    assert row["sd"] == pytest.approx(np.sqrt(np.sum(w * (s**2 + m**2)) - (w @ m) ** 2), abs=1e-10)
    regression = combine_forecasts([eq, var], method="regression")
    negative = regression.weights.groupby(["origin", "variable", "horizon"])["weight"].min() < 0
    if negative.any():  # no mixture density with negative weights
        o, v, h = negative[negative].index[0]
        assert regression.errors.query("origin == @o and variable == @v and horizon == @h")["density"].iloc[0] == ""
    assert [t for t, _ in __import__("pymetrix.views", fromlist=["x"]).view_sections(cb)][-2:] == ["Weights", "Forecasts"]
    with pytest.raises(SpecificationError, match="at least two"):
        combine_forecasts([eq])


def test_the_forecast_dialogs_are_steps_whose_code_replays(macro, backtests):
    from pymetrix.commands import CombineForecasts, CompareBacktests, CompareManyForecasts, Session

    eq, var, bvar = backtests
    session = Session()
    session.objects.update({"data": macro, "ols": eq, "var": var, "bvar": bvar})
    steps = [CombineForecasts("ols, var, bvar", "inverse_mse", name="combo"),
             CompareBacktests("ols", "var", "c", 1, "giacomini-white", name="gw"),
             CompareBacktests("ols", "var", "c", 2, "diebold-mariano", "crps", name="dm"),
             CompareBacktests("ols", "var", "c", 1, "encompassing", name="enc")]
    if __import__("importlib.util").util.find_spec("arch"):
        steps.append(CompareManyForecasts("ols, var, bvar", "c", 1, "crps", name="mcs"))
    for step in steps:
        session.run(step)
    namespace = dict(session.namespace) | {"ols": eq, "var": var, "bvar": bvar}
    exec("\n".join(r.code for r in session.history), namespace)  # the script, as History shows it
    assert namespace["combo"].errors["forecast"].equals(session["combo"].errors["forecast"])
    for name in ("gw", "dm", "enc"):
        assert namespace[name].statistic == pytest.approx(session[name].statistic), name
    if "mcs" in session.objects:
        assert namespace["mcs"].rows == session["mcs"].rows
    assert CombineForecasts("ols", "equal").validate(session) == ["name at least 2 backtests, separated by commas"]
    assert "no object named 'nope'" in CompareBacktests("ols", "nope", "c").validate(session)


# ---------------------------------------------------------------------------------------------
# asymmetric losses, binary forecasts, StepM


def test_asymmetric_losses_have_their_optimal_forecasts():
    from scipy.optimize import minimize_scalar

    from pymetrix.stats import linex_loss, linlin_loss

    e = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
    assert linlin_loss(e, 0.7) == pytest.approx([0.6, 0.15, 0.0, 0.7, 2.1])
    assert linlin_loss(e, 0.5) == pytest.approx(0.5 * np.abs(e))
    assert linex_loss(e, 1.0) == pytest.approx(np.exp(e) - e - 1)
    assert linex_loss(np.array([0.01]), 0.01)[0] == pytest.approx(0.01**2 * 0.01**2 / 2, rel=1e-3)  # a^2 e^2 / 2 near zero
    y = np.random.default_rng(5).normal(1.0, 2.0, 200_000)
    best = minimize_scalar(lambda c: float(np.mean(linlin_loss(y - c, 0.8))), bounds=(-5, 8), method="bounded", options={"xatol": 1e-6}).x
    assert best == pytest.approx(np.quantile(y, 0.8), abs=0.01)  # the tau quantile
    best = minimize_scalar(lambda c: float(np.mean(linex_loss(y - c, 0.5))), bounds=(-5, 8), method="bounded", options={"xatol": 1e-6}).x
    assert best == pytest.approx(1.0 + 0.5 * 4.0 / 2, abs=0.03)  # mean + a sd^2 / 2
    with pytest.raises(SpecificationError):
        linlin_loss(e, 1.2)
    with pytest.raises(SpecificationError):
        linex_loss(e, 0.0)


def test_backtests_compare_under_asymmetric_losses(backtests):
    from pymetrix.stats import linex_loss, linlin_loss

    eq, var, _ = backtests
    e = eq.errors[(eq.errors["variable"] == "c") & (eq.errors["horizon"] == 1)].set_index("target")["error"]
    assert eq.losses("c", 1, "lin-lin", tau=0.9).to_numpy() == pytest.approx(linlin_loss(e, 0.9))
    assert eq.losses("c", 1, "linex", a=-0.5).to_numpy() == pytest.approx(linex_loss(e, -0.5))
    result = eq.compare(var, "c", 1, loss="linex", a=0.5)
    assert result.details["loss"] == "linex" and np.isfinite(result.statistic)


def test_binary_scores_by_hand():
    from pymetrix.stats import binary_log_score, binary_scores, brier_score, kuipers_score, roc_auc, roc_curve

    p = np.array([0.9, 0.8, 0.8, 0.3, 0.6, 0.2, 0.1, 0.4])
    y = np.array([1, 1, 0, 1, 0, 0, 0, 1])
    assert brier_score(p, y) == pytest.approx(np.mean((p - y) ** 2))
    assert binary_log_score(p, y) == pytest.approx(np.mean(np.log(np.where(y == 1, p, 1 - p))))
    pairs = [(a, b) for a in p[y == 1] for b in p[y == 0]]
    brute = np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs])
    assert roc_auc(p, y) == pytest.approx(brute)
    curve = roc_curve(p, y)
    trapezoid = float(np.sum(np.diff(curve["false positive rate"]) * (curve["true positive rate"][1:].to_numpy() + curve["true positive rate"][:-1].to_numpy()) / 2))
    assert trapezoid == pytest.approx(brute) and curve["true positive rate"].iloc[-1] == 1.0  # the area under the curve is the AUC
    events = (p >= 0.5).astype(int)
    assert kuipers_score(events, y) == pytest.approx(np.mean(events[y == 1]) - np.mean(events[y == 0]))
    table = dict((r[0], r[1]) for r in binary_scores(p, y).rows)
    assert table["ROC area"] == pytest.approx(brute) and table["Brier score of the base rate"] == pytest.approx(0.25)
    with pytest.raises(SpecificationError, match="between 0 and 1"):
        brier_score([1.2], [1])


def test_pesaran_timmermann_is_the_independence_chi_squared_and_holds_its_size():
    from pymetrix.stats import pesaran_timmermann

    rng = np.random.default_rng(9)
    x = rng.normal(size=3000)
    y = 0.15 * x + rng.normal(size=3000)  # some directional skill
    result = pesaran_timmermann(y, x)
    table = np.array([[np.sum((y > 0) & (x > 0)), np.sum((y > 0) & (x <= 0))], [np.sum((y <= 0) & (x > 0)), np.sum((y <= 0) & (x <= 0))]])
    chi2 = sps.chi2_contingency(table, correction=False)[0]
    assert result.statistic**2 == pytest.approx(chi2, rel=0.01) and result.p_value < 0.01  # asymptotically the same test
    sizes = [pesaran_timmermann(rng.normal(size=200), rng.normal(size=200)).p_value < 0.05 for _ in range(400)]
    assert 0.02 < np.mean(sizes) < 0.09


def test_stepm_finds_the_superior_models(forecasts):
    pytest.importorskip("arch")
    from pymetrix.stats import stepm

    y, good, biased, _ = forecasts
    noisy = good + np.random.default_rng(4).normal(scale=0.3, size=len(y))
    table = stepm((y - biased) ** 2, {"good": (y - good) ** 2, "noisy": (y - noisy) ** 2, "same": (y - biased) ** 2}, reps=500, seed=2)
    superior = dict(zip(table.index, [r[1] for r in table.rows]))
    assert superior["good"] == "yes" and superior["same"] == "no" and table.index[0] == "good"
