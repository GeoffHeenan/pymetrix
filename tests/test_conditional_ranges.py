"""Richer conditional forecasts: ranges (truncated-normal Gibbs sampling), structural scenarios (only some shocks move,
conditions on shocks), the Bayesian VAR's posterior updated by the conditions (Waggoner and Zha 1999), ranges in
first-order model plans, and forecast rounds that record their conditions.

Checked independently: exact conditions against the Gaussian conditioning formula on a covariance built here from
the VAR's own recursion; a one-sided range in a one-shock VAR against the truncated-normal mean; ranges against
rejection sampling of unconditional paths; wide ranges against the unconditional forecast and narrow ones against
the exact condition; a structural scenario against the impact matrix; a binding range in a model plan against the
minimum-norm solution of its active constraint."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pymetrix import VAR, Bayes, Bounds, Dataset, ForecastRound, ForecastStore, Minnesota, Scenario, templates
from pymetrix import var_conditions as vc
from pymetrix.errors import SpecificationError
from pymetrix.models import ModelError
from pymetrix.var import _point_path

SPAN = ("2025Q1", "2025Q4")


@pytest.fixture(scope="module")
def macro():
    return Dataset.example("sample_macro")


@pytest.fixture(scope="module")
def var(macro):
    return VAR(["c", "y", "i"], lags=2).fit(macro)


@pytest.fixture(scope="module")
def bvar(macro):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VAR(["c", "y", "i"], lags=2).fit(macro, method=Bayes(prior=Minnesota(), draws=400))


def _stacked_moments(var, H):
    """Mean and covariance of the stacked path (period-major), from the VAR recursion written out here."""
    A, S, K = var.coefs, var.estimate.sigma, len(var.variables)
    psi = [np.eye(K)]
    for h in range(1, H):
        psi.append(sum(A[l] @ psi[h - 1 - l] for l in range(len(A)) if h - 1 - l >= 0))
    cov = np.zeros((H * K, H * K))
    for i in range(H):
        for j in range(H):
            cov[i * K:(i + 1) * K, j * K:(j + 1) * K] = sum(psi[i - s] @ S @ psi[j - s].T for s in range(min(i, j) + 1))
    return cov


def _draws(fit, data, conditions, shocks=None, n=4000, seed=0):
    """The simulated paths themselves (n, H, K), as the forecast draws them."""
    periods, D, history = fit._forecast_inputs(data, SPAN)
    d = len(fit.arrays.d_names)
    point = _point_path(fit.coefs, fit.estimate.B[:d], D, history, len(periods))
    conds = vc.parse(conditions, fit.variables, periods)
    rng = np.random.default_rng(seed)
    impact = fit.impact() if shocks else np.linalg.cholesky(fit.estimate.sigma)
    prepared = vc.prepare(fit.coefs, impact, point, conds, vc.driving_mask(shocks, fit.shocks, len(periods)), n, rng)
    return vc.draw_paths(prepared, rng)[0].reshape(n, len(periods), len(fit.variables)), point


# ---------------------------------------------------------------------------------------------
# exact conditions


def test_exact_conditions_match_gaussian_conditioning(macro, var):
    u = var.forecast(macro, SPAN).frame
    mean = u[["c", "y", "i"]].to_numpy().reshape(-1)
    cov = _stacked_moments(var, 4)
    idx = [1 * 3 + 1, 2 * 3 + 1, 0 * 3 + 2]  # y in 2025Q2 and Q3, i in 2025Q1
    values = np.array([600.0, 603.0, 64.5])
    gain = cov[:, idx] @ np.linalg.inv(cov[np.ix_(idx, idx)])
    expected_mean = (mean + gain @ (values - mean[idx])).reshape(4, 3)
    expected_sd = np.sqrt(np.clip(np.diag(cov - gain @ cov[idx]), 0, None)).reshape(4, 3)
    c = var.conditional_forecast(macro, SPAN, {"y": {"2025Q2": 600.0, "2025Q3": 603.0}, "i": {"2025Q1": 64.5}}).frame
    assert c[["c", "y", "i"]].to_numpy() == pytest.approx(expected_mean, rel=1e-10)
    assert c[["c_se", "y_se", "i_se"]].to_numpy() == pytest.approx(expected_sd, rel=1e-7, abs=1e-6)
    # every shock named: the structural route gives the same answer (identification does not matter)
    s = var.conditional_forecast(macro, SPAN, {"y": {"2025Q2": 600.0, "2025Q3": 603.0}, "i": {"2025Q1": 64.5}}, shocks=["c", "y", "i"]).frame
    assert np.allclose(s.to_numpy(), c.to_numpy(), rtol=1e-9, atol=1e-8)
    assert var.conditional_forecast(macro, SPAN, {"y": 600.0}).frame.shape == u.shape  # columns as before


def test_the_conditions_are_recorded_in_the_forecast(macro, var):
    f = var.conditional_forecast(macro, SPAN, {"y": {"2025Q1": 600.0}, "i": Bounds(upper=70)}, draws=200)
    rows = f.meta.extra["conditions"]
    assert rows[0] == {"variable": "y", "period": "2025Q1", "value": 600.0}
    assert rows[1] == {"variable": "i", "period": "2025Q1", "lower": None, "upper": 70.0} and len(rows) == 5
    assert f.meta.extra["shocks"] is None


# ---------------------------------------------------------------------------------------------
# ranges


def test_one_bound_in_a_one_shock_var_is_the_truncated_normal_mean(macro):
    ar = VAR(["y"], lags=1).fit(macro)
    u = ar.forecast(macro, SPAN).frame
    mu, sd = u["y"].iloc[0], u["y_se"].iloc[0]
    lo = mu + 0.5 * sd
    a = (lo - mu) / sd
    expected = mu + sd * stats.norm.pdf(a) / stats.norm.sf(a)
    variance = sd**2 * (1 + a * stats.norm.pdf(a) / stats.norm.sf(a) - (stats.norm.pdf(a) / stats.norm.sf(a)) ** 2)
    c = ar.conditional_forecast(macro, SPAN, {"y": {"2025Q1": Bounds(lower=lo)}}, draws=40_000, seed=3).frame
    mc = np.sqrt(variance / 40_000)
    assert abs(c["y"].iloc[0] - expected) < 4 * mc
    assert c["y_se"].iloc[0] == pytest.approx(np.sqrt(variance), rel=0.02)
    assert c["y_lower"].iloc[0] >= lo
    # one period on, the mean moves through the AR coefficient: E y2 = c + a1 E y1
    a1 = ar.coefs[0][0, 0]
    assert c["y"].iloc[1] == pytest.approx(u["y"].iloc[1] + a1 * (expected - mu), abs=5 * mc + 4 * u["y_se"].iloc[1] / 200)
    # an upper bound mirrors it
    up = ar.conditional_forecast(macro, SPAN, {"y": {"2025Q1": (None, 2 * mu - lo)}}, draws=40_000, seed=4).frame
    assert abs(up["y"].iloc[0] - (2 * mu - expected)) < 4 * mc


def test_every_draw_meets_the_bounds(macro, var):
    conditions = {"y": {"2025Q1": Bounds(598, 600), "2025Q2": Bounds(lower=601), "2025Q3": Bounds(upper=596)}, "i": {"2025Q4": 66.0}}
    paths, _ = _draws(var, macro, conditions, n=3000)
    y = paths[:, :, 1]
    assert (y[:, 0] >= 598).all() and (y[:, 0] <= 600).all() and (y[:, 1] >= 601).all() and (y[:, 2] <= 596).all()
    assert np.allclose(paths[:, 3, 2], 66.0)
    f = var.conditional_forecast(macro, SPAN, conditions, draws=3000).frame
    assert f["y_lower"].iloc[0] >= 598 and f["y_upper"].iloc[0] <= 600 and f["y_lower"].iloc[1] >= 601
    assert set(f.columns) >= {"y", "y_se", "y_median", "y_lower", "y_upper"}


def test_ranges_match_rejection_sampling(macro, var):
    """Two correlated bounds (y in two consecutive periods): the Gibbs draws against unconditional paths kept only
    when they fall in the box."""
    u = var.forecast(macro, SPAN).frame
    lo = u["y"].iloc[:2].to_numpy() - 0.3 * u["y_se"].iloc[:2].to_numpy()
    hi = u["y"].iloc[:2].to_numpy() + 0.6 * u["y_se"].iloc[:2].to_numpy()
    conditions = {"y": {"2025Q1": Bounds(lo[0], hi[0]), "2025Q2": Bounds(lo[1], hi[1])}}
    paths, point = _draws(var, macro, conditions, n=20_000, seed=5)
    rng = np.random.default_rng(6)
    M = vc.stacked(var.coefs, np.linalg.cholesky(var.estimate.sigma), 4)
    free = point.reshape(-1) + rng.standard_normal((400_000, 12)) @ M.T
    keep = free[(free[:, 1] >= lo[0]) & (free[:, 1] <= hi[0]) & (free[:, 4] >= lo[1]) & (free[:, 4] <= hi[1])]
    assert len(keep) > 20_000
    sd = keep.std(axis=0)
    got = paths.reshape(len(paths), -1)
    assert np.all(np.abs(got.mean(axis=0) - keep.mean(axis=0)) < 4 * sd * np.sqrt(1 / 20_000 + 1 / len(keep)) + 1e-9)
    assert got.std(axis=0)[sd > 0] == pytest.approx(sd[sd > 0], rel=0.04)


def test_a_wide_range_is_the_unconditional_forecast_and_a_narrow_one_the_exact(macro, var):
    u = var.forecast(macro, SPAN).frame
    wide = var.conditional_forecast(macro, SPAN, {"y": Bounds(-1e6, 1e6)}, draws=20_000, seed=2).frame
    for v in ("c", "y", "i"):
        assert np.all(np.abs(wide[v] - u[v]) < 4 * u[f"{v}_se"] / np.sqrt(20_000))
        assert wide[f"{v}_se"].to_numpy() == pytest.approx(u[f"{v}_se"].to_numpy(), rel=0.03)
    exact = var.conditional_forecast(macro, SPAN, {"y": 600.0}).frame
    narrow = var.conditional_forecast(macro, SPAN, {"y": Bounds(600 - 1e-7, 600 + 1e-7)}, draws=20_000, seed=3).frame
    for v in ("c", "i"):
        assert np.all(np.abs(narrow[v] - exact[v]) < 4 * exact[f"{v}_se"] / np.sqrt(20_000) + 1e-6)
        assert narrow[f"{v}_se"].to_numpy() == pytest.approx(exact[f"{v}_se"].to_numpy(), rel=0.03)
    assert narrow["y"].to_numpy() == pytest.approx(600.0, abs=1e-6)
    point = var.conditional_forecast(macro, SPAN, {"y": Bounds(600, 600)}, draws=500).frame  # a range that is a point
    assert point["y"].to_numpy() == pytest.approx(600.0) and np.allclose(point["y_se"], 0)


def test_truncated_standard_draws_far_in_the_tails():
    rng = np.random.default_rng(0)
    a, b = np.full(50_000, 8.0), np.full(50_000, np.inf)
    x = vc.truncated_standard(a, b, rng.random(50_000))
    assert (x >= 8).all() and x.mean() == pytest.approx(stats.truncnorm(8, np.inf).mean(), rel=1e-3)
    x = vc.truncated_standard(np.full(10, -np.inf), np.full(10, -40.0), rng.random(10))
    assert np.all(np.isfinite(x)) and (x <= -40).all()


def test_bad_ranges(macro, var):
    with pytest.raises(SpecificationError, match="lower bound"):
        Bounds(4, 3)
    with pytest.raises(SpecificationError, match="needs a lower bound"):
        Bounds()
    with pytest.raises(SpecificationError, match="two conditions"):
        var.conditional_forecast(macro, SPAN, {"y": {"2025Q1": 600.0, pd.Period("2025Q1", "Q"): Bounds(1, 2)}})
    with pytest.raises(SpecificationError, match="cannot"):  # a range on a value an exact condition fixes
        var.conditional_forecast(macro, SPAN, {"c": 450.0, "y": 590.0, "i": {"2025Q1": 64.0, "2025Q2": Bounds(60, 70)}}, shocks=["c", "y"])


# ---------------------------------------------------------------------------------------------
# structural scenarios


def test_only_the_named_shocks_move(macro, var):
    u = var.forecast(macro, SPAN).frame
    P = var.impact()  # Cholesky, c first: the y shock does not move c on impact
    r = float(u["y"].iloc[0]) + 4.0
    s = var.conditional_forecast(macro, SPAN, {"y": {"2025Q1": r}}, shocks=["y"]).frame
    assert s["c"].iloc[0] == pytest.approx(u["c"].iloc[0], rel=1e-12)
    assert s["c_se"].iloc[0] == pytest.approx(u["c_se"].iloc[0], rel=1e-9)  # the c shock keeps its distribution
    # i on impact: the y shock that closes the gap (net of the c shock's effect on y, zero on average)
    assert s["i"].iloc[0] == pytest.approx(u["i"].iloc[0] + P[2, 1] / P[1, 1] * 4.0, rel=1e-10)
    # differs from the all-shocks answer, which moves c too
    w = var.conditional_forecast(macro, SPAN, {"y": {"2025Q1": r}}).frame
    assert abs(w["c"].iloc[0] - u["c"].iloc[0]) > 0.1
    # the same with a narrow range, simulated
    sr = var.conditional_forecast(macro, SPAN, {"y": {"2025Q1": Bounds(r - 1e-7, r + 1e-7)}}, shocks=["y"], draws=20_000, seed=1).frame
    for v in ("c", "i"):
        assert abs(sr[v].iloc[0] - s[v].iloc[0]) < 4 * s[f"{v}_se"].iloc[0] / np.sqrt(20_000) + 1e-6
    with pytest.raises(SpecificationError, match="not a shock"):
        var.conditional_forecast(macro, SPAN, {"y": 600.0}, shocks=["g"])
    with pytest.raises(SpecificationError, match="cannot all be met"):  # one shock cannot fix two variables
        var.conditional_forecast(macro, SPAN, {"y": {"2025Q1": 600.0}, "i": {"2025Q1": 64.0}}, shocks=["y"])


def test_conditions_on_the_shocks(macro, var):
    u = var.forecast(macro, SPAN).frame
    quiet = var.conditional_forecast(macro, SPAN, {}, shock_conditions={"c": 0.0, "y": 0.0, "i": 0.0}).frame
    assert quiet[["c", "y", "i"]].to_numpy() == pytest.approx(u[["c", "y", "i"]].to_numpy()) and np.allclose(quiet[["c_se", "y_se", "i_se"]], 0, atol=1e-8)
    s = var.conditional_forecast(macro, SPAN, {"y": 600.0}, shock_conditions={"c": 0.0}).frame
    assert s["c_se"].iloc[0] == pytest.approx(0, abs=1e-6)  # with c first, only its own shock moves it on impact
    positive = var.conditional_forecast(macro, SPAN, {}, shock_conditions={"c": {"2025Q1": Bounds(lower=0)}}, draws=20_000).frame
    P = var.impact()
    assert positive["c"].iloc[0] == pytest.approx(u["c"].iloc[0] + P[0, 0] * stats.norm.pdf(0) / 0.5, rel=1e-3)
    with pytest.raises(SpecificationError, match="may move"):
        var.conditional_forecast(macro, SPAN, {"y": 600.0}, shocks=["y"], shock_conditions={"c": 0.0})


# ---------------------------------------------------------------------------------------------
# Bayesian VARs


def test_bayesian_ranges_draw_by_draw(macro, bvar):
    conditions = {"y": {"2025Q1": Bounds(598, 602), "2025Q2": Bounds(lower=600)}}
    a = bvar.conditional_forecast(macro, SPAN, conditions, seed=1).frame
    b = bvar.conditional_forecast(macro, SPAN, conditions, seed=1).frame
    assert a.equals(b)
    assert a["y_lower"].iloc[0] >= 598 and a["y_upper"].iloc[0] <= 602 and a["y_lower"].iloc[1] >= 600
    s = bvar.conditional_forecast(macro, SPAN, {"y": 600.0}, shocks=["y"], draws=100).frame
    assert np.allclose(s["y"], 600.0) and np.allclose(s["y_se"], 0, atol=1e-8)


def test_bayesian_wide_range_matches_the_predictive(macro, bvar):
    f = bvar.forecast(macro, SPAN, seed=4).frame
    w = bvar.conditional_forecast(macro, SPAN, {"y": Bounds(-1e6, 1e6)}, seed=5).frame
    for v in ("c", "y", "i"):
        assert np.all(np.abs(w[v] - f[v]) < 5 * f[f"{v}_se"] / np.sqrt(400))


def test_the_posterior_updated_by_the_conditions(macro, bvar):
    conditions = {"y": {"2025Q1": 595.0, "2025Q2": 600.0}}
    up = bvar.conditional_forecast(macro, SPAN, conditions, draws=300, seed=2, update_posterior=True)
    frame = up.frame
    assert frame["y"].iloc[:2].to_numpy() == pytest.approx([595.0, 600.0]) and np.allclose(frame["y_se"].iloc[:2], 0, atol=1e-8)
    assert up.meta.extra["posterior"].startswith("updated") and len(frame) == 4
    # no binding condition: the sampler's stationary distribution is the posterior, so the predictive is unchanged
    f = bvar.forecast(macro, SPAN, seed=4).frame
    w = bvar.conditional_forecast(macro, SPAN, {"y": Bounds(-1e6, 1e6)}, draws=600, seed=6, update_posterior=True).frame
    for v in ("c", "y", "i"):
        assert np.all(np.abs(w[v] - f[v]) < 0.6 * f[f"{v}_se"])
        assert w[f"{v}_se"].to_numpy() == pytest.approx(f[f"{v}_se"].to_numpy(), rel=0.3)
    with pytest.raises(SpecificationError, match="every shock"):
        bvar.conditional_forecast(macro, SPAN, conditions, shocks=["y"], update_posterior=True)


def test_the_independent_prior_updates_too(macro):
    from pymetrix import NormalWishart

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nw = VAR(["c", "y"], lags=1).fit(macro, method=Bayes(prior=NormalWishart(), draws=200))
    out = nw.conditional_forecast(macro, SPAN, {"y": {"2025Q2": Bounds(lower=600)}}, draws=100, seed=1, update_posterior=True).frame
    assert out["y_lower"].iloc[1] >= 600


# ---------------------------------------------------------------------------------------------
# the dialog


def test_the_dialog_takes_ranges_and_shocks(macro, var):
    from pymetrix.commands import ConditionalForecast, Session

    session = Session()
    session.objects.update({"data": macro, "var": var})
    step = ConditionalForecast("var", "2025Q1", "2025Q4", "y[2025Q1:2025Q2]=598..602, i>=60, i<=70", draws=300, seed=1,
                               shocks="y, i", shock_conditions="i[2025Q4]=0")
    assert step.validate(session) == []
    session.run(step)
    code = session.history[-1].code
    assert "{'y': {'2025Q1': {'lower': 598.0, 'upper': 602.0}, '2025Q2': {'lower': 598.0, 'upper': 602.0}}, 'i': {'lower': 60.0, 'upper': 70.0}}" in code
    assert "shocks=['y', 'i']" in code and "shock_conditions={'i': {'2025Q4': 0.0}}" in code
    namespace = dict(session.namespace) | {"data": macro, "var": var}
    exec(code, namespace)
    assert namespace["conditional"].frame.equals(session["conditional"].frame)
    assert ConditionalForecast("var", "2025Q1", "2025Q4", "y=4..3").validate(session)[0].startswith("conditions:")
    assert ConditionalForecast("var", "2025Q1", "2025Q4", "y>=3, y=2").validate(session)[0].startswith("conditions:")
    assert ConditionalForecast("var", "2025Q1", "2025Q4", "y=600", update_posterior=True).validate(session) == ["Update posterior is for Bayesian VARs"]


def test_the_dialog_refuses_mixed_conditions_on_a_variable(macro, var):
    """A condition for every period and one for some periods on the same variable: one used to replace the other
    silently, whichever came last (pi[2025Q1]=3, pi<=4 kept only pi<=4)."""
    from pymetrix.commands import ConditionalForecast, Session
    from pymetrix.commands.builtin import _conditions

    session = Session()
    session.objects.update({"data": macro, "var": var})
    for text in ("y[2025Q1]=600, y<=610", "y<=610, y[2025Q1]=600", "y>=590, y[2025Q2:2025Q3]<=610"):
        with pytest.raises(SpecificationError, match="every period and others for some periods"):
            _conditions(text)
        assert ConditionalForecast("var", "2025Q1", "2025Q4", text).validate(session)[0].startswith("conditions:")
    # the same variable's conditions all for every period, or all by period, still combine
    assert _conditions("y>=590, y<=610") == {"y": {"lower": 590.0, "upper": 610.0}}
    assert _conditions("y[2025Q1]>=590, y[2025Q1:2025Q2]<=610") == {"y": {"2025Q1": {"lower": 590.0, "upper": 610.0},
                                                                         "2025Q2": {"upper": 610.0}}}


# ---------------------------------------------------------------------------------------------
# forecast rounds


def test_rounds_record_the_conditions(macro, var, tmp_path):
    f = var.conditional_forecast(macro, SPAN, {"y": {"2025Q1": 600.0}, "i": {"2025Q2": Bounds(60, 70)}}, shocks=["y", "i"], draws=300)
    rnd = ForecastRound.from_forecast("conditional", f, data=macro)
    assert rnd.conditions == ({"variable": "y", "period": "2025Q1", "value": 600.0, "shocks": ["y", "i"]},
                              {"variable": "i", "period": "2025Q2", "lower": 60.0, "upper": 70.0, "shocks": ["y", "i"]})
    assert rnd.conditions_table().rows[1][2] == "60 to 70" and "Conditional on 2 conditions" in " ".join(rnd.table().notes)
    store = ForecastStore(tmp_path / "rounds")
    store.save(rnd)
    back = store.load("conditional")
    assert back.conditions == rnd.conditions
    store.save(rnd)  # saving it again (replaying a project) is harmless
    # rounds saved before conditions were recorded load with none
    import json

    record = json.loads((tmp_path / "rounds" / "conditional.json").read_text())
    del record["conditions"]
    (tmp_path / "rounds" / "conditional.json").write_text(json.dumps(record))
    assert store.load("conditional").conditions == ()
    plain = ForecastRound.from_forecast("plain", var.forecast(macro, SPAN), data=macro)
    assert plain.conditions == ()
    given = ForecastRound.from_forecast("given", var.forecast(macro, SPAN), conditions=[{"variable": "y", "period": "2025Q1", "value": 1.0}])
    assert given.conditions == ({"variable": "y", "period": "2025Q1", "value": 1.0},)


def test_rounds_record_a_model_plans_holds_and_ranges():
    sol = templates.new_keynesian().solve()
    path = sol.solve(None, ("2025Q1", "2026Q4"), Scenario(hold={"i": {"2025Q1": 0.5}}, hold_upper={"pi": {"2025Q2": 0.1}}, free=["ei", "epi"]))
    rnd = ForecastRound.from_forecast("plan", path, variables=["y", "pi", "i"])
    assert rnd.conditions == ({"variable": "i", "period": "2025Q1", "value": 0.5, "shocks": ["ei", "epi"]},
                              {"variable": "pi", "period": "2025Q2", "lower": None, "upper": 0.1, "shocks": ["ei", "epi"]})


def test_the_round_view_shows_the_conditions(macro, var):
    from pymetrix.views import sections

    rnd = ForecastRound.from_forecast("c", var.conditional_forecast(macro, SPAN, {"y": 600.0}), data=macro)
    assert "Conditions" in [title for title, _ in sections(rnd)]


# ---------------------------------------------------------------------------------------------
# first-order model plans


MODEL_SPAN = ("2025Q1", "2027Q4")
QUARTERS = [str(q) for q in pd.period_range("2025Q1", "2025Q4", freq="Q")]


@pytest.fixture(scope="module")
def sol():
    return templates.new_keynesian().solve()


def test_a_range_that_does_not_bind_changes_nothing(sol):
    base = sol.solve(None, MODEL_SPAN, Scenario(set={"ey": {"2025Q1": -1.0}}))
    y = float(base["y"].loc["2025Q1"])
    loose = sol.solve(None, MODEL_SPAN, Scenario(set={"ey": {"2025Q1": -1.0}}, hold_lower={"y": {"2025Q1": y - 1}}, free=["ei"]))
    np.testing.assert_allclose(loose.frame.loc["2025Q1":, ["y", "pi", "i", "ei"]], base.frame.loc["2025Q1":, ["y", "pi", "i", "ei"]], atol=1e-12)
    assert loose.provenance.method["ranges"].startswith("least-norm")


def test_a_binding_range_is_the_minimum_norm_on_its_bound(sol):
    """y kept at or above -0.2 in 2025Q1 and Q2 after a demand shock: only Q1 binds, so the shocks are the minimum-norm
    ones that put y on -0.2 in Q1, with the policy shock free in Q1 and Q2 (the last bounded period)."""
    scenario = Scenario(set={"ey": {"2025Q1": -1.0}}, hold_lower={"y": {"2025Q1": -0.2, "2025Q2": -0.2}}, free=["ei"])
    plan = sol.solve(None, MODEL_SPAN, scenario)
    base = sol.solve(None, MODEL_SPAN, Scenario(set={"ey": {"2025Q1": -1.0}}))
    responses = []
    for t in range(2):
        unit = sol.solve(None, MODEL_SPAN, Scenario(set={"ei": {QUARTERS[t]: 1.0}}))
        responses.append([unit["y"].loc["2025Q1"], unit["y"].loc["2025Q2"]])
    R = np.array(responses).T  # rows: y in Q1, Q2; columns: ei in Q1, Q2
    gap = -0.2 - float(base["y"].loc["2025Q1"])
    expected = np.linalg.pinv(R[:1]) @ [gap]
    np.testing.assert_allclose([plan["ei"].loc["2025Q1"], plan["ei"].loc["2025Q2"]], expected, atol=1e-6)
    assert plan["y"].loc["2025Q1"] == pytest.approx(-0.2, abs=1e-6) and plan["y"].loc["2025Q2"] >= -0.2 - 1e-8
    # with an exact hold as well, and an upper bound
    both = sol.solve(None, MODEL_SPAN, Scenario(set={"ey": {"2025Q1": -1.0}}, hold={"i": {"2025Q3": 0.0}},
                                                hold_upper={"pi": {"2025Q1": 0.0}}, free=["ei", "epi"]))
    assert both["i"].loc["2025Q3"] == pytest.approx(0.0, abs=1e-7) and both["pi"].loc["2025Q1"] <= 1e-7


def test_ranges_need_a_first_order_solution_and_free_shocks(sol):
    from pymetrix import PerfectForesight

    with pytest.raises(ModelError, match="free"):
        sol.solve(None, MODEL_SPAN, Scenario(hold_lower={"y": 0.0}))
    with pytest.raises(ModelError, match="cannot keep"):
        sol.solve(None, MODEL_SPAN, Scenario(hold_lower={"y": {"2025Q1": 1.0}}, hold_upper={"y": {"2025Q1": 1.0}},
                                              hold={"y": {"2025Q2": 0.0}, "pi": {"2025Q1": 0.0}}, free=["ei"]))
    nk = templates.new_keynesian()
    idx = pd.period_range("2024Q1", "2030Q4", freq="Q", name="time")
    flat = Dataset.from_frame(pd.DataFrame({v: 0.0 for v in [*nk.model.endogenous, *nk.model.exogenous]}, index=idx))
    with pytest.raises(ModelError, match="ranges"):
        nk.model.solve(flat, MODEL_SPAN, Scenario(hold_lower={"y": 0.0}, free=["ei"]), solver=PerfectForesight(terminal="steady_state"))


def test_scenarios_with_ranges_encode_and_read_back():
    from pymetrix.codec import decode, encode

    sc = Scenario(hold_lower={"y": {"2025Q1": -0.2}}, hold_upper={"pi": 3.0}, free=["ei"])
    back = decode(encode(sc))
    assert dict(back.hold_lower) == {"y": {"2025Q1": -0.2}} and dict(back.hold_upper) == {"pi": 3.0}
    assert decode(encode(Bounds(3, 4))) == Bounds(3, 4)
    old = encode(Scenario(hold={"i": 0.5}, free=["ei"]))
    del old["hold_lower"], old["hold_upper"]  # saved before ranges existed
    assert not decode(old).ranges
