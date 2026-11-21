"""Conditional forecasts from VARs (Waggoner and Zha 1999; Elliott and Timmermann 2016, section 9.5): checked against the
closed form for one condition, against a brute-force regression on simulated paths, for invariance to the ordering of
the variables, and for Bayesian VARs draw by draw."""

import warnings

import numpy as np
import pytest

from pymetrix import VAR, Bayes, Dataset, Minnesota
from pymetrix.errors import SpecificationError

SPAN = ("2025Q1", "2025Q4")


@pytest.fixture(scope="module")
def macro():
    return Dataset.example("sample_macro")


@pytest.fixture(scope="module")
def var(macro):
    return VAR(["c", "y", "i"], lags=2).fit(macro)


def test_one_condition_is_the_regression_on_the_shock_covariance(macro, var):
    u = var.forecast(macro, SPAN).frame
    r = float(u["y"].iloc[0]) + 4.0
    c = var.conditional_forecast(macro, SPAN, {"y": {"2025Q1": r}}).frame
    S = var.estimate.sigma  # c, y, i
    for j, name in ((0, "c"), (2, "i")):
        assert c[name].iloc[0] == pytest.approx(u[name].iloc[0] + S[j, 1] / S[1, 1] * 4.0, rel=1e-10)
        assert c[f"{name}_se"].iloc[0] == pytest.approx(np.sqrt(S[j, j] - S[j, 1] ** 2 / S[1, 1]), rel=1e-9)
    assert c["y"].iloc[0] == pytest.approx(r) and c["y_se"].iloc[0] == pytest.approx(0, abs=1e-9)
    assert c["y_se"].iloc[1] > 0  # later periods of y are free


def test_matches_a_regression_on_simulated_paths(macro, var):
    conditions = {"y": {"2025Q2": 600.0, "2025Q3": 603.0}, "i": {"2025Q1": 64.5}}
    c = var.conditional_forecast(macro, SPAN, conditions).frame
    # simulate the VAR forward from the same history, independently of pymetrix's moving-average algebra
    A, S = var.coefs, var.estimate.sigma
    d = len(var.arrays.d_names)
    const = var.estimate.B[0] if d else np.zeros(3)
    history = macro.frame.loc[["2024Q3", "2024Q4"], ["c", "y", "i"]].to_numpy(float)
    rng = np.random.default_rng(0)
    n = 200_000
    shocks = rng.standard_normal((n, 4, 3)) @ np.linalg.cholesky(S).T
    y1, y2 = np.tile(history[1], (n, 1)), np.tile(history[0], (n, 1))
    paths = np.empty((n, 4, 3))
    for h in range(4):
        new = const + y1 @ A[0].T + y2 @ A[1].T + shocks[:, h]
        paths[:, h] = new
        y2, y1 = y1, new
    flat = paths.reshape(n, 12)
    cond_idx = [1 * 3 + 1, 2 * 3 + 1, 0 * 3 + 2]  # y in 2025Q2 and Q3, i in 2025Q1 (period-major)
    values = np.array([600.0, 603.0, 64.5])
    X = np.column_stack([np.ones(n), flat[:, cond_idx]])
    beta, *_ = np.linalg.lstsq(X, flat, rcond=None)  # Gaussian: the regression is the conditional expectation
    expected = (np.concatenate([[1.0], values]) @ beta).reshape(4, 3)
    resid_sd = np.sqrt(np.mean((flat - X @ beta) ** 2, axis=0)).reshape(4, 3)
    for j, name in enumerate(["c", "y", "i"]):
        got, sd = c[name].to_numpy(), c[f"{name}_se"].to_numpy()
        free = sd > 1e-9
        assert np.all(np.abs(got - expected[:, j]) <= 0.02 * np.maximum(sd, 1e-9) + 1e-9 * ~free + 1e-6)
        assert sd[free] == pytest.approx(resid_sd[:, j][free], rel=0.01)


def test_the_ordering_of_the_variables_does_not_matter(macro, var):
    conditions = {"y": 600.0}
    a = var.conditional_forecast(macro, SPAN, conditions).frame
    b = VAR(["i", "y", "c"], lags=2).fit(macro).conditional_forecast(macro, SPAN, conditions).frame
    for name in ("c", "y", "i", "c_se", "i_se"):
        assert a[name].to_numpy() == pytest.approx(b[name].to_numpy(), rel=1e-9)


def test_conditions_on_everything_give_the_conditions(macro, var):
    c = var.conditional_forecast(macro, SPAN, {"c": 450.0, "y": 590.0, "i": 64.0}).frame
    assert np.allclose(c[["c", "y", "i"]].to_numpy(), [450.0, 590.0, 64.0]) and np.allclose(c[["c_se", "y_se", "i_se"]], 0, atol=1e-8)


def test_bad_conditions(macro, var):
    with pytest.raises(SpecificationError, match="not a variable"):
        var.conditional_forecast(macro, SPAN, {"g": 1.0})
    with pytest.raises(SpecificationError, match="outside the forecast"):
        var.conditional_forecast(macro, SPAN, {"y": {"2026Q1": 1.0}})
    with pytest.raises(SpecificationError, match="at least one condition"):
        var.conditional_forecast(macro, SPAN, {})


def test_bayesian_vars_meet_the_conditions_in_every_draw(macro):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bvar = VAR(["c", "y", "i"], lags=2).fit(macro, method=Bayes(prior=Minnesota(), draws=400))
        a = bvar.conditional_forecast(macro, SPAN, {"y": {"2025Q1": 595.0, "2025Q2": 600.0}}, draws=400, seed=1).frame
        b = bvar.conditional_forecast(macro, SPAN, {"y": {"2025Q1": 595.0, "2025Q2": 600.0}}, draws=400, seed=1).frame
    assert a.equals(b)  # reproducible with a seed
    assert a["y"].iloc[:2].to_numpy() == pytest.approx([595.0, 600.0]) and a["y_se"].iloc[:2].to_numpy() == pytest.approx([0, 0], abs=1e-9)
    assert a["y_se"].iloc[3] > 0 and (a["c_upper"] > a["c_lower"]).all()
    # one posterior draw and no free shocks to draw: each draw's path is the VAR formula with that draw's parameters
    from pymetrix.var import _condition_list, _conditional_moments, _point_path

    periods, D, history = bvar._forecast_inputs(macro, SPAN)
    A, S = bvar._draw_coefs(1)[0]
    point = _point_path(A, bvar.draws_B[0, : len(bvar.arrays.d_names)], D, history, 4)
    conds = _condition_list({"c": 450.0, "y": 590.0, "i": 64.0}, bvar.variables, periods)
    mean, cov, _, _ = _conditional_moments(A, S, point, conds)
    full = bvar.conditional_forecast(macro, SPAN, {"c": 450.0, "y": 590.0, "i": 64.0}, draws=1).frame
    assert full[["c", "y", "i"]].to_numpy() == pytest.approx(mean) and np.allclose(cov, 0, atol=1e-8)


def test_the_dialog_replays(macro, var):
    from pymetrix.commands import ConditionalForecast, Session

    session = Session()
    session.objects.update({"data": macro, "var": var})
    step = ConditionalForecast("var", "2025Q1", "2025Q4", "y[2025Q1:2025Q2]=600, i=64")
    assert step.validate(session) == []
    session.run(step)
    namespace = dict(session.namespace) | {"data": macro, "var": var}
    exec(session.history[-1].code, namespace)
    assert namespace["conditional"].frame.equals(session["conditional"].frame)
    assert session.history[-1].code == ("conditional = var.conditional_forecast(data, ('2025Q1', '2025Q4'), "
                                        "{'y': {'2025Q1': 600.0, '2025Q2': 600.0}, 'i': 64.0})")
    assert ConditionalForecast("var", "2025Q1", "2025Q4", "y=").validate(session)[0].startswith("conditions:")
