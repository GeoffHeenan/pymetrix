"""Interactive fixed effects (Bai 2009): the coefficients recovered from a panel with known factors that drive the
regressors (where two-way fixed effects fail), r = 0 equal to pooled least squares and to two-way fixed effects, the
number of factors by Bai and Ng's criteria, the bias correction against its formula with loops, and coverage."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, SpecificationError, interactive_fe


def _panel(N=80, T=40, r=2, seed=0, sd=1.0):
    rng = np.random.default_rng(seed)
    F, L = rng.normal(size=(T, r)), rng.normal(size=(N, r))
    common = L @ F.T
    x1 = common + rng.normal(size=(N, T))
    x2 = 0.5 * common + np.outer(L[:, 0], np.ones(T)) + rng.normal(size=(N, T))
    y = 1.0 + 1.0 * x1 + 2.0 * x2 + common + sd * rng.normal(size=(N, T))
    frame = pd.DataFrame({"unit": np.repeat(np.arange(N), T), "year": np.tile(np.arange(1980, 1980 + T), N),
                          "y": y.ravel(), "x1": x1.ravel(), "x2": x2.ravel()})
    return Dataset.from_frame(frame, entity="unit", time="year", freq="Y")


def test_recovers_beta_where_two_way_fixed_effects_fail():
    data = _panel()
    ife = interactive_fe(data, "y", ["x1", "x2"], factors=2)
    twfe = Equation("y ~ x1 + x2 + fe(unit) + fe(year)").fit(data)
    err_ife = np.abs(ife.beta[1:] - [1.0, 2.0])
    err_fe = np.abs(np.array([twfe.params["x1"], twfe.params["x2"]]) - [1.0, 2.0])
    assert err_ife.max() < 0.03 and err_fe.max() > 10 * err_ife.max()
    assert list(ife.params.names) == ["const", "x1", "x2"]
    assert ife.F.shape == (40, 2) and np.allclose(ife.F.T @ ife.F / 40, np.eye(2), atol=1e-10)  # F'F/T = I
    assert ife.converged and np.allclose(ife.resid + ife.fitted, data.frame["y"].reindex(ife.resid.index))


def test_the_estimate_is_a_fixed_point_and_no_worse_than_the_truth():
    data = _panel(seed=1)
    ife = interactive_fe(data, "y", ["x1", "x2"], factors=2)
    f = data.frame
    Y, X = f["y"].unstack().to_numpy(), np.stack([np.ones((80, 40)), f["x1"].unstack().to_numpy(), f["x2"].unstack().to_numpy()], axis=2)

    def ssr(b):
        W = Y - X @ b
        return np.linalg.eigvalsh(W.T @ W)[:-2].sum()

    assert ssr(ife.beta) <= ssr(np.array([1.0, 1.0, 2.0])) + 1e-8
    F = ife.F.to_numpy()
    M = np.eye(40) - F @ F.T / 40
    A = sum(X[i].T @ M @ X[i] for i in range(80))
    b = sum(X[i].T @ M @ Y[i] for i in range(80))
    assert np.allclose(np.linalg.solve(A, b), ife.beta, atol=1e-7)


def test_no_factors_is_pooled_ols_and_two_way_fixed_effects():
    data = _panel(seed=2)
    pooled = interactive_fe(data, "y", ["x1", "x2"], factors=0, cov="homoskedastic")
    ols = Equation("y ~ x1 + x2").fit(data)
    assert np.allclose(pooled.beta, ols.params.values, atol=1e-10) and np.allclose(pooled.se, ols.params.se, rtol=1e-8)
    twoway = interactive_fe(data, "y", ["x1", "x2"], factors=0, effects="twoway", cov="homoskedastic")
    fe = Equation("y ~ x1 + x2 + fe(unit) + fe(year)").fit(data)
    assert np.allclose(twoway.beta, fe.params.values, atol=1e-10) and np.allclose(twoway.se, fe.params.se, rtol=1e-8)
    unit = interactive_fe(data, "y", ["x1", "x2"], factors=0, effects="unit", cov="homoskedastic")
    assert np.allclose(unit.beta, Equation("y ~ x1 + x2 + fe(unit)").fit(data).params.values, atol=1e-10)


def test_additive_effects_with_factors():
    """Unit and period effects on top of the factor: the two-way transform keeps the factor structure."""
    rng = np.random.default_rng(3)
    N, T = 60, 30
    a, d = rng.normal(size=(N, 1)) * 3, rng.normal(size=(1, T)) * 3
    F, L = rng.normal(size=(T, 1)), rng.normal(size=(N, 1))
    x = L @ F.T + a + d + rng.normal(size=(N, T))
    y = 1.5 * x + a + d + L @ F.T + rng.normal(size=(N, T))
    frame = pd.DataFrame({"unit": np.repeat(np.arange(N), T), "year": np.tile(np.arange(T) + 1990, N), "y": y.ravel(), "x": x.ravel()})
    data = Dataset.from_frame(frame, entity="unit", time="year", freq="Y")
    fit = interactive_fe(data, "y", ["x"], factors=1, effects="twoway")
    assert abs(fit.beta[0] - 1.5) < 0.03 and list(fit.params.names) == ["x"]


def test_number_of_factors_by_bai_ng():
    fit = interactive_fe(_panel(seed=4), "y", ["x1", "x2"], factors="auto", max_factors=6)
    assert fit.factors == 2 and fit.criterion == "icp1" and list(fit.criteria.index) == list(range(7))
    assert fit.provenance.resolved == {"factors": 2}
    assert int(fit.criteria["pcp2"].idxmin()) == 2 and int(fit.criteria["icp2"].idxmin()) == 2


def test_bias_correction_against_the_formula_with_loops():
    data = _panel(N=30, T=12, r=1, seed=5)
    raw = interactive_fe(data, "y", ["x1", "x2"], factors=1, effects="twoway")
    fit = interactive_fe(data, "y", ["x1", "x2"], factors=1, effects="twoway", bias_correction=True)
    assert np.allclose(fit.beta_uncorrected, raw.beta)
    f = data.frame
    N, T = 30, 12

    def within(A):
        return A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()

    X = [np.column_stack([within(f[v].unstack().to_numpy())[i] for v in ("x1", "x2")]) for i in range(N)]
    E = raw.resid.unstack().to_numpy()
    F, L = raw.F.to_numpy(), raw.loadings.to_numpy()
    M = np.eye(T) - F @ F.T / T
    G = np.linalg.inv(L.T @ L / N)
    a = [[float(L[i] @ G @ L[k]) for k in range(N)] for i in range(N)]
    Z = [M @ X[i] - sum(a[i][k] * M @ X[k] for k in range(N)) / N for i in range(N)]
    D = sum(Z[i].T @ Z[i] for i in range(N)) / (N * T)
    sig_i = [np.mean(E[i] ** 2) for i in range(N)]
    Omega = np.diag((E**2).mean(axis=0))
    V = [sum(a[i][k] * X[k] for k in range(N)) / N for i in range(N)]
    B = -np.linalg.solve(D, sum((X[i] - V[i]).T @ F / T @ G @ L[i] * sig_i[i] for i in range(N)) / N)
    C = -np.linalg.solve(D, sum(X[i].T @ M @ Omega @ F @ G @ L[i] for i in range(N)) / (N * T))
    assert np.allclose(fit.beta, raw.beta - B / N - C / T, atol=1e-10)


def test_monte_carlo_coverage():
    cover = []
    for r in range(40):
        fit = interactive_fe(_panel(N=60, T=25, r=1, seed=100 + r), "y", ["x1", "x2"], factors=1)
        cover.append(abs(fit.beta[1] - 1.0) < 1.96 * fit.se[1])
    assert 0.85 <= np.mean(cover) <= 1.0


def test_errors():
    data = _panel(N=10, T=8)
    with pytest.raises(SpecificationError, match="balanced"):
        interactive_fe(Dataset.from_frame(data.frame.iloc[1:].reset_index(), entity="unit", time="year", freq="Y"), "y", ["x1"])
    with pytest.raises(SpecificationError, match="factors is"):
        interactive_fe(data, "y", ["x1"], factors=8)
    with pytest.raises(SpecificationError, match="criterion"):
        interactive_fe(data, "y", ["x1"], factors="auto", criterion="aic")
    with pytest.raises(SpecificationError, match="panel"):
        interactive_fe(Dataset.example("card"), "lwage", ["educ"])


def test_the_dialog_records_the_call_and_the_script_replays():
    from pymetrix.commands import COMMANDS, Session
    from pymetrix.views import sections

    s = Session()
    s.objects["data"] = _panel(N=30, T=15, r=1, seed=6)
    s.run(COMMANDS["FitInteractiveFE"].cls("y", "x1, x2", choose=True, max_factors=4, effects="twoway", cov="cluster"))
    assert s.history[-1].code == "ife = interactive_fe(data, 'y', ['x1', 'x2'], factors='auto', effects='twoway', cov='cluster', max_factors=4)"
    direct = interactive_fe(s.objects["data"], "y", ["x1", "x2"], factors="auto", effects="twoway", cov="cluster", max_factors=4)
    assert np.array_equal(s.objects["ife"].beta, direct.beta) and np.array_equal(s.objects["ife"].cov, direct.cov)
    assert "Number of factors" in [t for t, _ in sections(direct)]
    namespace = {"data": s.objects["data"]}
    exec(s.script(), namespace)  # noqa: S102 - the generated script reproduces the estimate
    assert np.array_equal(namespace["ife"].beta, direct.beta)
