"""Double/debiased machine learning (Chernozhukov et al. 2018): partially linear regression and IV, and the interactive
model's doubly robust ATE and ATT, cross-fitted, against Frisch-Waugh-Lovell, 2SLS, scores coded by hand and a small
Monte Carlo; against the DoubleML package where it is installed."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import IV, Dataset, Equation, SpecificationError, double_ml

CONTROLS = ["x1", "x2", "x3"]


def _plr(n=400, theta=1.5, seed=0, clusters=None):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    d = X @ [0.5, -0.3, 0.2] + rng.normal(size=n)
    y = theta * d + X @ [1.0, 1.0, -1.0] + rng.normal(size=n)
    z = rng.normal(size=n)
    frame = pd.DataFrame(X, columns=CONTROLS).assign(d=d, y=y, id=range(n), z=z)
    if clusters:
        frame["g"] = np.arange(n) % clusters
    return Dataset.from_frame(frame, entity="id")


def _irm(n=500, seed=0, effect=1.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    p = 1 / (1 + np.exp(-(X @ [0.8, -0.5, 0.3])))
    d = (rng.random(n) < p).astype(float)
    y = effect * d + X @ [1.0, 0.5, -0.5] + rng.normal(size=n)
    return Dataset.from_frame(pd.DataFrame(X, columns=CONTROLS).assign(d=d, y=y, id=range(n)), entity="id"), X, d, y


def _ols_resid(Xtr, ytr, Xte, yte):
    A = np.column_stack([np.ones(len(Xtr)), Xtr])
    b = np.linalg.lstsq(A, ytr, rcond=None)[0]
    return yte - np.column_stack([np.ones(len(Xte)), Xte]) @ b


def test_without_cross_fitting_plr_is_ols_frisch_waugh_lovell():
    data = _plr()
    fit = double_ml(data, "y", "d", CONTROLS, folds=1)
    ols = Equation("y ~ d + x1 + x2 + x3").fit(data)
    assert fit.theta == pytest.approx(ols.params["d"], abs=1e-10)


def test_without_cross_fitting_pliv_is_2sls():
    rng = np.random.default_rng(3)
    n = 500
    X, z, u = rng.normal(size=(n, 3)), rng.normal(size=n), rng.normal(size=n)
    d = 0.8 * z + X @ [0.3, 0.2, 0.0] + 0.6 * u + rng.normal(size=n)
    y = 2.0 * d + X @ [1.0, -1.0, 0.5] + u
    data = Dataset.from_frame(pd.DataFrame(X, columns=CONTROLS).assign(d=d, y=y, z=z, id=range(n)), entity="id")
    fit = double_ml(data, "y", "d", CONTROLS, model="pliv", instrument="z", folds=1)
    tsls = Equation("y ~ d + x1 + x2 + x3").fit(data, method=IV(endog=["d"], instruments=["z"]))
    assert fit.theta == pytest.approx(tsls.params["d"], abs=1e-10)


def test_cross_fitted_plr_by_hand():
    """The cross-fitted partialling-out estimate and its score variance, coded from the folds the result used."""
    data = _plr(seed=1)
    fit = double_ml(data, "y", "d", CONTROLS, folds=4, seed=7)
    f = data.frame
    X, y, d = f[CONTROLS].to_numpy(), f["y"].to_numpy(), f["d"].to_numpy()
    folds = fit.splits[:, 0]
    assert sorted(np.bincount(folds)) == [100, 100, 100, 100]
    u, v = np.empty(len(y)), np.empty(len(y))
    for k in range(4):
        te = folds == k
        u[te] = _ols_resid(X[~te], y[~te], X[te], y[te])
        v[te] = _ols_resid(X[~te], d[~te], X[te], d[te])
    theta = (u @ v) / (v @ v)
    psi = (u - theta * v) * v
    se = np.sqrt(np.mean(psi**2) / (len(y) * np.mean(v * v) ** 2))
    assert fit.theta == pytest.approx(theta, abs=1e-12) and fit.se == pytest.approx(se, rel=1e-10)


def test_repeated_splits_aggregate_by_the_median():
    fit = double_ml(_plr(seed=2), "y", "d", CONTROLS, folds=3, repeats=5, seed=1)
    assert fit.thetas.shape == (5,) and fit.splits.shape == (400, 5)
    assert fit.theta == pytest.approx(np.median(fit.thetas))
    assert fit.se == pytest.approx(np.sqrt(np.median(fit.ses**2 + (fit.thetas - fit.theta) ** 2)))
    again = double_ml(_plr(seed=2), "y", "d", CONTROLS, folds=3, repeats=5, seed=1)
    assert np.array_equal(again.thetas, fit.thetas) and fit.provenance.seed == 1  # seeded


def test_clusters_stay_whole_and_the_variance_sums_within_them():
    data = _plr(seed=4, clusters=40)
    fit = double_ml(data, "y", "d", CONTROLS, folds=5, cluster="g", seed=3)
    g = data.frame["g"].to_numpy()
    folds = fit.splits[:, 0]
    assert all(len(np.unique(folds[g == c])) == 1 for c in range(40)) and fit.clusters == 40
    X, y, d = data.frame[CONTROLS].to_numpy(), data.frame["y"].to_numpy(), data.frame["d"].to_numpy()
    u, v = np.empty(len(y)), np.empty(len(y))
    for k in range(5):
        te = folds == k
        u[te], v[te] = _ols_resid(X[~te], y[~te], X[te], y[te]), _ols_resid(X[~te], d[~te], X[te], d[te])
    psi = (u - fit.theta * v) * v
    sums = pd.Series(psi).groupby(g).sum().to_numpy()
    n = len(y)
    assert fit.se == pytest.approx(np.sqrt(40 / 39 * np.sum(sums**2) / (n**2 * np.mean(v * v) ** 2)), rel=1e-10)


def test_irm_ate_and_att_scores_by_hand():
    data, X, d, y = _irm(seed=5)
    fit = double_ml(data, "y", "d", CONTROLS, model="irm", folds=2, seed=2)
    folds = fit.splits[:, 0]
    g0, g1, m = np.empty(len(y)), np.empty(len(y)), np.empty(len(y))
    from pymetrix.dml import _logit_fit, _ols_fit

    for k in range(2):
        te, tr = folds == k, folds != k
        g0[te] = _ols_fit(X[tr & (d == 0)], y[tr & (d == 0)])(X[te])
        g1[te] = _ols_fit(X[tr & (d == 1)], y[tr & (d == 1)])(X[te])
        m[te] = _logit_fit(X[tr], d[tr])(X[te])
    m = np.clip(m, 0.01, 0.99)
    psi_b = g1 - g0 + d * (y - g1) / m - (1 - d) * (y - g0) / (1 - m)
    assert fit.theta == pytest.approx(psi_b.mean(), abs=1e-10)
    assert fit.se == pytest.approx(np.std(psi_b) / np.sqrt(len(y)), rel=1e-8)
    att = double_ml(data, "y", "d", CONTROLS, model="irm", score="ATTE", folds=2, seed=2)
    p = d.mean()
    num = d * (y - g0) / p - m * (1 - d) * (y - g0) / (p * (1 - m))
    assert att.theta == pytest.approx(num.sum() / (d / p).sum(), abs=1e-10)


def test_monte_carlo_unbiased_with_nominal_coverage():
    """Linear DGP, OLS learners, 5-fold cross-fitting: the PLR estimate centres on the truth and its 95% interval
    covers it about 95% of the time (150 replications; the band allows for simulation noise)."""
    est, cover = [], []
    for r in range(150):
        fit = double_ml(_plr(n=300, seed=100 + r), "y", "d", CONTROLS, folds=5, seed=r)
        est.append(fit.theta)
        lo, hi = fit.interval()
        cover.append(lo <= 1.5 <= hi)
    est = np.array(est)
    assert abs(est.mean() - 1.5) < 3 * est.std() / np.sqrt(len(est))
    assert 0.89 <= np.mean(cover) <= 0.99


def test_monte_carlo_irm_coverage():
    cover, est = [], []
    for r in range(100):
        data, *_ = _irm(n=400, seed=300 + r)
        fit = double_ml(data, "y", "d", CONTROLS, model="irm", folds=3, seed=r)
        lo, hi = fit.interval()
        cover.append(lo <= 1.0 <= hi)
        est.append(fit.theta)
    assert abs(np.mean(est) - 1.0) < 3 * np.std(est) / 10 and 0.87 <= np.mean(cover) <= 1.0


def test_scikit_learn_learners():
    pytest.importorskip("sklearn")
    from pymetrix import GradientBoosting, RandomForest

    data = _plr(n=400, seed=8)
    forest = double_ml(data, "y", "d", CONTROLS, learner="forest", folds=3, seed=1)
    lasso = double_ml(data, "y", "d", CONTROLS, learner="lasso", folds=3)
    boost = double_ml(data, "y", "d", CONTROLS, learner=GradientBoosting(trees=100), folds=3)
    assert abs(lasso.theta - 1.5) < 4 * lasso.se and abs(forest.theta - 1.5) < 0.3 and abs(boost.theta - 1.5) < 0.3
    assert "random forest" in forest.learners["l(X) = E[Y|X]"] and "lasso" in lasso.summary().notes[2]
    irm, *_ = _irm(n=400, seed=9)
    fit = double_ml(irm, "y", "d", CONTROLS, model="irm", learner=RandomForest(trees=100), learner_treatment="lasso", folds=3)
    assert "l1-penalised logit" in fit.learners["m(X) = P(D=1|X)"] and abs(fit.theta - 1.0) < 0.5


def test_against_the_doubleml_package():
    """With the same folds and least-squares learners, DoubleML's PLR (partialling out) and IRM (ATE) agree."""
    dml = pytest.importorskip("doubleml")
    from sklearn.linear_model import LinearRegression

    data = _plr(seed=11)
    fit = double_ml(data, "y", "d", CONTROLS, folds=5, seed=4)
    frame = data.frame.reset_index()
    obj = dml.DoubleMLData(frame, "y", "d", CONTROLS)
    model = dml.DoubleMLPLR(obj, LinearRegression(), LinearRegression(), n_folds=5, score="partialling out")
    folds = fit.splits[:, 0]
    model.set_sample_splitting([[(np.where(folds != k)[0], np.where(folds == k)[0]) for k in range(5)]])
    model.fit()
    assert fit.theta == pytest.approx(float(model.coef[0]), abs=1e-8) and fit.se == pytest.approx(float(model.se[0]), rel=1e-6)
    from sklearn.linear_model import LogisticRegression

    data, *_ = _irm(seed=12)
    ate = double_ml(data, "y", "d", CONTROLS, model="irm", folds=4, seed=6)
    obj = dml.DoubleMLData(data.frame.reset_index(), "y", "d", CONTROLS)
    model = dml.DoubleMLIRM(obj, LinearRegression(), LogisticRegression(penalty=None, tol=1e-12, max_iter=10_000), n_folds=4, score="ATE",
                            trimming_threshold=0.01)
    folds = ate.splits[:, 0]
    model.set_sample_splitting([[(np.where(folds != k)[0], np.where(folds == k)[0]) for k in range(4)]])
    model.fit()
    assert ate.theta == pytest.approx(float(model.coef[0]), abs=1e-5) and ate.se == pytest.approx(float(model.se[0]), rel=1e-4)  # our logit's tiny ridge


def test_errors():
    data = _plr()
    with pytest.raises(SpecificationError, match="needs controls"):
        double_ml(data, "y", "d", [])
    with pytest.raises(SpecificationError, match="needs an instrument"):
        double_ml(data, "y", "d", CONTROLS, model="pliv")
    with pytest.raises(SpecificationError, match="0/1 treatment"):
        double_ml(data, "y", "d", CONTROLS, model="irm")
    with pytest.raises(SpecificationError, match="learner"):
        double_ml(data, "y", "d", CONTROLS, learner="svm")
    with pytest.raises(SpecificationError, match="no series"):
        double_ml(data, "y", "nope", CONTROLS)


def test_the_dialog_records_the_call_and_the_script_replays():
    from pymetrix.commands import COMMANDS, Session
    from pymetrix.views import sections

    s = Session()
    s.objects["data"] = _plr(seed=12, clusters=20)
    s.run(COMMANDS["DoubleMachineLearning"].cls("y", "d", "x1, x2, x3", folds=3, repeats=2, cluster="g", seed=5))
    code = s.history[-1].code
    assert code == "dml = double_ml(data, 'y', 'd', ['x1', 'x2', 'x3'], folds=3, repeats=2, cluster='g', seed=5)"
    direct = double_ml(s.objects["data"], "y", "d", ["x1", "x2", "x3"], folds=3, repeats=2, seed=5, cluster="g")
    assert s.objects["dml"].theta == direct.theta and s.objects["dml"].se == direct.se
    assert [t for t, _ in sections(s.objects["dml"])][:3] == ["Summary", "Graph", "Splits"]
    namespace = {"data": s.objects["data"]}
    exec(s.script(), namespace)  # noqa: S102 - the generated script reproduces the estimate
    assert namespace["dml"].theta == direct.theta
    assert COMMANDS["DoubleMachineLearning"].cls("y", "d", "x1", model="pliv").validate(s) == ["the partially linear IV model needs an instrument"]
