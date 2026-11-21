"""Conditional logit: Chamberlain's fixed-effects logit against statsmodels' ConditionalLogit (to 1e-6), and McFadden's
choice model against its likelihood coded by hand and maximised by scipy; the two share one likelihood."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from pymetrix import Dataset, SpecificationError, choice_logit, conditional_logit
from pymetrix.clogit import _group_terms, conditional_loglike


@pytest.fixture(scope="module")
def wages():
    return Dataset.example("wage_panel").with_series(hours_k="hours / 1000")


def test_panel_against_statsmodels(wages):
    sm = pytest.importorskip("statsmodels.discrete.conditional_models")
    fit = conditional_logit(wages, "union", ["married", "hours_k", "exper"])
    f = wages.frame.reset_index()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sm.ConditionalLogit(f["union"], f[["married", "hours_k", "exper"]], groups=f["nr"]).fit(method="newton", tol=1e-12, maxiter=200, disp=0)
    assert np.allclose(fit.values, ref.params.to_numpy(), atol=1e-6)
    assert np.allclose(fit.se, ref.bse.to_numpy(), rtol=1e-6)
    assert fit.loglike == pytest.approx(ref.llf, abs=1e-6)
    assert fit.groups == 246 and fit.dropped_groups == 299 and fit.nobs == 1968  # 299 men always or never in a union


def test_the_recursion_against_enumerating_subsets():
    """Howard's recursion for the denominator (and its derivatives) against summing over every subset."""
    from itertools import combinations

    rng = np.random.default_rng(0)
    X, beta = rng.normal(size=(7, 2)), np.array([0.7, -1.2])
    for k in (1, 2, 3, 6):
        logden, g, H = _group_terms(X, k, beta)
        subsets = [list(c) for c in combinations(range(7), k)]
        terms = np.array([np.exp(X[c].sum(0) @ beta) for c in subsets])
        sums = np.array([X[c].sum(0) for c in subsets])
        w = terms / terms.sum()
        assert logden == pytest.approx(np.log(terms.sum()), abs=1e-12)
        assert np.allclose(g, w @ sums) and np.allclose(H, (sums * w[:, None]).T @ sums - np.outer(w @ sums, w @ sums))
    big = _group_terms(rng.normal(size=(40, 2)) * 30, 20, beta)  # rescaled as it goes: no overflow
    assert np.isfinite(big[0]) and np.all(np.isfinite(big[2]))


def _trips(n=400, seed=0):
    rng = np.random.default_rng(seed)
    alts = ["air", "bus", "car", "train"]
    rows = []
    for i in range(n):
        available = [a for a in alts if a != "air" or rng.random() < 0.6]  # air is not always available
        income = rng.normal(3, 1)
        u = []
        for a in available:
            cost, time = rng.normal(5, 2), rng.normal(3, 1)
            u.append({"air": 0.0, "bus": -0.5, "car": 0.5, "train": 0.2}[a] + {"air": 0.0, "bus": -0.2, "car": 0.3, "train": 0.0}[a] * income
                     - 0.4 * cost - 0.8 * time + rng.gumbel())
            rows.append({"trip": i, "mode": a, "cost": cost, "time": time, "income": income})
        best = int(np.argmax(u))
        for j in range(len(available)):
            rows[len(rows) - len(available) + j]["chosen"] = float(j == best)
    return Dataset.from_frame(pd.DataFrame(rows), entity=["trip", "mode"])


def test_choice_model_against_its_likelihood_by_hand():
    data = _trips()
    fit = choice_logit(data, "chosen", ["cost", "time"], alternative_constants=True, case_variables=["income"])
    f = data.frame.reset_index()
    others = ["bus", "car", "train"]  # air is the base
    D = np.column_stack([(f["mode"] == a).to_numpy(float) for a in others])
    X = np.column_stack([f[["cost", "time"]].to_numpy(), D, D * f[["income"]].to_numpy()])
    y, case = f["chosen"].to_numpy(), f["trip"].to_numpy()

    def negll(b):
        eta = X @ b
        logsum = pd.Series(np.exp(eta)).groupby(case).sum().map(np.log).to_numpy()
        return -(eta[y == 1].sum() - logsum.sum())

    ref = optimize.minimize(negll, np.zeros(X.shape[1]), method="BFGS", options={"gtol": 1e-9})
    assert np.allclose(fit.values, ref.x, atol=1e-4) and fit.loglike == pytest.approx(-ref.fun, abs=1e-8)
    assert list(fit.params.names) == ["cost", "time", "bus", "car", "train", "income:bus", "income:car", "income:train"]
    sets = f.groupby("trip").size().to_numpy()
    assert fit.loglike_null == pytest.approx(-np.log(sets).sum()) and fit.groups == 400  # choice sets of three or four
    probs = fit.probabilities.groupby(level="trip").sum()
    assert np.allclose(probs, 1.0) and abs(fit.values[0] + 0.4) < 0.15 and abs(fit.values[1] + 0.8) < 0.25


def test_one_likelihood_for_both():
    """McFadden's model is Chamberlain's likelihood with each case a group: the same estimates either way, and
    statsmodels' ConditionalLogit with the cases as groups agrees."""
    sm = pytest.importorskip("statsmodels.discrete.conditional_models")
    data = _trips(n=200, seed=1)
    choice = choice_logit(data, "chosen", ["cost", "time"])
    panel = conditional_logit(data, "chosen", ["cost", "time"], group="trip")
    assert np.allclose(choice.values, panel.values, atol=1e-10) and np.allclose(choice.se, panel.se)
    f = data.frame.reset_index()
    ref = sm.ConditionalLogit(f["chosen"], f[["cost", "time"]], groups=f["trip"]).fit(method="newton", tol=1e-12, disp=0)
    assert np.allclose(choice.values, ref.params.to_numpy(), atol=1e-6) and np.allclose(choice.se, ref.bse.to_numpy(), rtol=1e-6)
    ll, grad, _, _ = conditional_loglike(choice.values, [(g[["cost", "time"]].to_numpy(), g["chosen"].to_numpy()) for _, g in f.groupby("trip")])
    assert ll == pytest.approx(choice.loglike) and np.abs(grad).max() < 1e-6


def test_clustered_errors(wages):
    fit = conditional_logit(wages, "union", ["married", "exper"], cov="cluster")
    plain = conditional_logit(wages, "union", ["married", "exper"])
    assert np.allclose(fit.values, plain.values) and not np.allclose(fit.se, plain.se) and "clustered" in fit.summary().notes[-1]


def test_errors(wages):
    with pytest.raises(SpecificationError, match="0 or 1"):
        conditional_logit(wages, "lwage", ["married"])
    with pytest.raises(SpecificationError, match="do not vary within groups"):
        conditional_logit(wages, "union", ["married", "educ"])  # education is fixed for each man
    with pytest.raises(SpecificationError, match="varies across"):
        choice_logit(_trips(n=30), "chosen", ["time"], case_variables=["cost"])
    with pytest.raises(SpecificationError, match="base alternative"):
        choice_logit(_trips(n=30), "chosen", ["cost"], alternative_constants=True, base="plane")


def test_the_dialogs_record_the_calls_and_the_script_replays(wages):
    from pymetrix.commands import COMMANDS, Session

    s = Session()
    s.objects["data"] = wages
    s.objects["trips"] = _trips(n=100, seed=2)
    s.run(COMMANDS["FitConditionalLogit"].cls("union", "married, exper", cov="cluster"))
    assert s.history[-1].code == "clogit = conditional_logit(data, 'union', ['married', 'exper'], cov='cluster')"
    s.run(COMMANDS["FitChoiceLogit"].cls("chosen", "cost, time", data="trips", alternative_constants=True, case_variables="income", base="bus"))
    assert s.history[-1].code == ("choice = choice_logit(trips, 'chosen', regressors=['cost', 'time'], alternative_constants=True, "
                                  "case_variables=['income'], base='bus')")
    direct = choice_logit(s.objects["trips"], "chosen", ["cost", "time"], alternative_constants=True, case_variables=["income"], base="bus")
    assert np.array_equal(s.objects["choice"].values, direct.values) and "air" in direct.params.names
    namespace = {"data": wages, "trips": s.objects["trips"]}
    exec(s.script(), namespace)  # noqa: S102 - the generated script reproduces both estimates
    assert np.array_equal(namespace["choice"].values, direct.values)
    assert np.array_equal(namespace["clogit"].values, conditional_logit(wages, "union", ["married", "exper"], cov="cluster").values)
