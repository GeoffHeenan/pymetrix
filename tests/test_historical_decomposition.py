"""Historical decompositions of VARs.

- Recursive (Cholesky) identification: each shock's contribution against one computed independently from statsmodels'
  VAR estimates (coefficients, residuals, sigma_u with its degrees-of-freedom correction, which the contributions do
  not depend on), by the companion-form recursion c_j(t) = F c_j(t-1) + P_j e_j(t) rather than the moving-average sum,
  and the baseline as statsmodels' fitted values with zero shocks; to 1e-10.
- The parts add up to the data exactly, under long-run, maximum-share and sign restrictions (an identified set: its
  median-target member) and for a Bayesian VAR at its posterior mean.
- Simulated from known structural shocks, the shocks recovered are those shocks (up to the estimation error).
- The dialog records the call; refusals."""

import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.api import VAR as SMVAR

from pymetrix import VAR, Bayes, Cholesky, Dataset, LongRun, MaxShare, Minnesota, SignRestrictions
from pymetrix.commands import FitVAR, HistoricalDecomposition, Session

P_TRUE = np.array([[1.0, 0.0, 0.0], [0.4, 0.8, 0.0], [-0.2, 0.3, 0.6]])


@pytest.fixture(scope="module")
def simulated():
    rng = np.random.default_rng(11)
    T = 600
    A1 = np.array([[0.5, 0.1, 0.0], [0.2, 0.4, 0.1], [0.0, 0.3, 0.6]])
    A2 = np.array([[-0.2, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.0, -0.1]])
    e = rng.standard_normal((T, 3))
    y = np.zeros((T, 3))
    for t in range(2, T):
        y[t] = np.array([0.1, 0.2, -0.1]) + A1 @ y[t - 1] + A2 @ y[t - 2] + P_TRUE @ e[t]
    idx = pd.period_range("1900Q1", periods=T, freq="Q", name="time")
    return pd.DataFrame(y, columns=["a", "b", "c"], index=idx), pd.DataFrame(e, columns=["a", "b", "c"], index=idx)


def test_cholesky_against_statsmodels_by_recursion(simulated):
    frame, _ = simulated
    var = VAR(["a", "b", "c"], lags=2).fit(Dataset.from_frame(frame)).identify(Cholesky(["c", "a", "b"]))
    hd = var.historical_decomposition()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = SMVAR(frame).fit(2)
    K, p = 3, 2
    order = [2, 0, 1]  # c, a, b
    P = np.zeros((K, K))
    P[np.ix_(order, order)] = np.linalg.cholesky(np.asarray(ref.sigma_u)[np.ix_(order, order)])
    U = np.asarray(ref.resid)
    e = np.linalg.solve(P, U.T).T
    F = np.zeros((K * p, K * p))
    F[:K] = np.hstack(list(ref.coefs))
    F[K:, :-K] = np.eye(K * (p - 1))
    state = np.zeros((K, K * p))  # per shock
    for t in range(len(U)):
        for j in range(K):
            impulse = np.zeros(K * p)
            impulse[:K] = P[:, j] * e[t, j]
            state[j] = F @ state[j] + impulse
        for i, v in enumerate(["a", "b", "c"]):  # variable i's contribution from each shock j
            assert np.allclose(hd.contributions[v].iloc[t].to_numpy(), state[:, i], atol=1e-10)
    assert list(hd.contributions["a"].columns) == ["a", "b", "c"]
    # the baseline: statsmodels' intercept and coefficients from the first two observations, with no shocks
    history = list(frame.to_numpy()[:p])
    for t in range(len(U)):
        history.append(ref.intercept + sum(ref.coefs[l] @ history[-1 - l] for l in range(p)))
    assert np.allclose(hd.baseline.to_numpy(), np.array(history[p:]), atol=1e-10)
    assert "Cholesky (c, a, b)" in hd.title and "baseline" in str(hd.table("a"))


def test_the_parts_add_up_to_the_data(simulated):
    frame, _ = simulated
    var = VAR(["a", "b", "c"], lags=2).fit(Dataset.from_frame(frame))
    for scheme in (Cholesky(), LongRun(), MaxShare("a", horizon=8), SignRestrictions({"s1": {"a": "+", "b": "+"}}, rotations=50, seed=1)):
        hd = var.identify(scheme).historical_decomposition()
        total = hd.baseline + pd.DataFrame({v: hd.contributions[v].sum(axis=1) for v in ["a", "b", "c"]})
        assert np.allclose(total.to_numpy(), frame.to_numpy()[2:], atol=1e-10), scheme
        assert np.allclose(hd.residual.to_numpy(), 0, atol=1e-10)
        assert list(hd.shocks.columns) == var.identify(scheme).shocks


def test_a_bayesian_var_at_its_posterior_mean(simulated):
    frame, _ = simulated
    bvar = VAR(["a", "b", "c"], lags=2).fit(Dataset.from_frame(frame), method=Bayes(prior=Minnesota(), draws=10))
    hd = bvar.historical_decomposition()
    total = hd.baseline + pd.DataFrame({v: hd.contributions[v].sum(axis=1) for v in ["a", "b", "c"]})
    assert np.allclose(total.to_numpy(), frame.to_numpy()[2:], atol=1e-10)


def test_the_true_shocks_are_recovered(simulated):
    frame, shocks = simulated
    hd = VAR(["a", "b", "c"], lags=2).fit(Dataset.from_frame(frame)).historical_decomposition()  # Cholesky in VAR order
    recovered = hd.shocks.to_numpy()
    truth = shocks.to_numpy()[2:]
    for j in range(3):
        assert np.corrcoef(recovered[:, j], truth[:, j])[0, 1] > 0.98


def test_the_dialog(simulated):
    frame, _ = simulated
    session = Session()
    session.objects["data"] = Dataset.from_frame(frame)
    session.run(FitVAR("a, b, c", lags="2"))
    session.run(HistoricalDecomposition("var", ordering="c, b, a"))
    script = session.script()
    assert "hd = var.identify(Cholesky(['c', 'b', 'a'])).historical_decomposition()" in script
    ns = {"data": session["data"]}
    exec(script, ns)
    assert np.allclose(ns["hd"].contributions["a"].to_numpy(), session["hd"].contributions["a"].to_numpy())
    session["hd"].plot(["a", "b"])
