"""MSVAR.fit's starting values and speed (pymetrix.ms_var_em).

Checks: the vectorised filter and smoother equal the period-by-period recursions (Hamilton's filter, Kim's smoother,
kept in ms_var as the reference); on simulated VARs whose regimes are clearly separated (in the intercepts, in the
covariances, in everything) the ordinary fit reaches the maximum EM finds when started from the true regime path
(the fit before data-driven starts missed it: its random starts collapsed into two identical regimes) and its
smoothed probabilities classify the periods; on every fit of the earlier tests the log likelihood is no lower than
the one the earlier fit (quantile and random starts, EM then a Nelder-Mead polish of P) reached, recorded below; the
fit is reproducible by seed; collapsed regimes are recognised."""

import numpy as np
import pandas as pd
import pytest
from test_ms_var import _regimes
from test_var_decompositions import _two_regimes

from pymetrix import MSVAR, Dataset
from pymetrix import ms_var as ms
from pymetrix.ms_var import _design
from pymetrix.ms_var_em import collapsed, forward_backward, loglike_from_logdens, start_paths


def _separated(kind, T=300, seed=11):
    """Two persistent regimes (stay with probability 0.97) differing in the intercepts, the covariance, or
    everything: (data, true regime path, switching)."""
    rng = np.random.default_rng(seed)
    A, L = [np.array([[0.3, 0.1], [-0.1, 0.2]])] * 2, [np.array([[1.0, 0.0], [0.5, 0.8]])] * 2
    c = [np.zeros(2), np.zeros(2)]
    if kind == "intercept":
        c, switching = [np.zeros(2), np.array([4.0, -3.0])], "intercept"
    elif kind == "variance":
        L, switching = [L[0], 4 * L[0]], "intercept_variance"
    else:
        c = [np.zeros(2), np.array([2.0, -1.5])]
        A = [A[0], np.array([[0.7, -0.2], [0.3, -0.4]])]
        L, switching = [L[0], np.array([[2.5, 0.0], [-1.0, 1.5]])], "all"
    s, y = np.zeros(T, int), np.zeros((T, 2))
    for t in range(1, T):
        s[t] = s[t - 1] if rng.uniform() < 0.97 else 1 - s[t - 1]
        y[t] = c[s[t]] + A[s[t]] @ y[t - 1] + L[s[t]] @ rng.normal(size=2)
    data = Dataset.from_frame(pd.DataFrame(y, columns=["a", "b"], index=pd.period_range("1900Q1", periods=T, freq="Q")))
    return data, s, switching


def test_vectorised_filter_and_smoother_are_the_recursions():
    rng = np.random.default_rng(0)
    for T, M in [(1, 2), (2, 2), (7, 3), (300, 2), (301, 3), (64, 4)]:
        logd = rng.normal(size=(T, M)) * 3 - 50
        P = rng.dirichlet(np.ones(M), size=M)
        filt, pred, ll = ms.hamilton_filter(np.exp(logd), P)
        sm, trans = ms._smooth(filt, pred, P)
        r = forward_backward(logd, P)
        assert r["ll"] == pytest.approx(ll, abs=1e-10)
        np.testing.assert_allclose(r["filt"], filt, atol=1e-13)
        np.testing.assert_allclose(r["pred"], pred, atol=1e-13)
        np.testing.assert_allclose(r["smooth"], sm, atol=1e-13)
        np.testing.assert_allclose(r["trans"], trans, atol=1e-11)
        assert loglike_from_logdens(np.stack([logd, logd + 1]), np.stack([P, P])) == pytest.approx([ll, ll + T], abs=1e-10)
    # densities far below the floating-point range in one regime: the scaled products stay finite
    logd = np.column_stack([np.zeros(50), np.full(50, -2000.0)])
    r = forward_backward(logd, np.array([[0.9, 0.1], [0.2, 0.8]]))
    assert np.isfinite(r["ll"]) and np.allclose(r["smooth"][:, 0], 1)


# the log likelihood EM reaches from the true regime path (then polished), and the earlier fit's (the same with 1
# and with 10 starts): with the intercepts apart it missed, its random starts collapsing into two identical regimes
# (-893.7) and its quantile start stopping short (-892.8, 69% of periods classified right); the other two it found
SEPARATED = {"intercept": (-791.4939, -892.7744), "variance": (-985.8938, -985.8938), "all": (-897.8184, -897.8184)}


@pytest.mark.parametrize("kind", ["intercept", "variance", "all"])
def test_separated_regimes_are_found(kind):
    data, s, switching = _separated(kind)
    spec = MSVAR(["a", "b"], lags=1, regimes=2, switching=switching)
    Y, Xl = _design(data.frame[["a", "b"]].to_numpy(), 1)
    truth = spec._em(Y, Xl, np.column_stack([1 - s[1:], s[1:]]) * 0.9 + 0.05, 5000, 1e-12)
    assert truth["ll"] == pytest.approx(SEPARATED[kind][0], abs=1e-3)
    fit = spec.fit(data)
    assert fit.loglike >= truth["ll"] - 0.1
    path = fit.regime_path().to_numpy()
    accuracy = max((path == s[1:]).mean(), (path != s[1:]).mean())  # regimes are labelled by the intercepts
    assert accuracy > 0.98  # 1.0, 1.0, 0.993
    assert fit.loglike >= SEPARATED[kind][1] - 1e-4  # the earlier fit's


def _one_equation():
    s, rng = _regimes(500, 0)
    y = np.zeros(500)
    for t in range(1, 500):
        y[t] = [-0.5, 1.5][s[t]] + 0.5 * y[t - 1] + [1.0, 0.5][s[t]] * rng.normal()
    return Dataset.from_frame(pd.DataFrame({"y": y}, index=pd.period_range("1900Q1", periods=500, freq="Q")))


def _two_variables():
    s, rng = _regimes(500, 1)
    Y = np.zeros((500, 2))
    for t in range(1, 500):
        c = np.array([[-0.5, 0.2], [1.0, -0.4]])[s[t]]
        Y[t] = c + np.array([[0.4, 0.1], [0.0, 0.3]]) @ Y[t - 1] + rng.normal(size=2) * 0.6
    return Dataset.from_frame(pd.DataFrame(Y, columns=["a", "b"], index=pd.period_range("1900Q1", periods=500, freq="Q")))


def _macro():
    return Dataset.example("sample_macro").derive(dy="400*dlog(y)", di="400*dlog(i)")


def _gar():
    from _gallery import _gar_data

    return _gar_data()


# the earlier tests' fits: (variables, switching, data, starts, the earlier fit's log likelihood, its time in seconds)
EARLIER = {
    "one equation": (["y"], "intercept_variance", _one_equation, 10, -603.4343666188996, 34.8),
    "two variables, intercepts": (["a", "b"], "intercept", _two_variables, 4, -1057.5414020312392, 14.2),
    "two variables, everything": (["a", "b"], "all", _two_variables, 3, -1052.6210698435457, 14.8),
    "two regimes, intercept and variance": (["a", "b"], "intercept_variance", _two_regimes, 2, -520.9940485418425, 2.5),
    "two regimes, everything": (["a", "b"], "all", _two_regimes, 2, -517.6223019746874, 3.1),
    "two regimes, one variable": (["a"], "intercept_variance", _two_regimes, 1, -310.3129347691964, 1.7),
    "two regimes, one variable, intercepts": (["a"], "intercept", _two_regimes, 1, -322.9637619411655, 1.2),
    "macro, intercepts": (["dy", "di"], "intercept", _macro, 2, -573.6082928874933, 1.6),
    "growth": (["g"], "intercept_variance", _gar, 2, -166.4557281266644, 9.6),
}


@pytest.mark.parametrize("case", list(EARLIER))
def test_no_lower_than_the_earlier_fit(case):
    variables, switching, data, starts, before, _ = EARLIER[case]
    fit = MSVAR(variables, lags=1, regimes=2, switching=switching).fit(data(), starts=starts)
    assert fit.loglike >= before - 1e-7


def test_starts_reproducible_and_collapse():
    data = _two_regimes()
    spec = MSVAR(["a", "b"], lags=1, regimes=2, switching="intercept_variance")
    a, b = spec.fit(data, starts=3, seed=5), spec.fit(data, starts=3, seed=5)
    assert a.loglike == b.loglike
    np.testing.assert_array_equal(a.smoothed.to_numpy(), b.smoothed.to_numpy())
    Y, Xl = _design(data.frame[["a", "b"]].to_numpy(), 1)
    paths = start_paths(spec, Y, Xl, 4, np.random.default_rng(0))
    names = [n for n, _ in paths]
    assert names[0] == "quantiles of the first variable" and names.count("random split") == 3
    assert any("k-means" in n for n in names) and any("one-variable" in n for n in names) and any("variance" in n for n in names)
    assert all(w.shape == (len(Y), 2) and np.allclose(w.sum(axis=1), 1) for _, w in paths)
    r = {"C": np.zeros((2, 2)), "B": np.zeros((2, 2, 2)), "S": np.stack([np.eye(2)] * 2), "w": np.full((len(Y), 2), 0.5)}
    assert collapsed(spec, r, Y)  # identical regimes
    r["C"] = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert not collapsed(spec, r, Y)
    r["w"] = np.column_stack([np.ones(len(Y)), np.zeros(len(Y))])
    assert collapsed(spec, r, Y)  # a regime never in effect
