"""Structural VARs by restrictions on A and B, checked independently: just-identified recursive A-B and B models equal
to the Cholesky factor; long-run restrictions equal to Blanchard and Quah's closed form; an over-identified A-B model
against statsmodels' SVAR; the standard errors of a diagonal B against their closed form b / sqrt(2T); the
likelihood-ratio statistic by hand, its rejection of a false restriction and its size for a true one; combined short- and long-run restrictions
holding exactly and recovering a simulated model's impact; refusals; and the dialog."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from pymetrix import AB, VAR, Dataset, LongRun
from pymetrix.errors import SpecificationError

S = "*"


def _simulated(T=400, seed=0, P=None, A1=None):
    rng = np.random.default_rng(seed)
    A1 = np.array([[0.5, 0.1, 0.0], [0.2, 0.4, 0.1], [0.0, 0.2, 0.3]]) if A1 is None else A1
    if P is None:
        A0 = np.array([[1, 0, 0], [-0.5, 1, 0], [0.3, -0.4, 1.0]])
        P = np.linalg.solve(A0, np.diag([1.0, 0.7, 0.5]))
    y = np.zeros((T, 3))
    for t in range(1, T):
        y[t] = A1 @ y[t - 1] + P @ rng.normal(size=3)
    idx = pd.period_range("1900Q1", periods=T, freq="Q")
    return Dataset.from_frame(pd.DataFrame(y, columns=["a", "b", "c"], index=idx)), y


@pytest.fixture(scope="module")
def fitted():
    data, y = _simulated()
    return VAR(["a", "b", "c"], lags=1).fit(data), y


RECURSIVE_A = AB(A=[[1, 0, 0], [S, 1, 0], [S, S, 1]], B=[[S, 0, 0], [0, S, 0], [0, 0, S]])


def test_recursive_models_are_the_cholesky_factor(fitted):
    var, _ = fitted
    chol = var.impact()
    assert var.identify(RECURSIVE_A).impact() == pytest.approx(chol, abs=1e-7)
    assert var.identify(AB(B=[[S, 0, 0], [S, S, 0], [S, S, S]])).impact() == pytest.approx(chol, abs=1e-7)
    est = var.identify(RECURSIVE_A).structural()
    assert est.LR is None and est.df == 0 and est.loglike == pytest.approx(
        -est.nobs / 2 * (3 * np.log(2 * np.pi) + np.linalg.slogdet(var.estimate.sigma)[1] + 3), rel=1e-10)


def test_long_run_restrictions_are_blanchard_quah(fitted):
    var, _ = fitted
    scheme = AB(long_run=[[S, 0, 0], [S, S, 0], [S, S, S]])
    assert var.identify(scheme).impact() == pytest.approx(var.identify(LongRun()).impact(), abs=1e-7)


def test_an_over_identified_model_against_statsmodels(fitted):
    from statsmodels.tsa.vector_ar.svar_model import SVAR

    var, y = fitted
    A = np.array([[1, 0, 0], ["E", 1, 0], [0, "E", 1]], dtype=object)
    B = np.array([["E", 0, 0], [0, "E", 0], [0, 0, "E"]], dtype=object)
    ref = SVAR(y, svar_type="AB", A=A, B=B).fit(maxlags=1, trend="c")
    assert np.asarray(ref.sigma_u) == pytest.approx(var.estimate.sigma, rel=1e-10)  # the same reduced form
    est = var.identify(AB(A=[[1, 0, 0], [S, 1, 0], [0, S, 1]], B=[[S, 0, 0], [0, S, 0], [0, 0, S]])).structural()
    assert est.A == pytest.approx(ref.A, abs=1e-6) and np.abs(est.B) == pytest.approx(np.abs(ref.B), abs=1e-6)
    # the true A[c, a] is 0.3: setting it to zero is rejected, by the LR statistic computed by hand
    sigma, T = var.estimate.sigma, est.nobs
    P = est.impact
    ll = -T / 2 * (3 * np.log(2 * np.pi) + np.linalg.slogdet(P @ P.T)[1] + np.trace(np.linalg.solve(P @ P.T, sigma)))
    unrestricted = -T / 2 * (3 * np.log(2 * np.pi) + np.linalg.slogdet(sigma)[1] + 3)
    assert est.loglike == pytest.approx(ll, rel=1e-10) and est.LR == pytest.approx(2 * (unrestricted - ll), rel=1e-8)
    assert est.df == 1 and sps.chi2.sf(est.LR, 1) < 1e-6


def test_standard_errors_of_a_diagonal_b(fitted):
    var, _ = fitted
    est = var.identify(RECURSIVE_A).structural()
    b = np.diag(est.B)
    assert np.diag(est.B_se) == pytest.approx(b / np.sqrt(2 * est.nobs), rel=1e-3)  # the information matrix's closed form
    assert "A[b, a]" in est.summary().to_frame().index


def test_a_true_restriction_keeps_its_size():
    A0 = np.array([[1, 0, 0], [-0.5, 1, 0], [0.0, -0.4, 1.0]])  # A[c, a] is truly zero
    P = np.linalg.solve(A0, np.diag([1.0, 0.7, 0.5]))
    scheme = AB(A=[[1, 0, 0], [S, 1, 0], [0, S, 1]], B=[[S, 0, 0], [0, S, 0], [0, 0, S]])
    p = [sps.chi2.sf(VAR(["a", "b", "c"], lags=1).fit(_simulated(T=300, seed=50 + k, P=P)[0]).identify(scheme).structural().LR, 1)
         for k in range(100)]
    assert np.mean(np.array(p) < 0.05) == pytest.approx(0.05, abs=0.06)


def test_combined_short_and_long_run_restrictions():
    # b does not respond on impact to the third shock; a does not respond in the long run to the second or third
    A1 = np.array([[0.5, 0.1, 0.0], [0.2, 0.4, 0.1], [0.0, 0.2, 0.3]])
    Xi = np.linalg.inv(np.eye(3) - A1)
    L = np.array([[1.0, 0, 0], [0.4, 0.8, 0.3], [-0.2, 0.5, 0.6]])  # long-run effects with the zeros
    P = np.linalg.solve(Xi, L)
    P[:, 2] -= P[1, 2] / P[1, 1] * P[:, 1]  # rotate shocks 2 and 3 so that P[b, shock 3] = 0, keeping L[a, 2:] = 0
    data, _ = _simulated(T=20000, seed=3, P=P, A1=A1)
    var = VAR(["a", "b", "c"], lags=1).fit(data)
    scheme = AB(B=[[S, S, S], [S, S, 0], [S, S, S]], long_run=[[S, 0, 0], [S, S, S], [S, S, S]])
    est = var.identify(scheme).structural()
    Pi = est.impact
    long = np.linalg.inv(np.eye(3) - var.coefs.sum(axis=0)) @ Pi
    assert Pi[1, 2] == 0 and long[0, 1:] == pytest.approx([0, 0], abs=1e-10)
    assert Pi @ Pi.T == pytest.approx(var.estimate.sigma, abs=1e-7)  # just identified: Sigma reproduced
    assert np.abs(Pi) == pytest.approx(np.abs(P), abs=0.05)


def test_refusals(fitted):
    var, _ = fitted
    with pytest.raises(SpecificationError, match="not identified"):
        var.identify(AB())
    with pytest.raises(SpecificationError, match="3 x 3"):
        var.identify(AB(B=[[S, 0], [S, S]]))
    with pytest.raises(SpecificationError, match="a number"):
        var.identify(AB(B=[[S, 0, 0], [S, "x", 0], [S, S, S]]))
    with pytest.raises(SpecificationError, match="identity"):
        var.identify(AB(A=[[1, 0, 0], [S, 1, 0], [S, S, 1]], long_run=[[S, 0, 0], [S, S, 0], [S, S, S]]))
    with pytest.raises(SpecificationError, match="rank"):  # six parameters, but B[a] and B[b] are not separately identified
        var.identify(AB(B=[[S, S, 0], [S, S, 0], [0, 0, S]], A=[[1, 0, 0], [0, 1, 0], [S, 0, 1]]))
    with pytest.raises(SpecificationError, match="first"):
        var.structural()


def test_the_dialog():
    from pymetrix.commands import COMMANDS, Session

    data, _ = _simulated()
    s = Session()
    s.objects["data"] = data
    s.objects["var"] = VAR(["a", "b", "c"], lags=1).fit(data)
    s.run(COMMANDS["IdentifyStructuralVAR"].cls("var", A="1 0 0; * 1 0; * * 1", B="* 0 0; 0 * 0; 0 0 *"))
    assert s.history[-1].code == ("svar = var.identify(AB(A=[[1.0, 0.0, 0.0], ['*', 1.0, 0.0], ['*', '*', 1.0]], "
                                  "B=[['*', 0.0, 0.0], [0.0, '*', 0.0], [0.0, 0.0, '*']]))")
    assert s.objects["svar"].impact() == pytest.approx(s.objects["var"].impact(), abs=1e-7)
