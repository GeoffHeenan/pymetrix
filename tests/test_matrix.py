"""Matrices with names: algebra that keeps the names, checked against definitions (OLS by matrices against
Equation.fit, inverses, factors and eigenvectors by their defining identities, the Kronecker product and vec by
their formulas), and refusals when names do not meet."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import VAR, Dataset, Equation, Matrix
from pymetrix.errors import SpecificationError


@pytest.fixture(scope="module")
def macro():
    return Dataset.example("sample_macro")


def test_ols_by_matrices_is_the_regression(macro):
    fit = Equation("c ~ y + c[-1]").fit(macro)
    X, y = Matrix.design("c ~ y + c[-1]", macro)
    XtX_inv = (X.T @ X).inv()
    b = XtX_inv @ X.T @ y
    assert b.rows == ("const", "y", "c[-1]") and b.cols == ("c",)
    assert b.values[:, 0] == pytest.approx(Matrix.coef(fit).values[:, 0], rel=1e-8)  # normal equations against its solver
    u = y - X @ b
    n, k = X.shape
    s2 = (u.T @ u)[0, 0] / (n - k)
    V = XtX_inv * s2  # the classic covariance, s^2 (X'X)^-1
    assert V.values == pytest.approx(Matrix.cov(fit).values, rel=1e-8)
    assert V["y", "y"] ** 0.5 == pytest.approx(fit.params.se[1], rel=1e-8)  # a standard error, by name
    assert (X.T @ X).solve(X.T @ y).values == pytest.approx(b.values, rel=1e-8)  # solving, not inverting
    assert X.rows[0] == "2000Q2" and len(X.rows) == n  # rows named by period (the first is lost to the lag)


def test_factors_and_eigenvectors_by_their_identities():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(4, 4))
    S = Matrix(A @ A.T + 4 * np.eye(4), list("abcd"), list("abcd"))
    assert (S @ S.inv()).values == pytest.approx(np.eye(4), abs=1e-12)
    assert S.inv().rows == S.cols
    L = S.chol()
    assert np.allclose(np.triu(L.values, 1), 0) and (L @ L.T).values == pytest.approx(S.values, rel=1e-12)
    values, vectors = S.eig()
    for j in range(4):
        v = vectors.values[:, j]
        assert S.values @ v == pytest.approx(values.values[j, 0] * v, abs=1e-10)  # S v = lambda v
    assert np.all(np.diff(np.abs(values.values[:, 0])) <= 0)  # largest first
    assert S.det() == pytest.approx(np.prod(values.values), rel=1e-10) and S.trace() == pytest.approx(values.values.sum(), rel=1e-10)
    assert S.rank() == 4 and S.power(2).values == pytest.approx(S.values @ S.values)
    with pytest.raises(SpecificationError, match="positive definite"):
        Matrix([[1.0, 2.0], [2.0, 1.0]]).chol()
    with pytest.raises(SpecificationError, match="singular"):
        Matrix([[1.0, 2.0], [2.0, 4.0]]).inv()


def test_kronecker_vec_and_diag_by_their_formulas():
    A = Matrix([[1.0, 2.0], [3.0, 4.0]], ["a1", "a2"], ["x", "y"])
    B = Matrix([[0.0, 5.0], [6.0, 7.0], [1.0, 1.0]], ["b1", "b2", "b3"], ["u", "v"])
    K = A.kron(B)
    assert K.shape == (6, 4) and K.rows[0] == "a1:b1" and K.cols[-1] == "y:v"
    for i in range(2):
        for j in range(2):
            assert K.values[3 * i:3 * i + 3, 2 * j:2 * j + 2] == pytest.approx(A.values[i, j] * B.values)  # blocks a_ij B
    v = A.vec()
    assert list(v.values[:, 0]) == [1.0, 3.0, 2.0, 4.0] and v.rows == ("a1,x", "a2,x", "a1,y", "a2,y")  # columns stacked
    C = Matrix(np.arange(9.0).reshape(3, 3))
    X = Matrix(np.arange(6.0).reshape(3, 2))
    Y = Matrix(np.arange(8.0).reshape(2, 4))
    assert (X @ Y).vec().values == pytest.approx((Y.T.kron(Matrix.identity(3))).values @ X.vec().values)  # vec(XY) = (Y' ⊗ I) vec(X)
    assert list(C.diag().values[:, 0]) == [0.0, 4.0, 8.0] and C.diag().diag().values == pytest.approx(np.diag([0.0, 4.0, 8.0]))


def test_names_must_meet_and_selections_keep_them(macro):
    fit = Equation("c ~ y + i").fit(macro)
    V = Matrix.cov(fit)
    sub = V[["y", "i"], ["y", "i"]]
    assert sub.rows == ("y", "i") and sub.values == pytest.approx(V.values[1:, 1:])
    with pytest.raises(SpecificationError, match="do not match"):
        V @ V[["i", "y", "const"], :]  # columns const, y, i against rows i, y, const
    assert (V @ np.ones((3, 1))).rows == V.rows  # an unnamed matrix meets anything of its size
    with pytest.raises(SpecificationError, match="no row named 'z'"):
        V["z", "y"]
    with pytest.raises(SpecificationError, match="same size"):
        V + Matrix.identity(2)
    assert (V - V).values == pytest.approx(0) and (2 * V).values == pytest.approx(V.values * 2)
    with pytest.raises(SpecificationError, match="used twice"):
        Matrix(np.eye(2), ["a", "a"])
    frame = pd.DataFrame([[1.0, 2.0]], index=["r"], columns=["p", "q"])
    assert Matrix.from_frame(frame).to_frame().equals(frame)


def test_a_vars_matrices_by_name(macro):
    data = macro.with_series(gy="400*dlog(y)", gc="400*dlog(c)")
    var = VAR(["gy", "gc"], lags=2).fit(data)
    A2 = Matrix.var_lag(var, 2)
    assert A2.rows == ("gy", "gc") and A2.cols == ("gy[-2]", "gc[-2]")
    assert A2["gc", "gy[-2]"] == pytest.approx(var.coefs[1][1, 0])
    S = Matrix.var_sigma(var)
    assert S.values == pytest.approx(np.asarray(var.sigma, float)) and S.chol().rows == ("gy", "gc")
