"""Named matrices, the extensions: decompositions, functions of a matrix, the matrices of matrix calculus, matrix
equations and assembly.

Checked against what:
- every factor against NumPy or SciPy on the same matrix (1e-12), and by reconstruction: U diag(s) V' = A, Q R = A,
  P L U = A, Q T Q' = A, Q AA Z' = A and Q BB Z' = B (1e-12, relative to the matrix's size);
- the general eigenvalues against numpy.linalg.eig (complex, as a set), A v = λ v, and the symmetric ones unchanged;
- expm against its power series, logm and sqrtm by expm(logm A) = A and sqrtm(A)^2 = A, norms and condition numbers
  against numpy.linalg;
- null, orth and perp by their definitions (A N = 0, orthonormal columns, perp' A = 0 and [A perp] of full rank),
  perp also by the VECM identity that β⊥ (α⊥' Γ β⊥)^-1 α⊥' does not depend on the choice of complement;
- commutation K vec(A) = vec(A'), duplication D vech(A) = vec(A) for symmetric A, elimination L vec(A) = vech(A);
- the discrete Lyapunov equation against a VAR(2)'s unconditional covariance from the vec formula
  vec(Σ) = (I - F⊗F)^-1 vec(Q) on its companion form (assembled with hcat and vcat), the continuous one and the
  Sylvester equation against their vec formulas, the discrete Riccati equation against the Kalman filter's
  covariance recursion iterated to convergence;
- the matrix power against numpy.linalg.matrix_power and negative powers against the inverse (1e-12), the deprecated
  ``A ** n`` (a warning, the same values), and the upper Cholesky factor against scipy.linalg.cholesky(lower=False);
- the names each result carries, as documented, and the refusals when names do not meet.
"""

import warnings

import numpy as np
import pytest
from scipy import linalg

from pymetrix import VAR, Dataset, Matrix
from pymetrix.errors import SpecificationError

TOL = 1e-12


@pytest.fixture(scope="module")
def macro():
    return Dataset.example("sample_macro")


def _random(m, n, seed=0, rows=None, cols=None):
    rng = np.random.default_rng(seed)
    return Matrix(rng.normal(size=(m, n)), rows or [f"r{i}" for i in range(m)], cols or [f"c{j}" for j in range(n)])


def _close(a, b, tol=TOL):
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b)) <= tol * max(1.0, np.max(np.abs(b)))


def test_decompositions_reconstruct_and_match_numpy_and_scipy():
    A = _random(5, 3, 1)
    U, s, V = A.svd()
    assert U.rows == A.rows and V.rows == A.cols and U.cols == s.rows == V.cols == ("s1", "s2", "s3")
    assert _close((U @ s.diag() @ V.T).values, A.values)  # U diag(s) V' = A, with names that meet
    assert (U @ s.diag() @ V.T).rows == A.rows and (U @ s.diag() @ V.T).cols == A.cols
    assert _close(s.values[:, 0], np.linalg.svd(A.values, compute_uv=False))
    Q, R = A.qr()
    assert Q.rows == A.rows and R.cols == A.cols and Q.cols == R.rows == ("q1", "q2", "q3")
    assert _close((Q @ R).values, A.values) and _close((Q.T @ Q).values, np.eye(3)) and np.allclose(np.tril(R.values, -1), 0)
    Qn, Rn = np.linalg.qr(A.values)
    assert _close(Q.values, Qn) and _close(R.values, Rn)
    S = _random(4, 4, 2, rows=list("abcd"), cols=list("wxyz"))
    P, L, Uu = S.lu()
    assert _close((P @ L @ Uu).values, S.values) and (P @ L @ Uu).rows == S.rows and (P @ L @ Uu).cols == S.cols
    assert sorted(P.cols) == sorted(S.rows) and L.rows == P.cols  # L's rows are A's rows in pivot order
    Ps, Ls, Us = linalg.lu(S.values)
    assert _close(P.values, Ps) and _close(L.values, Ls) and _close(Uu.values, Us)
    assert np.allclose(np.triu(L.values, 1), 0) and np.allclose(np.diag(L.values), 1) and np.allclose(np.tril(Uu.values, -1), 0)
    T, Qs = S.schur()
    assert _close((Qs @ T @ Qs.T).values, S.values) and Qs.rows == S.rows and T.rows == T.cols == ("u1", "u2", "u3", "u4")
    assert _close(T.values, linalg.schur(S.values)[0])
    Tc, Qc = S.schur(complex=True)
    assert Tc.is_complex and _close((Qc.values @ Tc.values @ Qc.values.conj().T), S.values) and np.allclose(np.tril(Tc.values, -1), 0)
    B = _random(4, 4, 3, rows=list("abcd"), cols=list("wxyz"))
    AA, BB, Qz, Z = S.qz(B)
    assert _close((Qz @ AA @ Z.T).values, S.values) and _close((Qz @ BB @ Z.T).values, B.values)
    assert (Qz @ AA @ Z.T).rows == S.rows and (Qz @ AA @ Z.T).cols == S.cols
    assert _close(AA.values, linalg.qz(S.values, B.values)[0]) and np.allclose(np.tril(BB.values, -1), 0)
    with pytest.raises(SpecificationError, match="columns do not match"):
        S.qz(B.named(cols=list("pqrs")))
    with pytest.raises(SpecificationError, match="square"):
        A.schur()


def test_general_eigenvalues_are_complex_and_symmetric_ones_unchanged():
    A = Matrix([[0.5, -0.8, 0.1], [0.9, 0.3, 0.0], [0.0, 0.2, 0.7]], list("abc"), list("abc"))
    values, vectors = A.eig()
    assert values.is_complex and vectors.is_complex and values.rows == ("λ1", "λ2", "λ3") and vectors.rows == A.rows
    expected = np.linalg.eigvals(A.values)
    assert _close(np.sort_complex(values.values[:, 0]), np.sort_complex(expected))
    for j in range(3):
        v = vectors.values[:, j]
        assert _close(A.values @ v, values.values[j, 0] * v, 1e-12)  # A v = λ v
    assert np.all(np.diff(np.abs(values.values[:, 0])) <= 1e-15)  # largest first in size
    assert "+" in values.table().to_markdown() or "-" in values.table().to_markdown()  # shown as text, e.g. 0.4+0.84j
    rng = np.random.default_rng(4)
    G = rng.normal(size=(4, 4))
    S = Matrix(G @ G.T, list("abcd"), list("abcd"))
    old = S.eig()
    w, v = np.linalg.eigh(S.values)
    order = np.argsort(-np.abs(w), kind="stable")
    assert not old.values.is_complex and _close(old.values.values[:, 0], w[order]) and _close(old.vectors.values, v[:, order])
    assert _close(S.eigh().values.values, old.values.values) and _close(S.eig(symmetric=True).vectors.values, old.vectors.values)
    general = S.eig(symmetric=False)  # the general method: real, the same values
    assert not general.values.is_complex and _close(general.values.values, old.values.values, 1e-10)
    R = Matrix([[0.9, 0.2], [0.1, 0.5]])  # non-symmetric with real eigenvalues: real, as before
    assert not R.eig().values.is_complex and _close(np.sort(R.eig().values.values[:, 0]), np.sort(np.linalg.eigvals(R.values).real))
    with pytest.raises(SpecificationError, match="symmetric=True needs a symmetric matrix"):
        A.eigh()


def test_functions_of_a_matrix():
    A = Matrix([[0.2, 0.1, 0.0], [0.05, 0.3, 0.1], [0.0, 0.2, 0.1]], list("abc"), list("abc"))
    E = A.expm()
    series = sum(np.linalg.matrix_power(A.values, k) / float(np.prod(range(1, k + 1))) for k in range(30))
    assert _close(E.values, series) and _close(E.values, linalg.expm(A.values)) and E.rows == A.rows and E.cols == A.cols
    assert _close(E.logm().values, A.values, 1e-11)  # logm(expm(A)) = A for a small A
    S = Matrix(A.values @ A.values.T + np.eye(3), list("abc"), list("abc"))
    root = S.sqrtm()
    assert not root.is_complex and _close((root @ root).values, S.values) and _close(root.values, linalg.sqrtm(S.values))
    assert _close(S.logm().expm().values, S.values) and _close(S.logm().values, linalg.logm(S.values))
    negative = Matrix([[-1.0, 0.0], [0.0, 4.0]])
    assert negative.sqrtm().is_complex and _close((negative.sqrtm().values @ negative.sqrtm().values), negative.values)
    M = _random(4, 3, 5)
    for kind, ref in (("fro", "fro"), ("nuc", "nuc"), (1, 1), (2, 2), ("inf", np.inf), (-2, -2)):
        assert M.norm(kind) == pytest.approx(np.linalg.norm(M.values, ref), rel=TOL)
    assert S.cond() == pytest.approx(np.linalg.cond(S.values), rel=1e-12) and S.cond(1) == pytest.approx(np.linalg.cond(S.values, 1), rel=1e-12)
    with pytest.raises(SpecificationError, match="no norm"):
        M.norm("max")


def test_null_space_orth_and_perp():
    rng = np.random.default_rng(6)
    beta = Matrix(rng.normal(size=(4, 2)), ["y", "c", "i", "r"], ["ec1", "ec2"])
    perp = beta.perp()
    assert perp.shape == (4, 2) and perp.rows == beta.rows and perp.cols == ("perp1", "perp2")
    assert _close((perp.T @ beta).values, np.zeros((2, 2))) and (perp.T @ beta).cols == beta.cols  # perp' beta = 0
    assert _close((perp.T @ perp).values, np.eye(2))  # normalised to orthonormal columns
    assert Matrix.hcat(beta, perp).rank() == 4  # [beta perp] of full rank
    assert _close(np.abs((perp @ perp.T).values), np.abs(linalg.null_space(beta.values.T) @ linalg.null_space(beta.values.T).T))
    # Johansen's long-run impact beta_perp (alpha_perp' Gamma beta_perp)^-1 alpha_perp' does not depend on the complement
    alpha = Matrix(rng.normal(size=(4, 2)), beta.rows, ["ec1", "ec2"])
    Gamma = Matrix(np.eye(4) - 0.2 * rng.normal(size=(4, 4)), beta.rows, beta.rows)
    ap, bp = alpha.perp(), beta.perp()
    C = bp @ (ap.T @ Gamma @ bp).inv() @ ap.T
    H = Matrix(rng.normal(size=(2, 2)) + 2 * np.eye(2), ap.cols, ["h1", "h2"])  # another complement: perp times an invertible matrix
    ap2, bp2 = ap @ H, bp @ H.named(rows=bp.cols)
    C2 = bp2 @ (ap2.T @ Gamma @ bp2).inv() @ ap2.T
    assert _close(C2.values, C.values, 1e-10) and C.rows == C.cols == beta.rows
    assert _close((beta.T @ C).values, 0, 1e-12) and _close((C @ alpha).values, 0, 1e-12)  # beta' C = 0 and C alpha = 0
    M = Matrix(np.outer([1.0, 2.0, 3.0], [1.0, -1.0]), list("abc"), ["x", "y"])
    N = M.null()
    assert N.rows == M.cols and N.cols == ("null1",) and _close((M @ N).values, 0) and _close(N.values, linalg.null_space(M.values))
    O = M.orth()
    assert O.rows == M.rows and O.shape == (3, 1) and _close(np.abs(O.values), np.abs(linalg.orth(M.values)))
    assert Matrix.identity(3).perp().shape == (3, 0)  # a square matrix of full rank has no complement


def test_vech_commutation_duplication_and_elimination():
    A = _random(3, 4, 7, rows=["a", "b", "c"], cols=["w", "x", "y", "z"])
    K = Matrix.commutation(3, 4)
    assert _close((K @ A.vec()).values, A.T.vec().values)  # K vec(A) = vec(A')
    Kn = Matrix.commutation(A.rows, A.cols)
    assert (Kn @ A.vec()).rows == A.T.vec().rows and _close((Kn @ A.vec()).values, A.T.vec().values)
    assert _close((Matrix.commutation(4, 3) @ K).values, np.eye(12))  # K_nm K_mn = I
    G = np.random.default_rng(8).normal(size=(3, 3))
    S = Matrix(G + G.T, ["p", "q", "r"], ["p", "q", "r"])
    v = S.vech()
    assert v.rows == ("p,p", "q,p", "r,p", "q,q", "r,q", "r,r") and list(v.values[:, 0]) == [S.values[i, j] for j in range(3) for i in range(j, 3)]
    D = Matrix.duplication(3)
    assert D.shape == (9, 6) and _close((D @ v).values, S.vec().values)  # D vech(A) = vec(A)
    Dn = Matrix.duplication(S.rows)
    assert (Dn @ v).rows == S.vec().rows
    L = Matrix.elimination(S.rows)
    assert _close((L @ A[:, :3].named(rows=S.rows, cols=S.rows).vec()).values, A[:, :3].named(rows=S.rows, cols=S.rows).vech().values)
    assert _close((L @ Dn).values, np.eye(6))  # L D = I
    back = Matrix.unvech(v)
    assert back.rows == back.cols == S.rows and _close(back.values, S.values)
    assert Matrix.unvech([1.0, 2.0, 3.0], ["u", "v"]).values.tolist() == [[1.0, 2.0], [2.0, 3.0]]
    with pytest.raises(SpecificationError, match="n\\(n\\+1\\)/2"):
        Matrix.unvech([1.0, 2.0])


def test_lyapunov_is_a_vars_unconditional_covariance(macro):
    data = macro.with_series(gy="400*dlog(y)", gc="400*dlog(c)")
    var = VAR(["gy", "gc"], lags=2).fit(data)
    A1, A2, Sigma = Matrix.var_lag(var, 1), Matrix.var_lag(var, 2), Matrix.var_sigma(var)
    lagged = Matrix.identity(["gy[-1]", "gc[-1]"])
    F = Matrix.vcat(Matrix.hcat(A1, A2), Matrix.hcat(lagged, Matrix.zeros(lagged.rows, A2.cols)))  # the companion form
    assert F.rows == ("gy", "gc", "gy[-1]", "gc[-1]") and F.cols == ("gy[-1]", "gc[-1]", "gy[-2]", "gc[-2]")
    Q = Matrix.zeros(F.rows, F.rows).set(["gy", "gc"], ["gy", "gc"], Sigma)
    X = Matrix.lyapunov(F, Q)
    assert X.rows == X.cols == F.rows
    vecS = np.linalg.solve(np.eye(16) - np.kron(F.values, F.values), Q.values.reshape(-1, order="F"))  # (I - F⊗F)^-1 vec(Q)
    assert _close(X.values, vecS.reshape(4, 4, order="F"), 1e-10)
    assert _close(X.values, F.values @ X.values @ F.values.T + Q.values, 1e-10)
    assert _close(X[["gy", "gc"], ["gy[-1]", "gc[-1]"]].values, (F.named(cols=X.rows) @ X).values[:2, :2], 1e-10)  # the first autocovariance, F Σ (F's columns are the lagged state)
    C = Matrix([[-1.0, 0.3], [0.2, -0.5]], ["a", "b"], ["a", "b"])
    Qc = Matrix([[1.0, 0.2], [0.2, 0.5]], ["a", "b"], ["a", "b"])
    Xc = Matrix.lyapunov(C, Qc, continuous=True)  # A X + X A' + Q = 0
    vecC = np.linalg.solve(np.kron(np.eye(2), C.values) + np.kron(C.values, np.eye(2)), -Qc.values.reshape(-1, order="F"))
    assert _close(Xc.values, vecC.reshape(2, 2, order="F"))
    with pytest.raises(SpecificationError, match="A's rows and Q's rows do not match"):
        Matrix.lyapunov(F, Q.named(rows=["a", "b", "c", "d"]))


def test_sylvester_and_riccati():
    A = _random(3, 3, 9, rows=list("abc"), cols=list("abc"))
    B = _random(2, 2, 10, rows=["p", "q"], cols=["p", "q"])
    C = _random(3, 2, 11, rows=list("abc"), cols=["p", "q"])
    X = Matrix.sylvester(A, B, C)
    assert X.rows == C.rows and X.cols == C.cols
    assert _close((A @ X + X @ B).values, C.values, 1e-11) and _close(X.values, linalg.solve_sylvester(A.values, B.values, C.values))
    vecX = np.linalg.solve(np.kron(np.eye(2), A.values) + np.kron(B.values.T, np.eye(3)), C.values.reshape(-1, order="F"))
    assert _close(X.values, vecX.reshape(3, 2, order="F"), 1e-10)
    with pytest.raises(SpecificationError, match="C's columns and B's columns"):
        Matrix.sylvester(A, B, C.named(cols=["e", "f"]))
    # the Kalman filter's steady-state prediction covariance: iterate P = T P T' - T P Z' (Z P Z' + H)^-1 Z P T' + RQR'
    T = np.array([[0.9, 0.1, 0.0], [0.0, 0.7, 0.2], [0.1, 0.0, 0.5]])
    Z = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]])
    RQR, H = np.diag([0.5, 0.3, 0.2]), np.diag([0.4, 0.6])
    P = np.eye(3)
    for _ in range(3000):
        P = T @ P @ T.T - T @ P @ Z.T @ np.linalg.solve(Z @ P @ Z.T + H, Z @ P @ T.T) + RQR
    states, obs = ["level", "slope", "cycle"], ["y1", "y2"]
    Xr = Matrix.riccati(Matrix(T.T, states, states), Matrix(Z.T, states, obs), Matrix(RQR, states, states), Matrix(H, obs, obs))
    assert Xr.rows == Xr.cols == tuple(states) and _close(Xr.values, P, 1e-10)
    assert _close(Xr.values, linalg.solve_discrete_are(T.T, Z.T, RQR, H))
    with pytest.raises(SpecificationError, match="B's columns and R's rows"):
        Matrix.riccati(Matrix(T.T, states, states), Matrix(Z.T, states, obs), Matrix(RQR, states, states), Matrix(H, ["e1", "e2"], ["e1", "e2"]))


def test_assembly_setting_and_element_functions(macro):
    A = Matrix([[1.0, 2.0], [3.0, 4.0]], ["a", "b"], ["x", "y"])
    B = Matrix([[5.0], [6.0]], ["a", "b"], ["z"])
    H = Matrix.hcat(A, B)
    assert H.cols == ("x", "y", "z") and H.rows == A.rows and H.values.tolist() == [[1, 2, 5], [3, 4, 6]]
    V = Matrix.vcat(A, Matrix([[7.0, 8.0]], ["c"], ["x", "y"]))
    assert V.rows == ("a", "b", "c") and V.values[-1].tolist() == [7.0, 8.0]
    assert Matrix.hcat(np.ones((2, 1)), np.zeros((2, 2))).cols == ("1", "2", "3")  # unnamed stays unnamed
    with pytest.raises(SpecificationError, match="column names clash: 'x'"):
        Matrix.hcat(A, A)
    with pytest.raises(SpecificationError, match="rows do not match"):
        Matrix.hcat(A, B.named(rows=["b", "a"]))
    S = A.set("b", "y", 0.0)
    assert S["b", "y"] == 0.0 and A["b", "y"] == 4.0 and S.rows == A.rows  # a new matrix; the old one unchanged
    blk = Matrix.zeros(["a", "b", "c"], ["p", "q"]).set(["a", "b"], slice(None), A.named(cols=["p", "q"]))
    assert blk.values.tolist() == [[1, 2], [3, 4], [0, 0]]
    with pytest.raises(SpecificationError, match="block's rows"):
        Matrix.zeros(["a", "b", "c"], ["p", "q"]).set(["b", "a"], slice(None), A.named(cols=["p", "q"]))
    for fn in (np.log, np.exp, np.sqrt, np.abs):
        out = fn(A)
        assert isinstance(out, Matrix) and out.rows == A.rows and out.cols == A.cols and _close(out.values, fn(A.values))
    assert _close(A.log().values, np.log(A.values)) and _close(abs(-A).values, A.values) and _close(A.exp().values, np.exp(A.values))
    left = np.ones((2, 2)) + A  # an array on the left keeps the names now
    assert isinstance(left, Matrix) and left.rows == A.rows and _close(left.values, 1 + A.values)
    assert _close((1 / A).values, 1 / A.values) and _close((10 - A).values, 10 - A.values)
    assert isinstance(np.maximum(A, 2.5), Matrix) and np.maximum(A, 2.5).values.tolist() == [[2.5, 2.5], [3, 4]]
    with pytest.raises(SpecificationError, match="do not match"):
        np.add(A, A.named(rows=["b", "a"]))
    assert A.sum() == 10.0 and np.sum(A) == 10.0 and A.mean() == 2.5
    assert A.sum(axis=0).rows == ("sum",) and A.sum(axis=0).values.tolist() == [[4.0, 6.0]]
    assert A.mean(axis=1).cols == ("mean",) and A.mean(axis=1).rows == A.rows and A.mean(axis=1).values[:, 0].tolist() == [1.5, 3.5]
    a, b = Matrix([1.0, 2.0], ["p", "q"]), Matrix([3.0, 4.0, 5.0], ["u", "v", "w"])
    O = a.outer(b)
    assert O.rows == ("p", "q") and O.cols == ("u", "v", "w") and _close(O.values, np.outer([1, 2], [3, 4, 5]))
    assert a.inner(Matrix([3.0, 4.0], ["p", "q"])) == 11.0 and A.inner(A) == pytest.approx(np.sum(A.values ** 2))
    with pytest.raises(SpecificationError, match="names do not match"):
        a.inner(Matrix([3.0, 4.0], ["q", "p"]))
    m = Matrix.from_dataset(macro, ["c", "y"], "all")
    assert m.cols == ("c", "y") and m.rows[0] == "2000Q1" and _close(m.values, macro.frame[["c", "y"]].to_numpy())
    back = (m * 2).to_dataset()
    assert list(back.frame.columns) == ["c", "y"] and _close(back.frame["y"].to_numpy(), 2 * macro.frame["y"].to_numpy())
    assert str(back.frame.index[0]) == "2000Q1"
    with pytest.raises(SpecificationError, match="not periods"):
        A.to_dataset()


def test_complex_results_show_as_tables():
    from pymetrix.views import view

    values = Matrix([[0.0, -1.0], [1.0, 0.0]]).eig().values
    table = view(values, "Matrix")
    assert table.to_markdown().count("j") == 2
    assert Matrix([[1.0, 2.0], [3.0, 4.0]]).svd().U.to_frame("Matrix").shape == (2, 2)


def test_matrix_power_and_the_deprecated_double_star():
    A = Matrix([[0.5, 0.2, 0.1], [0.3, 0.9, 0.0], [0.2, 0.1, 0.7]], list("abc"), ["x", "y", "z"])
    for n in (0, 1, 2, 5):
        assert _close(A.power(n).values, np.linalg.matrix_power(A.values, n)) and A.power(n).rows == A.rows
    assert _close(A.power(-2).values, (A.inv() @ A.inv().named(rows=A.rows, cols=A.cols)).values)  # A^-2 = (A^-1)^2
    assert _close(A.power(-1).values, A.inv().values) and A.power(-1).rows == A.cols and A.power(-1).cols == A.rows
    assert _close(A.matrix_power(3).values, A.power(3).values)
    with pytest.warns(DeprecationWarning, match=r"A.power\(n\)"):
        old = A ** 2
    assert _close(old.values, A.values @ A.values)  # the result is unchanged for now
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning: element by element already
        assert _close((A ** 0.5).values, A.values ** 0.5)
        rect = Matrix([[1.0, 2.0, 3.0]])
        assert _close((rect ** 2).values, [[1.0, 4.0, 9.0]])
        assert _close(np.square(A).values, A.values ** 2) and _close((A * A).values, A.values ** 2)
    with pytest.raises(SpecificationError, match="whole number"):
        A.power(1.5)
    with pytest.raises(SpecificationError, match="no negative powers"):
        Matrix([[1.0, 2.0], [2.0, 4.0]]).power(-1)


def test_upper_cholesky_factor():
    G = np.random.default_rng(12).normal(size=(4, 4))
    S = Matrix(G @ G.T + np.eye(4), list("abcd"), list("abcd"))
    R = S.chol(upper=True)
    assert R.rows == S.rows and R.cols == S.cols and np.allclose(np.tril(R.values, -1), 0)
    assert _close(R.values, linalg.cholesky(S.values, lower=False)) and _close((R.T @ R).values, S.values)
    assert _close(R.values, S.chol().values.T)
