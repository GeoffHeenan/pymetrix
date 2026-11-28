"""Krusell-Smith, as sequence-jacobian's Krusell-Smith notebook solves it: the plugin's steady state, Jacobians,
impulse responses and nonlinear transition checked against

- the numbers the notebook prints (tests/fixtures/krusell_smith_notebook.json, sequence-jacobian 1.0.0);
- sequence-jacobian called directly, with its own defaults (the wrapper adds nothing numerically);
- hand formulas: the steady state's capital and wage, the distribution's mass and sums, Walras' law, the stationary
  distribution as an eigenvector of Young's transition matrix, the notebook's step-by-step chain rule for G, and
  the nonlinear transition converging to the linear one as the shock shrinks.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymetrix.models.solution import Solution
from pymetrix.var import IRF
from pymetrix_ha import AR1, HAOptions, HATransition
from pymetrix_ha.examples import _krusell_smith as nb
from pymetrix_ha.examples import krusell_smith

NOTEBOOK = json.loads((Path(__file__).parent / "fixtures" / "krusell_smith_notebook.json").read_text(encoding="utf-8"))
CELLS = NOTEBOOK["cells"]
T = 300


@pytest.fixture(scope="module")
def ks():
    return krusell_smith()


@pytest.fixture(scope="module")
def ss(ks):
    return ks.steady_state()


@pytest.fixture(scope="module")
def lin(ks, ss):
    return ks.linearize(ss)


@pytest.fixture(scope="module")
def path(ks, ss):
    return ks.solve(steady=ss)


@pytest.fixture(scope="module")
def direct():
    """The notebook's own calls, with the library's defaults (cells 13, 15, 30, 49 and 82)."""
    import sequence_jacobian as sj

    model = sj.create_model([nb.household_ext, nb.firm, nb.mkt_clearing], name="Krusell-Smith")
    ss = model.solve_steady_state(NOTEBOOK["calibration"], {"beta": 0.98, "Z": 0.85, "K": 3.0},
                                  {"r": 0.01, "Y": 1.0, "asset_mkt": 0.0}, solver="hybr")
    G = model.solve_jacobian(ss, ["K"], ["asset_mkt"], ["Z"], T=T)
    shock = {"Z": 0.01 * 0.8 ** np.arange(T)}
    nonlin = model.solve_impulse_nonlinear(ss, ["K"], ["asset_mkt"], shock, verbose=False)
    linear = model.solve_impulse_linear(ss, ["K"], ["asset_mkt"], shock)
    return {"model": model, "ss": ss, "G": G, "nonlin": nonlin, "lin": linear}


def _printed(x, digits):
    return np.round(np.asarray(x, dtype=float), digits)


# the notebook's printed numbers -------------------------------------------------------------------------------------

def test_the_household_jacobian_is_the_notebooks_to_every_printed_digit(ks, ss):
    """Cell 30: the 5 x 5 Jacobian of consumption on the interest rate, by the fake-news algorithm, with the
    notebook's grid (grids.agrid)."""
    J = ks.jacobian("household", ["r", "w"], T=5, steady=ss)
    cell = CELLS["J_ha_C_r_T5"]
    assert np.array_equal(_printed(J["C", "r"], cell["digits"]), np.array(cell["printed"]))
    assert J["C", "r"].shape == (5, 5) and J.T == 5
    assert J.table("C", "r").to_frame().shape == (5, 5)


def test_the_firms_jacobian_and_their_product_are_the_notebooks(ks, ss):
    """Cells 27 and 32: dY/dZ on the diagonal, and the household's J['C']['r'] times the firm's J['r']['Z']."""
    firm = ks.jacobian("firm", ["K", "Z"], T=5, steady=ss)
    cell = CELLS["J_firm_Y_Z_diagonal"]
    assert np.array_equal(_printed(np.diag(firm["Y", "Z"]), cell["digits"]), np.full(5, cell["printed"]))
    assert np.count_nonzero(firm["Y", "Z"] - np.diag(np.diag(firm["Y", "Z"]))) == 0
    household = ks.jacobian("household", ["r", "w"], T=5, steady=ss)
    cell = CELLS["J_ha_C_r_times_J_firm_r_Z_T5"]
    assert np.array_equal(_printed(household["C", "r"] @ firm["r", "Z"], cell["digits"]), np.array(cell["printed"]))


def test_the_steady_state_has_the_notebooks_aggregates(ss):
    """Cell 15 prints the steady state's keys (not its values)."""
    assert list(ss.values) == CELLS["steady_state_keys"]["printed"]


@pytest.mark.parametrize("size, cell", [(0.01, "nonlinear_errors_Z_1pct_rho_0.8"), (0.1, "nonlinear_errors_Z_10pct_rho_0.8")])
def test_the_newton_iterations_errors_are_the_notebooks(ks, ss, size, cell):
    """Cells 82 and 86: the largest error in asset-market clearing at each Newton iteration of the transition, as the
    library prints it (three significant digits): recorded in provenance instead of printed."""
    out = ks.solve({"Z": AR1(0.8, size)}, steady=ss)
    printed = [f"{e['asset_mkt']:.2E}" for e in out.errors]
    assert printed == CELLS[cell]["printed"]
    assert out.iterations == len(printed) - 1
    assert [r["asset_mkt"] for r in out.provenance.resolved["iterations"]] == [e["asset_mkt"] for e in out.errors]


# sequence-jacobian called directly -----------------------------------------------------------------------------------

def test_the_steady_state_is_the_librarys_own(ss, direct):
    raw = direct["ss"]
    for k in raw.toplevel:
        assert ss[k] == pytest.approx(float(raw[k]), abs=1e-12, rel=0), k
    for k in ("D", "a", "c", "Va"):
        np.testing.assert_allclose(ss.internals("household")[k], raw.internals["household"][k], rtol=0, atol=1e-12)


def test_g_and_the_impulse_responses_are_the_librarys_own(lin, direct):
    G = direct["G"]
    assert set(lin.outputs) == {str(o) for o in G.outputs}
    for o in lin.outputs:
        np.testing.assert_allclose(lin.jacobian(o, "Z"), np.asarray(G[o]["Z"]), rtol=0, atol=1e-12, err_msg=o)
    irf = lin.irf(40)
    dZ = 0.01 * 0.8 ** np.arange(T)
    for o in ("K", "C", "r", "w", "Y"):
        np.testing.assert_allclose(irf[o, "Z"].to_numpy(), (np.asarray(G[o]["Z"]) @ dZ)[:41], rtol=0, atol=1e-12)
        np.testing.assert_allclose(irf[o, "Z"].to_numpy(), direct["lin"][o][:41], rtol=0, atol=1e-12)


def test_the_transitions_are_the_librarys_own(ss, path, direct):
    dev = path.deviations()
    for o in ("K", "C", "r", "w", "Y", "A", "asset_mkt"):
        np.testing.assert_allclose(dev[o].to_numpy(), direct["nonlin"][o], rtol=0, atol=1e-12, err_msg=o)
        np.testing.assert_allclose(path.frame[o].to_numpy(), ss[o] + direct["nonlin"][o], rtol=0, atol=1e-12)
        np.testing.assert_allclose(path.linear.deviations()[o].to_numpy(), direct["lin"][o], rtol=0, atol=1e-12)


# hand checks -----------------------------------------------------------------------------------------------------------

def test_capital_wage_and_tfp_by_hand(ss, ks):
    """K = alpha Y / (r + delta), w = (1 - alpha) Y / L, Z = Y / (K^alpha L^(1-alpha)); and the firm's dY/dZ is
    K^alpha, the notebook's 1.13424196."""
    alpha, delta, Y, r, L = 0.11, 0.025, 1.0, 0.01, 1.0
    K = alpha * Y / (r + delta)
    assert ss["K"] == pytest.approx(K, rel=1e-12) and K == pytest.approx(3.142857142857143, rel=1e-15)
    assert ss["w"] == pytest.approx((1 - alpha) * Y / L, rel=1e-12)
    assert ss["Z"] == pytest.approx(Y / (K ** alpha * L ** (1 - alpha)), rel=1e-12)
    assert K ** alpha == pytest.approx(CELLS["J_firm_Y_Z_diagonal"]["printed"], abs=5e-9)


def test_the_distribution_by_hand(ss):
    """Mass 1, no negative entries, the aggregates the distribution's sums of the policies, and the untargeted goods
    market clear by Walras' law."""
    inner = ss.internals("household")
    D = inner["D"]
    assert D.sum() == pytest.approx(1.0, abs=1e-12) and D.min() >= 0
    assert ss["A"] == pytest.approx(float(np.vdot(D, inner["a"])), abs=1e-12)
    assert ss["C"] == pytest.approx(float(np.vdot(D, inner["c"])), abs=1e-12)
    assert abs(ss["goods_mkt"]) < 1e-8  # Y - C - delta K: not a target
    checks = ss.checks().to_frame()["value"]
    assert abs(checks["household: mass less 1"]) < 1e-12 and abs(checks["household: A less sum of D * a"]) < 1e-12


def test_the_stationary_distribution_is_the_eigenvector_of_youngs_matrix(ss):
    """Young's method by hand: each agent's saving split between the two grid points around it in proportion to the
    distance (a lottery), then moved by the income process's Markov matrix. The library's distribution is this map's
    eigenvector for eigenvalue 1, found here by SciPy's sparse eigensolver (1e-8)."""
    inner = ss.internals("household")
    grid, Pi, policy, D = inner["a_grid"], inner["Pi"], inner["a"], inner["D"]
    nE, nA = policy.shape
    j = np.clip(np.searchsorted(grid, policy, side="right") - 1, 0, nA - 2)
    pi = np.clip((grid[j + 1] - policy) / (grid[j + 1] - grid[j]), 0, 1)
    rows, cols, vals = [], [], []
    for e in range(nE):
        src = e * nA + np.arange(nA)
        for e2 in range(nE):
            if Pi[e, e2] == 0:
                continue
            for dest, w in ((j[e], pi[e]), (j[e] + 1, 1 - pi[e])):
                rows.append(e2 * nA + dest)
                cols.append(src)
                vals.append(Pi[e, e2] * w)
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nE * nA, nE * nA))
    value, vector = spla.eigs(M, k=1, sigma=1.0 + 1e-9)
    assert abs(value[0] - 1) < 1e-10
    stationary = np.real(vector[:, 0])
    stationary = (stationary / stationary.sum()).reshape(nE, nA)
    np.testing.assert_allclose(stationary, D, rtol=0, atol=1e-8)


def test_g_by_the_notebooks_chain_rule(ks, ss, lin):
    """The notebook's section 5.1 by hand: the household's and firm's Jacobians chained into dK = -H_K^-1 H_Z dZ,
    then r, w, Y and C from K; the same G as the library's own composition (its cell 51 asserts np.allclose)."""
    firm = ks.jacobian("firm", ["K", "Z"], T=T, steady=ss)
    hh = ks.jacobian("household", ["r", "w"], T=T, steady=ss)
    curlyK_K = hh["A", "r"] @ firm["r", "K"] + hh["A", "w"] @ firm["w", "K"]
    curlyK_Z = hh["A", "r"] @ firm["r", "Z"] + hh["A", "w"] @ firm["w", "Z"]
    G = {"K": -np.linalg.solve(curlyK_K - np.eye(T), curlyK_Z)}
    for o in ("r", "w", "Y"):
        G[o] = firm[o, "Z"] + firm[o, "K"] @ G["K"]
    G["C"] = hh["C", "r"] @ G["r"] + hh["C", "w"] @ G["w"]
    for o, m in G.items():
        np.testing.assert_allclose(lin.jacobian(o, "Z"), m, rtol=0, atol=1e-10, err_msg=o)


def test_a_small_shocks_transition_converges_to_the_linear_one(ks, ss, path):
    """The nonlinear and linear transitions differ at second order: a shock 10 times smaller leaves a gap about 100
    times smaller."""
    small = ks.solve({"Z": AR1(0.8, 0.001)}, steady=ss)
    gap = lambda p: np.max(np.abs(p.frame["r"].to_numpy() - p.linear.frame["r"].to_numpy()))  # noqa: E731
    ratio = gap(path) / gap(small)
    assert 70 < ratio < 130, ratio


# what the wrapper adds: pymetrix objects, provenance, silence ----------------------------------------------------------

def test_results_are_pymetrix_objects(ss, lin, path):
    table = ss.table().to_frame()
    assert table.loc["beta", "role"] == "unknown (solved)" and table.loc["r", "role"] == "target"
    assert table.loc["alpha", "role"] == "calibrated" and table.loc["C", "role"] == "computed"
    irf = lin.irf(20)
    assert isinstance(irf, IRF) and irf.values.shape == (21, len(irf.responses), 1)
    assert irf.responses[0] == "Z" and "asset_mkt" not in irf.responses and irf.shocks == ["Z"]
    assert irf["Z", "Z"].to_numpy() == pytest.approx(0.01 * 0.8 ** np.arange(21), abs=1e-15)
    assert isinstance(path, HATransition) and isinstance(path, Solution) and path.nonlinear and not path.linear.nonlinear
    assert path.span == ("2000Q1", "2074Q4") and len(path.frame) == T
    assert path.frame["K"].iloc[-1] == pytest.approx(ss["K"], abs=1e-6)  # back to the steady state


def test_provenance_records_the_grids_options_and_library(ss, lin, path):
    for result in (ss, lin, path):
        p = result.provenance
        assert p.backend["name"] == "sequence-jacobian" and p.backend["version"] == NOTEBOOK["package_version"]
        assert p.backend["numba"] and p.backend["numpy"] and p.method["plugin"] == "pymetrix-ha"
        assert p.seed is None  # nothing is drawn
        grid = p.spec["grids"]["household"]["a_grid"]
        assert grid["shape"] == [500] and grid["min"] == 0 and grid["max"] == 200 and len(grid["sha256"]) == 16
        assert p.spec["calibration"]["nA"] == 500 and p.spec["shocks"]["Z"] == {"process": "AR(1)", "rho": 0.8, "sd": 0.01}
        assert all(b["source"] for b in p.spec["blocks"])  # each block's source text fingerprinted
        assert p.resolved["options"] == {"solver": "", "ttol": 1e-12, "ctol": 1e-9, "backward_tol": 1e-8,
                                         "backward_maxit": 5000, "forward_tol": 1e-10, "forward_maxit": 100_000,
                                         "h": 1e-4, "twosided": False, "tol": 1e-8, "maxit": 30} | {"solver": "hybr"}
    assert ss.provenance.resolved["solver"] == "hybr" and ss.provenance.resolved["evaluations"] > 1
    het = ss.provenance.options["stages"]["solve_steady_state"]["options"]["household"]
    assert het == {"backward_tol": 1e-8, "backward_maxit": 5000, "forward_tol": 1e-10, "forward_maxit": 100_000}
    assert path.provenance.options["stages"]["solve_impulse_nonlinear"]["verbose"] is False
    assert path.provenance.sample == ("2000Q1", "2074Q4")


def test_the_library_prints_nothing(ks, ss, capsys):
    """The library prints each Newton iteration by default; the wrapper keeps it quiet."""
    ks.solve({"Z": AR1(0.8, 0.005)}, steady=ss)
    assert capsys.readouterr().out == ""


def test_the_options_reach_the_library():
    """A looser tolerance for the household's policies changes the steady state, so the options passed are used."""
    small = dict(nA=100)
    tight = krusell_smith(**small).steady_state()
    loose = krusell_smith(**small)
    loose.options = HAOptions(solver="hybr", backward_tol=1e-3)
    assert loose.steady_state()["beta"] != tight["beta"]
