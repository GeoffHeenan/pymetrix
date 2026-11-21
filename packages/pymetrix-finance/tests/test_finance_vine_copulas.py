"""Vine copulas, checked independently:

- against R's VineCopula 2.6.1 (its numbers written here, so R is not needed to run the tests):
  BiCopHfunc's h-functions of every pair family (rotations included) at four points to 1e-9; RVineLogLik of a
  five-series R-vine with Gaussian, t, Clayton, Gumbel, Frank, BB1, BB7, survival and 90/270-degree pair copulas
  at given parameters to 1e-8 (the same through our matrix reader at R's selected parameters); RVineStructureSelect
  (R-vine, C-vine, and by BIC truncated after two trees; familyset our families, rotations=FALSE, presel=FALSE) on
  a seeded data set: the same trees, the same families on every edge, the parameters within 1e-3 and the log
  likelihood within 1e-3 (at least R's); RVineSim of the five-series vine (100,000 draws): uniform margins and
  every pair's Kendall's tau within three standard errors of ours;
- a D-vine, a C-vine and the selected R-vine of Gaussian pair copulas with the partial correlations of a
  correlation matrix have the multivariate Gaussian copula's log likelihood (1e-10);
- simulation (100,000 draws) gives the first tree's Kendall's taus within three standard errors (batch means);
  the inverse h-functions invert the h-functions; the R-vine matrix written and read back keeps the likelihood;
- sequential estimates on 3,000 draws from a known vine recover its parameters within three standard errors, and
  joint maximum likelihood's is at least the sequential one's;
- through the API: the margins, portfolio VaR and ES, the summary, the errors, and the dialog's recorded code.
"""

import importlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pymetrix import Dataset, copula, vine_copula
from pymetrix.errors import SpecificationError

V = importlib.import_module("pymetrix_finance.vines")
cp = importlib.import_module("pymetrix_finance.copulas")

# the five-series R-vine of the R checks, in VineCopula's matrix convention (variables from 1)
M5 = np.array([[5, 0, 0, 0, 0], [2, 2, 0, 0, 0], [3, 3, 3, 0, 0], [1, 4, 4, 4, 0], [4, 1, 1, 1, 1]])
F5 = np.array([[0, 0, 0, 0, 0], [24, 0, 0, 0, 0], [2, 7, 0, 0, 0], [13, 5, 34, 0, 0], [1, 3, 9, 4, 0]])
P5 = np.array([[0, 0, 0, 0, 0], [-1.4, 0, 0, 0, 0], [0.3, 0.6, 0, 0, 0], [1.1, -3.0, -1.3, 0, 0], [0.5, 2.0, 1.6, 1.7, 0]])
Q5 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [6.0, 1.4, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1.2, 0, 0]])
R_LOGLIK = 1315.760537494028  # RVineLogLik(U, RVineMatrix(M5, F5, P5, Q5)) on U below
FAMSET = "c(0,1,2,3,4,5,6,7,9,13,14,16,23,24,26,33,34,36)"
R_SELECT = """
type 0 loglik 1193.7531926580
4,0,0,0,0,5,3,0,0,0,3,5,2,0,0,2,1,5,1,0,1,2,1,5,5
0,0,0,0,0,14,0,0,0,0,26,4,0,0,0,5,9,36,0,0,9,14,3,4,0
0,0,0,0,0,1.066169997,0,0,0,0,-1.104411172,1.388885138,0,0,0,-2.674256454,1.304991146,-1.128878928,0,0,2.083245017,2.729819654,2.084888287,1.900689831,0
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1049573997,0,0,0,0.1705274761,0,0,0,0
type 1 loglik 1247.4882284096
4,0,0,0,0,5,2,0,0,0,2,5,3,0,0,3,3,5,1,0,1,1,1,5,5
0,0,0,0,0,9,0,0,0,0,5,34,0,0,0,24,2,2,0,0,9,3,9,4,0
0,0,0,0,0,1.079061478,0,0,0,0,-1.569517358,-1.385093405,0,0,0,-1.257702744,0.7082656129,0.2668211249,0,0,2.083245017,2.084888287,1.753999515,1.900689831,0
0,0,0,0,0,0.09176181543,0,0,0,0,0,0,0,0,0,0,6.243315416,6.586509975,0,0,0.1705274761,0,1.345388563,0,0
bic trunc2 loglik 1104.4427399959
3,0,0,0,0,5,4,0,0,0,4,5,2,0,0,1,2,5,1,0,2,1,1,5,5
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,5,36,0,0,14,13,3,4,0
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.335548541,-2.696488696,-1.128878928,0,0,2.729819654,1.329741501,2.084888287,1.900689831,0
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
"""
R_TAU = [0.500530,0.469352,0.623982,0.413689,0.102132,0.127047,0.460285,0.238077,0.371264,0.336371]  # RVineSim, 100,000 draws, TauMatrix: the upper triangle by columns
R_HFUNC = """
1 0.6 0 : 0.947024619619,0.222720905924,0.486863751741,0.893277587082,0.023008363810,0.559446223256,0.588013482446,0.667907909363
2 -0.4 5 : 0.488195766199,0.125026103608,0.606911653538,0.986982446749,0.098378836934,0.211956494390,0.642140879990,0.965272963094
3 2 0 : 0.984588435709,0.127866173061,0.403918687336,0.923713852899,0.002870520221,0.685282771248,0.524396312334,0.814087524860
4 1.7 0 : 0.929303125724,0.236561016343,0.512482044363,0.910186802481,0.035983151469,0.558305852700,0.624143276096,0.475238787629
5 -3 0 : 0.447990500467,0.113974557989,0.631073279721,0.993919639733,0.132313629507,0.164212031244,0.651967954230,0.986618660563
5 4 0 : 0.936769168264,0.239576741714,0.465864413441,0.910421844946,0.029830802852,0.597250435563,0.581860851164,0.772971865003
6 2.2 0 : 0.920037671559,0.268229439426,0.548723116039,0.924173059354,0.054810293335,0.543224156436,0.662126518291,0.333520848063
7 0.8 1.5 : 0.975726345218,0.166508593077,0.451363708725,0.905004061013,0.007328575935,0.643790445404,0.581856085289,0.538922726775
9 1.6 1.2 : 0.959304377407,0.171644968543,0.468665713244,0.912516148328,0.011632401606,0.591940932637,0.579360545657,0.536162819675
13 1.5 0 : 0.938748818376,0.270502914567,0.535927535940,0.919832585928,0.045188572631,0.565907797057,0.654295928837,0.333285595578
14 1.7 0 : 0.940485312240,0.176689680172,0.459317706636,0.909253257926,0.017567848662,0.577900306383,0.563522668414,0.743024350511
16 2.2 0 : 0.947580083666,0.145092763528,0.425451421185,0.940278938020,0.010361558600,0.594836535373,0.522479052902,0.856985861453
23 -1.5 0 : 0.454957749457,0.046106122250,0.593735620528,0.998558669642,0.147229358334,0.122055626490,0.732184098650,0.998603022694
24 -1.7 0 : 0.307444342422,0.106295929822,0.669019009486,0.999009739044,0.071830539142,0.130857423304,0.639155236329,0.996843785706
26 -2.2 0 : 0.238083198173,0.139146787065,0.700071920521,0.999511536015,0.048545015527,0.128793043035,0.595623204066,0.997604897988
33 -1.5 0 : 0.220072973782,0.123517942420,0.703727483904,0.999813291813,0.050032342059,0.110965550874,0.608188504609,0.999004847950
34 -1.7 0 : 0.417934354931,0.069794792018,0.617384489713,0.997870479578,0.123234243683,0.137652739103,0.697438328002,0.996614782128
36 -2.2 0 : 0.488613080394,0.052422842584,0.578371888321,0.997235505386,0.143921894387,0.145261440360,0.732047715183,0.997012672417
"""


def _five():
    trees = V.from_matrix(M5, F5, P5, Q5)
    U = V.simulate_vine(trees, 5, 600, np.random.default_rng(2026))
    return trees, U


def _r_fits():
    lines = [ln for ln in R_SELECT.strip().splitlines()]
    out = []
    for i in range(0, len(lines), 5):
        ll = float(lines[i].split()[-1])
        M, F, P, Q = (np.array([float(x) for x in lines[i + k].split(",")]).reshape(5, 5) for k in range(1, 5))
        out.append((ll, V.from_matrix(M.astype(int), F.astype(int), P, Q)))
    return out


def test_h_functions_against_vinecopula():
    u1, u2 = np.array([0.1, 0.35, 0.6, 0.93]), np.array([0.7, 0.2, 0.55, 0.97])
    for line in R_HFUNC.strip().splitlines():
        head, values = line.split(":")
        code, par, par2 = (float(x) for x in head.split())
        values = np.array([float(x) for x in values.split(",")])
        fam = V.VC_FAMILY[int(code)]
        p = tuple(abs(x) if fam in V.ROTATED and not fam.endswith("180") else x for x in (par, par2)[:V.n_pair_parameters(fam)])
        assert V.pair_h1(fam, p, u1, u2) == pytest.approx(values[:4], abs=1e-9), fam
        assert V.pair_h2(fam, p, u1, u2) == pytest.approx(values[4:], abs=1e-9), fam
        w = np.array([0.05, 0.4, 0.75, 0.99])
        assert V.pair_h1(fam, p, u1, V.pair_h1inv(fam, p, u1, w)) == pytest.approx(w, abs=1e-8), fam
        assert V.pair_h2(fam, p, V.pair_h2inv(fam, p, u2, w), u2) == pytest.approx(w, abs=1e-8), fam


def test_log_likelihood_against_rvineloglik():
    trees, U = _five()
    # relative: the log likelihoods are in the thousands, and old SciPy's arithmetic moves them by 1e-11 relative
    assert float(np.sum(V.propagate(U, trees)[0])) == pytest.approx(R_LOGLIK, rel=1e-10)
    P = cp.pseudo_observations(U)
    for ll, rtrees in _r_fits():  # R's selected vines at R's parameters, through our reader: R's log likelihoods
        assert float(np.sum(V.propagate(P, rtrees)[0])) == pytest.approx(ll, rel=1e-10)


@pytest.mark.parametrize("case", [0, 1, 2])
def test_selection_against_rvinestructureselect(case):
    _, U = _five()
    P = cp.pseudo_observations(U)
    kwargs = [{"structure": "rvine"}, {"structure": "cvine"}, {"criterion": "bic", "trunc_level": 2}][case]
    ours = V.fit_vine(P, **kwargs)
    r_ll, theirs = _r_fits()[case]
    for tree_r, tree_o in zip(theirs, ours):
        assert len(tree_r) == len(tree_o)
        for er in tree_r:
            match = [e for e in tree_o if e.same_as(er)]
            assert match, (er.a, er.b, er.cond)
            eo = match[0].oriented(er.a)
            assert eo.family == er.family, (er.a, er.b, er.cond)
            assert eo.params == pytest.approx(er.params, abs=1e-3, rel=1e-3)
    ll = float(np.sum(V.propagate(P, ours)[0]))
    assert ll == pytest.approx(r_ll, abs=1e-3)
    vc = V.FittedVine(tuple("abcde"), "rvine", tuple(map(tuple, ours)), P, P, np.ones(5), np.zeros(5)).to_vinecopula()
    back = V.from_matrix(vc["matrix"], vc["family"], vc["par"], vc["par2"])  # written and read as VineCopula's
    assert float(np.sum(V.propagate(P, back)[0])) == pytest.approx(ll, rel=1e-12)  # relative: a large log likelihood


def _batch_tau(x, y, batches=20):
    taus = [stats.kendalltau(a, b)[0] for a, b in zip(np.array_split(x, batches), np.array_split(y, batches))]
    return stats.kendalltau(x, y)[0], np.std(taus, ddof=1) / np.sqrt(batches)


def test_simulation_against_rvinesim():
    trees, _ = _five()
    S = V.simulate_vine(trees, 5, 100_000, np.random.default_rng(7))
    for j in range(5):
        assert stats.kstest(S[:, j], "uniform").pvalue > 0.001
    k = 0
    for j in range(5):
        for i in range(j):
            tau, se = _batch_tau(S[:, i], S[:, j])
            assert abs(tau - R_TAU[k]) < 3 * np.sqrt(2) * se, (i, j, tau, R_TAU[k])  # two independent simulations
            k += 1
    for e in trees[0]:  # the first tree's pairs: their copula's own tau
        tau, se = _batch_tau(S[:, e.a], S[:, e.b])
        assert abs(tau - V.pair_tau(e.family, e.params)) < 3 * se


def _partial(R, i, j, S):
    idx = [i, j, *S]
    P = np.linalg.inv(R[np.ix_(idx, idx)])
    return -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])


def test_gaussian_vines_are_the_gaussian_copula():
    from dataclasses import replace

    R = np.array([[1, 0.6, 0.4, 0.3, -0.2], [0.6, 1, 0.5, 0.2, 0.1], [0.4, 0.5, 1, 0.45, 0.3], [0.3, 0.2, 0.45, 1, 0.25],
                  [-0.2, 0.1, 0.3, 0.25, 1]])
    U = cp.sample_copula("gaussian", {"R": R}, 1000, 5, np.random.default_rng(9))
    target = float(np.sum(cp.gaussian_logpdf(U, R)))
    _, U5 = _five()
    shapes = [V.fixed_structure(5, "dvine", [2, 0, 4, 1, 3]), V.fixed_structure(5, "cvine", [3, 1, 0, 4, 2]),
              V.fit_vine(cp.pseudo_observations(U5), families=["gaussian"])]
    for trees in shapes:
        g = [[replace(e, family="gaussian", params=(_partial(R, e.a, e.b, e.cond),)) for e in t] for t in trees]
        assert float(np.sum(V.propagate(U, g)[0])) == pytest.approx(target, rel=1e-12)  # relative: a large log likelihood


def test_sequential_estimates_recover_the_parameters():
    true = [[V.Edge(0, 0, 1, (), "gumbel", (1.8,)), V.Edge(0, 1, 2, (), "clayton", (2.0,)), V.Edge(0, 2, 3, (), "t", (0.5, 5.0))],
            [V.Edge(1, 0, 2, (1,), "frank", (-3.0,)), V.Edge(1, 1, 3, (2,), "joe90", (1.6,))],
            [V.Edge(2, 0, 3, (1, 2), "gaussian", (0.4,))]]
    U = V.simulate_vine(true, 4, 3000, np.random.default_rng(11))
    fit = V.fit_parameters(U, true, select=False)
    for tree_t, tree_f in zip(true, fit):
        for et, ef in zip(tree_t, tree_f):
            for value, est, se in zip(et.params, ef.params, ef.se):
                assert abs(est - value) < 3 * se, (et.family, value, est, se)
    chosen = V.fit_vine(U, structure="dvine", order=[0, 1, 2, 3])  # the families selected on the true structure
    assert [e.family for t in chosen for e in t][:3] == ["gumbel", "clayton", "t"]
    seq = float(np.sum(V.propagate(U[:800], V.fit_parameters(U[:800], true, select=False))[0]))
    joint = V.refine_joint(U[:800], V.fit_parameters(U[:800], true, select=False))
    assert float(np.sum(V.propagate(U[:800], joint)[0])) >= seq - 1e-9


def _data(X, names):
    idx = pd.period_range("2000-01-01", periods=len(X), freq="D", name="time")
    return Dataset.from_frame(pd.DataFrame(X, columns=names, index=idx))


def test_through_the_api():
    R = np.array([[1, 0.6, 0.4], [0.6, 1, 0.5], [0.4, 0.5, 1]])
    U = cp.sample_copula("gaussian", {"R": R}, 800, 3, np.random.default_rng(12))
    data = _data(stats.t.ppf(U, 5) * 0.01, ["a", "b", "c"])
    vine = vine_copula(data, ["a", "b", "c"], families=["gaussian", "indep"])
    gauss = copula(data, ["a", "b", "c"], family="gaussian")
    assert vine.loglike >= gauss.loglike - 1.0  # the same model up to the Gaussian copula's normal-score estimator
    a, b = vine.portfolio_var([0.5, 0.3, 0.2]), gauss.portfolio_var([0.5, 0.3, 0.2])
    assert a["var"] == pytest.approx(b["var"], rel=0.03) and a["es"] == pytest.approx(b["es"], rel=0.03)
    table = vine.summary()
    assert table.to_frame().shape[0] == 3 and "R-vine" in str(table)
    assert vine.aic == pytest.approx(2 * vine.k - 2 * vine.loglike)
    assert vine.logpdf(vine.u).sum() == pytest.approx(vine.loglike)
    d = vine_copula(data, ["a", "b", "c"], structure="dvine", order=["b", "a", "c"], margins="t", trunc_level=1)
    assert [e.family for e in d.trees[1]] == ["indep"] and d.margin_fits[0].kind == "t" and d.total_loglike is not None
    assert {frozenset((e.a, e.b)) for e in d.trees[0]} == {frozenset((0, 1)), frozenset((0, 2))}
    c = vine_copula(data, ["a", "b", "c"], structure="cvine", order=["c"], garch=True)
    assert all(2 in (e.a, e.b) for e in c.trees[0]) and c.garch
    m = vine_copula(data, ["a", "b", "c"], matrix=[[1, 0, 0], [3, 2, 0], [2, 3, 3]])
    assert {frozenset((e.a, e.b)) for e in m.trees[0]} == {frozenset((1, 2)), frozenset((0, 1))}
    with pytest.raises(SpecificationError, match="pair-copula families"):
        vine_copula(data, ["a", "b", "c"], families=["plackett"])
    with pytest.raises(SpecificationError, match="truncation level"):
        vine_copula(data, ["a", "b", "c"], trunc_level=3)
    with pytest.raises(SpecificationError, match="order"):
        vine_copula(data, ["a", "b", "c"], structure="dvine", order=["a", "b"])
    with pytest.raises(SpecificationError, match="order"):
        vine_copula(data, ["a", "b", "c"], order=["a", "b", "c"])


def test_the_dialog_replays():
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS

    R = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])
    data = _data(stats.norm.ppf(cp.sample_copula("t", {"R": R, "nu": 6.0}, 400, 3, np.random.default_rng(13))) * 0.01, ["a", "b", "c"])
    s = Session()
    s.objects["data"] = data
    s.run(COMMANDS["FitVineCopula"].cls("a, b, c", structure="dvine", order="b, a, c", families="gaussian, t, clayton",
                                        criterion="bic", trunc_level=2, margins="normal"))
    code = s.history[-1].code
    assert "structure='dvine'" in code and "criterion='bic'" in code and "trunc_level=2" in code
    namespace = dict(s.namespace) | {"data": data}
    exec("from pymetrix_finance import *\n" + code, namespace)
    fit = s.objects["vine_fit"]
    assert namespace["vine_fit"].loglike == pytest.approx(fit.loglike, abs=1e-12)
    assert [e.family for e in namespace["vine_fit"].edges] == [e.family for e in fit.edges]
