"""The copula families (Frank, Joe, BB1, BB7, the survival and rotated copulas, the Archimedean copulas of more than
two series), checked independently:

- each bivariate density integrates to one (scipy's 2-D adaptive quadrature), and is the mixed derivative of its CDF
  (central finite differences); the three-series densities are the third mixed derivative of the CDF; the
  h-functions are the CDF's derivative in the first argument (finite differences) and their inverses invert them;
- the densities and CDFs against R: the copula package 1.1-7 (Clayton, Gumbel, Frank, Joe in 2, 3 and 5 dimensions,
  Frank with negative dependence, the bivariate t and Gaussian CDFs, rotations) and VineCopula 2.6.1 (BB1, BB7, the
  survival and the 90 and 270 degree rotations: density, CDF, h-function and its inverse, Kendall's tau and tail
  dependence); the values were computed once in R and are written here, so the test needs no R;
- Kendall's tau against the closed forms (Clayton theta/(theta+2), Gumbel 1-1/theta, Frank by the Debye function
  computed here by its series, Joe by its series, BB1 1-2/(delta(theta+2)), BB7's Beta-function form for theta < 2)
  and against the general Archimedean formula 1 + 4 int phi/phi'; and against the sample tau of 200,000 draws (within
  three standard errors from 20 batches);
- tail dependence: the closed forms against the limit of C(q, q)/q (or the upper analogue) from the CDF at tiny q,
  and 400,000 draws' joint exceedances against C(q, q) at q = 0.01 (a binomial band);
- the Marshall-Olkin samplers against conditional inversion through the h-functions, the four-series samplers'
  pairwise tau, and the Sibuya sampler's frequencies against its probabilities.
"""

import importlib

import numpy as np
import pytest
from scipy import integrate, special, stats

cp = importlib.import_module("pymetrix_finance.copulas")

CASES = {"clayton": {"theta": 2.0}, "gumbel": {"theta": 1.8}, "frank": {"theta": 5.0}, "joe": {"theta": 2.5},
         "bb1": {"theta": 0.8, "delta": 1.5}, "bb7": {"theta": 1.6, "delta": 1.2}, "clayton180": {"theta": 2.0},
         "gumbel180": {"theta": 1.8}, "joe180": {"theta": 2.5}, "clayton90": {"theta": 2.0}, "gumbel90": {"theta": 1.8},
         "joe90": {"theta": 2.5}, "clayton270": {"theta": 2.0}, "gumbel270": {"theta": 1.8}, "joe270": {"theta": 2.5}}
NEGATIVE_FRANK = {"theta": -4.0}


def _all_cases():
    return [(f, p) for f, p in CASES.items()] + [("frank", NEGATIVE_FRANK)]


@pytest.mark.parametrize("family, params", _all_cases(), ids=lambda v: v if isinstance(v, str) else str(v.get("theta")))
def test_density_integrates_to_one_and_is_the_cdfs_mixed_derivative(family, params):
    f = lambda b, a: np.exp(cp.copula_logpdf(family, params, np.array([[a, b]])))[0]  # noqa: E731
    total = integrate.dblquad(f, 1e-9, 1 - 1e-9, 1e-9, 1 - 1e-9, epsabs=1e-7)[0]
    assert total == pytest.approx(1.0, abs=2e-3)
    u = np.random.default_rng(0).uniform(0.05, 0.95, (8, 2))
    a, b, e = u[:, 0], u[:, 1], 1e-4
    C = lambda x, y: cp.copula_cdf(family, params, np.column_stack([x, y]))  # noqa: E731
    mixed = (C(a + e, b + e) - C(a + e, b - e) - C(a - e, b + e) + C(a - e, b - e)) / (4 * e * e)
    assert mixed == pytest.approx(np.exp(cp.copula_logpdf(family, params, u)), rel=1e-4)
    h = cp.copula_h(family, params, a, b)
    assert h == pytest.approx((C(a + e, b) - C(a - e, b)) / (2 * e), abs=1e-6)
    assert cp.copula_hinv(family, params, a, h) == pytest.approx(b, abs=1e-8)
    assert C(a, np.ones_like(a)) == pytest.approx(a, abs=1e-9) and C(np.ones_like(b), b) == pytest.approx(b, abs=1e-9)  # uniform margins


@pytest.mark.parametrize("family, theta", [("clayton", 1.5), ("gumbel", 1.7), ("frank", 4.0), ("joe", 2.2)])
def test_three_series_density_is_the_cdfs_third_derivative(family, theta):
    p = {"theta": theta}
    u = np.random.default_rng(1).uniform(0.15, 0.85, (6, 3))
    e = 2e-3
    total = np.zeros(len(u))
    for s in np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3):
        total += np.prod(s) * cp.copula_cdf(family, p, u + e * s)
    assert total / (8 * e**3) == pytest.approx(np.exp(cp.copula_logpdf(family, p, u)), rel=2e-4)


U3 = np.array([[0.2, 0.5, 0.7], [0.9, 0.85, 0.6], [0.1, 0.15, 0.3], [0.5, 0.5, 0.5]])
U1, U2 = np.array([0.2, 0.9, 0.1, 0.5, 0.03]), np.array([0.5, 0.85, 0.15, 0.5, 0.97])
R_COPULA = {  # copula 1.1-7 (R 4.3.3): dCopula / pCopula of archmCopula(family, param, dim), frankCopula(-4), tCopula(0.6, df=5),
             # normalCopula(-0.7), rotCopula(...) at the rows of U3 (the first d columns; for d = 5 U3 then its first two)
    ("clayton", 2, "pdf"): [0.723146313155409, 2.010267889335038, 3.606933583515288, 1.481003649342279],
    ("clayton", 2, "cdf"): [0.1889822365046136, 0.7860015405766170, 0.0834945509987065, 0.3779644730092273],
    ("clayton", 3, "pdf"): [0.331334202410847, 2.148198626548379, 3.668802642008076, 2.428629243009313],
    ("clayton", 3, "cdf"): [0.1855647966226565, 0.5426111921136864, 0.0806988290684853, 0.3162277660168379],
    ("clayton", 5, "pdf"): [0.665829752550513, 10.439784315831721, 78.598829618056456, 7.382812499999978],
    ("clayton", 5, "cdf"): [0.1335819483554513, 0.4990601770697076, 0.0581238193719096, 0.2500000000000000],
    ("gumbel", 2, "pdf"): [1.02373436223479, 2.69965018719239, 2.01807403413603, 1.39235644652203],
    ("gumbel", 2, "cdf"): [0.1657895609547915, 0.8183882861112634, 0.0451521397178557, 0.3610487387073936],
    ("gumbel", 3, "pdf"): [0.711744776012729, 1.330074415805654, 2.797961482841702, 2.145823885318108],
    ("gumbel", 3, "cdf"): [0.1571205557980981, 0.5703519189910687, 0.0333718125337299, 0.2791171066803860],
    ("gumbel", 5, "pdf"): [1.24169905829283, 7.72900310128801, 13.72426738669527, 5.91617173719207],
    ("gumbel", 5, "cdf"): [0.06850387839391248, 0.54402770694490987, 0.00840334445342992, 0.18361898692327783],
    ("frank", 2, "pdf"): [0.87910720568537, 2.30516796622746, 2.30516796622746, 1.47356372458463],
    ("frank", 2, "cdf"): [0.1754952912619595, 0.7968954882290783, 0.0468954882290783, 0.3771485107465208],
    ("frank", 3, "pdf"): [0.457588501310487, 1.696232155885858, 3.219821686239781, 2.478709440067811],
    ("frank", 3, "cdf"): [0.1689670200549490, 0.5589273605702743, 0.0357008943338394, 0.3064346306040263],
    ("frank", 5, "pdf"): [0.813875808675055, 8.692535901628208, 19.495762012870724, 8.251683795549409],
    ("frank", 5, "cdf"): [0.0817301815404249, 0.5252437925898982, 0.0070014643111181, 0.2214356768222581],
    ("joe", 2, "pdf"): [1.08790075345443, 3.35550948874588, 1.84376202675273, 1.39414163544983],
    ("joe", 2, "cdf"): [0.1593124484948882, 0.8303820914883648, 0.0316754164818107, 0.3642213355541624],
    ("joe", 3, "pdf"): [0.712634089935391, 0.819974212258506, 2.740999912660312, 2.163642366092199],
    ("joe", 3, "cdf"): [0.1503798181048495, 0.5836647354636016, 0.0185040110570096, 0.2785438859644710],
    ("joe", 5, "pdf"): [1.19205074202011, 4.03306299088618, 10.59397119412082, 6.36350200292170],
    ("joe", 5, "cdf"): [0.04889197034472592, 0.56841630942972077, 0.00141238168250379, 0.17302439943922732],
    ("frankneg", 2, "pdf"): [0.934326710653689, 0.199837380970066, 0.199837380970066, 1.313035285499332],
    ("frankneg", 2, "cdf"): [0.0340886140737647, 0.7518788932838382, 0.0018788932838382, 0.1415547923792433],
    ("t", 2, "cdf"): [0.1654619960456107, 0.8046666769112797, 0.0546666769112798, 0.3524163823495667],
    ("gaussian", 2, "cdf"): [0.020789343954558684, 0.750128885425753622, 0.000128885425753608, 0.126591655553317528],
    ("clayton180", 2, "pdf"): [1.05422688410851, 3.60693358351530, 2.01026788933504, 1.48100364934228],
    ("clayton180", 2, "cdf"): [0.168164588784522, 0.833494550998707, 0.036001540576617, 0.377964473009227],
    ("gumbel90", 2, "cdf"): [0.023766977996157013, 0.751075035229311516, 0.000865719583396457, 0.138951261292606376],
    ("joe270", 2, "cdf"): [0.040687551505111719, 0.750538673867311878, 0.000944622670825623, 0.135778664445837638],
    ("joe270", 2, "pdf"): [1.087900753454434, 0.102622996557250, 0.170930857403138, 1.394141635449832],
}
R_VINE = {  # VineCopula 2.6.1: BiCopPDF, BiCopCDF, BiCopHfunc1, BiCopHinv1, BiCopPar2Tau, BiCopPar2TailDep at (U1, U2)
    ("bb1", "pdf"): [0.87129607654337016, 2.50408487368937527, 2.94253047360039899, 1.54949596520587352, 0.00566714389056347],
    ("bb1", "cdf"): [0.1836189314682390, 0.8111563250350862, 0.0730733978518412, 0.3782861495706971, 0.0299985134577781],
    ("bb1", "h"): [0.818373529923305, 0.576095962470049, 0.491335221404035, 0.480382804414482, 0.999889714098501],
    ("bb1", "hinv"): [0.266813431844196, 0.942050883624713, 0.057611452551985, 0.512667488738745, 0.407180224487278],
    ("bb1", "tau"): [0.523809523809524],
    ("bb1", "tail"): [0.561231024154687, 0.412598948031801],
    ("bb7", "pdf"): [0.9400364610863918, 2.2800889716740205, 2.6717884409893995, 1.4119047291456082, 0.0112879935976246],
    ("bb7", "cdf"): [0.1772249645303746, 0.8091806013678243, 0.0698816546254057, 0.3612954021636205, 0.0299971227717669],
    ("bb7", "h"): [0.766466289155243, 0.610867058604796, 0.455085953584166, 0.477451828569276, 0.999789167508109],
    ("bb7", "hinv"): [0.2901111528940069, 0.9378289251635916, 0.0587325354458429, 0.5159722888838267, 0.4918085743993452],
    ("bb7", "tau"): [0.470343303850493],
    ("bb7", "tail"): [0.561231024154687, 0.457789174592059],
    ("bb7b", "pdf"): [1.03100996912429199, 3.38202543776496833, 2.32473496404859414, 1.56595318873365152, 0.00364569530342728],
    ("bb7b", "cdf"): [0.1755423309487992, 0.8304914858403528, 0.0590042179044781, 0.3770431446109864, 0.0299992315041721],
    ("bb7b", "h"): [0.792030754079251, 0.446442042939436, 0.402674500130666, 0.537503718809782, 0.999956256097101],
    ("bb7b", "hinv"): [0.2852712965545581, 0.9374740220837683, 0.0569610865272573, 0.4759892684651756, 0.6020201940916883],
    ("bb7b", "tau"): [0.52992823887578],
    ("bb7b", "tail"): [0.371498572284237, 0.680492089227106],
    ("clayton90", "pdf"): [1.0542268841085132, 0.0483840632072579, 0.0913820799163556, 1.4810036493422789, 2.6792121273893912],
    ("clayton90", "cdf"): [0.031835411215477727, 0.750191490087986135, 0.000394273352577745, 0.122035526990772747, 0.027452490448727218],
    ("clayton90", "h"): [0.20041292328104549, 0.99426629087490981, 0.00459321871135127, 0.43195939772483111, 0.91747565239818740],
    ("clayton90", "hinv"): [0.722098996418426, 0.283486139514753, 0.491561496279055, 0.546390642842887, 0.989273962420798],
    ("clayton90", "tau"): [-0.5],
    ("clayton90", "tail"): [0, 0],
    ("clayton270", "pdf"): [0.7231463131554087, 0.0913820799163557, 0.0483840632072578, 1.4810036493422789, 17.6975725797728778],
    ("clayton270", "cdf"): [0.011017763495386385, 0.750394273352577690, 0.000191490087986121, 0.122035526990772747, 0.008782021982148600],
    ("clayton270", "h"): [0.15632930131868916, 0.99540678128864868, 0.00573370912509008, 0.56804060227516895, 0.64620782655776066],
    ("clayton270", "hinv"): [0.747502330288812, 0.508438503720945, 0.716513860485247, 0.453609357157113, 0.990193324479060],
    ("clayton270", "tau"): [-0.5],
    ("clayton270", "tail"): [0, 0],
    ("gumbel90", "pdf"): [0.896417025131662, 0.186452377771806, 0.154599041054108, 1.392356446522030, 10.356468828235956],
    ("gumbel90", "cdf"): [0.023766977996157013, 0.751075035229311516, 0.000865719583396457, 0.138951261292606376, 0.013779694932414044],
    ("gumbel90", "h"): [0.227690571010406, 0.985550367532330, 0.016366017807315, 0.530645784670857, 0.724427816789756],
    ("gumbel90", "hinv"): [0.715364943199255, 0.532796346308543, 0.554298467984292, 0.477951997453430, 0.993171381306745],
    ("gumbel90", "tau"): [-0.444444444444444],
    ("gumbel90", "tail"): [0, 0],
    ("gumbel270", "pdf"): [1.023734362234791, 0.154599041054108, 0.186452377771807, 1.392356446522030, 4.005120275127880],
    ("gumbel270", "cdf"): [0.03421043904520851, 0.75086571958339643, 0.00107503522931147, 0.13895126129260638, 0.02422206594922150],
    ("gumbel270", "h"): [0.2410360534771505, 0.9836339821926849, 0.0144496324676696, 0.4693542153291435, 0.8584661838579528],
    ("gumbel270", "hinv"): [0.703675230019033, 0.445701532015708, 0.467203653691458, 0.522048002546570, 0.994631450012146],
    ("gumbel270", "tau"): [-0.444444444444444],
    ("gumbel270", "tail"): [0, 0],
    ("joe90", "pdf"): [0.752994790195981, 0.170930857403138, 0.102622996557250, 1.394141635449831, 16.499334344809029],
    ("joe90", "cdf"): [0.016262045452965501, 0.750944622670825623, 0.000538673867312295, 0.135778664445837638, 0.009584003148767062],
    ("joe90", "h"): [0.198498556673413, 0.989727180790724, 0.013460450613333, 0.574134623943004, 0.659681960676126],
    ("joe90", "hinv"): [0.734946004550427, 0.557636239541613, 0.667124505278082, 0.446340422250652, 0.990800676953796],
    ("joe90", "tau"): [-0.448828392781576],
    ("joe90", "tail"): [0, 0],
    ("joe270", "pdf"): [1.087900753454434, 0.102622996557250, 0.170930857403138, 1.394141635449831, 2.296497544359108],
    ("joe270", "cdf"): [0.040687551505111774, 0.750538673867312323, 0.000944622670825651, 0.135778664445837638, 0.027846136078796352],
    ("joe270", "h"): [0.2358111233092214, 0.9865395493866670, 0.0102728192092761, 0.4258653760569961, 0.9297266867914169],
    ("joe270", "hinv"): [0.701768894073147, 0.332875494721918, 0.442363760458387, 0.553659577749348, 0.987336331106102],
    ("joe270", "tau"): [-0.448828392781576],
    ("joe270", "tail"): [0, 0],
    ("joe180", "pdf"): [0.7529947901959806, 1.8437620267527333, 3.3555094887458758, 1.3941416354498313, 0.0136068738613339],
    ("joe180", "cdf"): [0.1837379545470345, 0.7816754164818107, 0.0803820914883648, 0.3642213355541624, 0.0299952144471344],
    ("joe180", "h"): [0.801501443326587, 0.700819678135853, 0.551263574832833, 0.425865376056996, 0.999601205403466],
    ("joe180", "hinv"): [0.2650539954495728, 0.9275231862111559, 0.0625906383437898, 0.5536595777493478, 0.3000604944628167],
    ("joe180", "tau"): [0.448828392781576],
    ("joe180", "tail"): [0.680492089227106, 0.000000000000000],
    ("gumbel180", "pdf"): [0.8964170251316620, 2.0180740341360299, 2.6996501871923919, 1.3923564465220302, 0.0283987158669139],
    ("gumbel180", "cdf"): [0.1762330220038429, 0.7951521397178556, 0.0683882861112634, 0.3610487387073936, 0.0299886096742236],
    ("gumbel180", "h"): [0.772309428989594, 0.643862728319690, 0.456362794528623, 0.469354215329143, 0.999306227563582],
    ("gumbel180", "hinv"): [0.2846350568007449, 0.9418327369001115, 0.0594041202723761, 0.5220480025465702, 0.5773488881510747],
    ("gumbel180", "tau"): [0.444444444444444],
    ("gumbel180", "tail"): [0.530265507724401, 0.000000000000000],
    ("frank", "pdf"): [0.934326710653689, 0.199837380970066, 0.199837380970066, 1.313035285499332, 3.290376773823110],
    ("frank", "cdf"): [0.0340886140737647, 0.7518788932838382, 0.0018788932838382, 0.1415547923792433, 0.0267222161008298],
    ("frank", "h"): [0.2314752165009823, 0.9772888774576338, 0.0227111225423662, 0.5000000000000000, 0.8964879270025443],
    ("frank", "hinv"): [0.717213166803524, 0.525611598233757, 0.474388401766242, 0.500000000000000, 0.991590416190701],
    ("frank", "tau"): [-0.38814802],
    ("frank", "tail"): [0, 0],
}
R_PARAMS = {"clayton": ("clayton", {"theta": 2.0}), "gumbel": ("gumbel", {"theta": 1.8}), "frank": ("frank", {"theta": 5.0}),
            "joe": ("joe", {"theta": 2.5}), "frankneg": ("frank", NEGATIVE_FRANK),
            "t": ("t", {"R": np.array([[1, 0.6], [0.6, 1]]), "nu": 5.0}), "gaussian": ("gaussian", {"R": np.array([[1, -0.7], [-0.7, 1]])}),
            "clayton180": ("clayton180", {"theta": 2.0}), "gumbel90": ("gumbel90", {"theta": 1.8}), "joe270": ("joe270", {"theta": 2.5})}
VINE_PARAMS = {**{k: (k, v) for k, v in CASES.items()}, "bb7b": ("bb7", {"theta": 2.5, "delta": 0.7}), "frank": ("frank", NEGATIVE_FRANK)}


def test_against_the_r_copula_package():
    for (name, d, kind), expected in R_COPULA.items():
        family, params = R_PARAMS[name]
        u = U3[:, :2] if d == 2 else U3 if d == 3 else np.column_stack([U3, U3[:, :2]])
        got = np.exp(cp.copula_logpdf(family, params, u)) if kind == "pdf" else cp.copula_cdf(family, params, u)
        assert got == pytest.approx(expected, rel=1e-9 if family != "t" else 1e-8), (name, d, kind)


def test_against_r_vinecopula():
    U = np.column_stack([U1, U2])
    for (name, kind), expected in R_VINE.items():
        family, params = VINE_PARAMS[name]
        if kind == "pdf":
            got = np.exp(cp.copula_logpdf(family, params, U))
        elif kind == "cdf":
            got = cp.copula_cdf(family, params, U)
        elif kind == "h":
            got = cp.copula_h(family, params, U1, U2)
        elif kind == "hinv":
            got = cp.copula_hinv(family, params, U1, U2)
        elif kind == "tau":
            got = [cp.copula_tau(family, params)]
        else:
            td = cp.tail_dependence(family, params)
            got = [td["lower"], td["upper"]]
        assert got == pytest.approx(expected, rel=1e-8, abs=1e-8), (name, kind)


def _debye1_series(x):
    """D1(x) = 1 - x/4 + sum_k B_2k x^2k / ((2k + 1)(2k)!), |x| < 2 pi; beyond, D1(x) = (pi^2/6 - sum_k e^(-kx)(x/k + 1/k^2)) / x."""
    if abs(x) < 2:
        return 1 - x / 4 + sum(special.bernoulli(2 * k)[-1] * x ** (2 * k) / ((2 * k + 1) * special.factorial(2 * k)) for k in range(1, 15))
    if x < 0:
        return _debye1_series(-x) - x / 2
    k = np.arange(1, 400)
    return (np.pi**2 / 6 - np.sum(np.exp(-k * x) * (x / k + 1 / k**2))) / x


def test_kendalls_tau_closed_forms():
    arch = cp.Archimedean.tau  # 1 + 4 int phi/phi', the general formula, for every Archimedean family
    for th in (0.3, 2.0, 7.0):
        assert cp.copula_tau("clayton", {"theta": th}) == pytest.approx(th / (th + 2))
        assert arch(cp.ARCH["clayton"], [th]) == pytest.approx(th / (th + 2), abs=1e-8)
    for th in (1.2, 2.0, 5.0):
        assert cp.copula_tau("gumbel", {"theta": th}) == pytest.approx(1 - 1 / th)
        assert arch(cp.ARCH["gumbel"], [th]) == pytest.approx(1 - 1 / th, abs=1e-8)
        joe = 1 - 4 * sum(1 / (k * (th * k + 2) * (th * (k - 1) + 2)) for k in range(1, 200_000))
        assert cp.copula_tau("joe", {"theta": th}) == pytest.approx(joe, abs=1e-8)
        assert arch(cp.ARCH["joe"], [th]) == pytest.approx(joe, abs=1e-7)
    for th in (-6.0, -1.0, 0.5, 1.5, 5.736, 15.0):
        frank = 1 - 4 / th * (1 - _debye1_series(th))
        assert cp.copula_tau("frank", {"theta": th}) == pytest.approx(frank, abs=1e-10)
    assert cp.copula_tau("frank", {"theta": 5.736}) == pytest.approx(0.5, abs=1e-4)  # the often-quoted value
    for th, de in ((0.5, 1.2), (2.0, 3.0)):
        assert cp.copula_tau("bb1", {"theta": th, "delta": de}) == pytest.approx(1 - 2 / (de * (th + 2)))
        assert arch(cp.ARCH["bb1"], [th, de]) == pytest.approx(1 - 2 / (de * (th + 2)), abs=1e-8)
    for th, de in ((1.0, 2.0), (1.5, 0.8), (1.9, 3.0)):
        closed = 1 - 2 / (de * (2 - th)) + 4 / (th**2 * de) * special.beta(de + 2, 2 / th - 1)
        assert cp.copula_tau("bb7", {"theta": th, "delta": de}) == pytest.approx(closed, abs=1e-8)
    for family in ("clayton", "gumbel", "frank", "joe", "gumbel90", "joe180", "clayton270"):
        for tau in (-0.6, -0.2, 0.1, 0.45, 0.8):
            if (tau < 0) == family.endswith(("90", "270")) or family == "frank":
                th = cp.tau_to_parameter(family, tau)
                assert cp.copula_tau(family, {"theta": th}) == pytest.approx(tau, abs=1e-9)


def _batch_tau(U, batches=20):
    taus = [stats.kendalltau(b[:, 0], b[:, 1])[0] for b in np.array_split(U, batches)]
    return float(np.mean(taus)), float(np.std(taus, ddof=1) / np.sqrt(batches))


@pytest.mark.parametrize("family, params", _all_cases(), ids=lambda v: v if isinstance(v, str) else str(v.get("theta")))
def test_simulated_tau(family, params):
    U = cp.sample_copula(family, params, 200_000, 2, np.random.default_rng(7))
    assert U.min() > 0 and U.max() < 1
    mean, se = _batch_tau(U)
    assert abs(mean - cp.copula_tau(family, params)) < 3 * se
    for j in (0, 1):  # uniform margins
        assert stats.kstest(U[:, j], "uniform").pvalue > 1e-3


def test_tail_dependence():
    for family, params in _all_cases():
        td = cp.tail_dependence(family, params)
        q = 1e-7
        C = lambda a, b: cp.copula_cdf(family, params, np.array([[a, b]]))[0]  # noqa: E731
        assert C(q, q) / q == pytest.approx(td["lower"], abs=2e-3), family
        assert (2 * q - 1 + C(1 - q, 1 - q)) / q == pytest.approx(td["upper"], abs=2e-3), family
        if family.endswith(("90", "270")):
            assert (q - C(q, 1 - q)) / q == pytest.approx(td["lower_upper"], abs=2e-3)  # first low, second high
            assert (q - C(1 - q, q)) / q == pytest.approx(td["upper_lower"], abs=2e-3)
    th, de = 0.8, 1.5
    assert cp.tail_dependence("bb1", {"theta": th, "delta": de}) == {"lower": pytest.approx(2 ** (-1 / (th * de))), "upper": pytest.approx(2 - 2 ** (1 / de))}
    assert cp.tail_dependence("bb7", {"theta": 1.6, "delta": 1.2}) == {"lower": pytest.approx(2 ** (-1 / 1.2)), "upper": pytest.approx(2 - 2 ** (1 / 1.6))}
    assert cp.tail_dependence("joe", {"theta": 2.5})["upper"] == pytest.approx(2 - 2 ** 0.4)
    assert cp.tail_dependence("frank", {"theta": 5.0}) == {"lower": 0.0, "upper": 0.0}


@pytest.mark.parametrize("family", ["clayton", "gumbel", "joe", "bb1", "bb7", "gumbel180", "joe90"])
def test_simulated_tails(family):
    params = CASES[family]
    n, q = 400_000, 0.01
    U = cp.sample_copula(family, params, n, 2, np.random.default_rng(8))
    C = lambda a, b: cp.copula_cdf(family, params, np.array([[a, b]]))[0]  # noqa: E731
    corners = {"lower": (np.mean((U[:, 0] < q) & (U[:, 1] < q)), C(q, q)),
               "upper": (np.mean((U[:, 0] > 1 - q) & (U[:, 1] > 1 - q)), 2 * q - 1 + C(1 - q, 1 - q)),
               "off": (np.mean((U[:, 0] > 1 - q) & (U[:, 1] < q)), q - C(1 - q, q))}
    for name, (share, p) in corners.items():
        assert abs(share - p) < 4 * np.sqrt(p * (1 - p) / n) + 1e-6, name


@pytest.mark.parametrize("family", ["clayton", "gumbel", "frank", "joe", "bb1", "bb7", "joe180", "gumbel270"])
def test_marshall_olkin_against_conditional_inversion(family):
    params = CASES[family]
    a = cp.sample_copula(family, params, 100_000, 2, np.random.default_rng(9))
    b = cp.sample_conditional(family, params, 100_000, np.random.default_rng(10))
    (ta, sa), (tb, sb) = _batch_tau(a), _batch_tau(b)
    assert abs(ta - tb) < 3 * np.hypot(sa, sb)
    assert stats.ks_2samp(a[:, 1], b[:, 1]).pvalue > 1e-3
    for k in range(4):  # the joint CDF at a few points, against both
        pt = np.array([[0.3, 0.4], [0.7, 0.8], [0.1, 0.9], [0.5, 0.5]])[k]
        expected = cp.copula_cdf(family, params, pt[None, :])[0]
        for sample in (a, b):
            share = np.mean(np.all(sample <= pt, axis=1))
            assert abs(share - expected) < 4 * np.sqrt(expected * (1 - expected) / len(sample))


@pytest.mark.parametrize("family, theta", [("clayton", 1.5), ("gumbel", 1.7), ("frank", 4.0), ("joe", 2.2)])
def test_four_series_samplers(family, theta):
    U = cp.sample_copula(family, {"theta": theta}, 100_000, 4, np.random.default_rng(11))
    tau = cp.copula_tau(family, {"theta": theta})
    for i, j in ((0, 1), (1, 3), (0, 2)):
        mean, se = _batch_tau(U[:, [i, j]])
        assert abs(mean - tau) < 3 * se
    pt = np.array([0.4, 0.5, 0.6, 0.7])
    share, expected = np.mean(np.all(U <= pt, axis=1)), cp.copula_cdf(family, {"theta": theta}, pt[None, :])[0]
    assert abs(share - expected) < 4 * np.sqrt(expected * (1 - expected) / len(U))


def test_sibuya_sampler():
    a, n = 0.4, 200_000
    v = cp._sibuya(a, n, np.random.default_rng(12))[:, 0]
    for k in range(1, 8):
        p = (-1) ** (k + 1) * special.binom(a, k)  # P(V = k)
        assert abs(np.mean(v == k) - p) < 4 * np.sqrt(p * (1 - p) / n)
    k = 1000  # the tail: P(V > k) = Gamma(k + 1 - a) / (k! Gamma(1 - a))
    p = np.exp(special.gammaln(k + 1 - a) - special.gammaln(k + 1) - special.gammaln(1 - a))
    assert abs(np.mean(v > k) - p) < 4 * np.sqrt(p * (1 - p) / n)
