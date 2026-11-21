"""The Bartlett long-run covariance and its automatic bandwidth, whatever arch is installed.

- Lags: every lag j < b + 1 has the weight 1 - j/(b+1); the estimate equals a brute-force sum of the Bartlett
  kernel over all T - 1 lags, its spectral window is non-negative for every b on a grid (so the estimate is
  positive semi-definite), and the long-run variance of an AR(1) with phi = -0.9 at b = 1.5 is positive (pymetrix
  0.1.0a18 gave -0.542).
- The default rule ("newey-west") against numbers recorded from arch 9.0.0 (``arch.unitroot.cointegration`` and
  ``arch.covariance.kernel.Bartlett``) and from R's sandwich 3.1.0 (``bwNeweyWest``, ``kernHAC`` and ``NeweyWest``,
  prewhite = FALSE, adjust = FALSE, on DOLS residuals).
- The legacy rule against numbers recorded from pymetrix 0.1.0a18 on the same fixtures, to 1e-12.
- The dialog's Bandwidth rule against the API call, and its generated script replayed.
"""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, cointegrating_regression
from pymetrix.cointegrating_regression import bartlett_weights, long_run_covariance, newey_west_bandwidth
from pymetrix.errors import SpecificationError


def _system() -> pd.DataFrame:
    """The simulated system of tests/test_cointegrating_regression.py (T = 400, seed 0)."""
    rng = np.random.default_rng(0)
    T = 400
    v = rng.normal(size=(T, 2))
    x = np.cumsum(v, axis=0)
    u = np.zeros(T)
    e = 0.5 * v[:, 0] + rng.normal(size=T)
    for t in range(1, T):
        u[t] = 0.5 * u[t - 1] + e[t]
    y = 1 + 2 * x[:, 0] - x[:, 1] + u
    return pd.DataFrame({"y": y, "a": x[:, 0], "b": x[:, 1]}, index=pd.period_range("1900Q1", periods=T, freq="Q"))


@pytest.fixture(scope="module")
def cases():
    macro = Dataset.example("sample_macro").derive(lc="log(c)", ly="log(y)")
    return {"system": (Dataset.from_frame(_system()), "y", ["a", "b"]), "sample_macro": (macro, "lc", ["ly"])}


def _ar1(phi: float = -0.9, T: int = 400, seed: int = 0) -> np.ndarray:
    z = np.random.default_rng(seed).normal(size=T)
    e = np.zeros(T)
    for t in range(1, T):
        e[t] = phi * e[t - 1] + z[t]
    return e


# (bandwidth, coefficients, standard errors) of the cases, (case, method, trend)
# from arch 9.0.0's FullyModifiedOLS, CanonicalCointegratingReg and DynamicOLS with their automatic bandwidth
ARCH9 = {
    ("system", "fmols", "c"): (3.0, [1.9982161430029137, -1.0058445887531744, 0.9107624907084113], [0.009822256313864717, 0.02237549815646016, 0.23878528381942674]),
    ("system", "fmols", "ct"): (0.0, [2.050225328354402, -0.9949258826321941, 0.7154031363505831, 0.004701320585888062], [0.011925151277891117, 0.01440856387759478, 0.15651002271278897, 0.0009498834134919652]),
    ("system", "ccr", "c"): (3.0, [1.9981871772782205, -1.005924261851905, 0.9113278895312187], [0.009839371838004537, 0.02237118362025292, 0.2368500687320953]),
    ("system", "ccr", "ct"): (0.0, [2.0502210399340055, -0.994921691134962, 0.7203337362681177, 0.00469949880543839], [0.011925953014388984, 0.014361841744182344, 0.15500796478597134, 0.0009513864023292998]),
    ("system", "dols", "c"): (10.078906998959248, [1.9998208785415867, -1.0041536756767873, 0.9171138109386135], [0.011879023852790695, 0.027102671699019823, 0.2867797869881204]),
    ("system", "dols", "ct"): (9.301271571354437, [2.050064504926731, -0.9950126837030494, 0.7484536510618904, 0.004691166098209241], [0.020694887640472573, 0.025028804613963205, 0.2692241061350176, 0.0016436730498034279]),
    ("sample_macro", "fmols", "c"): (6.0, [0.9590527378656142, -0.0022208875886714045], [0.0013818756306559661, 0.008281596187262126]),
    ("sample_macro", "ccr", "c"): (6.0, [0.9590456392666898, -0.0021795566662731183], [0.0013588376360188707, 0.008119301232198494]),
    ("sample_macro", "dols", "c"): (3.191390398811027, [0.9593021681249401, -0.003770867518657656], [0.0014962469502196093, 0.009173448705969303]),
}

# from pymetrix 0.1.0a18 (the same as arch 8.0.0's)
A18 = {
    ("system", "fmols", "c"): (2.0, [1.9986141927749652, -1.0050694375740745, 0.9095301855169143], [0.009083334208654169, 0.020692203638929818, 0.22082161854996232]),
    ("system", "fmols", "ct"): (4.0, [2.045581702044642, -0.9966450864798534, 0.7229133203408179, 0.0043560059954339415], [0.018714219833228973, 0.022611455871955673, 0.24561222771077162, 0.0014906584077455994]),
    ("system", "ccr", "c"): (2.0, [1.9985873378800574, -1.0051435835071132, 0.9100659187937231], [0.009098755111937175, 0.02066970632731359, 0.21897512792047136]),
    ("system", "ccr", "ct"): (4.0, [2.045636920001432, -0.9966007978423693, 0.7273104683390512, 0.004358090356405142], [0.01867877930907849, 0.022586926697804367, 0.24361956971058005, 0.0014905718240922187]),
    ("system", "dols", "c"): (11.70548871080216, [1.9998208785415872, -1.0041536756767877, 0.9171138109386094], [0.012136715305123537, 0.027690609472255278, 0.29300089578666116]),
    ("system", "dols", "ct"): (10.541536685476277, [2.0500645049267305, -0.9950126837030497, 0.7484536510618888, 0.004691166098208993], [0.02092700418386621, 0.02530953093213399, 0.2722437586211751, 0.0016621086998740043]),
    ("sample_macro", "fmols", "c"): (8.0, [0.9589984613250936, -0.001851429339398825], [0.0012631856613225779, 0.0075702858669324646]),
    ("sample_macro", "ccr", "c"): (8.0, [0.9590080236283025, -0.0019108729494810113], [0.001237548829127745, 0.007390675898134317]),
    ("sample_macro", "dols", "c"): (3.3994881609819174, [0.9593021681249418, -0.0037708675186625485], [0.0015057466633202332, 0.009231691184483432]),
}


def test_every_lag_below_b_plus_one_is_used():
    assert bartlett_weights(1.5) == pytest.approx([1 - 1 / 2.5, 1 - 2 / 2.5], abs=1e-15)  # lag 2 = floor(1.5) + 1 included
    assert bartlett_weights(1.5, "legacy") == pytest.approx([1 - 1 / 2.5], abs=1e-15)  # 0.1.0a18 left it out
    assert len(bartlett_weights(3.0)) == len(bartlett_weights(3.0, "legacy")) == 3  # a whole number: lags 1..b either way
    assert len(bartlett_weights(0.0)) == 0 and len(bartlett_weights(0.3)) == 1
    x = np.random.default_rng(1).normal(size=(120, 2)).cumsum(0) * 0.05 + np.random.default_rng(2).normal(size=(120, 2))
    T = len(x)
    for b in (0.4, 1.5, 2.3, 5.9, 11.7, 3.0):
        omega, lam, sigma, used = long_run_covariance(x, b, force_int=False)
        gamma = [x[j:].T @ x[:T - j] / T for j in range(T)]
        k = [max(0.0, 1 - j / (b + 1)) for j in range(T)]  # the Bartlett kernel at every lag, by brute force
        one_sided = sum(k[j] * gamma[j] for j in range(T))
        assert used == b
        assert lam == pytest.approx(one_sided, abs=1e-12)
        assert omega == pytest.approx(one_sided + one_sided.T - gamma[0], abs=1e-12)


def test_the_spectral_window_is_not_negative():
    lam = np.linspace(0, np.pi, 4001)
    worst_legacy = 0.0
    for b in np.arange(0.0, 15.01, 0.05):
        w = bartlett_weights(b)
        window = 1 + 2 * np.cos(np.outer(lam, np.arange(1, len(w) + 1))) @ w
        assert window.min() >= -1e-12, b
        wl = bartlett_weights(b, "legacy")
        worst_legacy = min(worst_legacy, (1 + 2 * np.cos(np.outer(lam, np.arange(1, len(wl) + 1))) @ wl).min())
    assert worst_legacy < -0.19  # the legacy truncation is not a kernel: b = 1.5 reaches -0.200


def test_the_ar1_example_is_positive():
    e = _ar1()
    new = long_run_covariance(e, 1.5, force_int=False)[0][0, 0]
    old = long_run_covariance(e, 1.5, force_int=False, bandwidth_rule="legacy")[0][0, 0]
    assert new == pytest.approx(1.4006582323389827, rel=1e-12)  # arch 9.0.0's Bartlett(e, bandwidth=1.5, center=False)
    assert old == pytest.approx(-0.5419578388580346, rel=1e-12)  # pymetrix 0.1.0a18 and arch 8: negative
    x = np.column_stack([e, _ar1(0.6, seed=3), _ar1(-0.5, seed=4)])
    for b in np.arange(0.0, 12.01, 0.1):  # positive semi-definite for every bandwidth
        assert np.linalg.eigvalsh(long_run_covariance(x, b, force_int=False)[0]).min() > -1e-12


@pytest.mark.parametrize("key", sorted(ARCH9))
def test_the_default_rule_equals_arch_9(cases, key):
    case, method, trend = key
    data, y, x = cases[case]
    ours = cointegrating_regression(data, y, x, method=method, trend=trend)
    bw, params, se = ARCH9[key]
    assert ours.bandwidth == pytest.approx(bw, rel=1e-10, abs=1e-12)
    # abs: a coefficient near 0 moves by up to 2e-12 with OpenBLAS's kernel (Haswell or Zen on CI's runners, SkylakeX
    # here); 1e-10 is the agreement claimed and still about 1e-10 of a standard error
    assert ours.params == pytest.approx(params, rel=1e-10, abs=1e-10)
    assert ours.se == pytest.approx(se, rel=1e-10)


@pytest.mark.parametrize("key", sorted(A18))
def test_the_legacy_rule_reproduces_0_1_0a18(cases, key):
    case, method, trend = key
    data, y, x = cases[case]
    ours = cointegrating_regression(data, y, x, method=method, trend=trend, bandwidth_rule="legacy")
    bw, params, se = A18[key]
    assert ours.bandwidth == pytest.approx(bw, rel=1e-12)
    # abs: a coefficient near 0 differs by 7e-13 with numpy 1.26 and by up to 1.5e-12 with another OpenBLAS kernel
    assert ours.params == pytest.approx(params, rel=1e-12, abs=1e-10)
    assert ours.se == pytest.approx(se, rel=1e-12)
    assert ours.bandwidth_rule == "legacy" and "legacy rule" in str(ours.summary())


def test_against_r_sandwich():
    """DOLS residuals of log consumption on log income: sample_macro (97 rows) and statsmodels' macrodata (199 rows).
    R: m <- lm(e ~ 1); bwNeweyWest(m, kernel = "Bartlett", prewhite = FALSE, adjust = FALSE); kernHAC(m, bw =
    bwNeweyWest, kernel = "Bartlett", prewhite = FALSE, adjust = FALSE, sandwich = FALSE); NeweyWest(m, prewhite =
    FALSE, adjust = FALSE, sandwich = FALSE). The residuals have mean zero (a constant in the regression), so
    sandwich's centring changes nothing to these digits."""
    macro = Dataset.example("sample_macro").derive(lc="log(c)", ly="log(y)")
    md = pytest.importorskip("statsmodels.datasets.macrodata").load_pandas().data
    us = Dataset.from_frame(pd.DataFrame({"lc": np.log(md.realcons.to_numpy()), "ly": np.log(md.realdpi.to_numpy())},
                                         index=pd.period_range("1959Q1", periods=len(md), freq="Q")))
    sandwich = [(macro, 97, 4.19139039881267, 1.06052151299027e-05, 1.08479383514964e-05),
                (us, 199, 11.023028860907, 0.00345378918218207, 0.00370084713235439)]
    for data, n, S, kern_hac, newey_west in sandwich:
        fit = cointegrating_regression(data, "lc", ["ly"], method="dols")
        e = fit.resid.to_numpy()[:, None]
        assert len(e) == n
        assert newey_west_bandwidth(e) == pytest.approx(S, rel=1e-10)
        assert long_run_covariance(e, force_int=False)[0][0, 0] == pytest.approx(kern_hac, rel=1e-10)
        assert fit.long_run_variance == pytest.approx(kern_hac, rel=1e-10)
        omega, _, _, b = long_run_covariance(e)  # FMOLS's rounding: b = ceil(S) - 1 = sandwich's lag floor(S)
        assert b == np.floor(S) and omega[0, 0] == pytest.approx(newey_west, rel=1e-10)


def test_bandwidth_is_capped_and_rules_checked():
    e = _ar1(0.99, T=30, seed=2)[:, None]
    assert newey_west_bandwidth(e) <= 29
    with pytest.raises(SpecificationError, match="bandwidth_rule"):
        long_run_covariance(e, bandwidth_rule="arch8")
    with pytest.raises(SpecificationError, match="bandwidth_rule"):
        cointegrating_regression(Dataset.from_frame(_system()), "y", ["a", "b"], bandwidth_rule="bad")


def test_the_dialog_s_bandwidth_rule():
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import FitCointegratingRegression

    data = Dataset.from_frame(_system())
    s = Session()
    s.objects["data"] = data
    s.run(FitCointegratingRegression("y", "a, b", method="dols", bandwidth_rule="legacy", name="old"))
    s.run(FitCointegratingRegression("y", "a, b", method="dols", name="new"))
    by_api = cointegrating_regression(data, "y", ["a", "b"], method="dols", bandwidth_rule="legacy")
    np.testing.assert_array_equal(s["old"].cov, by_api.cov)
    np.testing.assert_array_equal(s["new"].cov, cointegrating_regression(data, "y", ["a", "b"], method="dols").cov)
    assert not np.allclose(s["old"].cov, s["new"].cov)
    script = s.script()
    assert "bandwidth_rule='legacy'" in script and script.count("bandwidth_rule") == 1  # the default is left out
    ns = dict(s.objects)
    exec(script, ns)
    np.testing.assert_array_equal(ns["old"].cov, by_api.cov)
    np.testing.assert_array_equal(ns["new"].cov, s["new"].cov)
