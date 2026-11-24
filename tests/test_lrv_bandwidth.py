"""The Bartlett long-run covariance and its automatic bandwidth, whatever arch is installed.

- Lags: every lag j < b + 1 has the weight 1 - j/(b+1); the estimate equals a brute-force sum of the Bartlett
  kernel over all T - 1 lags, its spectral window is non-negative for every b on a grid (so the estimate is
  positive semi-definite), and the long-run variance of an AR(1) with phi = -0.9 at b = 1.5 is positive (pymetrix
  0.1.0a18 gave -0.542).
- The default rule ("newey-west") against numbers recorded from arch 9.0.0 (``arch.unitroot.cointegration`` and
  ``arch.covariance.kernel.Bartlett``) and from R's sandwich 3.1.0 (``bwNeweyWest``, ``kernHAC`` and ``NeweyWest``,
  prewhite = FALSE, adjust = FALSE, on DOLS residuals).
- The removed "legacy" rule (pymetrix 0.1.0a18's, kept for 0.1.0a19 and 0.1.0a20): asking for it, in a call, a
  replayed script or a dialog step saved by those versions, stops with an error that names it; a saved step with the
  default rule loads and runs as before.
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

def test_every_lag_below_b_plus_one_is_used():
    assert bartlett_weights(1.5) == pytest.approx([1 - 1 / 2.5, 1 - 2 / 2.5], abs=1e-15)  # lag 2 = floor(1.5) + 1 included
    assert len(bartlett_weights(3.0)) == 3  # a whole number: lags 1..b
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
    for b in np.arange(0.0, 15.01, 0.05):
        w = bartlett_weights(b)
        window = 1 + 2 * np.cos(np.outer(lam, np.arange(1, len(w) + 1))) @ w
        assert window.min() >= -1e-12, b


def test_the_ar1_example_is_positive():
    e = _ar1()
    new = long_run_covariance(e, 1.5, force_int=False)[0][0, 0]
    assert new == pytest.approx(1.4006582323389827, rel=1e-12)  # arch 9.0.0's Bartlett(e, bandwidth=1.5, center=False)
    # (pymetrix 0.1.0a18 and arch 8, which left out lag 2, gave -0.5419578388580346)
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
    assert long_run_covariance(e, bandwidth_rule="newey-west")[0] == pytest.approx(long_run_covariance(e)[0], abs=0)
    with pytest.raises(SpecificationError, match="bandwidth_rule"):
        long_run_covariance(e, bandwidth_rule="arch8")
    with pytest.raises(SpecificationError, match="bandwidth_rule"):
        cointegrating_regression(Dataset.from_frame(_system()), "y", ["a", "b"], bandwidth_rule="bad")


def test_the_removed_legacy_rule_says_so():
    """A call or a replayed script from 0.1.0a19 or 0.1.0a20 that asked for "legacy" stops with an error naming it."""
    from pymetrix.calls import make_call
    from pymetrix.codec import Ref
    from pymetrix.commands import Session

    data = Dataset.from_frame(_system())
    for call in (lambda: cointegrating_regression(data, "y", ["a", "b"], method="dols", bandwidth_rule="legacy"),
                 lambda: long_run_covariance(_ar1()[:, None], bandwidth_rule="legacy"),
                 lambda: bartlett_weights(1.5, "legacy"), lambda: newey_west_bandwidth(_ar1()[:, None], "legacy")):
        with pytest.raises(SpecificationError, match="legacy.*removed after 0.1.0a20"):
            call()
    s = Session()
    s.objects["data"] = data
    saved = make_call("pymetrix:cointegrating_regression", output="old", data=Ref("data"), y="y", x=["a", "b"],
                      method="dols", trend="c", bandwidth_rule="legacy")  # as the 0.1.0a20 dialog recorded it
    with pytest.raises(SpecificationError, match="removed after 0.1.0a20"):
        saved.run(s)
    with pytest.raises(SpecificationError, match="removed after 0.1.0a20"):
        exec("from pymetrix import cointegrating_regression\n" + saved.to_code(), dict(s.objects))


def test_dialog_steps_saved_with_a_bandwidth_rule():
    """Steps saved by 0.1.0a19 or 0.1.0a20 (schema 1, with the Bandwidth rule field): the default rule loads as the
    same step without it; "legacy" stops with a MigrationError naming the rule. The dialog's call equals the API's."""
    from pymetrix.commands import Session
    from pymetrix.commands.builtin import FitCointegratingRegression
    from pymetrix.commands.registry import MigrationError, from_json, to_json

    fields = {"y": "y", "x": "a, b", "data": "data", "method": "dols", "trend": "c", "name": "coint", "bandwidth": None,
              "leads": None, "lags": None, "start": "", "end": "", "where": ""}
    saved = {"kind": "command", "id": "pymetrix:fit_cointegrating_regression", "version": 1,
             "fields": {**fields, "bandwidth_rule": "newey-west"}}
    step = from_json(saved)
    assert step == FitCointegratingRegression("y", "a, b", method="dols")
    assert to_json(step)["version"] == 2 and "bandwidth_rule" not in to_json(step)["fields"]
    with pytest.raises(MigrationError, match="Bandwidth rule = legacy"):
        from_json({**saved, "fields": {**fields, "bandwidth_rule": "legacy"}})
    data = Dataset.from_frame(_system())
    s = Session()
    s.objects["data"] = data
    s.run(step)
    np.testing.assert_array_equal(s["coint"].cov, cointegrating_regression(data, "y", ["a", "b"], method="dols").cov)
    script = s.script()
    assert "bandwidth_rule" not in script
    ns = dict(s.objects)
    exec(script, ns)
    np.testing.assert_array_equal(ns["coint"].cov, s["coint"].cov)
