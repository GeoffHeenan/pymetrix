"""Roots plots for VARs and VECMs, and VECMs' cointegrating relations.

- Cointegrating relations against R's urca 1.3.3 (ca.jo, spec="transitory", K = 3; cajorls's normalised beta),
  applied to the data: a constant restricted to the relation (urca ecdet="const", case ci) and an unrestricted one
  (ecdet="none", case co), to 1e-8; dated at the levels they use.
- A VAR's companion eigenvalues against statsmodels' VAR roots (their inverses), to 1e-10; a VECM's levels VAR has
  K - rank unit roots and the rest inside the unit circle.
- The plots: the unit circle with the roots, and each relation over time; result-window tabs.
- Recursive eigenvalues (Hansen and Johansen 1999) against urca on truncated samples, in both cases, to 1e-9."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import VAR, VECM, Dataset
from pymetrix.views import sections

URCA = {"ci": [1.0, -1.2046985093330260, -0.0381997289619258, 1.0830904622613511],
        "co": [1.0, -1.2046886463759774, -0.0382069014333444]}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    T = 300
    x = np.cumsum(rng.normal(size=T))
    y = 0.8 * x + rng.normal(size=T) * 0.5 + 1
    z = 0.5 * x + np.cumsum(rng.normal(size=T)) * 0.5
    return Dataset.from_frame(pd.DataFrame({"x": x, "y": y, "z": z}, index=pd.period_range("1950Q1", periods=T, freq="Q", name="time")))


@pytest.mark.parametrize("case", ["ci", "co"])
def test_relations_against_urca(data, case):
    fit = VECM(["x", "y", "z"], lags=2, rank=1, deterministic=case).fit(data)
    beta = np.array(URCA[case])
    X = data.frame[["x", "y", "z"]].to_numpy()
    if case == "ci":
        X = np.column_stack([X, np.ones(len(X))])
    theirs = pd.Series(X @ beta, index=data.frame.index)
    ours = fit.relations().frame["ce1"]
    assert np.allclose(ours.to_numpy(), theirs.reindex(ours.index).to_numpy(), atol=1e-8)
    assert ours.index[0] == data.frame.index[2] and ours.index[-1] == data.frame.index[-2]  # y(t-1) for t in the sample


def test_roots(data):
    from statsmodels.tsa.api import VAR as SMVAR

    diffs = data.frame.diff().dropna()
    var = VAR(["x", "y", "z"], lags=2).fit(Dataset.from_frame(diffs))
    ref = SMVAR(diffs).fit(2)
    assert np.allclose(np.sort_complex(var.eigenvalues), np.sort_complex(1 / ref.roots), atol=1e-10)
    assert np.all(np.diff(var.roots) <= 1e-12) and var.is_stable
    vecm = VECM(["x", "y", "z"], lags=2, rank=1).fit(data)
    assert np.allclose(vecm.roots[:2], 1.0, atol=1e-8) and vecm.roots[2] < 1  # K - rank unit roots


def test_plots_and_tabs(data):
    vecm = VECM(["x", "y", "z"], lags=2, rank=1, deterministic="ci").fit(data)
    fig = vecm.roots_plot()
    assert "2 unit root(s) from 3 variables and rank 1" in fig.footnote
    assert len(vecm.relations_plot().panels) == 1
    tabs = dict(sections(vecm))
    assert {"Relations", "Relations data", "Roots", "Roots graph"} <= set(tabs)
    assert list(tabs["Roots"].to_frame().columns) == ["real", "imaginary", "modulus"]
    var = VAR(["x", "y"], lags=2).fit(Dataset.from_frame(data.frame.diff().dropna()))
    assert "stable: every root inside" in var.roots_plot().footnote and "Roots graph" in dict(sections(var))


URCA_RECURSIVE = {  # urca 1.3.3 ca.jo(data[1:end, ], K = 3, spec = "transitory")@lambda
    "ci": {"1975Q3": [0.374616877334076, 0.113091366304098, 0.0142709149776902],
           "1988Q1": [0.354913318569858, 0.0226216682731184, 0.0184779638647869],
           "2024Q4": [0.316756782855139, 0.0307843370363063, 0.00339266006589753]},
    "co": {"1975Q3": [0.374614652794214, 0.112800755177921, 0.00691674083981873],
           "1988Q1": [0.354874073237591, 0.0215867344610671, 0.0168422566936118],
           "2024Q4": [0.316707978005589, 0.0307299053661519, 0.00183756421364784]},
}


@pytest.mark.parametrize("case", ["ci", "co"])
def test_recursive_eigenvalues_against_urca(data, case):
    """Hansen-Johansen recursive eigenvalues (everything re-estimated) against urca on each truncated sample, to
    1e-9; the R representation (short run at the full-sample estimates) ends where the Z one does."""
    fit = VECM(["x", "y", "z"], lags=2, rank=1, deterministic=case).fit(data)
    Z = fit.recursive_eigenvalues().frame
    assert str(Z.index[0]) == "1975Q3" and str(Z.index[-1]) == "2024Q4"  # a third of the way in, to the end
    for end, values in URCA_RECURSIVE[case].items():
        assert np.allclose(Z.loc[pd.Period(end, freq="Q")].to_numpy(), values, atol=1e-9), end
    R = fit.recursive_eigenvalues(base="R").frame
    assert np.allclose(R.iloc[-1], Z.iloc[-1], atol=1e-12) and not np.allclose(R.iloc[0], Z.iloc[0])
    later = fit.recursive_eigenvalues(start="2000Q1").frame
    assert str(later.index[0]) == "2000Q1" and np.allclose(later.to_numpy(), Z.loc[later.index].to_numpy())
    figure = fit.recursive_eigenvalues_plot()
    assert len(figure.panels) == 1 and "Recursive eigenvalues" in dict(sections(fit))


def test_recursive_eigenvalues_with_seasons_and_exogenous_terms(data):
    frame = data.frame.assign(w=np.random.default_rng(4).normal(size=len(data.frame)))
    fit = VECM(["x", "y", "z"], lags=1, rank=1, deterministic="colo", seasons=4, exog=["w"]).fit(Dataset.from_frame(frame))
    Z = fit.recursive_eigenvalues().frame
    assert Z.shape[1] == 3 and np.all((Z.to_numpy() >= -1e-12) & (Z.to_numpy() < 1))
    with pytest.raises(Exception, match="base is 'Z'"):
        fit.recursive_eigenvalues(base="X")


def _two(T=200, seed=0, broken=False):
    rng = np.random.default_rng(seed)
    y = np.zeros((T, 2))
    for t in range(1, T):
        alpha = np.array([-0.5, 0.0]) if not (broken and t > T // 2) else np.array([0.0, 0.0])  # cointegration lost mid-sample
        y[t] = y[t - 1] + alpha * (y[t - 1, 0] - y[t - 1, 1]) + rng.normal(size=2)
    return Dataset.from_frame(pd.DataFrame(y, columns=["a", "b"], index=pd.period_range("1900Q1", periods=T, freq="Q", name="time")))


def test_fluctuation_test():
    """The statistic from the R-representation recursive eigenvalues by hand; a bootstrap p-value, reproducible; broken
    samples with smaller p-values than stable ones (size and power over 200 samples are in validation.md)."""
    fit = VECM(["a", "b"], lags=1, rank=1).fit(_two())
    test = fit.fluctuation_test(draws=99, seed=3)
    lam = fit.recursive_eigenvalues(base="R").frame["lambda1"]
    xi = np.log(lam / (1 - lam)).to_numpy()
    T = len(np.asarray(fit.results.resid))
    n = np.arange(T - len(xi) + 1, T + 1)
    by_hand = np.max(np.sqrt(T) * (n / T) * np.abs(xi - xi[-1]))
    assert test.statistic == pytest.approx(by_hand, rel=1e-10)
    assert test.details["by eigenvalue"].loc["eigenvalue 1", "statistic"] == pytest.approx(by_hand, rel=1e-10)
    assert test.p_value == fit.fluctuation_test(draws=99, seed=3).p_value and 0 < test.p_value <= 1
    assert test.p_value > 0.05 and test.critical_values["5%"] > test.statistic
    # a break: smaller p-values than stable samples, over several (power is moderate: 54% at 5% in validation.md)
    p = {b: [VECM(["a", "b"], lags=1, rank=1).fit(_two(seed=s, broken=b)).fluctuation_test(draws=49, seed=s).p_value for s in range(6)]
         for b in (False, True)}
    assert np.mean(p[True]) < np.mean(p[False]) and min(p[True]) < 0.1
    figure = fit.fluctuation_plot(draws=19)
    assert len(figure.panels) == 2 and "bootstrap 5% critical value" in figure.footnote
