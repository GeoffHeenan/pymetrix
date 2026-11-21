"""Factor-augmented VARs, checked independently: the factors normalised (F'F / T = I) and spanning the same space as
scikit-learn's or numpy's principal components; the loadings by least squares by hand; a panel series' response equal
to its loadings times the VAR's responses, in its own units; Bernanke-Boivin-Eliasz's purge leaving the factors with
no part explained by the observed variable beyond the slow series' factors; and a simulated series' true response
tracked. Also: Bai-Ng's criteria by hand, lags by criterion as select_order, the bootstrap of the whole procedure
(reproducible, wider than the VAR's alone, covering the true policy response), the VAR-only bands taken draw by draw,
the common components' variance decomposition by hand, and the dialog."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, favar
from pymetrix.errors import SpecificationError


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(0)
    T, N = 300, 60
    A = np.array([[0.7, 0.0, 0.1], [0.1, 0.5, 0.0], [0.2, 0.1, 0.6]])
    s = np.zeros((T, 3))
    for t in range(1, T):
        s[t] = A @ s[t - 1] + rng.normal(size=3) * np.array([1, 1, 0.5])
    Lam = rng.normal(size=(N, 2))
    g = np.where(np.arange(N) < 30, 0.0, rng.normal(size=N) * 0.5)  # the first 30 series do not respond to r within the period
    X = s[:, :2] @ Lam.T + s[:, 2:3] * g + rng.normal(size=(T, N)) * 0.5
    cols = {f"x{i}": X[:, i] * (i + 1) for i in range(N)}  # different units
    cols["r"] = s[:, 2]
    data = Dataset.from_frame(pd.DataFrame(cols, index=pd.period_range("1950-01", periods=T, freq="M")))
    return data, A, Lam, g, X


NAMES = [f"x{i}" for i in range(60)]


def test_factors_loadings_and_responses(panel):
    data, *_ = panel
    fv = favar(data, NAMES, ["r"], factors=2, lags=1)
    F = fv.factors.to_numpy()
    assert F.T @ F / len(F) == pytest.approx(np.eye(2), abs=1e-10)
    Z = ((data.frame[NAMES] - data.frame[NAMES].mean()) / data.frame[NAMES].std(ddof=0)).to_numpy()
    U = np.linalg.svd(Z, full_matrices=False)[0][:, :2]
    assert F @ np.linalg.lstsq(F, U, rcond=None)[0] == pytest.approx(U, abs=1e-10)  # the same space
    R = np.column_stack([np.ones(len(F)), F, data.frame["r"].to_numpy()])
    lam = np.linalg.lstsq(R, Z[:, 5], rcond=None)[0][1:]
    assert fv.loadings.loc["x5"].to_numpy() == pytest.approx(lam, abs=1e-10)
    base = fv.var.irf(12, draws=0).values
    resp = fv.irf(["x5"], 12).values[:, 0, :]
    assert resp == pytest.approx(np.einsum("k,hkj->hj", lam * data.frame["x5"].std(ddof=0), base), abs=1e-10)


def test_the_slow_purge_and_a_true_response(panel):
    data, A, Lam, g, _ = panel
    slow = NAMES[:30]
    fv = favar(data, NAMES, ["r"], factors=2, lags=1, slow=slow)
    Zs = ((data.frame[slow] - data.frame[slow].mean()) / data.frame[slow].std(ddof=0)).to_numpy()
    Fs = np.sqrt(len(Zs)) * np.linalg.svd(Zs, full_matrices=False)[0][:, :2]
    F, r = fv.factors.to_numpy(), data.frame["r"].to_numpy()
    coef = np.linalg.lstsq(np.column_stack([Fs, r]), F, rcond=None)[0]
    assert coef[2] == pytest.approx([0, 0], abs=1e-10)  # nothing left that the observed variable explains beyond the slow factors
    Phi = [np.eye(3)]
    for _ in range(12):
        Phi.append(A @ Phi[-1])
    true = np.array([(Lam[40] @ P[:2, 2] + g[40] * P[2, 2]) * 0.5 * 41 for P in Phi])  # x40 in its units (scaled by 41)
    est = fv.irf(["x40"], 12).values[:, 0, 2]
    assert np.corrcoef(true, est)[0, 1] > 0.85
    with pytest.raises(SpecificationError, match="factors"):
        favar(data, NAMES[:3], ["r"], factors=3)


def test_bai_ng_criteria_by_hand():
    """ICp1, ICp2, ICp3 from their definitions; on a panel of three factors with equal idiosyncratic noise each picks three."""
    from pymetrix.favar import factor_criteria

    rng = np.random.default_rng(5)
    T, N = 200, 80
    Z = rng.normal(size=(T, 3)) @ rng.normal(size=(3, N)) + rng.normal(size=(T, N))
    Z = (Z - Z.mean(0)) / Z.std(0)
    crit = factor_criteria(Z, 6)
    for k in range(1, 7):
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        V = ((Z - U[:, :k] * s[:k] @ Vt[:k]) ** 2).sum() / (N * T)
        assert crit.loc[k, "ICp1"] == pytest.approx(np.log(V) + k * (N + T) / (N * T) * np.log(N * T / (N + T)), abs=1e-10)
        assert crit.loc[k, "ICp2"] == pytest.approx(np.log(V) + k * (N + T) / (N * T) * np.log(min(N, T)), abs=1e-10)
        assert crit.loc[k, "ICp3"] == pytest.approx(np.log(V) + k * np.log(min(N, T)) / min(N, T), abs=1e-10)
    assert [int(crit[c].idxmin()) for c in ("ICp1", "ICp2", "ICp3")] == [3, 3, 3]


def test_chosen_factors_and_lags(panel):
    from pymetrix import VAR

    data, *_ = panel
    fv = favar(data, NAMES, ["r"], factors="icp2", lags="bic", max_lags=4)
    k = len(fv.factor_names)
    assert fv.factor_selection is not None and f"Chosen by ICp2: {k} factors" in fv.factor_selection.notes[1]
    block = Dataset.from_frame(pd.concat([fv.factors, data.frame[["r"]]], axis=1))
    assert fv.var.lags == VAR([*fv.factor_names, "r"], lags="bic", max_lags=4).select_order(block).best["bic"]
    assert "Number of factors" in dict(__import__("pymetrix.views", fromlist=["sections"]).sections(fv))


def test_var_ols_and_the_bootstraps(panel):
    from pymetrix.favar import _var_ols

    data, A, Lam, g, _ = panel
    fv = favar(data, NAMES, ["r"], factors=2, lags=2, slow=NAMES[:30])
    R = np.column_stack([fv.factors.to_numpy(), data.frame["r"].to_numpy()])
    coefs, sigma = _var_ols(R, 2)
    assert np.allclose(coefs, fv.var.coefs, atol=1e-10) and np.allclose(sigma, fv.var.estimate.sigma, atol=1e-10)
    # the VAR-only bands: each draw through the loadings, then quantiles (not the VAR's bands through the loadings)
    alone = fv.irf(["x40", "x5"], 12, draws=100, seed=2, bootstrap="var")
    L = fv.loadings.loc[["x40", "x5"]].to_numpy() * fv.scale[["x40", "x5"]].to_numpy()[:, None]
    sims = np.stack([np.einsum("sk,hkj->hsj", L, fv.var._responses(a, s, 12, True, False)) for a, s in fv.var._bootstrap(100, 2)])
    assert np.allclose(alone.lower, np.quantile(sims, 0.025, axis=0)) and np.all(alone.lower <= alone.upper)
    whole = fv.irf(["x40", "x5"], 12, draws=150, seed=2)
    assert np.allclose(whole.lower, fv.irf(["x40", "x5"], 12, draws=150, seed=2).lower)  # reproducible
    assert np.all(whole.lower <= whole.upper) and "whole procedure" in whole.identification
    assert np.median(whole.upper - whole.lower) > np.median(alone.upper - alone.lower)  # factor uncertainty too
    Phi = [np.eye(3)]
    for _ in range(12):
        Phi.append(A @ Phi[-1])
    true = np.array([(Lam[40] @ P[:2, 2] + g[40] * P[2, 2]) * 0.5 * 41 for P in Phi])
    inside = (whole.lower[:, 0, 2] <= true) & (true <= whole.upper[:, 0, 2])  # the policy shock: invariant to factor rotation
    assert inside.mean() >= 0.75


def test_variance_decomposition_of_panel_series(panel):
    from pymetrix.var import _ma

    data, *_ = panel
    fv = favar(data, NAMES, ["r"], factors=2, lags=1)
    d = fv.fevd(["x40", "x3"], 24)
    theta = np.einsum("k,hkj->hj", fv.loadings.loc["x40"].to_numpy(), _ma(fv.var.coefs, 23) @ fv.var.impact())
    by_hand = (theta**2).sum(axis=0) / (theta**2).sum()
    assert np.allclose(d.values[-1, 0], by_hand) and np.allclose(d.values.sum(axis=2), 1)


def test_the_var_points_to_favar_and_the_dialog(panel):
    from pymetrix import VAR
    from pymetrix.commands import FitFAVAR, Session

    data, *_ = panel
    with pytest.raises(SpecificationError, match="favar"):
        VAR(["r"], lags=1, factors=object()).fit(data)
    session = Session()
    session.objects["data"] = data
    session.run(FitFAVAR(", ".join(NAMES), "r", factors="icp2", lags="2", slow=", ".join(NAMES[:30])))
    script = session.script()
    assert "fv = favar(data, ['x0', " in script and "['r'], factors='icp2', slow=['x0', " in script
    ns = {"data": data}
    exec(script, ns)
    assert np.allclose(ns["fv"].loadings.to_numpy(), session["fv"].loadings.to_numpy())


def test_small_panels_and_few_slow_series(panel):
    """Bai-Ng's search stops at what the panel allows; with fewer slow series than factors, the purge uses as many of
    their components as they have, and still leaves nothing the observed variable explains beyond them."""
    data, *_ = panel
    small = favar(data, NAMES[:5], ["r"], factors="icp2")  # max_factors 8 > 4 the panel allows
    assert len(small.factor_selection.to_frame()) == 4
    fv = favar(data, NAMES, ["r"], factors=3, lags=1, slow=NAMES[:2])
    Zs = ((data.frame[NAMES[:2]] - data.frame[NAMES[:2]].mean()) / data.frame[NAMES[:2]].std(ddof=0)).to_numpy()
    Fs = np.sqrt(len(Zs)) * np.linalg.svd(Zs, full_matrices=False)[0][:, :2]
    coef = np.linalg.lstsq(np.column_stack([Fs, data.frame["r"].to_numpy()]), fv.factors.to_numpy(), rcond=None)[0]
    assert coef[2] == pytest.approx([0, 0, 0], abs=1e-10)
