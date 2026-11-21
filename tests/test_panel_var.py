"""Panel VARs, checked independently: forward orthogonal deviations against their formula by a plain loop, and
keeping i.i.d. errors i.i.d.; the within estimator and its clustered standard errors against linearmodels' PanelOLS
with entity effects; the just-identified GMM estimate and its clustered errors against linearmodels' IV2SLS on data
transformed by hand; known coefficients recovered by GMM (and the within estimator's Nickell bias seen); Hansen's J
and the Granger tests by hand; the impulse responses' first steps from the coefficients."""

import importlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pymetrix import VAR, Dataset
from pymetrix.errors import SpecificationError

pv = importlib.import_module("pymetrix.panelvar")
A_TRUE = np.array([[0.5, 0.2], [-0.1, 0.4]])


def _panel(N=300, T=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N):
        mu = rng.normal(0, 1, 2)
        y = mu + rng.normal(0, 1, 2)
        for t in range(-20, T):  # a burn-in, so each entity starts from its own stationary distribution
            y = (np.eye(2) - A_TRUE) @ mu + A_TRUE @ y + rng.normal(0, 1, 2)
            if t >= 0:
                rows.append({"firm": i, "year": 2000 + t, "x": y[0], "z": y[1]})
    return Dataset.from_frame(pd.DataFrame(rows), entity="firm", time="year", freq="Y")


def test_forward_orthogonal_deviations():
    x = np.random.default_rng(1).normal(size=(6, 2))
    by_hand = []
    for t in range(5):
        later = x[t + 1:]
        by_hand.append(np.sqrt(len(later) / (len(later) + 1)) * (x[t] - later.mean(axis=0)))
    assert pv.forward_orthogonal(x) == pytest.approx(np.array(by_hand), rel=1e-12)
    e = np.random.default_rng(2).normal(size=(20000, 8, 1))  # i.i.d. errors stay i.i.d. with the same variance
    f = np.stack([pv.forward_orthogonal(block)[:, 0] for block in e])
    assert np.cov(f.T) == pytest.approx(np.eye(7), abs=0.03)


def _lagged_frame(data):
    frame = data.frame.copy()
    for v in ("x", "z"):
        frame[f"{v}_1"] = data.lagged(v, -1)
    frame = frame.dropna(subset=["x_1", "z_1"])
    frame.index = pd.MultiIndex.from_arrays([frame.index.get_level_values(0), frame.index.get_level_values(1).year], names=["firm", "year"])
    return frame  # linearmodels takes numeric years


def test_within_against_panelols():
    lm = pytest.importorskip("linearmodels")
    data = _panel(N=150)
    fit = VAR(["x", "z"], lags=1, by="firm").fit(data, transform="within")
    frame = _lagged_frame(data)
    for j, eq in enumerate(("x", "z")):
        ref = lm.PanelOLS(frame[eq], frame[["x_1", "z_1"]], entity_effects=True).fit(
            cov_type="clustered", cluster_entity=True, group_debias=False, debiased=False, auto_df=False, count_effects=False)
        assert fit.B[:, j] == pytest.approx(ref.params.to_numpy(), rel=1e-10)
        assert fit.se[:, j] == pytest.approx(ref.std_errors.to_numpy(), rel=1e-8)  # no small-sample factors on either side


def test_just_identified_gmm_against_iv2sls():
    lm = pytest.importorskip("linearmodels")
    data = _panel(N=150)
    fit = VAR(["x", "z"], lags=1, by="firm").fit(data)
    Ys, Xs, Zs, groups = [], [], [], []
    frame = _lagged_frame(data)
    for key, g in frame.groupby(level="firm"):
        y = g[["x", "z"]].to_numpy()
        x = g[["x_1", "z_1"]].to_numpy()
        Ys.append(pv.forward_orthogonal(y))
        Xs.append(pv.forward_orthogonal(x))
        Zs.append(x[:-1])
        groups += [key] * (len(g) - 1)
    Y, X, Z = np.vstack(Ys), np.vstack(Xs), np.vstack(Zs)
    for j in range(2):
        ref = lm.IV2SLS(pd.Series(Y[:, j]), None, pd.DataFrame(X, columns=["x1", "z1"]), pd.DataFrame(Z, columns=["zx", "zz"])).fit(
            cov_type="clustered", clusters=pd.Series(groups), debiased=False)
        assert fit.B[:, j] == pytest.approx(ref.params.to_numpy(), rel=1e-10)
        assert fit.se[:, j] == pytest.approx(ref.std_errors.to_numpy(), rel=0.02)  # linearmodels adds a small-sample factor


def test_gmm_recovers_the_coefficients_and_within_is_biased():
    data = _panel(N=400, T=10, seed=3)
    gmm = VAR(["x", "z"], lags=1, by="firm").fit(data)
    within = VAR(["x", "z"], lags=1, by="firm").fit(data, transform="within")
    assert np.all(np.abs(gmm.coefs[0] - A_TRUE) < 3.5 * gmm.se.T)
    assert within.coefs[0][0, 0] < A_TRUE[0, 0] - 0.08 and within.coefs[0][1, 1] < A_TRUE[1, 1] - 0.08  # Nickell bias
    assert gmm.is_stable and gmm.entities == 400 and gmm.nobs == 400 * 8


def test_hansen_j_granger_and_responses():
    data = _panel(N=300, seed=4)
    over = VAR(["x", "z"], lags=1, by="firm").fit(data, instrument_lags=3)
    assert over.Jdf == 2 * (6 - 2) and over.stats["J_p"] > 0.01  # a correct model's restrictions are not rejected
    fit = VAR(["x", "z"], lags=1, by="firm").fit(data)
    g = fit.granger().to_frame()
    b, V = fit.B[1, 0], fit.cov[1, 1]  # z's lag in the x equation
    assert g.loc["z -> x", "chi-squared"] == pytest.approx(b**2 / V)
    assert g.loc["z -> x", "p-value"] == pytest.approx(stats.chi2.sf(b**2 / V, 1))
    irf = fit.irf(4, draws=50)
    P = np.linalg.cholesky(fit.sigma)
    assert irf.values[0] == pytest.approx(P) and irf.values[1] == pytest.approx(fit.coefs[0] @ P)
    assert np.all(irf.lower <= irf.values + 1e-12) and np.all(irf.values <= irf.upper + 1e-12)
    assert fit.fevd(5).values[-1].sum(axis=1) == pytest.approx([1, 1])


def test_refusals_and_the_dialog():
    data = _panel(N=40)
    with pytest.raises(SpecificationError, match="entity axis"):
        VAR(["x", "z"], lags=1, by="country").fit(data)
    with pytest.raises(SpecificationError, match="instrument_lags"):
        VAR(["x", "z"], lags=2, by="firm").fit(data, instrument_lags=1)
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS
    from pymetrix.views import sections

    s = Session()
    s.objects["data"] = data
    s.run(COMMANDS["FitPanelVAR"].cls("x, z", "firm", lags=1))
    assert s.history[-1].code == "pvar = VAR(['x', 'z'], lags=1, by='firm').fit(data)"
    assert [t for t, _ in sections(s.objects["pvar"])][:2] == ["Summary", "Granger causality"]


def _panel2(N=500, T=12, seed=3):
    """A panel VAR(2)."""
    A1, A2 = np.array([[0.4, 0.1], [0.0, 0.3]]), np.array([[0.3, 0.0], [0.1, 0.3]])
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N):
        mu = rng.normal(0, 1, 2)
        y1 = y2 = mu
        for t in range(-30, T):
            y = (np.eye(2) - A1 - A2) @ mu + A1 @ y1 + A2 @ y2 + rng.normal(0, 1, 2)
            y2, y1 = y1, y
            if t >= 0:
                rows.append({"firm": i, "year": 2000 + t, "x": y[0], "z": y[1]})
    return Dataset.from_frame(pd.DataFrame(rows), entity="firm", time="year", freq="Y")


def test_lag_selection_by_andrews_lu():
    """MBIC, MAIC and MQIC from Hansen's J by their formulas (each J that of a fit with the same instruments), CD by
    hand; the true lag length found in a VAR(1) and a VAR(2) panel; lags="bic" fits the chosen lags."""
    data = _panel(N=500)
    spec = VAR(["x", "z"], lags="bic", max_lags=3, by="firm")
    table = spec.select_order(data).table.to_frame()
    table.index = table.index.astype(int)  # the table keeps its row labels as text
    K, q = 2, 4
    for p in (1, 2, 3):
        fit = VAR(["x", "z"], lags=p, by="firm").fit(data, instrument_lags=q)
        J, n = fit.J, fit.nobs
        assert table.loc[p, "J"] == pytest.approx(J, rel=1e-10) and table.loc[p, "nobs"] == n
        assert table.loc[p, "J p-value"] == pytest.approx(stats.chi2.sf(J, fit.Jdf), rel=1e-10)
        assert table.loc[p, "MBIC"] == pytest.approx(J - (q - p) * K**2 * np.log(n), rel=1e-10)
        assert table.loc[p, "MAIC"] == pytest.approx(J - 2 * (q - p) * K**2, rel=1e-10)
        assert table.loc[p, "MQIC"] == pytest.approx(J - 2.1 * (q - p) * K**2 * np.log(np.log(n)), rel=1e-10)
        blocks, _ = pv._entity_blocks(VAR(["x", "z"], lags=p, by="firm"), data, p, q, "fod", None)
        Y = np.vstack([b[1] for b in blocks])
        cd = 1 - np.linalg.det(fit.sigma) / np.linalg.det(np.cov(Y.T, bias=True))
        assert table.loc[p, "CD"] == pytest.approx(cd, rel=1e-10)
    assert spec.select_order(data).best == {"bic": 1, "aic": 1, "hq": 1}
    assert spec.fit(data).lags == 1
    two = _panel2()
    assert VAR(["x", "z"], lags="bic", max_lags=4, by="firm").select_order(two).best["bic"] == 2
    assert VAR(["x", "z"], lags="bic", max_lags=4, by="firm").fit(two).lags == 2
    within = pv.select_order(spec, data, 3, transform="within")
    assert list(within.table.to_frame().columns) == ["CD", "BIC", "AIC", "HQ", "nobs"]
    with pytest.raises(SpecificationError, match="'bic', 'aic' or 'hq'"):
        VAR(["x", "z"], lags="fpe", by="firm").fit(data)


def test_lag_choice_in_the_dialogs():
    from pymetrix.commands import Session, VARLagOrder
    from pymetrix.commands.registry import COMMANDS

    data = _panel(N=200)
    s = Session()
    s.objects["data"] = data
    s.run(COMMANDS["FitPanelVAR"].cls("x, z", "firm", choose_lags="bic", max_lags=3))
    assert s.history[-1].code == "pvar = VAR(['x', 'z'], lags='bic', max_lags=3, by='firm').fit(data)"
    s.run(VARLagOrder("x, z", max_lags=3, by="firm"))
    assert "MBIC" in str(s.objects["lag_order"])
