"""Bootstrap bands for VECM impulse responses.

- The residual bootstrap rebuilds each series from the VECM: with the actual residuals it gives back the data, in each
  of the five deterministic cases, with and without seasonal dummies and an exogenous series; and the deterministic
  part it carries over is a function of time alone (constant, trend, dummies and the exogenous series explain it
  exactly), so the short-run and error-correction terms are taken out correctly.
- The bootstrap spread of the levels-VAR coefficients against statsmodels' asymptotic standard errors
  (VECMResults.cov_var_repr, Luetkepohl's delta method), within 10% each (1,000 draws, 400 observations).
- Bands: contain the estimates, narrow with coverage, reproducible by seed; cumulative and reduced-form responses.
- Historical decompositions of VECMs: each shock's contribution against the error-correction recursion from
  statsmodels' alpha, beta and gamma, in each deterministic case, to 1e-8; the parts add up to the data."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import VECM, Cholesky, Dataset


def _data(T=400, seed=0):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=T))
    y = 0.8 * x + rng.normal(size=T) * 0.5 + 0.01 * np.arange(T)
    z = 0.5 * x + np.cumsum(rng.normal(size=T)) * 0.5
    w = rng.normal(size=T)
    return Dataset.from_frame(pd.DataFrame({"x": x, "y": y, "z": z, "w": w},
                                           index=pd.period_range("1900Q1", periods=T, freq="Q", name="time")))


@pytest.mark.parametrize("deterministic", ["n", "ci", "co", "coli", "colo"])
@pytest.mark.parametrize("extra", [{}, {"seasons": 4, "exog": ["w"]}], ids=["plain", "seasons-exog"])
def test_the_bootstrap_rebuilds_the_data(deterministic, extra):
    data = _data(240)
    fit = VECM(["x", "y", "z"], lags=2, rank=1, deterministic=deterministic, **extra).fit(data)
    det = fit._deterministic()
    rebuilt = fit._simulate(np.asarray(fit.results.resid), det)
    assert np.allclose(rebuilt, np.asarray(fit.results.model.endog), atol=1e-10)
    n = len(det)
    terms = [np.ones(n), np.arange(n)]
    if extra:
        season = (np.arange(n) + 240 - n) % 4
        terms += [(season == k).astype(float) for k in range(3)] + [data.frame.w.to_numpy()[-n:]]
    X = np.column_stack(terms)
    assert np.allclose(det, X @ np.linalg.lstsq(X, det, rcond=None)[0], atol=1e-10)


def test_the_spread_against_statsmodels_asymptotics():
    fit = VECM(["x", "y", "z"], lags=2, rank=1, deterministic="co").fit(_data())
    se = np.sqrt(np.diag(np.asarray(fit.results.cov_var_repr)))  # vec of [A1 A2 A3], column by column
    sd = np.stack([A for A, _ in fit._bootstrap(1000, 1)]).std(axis=0)  # (lag, row, column)
    K, P = 3, sd.shape[0]
    ours = np.array([sd[col // K, i, col % K] for col in range(K * P) for i in range(K)])
    assert np.allclose(ours, se, rtol=0.10)


def test_the_bands():
    fit = VECM(["x", "y", "z"], lags=1, rank=1).fit(_data(200)).identify(Cholesky(["z", "y", "x"]))
    irf = fit.irf(12, draws=200, seed=3)
    assert irf.lower.shape == irf.values.shape and np.all(irf.lower <= irf.upper)
    inside = (irf.lower <= irf.values + 1e-12) & (irf.values <= irf.upper + 1e-12)
    assert inside.mean() > 0.95  # percentile bands around the estimates (not always, by their skew)
    narrow = fit.irf(12, draws=200, seed=3, alpha=0.32)
    assert np.all(narrow.upper - narrow.lower <= irf.upper - irf.lower + 1e-12)
    assert np.allclose(fit.irf(12, draws=200, seed=3).lower, irf.lower)
    assert "residual bootstrap, 200 draws" in irf.identification and irf.coverage == 0.95
    cumulative = fit.irf(12, draws=50, cumulative=True, orthogonal=False)
    assert np.allclose(cumulative.values, np.cumsum(fit.irf(12, orthogonal=False).values, axis=0))
    assert cumulative.lower is not None and fit.irf(12).lower is None


@pytest.mark.parametrize("deterministic", ["n", "ci", "co", "coli", "colo"])
def test_historical_decomposition_in_error_correction_form(deterministic):
    """Each shock's contribution rebuilt in error-correction form from statsmodels' alpha, beta and gamma (not the
    levels VAR the decomposition uses), from zero; the parts adding up to the data, with seasons and an exogenous
    series too."""
    data = _data(240)
    fit = VECM(["x", "y", "z"], lags=2, rank=1, deterministic=deterministic).fit(data).identify(Cholesky(["z", "x", "y"]))
    hd = fit.historical_decomposition()
    r = fit.results
    alpha, beta, gamma = np.asarray(r.alpha), np.asarray(r.beta), np.asarray(r.gamma)
    U = np.asarray(r.resid)
    P = fit.impact()
    e = np.linalg.solve(P, U.T).T
    T, K, p = len(U), 3, 2
    for j, shock in enumerate(["x", "y", "z"]):
        c = np.zeros((T + p + 1, K))  # zero before the sample
        for t in range(T):
            i = t + p + 1
            short = sum(gamma[:, l * K : (l + 1) * K] @ (c[i - 1 - l] - c[i - 2 - l]) for l in range(p))
            c[i] = c[i - 1] + alpha @ (beta.T @ c[i - 1]) + short + P[:, j] * e[t, j]
        for k, v in enumerate(["x", "y", "z"]):
            assert np.allclose(hd.contributions[v][shock].to_numpy(), c[p + 1 :, k], atol=1e-8)
    Y = np.asarray(r.model.endog)[-T:]
    total = hd.baseline + pd.DataFrame({v: hd.contributions[v].sum(axis=1) for v in ["x", "y", "z"]})
    assert np.allclose(total.to_numpy(), Y, atol=1e-8) and np.allclose(hd.residual.to_numpy(), 0, atol=1e-8)
    seasonal = VECM(["x", "y", "z"], lags=1, rank=1, deterministic=deterministic, seasons=4, exog=["w"]).fit(data)
    hd = seasonal.historical_decomposition()
    total = hd.baseline + pd.DataFrame({v: hd.contributions[v].sum(axis=1) for v in ["x", "y", "z"]})
    assert np.allclose(total.to_numpy(), data.frame[["x", "y", "z"]].to_numpy()[-len(total):], atol=1e-8)


def test_the_vecm_historical_decomposition_dialog():
    from pymetrix.commands import FitVECM, HistoricalDecomposition, Session

    session = Session()
    session.objects["data"] = _data(200)
    session.run(FitVECM("x, y, z", lags=1, rank="1"))
    session.run(HistoricalDecomposition("vecm", ordering="z, y, x"))
    script = session.script()
    assert "hd = vecm.identify(Cholesky(['z', 'y', 'x'])).historical_decomposition()" in script
    ns = {"data": session["data"]}
    exec(script, ns)
    assert np.allclose(ns["hd"].contributions["x"].to_numpy(), session["hd"].contributions["x"].to_numpy())
