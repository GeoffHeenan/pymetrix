"""GMM on moment functions written in Python, checked against statsmodels' generic GMM and closed forms.

- A consumption Euler equation, E[(beta g^-gamma R - 1) z] = 0 with three instruments: one-step (W = I) estimates and
  sandwich standard errors, two-step and iterated estimates, standard errors and J, against statsmodels' GMM
  (statsmodels.sandbox.regression.gmm.GMM, uncentred 'cov' weights), to 1e-6; the continuously updated estimator
  reaches an objective no higher than at statsmodels' estimates; Newey-West weighting agrees on the estimates and J,
  and on the standard errors to 1e-3 (statsmodels takes them from the last step's weight matrix, not S at the
  estimates).
- Least squares written as moments: exactly identified, so the estimates are OLS's, the standard errors White's
  (HC0) and J is zero."""

import warnings

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, MomentGMM
from pymetrix.errors import SpecificationError


def _euler_data(n=600):
    rng = np.random.default_rng(0)
    lc = np.zeros(n)
    for t in range(1, n):
        lc[t] = 0.01 + 0.5 * lc[t - 1] + rng.normal(0, 0.02)
    cg = np.exp(lc)
    r = (1 / 0.97) * cg ** 2 * np.exp(rng.normal(0, 0.05, n) - 0.05 ** 2 / 2)  # E[0.97 cg^-2 r | past] = 1
    return Dataset.from_frame(pd.DataFrame({"cg": cg, "r": r}, index=pd.period_range("1900Q1", periods=n, freq="Q", name="time")))


def euler(p, f):
    u = p["beta"] * f.cg ** -p["gamma"] * f.r - 1
    return pd.DataFrame({"u": u, "u_cg1": u * f.cg.shift(1), "u_r1": u * f.r.shift(1)})


START = {"beta": 0.99, "gamma": 1.0}


@pytest.fixture(scope="module")
def reference():
    from statsmodels.sandbox.regression.gmm import GMM

    data = _euler_data()
    f = data.frame.iloc[1:]
    Z = np.column_stack([np.ones(len(f)), data.frame.cg.shift(1).iloc[1:], data.frame.r.shift(1).iloc[1:]])

    class Euler(GMM):
        def momcond(self, params):
            u = params[0] * self.endog[:, 0] ** -params[1] * self.endog[:, 1] - 1
            return u[:, None] * self.instrument

    model = Euler(f[["cg", "r"]].to_numpy(), np.ones((len(f), 2)), Z, k_moms=3, k_params=2)

    def fit(maxiter, method="cov", **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.fit([0.99, 1.0], maxiter=maxiter, inv_weights=np.eye(3), weights_method=method, optim_method="bfgs",
                             optim_args={"gtol": 1e-12, "disp": 0}, **kw)

    return data, fit


def test_one_and_two_step_and_iterated_match_statsmodels(reference):
    data, fit = reference
    one = MomentGMM(euler, START, weights="identity").fit(data)
    theirs = fit(1, wargs={"centered": False}, has_optimal_weights=False)
    assert np.allclose(one.params.values, theirs.params, rtol=1e-6) and np.allclose(one.params.se, theirs.bse, rtol=1e-5)
    for steps, maxiter in (("two", 2), ("iterated", 50)):
        ours, theirs = MomentGMM(euler, START, steps=steps).fit(data), fit(maxiter, wargs={"centered": False})
        assert np.allclose(ours.params.values, theirs.params, rtol=1e-6), steps
        assert np.allclose(ours.params.se, theirs.bse, rtol=1e-5), steps
        assert np.isclose(ours.J, theirs.jval, rtol=1e-6), steps
    two = MomentGMM(euler, START).fit(data)
    assert two.stats["J_df"] == 1 and 0 < two.stats["J_pvalue"] < 1 and two.stats["nobs"] == len(data.frame) - 1
    assert list(two.contributions.columns) == ["u", "u_cg1", "u_r1"] and two.provenance.nobs == 599
    assert "GMM on moment conditions: 3 moments for 2 parameters" in str(two.summary())


def test_cue_and_newey_west(reference):
    data, fit = reference
    ours, theirs = MomentGMM(euler, START, steps="cue").fit(data), fit("cue", wargs={"centered": False})
    assert np.allclose(ours.params.values, theirs.params, atol=1e-4)
    g = lambda th: euler({"beta": th[0], "gamma": th[1]}, data.frame).to_numpy()[1:]  # noqa: E731

    def objective(th):
        G = g(th)
        return len(G) * G.mean(0) @ np.linalg.solve(G.T @ G / len(G), G.mean(0))

    assert objective(ours.params.values) <= objective(np.asarray(theirs.params)) + 1e-12
    hac, theirs = MomentGMM(euler, START, cov="hac", hac_lags=4).fit(data), fit(2, method="hac", wargs={"maxlag": 4})
    assert np.allclose(hac.params.values, theirs.params, rtol=1e-6) and np.isclose(hac.J, theirs.jval, rtol=1e-6)
    assert np.allclose(hac.params.se, theirs.bse, rtol=1e-3)


def test_least_squares_as_moments_is_ols_with_white_errors():
    rng = np.random.default_rng(3)
    n = 400
    x = rng.normal(size=n)
    y = 1 + 2 * x + rng.normal(size=n) * (1 + np.abs(x))
    frame = pd.DataFrame({"y": y, "x": x})
    fit = MomentGMM(lambda p, f: np.column_stack([f.y - p["a"] - p["b"] * f.x, (f.y - p["a"] - p["b"] * f.x) * f.x]),
                    start={"a": 0.0, "b": 0.0}).fit(frame)
    X = np.column_stack([np.ones(n), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    bread = np.linalg.inv(X.T @ X)
    white = bread @ (X.T * e ** 2) @ X @ bread
    assert np.allclose(fit.params.values, b, atol=1e-8)
    assert np.allclose(fit.params.se, np.sqrt(np.diag(white)), rtol=1e-5)
    assert abs(fit.J) < 1e-12 and fit.stats["J_df"] == 0


def test_span_bounds_and_refusals(reference):
    data, _ = reference
    short = MomentGMM(euler, START).fit(data, span=("1950Q1", "1999Q4"))
    assert short.stats["nobs"] == 200 and short.provenance.sample == ("1950Q1", "1999Q4")
    bounded = MomentGMM(euler, START, bounds={"gamma": (0.0, 2.0)}, weights="identity").fit(data)
    assert bounded.params.as_dict()["gamma"] <= 2.0 + 1e-12
    for kwargs, message in (({"moments": 1}, "is a function"), ({"start": {}}, "starting values"),
                            ({"bounds": {"delta": (0, 1)}}, "not parameters"), ({"steps": "cue", "weights": "identity"}, "weights optimally")):
        with pytest.raises(SpecificationError, match=message):
            MomentGMM(**{"moments": euler, "start": START, **kwargs})
    with pytest.raises(SpecificationError, match="underidentified"):
        MomentGMM(lambda p, f: (f.r - p["a"]).to_numpy(), start={"a": 1.0, "b": 0.0}).fit(data)
    with pytest.raises(SpecificationError, match="one row per row"):
        MomentGMM(lambda p, f: np.zeros((3, 2)), start={"a": 1.0}).fit(data)
    with pytest.raises(SpecificationError, match="cov='white' or 'hac'"):
        MomentGMM(euler, START, cov="classic").fit(data)
    with pytest.raises(SpecificationError, match="HAC weighting needs a time series"):
        MomentGMM(lambda p, f: np.column_stack([f.y - p["a"], (f.y - p["a"]) ** 3]), start={"a": 0.0}, cov="hac").fit(
            pd.DataFrame({"y": np.random.default_rng(1).normal(size=50)}))
