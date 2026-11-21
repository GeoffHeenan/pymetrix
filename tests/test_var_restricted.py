"""VARs with zero restrictions (method=Restricted(...)), estimated by feasible GLS.

- With no restrictions: least squares exactly (coefficients, sigma), the standard errors from U'U / T rather than
  the degrees-of-freedom-adjusted sigma (a factor sqrt((T - k) / T)).
- A block-exogenous variable and a single zero against linearmodels' SUR (GLS, unadjusted covariance): estimates and
  standard errors to 1e-10; iterated GLS against its iterate=True.
- Bootstrap draws re-estimated under the same restrictions; refusals; the VAR dialog."""

import warnings

import numpy as np
import pandas as pd
import pytest

from pymetrix import VAR, Dataset, Restricted
from pymetrix.errors import SpecificationError


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(3)
    T = 300
    A = np.array([[0.5, 0.0, 0.0], [0.2, 0.4, 0.1], [0.1, 0.3, 0.6]])
    y = np.zeros((T, 3))
    for t in range(1, T):
        y[t] = 0.1 + A @ y[t - 1] + rng.multivariate_normal(np.zeros(3), [[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1]])
    return pd.DataFrame(y, columns=["oil", "a", "b"], index=pd.period_range("1950Q1", periods=T, freq="Q", name="time"))


RESTRICTION = Restricted(block_exogenous=["oil"], zeros={"b": ["a[-2]"]})


def _sur(frame, iterate):
    SUR = pytest.importorskip("linearmodels.system").SUR  # an optional library: skipped in the core suite

    lags = pd.concat({f"{v}[-{l}]": frame[v].shift(l) for l in (1, 2) for v in frame.columns}, axis=1).iloc[2:]
    lags.insert(0, "const", 1.0)
    Y = frame.iloc[2:]
    eqs = {"oil": {"dependent": Y.oil, "exog": lags[["const", "oil[-1]", "oil[-2]"]]},
           "a": {"dependent": Y.a, "exog": lags},
           "b": {"dependent": Y.b, "exog": lags.drop(columns=["a[-2]"])}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return eqs, SUR(eqs).fit(method="gls", cov_type="unadjusted", iterate=iterate)


def _stack(fit, eqs, values):
    names = fit.regressors
    return np.concatenate([values[[names.index(c) for c in eqs[e]["exog"].columns], j] for j, e in enumerate(eqs)])


def test_no_restrictions_is_least_squares(frame):
    data = Dataset.from_frame(frame)
    ols = VAR(["oil", "a", "b"], lags=2).fit(data)
    gls = VAR(["oil", "a", "b"], lags=2).fit(data, method=Restricted())
    assert np.allclose(gls.estimate.B, ols.estimate.B, atol=1e-12) and np.allclose(gls.estimate.sigma, ols.estimate.sigma, atol=1e-12)
    T, k = len(gls.estimate.U), gls.estimate.B.shape[0]
    assert np.allclose(gls.params.se, ols.params.se * np.sqrt((T - k) / T), rtol=1e-10)


@pytest.mark.parametrize("iterate", [False, True], ids=["fgls", "iterated"])
def test_against_linearmodels_sur(frame, iterate):
    fit = VAR(["oil", "a", "b"], lags=2).fit(Dataset.from_frame(frame), method=Restricted(RESTRICTION.zeros, RESTRICTION.block_exogenous, iterate))
    eqs, ref = _sur(frame, iterate)
    assert np.allclose(_stack(fit, eqs, fit.estimate.B), ref.params.to_numpy(), atol=1e-10)
    se = np.sqrt(np.diag(fit.estimate.cov)).reshape(3, -1).T
    # linearmodels before 7.1 took iterated GLS's standard errors from the first step's residual covariance; from 7.1,
    # as here, from the covariance at the final estimates. Under the older convention they differ by about 2e-4.
    import linearmodels
    from packaging.version import Version

    old = iterate and Version(linearmodels.__version__) < Version("7.1")
    assert np.allclose(_stack(fit, eqs, se), ref.std_errors.to_numpy(), rtol=1e-3 if old else 1e-8)
    assert fit.coefs[0][0, 1] == 0 and fit.coefs[1][2, 1] == 0  # oil leaves out a; b leaves out a[-2]
    assert "5 coefficients restricted to zero" in str(fit.summary())


def test_bootstrap_keeps_the_restrictions_refusals_and_the_dialog(frame):
    data = Dataset.from_frame(frame)
    fit = VAR(["oil", "a", "b"], lags=2).fit(data, method=RESTRICTION)
    for A, _ in fit._bootstrap(5, 1):
        assert A[0][0, 1] == 0 and A[0][0, 2] == 0 and A[1][2, 1] == 0
    assert fit.irf(8, draws=30).lower is not None
    with pytest.raises(SpecificationError, match="not equations of the VAR"):
        VAR(["oil", "a", "b"], lags=2).fit(data, method=Restricted(block_exogenous=["gdp"]))
    with pytest.raises(SpecificationError, match="not a regressor"):
        VAR(["oil", "a", "b"], lags=2).fit(data, method=Restricted(zeros={"a": ["c[-1]"]}))
    from pymetrix.commands import FitVAR, Session

    s = Session()
    s.objects["data"] = data
    s.run(FitVAR("oil, a, b", lags="2", zeros="b: a[-2]", block_exogenous="oil"))
    script = s.script()
    assert "method=Restricted(zeros={'b': ['a[-2]']}, block_exogenous=['oil'])" in script
    ns = {"data": data}
    exec(script, ns)
    assert np.allclose(ns["var"].estimate.B, fit.estimate.B)
