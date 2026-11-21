"""HAC standard errors with and without the n / (n - k) small-sample factor.

The default (hac_correction=True) multiplies the Newey-West covariance by n / (n - k), as EViews and Stata's newey do and
as statsmodels does with use_correction=True; hac_correction=False leaves it out, which is statsmodels' default. OLS and
IV take the option; the summary says when the factor is left out."""

import numpy as np
import pandas as pd
import pytest

sm = pytest.importorskip("statsmodels.api")

from pymetrix import IV, OLS, Dataset, Equation  # noqa: E402


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    n = 120
    e = np.zeros(n)
    for t in range(1, n):  # autocorrelated errors, so the lags matter
        e[t] = 0.6 * e[t - 1] + rng.normal()
    x, z = rng.normal(size=(2, n))
    df = pd.DataFrame({"y": 1 + 0.5 * x + e, "x": x, "z": z + 0.8 * x},
                      index=pd.period_range("1990Q1", periods=n, freq="Q", name="time"))
    return df, Dataset.from_frame(df)


@pytest.mark.parametrize("correction", [True, False])
def test_ols_matches_statsmodels(data, correction):
    df, ds = data
    fit = Equation("y ~ x").fit(ds, method=OLS(cov="hac", hac_lags=4, hac_correction=correction))
    ref = sm.OLS(df.y.to_numpy(), sm.add_constant(df.x.to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": correction})
    np.testing.assert_allclose(fit.params.cov, ref.cov_params(), rtol=1e-10)


def test_default_is_corrected_and_the_ratio_is_n_over_df(data):
    _, ds = data
    default = Equation("y ~ x").fit(ds, method=OLS(cov="hac", hac_lags=4))
    corrected = Equation("y ~ x").fit(ds, method=OLS(cov="hac", hac_lags=4, hac_correction=True))
    plain = Equation("y ~ x").fit(ds, method=OLS(cov="hac", hac_lags=4, hac_correction=False))
    np.testing.assert_array_equal(default.params.cov, corrected.params.cov)
    n = len(ds.frame)
    np.testing.assert_allclose(corrected.params.cov, plain.params.cov * n / (n - 2), rtol=1e-12)
    assert "without the n / (n - k) correction" in str(plain.summary())
    assert "without" not in str(default.summary())
    # other covariances ignore the option
    white = Equation("y ~ x").fit(ds, method=OLS(cov="white"))
    np.testing.assert_array_equal(white.params.cov, Equation("y ~ x").fit(ds, method=OLS(cov="white", hac_correction=False)).params.cov)


def test_iv(data):
    _, ds = data
    corrected = Equation("y ~ x").fit(ds, method=IV(["x"], ["z"], cov="hac", hac_lags=4))
    plain = Equation("y ~ x").fit(ds, method=IV(["x"], ["z"], cov="hac", hac_lags=4, hac_correction=False))
    np.testing.assert_array_equal(corrected.params.values, plain.params.values)
    n = len(ds.frame)
    np.testing.assert_allclose(corrected.params.cov, plain.params.cov * n / (n - 2), rtol=1e-12)
    assert "without the n / (n - k) correction" in str(plain.summary())


def test_positional_order_unchanged():
    """The new field comes last, so a call written positionally before means the same."""
    assert OLS("hac", 4).hac_lags == 4 and OLS("hac", 4).hac_correction is True
    assert IV(["x"], ["z"], "hac", 3, {"x": 1.0}).start == {"x": 1.0}
