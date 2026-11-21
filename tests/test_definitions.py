"""Canonical definitions (plan item 4): each statistic computed one way, labelled, and checked against the package
whose definition it shares; differences from a library kept and labelled, never passed through silently."""

import warnings

import numpy as np
import pandas as pd
import pytest

from pymetrix import ARDL, OLS, VAR, Cluster, Dataset, Equation, ExponentialSmoothing
from pymetrix.estimation import criteria
from pymetrix.estimation.criteria import DEFINITIONS, definitions

sm = pytest.importorskip("statsmodels.api")


@pytest.fixture(scope="module")
def regression():
    rng = np.random.default_rng(5)
    n = 200
    x = rng.normal(size=n)
    u = rng.normal(size=n) * (1 + np.abs(x))
    y = 1 + 0.5 * x + u
    g = rng.integers(0, 20, size=n)
    h = rng.integers(0, 7, size=n)
    d = Dataset.from_frame(pd.DataFrame({"y": y, "x": x, "g": g, "h": h}, index=pd.period_range("1970Q1", periods=n, freq="Q")))
    return d, y, sm.add_constant(x), g, h


def test_least_squares_statistics_match_statsmodels(regression):
    d, y, X, _, _ = regression
    fit, ref = Equation("y ~ x").fit(d), sm.OLS(y, X).fit()
    s = fit.stats
    assert np.isclose(s["loglike"], ref.llf) and np.isclose(s["aic"], ref.aic) and np.isclose(s["bic"], ref.bic)
    assert np.isclose(s["r2"], ref.rsquared) and np.isclose(s["r2_adj"], ref.rsquared_adj)
    assert np.isclose(s["se_regression"], np.sqrt(ref.scale)) and s["df_resid"] == ref.df_resid
    from statsmodels.stats.stattools import durbin_watson

    assert np.isclose(s["durbin_watson"], durbin_watson(ref.resid))
    assert definitions(fit) == {"nobs": "nobs", "k": "k.estimated", "df_resid": "df_resid", "ssr": "ssr", "se_regression": "se_regression",
                                "r2": "r2.centred", "r2_adj": "r2.adjusted", "loglike": "loglike.gaussian", "aic": "aic.total",
                                "bic": "bic.total", "hq": "hq.total", "durbin_watson": "durbin_watson"}


def test_without_a_constant_r2_is_uncentred(regression):
    d, y, X, _, _ = regression
    fit, ref = Equation("y ~ x - 1").fit(d), sm.OLS(y, X[:, 1:]).fit()
    assert np.isclose(fit.stats["r2"], ref.rsquared) and definitions(fit)["r2"] == "r2.uncentred"


@pytest.mark.parametrize(("cov", "kwargs"), [
    ("classic", {}),
    ("white", {"cov_type": "HC1"}),
    ("hac", {"cov_type": "HAC", "cov_kwds": {"maxlags": 4, "use_correction": True}}),
])
def test_covariance_estimators_are_the_documented_ones(regression, cov, kwargs):
    d, y, X, _, _ = regression
    fit = Equation("y ~ x").fit(d, method=OLS(cov=cov, hac_lags=4 if cov == "hac" else None))
    assert np.allclose(fit.params.se, sm.OLS(y, X).fit(**kwargs).bse)


def test_cluster_covariances(regression):
    d, y, X, g, h = regression
    one = Equation("y ~ x").fit(d, method=OLS(cov=Cluster("g")))
    assert np.allclose(one.params.se, sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g}).bse)
    two = Equation("y ~ x").fit(d, method=OLS(cov=Cluster("g", "h")))
    ref = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": np.column_stack([g, h])})
    assert np.allclose(two.params.se, ref.bse, rtol=1e-8)


def test_the_default_hac_lags_rule_is_recorded(regression):
    d, *_ = regression
    fit = Equation("y ~ x").fit(d, method=OLS(cov="hac"))
    assert fit.stats["hac_lags"] == int(np.floor(4 * (200 / 100) ** (2 / 9)))


def test_var_criteria_are_lutkepohls_with_totals_beside_them():
    data = Dataset.example().with_series(lc="log(c)", ly="log(y)")
    fit = VAR(["lc", "ly"], lags=2).fit(data)
    frame = data.to_frame(["lc", "ly"])
    ref = sm.tsa.VAR(frame.to_numpy()).fit(2)
    s = fit.stats
    assert np.isclose(s["aic"], ref.aic) and np.isclose(s["bic"], ref.bic) and np.isclose(s["hq"], ref.hqic)
    T, K, m = s["nobs"], 2, 1 + 2 * 2
    assert np.isclose(s["aic_total"], -2 * s["loglike"] + 2 * K * m)
    assert np.isclose(s["aic_total"] / T, s["aic"] + K * (1 + np.log(2 * np.pi)))  # the same ranking for a fixed T
    d = definitions(fit)
    assert d["aic"] == "aic.lutkepohl" and d["aic_total"] == "aic.total" and d["fpe"] == "fpe.lutkepohl"


def test_ardl_and_ets_are_normalised_with_the_librarys_values_kept():
    data = Dataset.example().with_series(lc="log(c)", ly="log(y)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ardl = ARDL("lc ~ ly", lags=2, order=1).fit(data)
        ets = ExponentialSmoothing("c", trend="add").fit(data)
    for fit in (ardl, ets):
        s = fit.stats
        assert np.isclose(s["aic"], -2 * s["loglike"] + 2 * s["k"])
        assert np.isclose(s["aic_backend"], s["aic"] + 2)  # statsmodels counts the error variance
        assert np.isclose(s["bic_backend"], s["bic"] + np.log(s["nobs"]))
        assert definitions(fit)["aic_backend"] == "aic.k_plus_variance"


def test_quasi_likelihood_criteria_are_labelled(regression):
    d, *_ = regression
    import pymetrix

    fit = Equation("y ~ x").fit(d, method=pymetrix.Quantile(q=0.5))
    assert definitions(fit)["aic"] == "aic.gaussian_quasi"


def test_the_statistics_view_explains_non_default_definitions():
    from pymetrix.views import view

    data = Dataset.example().with_series(lc="log(c)", ly="log(y)")
    notes = view(VAR(["lc", "ly"], lags=2).fit(data), "Statistics").notes
    assert any(n.startswith("aic: Akaike criterion (Lütkepohl)") for n in notes)
    assert any("EViews divides them by n" in n for n in notes)


def test_every_definition_used_is_documented():
    from pathlib import Path

    text = (Path(__file__).parents[1] / "docs" / "design" / "definitions.md").read_text(encoding="utf-8")
    used = set(criteria.DEFAULT_KEYS.values()) | {"aic.lutkepohl", "bic.lutkepohl", "hq.lutkepohl", "fpe.lutkepohl", "aic.gaussian_quasi",
                                                  "r2.uncentred", "loglike.gaussian", "aic.per_observation"}
    assert used <= set(DEFINITIONS)
    missing = sorted(d for d in used if d not in ("nobs", "ssr", "backend") and f"`{d}`" not in text)
    assert not missing, f"definitions.md does not describe {missing}"


def test_criteria_functions():
    assert criteria.aic(-10.0, 3) == 26.0 and np.isclose(criteria.bic(-10.0, 3, 100), 20 + 3 * np.log(100))
    assert np.isclose(criteria.hq(-10.0, 3, 100), 20 + 6 * np.log(np.log(100)))
    assert criteria.per_observation(26.0, 13) == 2.0
    assert np.isnan(criteria.r2(1.0, 0.0))
