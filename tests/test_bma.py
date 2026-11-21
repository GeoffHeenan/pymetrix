"""Bayesian model averaging (Elliott and Timmermann 2016, section 14.6): the closed-form Bayes factor against numerical
integration of the posterior from its definition, coefficients against statsmodels' least squares, averaged forecasts
by hand, backtests, and the dialog."""

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from pymetrix import BMA, Dataset, backtest
from pymetrix.bma import log_marginal_likelihood
from pymetrix.errors import SpecificationError

sm = pytest.importorskip("statsmodels.api")


def test_the_bayes_factor_is_the_integral_of_its_definition():
    """The evidence of y = alpha + beta x + e against y = alpha + e, from the priors: alpha flat, p(sigma) = 1/sigma, and
    beta ~ N(0, g sigma^2 / x'x) on the centred x. Alpha and sigma are integrated analytically, beta numerically
    (multi-dimensional quadrature is only good to about 1e-4 here); the closed form integrates beta as well."""
    from scipy.special import gammaln

    rng = np.random.default_rng(3)
    n, g = 12, 5.0
    x = rng.normal(size=n)
    y = 1.0 + 0.6 * x + rng.normal(scale=0.8, size=n)
    xc = x - x.mean()
    sxx = float(xc @ xc)
    common = -(n - 1) / 2 * np.log(2 * np.pi) - 0.5 * np.log(n)  # from integrating alpha out of the likelihood

    def m0():  # int s^-(n-1) exp(-SST / 2 s^2) ds / s
        sst = float(np.sum((y - y.mean()) ** 2))
        return np.exp(common + np.log(0.5) + gammaln((n - 1) / 2) - (n - 1) / 2 * np.log(sst / 2))

    def m1():
        def at(beta):
            r = y - beta * x
            q = float(np.sum((r - r.mean()) ** 2)) + beta * beta * sxx / g  # the prior on beta adds beta^2 x'x / g sigma^2
            return np.exp(common - 0.5 * np.log(2 * np.pi * g / sxx) + np.log(0.5) + gammaln(n / 2) - n / 2 * np.log(q / 2))

        return integrate.quad(at, -np.inf, np.inf, epsabs=0, epsrel=1e-12, limit=500)[0]

    r2 = sm.OLS(y, sm.add_constant(x)).fit().rsquared
    closed = np.exp(log_marginal_likelihood(r2, n, 1, g) - log_marginal_likelihood(0.0, n, 0, g))
    assert closed == pytest.approx(m1() / m0(), rel=1e-10)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(8)
    idx = pd.period_range("1990Q1", periods=120, freq="Q", name="time")
    x1, x2, x3 = rng.normal(size=(3, 120))
    y = 0.5 + 0.8 * x1 + 0.3 * x2 + rng.normal(scale=0.7, size=120)  # x3 does not belong
    return Dataset.from_frame(pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3}, index=idx))


def test_posteriors_coefficients_and_inclusion_by_hand(data):
    fit = BMA("y", ["x1", "x2", "x3"]).fit(data)
    frame = data.frame
    n, g = 120, 120.0  # the benchmark: max(n, K^2)
    assert fit.g == g
    logml, slopes = {}, {}
    for size in range(4):
        for subset in combinations(["x1", "x2", "x3"], size):
            ols = sm.OLS(frame["y"], sm.add_constant(frame[list(subset)]) if subset else np.ones(n)).fit()
            logml[subset] = float(log_marginal_likelihood(ols.rsquared if subset else 0.0, n, size, g))
            slopes[subset] = {v: g / (1 + g) * ols.params[v] for v in subset}
    weights = np.exp(np.array(list(logml.values())) - max(logml.values()))
    post = dict(zip(logml, weights / weights.sum()))
    pip = {v: sum(p for s, p in post.items() if v in s) for v in ("x1", "x2", "x3")}
    assert fit.inclusion == pytest.approx([pip["x1"], pip["x2"], pip["x3"]], rel=1e-9)
    means = [sum(post[s] * slopes[s].get(v, 0.0) for s in post) for v in ("x1", "x2", "x3")]
    assert fit.mean == pytest.approx(means, rel=1e-8)
    assert pip["x1"] > 0.99 and pip["x2"] > 0.9 and pip["x3"] < 0.3  # it finds the regressors that belong
    assert fit.summary().index[0] in ("x1", "x2") and fit.models_table().rows[0][0] == pytest.approx(max(post.values()))


def test_always_included_regressors_are_partialled_out(data):
    fit = BMA("y", ["x2", "x3"], always=("x1",)).fit(data)
    frame = data.frame
    full = sm.OLS(frame["y"], sm.add_constant(frame[["x1", "x2"]])).fit()
    base = sm.OLS(frame["y"], sm.add_constant(frame[["x1"]])).fit()
    partial_r2 = 1 - full.ssr / base.ssr  # the R^2 of x2 after the intercept and x1
    m = [i for i, cols in enumerate(fit.models) if cols == [0]][0]
    expected = log_marginal_likelihood(partial_r2, 120, 1, fit.g, q=2) - log_marginal_likelihood(0.0, 120, 0, fit.g, q=2)
    empty = [i for i, cols in enumerate(fit.models) if cols == []][0]
    assert np.log(fit.posterior[m] / fit.posterior[empty]) == pytest.approx(expected, rel=1e-9)
    assert fit.coefs[m, 0] == pytest.approx(fit.g / (1 + fit.g) * full.params["x2"], rel=1e-9)


def test_forecasts_average_the_models(data):
    train = Dataset.from_frame(data.frame.iloc[:100])
    fit = BMA("y", ["x1", "x2"]).fit(train)
    fc = fit.forecast(data, ("2015Q1", "2015Q4")).frame
    frame = data.frame
    est = frame.iloc[:100]
    expected = np.zeros(4)
    for m, cols in enumerate(fit.models):
        names = ["x1", "x2"]
        chosen = [names[j] for j in cols]
        beta = np.array([fit.coefs[m, j] for j in cols])
        intercept = est["y"].mean() - (est[chosen].mean().to_numpy() @ beta if chosen else 0.0)
        future = frame.loc["2015Q1":"2015Q4"]
        expected += fit.posterior[m] * (intercept + (future[chosen].to_numpy() @ beta if chosen else 0.0))
    assert fc["forecast"].to_numpy() == pytest.approx(expected, rel=1e-9)
    assert (fc["se"] > 0).all() and fc["actual"].to_numpy() == pytest.approx(frame["y"].loc["2015Q1":"2015Q4"].to_numpy())


def test_backtests_and_expression_targets():
    macro = Dataset.example("sample_macro")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = BMA("dlog(c)", ["dlog(y)", "dlog(c)[-1]"]).fit(macro, span=(None, "2018Q4"))
        bt = backtest(fit, macro, ("2016Q1", "2017Q4"), horizon=1, label="bma")
    row = bt.errors.iloc[3]
    own = BMA("dlog(c)", ["dlog(y)", "dlog(c)[-1]"]).fit(macro, span=(fit.provenance.sample[0], row["origin"]))
    assert row["forecast"] == pytest.approx(float(own.forecast(macro, (str(row["target"]), str(row["target"]))).frame["forecast"].iloc[0]))
    assert row["actual"] == pytest.approx(float(macro.eval("dlog(c)")[row["target"]]))  # the outcome of an expression target
    assert row["density"] == "normal" and np.isfinite(row["crps"])


def test_specification_errors():
    with pytest.raises(SpecificationError, match="at least one candidate"):
        BMA("y", [])
    with pytest.raises(SpecificationError, match="at most 15"):
        BMA("y", [f"x{i}" for i in range(16)])
    with pytest.raises(SpecificationError, match="listed twice"):
        BMA("y", ["x1"], always=("x1",))


def test_the_dialog_replays(data):
    from pymetrix.commands import FitBMA, Session

    session = Session()
    session.objects["data"] = data
    session.run(FitBMA("y", "x1, x2, x3"))
    namespace = dict(session.namespace) | {"data": data}
    exec(session.history[-1].code, namespace)
    assert namespace["bma"].inclusion == pytest.approx(session["bma"].inclusion)
    assert FitBMA("y", "").validate(session) == ["give at least one candidate regressor"]
