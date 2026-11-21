"""Temporal disaggregation, extended: aggregation matrices (calendar and fiscal years mixed), soft constraints, Denton in
seasonal differences, Litterman, the search range of rho, and two-way reconciliation.

The references are written here from the formulas, by a different route from the package's: GLS with the
covariances built from their processes, Denton by elimination over the constraints' null space, soft Denton through its
normal equations, reconciliation through the full KKT system or the pro-rata rule."""

import numpy as np
import pandas as pd
import pytest
from scipy import linalg

from pymetrix import (
    Dataset,
    Matrix,
    TemporalDisaggregation,
    aggregation_matrix,
    reconcile,
)
from pymetrix.disaggregation import conversion_matrix
from pymetrix.errors import DataError, NotFoundError, SpecificationError


@pytest.fixture(scope="module")
def economy():
    """Quarterly truth y = 10 + 2 ip + AR(1) (rho 0.8), 2000Q1-2023Q4; ip to 2024Q2."""
    rng = np.random.default_rng(11)
    q = pd.period_range("2000Q1", "2023Q4", freq="Q", name="time")
    n = len(q)
    ip = 100 + np.cumsum(rng.normal(0.5, 1, n + 2))
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.8 * u[t - 1] + rng.normal(0, 1)
    y = 10 + 2 * ip[:n] + u
    quarterly = Dataset.from_frame(pd.DataFrame({"ip": ip}, index=pd.period_range("2000Q1", periods=n + 2, freq="Q", name="time")))
    annual = Dataset.from_frame(pd.DataFrame({"gdp": pd.Series(y, index=q).groupby(q.year).sum().to_numpy()},
                                             index=pd.period_range("2000", "2023", freq="Y", name="time")))
    return annual, quarterly, pd.Series(y, index=q)


def _mixed(quarterly, truth):
    """Calendar years 2000-2011 and July-June fiscal years 2011-2023 (fiscal 2011 overlaps calendar 2010 and 2011)."""
    periods = [pd.Period(str(a), "Y") for a in range(2000, 2012)] + [pd.Period(str(a), "Y-JUN") for a in range(2011, 2024)]
    C = aggregation_matrix(quarterly, periods)
    values = C.values[:, : len(truth)] @ truth.to_numpy()
    low = Dataset.from_frame(pd.DataFrame({"gdp": values, "period": list(C.rows)}), entity="period")
    return C, low


def _gls_reference(Y, X, C, Sigma):
    """Chow-Lin's BLUE written out: beta by GLS on the aggregates, the residual spread by Sigma C' (C Sigma C')^-1."""
    V = C @ Sigma @ C.T
    Vi = np.linalg.inv(V)
    Xl = C @ X
    beta = np.linalg.solve(Xl.T @ Vi @ Xl, Xl.T @ Vi @ Y)
    return beta, X @ beta + Sigma @ C.T @ Vi @ (Y - Xl @ beta)


def _denton_reference(Y, x, C, D, proportional):
    """min |D (y - x)|^2 (or of (y - x) / x) subject to C y = Y, by elimination: y = y0 + N v over the null space of C."""
    W = D @ np.diag(1 / x) if proportional else D
    y0 = np.linalg.lstsq(C, Y, rcond=None)[0]
    N = linalg.null_space(C)
    v = np.linalg.lstsq(W @ N, W @ (x - y0), rcond=None)[0]
    return y0 + N @ v


# -- aggregation matrix ----------------------------------------------------------------------------------------------

def test_the_aggregation_matrix_names_periods_of_any_frequency(economy):
    _, quarterly, _ = economy
    C = aggregation_matrix(quarterly, [pd.Period("2019", "Y"), pd.Period("2020", "Y-JUN"), "2021Q3"])
    assert C.rows == ("2019", "2020 Y-JUN", "2021Q3") and C.cols[0] == "2000Q1"
    j = {c: i for i, c in enumerate(C.cols)}
    assert C.values[1].sum() == 4 and C.values[1, j["2019Q3"]] == C.values[1, j["2020Q2"]] == 1 and C.values[1, j["2020Q3"]] == 0
    assert C.values[2].sum() == 1 and C.values[2, j["2021Q3"]] == 1
    assert np.allclose(aggregation_matrix(quarterly, ["2019", "2020"], "average").values.sum(axis=1), 1)


@pytest.mark.parametrize("method", ["chow-lin", "fernandez", "litterman", "denton", "denton-additive"])
def test_mixed_calendar_and_fiscal_years_hold_exactly(economy, method):
    _, quarterly, truth = economy
    C, low = _mixed(quarterly, truth)
    fit = TemporalDisaggregation("gdp ~ ip", method=method).fit(low, quarterly, aggregation=C)
    Y = low["gdp"].reindex(list(C.rows)).to_numpy()
    assert np.max(np.abs(fit.C @ fit.series.to_numpy() - Y)) < 1e-10 * np.abs(Y).max()  # C y = Y
    assert np.corrcoef(fit.series.loc[truth.index], truth)[0, 1] > 0.998
    assert "aggregation matrix of 25 constraints" in str(fit.summary())
    assert fit.provenance.spec["aggregation"] == "matrix (25 constraints)"
    if method in ("chow-lin", "fernandez", "litterman"):
        assert str(fit.series.index[-1]) == "2024Q2"  # extrapolated with the indicator, past the matrix's last column of weight


def test_a_matrix_gives_chow_lin_s_gls_and_the_conversion_s_results(economy):
    annual, quarterly, truth = economy
    C, low = _mixed(quarterly, truth)
    fit = TemporalDisaggregation("gdp ~ ip", rho=0.6).fit(low, quarterly, aggregation=C)
    n = len(quarterly)
    i = np.arange(n)
    Sigma = 0.6 ** np.abs(i[:, None] - i[None, :]) / (1 - 0.36)
    X = np.column_stack([np.ones(n), quarterly["ip"].to_numpy()])
    beta, y = _gls_reference(low["gdp"].reindex(list(C.rows)).to_numpy(), X, C.values, Sigma)
    assert np.allclose(fit.params, beta, rtol=1e-10) and np.allclose(fit.series, y, rtol=1e-10)
    calendar = aggregation_matrix(quarterly, pd.period_range("2000", "2023", freq="Y"))
    for method in ("chow-lin", "denton"):
        by_matrix = TemporalDisaggregation("gdp ~ ip", method=method).fit(annual, quarterly, aggregation=calendar)
        by_conversion = TemporalDisaggregation("gdp ~ ip", method=method).fit(annual, quarterly)
        assert np.allclose(by_matrix.series.loc[by_conversion.series.index], by_conversion.series, rtol=1e-8)


def test_every_form_of_the_matrix_gives_the_same_series(economy, tmp_path):
    annual, quarterly, _ = economy
    C = aggregation_matrix(quarterly, pd.period_range("2000", "2023", freq="Y"))
    reference = TemporalDisaggregation("gdp ~ ip").fit(annual, quarterly, aggregation=C).series
    frame = C.to_frame()
    weights = Dataset.from_frame(pd.DataFrame(C.values.T, index=quarterly.frame.index, columns=[f"y{p}" for p in C.rows]))
    path = tmp_path / "weights.csv"
    frame.to_csv(path)
    positional = Matrix(C.values)  # unnamed rows and columns: matched by position
    for form in (frame, C.values, positional, str(path)):
        fit = TemporalDisaggregation("gdp ~ ip").fit(annual, quarterly, aggregation=form)
        assert np.allclose(fit.series, reference, rtol=1e-12)
    low = Dataset.from_frame(pd.DataFrame({"gdp": annual["gdp"].to_numpy(), "period": [f"y{p}" for p in C.rows]}), entity="period")
    assert np.allclose(TemporalDisaggregation("gdp ~ ip").fit(low, quarterly, aggregation=weights).series, reference, rtol=1e-12)


def test_missing_low_frequency_values_drop_their_rows(economy):
    _, quarterly, truth = economy
    C, low = _mixed(quarterly, truth)
    gaps = low.frame.copy()
    gaps.loc["2005", "gdp"] = np.nan
    holes = Dataset.from_frame(gaps.reset_index(), entity="period")
    fit = TemporalDisaggregation("gdp ~ ip").fit(holes, quarterly, aggregation=C)
    keep = [r for r in C.rows if r != "2005"]
    fewer = TemporalDisaggregation("gdp ~ ip").fit(low, quarterly, aggregation=C[keep, :])
    assert np.allclose(fit.series, fewer.series, rtol=1e-12) and list(fit.low_index) == keep
    assert fit.provenance.dropped == 1 and "1 dropped" in str(fit.summary())


def test_the_matrix_is_checked(economy):
    annual, quarterly, truth = economy
    C, low = _mixed(quarterly, truth)
    twice = Matrix(np.vstack([C.values, C.values[:1]]), (*C.rows, "again"), C.cols)
    low_twice = Dataset.from_frame(pd.DataFrame({"gdp": [*low["gdp"].reindex(list(C.rows)), low["gdp"]["2000"]],
                                                 "period": [*C.rows, "again"]}), entity="period")
    with pytest.raises(SpecificationError, match="not independent"):
        TemporalDisaggregation("gdp ~ ip").fit(low_twice, quarterly, aggregation=twice)
    with pytest.raises(SpecificationError, match="one column per high-frequency period"):
        TemporalDisaggregation("gdp ~ ip").fit(annual, quarterly, aggregation=np.ones((24, 5)))
    with pytest.raises(SpecificationError, match="replaces conversion"):
        TemporalDisaggregation("gdp ~ ip", conversion="average").fit(low, quarterly, aggregation=C)
    short = quarterly.with_series(ip=quarterly["ip"].where(quarterly.frame.index >= pd.Period("2001Q1", "Q")))
    with pytest.raises(SpecificationError, match="outside the indicators' sample"):
        TemporalDisaggregation("gdp ~ ip").fit(low, short, aggregation=C)
    with pytest.raises(SpecificationError, match="rows for"):
        TemporalDisaggregation("gdp ~ ip").fit(annual, quarterly, aggregation=C.values[:5])
    with pytest.raises(DataError, match="no row"):
        TemporalDisaggregation("gdp ~ ip").fit(annual, quarterly, aggregation=Matrix(C.values, tuple(f"x{r}" for r in C.rows), C.cols))


# -- soft constraints ------------------------------------------------------------------------------------------------

def test_soft_regression_runs_from_hard_to_the_indicator_path(economy):
    annual, quarterly, _ = economy
    X = np.column_stack([np.ones(len(quarterly)), quarterly["ip"].to_numpy()])
    for method in ("chow-lin", "fernandez", "litterman"):
        kw = {"rho": 0.5} if method != "fernandez" else {}
        hard = TemporalDisaggregation("gdp ~ ip", method=method, **kw).fit(annual, quarterly)
        tight = TemporalDisaggregation("gdp ~ ip", method=method, soft=1e-7, **kw).fit(annual, quarterly)
        loose = TemporalDisaggregation("gdp ~ ip", method=method, soft=1e7, **kw).fit(annual, quarterly)
        assert np.allclose(tight.series, hard.series, atol=1e-5) and np.allclose(tight.params, hard.params, rtol=1e-6)
        assert np.max(np.abs(loose.series.to_numpy() - X @ loose.params)) < 1e-6 * np.abs(loose.series).max()  # y = X b
        assert loose.check > 1  # no longer meets the totals


def test_soft_regression_is_gls_with_measurement_error(economy):
    """At the fit's own s2: b and y by the GLS formulas with V = s2 C Sigma C' + diag(sd^2); s2 maximises the likelihood."""
    annual, quarterly, _ = economy
    sd = {"2003": 20.0, "2010": 5.0, "2017": 50.0}  # the other years stay exact
    fit = TemporalDisaggregation("gdp ~ ip", rho=0.7, soft=sd).fit(annual, quarterly)
    n, Y = len(quarterly), annual["gdp"].to_numpy()
    C = conversion_matrix(annual.frame.index, quarterly.frame.index[:96], "sum")
    C = np.hstack([C, np.zeros((24, n - 96))])
    i = np.arange(n)
    Sigma = 0.7 ** np.abs(i[:, None] - i[None, :]) / (1 - 0.49)
    X = np.column_stack([np.ones(n), quarterly["ip"].to_numpy()])
    var = np.array([sd.get(str(p), 0.0) ** 2 for p in annual.frame.index])

    def at(s2):
        V = s2 * C @ Sigma @ C.T + np.diag(var)
        Vi = np.linalg.inv(V)
        b = np.linalg.solve(X.T @ C.T @ Vi @ C @ X, X.T @ C.T @ Vi @ Y)
        u = Y - C @ X @ b
        ll = -0.5 * (np.linalg.slogdet(2 * np.pi * V)[1] + u @ Vi @ u)
        return b, X @ b + s2 * Sigma @ C.T @ Vi @ u, ll

    s2 = fit.info["sigma2"]
    b, y, ll = at(s2)
    assert np.allclose(fit.params, b, rtol=1e-8) and np.allclose(fit.series, y, rtol=1e-8)
    assert ll >= at(s2 * 1.02)[2] and ll >= at(s2 / 1.02)[2]
    gaps = fit.gaps
    exact = [str(p) for p in annual.frame.index if str(p) not in sd]
    assert np.max(np.abs(gaps[exact])) < 1e-8 and np.all(np.abs(gaps[list(sd)]) > 1e-6)
    assert "Soft constraints (3 of 24)" in str(fit.summary())


def test_soft_denton_from_hard_to_the_indicator(economy):
    annual, quarterly, _ = economy
    quarterly = quarterly.select(span=(None, "2023Q4"))
    x = quarterly["ip"].to_numpy()
    C = conversion_matrix(annual.frame.index, quarterly.frame.index, "sum")
    Y = annual["gdp"].to_numpy()
    for method in ("denton", "denton-additive"):
        hard = TemporalDisaggregation("gdp ~ ip", method=method).fit(annual, quarterly)
        tight = TemporalDisaggregation("gdp ~ ip", method=method, soft=1e-8).fit(annual, quarterly)
        loose = TemporalDisaggregation("gdp ~ ip", method=method, soft=1e8).fit(annual, quarterly)
        assert np.allclose(tight.series, hard.series, rtol=1e-9)
        level = loose.series.to_numpy() / x if method == "denton" else loose.series.to_numpy() - x
        assert np.ptp(np.diff(level)) < 1e-6 * np.abs(level).max()  # the indicator, moved by a constant ratio or amount
        middle = TemporalDisaggregation("gdp ~ ip", method=method, soft=3.0).fit(annual, quarterly)
        n = len(x)
        D = (np.eye(n) - np.eye(n, k=-1))[1:]
        W = D @ np.diag(1 / x) if method == "denton" else D
        A = W.T @ W  # penalised least squares: (A + C'C / sd^2) y = A x + C'Y / sd^2
        ref = np.linalg.solve(A + C.T @ C / 9.0, A @ x + C.T @ Y / 9.0)
        assert np.allclose(middle.series, ref, rtol=1e-8)


def test_soft_is_checked(economy):
    annual, quarterly, _ = economy
    with pytest.raises(SpecificationError, match="zero .exact. or positive"):
        TemporalDisaggregation("gdp ~ ip", soft=-1)
    with pytest.raises(NotFoundError, match="1999"):
        TemporalDisaggregation("gdp ~ ip", soft={"1999": 1.0}).fit(annual, quarterly)
    exact = TemporalDisaggregation("gdp ~ ip", soft=0.0).fit(annual, quarterly)  # zero everywhere: exact
    assert np.array_equal(exact.series, TemporalDisaggregation("gdp ~ ip").fit(annual, quarterly).series)


# -- Denton in seasonal differences ----------------------------------------------------------------------------------

def _seasonal_economy(freq, s, start, years=15, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start, periods=s * years, freq=freq, name="time")
    season = np.tile(np.linspace(-10, 10, s) ** 2 / 5 - 5, years)
    x = 100 + 0.5 * np.arange(len(idx)) + season
    totals = x.reshape(years, s).sum(axis=1) + s * rng.normal(0, 3, years).cumsum()
    high = Dataset.from_frame(pd.DataFrame({"x": x}, index=idx))
    low = Dataset.from_frame(pd.DataFrame({"y": totals}, index=pd.period_range(str(idx[0].year), periods=years, freq="Y", name="time")))
    return low, high, x


@pytest.mark.parametrize(("freq", "s", "start"), [("Q", 4, "2000Q1"), ("M", 12, "2000-01")])
def test_seasonal_denton_keeps_the_indicator_s_seasonal_pattern(freq, s, start):
    low, high, x = _seasonal_economy(freq, s, start)
    years = len(x) // s

    def drift(adjustment):  # how much the adjustment's within-year pattern changes from year to year
        a = adjustment.reshape(years, s)
        return np.ptp(a - a.mean(axis=1, keepdims=True), axis=0).max()

    first = TemporalDisaggregation("y ~ x", "denton-additive").fit(low, high)
    for penalty, bound in ((None, 0.1), (1e-6, 1e-4)):
        seasonal = TemporalDisaggregation("y ~ x", "denton-additive", differences="seasonal", level_penalty=penalty).fit(low, high)
        assert seasonal.check < 1e-9
        assert drift(seasonal.series.to_numpy() - x) < bound < drift(first.series.to_numpy() - x)
    assert f"seasonal differences (lag {s}), first-difference weight 0.001" in str(
        TemporalDisaggregation("y ~ x", "denton", differences="seasonal").fit(low, high).summary())


@pytest.mark.parametrize("method", ["denton", "denton-additive"])
def test_seasonal_denton_against_the_quadratic_programme(method):
    low, high, x = _seasonal_economy("Q", 4, "2000Q1", years=8, seed=4)
    n = len(x)
    C = conversion_matrix(low.frame.index, high.frame.index, "sum")
    Y = low["y"].to_numpy()
    lam = 0.01
    D4 = (np.eye(n) - np.eye(n, k=-4))[4:]
    D1 = (np.eye(n) - np.eye(n, k=-1))[1:]
    D = np.vstack([D4, np.sqrt(lam) * D1])  # |D4 a|^2 + lam |D1 a|^2 as one stacked least-squares criterion
    ref = _denton_reference(Y, x, C, D, proportional=method == "denton")
    fit = TemporalDisaggregation("y ~ x", method, differences="seasonal", level_penalty=lam).fit(low, high)
    assert np.allclose(fit.series, ref, rtol=1e-9)
    first = TemporalDisaggregation("y ~ x", method).fit(low, high)  # and the first-difference form, the same way
    assert np.allclose(first.series, _denton_reference(Y, x, C, D1, proportional=method == "denton"), rtol=1e-9)


def test_seasonal_differences_are_checked():
    low, _, _ = _seasonal_economy("Q", 4, "2000Q1")
    with pytest.raises(SpecificationError, match="for Denton"):
        TemporalDisaggregation("y ~ x", "chow-lin", differences="seasonal")
    annual_high = Dataset.from_frame(pd.DataFrame({"x": np.arange(1.0, 11)}, index=pd.period_range("2000", periods=10, freq="Y", name="time")))
    with pytest.raises(SpecificationError, match="quarterly or monthly"):
        TemporalDisaggregation("y ~ x", "denton", differences="seasonal").fit(low.select(span=("2000", "2009")), annual_high)
    with pytest.raises(SpecificationError, match="zero or positive"):
        TemporalDisaggregation("y ~ x", "denton", differences="seasonal", level_penalty=-1)


# -- Litterman -------------------------------------------------------------------------------------------------------

def _litterman_sigma(n, rho):
    """The covariance of u_t = u_(t-1) + e_t, e_t = rho e_(t-1) + eps_t (starting from zero), by its moving average."""
    L = np.zeros((n, n))  # u = L eps
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        for t in range(j + 1, n):
            e[t] = rho * e[t - 1]
        L[:, j] = np.cumsum(e)
    return L @ L.T


def test_litterman_is_fernandez_at_rho_zero(economy):
    annual, quarterly, _ = economy
    fernandez = TemporalDisaggregation("gdp ~ ip", "fernandez").fit(annual, quarterly)
    litterman = TemporalDisaggregation("gdp ~ ip", "litterman", rho=0.0).fit(annual, quarterly)
    assert np.allclose(litterman.series, fernandez.series, rtol=1e-12) and np.allclose(litterman.params, fernandez.params, rtol=1e-10)


def test_litterman_against_gls_and_its_likelihood(economy):
    annual, quarterly, _ = economy
    n = len(quarterly)
    C = np.hstack([conversion_matrix(annual.frame.index, quarterly.frame.index[:96], "sum"), np.zeros((24, n - 96))])
    X = np.column_stack([np.ones(n), quarterly["ip"].to_numpy()])
    Y = annual["gdp"].to_numpy()
    fixed = TemporalDisaggregation("gdp ~ ip", "litterman", rho=0.4).fit(annual, quarterly)
    beta, y = _gls_reference(Y, X, C, _litterman_sigma(n, 0.4))
    assert np.allclose(fixed.params, beta, rtol=1e-8) and np.allclose(fixed.series, y, rtol=1e-8)

    def loglike(rho):  # the concentrated likelihood of the aggregates
        V = C @ _litterman_sigma(n, rho) @ C.T
        Vi = np.linalg.inv(V)
        b = np.linalg.solve(X.T @ C.T @ Vi @ C @ X, X.T @ C.T @ Vi @ Y)
        u = Y - C @ X @ b
        return -len(Y) / 2 * (np.log(2 * np.pi * (u @ Vi @ u) / len(Y)) + 1) - np.linalg.slogdet(V)[1] / 2

    free = TemporalDisaggregation("gdp ~ ip", "litterman", rho_bounds=(-0.9, 0.9)).fit(annual, quarterly)
    grid = np.linspace(-0.9, 0.9, 181)
    assert free.rho == pytest.approx(grid[np.argmax([loglike(r) for r in grid])], abs=0.011)
    assert free.info["loglike"] == pytest.approx(loglike(free.rho), rel=1e-9)
    assert free.check < 1e-9 and "maximum likelihood" in str(free.summary())
    minrss = TemporalDisaggregation("gdp ~ ip", "litterman-minrss").fit(annual, quarterly)
    assert 0 <= minrss.rho <= 0.999 and "minimum residual sum of squares" in str(minrss.summary())


# -- rho's search range ----------------------------------------------------------------------------------------------

def test_an_estimate_at_the_end_of_rho_s_range_is_noted(economy):
    annual, quarterly, _ = economy
    rng = np.random.default_rng(2)
    noise = annual.with_series(gdp=annual["gdp"] + rng.normal(0, 40, len(annual)))  # rough annual data push rho down
    at_zero = TemporalDisaggregation("gdp ~ ip").fit(noise, quarterly)
    assert at_zero.rho < 1e-4 and "lower end of its search range [0, 0.999]" in str(at_zero.summary())
    assert "rho_bounds=(-0.999, 0.999)" in at_zero.provenance.messages[0]
    wider = TemporalDisaggregation("gdp ~ ip", rho_bounds=(-0.999, 0.999)).fit(noise, quarterly)
    assert wider.rho < -0.5 and "rho_note" not in wider.info and wider.info["loglike"] > at_zero.info["loglike"]
    inside = TemporalDisaggregation("gdp ~ ip").fit(annual, quarterly)
    assert "search range" not in str(inside.summary()) and inside.provenance.messages == ()
    with pytest.raises(SpecificationError, match="between -1 and 1"):
        TemporalDisaggregation("gdp ~ ip", rho_bounds=(0.5, 0.1))


# -- reconciliation --------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def accounts():
    """Three components and their total, quarterly 2005-2019; annual totals of each; preliminary quarterly values that miss."""
    rng = np.random.default_rng(5)
    q = pd.period_range("2005Q1", "2019Q4", freq="Q", name="time")
    n = len(q)
    truth = {c: 100 * (i + 1) + np.cumsum(rng.normal(0.5, 1, n)) for i, c in enumerate(["cons", "inv", "gov"])}
    prelim = {c: v * (1 + rng.normal(0, 0.02, n)) for c, v in truth.items()}
    prelim["gov"] = truth["gov"]  # gov is right already: it can be held fixed
    total = sum(truth.values())
    quarterly = Dataset.from_frame(pd.DataFrame({**prelim, "gdp": total}, index=q))
    annual = Dataset.from_frame(pd.DataFrame({c: pd.Series(v, index=q).groupby(q.year).sum().to_numpy() for c, v in truth.items()},
                                             index=pd.period_range("2005", "2019", freq="Y", name="time")))
    return quarterly, annual


def _kkt_reference(P, z, blocks, constraints):
    """min sum_j (y_j - p_j)' A_j (y_j - p_j) s.t. the stacked constraints, from the full KKT system."""
    A = linalg.block_diag(*blocks)
    G = np.vstack([g for g, _ in constraints])
    g = np.concatenate([v for _, v in constraints])
    m = G.shape[0]
    K = np.block([[A, G.T], [G, np.zeros((m, m))]])
    p = np.concatenate(P)
    sol = np.linalg.lstsq(K, np.r_[A @ p, g], rcond=None)[0]
    return sol[: len(p)].reshape(len(P), -1)


@pytest.mark.parametrize("method", ["denton", "denton-additive", "least-squares"])
def test_reconciled_components_add_up_and_keep_their_totals(accounts, method):
    quarterly, annual = accounts
    rec = reconcile(quarterly, ["cons", "inv", "gov"], "gdp", annual, fixed=["gov"], method=method)
    total = quarterly["gdp"].to_numpy()
    assert np.max(np.abs(rec.series.sum(axis=1).to_numpy() - total)) < 1e-10 * total.max()
    assert np.array_equal(rec.series["gov"].to_numpy(), quarterly["gov"].to_numpy())  # fixed: unchanged
    for c in ("cons", "inv"):
        yearly = rec.series[c].groupby(rec.series.index.year).sum().to_numpy()
        assert np.max(np.abs(yearly - annual[c].to_numpy())) < 1e-10 * yearly.max()
    assert rec.check < 1e-9
    # the reference: the same criterion and constraints, through the KKT system
    n = len(quarterly)
    D = (np.eye(n) - np.eye(n, k=-1))[1:]
    P = [quarterly[c].to_numpy() for c in ("cons", "inv")]
    if method == "denton":
        blocks = [(D @ np.diag(1 / p)).T @ (D @ np.diag(1 / p)) for p in P]
    elif method == "denton-additive":
        blocks = [D.T @ D] * 2
    else:
        blocks = [np.diag(1 / np.abs(p)) for p in P]
    C = conversion_matrix(annual.frame.index, quarterly.frame.index, "sum")
    Z = np.zeros_like(C)
    ref = _kkt_reference(P, total - quarterly["gov"].to_numpy(), blocks,
                         [(np.hstack([np.eye(n), np.eye(n)]), total - quarterly["gov"].to_numpy()),
                          (np.hstack([C, Z]), annual["cons"].to_numpy()), (np.hstack([Z, C]), annual["inv"].to_numpy())])
    assert np.allclose(rec.series[["cons", "inv"]].to_numpy().T, ref, rtol=1e-7)
    table = rec.diagnostics()
    assert table.index[-1] == "gdp (sum of components)" and table.rows[2][0] == "yes" and table.rows[2][2] == 0


def test_least_squares_balancing_without_totals_is_pro_rata(accounts):
    """With only the total to meet, each period's gap is shared in proportion to the values (Stone's rule, by hand)."""
    quarterly, _ = accounts
    rec = reconcile(quarterly, "cons inv gov", "gdp", method="least-squares")
    P = quarterly.frame[["cons", "inv", "gov"]].to_numpy()
    gap = quarterly["gdp"].to_numpy() - P.sum(axis=1)
    assert np.allclose(rec.series.to_numpy(), P + P / P.sum(axis=1, keepdims=True) * gap[:, None], rtol=1e-12)
    heavier = reconcile(quarterly, "cons inv gov", "gdp", method="least-squares", weights={"inv": 4})
    w = np.array([1.0, 4.0, 1.0]) * P
    assert np.allclose(heavier.series.to_numpy(), P + w / w.sum(axis=1, keepdims=True) * gap[:, None], rtol=1e-12)


def test_reconciliation_reports_contradictions_and_bad_input(accounts):
    quarterly, annual = accounts
    wrong = annual.with_series(cons=annual["cons"] + 5)  # the components' annual totals no longer add up to the total's
    with pytest.raises(SpecificationError, match="contradict"):
        reconcile(quarterly, ["cons", "inv", "gov"], "gdp", wrong)
    with pytest.raises(SpecificationError, match="every component is fixed"):
        reconcile(quarterly, ["cons", "inv"], "gdp", fixed=["cons", "inv"])
    with pytest.raises(NotFoundError, match="not one of the components"):
        reconcile(quarterly, ["cons", "inv"], "gdp", fixed=["gov"])
    with pytest.raises(SpecificationError, match="method"):
        reconcile(quarterly, ["cons", "inv"], "gdp", method="ras")
    rec = reconcile(quarterly, ["cons", "inv", "gov"], "gdp", annual)
    assert rec.provenance.data_fingerprint and rec.provenance.sample == ("2005Q1", "2019Q4")
    assert "Di Fonzo-Marini" in str(rec.summary()) and rec.plot().panels


# -- dialogs ---------------------------------------------------------------------------------------------------------

def _replay(s, name, **objects):
    ns = dict(objects)
    exec(s.script(), ns)  # noqa: S102 - the session's own script
    return ns[name]


def test_the_disaggregation_dialog_with_a_matrix_soft_and_seasonal_options(economy, tmp_path):
    from pymetrix.commands import Disaggregate, Session, load_project, save_project

    annual, quarterly, truth = economy
    C, low = _mixed(quarterly, truth)
    s = Session()
    for name, obj in (("low", low), ("data", quarterly), ("C", C)):
        s.objects[name] = obj
    fit = s.run(Disaggregate("gdp ~ ip", "low", method="litterman", aggregation="C", soft=2.0, add_to_high=False))
    api = TemporalDisaggregation("gdp ~ ip", "litterman", soft=2.0).fit(low, quarterly, aggregation=C)
    assert np.array_equal(fit.series, api.series)
    script = s.script()
    assert "TemporalDisaggregation(formula='gdp ~ ip', method='litterman', soft=2.0).fit(low, data, aggregation=C)" in script
    assert np.array_equal(_replay(s, "disaggregation", low=low, data=quarterly, C=C).series, fit.series)
    path = tmp_path / "C.csv"
    C.to_frame().to_csv(path)
    s.run(Disaggregate("gdp ~ ip", "low", method="denton-additive", aggregation_file=str(path), name="by_file", add_to_high=False))
    assert np.allclose(s["by_file"].series, TemporalDisaggregation("gdp ~ ip", "denton-additive").fit(low, quarterly, aggregation=C).series)
    s2 = Session()  # a project keeps its datasets and steps (a matrix typed in code is not a step)
    s2.objects["annual"], s2.objects["data"] = annual, quarterly
    s2.run(Disaggregate("gdp ~ ip", "annual", method="denton", differences="seasonal", level_penalty=0.01, name="seasonal"))
    assert "differences='seasonal', level_penalty=0.01" in s2.script()
    replayed = _replay(s2, "seasonal", data=quarterly, annual=annual)
    assert np.array_equal(replayed.series, s2["seasonal"].series)
    reopened = load_project(save_project(s2, tmp_path / "td.pymx"))
    assert np.array_equal(reopened["seasonal"].series, s2["seasonal"].series)
    assert Disaggregate("gdp ~ ip", "low", aggregation="C", aggregation_file="x.csv").validate(s)
    assert "no matrix or dataset" in Disaggregate("gdp ~ ip", "low", aggregation="D").validate(s)[0]
    assert "replaces the conversion" in Disaggregate("gdp ~ ip", "low", aggregation="C", conversion="average").validate(s)[0]


def test_the_reconciliation_dialog(accounts, tmp_path):
    from pymetrix.commands import Reconcile, Session, load_project, save_project

    quarterly, annual = accounts
    s = Session()
    s.objects["data"], s.objects["annual"] = quarterly, annual
    rec = s.run(Reconcile("cons, inv, gov", "gdp", low="annual", fixed="gov", method="denton-additive", weights="cons=2"))
    api = reconcile(quarterly, ["cons", "inv", "gov"], "gdp", annual, fixed=["gov"], method="denton-additive", weights={"cons": 2.0})
    pd.testing.assert_frame_equal(rec.series, api.series)
    assert np.array_equal(s["data"]["cons"].to_numpy(), rec.series["cons"].to_numpy())  # the components replaced
    script = s.script()
    assert ("reconciliation = reconcile(data, ['cons', 'inv', 'gov'], 'gdp', low=annual, fixed=['gov'], method='denton-additive', "
            "weights={'cons': 2.0})") in script
    assert "data = data.with_series(cons=reconciliation.series['cons'], inv=reconciliation.series['inv'])" in script
    replayed = _replay(s, "reconciliation", data=quarterly, annual=annual)
    pd.testing.assert_frame_equal(replayed.series, rec.series)
    reopened = load_project(save_project(s, tmp_path / "rec.pymx"))
    pd.testing.assert_frame_equal(reopened["reconciliation"].series, rec.series)
    assert "name=number" in Reconcile("cons, inv", "gdp", weights="cons 2").validate(s)[0]
    assert "two or more" in Reconcile("cons", "gdp").validate(s)[0]
    from pymetrix.ui.render import view_sections

    assert [t for t, _ in view_sections(rec)][:1] == ["Summary"]


def test_the_reconciliation_dialog_records_its_call(accounts):
    from pymetrix.commands import Reconcile, Session

    quarterly, annual = accounts
    s = Session()
    s.objects["data"], s.objects["annual"] = quarterly, annual
    step = Reconcile("cons, inv, gov", "gdp", low="annual", fixed="gov", replace=False, name="rec")
    call = step.to_call()
    assert call.function == "pymetrix:reconcile" and call.output == "rec"
    assert call.to_code() == step.to_code()
    pd.testing.assert_frame_equal(call.run(s).series, s.run(step).series)
