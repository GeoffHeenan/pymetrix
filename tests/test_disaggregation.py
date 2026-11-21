"""Temporal disaggregation: Chow-Lin, Fernandez, Denton; the totals always hold."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, TemporalDisaggregation
from pymetrix.disaggregation import _ar1_cov, conversion_matrix


@pytest.fixture(scope="module")
def economy():
    """Quarterly truth y = 10 + 2 ip + AR(1) (rho 0.8); annual sums observed to 2023; ip to 2024Q3."""
    rng = np.random.default_rng(3)
    q = pd.period_range("2000Q1", "2023Q4", freq="Q", name="time")
    n = len(q)
    ip = 100 + np.cumsum(rng.normal(0.5, 1, n + 3))
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.8 * u[t - 1] + rng.normal(0, 1)
    y = 10 + 2 * ip[:n] + u
    quarterly = Dataset.from_frame(pd.DataFrame({"ip": ip}, index=pd.period_range("2000Q1", periods=n + 3, freq="Q", name="time")))
    sums = pd.Series(y, index=q).groupby(q.year).sum()
    annual = Dataset.from_frame(pd.DataFrame({"gdp": sums.to_numpy()}, index=pd.period_range("2000", "2023", freq="Y", name="time")))
    return annual, quarterly, pd.Series(y, index=q)


@pytest.mark.parametrize("method", ["chow-lin", "chow-lin-minrss", "fernandez", "denton", "denton-additive"])
def test_every_method_meets_the_totals(economy, method):
    annual, quarterly, truth = economy
    fit = TemporalDisaggregation("gdp ~ ip", method=method).fit(annual, quarterly)
    assert fit.check < 1e-8
    sums = fit.series.loc[truth.index].groupby(truth.index.year).sum()
    assert np.allclose(sums.to_numpy(), annual["gdp"].to_numpy())
    assert np.corrcoef(fit.series.loc[truth.index], truth)[0, 1] > 0.998
    if method in ("chow-lin", "chow-lin-minrss", "fernandez"):
        assert str(fit.series.index[-1]) == "2024Q3"  # extrapolated with the indicator


def test_chow_lin_estimates_and_its_gls_step(economy):
    sm = pytest.importorskip("statsmodels.api")
    annual, quarterly, _ = economy
    fit = TemporalDisaggregation("gdp ~ ip").fit(annual, quarterly)
    assert 0.6 < fit.rho < 0.95 and fit.params[1] == pytest.approx(2.0, abs=0.1)
    fixed = TemporalDisaggregation("gdp ~ ip", rho=0.7).fit(annual, quarterly)
    lo = pd.period_range("2000", "2023", freq="Y")
    C = conversion_matrix(lo, quarterly.frame.index, "sum")
    X = np.column_stack([np.ones(len(quarterly)), quarterly["ip"].to_numpy()])
    ref = sm.GLS(annual["gdp"].to_numpy(), C @ X, sigma=C @ _ar1_cov(len(quarterly), 0.7) @ C.T).fit()
    assert np.allclose(fixed.params, ref.params) and np.allclose(np.sqrt(np.diag(fixed.cov)), ref.bse)
    assert "rho = " in str(fit.summary()) and "maximum likelihood" in str(fit.summary())


def test_denton_recovers_a_series_proportional_to_its_indicator(economy):
    annual, quarterly, _ = economy
    truth = 1.7 * quarterly["ip"].loc["2000Q1":"2023Q4"]
    sums = truth.groupby(truth.index.year).sum()
    target = Dataset.from_frame(pd.DataFrame({"gdp": sums.to_numpy()}, index=pd.period_range("2000", "2023", freq="Y", name="time")))
    fit = TemporalDisaggregation("gdp ~ ip", method="denton").fit(target, quarterly.select(span=(None, "2023Q4")))
    assert np.allclose(fit.series, truth, rtol=1e-9)
    smooth = TemporalDisaggregation("gdp ~ 1", method="denton").fit(target, quarterly.select(span=(None, "2023Q4")))
    assert smooth.check < 1e-8 and np.std(np.diff(smooth.series, 2)) < np.std(np.diff(fit.series, 2)) + 1


def test_conversions(economy):
    annual, quarterly, truth = economy
    averages = Dataset.from_frame(pd.DataFrame({"rate": truth.groupby(truth.index.year).mean().to_numpy()},
                                               index=pd.period_range("2000", "2023", freq="Y", name="time")))
    fit = TemporalDisaggregation("rate ~ ip", conversion="average").fit(averages, quarterly)
    assert np.allclose(fit.series.loc[truth.index].groupby(truth.index.year).mean(), averages["rate"].to_numpy())
    last = Dataset.from_frame(pd.DataFrame({"stock": truth.groupby(truth.index.year).last().to_numpy()},
                                           index=pd.period_range("2000", "2023", freq="Y", name="time")))
    fit = TemporalDisaggregation("stock ~ ip", conversion="last").fit(last, quarterly)
    assert np.allclose(fit.series.loc[[p for p in truth.index if p.quarter == 4]], last["stock"].to_numpy())
    C = conversion_matrix(pd.period_range("2020", "2021", freq="Y"), pd.period_range("2020Q1", "2021Q4", freq="Q"), "first")
    assert C.tolist() == [[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0]]
    with pytest.raises(ValueError, match="not fully covered"):
        conversion_matrix(pd.period_range("2020", "2021", freq="Y"), pd.period_range("2020Q1", "2021Q3", freq="Q"), "sum")


def test_errors(economy):
    annual, quarterly, _ = economy
    with pytest.raises(KeyError, match="no series 'gnp'"):
        TemporalDisaggregation("gnp ~ ip").fit(annual, quarterly)
    with pytest.raises(ValueError, match="one indicator"):
        TemporalDisaggregation("gdp ~ ip + ip[-1]", method="denton").fit(annual, quarterly)


def test_dialog_and_view(economy, tmp_path):
    from pymetrix.commands import Disaggregate, Session, load_project, save_project

    annual, quarterly, _ = economy
    s = Session()
    s.objects["annual"], s.objects["data"] = annual, quarterly
    s.run(Disaggregate("gdp ~ ip", "annual"))
    assert "gdp" in s["data"].names and s["disaggregation"].check < 1e-8
    script = s.script()
    assert "disaggregation = TemporalDisaggregation(formula='gdp ~ ip').fit(annual, data)" in script
    assert "data = data.with_series(gdp=disaggregation.series)" in script
    reopened = load_project(save_project(s, tmp_path / "td.pymx"))
    assert np.allclose(reopened["data"]["gdp"].dropna(), s["data"]["gdp"].dropna())
    from pymetrix.ui.render import view_sections

    titles = [t for t, _ in view_sections(s["disaggregation"])]
    assert titles[0] == "Summary" and titles[-1] == "Series"
