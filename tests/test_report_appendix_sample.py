"""The sample a report's provenance appendix gives for results made of re-estimates: a backtest's origins (it gave
the first two, "2015Q1 to 2015Q2", for origins running to 2022Q4), rolling and recursive fits' windows, combinations
and quantile processes. Checked against the origins and windows counted directly and the samples of fits made
directly on each origin's data, in every report format."""

from dataclasses import replace

import pandas as pd
import pytest

from pymetrix import VAR, Dataset, Equation, Report, backtest, combine_forecasts
from pymetrix.views import provenance_table, sample_text


@pytest.fixture(scope="module")
def data():
    return Dataset.example("sample_macro").with_series(ly="log(y)", lc="log(c)")


@pytest.fixture(scope="module")
def bt(data):
    return backtest(VAR(["ly", "lc"], lags=2).fit(data), data, ("2015Q1", "2016Q4"), horizon=2)


def _direct(data, end):
    """The VAR fitted directly on the data to ``end``: the estimation sample a backtest's re-estimate has there."""
    return VAR(["ly", "lc"], lags=2).fit(data.select(span=(None, end))).provenance.sample


def test_a_backtest_gives_its_origins_and_estimation_samples(data, bt):
    origins = [str(p) for p in pd.period_range("2015Q1", "2016Q4", freq="Q")]
    assert len(origins) == 8 and sorted(bt.errors["origin"].unique()) == origins
    assert bt.provenance.sample == ("2015Q1", "2016Q4")  # was ("2015Q1", "2015Q2"): the first two origins
    first, last = _direct(data, "2015Q1"), _direct(data, "2016Q4")
    assert first == ("2000Q3", "2015Q1") and last == ("2000Q3", "2016Q4")  # the data start in 2000Q1; two lags
    record = bt.provenance.spec["re-estimates"]
    assert record == {"over": "origins", "count": 8, "first": list(first), "last": list(last)}
    expected = "origins 2015Q1 to 2016Q4 (8 re-estimates), estimated on 2000Q3 to 2015Q1 at the first and 2000Q3 to 2016Q4 at the last"
    assert sample_text(bt.provenance) == expected
    assert expected in str(provenance_table(bt.provenance))  # the result's Provenance tab says the same


def test_a_rolling_window_backtest_moves_its_start(data):
    bt = backtest(VAR(["ly", "lc"], lags=2).fit(data), data, ("2015Q1", "2015Q4"), horizon=1, window=30)
    start = lambda end: str(pd.Period(end, "Q") - 29)  # noqa: E731 - the last 30 periods to the origin (lags from before them)
    assert (start("2015Q1"), start("2015Q4")) == ("2007Q4", "2008Q3")
    assert VAR(["ly", "lc"], lags=2).fit(data, span=("2007Q4", "2015Q1")).provenance.sample == ("2007Q4", "2015Q1")
    assert sample_text(bt.provenance) == ("origins 2015Q1 to 2015Q4 (4 re-estimates), estimated on 2007Q4 to 2015Q1 "
                                          "at the first and 2008Q3 to 2015Q4 at the last")


def test_a_backtest_saved_before_the_record_shows_its_origins(bt):
    """A backtest saved before the fix: its stored sample is wrong and it has no record; its origins are in its spec."""
    spec = {k: v for k, v in bt.provenance.spec.items() if k != "re-estimates"}
    old = replace(bt.provenance, spec=spec, sample=("2015Q1", "2015Q2"))
    assert sample_text(old) == "origins 2015Q1 to 2016Q4 (8 re-estimates)"


def test_rolling_and_recursive_fits_give_their_windows(data):
    eq = Equation("c ~ y")
    rolling, recursive = eq.fit_rolling(data, window=40, step=4), eq.fit_recursive(data, min_obs=60, step=10)
    ends = [str(e) for e in pd.period_range("2009Q4", "2024Q4", freq="Q")][::4]
    assert [str(e) for e in rolling.fits] == ends  # 2000Q1 + 39 quarters, every fourth window to the end of the data
    assert eq.fit(data, span=("2000Q1", "2009Q4")).provenance.sample == ("2000Q1", "2009Q4")
    assert sample_text(rolling.provenance) == (f"windows ending 2009Q4 to 2024Q4 ({len(ends)} re-estimates), estimated on 2000Q1 to 2009Q4 "
                                               f"at the first and {pd.Period('2024Q4', 'Q') - 39} to 2024Q4 at the last")
    assert rolling.provenance.nobs == 100 and rolling.provenance.method["type"] == "rolling (40-period) estimation"
    assert [str(e) for e in recursive.fits] == ["2014Q4", "2017Q2", "2019Q4", "2022Q2", "2024Q4"]
    assert sample_text(recursive.provenance) == ("windows ending 2014Q4 to 2024Q4 (5 re-estimates), estimated on 2000Q1 to 2014Q4 "
                                                 "at the first and 2000Q1 to 2024Q4 at the last")
    assert recursive.provenance.nobs == 100 and recursive.provenance.data_fingerprint == recursive.fits[max(recursive.fits)].provenance.data_fingerprint


def test_combinations_and_quantile_processes(data, bt):
    from pymetrix.estimation.quantile import quantile_process

    other = backtest(VAR(["ly", "lc"], lags=1).fit(data), data, ("2015Q1", "2016Q4"), horizon=2)
    combined = combine_forecasts({"two lags": bt, "one lag": other})
    assert sample_text(combined.provenance) == "origins 2015Q1 to 2016Q4 (2 backtests combined)"
    qp = quantile_process(Equation("c ~ y"), data, quantiles=(0.25, 0.75))
    assert sample_text(qp.provenance) == "2000Q1 to 2024Q4" and qp.provenance.method["quantiles"] == [0.25, 0.75]
    assert sample_text(Equation("c ~ y").fit(data, span=("2001Q1", "2020Q4")).provenance) == "2001Q1 to 2020Q4"  # one fit: as before


def test_every_report_format_shows_it(data, bt, tmp_path):
    rolling = Equation("c ~ y").fit_rolling(data, window=40, step=20)
    report = Report("Track record").add(bt.accuracy(), caption="Accuracy")
    report._sources.append(("VAR backtest", bt.provenance))  # as Report.add records a result's provenance, without its graphs
    report._sources.append(("Rolling consumption", rolling.provenance))
    rows = {r[0]: r for r in report._appendix().rows}
    assert rows["VAR backtest"][1] == "pseudo out-of-sample backtest" and rows["VAR backtest"][2] == sample_text(bt.provenance)
    assert rows["VAR backtest"][3] == bt.provenance.nobs == len(bt.errors)
    assert rows["Rolling consumption"][2].startswith("windows ending 2009Q4 to 2024Q4 (4 re-estimates)")
    wanted = "origins 2015Q1 to 2016Q4 (8 re-estimates)"
    for ext in ("md", "html"):
        text = report.save(tmp_path / f"r.{ext}").read_text(encoding="utf-8")
        assert wanted in text and "2015Q1 to 2015Q2" not in text
    if _have("openpyxl"):
        from openpyxl import load_workbook

        cells = [c for r in load_workbook(report.save(tmp_path / "r.xlsx")).worksheets[-1].iter_rows(values_only=True) for c in r]
        assert any(isinstance(c, str) and c.startswith(wanted) for c in cells)
    if _have("docx"):
        import docx

        doc = docx.Document(str(report.save(tmp_path / "r.docx")))
        assert any(c.text.startswith(wanted) for t in doc.tables for row in t.rows for c in row.cells)


def _have(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None
