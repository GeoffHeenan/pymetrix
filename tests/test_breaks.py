"""Structural breaks at unknown dates, checked independently: Quandt-Andrews' simulated critical values against
Andrews' (1993) published ones; its statistic against a Chow test at every date; Bai-Perron's dynamic programming
against a brute-force search over every partition; and known breaks recovered from simulated data."""

import itertools

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, bai_perron
from pymetrix.errors import SpecificationError
from pymetrix.stats.breaks import (
    _null_quantiles,
    _segment_ssr,
    _sup_wald_null,
    partition,
)


def _data(n=200, breaks=(120,), levels=(1.0, 3.0), seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    regime = np.searchsorted(np.asarray(breaks), np.arange(n), side="right")
    y = np.asarray(levels)[regime] + 0.5 * x + rng.normal(0, 1, n)
    return Dataset.from_frame(pd.DataFrame({"y": y, "x": x}, index=pd.period_range("1970Q1", periods=n, freq="Q", name="time")))


def test_the_simulated_null_matches_andrews_table():
    # Andrews (1993), as in Stock and Watson's QLR table with 15% trimming (F = Wald / q): q = 1: 7.12, 8.68, 12.16;
    # q = 2: 5.00, 5.86, 7.78
    one = _sup_wald_null(1, 0.15)
    assert [np.quantile(one, q) for q in (0.90, 0.95, 0.99)] == pytest.approx([7.12, 8.68, 12.16], rel=0.02)
    two = _sup_wald_null(2, 0.15) / 2
    assert [np.quantile(two, q) for q in (0.90, 0.95)] == pytest.approx([5.00, 5.86], rel=0.02)


def test_the_shipped_table_matches_andrews_table():
    # the same published values as above, read from the shipped table (quandt_andrews.npz) instead of simulated
    for k, published in ((1, [7.12, 8.68, 12.16]), (2, [5.00, 5.86, 7.78])):
        probs, quantiles, tabulated = _null_quantiles(k, 0.15)
        assert tabulated
        assert [np.interp(q, probs, quantiles) / k for q in (0.90, 0.95, 0.99)] == pytest.approx(published, rel=0.02)


def test_the_table_agrees_with_a_fresh_simulation():
    # off the table's trimmings the test simulates; just beside one, the two must agree (a sign the table is right)
    probs, table, _ = _null_quantiles(1, 0.15)
    _, simulated, tabulated = _null_quantiles(1, 0.151)
    assert not tabulated
    assert [np.interp(q, probs, simulated) for q in (0.90, 0.95, 0.99)] == pytest.approx(
        [np.interp(q, probs, table) for q in (0.90, 0.95, 0.99)], rel=0.03)


def test_quandt_andrews_is_the_largest_chow_statistic():
    data = _data()
    fit = Equation("y ~ x").fit(data)
    qa = fit.test("quandt_andrews")
    n = 200
    candidates = range(int(np.ceil(0.15 * n)), int(np.floor(0.85 * n)) + 1)
    chows = {t: fit.test("chow", at=str(data.frame.index[t])).statistic for t in candidates}  # each date's own Chow test
    best = max(chows, key=chows.get)
    assert qa.statistic == pytest.approx(chows[best], rel=1e-10) and qa.details["break date"] == str(data.frame.index[best])
    assert qa.details["break date"] == "2000Q1" and qa.p_value < 0.001
    calm = Equation("y ~ x").fit(_data(breaks=(), levels=(1.0,), seed=3)).test("quandt_andrews")
    assert calm.p_value > 0.05 and "5%" in calm.critical_values
    assert "(tabulated)" in qa.distribution
    assert "p-value is below" in qa.details["note"] and "note" not in calm.details  # qa's sup F is beyond the table


def test_the_dynamic_programme_finds_the_brute_force_partition():
    rng = np.random.default_rng(9)
    n, h = 48, 6
    x = rng.normal(size=n)
    y = np.r_[np.full(15, 0.0), np.full(18, 2.0), np.full(15, -1.0)] + x + rng.normal(0, 0.5, n)
    X = np.column_stack([np.ones(n), x])

    def ssr(a, b):
        coef = np.linalg.lstsq(X[a:b], y[a:b], rcond=None)[0]
        return float(np.sum((y[a:b] - X[a:b] @ coef) ** 2))

    S = _segment_ssr(y, X, h)
    for m in (1, 2):
        best = min((sum(ssr(a, b) for a, b in zip((0, *c), (*c, n))), c)
                   for c in itertools.combinations(range(h, n - h + 1), m) if all(b - a >= h for a, b in zip((0, *c), (*c, n))))
        total, dates = partition(y, X, m, h, S)
        assert total == pytest.approx(best[0], rel=1e-9) and tuple(dates) == best[1]


def test_bai_perron_recovers_two_breaks_and_chooses_their_number():
    data = _data(n=240, breaks=(80, 160), levels=(0.0, 3.0, 1.0), seed=5)
    found = bai_perron(Equation("y ~ x").fit(data), max_breaks=4)
    assert found.chosen == 2
    assert [abs((pd.Period(d, "Q") - pd.Period(t, "Q")).n) <= 2 for d, t in zip(found.dates, ("1990Q1", "2010Q1"))] == [True, True]
    table = found.segments()
    assert len(table.rows) == 6 and table.index[0].endswith(": const")
    consts = [r[0] for i, r in zip(table.index, table.rows) if i.endswith(": const")]
    assert consts == pytest.approx([0.0, 3.0, 1.0], abs=0.35)
    assert list(found.by_count.index) == [0, 1, 2, 3, 4] and found.summary().notes[0].startswith("Chosen by BIC: 2")
    with pytest.raises(SpecificationError, match="trim"):
        bai_perron(Equation("y ~ x").fit(data), trim=0.6)


def test_the_dialog_records_the_call():
    from pymetrix.commands import FindBreaks, FitEquation, Session

    s = Session()
    s.objects["data"] = _data()
    s.run(FitEquation("y ~ x", name="fit"))
    s.run(FindBreaks("fit", max_breaks=3))
    assert s.history[-1].code == "breaks = bai_perron(fit, max_breaks=3)"
    assert s.objects["breaks"].dates == ["2000Q1"]


def test_the_nile_breaks_where_strucchange_finds_it():
    # R: breakpoints(Nile ~ 1) in strucchange: one break chosen by BIC, after observation 28 (1898), so the new
    # regime starts in 1899; the flow falls from about 1097 to about 850
    nile = Dataset.example("nile")
    fit = Equation("volume ~ 1").fit(nile)
    found = fit.breaks()
    assert found.chosen == 1 and found.dates == ["1899"]
    means = [r[0] for r in found.segments().rows]
    assert means == pytest.approx([1097.75, 849.97], abs=0.01)
    qa = fit.test("quandt_andrews")
    assert qa.details["break date"] == "1899" and qa.p_value < 0.001
