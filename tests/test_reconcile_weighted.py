"""Weighted least-squares reconciliation, ``reconcile(..., method="weighted")``: min sum_c sum_t e_ct^2 / w_c under the
same constraints as the other methods (the components add up to the total each period, keep their low-frequency
totals, fixed ones held).

The references are written here by other routes: the full KKT system built with NumPy; with equal weights the
minimum-norm adjustment (``numpy.linalg.lstsq`` and the pseudo-inverse); with only the total to meet, each period's
gap shared in proportion to the weights (by hand). A component entering with a negative sign (imports) is adjusted
as one of the same size with a positive sign. The other methods' results are recorded from before the method was
added, so a change to them shows."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, reconcile
from pymetrix.disaggregation import conversion_matrix
from pymetrix.errors import SpecificationError

COMPONENTS = ["cons", "inv", "gov", "imp"]


@pytest.fixture(scope="module")
def accounts():
    """Quarterly 2010-2019: consumption, investment, government and imports (negative), their total and annual totals;
    preliminary quarterly values that miss both; government right already (it can be held fixed)."""
    rng = np.random.default_rng(23)
    q = pd.period_range("2010Q1", "2019Q4", freq="Q", name="time")
    n = len(q)
    levels = {"cons": 300.0, "inv": 120.0, "gov": 90.0, "imp": -150.0}
    truth = {c: lv + np.cumsum(rng.normal(0.4, 1, n)) * np.sign(lv) for c, lv in levels.items()}
    prelim = {c: v + rng.normal(0, 2, n) for c, v in truth.items()}
    prelim["gov"] = truth["gov"]
    quarterly = Dataset.from_frame(pd.DataFrame({**prelim, "gdp": sum(truth.values())}, index=q))
    annual = Dataset.from_frame(pd.DataFrame({c: pd.Series(v, index=q).groupby(q.year).sum().to_numpy() for c, v in truth.items()},
                                             index=pd.period_range("2010", "2019", freq="Y", name="time")))
    return quarterly, annual


def _kkt(P, z, weights, C=None, lows=None):
    """min sum_j (y_j - p_j)' (y_j - p_j) / w_j subject to sum_j y_j = z and C y_j = low_j, from the full KKT system
    [[A, G'], [G, 0]] [y; lambda] = [A p; g]."""
    k, n = len(P), len(z)
    A = np.diag(np.concatenate([np.full(n, 1 / w) for w in weights]))
    G, g = [np.hstack([np.eye(n)] * k)], [z]
    for j, low in enumerate(lows or []):
        if low is not None:
            row = np.zeros((C.shape[0], k * n))
            row[:, j * n : (j + 1) * n] = C
            G.append(row)
            g.append(low)
    G, g = np.vstack(G), np.concatenate(g)
    m = G.shape[0]
    K = np.block([[A, G.T], [G, np.zeros((m, m))]])
    p = np.concatenate(P)
    sol = np.linalg.lstsq(K, np.r_[A @ p, g], rcond=None)[0]
    return sol[: k * n].reshape(k, n)


def test_weighted_equals_the_kkt_solution_and_meets_the_constraints(accounts):
    quarterly, annual = accounts
    weights = {"cons": 1.0, "inv": 4.0, "imp": 2.5}
    rec = reconcile(quarterly, COMPONENTS, "gdp", annual, fixed=["gov"], method="weighted", weights=weights)
    free = ["cons", "inv", "imp"]
    z = quarterly["gdp"].to_numpy() - quarterly["gov"].to_numpy()
    C = conversion_matrix(annual.frame.index, quarterly.frame.index, "sum")
    ref = _kkt([quarterly[c].to_numpy() for c in free], z, [weights[c] for c in free], C, [annual[c].to_numpy() for c in free])
    assert np.max(np.abs(rec.series[free].to_numpy().T - ref)) < 1e-10
    # rows add to the total; low-frequency totals unchanged; the fixed component does not move
    total = quarterly["gdp"].to_numpy()
    assert np.max(np.abs(rec.series.sum(axis=1).to_numpy() - total)) < 1e-10
    for c in free:
        yearly = rec.series[c].groupby(rec.series.index.year).sum().to_numpy()
        assert np.max(np.abs(yearly - annual[c].to_numpy())) < 1e-10
    assert np.array_equal(rec.series["gov"].to_numpy(), quarterly["gov"].to_numpy())
    assert rec.check < 1e-10 and rec.method == "weighted" and rec.provenance.method["method"] == "weighted"
    assert "weighted least squares" in str(rec.summary())


def test_equal_weights_give_the_minimum_norm_adjustment(accounts):
    quarterly, annual = accounts
    n = len(quarterly)
    C = conversion_matrix(annual.frame.index, quarterly.frame.index, "sum")
    k = len(COMPONENTS)
    G = np.vstack([np.hstack([np.eye(n)] * k)] + [np.hstack([C if i == j else np.zeros_like(C) for i in range(k)]) for j in range(k)])
    p = np.concatenate([quarterly[c].to_numpy() for c in COMPONENTS])
    g = np.concatenate([quarterly["gdp"].to_numpy()] + [annual[c].to_numpy() for c in COMPONENTS])
    by_lstsq = np.linalg.lstsq(G, g - G @ p, rcond=None)[0]  # the smallest |e| with G (p + e) = g
    # G is rank-deficient (the components' annual totals add up to the total's), so the pseudo-inverse drops the
    # singular values at rounding level, as lstsq does
    by_pinv = np.linalg.pinv(G, rcond=1e-10) @ (g - G @ p)
    for scale in (1.0, 3.0):  # equal weights of any size give the same answer
        rec = reconcile(quarterly, COMPONENTS, "gdp", annual, method="weighted", weights=dict.fromkeys(COMPONENTS, scale))
        e = rec.adjustments[COMPONENTS].to_numpy().T.ravel()
        assert np.max(np.abs(e - by_lstsq)) < 1e-10 and np.max(np.abs(e - by_pinv)) < 1e-10
    unweighted = reconcile(quarterly, COMPONENTS, "gdp", annual, method="weighted")  # weights default to 1 each
    np.testing.assert_allclose(unweighted.adjustments[COMPONENTS].to_numpy().T.ravel(), by_lstsq, rtol=0, atol=1e-10)


def test_only_the_total_shares_each_gap_in_proportion_to_the_weights(accounts):
    """With the total alone to meet, e_ct = w_c / sum_d w_d * gap_t, whatever the components' levels and signs."""
    quarterly, _ = accounts
    weights = {"cons": 1.0, "inv": 4.0, "gov": 0.5, "imp": 2.5}
    rec = reconcile(quarterly, COMPONENTS, "gdp", method="weighted", weights=weights)
    P = quarterly.frame[COMPONENTS].to_numpy()
    gap = quarterly["gdp"].to_numpy() - P.sum(axis=1)
    w = np.array([weights[c] for c in COMPONENTS])
    np.testing.assert_allclose(rec.adjustments[COMPONENTS].to_numpy(), np.outer(gap, w / w.sum()), rtol=0, atol=1e-10)


def test_a_negative_component_is_adjusted_as_a_positive_one_of_the_same_size(accounts):
    """imp (negative) and its mirror |imp| entering with a plus sign, with the total and the annual totals shifted so
    each constraint's gap is the same: the same adjustments, to rounding."""
    quarterly, annual = accounts
    imp, imp_low = quarterly["imp"].to_numpy(), annual["imp"].to_numpy()
    assert (imp < 0).all()
    flipped_q = quarterly.with_series(imp=pd.Series(-imp, index=quarterly.frame.index),
                                      gdp=quarterly["gdp"] - 2 * quarterly["imp"])
    C = conversion_matrix(annual.frame.index, quarterly.frame.index, "sum")
    flipped_a = annual.with_series(imp=pd.Series(imp_low - 2 * C @ imp, index=annual.frame.index))  # same gap: low - C x
    weights = {"cons": 1.0, "inv": 4.0, "imp": 2.5}
    for low_q, low_a in ((quarterly, None), (quarterly, annual)):
        a = reconcile(low_q, COMPONENTS, "gdp", low_a, fixed=["gov"], method="weighted", weights=weights)
        b = reconcile(flipped_q, COMPONENTS, "gdp", flipped_a if low_a is not None else None, fixed=["gov"], method="weighted",
                      weights=weights)
        np.testing.assert_allclose(a.adjustments.to_numpy(), b.adjustments.to_numpy(), rtol=0, atol=1e-10)
        assert b.check < 1e-10


def test_weights_are_positive(accounts):
    quarterly, _ = accounts
    for bad in (0, -1.0):
        with pytest.raises(SpecificationError, match="weights are positive.*fixed"):
            reconcile(quarterly, COMPONENTS, "gdp", method="weighted", weights={"inv": bad})


# Recorded with the code before method="weighted" was added (commit f17ba45), on the data of
# tests/test_disaggregation_extensions.py's accounts fixture: cons in the first quarter, inv in the last, the sum of all.
# Denton without low-frequency totals is left out: a constant moved from one component to another changes neither the
# total nor any first difference, so its minimum is not unique and the one returned depends on the LAPACK build.
RECORDED = {
    ("denton", True): (97.81671919233833, 223.1582035282029, 38215.10149229319),
    ("denton-additive", True): (98.5788442325442, 223.457410064004, 38215.10149229319),
    ("least-squares", False): (99.63671986662813, 224.61312891772417, 38215.10149229319),
    ("least-squares", True): (98.38301562488311, 223.02992592907458, 38215.10149229319),
}


@pytest.mark.parametrize(("method", "with_low"), list(RECORDED))
def test_the_other_methods_are_unchanged(method, with_low):
    rng = np.random.default_rng(5)
    q = pd.period_range("2005Q1", "2019Q4", freq="Q", name="time")
    n = len(q)
    truth = {c: 100 * (i + 1) + np.cumsum(rng.normal(0.5, 1, n)) for i, c in enumerate(["cons", "inv", "gov"])}
    prelim = {c: v * (1 + rng.normal(0, 0.02, n)) for c, v in truth.items()}
    prelim["gov"] = truth["gov"]
    quarterly = Dataset.from_frame(pd.DataFrame({**prelim, "gdp": sum(truth.values())}, index=q))
    annual = Dataset.from_frame(pd.DataFrame({c: pd.Series(v, index=q).groupby(q.year).sum().to_numpy() for c, v in truth.items()},
                                             index=pd.period_range("2005", "2019", freq="Y", name="time")))
    kw = {"low": annual, "fixed": ["gov"], "weights": {"inv": 4.0}} if with_low else {}
    out = reconcile(quarterly, ["cons", "inv", "gov"], "gdp", method=method, **kw).series
    first, last, total = RECORDED[(method, with_low)]
    assert out["cons"].iloc[0] == pytest.approx(first, rel=1e-12)
    assert out["inv"].iloc[-1] == pytest.approx(last, rel=1e-12)
    assert float(out.to_numpy().sum()) == pytest.approx(total, rel=1e-12)
    default = reconcile(quarterly, ["cons", "inv", "gov"], "gdp", **kw).series  # the default is still denton
    pd.testing.assert_frame_equal(default, reconcile(quarterly, ["cons", "inv", "gov"], "gdp", method="denton", **kw).series)


def test_the_dialog_offers_weighted_and_its_script_replays(accounts, tmp_path):
    from pymetrix.commands import Reconcile, Session, load_project, save_project
    from pymetrix.wording import choice_label

    quarterly, annual = accounts
    s = Session()
    s.objects["data"], s.objects["annual"] = quarterly, annual
    step = Reconcile("cons, inv, gov, imp", "gdp", low="annual", fixed="gov", method="weighted", weights="inv=4, imp=2.5")
    rec = s.run(step)
    api = reconcile(quarterly, COMPONENTS, "gdp", annual, fixed=["gov"], method="weighted", weights={"inv": 4.0, "imp": 2.5})
    pd.testing.assert_frame_equal(rec.series, api.series)
    script = s.script()
    assert ("reconciliation = reconcile(data, ['cons', 'inv', 'gov', 'imp'], 'gdp', low=annual, fixed=['gov'], method='weighted', "
            "weights={'inv': 4.0, 'imp': 2.5})") in script
    call = Reconcile("cons, inv, gov, imp", "gdp", low="annual", fixed="gov", method="weighted", weights="inv=4, imp=2.5",
                     replace=False, name="rec").to_call()
    assert call.arguments["method"] == "weighted" and "method='weighted'" in call.to_code()
    pd.testing.assert_frame_equal(call.run(s).series, rec.series)
    ns = {"data": quarterly, "annual": annual}
    exec(script, ns)  # noqa: S102 - the session's own script
    pd.testing.assert_frame_equal(ns["reconciliation"].series, rec.series)
    reopened = load_project(save_project(s, tmp_path / "weighted.pymx"))
    pd.testing.assert_frame_equal(reopened["reconciliation"].series, rec.series)
    assert choice_label("Reconcile", "method", "weighted") == "weighted least squares (in proportion to the weights)"
