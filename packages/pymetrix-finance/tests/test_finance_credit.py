"""Credit risk (the finance plugin), checked independently: Merton's equations round-tripped through Black-Scholes;
the KMV iteration recovering a simulated firm's asset path and volatility; CDS intensities that reprice every CDS at
zero, equal to the credit triangle on a flat curve without discounting; and the dialogs."""

import importlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pymetrix import Dataset, cds_curve, merton

cr = importlib.import_module("pymetrix_finance.credit")


def test_merton_round_trip_and_the_kmv_iteration():
    from pymetrix_finance.options import black_scholes

    V, sv, D, r = 120.0, 0.25, 80.0, 0.03
    E = black_scholes(V, D, 1, r, sv)
    d1 = (np.log(V / D) + r + sv**2 / 2) / sv
    point = cr.merton_point(E, stats.norm.cdf(d1) * sv * V / E, D, r)
    assert (point["assets"], point["asset vol"]) == pytest.approx((V, sv), rel=1e-9)
    assert point["PD"] == pytest.approx(stats.norm.cdf(-(np.log(V / D) + r - sv**2 / 2) / sv))

    rng = np.random.default_rng(0)
    T = 500
    Vt = 100 * np.exp(np.cumsum(rng.normal((0.05 - 0.3**2 / 2) / 252, 0.3 / np.sqrt(252), T)))
    Et = np.array([black_scholes(v, 70.0, 1, 0.02, 0.3) for v in Vt])
    idx = pd.PeriodIndex(pd.bdate_range("2020-01-01", periods=T), freq="D", name="time")
    m = merton(Dataset.from_frame(pd.DataFrame({"mcap": Et, "debt": np.full(T, 70.0)}, index=idx)), "mcap", "debt", rate=0.02,
               periods_per_year=252)
    realised = np.std(np.diff(np.log(Vt)), ddof=1) * np.sqrt(252)
    assert m.asset_vol == pytest.approx(realised, rel=0.02)  # the fixed point is near the path's own volatility
    assert m.frame["assets"].to_numpy() == pytest.approx(Vt, rel=0.02)


def test_cds_bootstrap():
    flat = cds_curve([5], [0.012], recovery=0.4, rate=0.0)
    assert flat.hazards[0] == pytest.approx(0.012 / 0.6, rel=1e-6)  # the credit triangle, exactly with accrual and no discounting
    c = cds_curve([1, 3, 5, 7, 10], [0.008, 0.011, 0.014, 0.015, 0.016], recovery=0.4, rate=0.03)
    for i, (mat, s) in enumerate(zip(c.maturities, c.spreads)):  # each CDS repriced at zero by the bootstrapped curve
        prem, prot = cr._premium_and_protection(c.maturities[:i + 1], lambda t: float(c.hazard(t)[0]), 0.03, 0.4)
        assert s * prem[-1] - prot[-1] == pytest.approx(0, abs=1e-12)
    assert c.survival([0])[0] == 1 and c.survival([10])[0] == pytest.approx(np.exp(-np.sum(c.hazards * np.diff([0, 1, 3, 5, 7, 10]))))


def test_the_dialogs():
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS

    s = Session()
    s.run(COMMANDS["BootstrapCDS"].cls(spreads="80, 110, 140, 150, 160", recovery=0.4, rate=0.03))
    assert s.objects["cds"].spreads[0] == pytest.approx(0.008)
    assert s.history[-1].code.startswith("cds = cds_curve([1.0, 3.0, 5.0, 7.0, 10.0], [0.008")
