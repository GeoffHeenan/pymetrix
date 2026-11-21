"""X-13ARIMA-SEATS options, each checked against the program itself: a specification written out here by hand is run
with x13as, and its saved adjusted series (X-11's D11, SEATS' S11) must equal pymetrix's; with every option at its
default, the result is statsmodels' x13_arima_analysis's. Skipped where x13as is not installed (set X13PATH)."""

import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pymetrix_seasonal import X13, x13_program

from pymetrix import Dataset
from pymetrix.forms import build
from pymetrix.plugins import registry

FOLDER = x13_program()
pytestmark = pytest.mark.skipif(FOLDER is None, reason="x13as is not installed (set X13PATH)")


def _series(freq="M", n=180, seed=0):
    rng = np.random.default_rng(seed)
    per = 12 if freq == "M" else 4
    idx = pd.period_range("2005-01" if freq == "M" else "2005Q1", periods=n, freq=freq, name="time")
    season = np.tile(1 + 0.08 * np.sin(2 * np.pi * np.arange(per) / per), n // per + 1)[:n]
    y = 100 * np.exp(0.003 * np.arange(n)) * season * np.exp(rng.normal(0, 0.012, n))
    return Dataset.from_frame(pd.DataFrame({"y": y}, index=idx)), y, idx


def _by_hand(y, idx, body, table, tmp_path):
    """x13as on a specification written here: the series block, then ``body``; the saved ``table``'s values."""
    exe = next(Path(FOLDER) / n for n in ("x13as.exe", "x13as") if (Path(FOLDER) / n).is_file())
    period = 12 if idx.freqstr.startswith("M") else 4
    first = idx[0].month if period == 12 else idx[0].quarter
    values = "\n".join(repr(float(v)) for v in y)
    (tmp_path / "hand.spc").write_text(f"series{{ title=\"hand\" start={idx[0].year}.{first} period={period}\n"
                                       f"  appendbcst=no appendfcst=no data=(\n{values}\n) }}\n{body}\n")
    subprocess.run([str(exe), "hand"], cwd=tmp_path, capture_output=True, timeout=300)
    lines = (tmp_path / f"hand.{table}").read_text().splitlines()[2:]
    return np.array([float(line.split()[1]) for line in lines if line.strip()]), (tmp_path / "hand.out").read_text(errors="replace")


def test_defaults_are_statsmodels_x13():
    x13 = pytest.importorskip("statsmodels.tsa.x13")
    data, y, idx = _series()
    for options in ({}, {"trading": True, "outlier": False}, {"log": False}):
        ours = data.seasonal_adjust("y", X13(**options), derived=False)["y_sa"].to_numpy()
        theirs = x13.x13_arima_analysis(pd.Series(y, index=idx.to_timestamp()), x12path=FOLDER, freq="MS",
                                        log=options.get("log", True), outlier=options.get("outlier", True),
                                        trading=options.get("trading", False)).seasadj.to_numpy()
        assert np.allclose(ours, theirs, rtol=1e-10), options


@pytest.mark.parametrize("options, body, table", [
    ({"log": False, "mode": "add"}, "transform{function=none}\noutlier{}\nautomdl{maxorder=(2 1) maxdiff=(2 1)}\nx11{mode=add save=(d11)}", "d11"),
    ({"seasonal_filter": "s3x9", "trend_filter": 13, "sigma_limits": "1.8 2.8"},
     "transform{function=log}\noutlier{}\nautomdl{maxorder=(2 1) maxdiff=(2 1)}\nx11{seasonalma=s3x9 trendma=13 sigmalim=(1.8 2.8) save=(d11)}",
     "d11"),
    ({"arima": "(0 1 1)(0 1 1)", "outlier_types": "ao ls tc", "critical": 3.5},
     "transform{function=log}\noutlier{types=(ao ls tc) critical=3.5}\narima{model=(0 1 1)(0 1 1)}\nx11{save=(d11)}", "d11"),
    ({"trading": True, "easter": 8, "forecast": 24},
     "transform{function=log}\nregression{variables=(td easter[8])}\noutlier{}\nautomdl{maxorder=(2 1) maxdiff=(2 1)}\n"
     "forecast{maxlead=24}\nx11{save=(d11)}", "d11"),
    ({"max_order": "3 1", "max_diff": "1 1", "outlier": False},
     "transform{function=log}\nautomdl{maxorder=(3 1) maxdiff=(1 1)}\nx11{save=(d11)}", "d11"),
    ({"adjustment": "seats"}, "transform{function=log}\noutlier{}\nautomdl{maxorder=(2 1) maxdiff=(2 1)}\nseats{save=(s11)}", "s11"),
])
def test_each_option_against_the_program(options, body, table, tmp_path):
    data, y, idx = _series()
    out = data.seasonal_adjust("y", X13(**options), derived=False)
    hand, _ = _by_hand(y, idx, body, table, tmp_path)
    assert np.allclose(out["y_sa"].to_numpy(), hand, rtol=1e-8), options
    seasonal = out["y_seasonal"].to_numpy()
    if options.get("mode") == "add":  # an additive component: y - sa; otherwise a factor: y / sa
        assert np.allclose(seasonal, y - hand, atol=1e-6) and abs(seasonal.mean()) < 5
    else:
        assert np.allclose(seasonal, y / hand, rtol=1e-8) and abs(seasonal.mean() - 1) < 0.05
    spec = out.variables["y_sa"].extra["backend_options"]
    assert "data=(...)" in spec and ("seats{" in spec) == (options.get("adjustment") == "seats")


def test_the_programs_own_refusals_reach_the_user():
    data, _, _ = _series("Q", 80, seed=3)
    with pytest.raises(ValueError, match="could not adjust y: The innovation variance of the seasonal adjustment is less than zero"):
        data.seasonal_adjust("y", X13(adjustment="seats"), derived=False)  # SEATS's message does not start with ERROR
    data, _, _ = _series()
    with pytest.raises(ValueError, match="Pseudo-additive seasonal adjustment cannot be performed"):
        data.seasonal_adjust("y", X13(mode="pseudoadd"), derived=False)  # X-13's rule: not with regARIMA preadjustment


def test_quarterly_seats_and_extra_spec_text(tmp_path):
    data, y, idx = _series("Q", 100, seed=1)
    out = data.seasonal_adjust("y", X13(adjustment="seats", spec="check{print=all}"), derived=False)
    hand, output = _by_hand(y, idx, "transform{function=log}\noutlier{}\nautomdl{maxorder=(2 1) maxdiff=(2 1)}\nseats{save=(s11)}\n"
                            "check{print=all}", "s11", tmp_path)
    assert np.allclose(out["y_sa"].to_numpy(), hand, rtol=1e-8)
    assert out.variables["y_sa"].extra["x13_model"] and "check{print=all}" in out.variables["y_sa"].extra["backend_options"]


def test_settings_from_the_dialog_and_refusals():
    method = build(registry("seasonal").get("X13"), "mode='add', seasonal_filter='s3x5', trend_filter=13, easter=8, arima='(0 1 1)(0 1 1)'")
    assert (method.mode, method.seasonal_filter, method.trend_filter, method.easter, method.arima) == ("add", "s3x5", 13, 8, "(0 1 1)(0 1 1)")
    for bad, message in (({"adjustment": "seats", "mode": "add"}, "X-11's"), ({"trend_filter": 12}, "odd"),
                         ({"seasonal_filter": "s4x4"}, "seasonal_filter"), ({"arima": "011"}, "arima"),
                         ({"outlier_types": "ao xx"}, "outlier_types"), ({"max_order": "2"}, "two numbers")):
        with pytest.raises(ValueError, match=re.escape(message)):
            X13(**bad)
