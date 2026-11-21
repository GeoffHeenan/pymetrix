"""X-13ARIMA-SEATS in the seasonal plugin: the program found or asked for, what its runs report (version, model,
outliers), its failures named by series, and its provenance in a panel. The runs are skipped where x13as is not
installed (set X13PATH)."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pymetrix_seasonal import X13, x13_notes, x13_program

from pymetrix import STL, Dataset
from pymetrix.errors import DataError
from pymetrix.metadata import Variable
from pymetrix.views import view_sections


@pytest.fixture(scope="module")
def monthly():
    rng = np.random.default_rng(0)
    idx = pd.period_range("2005-01", "2024-12", freq="M", name="time")
    n = len(idx)
    factor = np.tile(1 + 0.1 * np.sin(2 * np.pi * np.arange(12) / 12), n // 12)
    trend = 100 * np.exp(0.002 * np.arange(n))
    frame = pd.DataFrame({"sales": trend * factor * np.exp(rng.normal(0, 0.01, n)), "balance": rng.normal(0, 1, n) + 5 * (factor - 1)}, index=idx)
    return Dataset.from_frame(frame), factor


def test_x13_needs_its_program(monthly, monkeypatch):
    data, _ = monthly
    if x13_program() is not None:  # pragma: no cover - only where x13as is installed
        out = data.seasonal_adjust("sales", X13())
        assert np.corrcoef(out["sales_seasonal"], monthly[1])[0, 1] > 0.99
        return
    monkeypatch.delenv("X13PATH", raising=False)
    with pytest.raises(ValueError, match="needs the x13as program"):
        data.seasonal_adjust("sales", X13(), derived=False)


def test_x13_notes_read_the_version_model_and_outliers():
    console = b"\r\n X-13ARIMA-SEATS Seasonal Adjustment Program\r\n Version Number 1.1 Build 62\r\n"
    output = ("  Final automatic model choice : (1 1 1)(0 1 1)\n Automatically Identified Outliers\n"
              "   AO2005.Jan                   0.0928         0.00990         9.38\n"
              "   LS2009.Mar                  -0.0400         0.00990        -4.04\n ------\n\n ARIMA Model:  (0 1 1)(0 1 1)\n")
    assert x13_notes(console, output) == {"x13_version": "1.1 Build 62", "x13_model": "(0 1 1)(0 1 1)", "x13_outliers": "AO2005.Jan, LS2009.Mar"}
    looked = "   131: outlier{}\n  Final automatic model choice : (0 1 1)(0 1 1)\n"
    assert x13_notes("", looked) == {"x13_model": "(0 1 1)(0 1 1)", "x13_outliers": "none"}
    assert x13_notes("", "                          Release Version 1.1 Build 61\n") == {"x13_version": "1.1 Build 61"}


def _x13_folder():
    """The x13as program: X13PATH or the PATH, or the copy kept in the repository's .tools (not committed)."""
    local = Path(__file__).resolve().parents[3] / ".tools" / "x13as" / "x13as"
    return x13_program() or (x13_program(str(local)) if local.is_dir() else None)


def _run_x13_directly(folder, y, start, tmp_path):
    """x13as run on a spec of our own, without statsmodels: its console, its output, and the adjusted series (D11)."""
    exe = next(Path(folder) / n for n in ("x13as.exe", "x13as", "x13as_ascii.exe", "x13as_ascii") if (Path(folder) / n).is_file())
    values = "\n".join(f"{v:.10g}" for v in y)
    (tmp_path / "own.spc").write_text(f"series{{ title=\"own\" start={start.year}.{start.month:02d} period=12 data=(\n{values}\n) }}\n"
                                      "transform{ function=log }\nautomdl{ }\noutlier{ }\nx11{ save=(d11) }\n")
    run = subprocess.run([str(exe), "own"], cwd=tmp_path, capture_output=True, timeout=120)
    output = (tmp_path / "own.out").read_text(errors="replace")
    table = (tmp_path / "own.d11").read_text().splitlines()[2:]  # two header lines, then date and value
    return run.stdout.decode("latin-1"), output, np.array([float(line.split()[1]) for line in table])


def test_x13_provenance_matches_the_program_run_directly(monthly, tmp_path):
    folder = _x13_folder()
    if folder is None:
        pytest.skip("x13as is not installed")
    data, _ = monthly
    y = data["sales"].to_numpy().copy()
    y[60] *= 1.15  # an outlier, in January 2010
    data = data.with_series(sales=pd.Series(y, index=data.frame.index))
    out = data.seasonal_adjust("sales", X13(program=folder))
    console, output, d11 = _run_x13_directly(folder, y, data.frame.index[0], tmp_path)
    extra = out.variables["sales_sa"].extra
    assert extra["x13_program"] == folder
    assert extra["x13_version"] in " ".join(console.split())  # "1.1 Build 62", from the program's own banner
    assert re.search(r"ARIMA Model:\s+" + re.escape(extra["x13_model"]), output)
    assert "AO2010.Jan" in extra["x13_outliers"].split(", ")
    assert np.allclose(out["sales_sa"], d11, rtol=1e-5)  # the program's own adjusted series, as saved
    assert out.variables["sales_seasonal"].extra == extra
    assert Variable.from_dict(out.variables["sales_sa"].to_dict()).extra == extra  # kept in Parquet files and projects
    dictionary = dict(view_sections(out))["Variables"].to_frame()
    assert dictionary.loc["sales_sa", "x13_model"] == extra["x13_model"]  # shown in the data dictionary
    static = data.seasonal_adjust("sales", X13(program=folder), derived=False)
    assert static.variables["sales_sa"].extra == extra and np.allclose(static["sales_sa"], out["sales_sa"])
    clean = out.edit({"sales": {"2010-01": float(y[60] / 1.15)}})  # recomputed: the notes follow the new run
    assert "AO2010.Jan" not in clean.variables["sales_sa"].extra.get("x13_outliers", "")
    assert set(clean.variables["sales_sa"].extra) >= {"x13_program", "x13_version", "x13_model"}
    assert not any(k.startswith("x13_") for k in clean.seasonal_adjust("sales", STL()).variables["sales_sa"].extra)


def test_x13_failures_name_the_series(monthly, tmp_path):
    if sys.platform == "win32":
        pytest.skip("the stand-in program is a shell script")
    fake = tmp_path / "x13as"  # a stand-in program that fails as x13as does: its error file, after "spc:"
    fake.write_text("#!/bin/sh\nprintf 'In pmx.spc:\\n ERROR: Cannot compute outlier t-statistic for outlier backward deletion -\\n"
                    "        the residual root mean square error is zero.\\n' > \"$1.err\"\n")
    fake.chmod(0o755)
    data, _ = monthly
    message = "X-13ARIMA-SEATS could not adjust sales: Cannot compute outlier t-statistic for outlier backward deletion - the residual"
    with pytest.raises(DataError, match=re.escape(message)):
        data.seasonal_adjust("sales", X13(program=str(tmp_path)), derived=False)
    with pytest.raises(DataError, match=r"sales_\w+ \(.*\) could not be recomputed: X-13ARIMA-SEATS could not adjust sales"):
        data.seasonal_adjust("sales", X13(program=str(tmp_path)))


def test_x13_failures_from_the_program(monthly):
    folder = _x13_folder()
    if folder is None:
        pytest.skip("x13as is not installed")
    data, _ = monthly
    t = np.arange(len(data.frame))
    exact = data.with_series(sales=pd.Series(100 * np.exp(0.002 * t + 0.05 * np.sin(2 * np.pi * t / 12)), index=data.frame.index))
    with pytest.raises(DataError, match="X-13ARIMA-SEATS could not adjust sales: .*mean square error .*is zero"):
        exact.seasonal_adjust("sales", X13(program=folder), derived=False)  # no noise at all: X-13 refuses it


def test_x13_provenance_in_a_panel_names_each_entity(monthly):
    folder = _x13_folder()
    if folder is None:
        pytest.skip("x13as is not installed")
    data, _ = monthly
    frame = data.frame.reset_index()
    spiked = frame.sales.to_numpy().copy()
    spiked[60] *= 1.15
    panel = Dataset.from_frame(pd.concat([frame.assign(region="north"), frame.assign(region="south", sales=spiked)]),
                               time="time", entity="region")
    extra = panel.seasonal_adjust("sales", X13(program=folder), derived=False).variables["sales_sa"].extra
    assert extra["x13_program"] == folder and "; " not in extra["x13_version"]  # one program: said once
    north, south = extra["x13_outliers"].split("; ")
    assert north.startswith("north: ") and south.startswith("south: ") and "AO2010.Jan" in south and "AO2010.Jan" not in north


