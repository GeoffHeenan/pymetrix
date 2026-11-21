"""The command and function reference (docs/reference, scripts/make_api_reference.py): built from the code, it lists
every dialog, API call, method and test, says which library function each wrapper calls, and which PyMetrix code
calls each library function."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    out = tmp_path_factory.mktemp("reference")
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_api_reference.py"), str(out)],
                          capture_output=True, text=True, timeout=600, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    return {name: list(csv.DictReader((out / f"{name}.csv").open(encoding="utf-8")))
            for name in ("commands", "calls", "settings", "other", "library")} | {"md": (out / "reference.md").read_text("utf-8")}


def test_everything_registered_is_listed(reference):
    from pymetrix import _api
    from pymetrix.commands import COMMANDS

    first_party = (None, "", "pymetrix", "pymetrix-finance", "pymetrix-labbook")  # not what other tests register
    calls = {r["id"] for r in reference["calls"]}
    assert {c for c, spec in _api.API_CALLS.items() if spec.plugin in first_party} <= calls
    assert {c for c, info in COMMANDS.items() if info.plugin in first_party} <= {r["id"] for r in reference["commands"]}
    for section in ("## 1. Dialog commands", "## 2. API functions", "## 3. Methods, tests", "## 5. External library"):
        assert section in reference["md"]


def test_wrappers_name_the_library_function_they_call(reference):
    settings = {r["name"]: r for r in reference["settings"]}
    assert (settings["ADF"]["implementation"], settings["ADF"]["library_function"]) == ("wrapper", "statsmodels.tsa.stattools.adfuller")
    if "X13" in settings:  # the seasonal plugin's methods, where it is installed
        assert settings["TramoSeats"]["library"] == "JDemetra+" and settings["X13"]["library"] == "x13as"
    other = {r["name"]: r for r in reference["other"]}
    assert other["johansen"]["library_function"] == "statsmodels.tsa.vector_ar.vecm.coint_johansen"  # through the adapter
    calls = {r["call"]: r for r in reference["calls"]}
    assert calls["Dataset.johansen"]["implementation"] == "wrapper"  # and one step further, through johansen()
    assert calls["Equation.fit"]["implementation"].startswith("pymetrix (dispatches")


def test_dialogs_name_their_calls_and_the_library_lists_its_callers(reference):
    commands = {r["command"]: r for r in reference["commands"]}
    assert commands["Johansen"]["api_calls"] == "dataset.johansen" and commands["Johansen"]["menu"].startswith("Test →")
    assert "fitted_var.irf" in commands["ImpulseResponse"]["api_calls"]
    library = {r["function"]: r for r in reference["library"]}
    assert "ADF (tests)" in library["statsmodels.tsa.stattools.adfuller"]["used_by"]
    assert not [f for f in library if f.startswith(("numpy.linalg", "scipy.linalg"))]  # arithmetic, left out
