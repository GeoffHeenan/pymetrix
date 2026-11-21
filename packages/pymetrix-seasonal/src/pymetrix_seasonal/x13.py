"""X-13ARIMA-SEATS (US Census Bureau): the ``x13as`` program run on a specification written here.

``X13`` runs the program (on the PATH, or in the folder named by ``X13PATH`` or ``program=``) on X-11 (its mode,
seasonal and Henderson filters, sigma limits) or SEATS, an ARIMA model given or chosen automatically, outliers (types,
critical value), trading-day and Easter effects, a forecast extension, and any other specification text. Settings left
at their defaults leave their options out, so X-13 uses its own defaults. Series it adjusts record, in their
variables' ``extra``, the program's folder and version, the ARIMA model it estimated and the outliers it found
(``x13_program``, ``x13_version``, ``x13_model``, ``x13_outliers``), with ``backend`` and ``backend_options``.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
import pandas as pd

from pymetrix.commands import Widget
from pymetrix.errors import DataError, SpecificationError

__all__ = ["X13", "run_x13", "x13_notes", "x13_program"]

_LOG = Widget("choice", help="multiplicative (logs) or additive; empty = multiplicative when every value is positive")


def _use_log(log: bool | None, y: np.ndarray, name: str) -> bool:
    """Logs (multiplicative) or levels: as given, else logs when every value is positive (as pymetrix's STL)."""
    if log is None:
        return bool(np.all(y > 0))
    if log and np.any(y <= 0):
        raise SpecificationError(f"{name} has zero or negative values: multiplicative adjustment (log=True) needs positive values")
    return bool(log)


def x13_program(path: str | None = None) -> str | None:
    """Where the X-13ARIMA-SEATS program is: ``path``, the folder in ``X13PATH``, or the PATH."""
    for folder in (path, os.environ.get("X13PATH")):
        if folder:
            for exe in ("x13as.exe", "x13as", "x13as_ascii.exe", "x13as_ascii"):
                if os.path.isfile(os.path.join(folder, exe)):
                    return folder
            if os.path.isfile(folder):
                return os.path.dirname(folder)
    found = shutil.which("x13as") or shutil.which("x13as_ascii")
    return os.path.dirname(found) if found else None


def x13_notes(stdout: bytes | str, output: str) -> dict[str, str]:
    """What an X-13 run reports about itself: the program's version, the ARIMA model it estimated, and
    the outliers it found ("none" when it looked and found none), read from its console and output file."""
    console = stdout.decode("latin-1") if isinstance(stdout, bytes) else stdout
    notes: dict[str, str] = {}
    version = re.search(r"Version Number\s+(\S+\s+Build\s+\d+)", console) or re.search(r"Release Version\s+(\S+\s+Build\s+\d+)", output)
    if version:
        notes["x13_version"] = " ".join(version.group(1).split())
    model = re.search(r"^\s*ARIMA Model:\s*(.+?)\s*$", output, re.M) or re.search(r"Final automatic model choice\s*:\s*(.+?)\s*$", output, re.M)
    if model:
        notes["x13_model"] = " ".join(model.group(1).split())
    start = output.find("Automatically Identified Outliers")
    if start >= 0:
        found = []
        for line in output[start:].splitlines()[1:]:
            if line.strip().startswith("---"):
                break
            if code := re.match(r"\s*((?:AO|LS|TC|SO)\d{4}\.\S+)", line):
                found.append(code.group(1))
        notes["x13_outliers"] = ", ".join(found) or "none"
    elif re.search(r"^\s*\d+:\s*outlier\s*\{", output, re.M):  # its echo of the spec: it looked, and found none
        notes["x13_outliers"] = "none"
    return notes


_SEASONAL_FILTERS = ("", "s3x1", "s3x3", "s3x5", "s3x9", "s3x15", "stable", "msr", "x11default")


def _pairs(text: str, what: str) -> str:
    """'2 1' or '(2 1)' or '2,1' -> '(2 1)', as X-13 writes a pair."""
    parts = re.sub(r"[(),]", " ", str(text)).split()
    if len(parts) != 2 or not all(p.replace(".", "", 1).isdigit() for p in parts):
        raise SpecificationError(f"{what} is two numbers, such as '2 1'")
    return f"({parts[0]} {parts[1]})"


@dataclass(frozen=True)
class X13:
    """X-13ARIMA-SEATS (US Census Bureau), the x13as program run on a specification written here: an ARIMA model
    (chosen automatically, or given), outliers, trading-day and Easter effects, then X-11 or SEATS adjustment.

    Each setting left at its default leaves that option out of the specification, so X-13 uses its own default; with
    every default the specification is the one statsmodels' x13_arima_analysis writes. ``spec`` adds any other X-13
    specification text (for example ``check{print=all}``); options the fields here write cannot be given twice."""

    label: ClassVar[str] = "X-13ARIMA-SEATS"
    needs_periods: ClassVar[bool] = True  # decompose() is given the series' periods (pymetrix.seasonal)
    topics: ClassVar[tuple[str, ...]] = ("seasonal-adjustment",)

    log: Annotated[bool | None, _LOG] = None
    outlier: Annotated[bool, Widget("checkbox", help="detect and adjust for outliers")] = True
    trading: Annotated[bool, Widget("checkbox", help="estimate trading-day effects (td)")] = False
    program: Annotated[str | None, Widget("text", advanced=True, help="folder of the x13as program; empty = X13PATH or the PATH")] = None
    adjustment: Annotated[Literal["x11", "seats"], Widget("choice", help="X-11 moving averages, or SEATS (signal extraction from the ARIMA model)")] = "x11"
    mode: Annotated[Literal["auto", "mult", "add", "pseudoadd", "logadd"], Widget(
        "choice", help="X-11 decomposition: multiplicative, additive, pseudo-additive or log-additive; auto = X-13's default")] = "auto"
    easter: Annotated[int, Widget("number", min=0, max=25, help="days before Easter in an Easter regressor (often 8); 0 = none")] = 0
    seasonal_filter: Annotated[Literal["", "s3x1", "s3x3", "s3x5", "s3x9", "s3x15", "stable", "msr", "x11default"], Widget(
        "choice", advanced=True, help="X-11 seasonal moving average (3x3, 3x5, 3x9, ...); empty = chosen by X-13 (msr)")] = ""
    trend_filter: Annotated[int | None, Widget("number", min=3, max=101, step=2, advanced=True,
                                               help="X-11 Henderson trend filter length (odd); empty = chosen by X-13")] = None
    sigma_limits: Annotated[str, Widget("text", advanced=True, help="X-11 extreme-value limits in standard deviations, e.g. 1.5 2.5; empty = X-13's")] = ""
    arima: Annotated[str, Widget("text", advanced=True, help="an ARIMA model such as (0 1 1)(0 1 1); empty = chosen automatically")] = ""
    max_order: Annotated[str, Widget("text", advanced=True, help="automatic model: largest regular and seasonal ARMA orders")] = "2 1"
    max_diff: Annotated[str, Widget("text", advanced=True, help="automatic model: largest regular and seasonal differences")] = "2 1"
    outlier_types: Annotated[str, Widget("text", advanced=True, help="outlier types to look for: ao ls tc so (or all); empty = X-13's (ao ls)")] = ""
    critical: Annotated[float | None, Widget("number", min=2, max=10, step=0.1, advanced=True,
                                             help="outlier critical t-value; empty = set by X-13 from the series' length")] = None
    forecast: Annotated[int | None, Widget("number", min=0, max=120, advanced=True,
                                           help="periods forecast to extend the series before filtering; empty = X-13's default")] = None
    spec: Annotated[str, Widget("text", advanced=True, help="other X-13 specification text, added as written")] = ""

    def __post_init__(self) -> None:
        if self.adjustment not in ("x11", "seats"):
            raise SpecificationError("adjustment is x11 or seats")
        if self.mode not in ("auto", "mult", "add", "pseudoadd", "logadd"):
            raise SpecificationError("mode is auto, mult, add, pseudoadd or logadd")
        if self.seasonal_filter not in _SEASONAL_FILTERS:
            raise SpecificationError(f"seasonal_filter is one of {', '.join(f for f in _SEASONAL_FILTERS if f)}, or empty")
        if self.trend_filter is not None and (self.trend_filter < 3 or self.trend_filter > 101 or self.trend_filter % 2 == 0):
            raise SpecificationError("trend_filter is an odd length from 3 to 101")
        if self.adjustment == "seats" and (self.mode != "auto" or self.seasonal_filter or self.trend_filter or self.sigma_limits):
            raise SpecificationError("mode, filters and sigma limits are X-11's: leave them empty with SEATS")
        if self.arima and not re.fullmatch(r"\s*(\(\s*\d+\s+\d+\s+\d+\s*\)\s*){1,2}\s*(\d+\s*)?", self.arima):
            raise SpecificationError("arima is a model such as (0 1 1)(0 1 1)")
        for name, text in (("max_order", self.max_order), ("max_diff", self.max_diff)):
            _pairs(text, name)
        if self.sigma_limits:
            _pairs(self.sigma_limits, "sigma_limits")
        kinds = self.outlier_types.lower().split()
        if any(k not in ("ao", "ls", "tc", "so", "all") for k in kinds):
            raise SpecificationError("outlier_types are among ao, ls, tc, so, or all")

    def specification(self, y: np.ndarray, index: pd.PeriodIndex, period: int, name: str, log: bool) -> str:
        """The X-13 specification for one series (its data included)."""
        start = index[0]
        first = start.month if period == 12 else start.quarter
        values = "\n".join(repr(float(v)) for v in y)
        title = re.sub(r'["{}]', "", name)[:70] or "series"
        out = [f'series{{\n  title="{title}"\n  name="{title}"\n  start={start.year}.{first}\n  period={period}\n'
               f"  appendbcst=no\n  appendfcst=no\n  data=(\n{values}\n  )\n}}",
               f"transform{{function={'log' if log else 'none'}}}"]
        regressors = (["td"] if self.trading else []) + ([f"easter[{int(self.easter)}]"] if self.easter else [])
        if regressors:
            out.append(f"regression{{\n  variables = ({' '.join(regressors)})\n}}")
        if self.outlier:
            options = []
            if self.outlier_types.strip():
                options.append(f"types = ({' '.join(self.outlier_types.lower().split())})")
            if self.critical is not None:
                options.append(f"critical = {float(self.critical):g}")
            out.append("outlier{" + ("\n  " + "\n  ".join(options) + "\n" if options else "") + "}")
        if self.arima.strip():
            out.append(f"arima{{model = {' '.join(self.arima.split())}}}")
        else:
            out.append(f"automdl{{\n  maxorder = {_pairs(self.max_order, 'max_order')}\n  maxdiff = {_pairs(self.max_diff, 'max_diff')}\n}}")
        if self.forecast is not None:
            out.append(f"forecast{{maxlead = {int(self.forecast)}}}")
        if self.adjustment == "seats":
            out.append("seats{save=(s10 s11)}")
        else:
            options = [f"mode = {self.mode}"] if self.mode != "auto" else []
            options += [f"seasonalma = {self.seasonal_filter}"] if self.seasonal_filter else []
            options += [f"trendma = {int(self.trend_filter)}"] if self.trend_filter else []
            options += [f"sigmalim = {_pairs(self.sigma_limits, 'sigma_limits')}"] if self.sigma_limits else []
            out.append("x11{" + "".join(f"\n  {o}" for o in options) + "\n  save=(d10 d11 d12 d13)\n  savelog=(fd8 m7 q)\n}")
        if self.spec.strip():
            out.append(self.spec.strip())
        return "\n".join(out) + "\n"

    def decompose(self, y: np.ndarray, period: int, name: str, index: pd.PeriodIndex | None = None,
                  notes: dict[str, str] | None = None) -> tuple[np.ndarray, bool]:
        """The seasonal factor (or component); ``notes``, if given, gets the program's folder, the specification it
        ran (``backend_options``, the data left out) and what the run reports (``x13_notes``)."""
        folder = x13_program(self.program)
        if folder is None:
            raise SpecificationError("X-13ARIMA-SEATS needs the x13as program from the US Census Bureau "
                             "(https://www.census.gov/data/software/x13as.html): put it on the PATH, "
                             "or set X13PATH to its folder")
        log = _use_log(self.log, y, name)
        if index is None:
            raise SpecificationError("X-13 needs the series' periods")
        spec = self.specification(y, index, period, name, log)
        stdout, output, sa, warnings_ = run_x13(folder, spec, "s11" if self.adjustment == "seats" else "d11", name)
        if len(sa) != len(y):
            raise DataError(f"X-13ARIMA-SEATS returned {len(sa)} adjusted values for the {len(y)} of {name}")
        if notes is not None:
            notes.update({"x13_program": folder, **x13_notes(stdout, output)})
            notes["backend"] = "X-13ARIMA-SEATS (x13as, run by pymetrix)"
            notes["backend_options"] = re.sub(r"data=\(\n.*?\n  \)", "data=(...)", spec, flags=re.S).replace("\n", " ")
            if warnings_:
                notes["backend_warnings"] = warnings_
        additive = (self.mode == "add") if self.adjustment == "x11" and self.mode != "auto" else not log
        return (y - sa, False) if additive else (y / sa, True)

    def describe(self) -> str:
        parts = ["SEATS" if self.adjustment == "seats" else ""] + (["trading days"] if self.trading else []) \
            + ([f"Easter[{self.easter}]"] if self.easter else []) + ([self.mode] if self.mode != "auto" else [])
        parts = [p for p in parts if p]
        return "X-13ARIMA-SEATS" + (f" ({', '.join(parts)})" if parts else "")


def run_x13(folder: str, spec: str, table: str, name: str) -> tuple[str, str, np.ndarray, str]:
    """Run x13as on ``spec`` in a scratch folder: its console, its output file, the saved ``table`` (d11, s11, ...)
    as values, and its warnings. An error from the program is a DataError naming the series."""
    import subprocess
    import tempfile

    exe = next((os.path.join(folder, n) for n in ("x13as.exe", "x13as", "x13as_ascii.exe", "x13as_ascii")
                if os.path.isfile(os.path.join(folder, n))), None)
    if exe is None:
        raise SpecificationError(f"no x13as program in {folder}")
    with tempfile.TemporaryDirectory(prefix="pymetrix-x13-") as work:
        with open(os.path.join(work, "pmx.spc"), "w", encoding="latin-1") as f:
            f.write(spec)
        run = subprocess.run([exe, "pmx"], cwd=work, capture_output=True, timeout=600)
        stdout = run.stdout.decode("latin-1")

        def read(ext: str) -> str:
            path = os.path.join(work, f"pmx.{ext}")
            return open(path, encoding="latin-1").read() if os.path.isfile(path) else ""

        errors = read("err")
        errors = errors[errors.find("spc:") + 4:].strip() if "spc:" in errors else errors.strip()
        if "ERROR" in errors:
            message = re.sub(r"^\s*ERROR:\s*", "", " ".join(errors.split()))
            raise DataError(f"X-13ARIMA-SEATS could not adjust {name}: {message}")
        saved = read(table)
        if not saved:  # some messages that halt the run do not start with ERROR: give the program's own words
            said = " ".join(errors.split()) or " ".join(stdout.split())[-300:] or "no message"
            raise DataError(f"X-13ARIMA-SEATS could not adjust {name}: {said}")
        values = np.array([float(line.split()[1]) for line in saved.splitlines()[2:] if line.strip()])
        warnings_ = " ".join(errors.split()) if "WARNING" in errors else ""
        return stdout, read("out"), values, warnings_
