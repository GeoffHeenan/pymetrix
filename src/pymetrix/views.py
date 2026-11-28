"""Views: what a result shows, as named tables and figures, independent of any user interface.

A view turns a result into an ordered list of ``(title, Table or Figure)`` sections: the tabs of its window
in the desktop app, the parts of its display in Jupyter, the blocks a report adds, and what
``pymetrix.testing.fingerprint`` compares. Register views for a new result type with ``register_view``;
plugins do so from a ``pymetrix.views`` entry point. Frontends only render sections; none defines its own.
"""

from __future__ import annotations

import re

from collections.abc import Mapping
from functools import singledispatch
from typing import Any

from ._api import api
from .errors import NotFoundError
from .data import Dataset
from .estimation import Equation, FittedEquation
from .figures import Figure
from .models import Solution
from .output import Table
from .saved import SavedResult
from .stats import TestResult

__all__ = ["Section", "figure_of", "provenance_table", "register_view", "sections", "titles", "view", "view_sections"]

# Structured views: an object as named tables, for tabbed windows with copy and export.

Section = tuple[str, Any]  # (title, Table or Figure)


def _figure(make: Any) -> list[Section]:
    """A "Graph" tab, if the object can be drawn (and matplotlib is installed)."""
    import importlib.util

    if importlib.util.find_spec("matplotlib") is None:
        return []
    try:
        return [("Graph", make())]
    except (ValueError, KeyError, TypeError):
        return []


def _frame_table(frame: Any, title: str = "", notes: Any = ()) -> Table:
    import pandas as pd

    frame = frame.copy()
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    frame.index = [str(i) if not isinstance(i, tuple) else " ".join(map(str, i)) for i in frame.index]
    return Table.from_frame(frame, title=title, notes=notes)


def _pairs(mapping: Any, title: str) -> Table:
    rows = [[str(k), v if isinstance(v, (int, float)) else str(v)] for k, v in mapping.items()]
    return Table(["", "value"], rows, title=title)


def _statistics(obj: Any) -> Table:
    """A result's statistics, with a note on the definitions that are not the usual ones (estimation.criteria)."""
    from dataclasses import replace

    from .estimation.criteria import DEFINITIONS, definitions

    table = _pairs(obj.stats, "Statistics")
    used = definitions(obj)
    shown = {"aic.total", "bic.total", "hq.total"}
    totals = [k for k, v in used.items() if v in shown]  # name them: a VAR also shows per-observation criteria
    notes = [f"{', '.join(totals)}: totals, -2 log L + 2k (AIC), -2 log L + k ln n (BIC) and -2 log L + 2k ln ln n (HQ); "
             "EViews divides them by n."] if totals else []
    special = {k: v for k, v in used.items() if v not in shown and v in DEFINITIONS and v.split(".")[0] in ("aic", "bic", "hq", "fpe")}
    notes += [f"{k}: {DEFINITIONS[v].name}, {DEFINITIONS[v].formula}" + (f" ({DEFINITIONS[v].matches})" if DEFINITIONS[v].matches else "")
              for k, v in special.items()]
    return replace(table, notes=tuple(notes)) if notes else table


@singledispatch
def sections(obj: Any) -> list[Section]:
    """Named tables describing ``obj``; the first is the main view."""
    for attr in ("table", "summary"):
        value = getattr(obj, attr, None)
        value = value() if callable(value) else value
        if isinstance(value, Table):
            return [("Results", value)]
    return [("Value", Table(["value"], [[repr(obj)]]))]


@sections.register
def _(obj: Figure) -> list[Section]:
    return [("Figure", obj)]


@sections.register
def _(obj: Table) -> list[Section]:
    return [("Table", obj)]


@sections.register
def _(obj: TestResult) -> list[Section]:
    return [("Result", obj.to_table())]


@sections.register
def _(obj: SavedResult) -> list[Section]:
    rows = [["Result", obj.kind], ["Name", obj.name], ["Saved", obj.saved.replace("T", " ")], ["PyMetrix version", obj.package_version]]
    commit = _commit_text(getattr(obj, "build", None))
    rows += [["PyMetrix commit", commit]] if commit else []
    rows += [["File", obj.source]] if obj.source else []
    note = "A record of the result as it was saved: its views can be read and copied, but it cannot be estimated again."
    return [*obj.views, ("Saved", Table(["", "value"], rows, title="Saved result", notes=(note,)))]


@sections.register
def _(obj: Dataset) -> list[Section]:
    import pandas as pd

    rows = []
    for var in obj.variables.values():
        row = {k: v for k, v in var.to_dict().items() if k not in ("extra", "recipe")}
        row.update(_described(row))
        row.update({k: v for k, v in var.extra.items() if isinstance(v, (str, int, float)) and k not in row})  # e.g. X-13's model
        row["kept up to date"] = "yes" if var.recipe is not None else None
        if var.name in obj.frame.columns:
            values = obj.frame[var.name]
            valid = values.notna().to_numpy()
            row["observations"] = int(valid.sum())
            if valid.any():
                first, last = valid.argmax(), len(valid) - 1 - valid[::-1].argmax()
                row["first"], row["last"] = str(values.index[first]), str(values.index[last])
                row["gaps"] = int((~valid[first:last + 1]).sum())  # missing between the first and last observation
        rows.append(row)
    meta = pd.DataFrame(rows)
    meta = meta.set_index("name").dropna(axis=1, how="all")
    return [("Data", _frame_table(obj.in_sample(), dataset_status(obj))), *_figure(lambda: obj.plot(layout="each") if not _forecast(obj) else obj.plot()),
            ("Variables", _frame_table(meta, "Data dictionary: metadata, sample and gaps of each series"))]


def _described(row: dict[str, Any]) -> dict[str, Any]:
    """A series' coded description in words for the data dictionary: adjustment "nsa" -> "not seasonally adjusted"."""
    from .wording import choice_label

    return {k: choice_label("Variable", k, row[k]) or row[k] for k in ("adjustment", "prices") if row.get(k)}


def dataset_overview(data: Any, name: str = "") -> Table:
    """A dataset at a glance: name, dimensions, series and observations (original and derived), the active sample."""
    import pandas as pd

    frame = data.frame
    derived = [n for n in data.names if (v := data.variables.get(n)) is not None and (v.recipe is not None or v.derived_from)]
    original = [n for n in data.names if n not in derived]
    kept = [n for n in derived if data.variables[n].recipe is not None]

    def observations(names: list[str]) -> int:
        return int(frame[names].notna().to_numpy().sum()) if names else 0

    dims = []
    t = data.time_axis
    if t is not None:
        periods = pd.PeriodIndex(frame.index.get_level_values(t.name)).unique().sort_values()
        names = {"Q": "quarterly", "M": "monthly", "Y": "annual", "A": "annual", "W": "weekly", "D": "daily", "B": "business daily"}
        freq = names.get(periods.freqstr.split("-")[0].rstrip("ES"), periods.freqstr) if len(periods) else ""
        dims.append(f"{t.name}: {len(periods)} {freq} periods, {periods[0]} to {periods[-1]}" if len(periods) else f"{t.name}: none")
    for axis in data.entity_axes:
        dims.append(f"{axis.name}: {frame.index.get_level_values(axis.name).nunique()}")
    rows = [["name", name or "(unnamed)"],
            ["kind", data.kind.replace("_", " ")],
            ["dimensions", "; ".join(dims) or "rows only"],
            ["rows", f"{len(frame)}"],
            ["series", f"{len(data.names)} ({len(original)} original, {len(derived)} derived"
                       + (f", {len(kept)} kept up to date" if kept else "") + ")"],
            ["observations", f"{observations(data.names)} ({observations(original)} in original series, "
                             f"{observations(derived)} in derived)"],
            ["missing values", f"{int(frame.isna().to_numpy().sum()) if len(frame.columns) else 0}"]]
    if data.meta.sample is not None:
        rows.append(["active sample", f"{data.sample} ({data.sample.describe(data)})"])
    else:  # a time series' rows are its periods
        rows.append(["active sample", "all periods" if t is not None and not data.entity_axes else "all rows"])
    title = data.meta.title or name
    kind = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", type(data).__name__).capitalize()  # a Solution's overview: "Solution g_solution"
    return Table(["value"], [[v] for _, v in rows], index=[k for k, _ in rows], title=f"{kind} {title}".strip(),
                 notes=[n for n in [data.meta.description] if n])


def dataset_status(data: Any) -> str:
    """One line about a dataset: series, frequency, sample, rows, series kept up to date, missing values."""
    import pandas as pd

    frame = data.frame
    parts = [f"{len(data.names)} series"]
    index = frame.index
    if isinstance(index, pd.PeriodIndex) and len(index):
        names = {"Q": "quarterly", "M": "monthly", "Y": "annual", "A": "annual", "W": "weekly", "D": "daily", "B": "business daily"}
        code = index.freqstr.split("-")[0].rstrip("ES")
        parts.append(names.get(code, index.freqstr))
        parts.append(f"{index[0]} to {index[-1]}")
    unit = "periods" if isinstance(index, pd.PeriodIndex) else "rows"  # a time series' rows are its periods
    parts.append(f"{len(frame)} {unit}")
    if getattr(data, "meta", None) is not None and data.meta.sample is not None:
        parts.append(f"sample {data.sample}: {int(data.sample_mask().sum())} {unit} shown")
    kept = sum(v.recipe is not None for v in data.variables.values())
    if kept:
        parts.append(f"{kept} kept up to date")
    sourced = sum(bool(v.source or v.external_ref) for v in data.variables.values())
    if sourced:
        parts.append(f"{sourced} with a source")
    missing = int(frame.isna().to_numpy().sum()) if len(frame.columns) else 0
    parts.append(f"{missing} missing values" if missing else "no missing values")
    return " · ".join(parts)


def _forecast(obj: Dataset) -> bool:
    return "forecast" in obj.frame.columns or any(f"{c}_lower" in obj.frame.columns for c in obj.names)


@sections.register
def _(obj: Solution) -> list[Section]:
    solved = obj.solved()
    cols = list(obj.model.endogenous) if obj.model is not None else solved.names
    out: list[Section] = [("Solution", _frame_table(solved.frame[cols], repr(obj)))]
    sims = getattr(obj, "simulation", None)
    if sims is not None:  # a stochastic simulation: its fan and quantiles
        info = sims.info
        notes = [f"{info.get('draws')} draws, seed {info.get('seed')}; {info.get('failed', 0)} failed",
                 str(info.get("method", "")),
                 "standard deviations: " + ", ".join(f"{k} {v:.4g}" for k, v in dict(info.get("sd") or info.get("shock_sd") or {}).items())]
        out += [("Fan chart", figure) for _, figure in _figure(obj.fan)]
        out.append(("Quantiles", _frame_table(obj.quantiles((0.05, 0.25, 0.5, 0.75, 0.95)).frame, "Stochastic simulation: quantiles", notes)))
    convergence = getattr(obj, "convergence", None)  # a solution saved before it was recorded has none
    if convergence is not None and len(convergence):
        out.append(("Convergence", _frame_table(convergence, "Convergence by period",
                                                ["iterations; the largest absolute residual at the end and its equation"])))
    return [*out, *sections.dispatch(Dataset)(obj)]


@sections.register
def _(obj: FittedEquation) -> list[Section]:
    import pandas as pd

    out: list[Section] = [("Summary", obj.summary()), *_figure(obj.plot)]
    equation = _equation_table(obj)
    if equation is not None:
        out.append(("Equation", equation))
    p = obj.params
    coefs = pd.DataFrame({"coef": p.values, "std err": p.se}, index=list(p.names))
    out.append(("Coefficients", _frame_table(coefs, "Coefficients")))
    if p.cov is not None:
        out.append(("Covariance", _frame_table(pd.DataFrame(p.cov, index=list(p.names), columns=list(p.names)), "Coefficient covariance")))
    out.append(("Statistics", _statistics(obj)))
    from .estimation.families import Gaussian

    if not isinstance(obj.equation.family, Gaussian) and not obj.equation.has_error_model:  # logit, counts, ordered, Tobit...
        try:
            out.append(("Marginal effects", obj.marginal_effects()))
        except Exception as exc:  # noqa: BLE001 - a convenience: say why there are none
            out.append(("Marginal effects", Table(["note"], [[str(exc)]])))
    if obj.resid is not None:
        cols = {"fitted": obj.fitted, "resid": obj.resid}
        cols.update({k: obj.extra[k] for k in ("std_resid", "volatility") if k in obj.extra})
        out.append(("Residuals", _frame_table(pd.DataFrame(cols), "Fitted values and residuals")))
        try:
            out.append(("Diagnostics", obj.diagnostics()))
        except Exception as exc:  # diagnostics are a convenience; some fits have none
            out.append(("Diagnostics", Table(["note"], [[str(exc)]])))
        out += [("Q-Q plot", figure) for _, figure in _figure(obj.qq_plot)]
        out += _correlograms(obj)
    if obj.provenance is not None:
        prov = obj.provenance
        out.append(("Provenance", provenance_table(prov)))
    return out


def _correlograms(obj: FittedEquation) -> list[Section]:
    """For a fit to a time series: the correlogram of its residuals (standardised after ARMA errors or GARCH), as a
    table and a chart, up to 20 lags (a quarter of the observations if fewer); with a variance model, of the squared
    residuals too."""
    import pandas as pd

    from .errors import SpecificationError

    if not isinstance(obj.resid.index, pd.PeriodIndex):
        return []
    lags = min(20, int(obj.resid.notna().sum()) // 4)
    if lags < 1:
        return []
    out: list[Section] = []
    for squared, name in ((False, "Correlogram"), (True, "Correlogram of squares")):
        if squared and obj.equation.variance is None:
            continue
        try:
            table = obj.correlogram(lags, squared=squared)
        except (SpecificationError, ValueError):
            continue
        from .stats.timeseries import correlogram_figure

        out.append((name, table))
        out += [(f"{name} graph", figure) for _, figure in _figure(lambda: correlogram_figure(table))]
    return out


def _residual_correlogram(obj: Any) -> list[Section]:
    """A VAR's or VECM's residual correlogram (12 lags, fewer for short samples): the table, then the panels."""
    lags = min(12, len(obj.resid.frame) // 4)
    if lags < 1:
        return []
    rc = obj.residual_correlogram(lags)
    return [("Residual correlogram", rc.table()), *[("Residual correlogram graph", f) for _, f in _figure(rc.plot)]]


def _roots(obj: Any) -> list[Section]:
    """A VAR's or VECM's companion-matrix roots: the table (real and imaginary parts, modulus), then the unit-circle
    graph."""
    import pandas as pd

    z = obj.eigenvalues
    table = _frame_table(pd.DataFrame({"real": z.real, "imaginary": z.imag, "modulus": abs(z)}), "Companion-matrix roots")
    return [("Roots", table), *[("Roots graph", f) for _, f in _figure(obj.roots_plot)]]


def _equation_table(obj: Any) -> Table | None:
    """An equation in code, in algebraic notation and in LaTeX, and for a fit, with its estimates."""
    eq = obj.equation if isinstance(obj, FittedEquation) else obj
    if eq.fixed_effects or eq.categoricals:
        return None
    try:
        rows = [["code", eq.text], ["algebraic", eq.algebraic()]]
        if isinstance(obj, FittedEquation) and not eq.is_identity:
            rows.append(["estimated", obj.algebraic(estimates=True)])
        rows.append(["LaTeX", eq.algebraic("latex")])
    except Exception:  # an expression the notation cannot write is shown in code only
        return None
    return Table(["", "equation"], rows, title=eq.label if isinstance(eq, Equation) else "Equation")


@sections.register
def _(obj: Equation) -> list[Section]:
    equation = _equation_table(obj)
    return [("Equation", equation if equation is not None else Table(["equation"], [[obj.text]]))]


def provenance_table(prov: Any) -> Table:
    """Where a result came from, for any result with ``Provenance``."""
    backend = getattr(prov, "backend", None)
    dropped = getattr(prov, "dropped_rows", None)
    runs = getattr(prov, "rows", None)
    rows = {"sample": sample_text(prov) or None, "observations": prov.nobs, "dropped": prov.dropped,
            "dropped by reason": ", ".join(f"{k} {v}" for k, v in dropped["by_reason"].items()) if dropped else None,
            "gaps": f"{len(runs)} runs of periods: " + ", ".join(f"{a} to {b}" for a, b in runs[:6]) + (" ..." if len(runs) > 6 else "")
            if isinstance(runs, list) and len(runs) > 1 else None,
            "presample": ", ".join(getattr(prov, "presample", ()) or ()) or None, "method": prov.method,
            "backend": f"{backend['name']} {backend.get('version', '')} ({backend.get('wraps', '')})".strip() if backend else None,
            "backend options": dict((getattr(prov, "options", None) or {}).get("backend_options") or {}) or None,
            "data asked for": _vintage_text(getattr(prov, "vintage_request", None)),
            "data fingerprint": prov.data_fingerprint or None, "seed": prov.seed,
            "random generator": getattr(prov, "random_generator", None) if prov.seed is not None else None, "converged": prov.converged,
            "warnings": "; ".join(prov.warnings) or "none", "messages": "; ".join(getattr(prov, "messages", ()) or ()) or None,
            "package": prov.package_version, "commit": _commit_text(getattr(prov, "build", None)), "created": prov.created,
            **{f"vintage {k}": v for k, v in prov.data_vintages.items()}}
    blank = [k for k, v in rows.items() if v is None or (isinstance(v, float) and v != v)]  # unknown: left out
    return _pairs({k: _readable(k, v) for k, v in rows.items() if k not in blank}, "Provenance")


def _commit_text(build: Any) -> str | None:
    """The pymetrix commit a result was made by, in words (None: not recorded)."""
    if not isinstance(build, Mapping):
        return None
    commit = build.get("commit")
    if not commit:
        return build.get("note") or None
    dirty = build.get("dirty")
    return f"{str(commit)[:12]}" + (", with uncommitted changes" if dirty else "" if dirty is False else ", changes not known")


def sample_text(prov: Any) -> str:
    """A result's sample in words, for its Provenance tab and a report's appendix: "2000Q3 to 2024Q4"; for a result
    made of re-estimates (backtests, combinations, rolling and recursive fits), what they ran over and the samples
    they were estimated on: "origins 2015Q1 to 2022Q4 (32 re-estimates), estimated on 2000Q3 to 2015Q1 at the first
    and 2000Q3 to 2022Q4 at the last"."""
    sample = getattr(prov, "sample", None)
    spec = prov.spec if isinstance(getattr(prov, "spec", None), Mapping) else {}
    record = spec.get("re-estimates")
    origins = spec.get("origins")
    if not isinstance(record, Mapping) and isinstance(origins, (list, tuple)) and origins:  # a backtest saved before
        record, sample = {"over": "origins", "count": len(origins)}, (origins[0], origins[-1])  # the record (its sample was wrong)
    span = f"{sample[0]} to {sample[1]}" if sample else ""
    if not isinstance(record, Mapping):
        return span
    text = f"{record.get('over', '')} {span}".strip()
    count, failed = record.get("count"), record.get("failed")
    if count is not None:
        text += f" ({count} re-estimate{'' if count == 1 else 's'}" + (f", {failed} failed" if failed else "") + ")"
    elif record.get("note"):
        text += f" ({record['note']})"
    first, last = record.get("first"), record.get("last")
    if first and last:
        text += f", estimated on {first[0]} to {first[1]}"
        if list(first) != list(last):
            text += f" at the first and {last[0]} to {last[1]} at the last"
    return text


def _vintage_text(request: Any) -> str | None:
    """The vintage request in words: "latest, retrieved 2026-10-24T09:12", "as of 2020-01-30", "file data.csv"."""
    if not request:
        return None
    policy = request.get("policy", "unknown")
    if policy == "latest":
        return f"latest release, retrieved {request.get('retrieved', '?')}"
    if policy == "as_of":
        return f"as published on {request.get('date', '?')}"
    if policy == "file":
        return f"file {request.get('source', '')}" + (f", modified {request['modified']}" if request.get("modified") else "")
    if policy == "example":
        return f"example data {request.get('name', '')} (pymetrix {request.get('package_version', '')})"
    return {"user": "data given in code (pandas)", "direct": "arrays passed directly to the call (pymetrix.ols)",
            "unknown": "not recorded"}.get(policy, policy)


def _readable(key: str, value: Any) -> Any:
    """A provenance value as read: 2000Q2 to 2024Q4, "OLS; cov: classic", a time to the minute."""
    from datetime import datetime

    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, bool):
        return "yes" if value else "no"
    if key == "sample" and isinstance(value, (tuple, list)) and len(value) == 2:
        return f"{value[0]} to {value[1]}"
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            if v is None or v == "" or v == [] or v == {}:
                continue
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], dict):
                v = f"{len(v)} (see Fields)"  # e.g. the starts of an estimation: one line each would swamp the table
            elif isinstance(v, dict):
                v = f"{len(v)} entries (see Fields)"
            elif isinstance(v, (list, tuple)):
                v = ", ".join(str(x) for x in v)
            parts.append(str(v) if k == "type" else f"{str(k).replace('_', ' ')}: {v}")
        return "; ".join(parts)
    if isinstance(value, (tuple, list)):
        return ", ".join(str(x) for x in value)
    return value


def _provenance(obj: Any) -> list[Section]:
    prov = getattr(obj, "provenance", None)
    return [("Provenance", provenance_table(prov))] if prov is not None else []


def _describe(value: Any) -> tuple[str, str]:
    """(size, value) for the Fields table: a short value for scalars, a shape for arrays, tables and data."""
    import numpy as np
    import pandas as pd

    if value is None or isinstance(value, (bool, int, float, str, np.number)):
        text = repr(value) if not isinstance(value, float) else f"{value:.6g}"
        return "", text if len(text) <= 60 else text[:57] + "..."
    if isinstance(value, (pd.DataFrame, np.ndarray)):
        return " × ".join(map(str, value.shape)), ""
    if isinstance(value, pd.Series):
        return f"{len(value)}", ""
    if isinstance(value, pd.Index):  # an index's .names are its level names (often None): show its span instead
        return f"{len(value)}", f"{value[0]} to {value[-1]}" if len(value) else ""
    if hasattr(value, "frame") and hasattr(value, "names"):  # a Dataset
        return f"{len(value.frame)} × {len(value.names)}", ", ".join(list(value.names)[:6]) + (", ..." if len(value.names) > 6 else "")
    if hasattr(value, "names") and hasattr(value, "values") and not callable(value.values):  # a PointEstimate
        return f"{len(value.names)}", ", ".join(list(value.names)[:6]) + (", ..." if len(value.names) > 6 else "")
    if isinstance(value, dict) or hasattr(value, "keys") and callable(value.keys):
        keys = list(value.keys())
        return f"{len(keys)}", ", ".join(map(str, keys[:6])) + (", ..." if len(keys) > 6 else "")
    if isinstance(value, (list, tuple)):
        return f"{len(value)}", ", ".join(map(str, value[:6])) + (", ..." if len(value) > 6 else "")
    return "", ""


def fields(obj: Any, name: str = "result", code: str = "") -> Table:
    """The named fields of a result, with the Python that reaches each one: everything shown in its views comes from
    these. ``code`` (the command that made it) is added as a note."""
    import dataclasses

    names: list[str] = []
    if dataclasses.is_dataclass(obj):
        names += [f.name for f in dataclasses.fields(obj) if not f.name.startswith("_")]
    else:
        names += [k for k in getattr(obj, "__dict__", {}) if not k.startswith("_")]
    for klass in type(obj).__mro__:
        names += [k for k, v in vars(klass).items() if isinstance(v, property) and not k.startswith("_") and k not in names]
    rows, index = [], []
    for field in names:
        try:
            value = getattr(obj, field)
        except Exception as exc:  # a property that cannot be computed for this result
            rows.append(["", "", f"not available: {exc}"])
            index.append(f"{name}.{field}")
            continue
        if callable(value) and not hasattr(value, "frame") or type(value).__name__.startswith("_"):
            continue  # methods, and internal working objects
        rows.append([type(value).__name__, *_describe(value)])
        index.append(f"{name}.{field}")
        if isinstance(value, dict) and 0 < len(value) <= 40 and all(isinstance(k, str) for k in value):
            for key, item in value.items():  # e.g. fit.stats["aic"]: the Python for each entry
                if not callable(item):
                    rows.append([type(item).__name__, *_describe(item)])
                    index.append(f"{name}.{field}[{key!r}]".replace("'", '"'))
        if field == "provenance" and dataclasses.is_dataclass(value):
            for sub in dataclasses.fields(value):
                rows.append([type(getattr(value, sub.name)).__name__, *_describe(getattr(value, sub.name))])
                index.append(f"{name}.provenance.{sub.name}")
    notes = [f"Made by: {code}"] if code else []
    return Table(["type", "size", "value"], rows, index=index, index_name="Python", title=f"Fields of {name}", notes=notes)


def titles(obj: Any) -> list[str]:
    """The names of an object's views (its window's tabs): ``["Summary", "Graph", "Coefficients", ...]``."""
    return [t for t, _ in view_sections(obj)]


def view(obj: Any, title: str) -> Any:
    """One of an object's views by name: a ``Table`` or ``Figure``."""
    sections = dict(view_sections(obj))
    if title not in sections:
        raise NotFoundError(f"no view {title!r}; views: {list(sections)}")
    return sections[title]


def summary_text(obj: Any) -> str:
    """A result as plain text: its summary() as printed, or else its first table view."""
    from .output import Table

    summary = getattr(obj, "summary", None)
    if callable(summary):
        try:
            return str(summary())
        except TypeError:  # a summary that needs arguments
            pass
    first = next((v for _, v in view_sections(obj) if isinstance(v, Table)), None)
    return str(first) if first is not None else repr(obj)


def _views_to_frame(self: Any, view: str | None = None) -> Any:
    """One of the object's views as a DataFrame: the first table by default, or a view by name (the tabs of its
    window; case does not matter). ``views.titles(obj)`` lists them."""
    from .output import Table

    parts = view_sections(self)
    tables = [(t, v) for t, v in parts if isinstance(v, Table)]
    if view is None:
        if not tables:
            raise NotFoundError(f"{type(self).__name__} has no table views; views: {[t for t, _ in parts]}")
        return tables[0][1].to_frame()
    titles = {t.lower(): t for t, _ in parts}
    found = dict(parts).get(titles.get(view.lower(), view))
    if found is None:
        raise NotFoundError(f"no view {view!r}; views: {[t for t, _ in parts]}")
    if not isinstance(found, Table):
        raise NotFoundError(f"{view!r} is a figure: use .plot(), or a table view: {[t for t, _ in tables]}")
    return found.to_frame()


def _views_html(self: Any) -> str:
    """In a notebook: the object's first table view (its window's first table tab)."""
    from .output import Table

    first = next((v for _, v in view_sections(self) if isinstance(v, Table)), None)
    return first._repr_html_() if first is not None else f"<pre>{repr(self)}</pre>"


def _dataset_html(self: Any) -> str:
    """In a notebook: a dataset's one-line description, then its data (pandas' own display, rows elided)."""
    import html

    return f"<p><b>{html.escape(dataset_status(self))}</b></p>" + self.frame._repr_html_()


def attach_view_methods(cls: type) -> None:
    """Give a pymetrix type with views ``to_frame(view)`` and a notebook display, unless it has its own. Every
    result then reaches any of its views from Python as the app shows them (``fit.to_frame("Statistics")``)."""
    if not cls.__module__.startswith("pymetrix") or cls is object or cls.__name__ in ("Figure", "Table"):
        return  # a figure's numbers are .data(); a table is its own view
    dataset = any(b.__name__ == "Dataset" for b in cls.__mro__)
    if not hasattr(cls, "to_frame"):
        cls.to_frame = _views_to_frame
    if not hasattr(cls, "_repr_html_"):
        cls._repr_html_ = _dataset_html if dataset else _views_html


def _attach_all() -> None:
    from .results import attach_verbs

    for cls in list(sections.registry):
        attach_view_methods(cls)
    attach_verbs(list(sections.registry))  # the generic verbs and tidy tables on every estimate (pymetrix.results)


def register_view(cls: type, fn: Any = None, *, plugin: Any = None) -> Any:
    """Register ``fn(obj) -> [(title, Table), ...]``: the tabs of an object window for result type ``cls``.

    Usable as a decorator: ``@register_view(MyResult)``. Plugins with new result
    types call this from a function named by a ``pymetrix.views`` entry point.
    """
    from .plugins import register

    if fn is None:
        return lambda f: register_view(cls, f, plugin=plugin)
    sections.register(cls)(fn)
    attach_view_methods(cls)  # a plugin's result type reaches its views from Python too
    from .results import attach_verbs

    attach_verbs([cls])
    register("views", cls, label=cls.__name__, plugin=plugin)
    return fn


@api("pymetrix:figure_of")
def figure_of(obj: Any, tab: str = "Graph") -> Any:
    """The figure on a result's ``tab`` (its Graph tab by default), to change with ``with_options``, e.g.
    ``figure_of(fit).with_options(title="Consumption", shade="2008Q3:2009Q2")``. A figure is returned as it is."""
    from .errors import NotFoundError
    from .figures import Figure

    if isinstance(obj, Figure):
        from dataclasses import replace

        return replace(obj)  # a copy, not the named figure itself, so figure_of(chart).line(...) records as one step
    found = [(t, v) for t, v in view_sections(obj) if isinstance(v, Figure)]
    for title, value in found:
        if title.lower() == tab.lower():
            return value
    raise NotFoundError(f"no figure on a tab {tab!r}; " + (f"the figures are on {', '.join(t for t, _ in found)}" if found else "this result has no figures"))


def reportable(obj: Any) -> bool:
    """Whether ``obj`` has views of its own (a result, a table, a figure, a dataset), not just its repr."""
    from .plugins import load

    load("views")
    return sections.dispatch(type(obj)) is not sections.dispatch(object) or any(
        isinstance(getattr(obj, a, None), Table) or callable(getattr(obj, a, None)) for a in ("table", "summary"))


def view_sections(obj: Any) -> list[Section]:
    """``sections`` after loading plugins' views."""
    from .plugins import load

    load("views")
    out = []
    for title, value in sections(obj):  # every section a Table or a Figure, whatever a view (or a plugin's) returns
        if type(value).__name__ in ("DataFrame", "Series"):
            value = _frame_table(value, title)
        out.append((title, value))
    return out


def _matrix_sections(obj: Any) -> list[Section]:
    """NumPy arrays and pandas objects made in code, as tables (with their shape)."""
    import numpy as np
    import pandas as pd

    if isinstance(obj, pd.Series):
        return [("Values", _frame_table(obj.to_frame(obj.name or "value"), f"Series: {len(obj)} values"))]
    if isinstance(obj, pd.DataFrame):
        return [("Values", _frame_table(obj, f"Table: {obj.shape[0]} × {obj.shape[1]}"))]
    arr = np.asarray(obj)
    if arr.ndim == 0:
        return [("Value", Table(["value"], [[arr.item()]]))]
    if arr.ndim == 1:
        return [("Values", Table(["", "value"], [[i, v] for i, v in enumerate(arr.tolist())], title=f"Vector: {arr.shape[0]}"))]
    if arr.ndim == 2:
        frame = pd.DataFrame(arr)
        return [("Values", _frame_table(frame, f"Matrix: {arr.shape[0]} × {arr.shape[1]}"))]
    return [("Value", Table(["value"], [[repr(arr)]], title=f"Array: {' × '.join(map(str, arr.shape))}"))]


def _register_late() -> None:
    """Sections for result types defined outside the core imports above."""
    import numpy as np
    import pandas as pd

    from .estimation import RollingFit
    from .var import FEVD, IRF, FittedVAR
    from .selection import Results, Selection

    for cls in (np.ndarray, pd.Series, pd.DataFrame):
        sections.register(cls)(_matrix_sections)
    from .statespace import FittedStateSpace
    from .projections import FittedLocalProjection

    from .growth_at_risk import GrowthAtRisk

    @register_view(GrowthAtRisk)
    def _(obj: GrowthAtRisk) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot),
                *[(f"At risk, h={h}", _frame_table(obj.at_risk(h), f"Growth at risk and expected shortfall, {h} periods ahead")) for h in obj.horizons],
                *[(f"Quantiles, h={h}", _frame_table(obj.quantiles(h), f"Predicted quantiles, {h} periods ahead")) for h in obj.horizons]]

    from .smooth_lp import FittedSmoothLocalProjection

    @register_view(FittedSmoothLocalProjection)
    def _(obj: FittedSmoothLocalProjection) -> list[Section]:
        out = [("Responses", obj.summary()), *_figure(obj.plot)]
        if obj.cv is not None:
            out.append(("Cross-validation", _frame_table(obj.cv, "Cross-validation error by penalty")))
        return out

    @register_view(FittedLocalProjection)
    def _(obj: FittedLocalProjection) -> list[Section]:
        return [("Responses", obj.table()), *_figure(obj.plot), ("Regressions", obj.fits.coefficients()), ("Statistics", obj.fits.stats())]

    @register_view(FittedStateSpace)
    def _(obj: FittedStateSpace) -> list[Section]:
        states = list(obj.model.states)
        smoothed = obj.smoothed.frame[[c for s in states for c in (s, f"{s}_se")]]
        filtered = obj.filtered.frame[[c for s in states for c in (s, f"{s}_se")]]
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Smoothed states", _frame_table(smoothed, "Smoothed (two-sided) states and standard errors")),
                ("Filtered states", _frame_table(filtered, "Filtered (real-time) states and standard errors")),
                ("Innovations", _frame_table(obj.innovations(), "Standardised one-step-ahead prediction errors")),
                ("Responses", _frame_table(obj.irf(20), "Responses to one-standard-deviation shocks"))]
    from .systems import FittedSystem

    from .forecasting import Backtest, ForecastRound
    from .disaggregation import FittedDisaggregation
    from .estimation.quantile import QuantileProcess
    from .models.dynare import DynareModel
    from .midas import FittedMIDAS
    from .ardl import FittedARDL
    from .markov import FittedMarkovSwitching

    @register_view(FittedMarkovSwitching)
    def _(obj: FittedMarkovSwitching) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot),
                ("Transition probabilities", _frame_table(obj.transition, "Transition probabilities (from row to column)")),
                ("Expected durations", _frame_table(obj.durations, "Expected duration of each regime (periods)")),
                ("Regime probabilities", _frame_table(obj.probabilities(), "Smoothed regime probabilities")), *_provenance(obj)]

    @register_view(FittedARDL)
    def _(obj: FittedARDL) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Long run", obj.long_run()), ("Bounds test", obj.bounds_test().to_table()),
                ("Error correction", obj.ecm()), *_provenance(obj)]

    @register_view(FittedMIDAS)
    def _(obj: FittedMIDAS) -> list[Section]:
        weights = pd.DataFrame(obj.weights)
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Weights", _frame_table(weights, "Lag weights")), *_provenance(obj)]

    @register_view(DynareModel)
    def _(obj: DynareModel) -> list[Section]:
        equations = Table(["equation"], [[e] for e in obj.equations], title=f"Model: {len(obj.endogenous)} equations")
        params = Table(["value"], [[v] for v in obj.params.values()], index=list(obj.params), title="Parameters")
        out: list[Section] = [("Equations", equations), ("Parameters", params)]
        if obj.shock_sd:
            out.append(("Shocks", Table(["standard deviation"], [[v] for v in obj.shock_sd.values()], index=list(obj.shock_sd), title="Shocks")))
        if obj.commands:
            out.append(("Commands", Table(["not run"], [[c] for c in obj.commands], title="Commands in the file")))
        return out

    @register_view(QuantileProcess)
    def _(obj: QuantileProcess) -> list[Section]:
        return [("Coefficients", obj.table()), *_figure(obj.plot), *_provenance(obj)]

    @register_view(FittedDisaggregation)
    def _(obj: FittedDisaggregation) -> list[Section]:
        frame = obj.series.to_frame()
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Series", _frame_table(frame, f"{obj.series.name}, disaggregated"))]

    from .reconciliation import Reconciliation

    @register_view(Reconciliation)
    def _(obj: Reconciliation) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Series", _frame_table(obj.frame, "Reconciled components and the total")),
                ("Adjustments", _frame_table(obj.adjustments, "Reconciled minus preliminary")), *_provenance(obj)]

    @register_view(ForecastRound)
    def _(obj: ForecastRound) -> list[Section]:
        out: list[Section] = [("Paths", obj.table()), *_figure(obj.plot)]
        if obj.adjustments:
            out.append(("Judgement", obj.judgement()))
        if obj.conditions:
            out.append(("Conditions", obj.conditions_table()))
        return out

    from .ets import FittedExponentialSmoothing

    @register_view(FittedExponentialSmoothing)
    def _(obj: FittedExponentialSmoothing) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("States", _frame_table(obj.states, "Level, trend and seasonal states")),
                ("Residuals", _frame_table(pd.concat([obj.data.rename(obj.dependent), obj.fitted, obj.resid], axis=1), "One-step-ahead fit"))]

    from .arima import FittedSARIMA, FittedSTLForecast, FittedTheta
    from .stats.breaks import StructuralBreaks
    from .treatment import DiD, RDD, EventStudy, SyntheticControl, TreatmentEffect

    @register_view(DiD)
    def _(obj: DiD) -> list[Section]:
        out: list[Section] = [("Summary", obj.summary()), *_figure(obj.plot)]
        if obj.means is not None:
            out.append(("Means", _frame_table(obj.means, "Mean outcome before and after")))
        return out + [("Regression", obj.fit.summary())]

    @register_view(EventStudy)
    def _(obj: EventStudy) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot)]

    @register_view(RDD)
    def _(obj: RDD) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Bandwidths", obj.sensitivity_table())]

    @register_view(TreatmentEffect)
    def _(obj: TreatmentEffect) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot)]

    @register_view(SyntheticControl)
    def _(obj: SyntheticControl) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Weights", obj.weights_table()),
                ("Paths", _frame_table(obj.to_frame(), "Actual, synthetic and the gap"))]

    @register_view(StructuralBreaks)
    def _(obj: StructuralBreaks) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Regimes", obj.segments())]
    from .matrix import Matrix

    @register_view(Matrix)
    def _(obj: Matrix) -> list[Section]:
        out: list[Section] = [("Matrix", obj.table())]
        if 1 < obj.shape[0] <= 60 and 1 < obj.shape[1] <= 60 and not obj.is_complex:  # complex numbers have no colour
            from .figures import heatmap_figure

            out.append(("Heatmap", heatmap_figure(obj.to_frame(), title=obj.title or "Matrix")))
        return out

    from .matrix import LU, QR, QZ, SVD, Eig, Schur

    def _factors(obj: Any) -> list[Section]:  # a decomposition: a table for each factor, by its name
        return [(name, getattr(obj, name).table()) for name in obj._fields]

    for cls in (SVD, QR, LU, Eig, Schur, QZ):
        register_view(cls, _factors)

    @register_view(FittedSARIMA)
    def _(obj: FittedSARIMA) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot),
                ("Residuals", _frame_table(pd.concat([obj.data.rename(obj.dependent), obj.fitted, obj.resid], axis=1), "One-step-ahead fit"))]

    @register_view(FittedTheta)
    def _(obj: FittedTheta) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot)]

    @register_view(FittedSTLForecast)
    def _(obj: FittedSTLForecast) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Components", _frame_table(obj.components, "STL decomposition"))]

    from .bma import FittedBMA
    from .nonlinear import FittedThresholdAR, terasvirta
    from .ext.sklearn import FittedLearner

    @register_view(FittedLearner)
    def _(obj: FittedLearner) -> list[Section]:
        frame = pd.concat([obj.y.rename(str(obj.equation.lhs)), obj.fitted, obj.resid], axis=1)
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Importance", obj.importance()),
                ("Cross-validation", _frame_table(obj.cv_resid.to_frame(), "Out-of-sample errors of the validation windows")),
                ("Residuals", _frame_table(frame, "In-sample fit"))]

    from .ml_conformal import ConformalIntervals
    from .ml_explain import ShapValues

    @register_view(ShapValues)
    def _(obj: ShapValues) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot),
                ("Values", _frame_table(obj.to_frame(), "Each regressor's contribution, the base value and the prediction"))]

    @register_view(ConformalIntervals)
    def _(obj: ConformalIntervals) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Intervals", _frame_table(obj.frame, "Conformal prediction intervals")),
                ("Scores", _frame_table(obj.scores, "Conformity scores of the calibration periods"))]

    @register_view(FittedThresholdAR)
    def _(obj: FittedThresholdAR) -> list[Section]:
        from .nonlinear import _series_data

        tests = [obj.linearity_test(reps=199), terasvirta(_series_data(obj.data), obj.dependent, obj.spec.lags, obj.delay, span=obj.span)]
        rows = [[t.name, t.statistic, t.p_value] for t in tests]
        table = Table(["test", "statistic", "p-value"], rows, title="Tests of linearity",
                      notes=("Hansen's p-value is bootstrapped (199 replications here; linearity_test(reps=...) for more). "
                             "Small p-values reject the linear AR.",))
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Tests", table)]

    from .smooth_transition import FittedSmoothTransitionAR

    @register_view(FittedSmoothTransitionAR)
    def _(obj: FittedSmoothTransitionAR) -> list[Section]:
        from .nonlinear import _series_data

        test = terasvirta(_series_data(obj.data), obj.dependent, obj.spec.lags, obj.delay, span=obj.span)
        table = Table(["test", "statistic", "p-value"], [[test.name, test.statistic, test.p_value]], title="Test of linearity",
                      notes=("Teräsvirta's LM test of the linear AR against smooth transition; a small p-value supports the model.",))
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Transition", _frame_table(obj.transition_path(), "Weight on the second regime")),
                ("Tests", table)]

    @register_view(FittedBMA)
    def _(obj: FittedBMA) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Models", obj.models_table())]

    @register_view(Backtest)
    def _(obj: Backtest) -> list[Section]:
        frame = obj.errors.copy()
        frame["target"] = frame["target"].astype(str)
        out: list[Section] = [("Accuracy", obj.accuracy()), *_figure(obj.plot)]
        if obj._has_density():
            out += [("Density scores", obj.density_accuracy()), *[("PIT", f) for _, f in _figure(obj.plot_pit)]]
        out.append(("Tests", obj.evaluation()))
        if obj.weights is not None and len(obj.weights):
            out.append(("Weights", obj.weights_table()))
        return [*out, ("Forecasts", Table.from_frame(frame.set_index("origin"), "Forecasts and errors"))]

    @register_view(FittedSystem)
    def _(obj: FittedSystem) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Equations", obj.equation_stats()),
                ("Residual covariance", _frame_table(obj.sigma, "Residual covariance")),
                ("Residual correlation", _frame_table(obj.correlation, "Residual correlation")),
                ("Residuals", _frame_table(obj.resid, "Residuals"))]

    from .models.linear import LinearSolution, SteadyState
    from .models.system import ModelCheck, ModelSystem, ParameterSet

    @register_view(ModelSystem)
    def _(obj: ModelSystem) -> list[Section]:
        return [("Equations", obj.equations_table()), ("Variables", obj.variables_table()), ("Parameters", obj.parameters.table()),
                ("Incidence", obj.incidence_table())]

    @register_view(ModelCheck)
    def _(obj: ModelCheck) -> list[Section]:
        missing = getattr(obj, "missing", None)
        extra = [("Missing inputs", _frame_table(missing.set_index("period"), "Inputs the equations lack, by period"))] \
            if missing is not None and len(missing) else []
        return [("Checks", obj.table()), *extra, ("Equations", obj.system.equations_table()),
                ("Variables", obj.system.variables_table()), ("Parameters", obj.system.parameters.table())]

    @register_view(ParameterSet)
    def _(obj: ParameterSet) -> list[Section]:
        return [("Parameters", obj.table())]

    from .models.measurement import MeasurementMap
    from .models.templates import Template
    from .statespace import StateSpaceSystem

    @register_view(MeasurementMap)
    def _(obj: MeasurementMap) -> list[Section]:
        return [("Measurement", obj.table())]

    @register_view(StateSpaceSystem)
    def _(obj: StateSpaceSystem) -> list[Section]:
        out: list[Section] = [("Measurement", obj.summary())] if obj.measurement is not None else []
        return [*out, ("Measurement matrix", obj.measurement_matrix()), ("Transition", obj.transition()), ("Shocks", obj.shock_impact()),
                ("Shock sizes", obj.covariances())]

    from .factors import FittedFactorModel, FittedPCA
    from .models.calibrate import Calibration
    from .nowcast import FittedBridge, FittedDynamicFactor, News

    @register_view(FittedBridge)
    def _(obj: FittedBridge) -> list[Section]:
        return [("Summary", obj.summary()), *[(t, v) for t, v in sections(obj.equation) if t != "Summary"]]

    @register_view(FittedDynamicFactor)
    def _(obj: FittedDynamicFactor) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Factors", _frame_table(obj.factors, "Smoothed factors")),
                *_provenance(obj)]

    @register_view(News)
    def _(obj: News) -> list[Section]:
        return [("News", obj.summary()), ("By series", _frame_table(obj.by_series.to_frame("impact"), "Impact of each series' new figures"))]

    @register_view(FittedFactorModel)
    def _(obj: FittedFactorModel) -> list[Section]:
        out: list[Section] = [("Summary", obj.summary()), ("Variance explained", _frame_table(obj.explained, "Sums of squared loadings")),
                              ("Test", obj.test.to_table())]
        if obj.factor_correlation is not None:
            out.append(("Factor correlations", _frame_table(obj.factor_correlation, "Correlations of the rotated factors")))
        return [*out, ("Scores", _frame_table(obj.scores(), "Regression scores, observation by observation")), *_provenance(obj)]

    @register_view(FittedPCA)
    def _(obj: FittedPCA) -> list[Section]:
        out: list[Section] = [("Summary", obj.summary()), ("Variance explained", _frame_table(obj.explained.to_frame(), "Share of the variance"))]
        if len(obj.loadings) >= 3:
            out.append(("Number of factors", obj.factor_criteria()))
        return [*out, ("Scores", _frame_table(obj.scores, "Components, period by period"))]
    from .models.estimation import FittedModel

    @register_view(Calibration)
    def _(obj: Calibration) -> list[Section]:
        return [("Summary", obj.summary()), ("Steady state", obj.steady.table()), ("Parameters", obj.model.parameters.table())]

    from .models.decompose import Decomposition
    from .models.moments import Moments

    @register_view(Decomposition)
    def _(obj: Decomposition) -> list[Section]:
        bands: list[Section] = []
        if obj.has_bands:
            bands = [*[("Bands graph", fig) for _, fig in _figure(obj.plot_bands)],
                     *[(f"Bands: {v}", obj.bands_table(v)) for v in obj.contributions]]
        return [*_figure(obj.plot), *[(f"Table: {v}", obj.table(v)) for v in obj.contributions], *bands,
                *([("Shocks", _frame_table(obj.shocks, "Shocks recovered from the path"))] if not obj.shocks.empty else [])]

    @register_view(Moments)
    def _(obj: Moments) -> list[Section]:
        return [("Moments", obj.summary()), ("Variance decomposition", obj.table()),
                ("Correlations", _frame_table(obj.correlation, "Correlations"))]

    from .models.bayes import Identification, Posterior
    from .sweeps import ScenarioSet, SweepResult

    @register_view(SweepResult)
    def _(obj: SweepResult) -> list[Section]:
        out: list[Section] = [("Runs", obj.summary())]
        first = next((r for r in obj.results if r is not None), None)
        if obj.what == "irf" and first is not None:
            out += _figure(obj.plot)
            pairs = [(r, s) for s in list(first.shocks)[:4] for r in list(first.responses)[:4]]
            chosen = (obj.options.get("response"), obj.options.get("shock"))
            if chosen in pairs:  # the pair the graph shows, first
                pairs.remove(chosen)
                pairs.insert(0, chosen)
            for response, shock in pairs:
                out.append((f"{response}, shock to {shock}", obj.table(response, shock)))
        elif obj.what == "steady_state" and first is not None:
            out.append(("Steady states", _frame_table(obj.values(lambda r: {v: float(x) for v, x in r.values.items()}), "Steady state of each run")))
        elif obj.what == "moments" and first is not None:
            out.append(("Standard deviations", _frame_table(obj.values(lambda r: r.std.to_dict()), "Standard deviation of each variable, by run")))
        return out

    from .models.occbin import OccBinPath
    from .models.occbin_filter import OccBinFilter

    from .models.perturbation import SecondOrderSolution

    @register_view(SecondOrderSolution)
    def _(obj: SecondOrderSolution) -> list[Section]:
        steady = pd.Series({v: obj.steady.values[v] for v in obj.endogenous})
        mean = obj.mean()
        table = pd.DataFrame({"steady state": steady, "ergodic mean": mean, "risk correction": mean - steady})
        out: list[Section] = [("Decision rules", obj.rules()),
                              ("Means", _frame_table(table, "The steady state, and the mean once risk is taken into account"))]
        irf = obj.irf(20)
        out += _figure(irf.plot)
        out.append(("First order", obj.first.rules()))
        return out

    from .models.particle import ParticleFilter

    @register_view(ParticleFilter)
    def _(obj: ParticleFilter) -> list[Section]:
        diagnostics = pd.DataFrame({"effective sample size": obj.ess, "resampled": obj.resampled.astype(int),
                                    "log likelihood": obj.loglike_path})
        return [("Summary", obj.summary()),
                ("Filtered", _frame_table(obj.filtered.frame, "Filtered means (data up to each period), levels and fitted observables")),
                ("Quantiles", _frame_table(obj.quantiles.set_axis([f"{v} q{q:g}" for q, v in obj.quantiles.columns], axis=1),
                                           "Weighted quantiles of the filtered particles")),
                ("Diagnostics", _frame_table(diagnostics, "Effective sample size, resampling and each period's log likelihood"))]

    from .models.policy import OptimalPolicy, OSRResult
    from .data.validate import Validation

    @register_view(Validation)
    def _(obj: Validation) -> list[Section]:
        return [("Checks", obj.summary())]

    @register_view(OptimalPolicy)
    def _(obj: OptimalPolicy) -> list[Section]:
        m = obj.moments()
        return [("Summary", obj.summary()), *_figure(lambda: obj.irf(20).plot()),
                ("Moments", m.summary()), ("Decision rules", obj.solution.rules())]

    @register_view(OSRResult)
    def _(obj: OSRResult) -> list[Section]:
        return [("Optimal rule", obj.summary()), ("Decision rules", obj.template.solve().rules())]

    @register_view(OccBinPath)
    def _(obj: OccBinPath) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot),
                ("Paths", _frame_table(obj.frame, "Levels with the constraints")),
                ("Without constraints", _frame_table(obj.linear, "Levels of the linear solution (the reference regime throughout)")),
                ("Binding", _frame_table(obj.binding.astype(int), "1 where the constraint binds")),
                ("Shocks", _frame_table(obj.shocks, "Shocks, period by period"))]

    @register_view(OccBinFilter)
    def _(obj: OccBinFilter) -> list[Section]:
        shocks = [c for c in obj.smoothed.frame.columns if c not in obj.filtered.columns and not c.endswith("_fitted")]
        import pandas as pd

        parts = [obj.binding.astype(int)]
        if obj.smoother == "iterated":
            parts.append(obj.regimes.astype(int).rename(columns=lambda c: f"{c} (smoothed)"))
        parts.append(obj.ex_post.astype(int).rename(columns=lambda c: f"{c} (ex post)"))
        both = pd.concat(parts, axis=1)
        return [("Summary", obj.summary()), *_figure(obj.plot),
                ("Smoothed", _frame_table(obj.smoothed.frame.drop(columns=shocks), "Smoothed estimates (levels) and fitted observables")),
                ("Filtered", _frame_table(obj.filtered, "Filtered estimates (levels), from data up to each period")),
                ("Binding", _frame_table(both, "1 where the constraint binds: the filter's regimes, the smoother's when iterated, and on the smoothed path (ex post)")),
                ("Shocks", _frame_table(obj.smoothed.frame[shocks], "Smoothed shocks"))]

    @register_view(ScenarioSet)
    def _(obj: ScenarioSet) -> list[Section]:
        variables = obj.variables[:12]
        out: list[Section] = [("Scenarios", obj.summary())]
        base_ok = obj.solutions.get(obj.baseline) is not None
        if base_ok and obj.alternatives:
            out.append(("Impact", obj.impact()))
        if obj.loss:
            out.append(("Loss", obj.losses()))
        out += _figure(obj.plot)
        if base_ok and obj.alternatives:
            out += [("Less the baseline", fig) for _, fig in _figure(lambda: obj.plot(deviations=True))]
        out += [(v, obj.table(v)) for v in variables]
        if base_ok:
            out += [(f"{n} less {obj.baseline}", obj.difference(n)) for n in obj.alternatives]
        return out

    @register_view(Posterior)
    def _(obj: Posterior) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), *[("Chains", s) for _, s in _figure(obj.trace)],
                ("Draws", _frame_table(obj.to_frame().set_index(["chain", "draw"]), "Kept draws, with the log posterior"))]

    @register_view(Identification)
    def _(obj: Identification) -> list[Section]:
        return [("Identification", obj.summary())]

    @register_view(FittedModel)
    def _(obj: FittedModel) -> list[Section]:
        smoothed = obj.smoothed.frame
        moments = []
        try:  # a model with a unit root has no unconditional moments
            m = obj.moments()
            moments = [("Moments", m.summary()), ("Variance decomposition", m.table())]
        except Exception:  # noqa: BLE001 - a view must open whatever the model
            pass
        return [("Summary", obj.summary()), ("Parameters", obj.parameters.table()), *moments,
                ("Shock sizes", Table(["standard deviation"], [[v] for v in obj.shock_sd.values()], index=list(obj.shock_sd), title="Shocks")),
                ("Smoothed variables", _frame_table(smoothed, "The model's variables estimated from the data (Kalman smoother)")),
                ("Measurement", obj.measurement.table()), ("Decision rules", obj.solution.rules()), *_provenance(obj)]

    from .models.dsge_var import FittedDSGEVAR

    @register_view(FittedDSGEVAR)
    def _(obj: FittedDSGEVAR) -> list[Section]:
        out: list[Section] = [("Summary", obj.summary())]
        if obj.grid is not None:
            out += [("Marginal density by lambda", obj.lambdas()), *[("Graph: lambda", s) for _, s in _figure(obj.lambda_plot)]]
        try:  # the model's rotation needs as many shocks as observables
            irf = obj.irf(20, draws=200 if obj.posterior is not None else 0, seed=0)
            out += [*_figure(irf.plot), ("Responses", irf.table()), ("Model's own responses", obj.dsge_irf(20).table()),
                    ("Variance decomposition", obj.fevd(20).table())]
        except Exception:  # noqa: BLE001 - a view must open whatever the model
            pass
        prior, _ = obj.var_prior()
        mean, sigma = obj.var_posterior()
        out += [("VAR prior (the model's VAR)", _frame_table(prior, "Phi*: the model's VAR approximation, the prior's mean")),
                ("VAR posterior", _frame_table(mean, "Posterior mean of the VAR's coefficients given the mode")),
                ("Residual covariance", _frame_table(sigma, "Sigma~: the posterior scale of the residual covariance given the mode"))]
        if obj.posterior is not None:
            out += [*[("Chains", s) for _, s in _figure(obj.posterior.trace)],
                    ("Draws", _frame_table(obj.posterior.to_frame().set_index(["chain", "draw"]), "Kept draws, with the log posterior"))]
        return [*out, *_provenance(obj)]

    from .data import Sample

    @register_view(Sample)
    def _(obj: Sample) -> list[Section]:
        return [("Sample", Table(["value"], [[str(obj)], [repr(obj)]], index=["selects", "code"], title="Sample",
                                 notes=["Set it as a dataset's active sample with Data > Sample > Set sample (data.with_sample(...))."]))]

    @register_view(Template)
    def _(obj: Template) -> list[Section]:
        system = obj.model.compile()
        out: list[Section] = [("Equations", system.equations_table()), ("Parameters", system.parameters.table())]
        if obj.measurement is not None:
            out.append(("Measurement", obj.measurement.table()))
        out.append(("Shocks", Table(["standard deviation"], [[v] for v in obj.shock_sd.values()], index=list(obj.shock_sd), title="Shocks")))
        return out

    @register_view(LinearSolution)
    def _(obj: LinearSolution) -> list[Section]:
        return [("Decision rules", obj.rules()), *_figure(obj.plot), ("Steady state", obj.model.steady.table()), ("Roots", obj.roots())]

    @register_view(SteadyState)
    def _(obj: SteadyState) -> list[Section]:
        return [("Steady state", obj.table())]
    from .vecm import FittedVECM

    @register_view(Selection)
    def _(obj: Selection) -> list[Section]:
        return [("Series", Table(["series"], [[n] for n in obj], title=repr(obj)))]

    @register_view(Results)
    def _(obj: Results) -> list[Section]:
        out: list[Section] = [("Summary", obj.summary())]
        if obj._kind() == "fits":
            out.append(("Statistics", obj.stats()))
        if obj.failed:
            out.append(("Failed", obj.failures()))
        return out

    from .structural import AB, StructuralEstimate
    from .identification_extra import HeteroskedasticitySVAR

    @register_view(HeteroskedasticitySVAR)
    def _(obj: HeteroskedasticitySVAR) -> list[Section]:
        return [("Summary", obj.summary()), ("Responses", obj.irf().table())]
    from .nonparametric import KernelDensity, KernelRegression, PartiallyLinear, SemiparametricGARCH
    from .count_series import SeriesModel
    from .nonlinear_extra import FittedThresholdVAR
    from .varma import FittedVARMA
    from .synthdid import SyntheticDiD
    from .cointegrating_regression import CointegratingRegression
    from .panel_time_series import PanelMeanGroup, PooledMeanGroup

    @register_view(CointegratingRegression)
    def _(obj: CointegratingRegression) -> list[Section]:
        return [("Summary", obj.summary()), ("Residuals", _frame_table(obj.resid.to_frame(), "Residuals"))]

    @register_view(PanelMeanGroup)
    def _(obj: PanelMeanGroup) -> list[Section]:
        return [("Summary", obj.summary()), ("Units", _frame_table(obj.units, "Each unit's coefficients"))]

    @register_view(PooledMeanGroup)
    def _(obj: PooledMeanGroup) -> list[Section]:
        return [("Summary", obj.summary()), ("Hausman", obj.hausman().to_table()), ("Units", _frame_table(obj.to_frame(), "Each unit's adjustment and short run"))]

    @register_view(SyntheticDiD)
    def _(obj: SyntheticDiD) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot),
                ("Unit weights", _frame_table(obj.unit_weights.to_frame(), "Control units' weights (omega)")),
                ("Time weights", _frame_table(obj.time_weights.to_frame(), "Pre-treatment periods' weights (lambda)"))]

    @register_view(FittedThresholdVAR)
    def _(obj: FittedThresholdVAR) -> list[Section]:
        return [("Summary", obj.summary()), ("Coefficients", obj.regime_table()),
                ("Regimes", _frame_table(obj.regime.to_frame(), "Regime by period (1: at or below the threshold)"))]

    @register_view(FittedVARMA)
    def _(obj: FittedVARMA) -> list[Section]:
        return [("Summary", obj.summary()), ("Responses", obj.irf().table()),
                ("Residual covariance", _frame_table(pd.DataFrame(obj.sigma, index=list(obj.variables), columns=list(obj.variables)), "Residual covariance"))]

    @register_view(KernelDensity)
    def _(obj: KernelDensity) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Density", _frame_table(obj.to_frame(), "Density on the grid"))]

    @register_view(KernelRegression)
    def _(obj: KernelRegression) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Fitted", _frame_table(obj.to_frame(), "Fitted values and residuals"))]

    @register_view(PartiallyLinear)
    def _(obj: PartiallyLinear) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot)]

    @register_view(SemiparametricGARCH)
    def _(obj: SemiparametricGARCH) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Volatility", _frame_table(obj.volatility.to_frame(), "Conditional volatility"))]

    @register_view(SeriesModel)
    def _(obj: SeriesModel) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Conditional mean", _frame_table(obj.fitted.to_frame(), "Conditional mean"))]

    @register_view(StructuralEstimate)
    def _(obj: StructuralEstimate) -> list[Section]:
        names = list(obj.variables)
        shocks = [f"shock {j + 1}" for j in range(len(names))]
        return [("Summary", obj.summary()), ("A", _frame_table(pd.DataFrame(obj.A, index=names, columns=names), "A")),
                ("B", _frame_table(pd.DataFrame(obj.B, index=names, columns=shocks), "B")),
                ("Impact", _frame_table(pd.DataFrame(obj.impact, index=names, columns=shocks), "Impact matrix A^-1 B"))]

    @register_view(FittedVAR)
    def _(obj: FittedVAR) -> list[Section]:
        if isinstance(obj.identification, AB):
            return [("Structural", obj.structural().summary()), ("Summary", obj.summary()),
                    ("Residual covariance", _frame_table(obj.sigma, "Residual covariance")),
                    *_roots(obj), ("Statistics", _statistics(obj)), ("Residuals", _frame_table(obj.resid.frame, "Residuals")),
                *_residual_correlogram(obj)]
        return [("Summary", obj.summary()), ("Residual covariance", _frame_table(obj.sigma, "Residual covariance")),
                *_roots(obj), ("Statistics", _statistics(obj)), ("Residuals", _frame_table(obj.resid.frame, "Residuals")),
                *_residual_correlogram(obj)]

    from .panelvar import FittedPanelVAR
    from .staggered import BaconDecomposition, StaggeredDiD

    @register_view(StaggeredDiD)
    def _(obj: StaggeredDiD) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("By cohort", _frame_table(obj.cohorts, "ATT by cohort")),
                ("Group-time", _frame_table(obj.att, "Group-time effects" if obj.method == "callaway_santanna" else "Cohort-by-event coefficients"))]

    @register_view(BaconDecomposition)
    def _(obj: BaconDecomposition) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Comparisons", _frame_table(obj.comparisons, "Each 2x2 comparison"))]
    from .proxy import ProxySVAR
    from .tvpvar import FittedTVPVAR
    from .bvar_sv import FittedBVARSV
    from .favar import FittedFAVAR
    from .ms_var import FittedMSVAR
    from .quantile_var import FittedQuantileVAR
    from .spatial import SpatialModel

    @register_view(FittedQuantileVAR)
    def _(obj: FittedQuantileVAR) -> list[Section]:
        return [("Summary", obj.summary()), ("Responses", obj.irf().table())]

    @register_view(SpatialModel)
    def _(obj: SpatialModel) -> list[Section]:
        return [("Summary", obj.summary())]
    from .spectral import CrossSpectrum, FittedARFIMA, Spectrum

    @register_view(Spectrum)
    def _(obj: Spectrum) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Density", _frame_table(obj.to_frame(), "Spectral density by frequency"))]

    @register_view(CrossSpectrum)
    def _(obj: CrossSpectrum) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("By frequency", _frame_table(obj.to_frame(), "Coherence, phase and gain"))]

    @register_view(FittedARFIMA)
    def _(obj: FittedARFIMA) -> list[Section]:
        return [("Summary", obj.summary()), ("Residuals", _frame_table(obj.resid.to_frame(), "Residuals"))]

    @register_view(FittedBVARSV)
    def _(obj: FittedBVARSV) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Volatilities", _frame_table(obj.volatilities(), "Shock standard deviations (median and 68% bands)")),
                ("Responses, last period", obj.irf().table())]

    @register_view(FittedFAVAR)
    def _(obj: FittedFAVAR) -> list[Section]:
        out = [("Summary", obj.summary()), ("Loadings", _frame_table(obj.loadings, "Loadings on the factors and observed variables")),
               ("R-squared", _frame_table(obj.r2.sort_values(ascending=False).to_frame(), "Each series' fit by the factors and observed variables")),
               ("Factors", _frame_table(obj.factors, "Estimated factors")), ("VAR", obj.var.summary())]
        if obj.factor_selection is not None:
            out.insert(1, ("Number of factors", obj.factor_selection))
        return out

    @register_view(FittedMSVAR)
    def _(obj: FittedMSVAR) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Probabilities", _frame_table(obj.smoothed, "Smoothed regime probabilities"))]
    from .simulation_estimators import SimulationEstimate
    from .estimation.gmm import MinimumDistance

    @register_view(MinimumDistance)
    def _(obj: MinimumDistance) -> list[Section]:
        return [("Summary", obj.summary()),
                ("Covariance", _frame_table(pd.DataFrame(obj.cov, index=list(obj.names), columns=list(obj.names)), "Parameter covariance"))]

    @register_view(SimulationEstimate)
    def _(obj: SimulationEstimate) -> list[Section]:
        return [("Summary", obj.summary()),
                ("Covariance", _frame_table(pd.DataFrame(obj.cov, index=list(obj.names), columns=list(obj.names)), "Parameter covariance"))]

    @register_view(FittedTVPVAR)
    def _(obj: FittedTVPVAR) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Volatilities", _frame_table(obj.volatilities(), "Shock standard deviations (median and 68% bands)")),
                ("Responses, last period", obj.irf().table()), *[("Responses over time", f) for _, f in _figure(obj.irf_surface)]]

    @register_view(ProxySVAR)
    def _(obj: ProxySVAR) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Responses", _frame_table(obj.irf.to_frame(), "Responses (with bands)"))]

    from .historical_extra import PanelDecomposition

    @register_view(PanelDecomposition)
    def _(obj: PanelDecomposition) -> list[Section]:  # the entities, then the selected entity's decomposition
        return [("Entities", obj.summary()), *sections(obj.selected)]

    @register_view(FittedPanelVAR)
    def _(obj: FittedPanelVAR) -> list[Section]:
        return [("Summary", obj.summary()), ("Granger causality", obj.granger()),
                ("Residual covariance", _frame_table(pd.DataFrame(obj.sigma, index=list(obj.variables), columns=list(obj.variables)), "Residual covariance")),
                ("Roots", _frame_table(pd.Series(obj.roots, name="modulus"), "Companion-matrix roots"))]

    @register_view(FittedVECM)
    def _(obj: FittedVECM) -> list[Section]:
        return [("Summary", obj.summary()), ("Short run", obj.short_run()), ("Alpha", _frame_table(obj.alpha, "Adjustment")),
                ("Beta", _frame_table(obj.beta, "Cointegrating vectors")),
                *[("Relations", f) for _, f in _figure(obj.relations_plot)],
                ("Relations data", _frame_table(obj.relations().frame, "Cointegrating relations")),
                *[("Recursive eigenvalues", f) for _, f in _figure(obj.recursive_eigenvalues_plot)],
                ("Residuals", _frame_table(obj.resid.frame, "Residuals")), *_residual_correlogram(obj),
                *_roots(obj)]

    @register_view(IRF)
    def _(obj: IRF) -> list[Section]:
        return [*_figure(obj.plot), *[("Heatmap", f) for _, f in _figure(obj.heatmap)], ("Selected horizons", obj.table()),
                ("All horizons", _frame_table(obj.to_frame(), "Impulse responses (with bands)"))]

    @register_view(FEVD)
    def _(obj: FEVD) -> list[Section]:
        return ([("Decomposition", obj.table()), *[("Heatmap", f) for _, f in _figure(obj.heatmap)]]
                + [(v, obj.table(v)) for v in obj.variables]
                + ([(f"Bands: {v}", obj.bands_table(v)) for v in obj.variables] if obj.has_bands else []))

    @register_view(RollingFit)
    def _(obj: RollingFit) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Coefficients", _frame_table(obj.params.frame, "Coefficients by window")),
                ("Std errors", _frame_table(obj.se.frame, "Standard errors by window")), ("Statistics", _frame_table(obj.stats.frame, "Statistics by window")), *_provenance(obj)]



def _register_model_dialogs() -> None:
    """Views for the results of the model dialogs (Model > Export model, Import > FRB/US XML file and EViews state-space
    object, Add factors, Policy > Optimal control)."""
    import pandas as pd

    from .models.control import OptimalControl
    from .models.eviews import EViewsXmlModel
    from .models.writers import Translation
    from .statespace import StateSpace

    @register_view(Translation)
    def _(obj: Translation) -> list[Section]:
        lines = obj.text.splitlines()
        return [("Notes", obj.summary()),
                ("Text", Table(["line", "text"], [[i + 1, line] for i, line in enumerate(lines)],
                               title=f"The model in {obj.syntax}'s syntax: {len(lines)} lines"))]

    @register_view(EViewsXmlModel)
    def _(obj: EViewsXmlModel) -> list[Section]:
        names = sorted(obj.kinds)
        rows = [[n, obj.kinds.get(n, ""), obj.sectors.get(n, ""), obj.mce.get(n, ""), obj.stochastic.get(n, ""),
                 obj.template.labels.get(n, "")] for n in names]
        variables = Table(["kind", "sector", "MCE group", "stochastic", "definition"], [r[1:] for r in rows], index=[r[0] for r in rows],
                          title=f"{obj.template.name}: {len(obj.equations)} equations, {len(obj.model.exogenous)} exogenous")
        equations = Table(["equation (EViews)"], [[obj.equations[n]] for n in obj.equations], index=list(obj.equations),
                          title="The EViews text of each equation used")
        return [("Variables", variables), ("Equations", equations), ("Parameters", obj.model.parameters.table())]

    @register_view(StateSpace)
    def _(obj: StateSpace) -> list[Section]:
        rows = ([["transition", e] for e in obj.transition] + [["measurement", e] for e in obj.measurement]
                + [["shock", f"{k}: sd {v}"] for k, v in obj.shocks.items()])
        params = Table(["start"], [[p.start] for p in obj.params], index=[p.name for p in obj.params], title="Parameters and starting values")
        return [("Equations", Table(["kind", "equation"], rows, title=f"State space: {len(obj.states)} states ({', '.join(obj.states)})")),
                ("Parameters", params)]

    @register_view(OptimalControl)
    def _(obj: OptimalControl) -> list[Section]:
        names = [v for v in obj.weights if v in obj.solution.frame.columns]
        solved, baseline = obj.solution.solved().frame, obj.baseline.solved().frame  # the solved periods: the loss's span
        paths = pd.concat({"optimal": solved[names], "baseline": baseline[names]}, axis=1)
        return [("Summary", obj.summary()), ("Moves", _frame_table(obj.moves, "The instruments' optimal moves, added to their paths")),
                ("Paths", _frame_table(paths, "The loss's variables with the optimal moves and without them")),
                ("Gradient", _frame_table(obj.gradient, "The loss's derivatives in the moves, at the optimum"))]



_register_late()
_register_model_dialogs()


def _register_micro() -> None:
    """Applied microeconometrics: double machine learning, shift-share IV, interactive fixed effects, conditional logit."""
    from .clogit import FittedConditionalLogit
    from .clogit_effects import ConditionalLogitEffects
    from .dml import FittedDoubleML
    from .interactive_fe import FittedInteractiveFE
    from .shiftshare import FittedShiftShare

    @register_view(FittedDoubleML)
    def _(obj: FittedDoubleML) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Splits", _frame_table(obj.splits_table(), "Each sample split's estimate")),
                ("Nuisance fit", _frame_table(obj.nuisance_rmse, "Out-of-fold RMSE of each nuisance function")), *_provenance(obj)]

    @register_view(FittedShiftShare)
    def _(obj: FittedShiftShare) -> list[Section]:
        return [("Summary", obj.summary()), *_figure(obj.plot), ("Rotemberg weights", obj.rotemberg_table()), ("Shock level", obj.shock_level()),
                ("Industries", _frame_table(obj.rotemberg(), "Rotemberg weight, shock, just-identified estimate and exposure of each industry")),
                *_provenance(obj)]

    @register_view(FittedInteractiveFE)
    def _(obj: FittedInteractiveFE) -> list[Section]:
        out = [("Summary", obj.summary()), *_figure(obj.plot), ("Statistics", _statistics(obj))]
        if obj.factors:
            out += [("Factors", _frame_table(obj.F, "Estimated factors (F'F/T = I)")), ("Loadings", _frame_table(obj.loadings, "Each unit's loadings"))]
        if obj.criteria is not None:
            out.append(("Number of factors", _frame_table(obj.criteria, "Bai and Ng's criteria (smallest wins)")))
        return [*out, *_provenance(obj)]

    @register_view(FittedConditionalLogit)
    def _(obj: FittedConditionalLogit) -> list[Section]:
        out = [("Summary", obj.summary()), *_figure(obj.plot), ("Odds ratios", _frame_table(obj.odds_ratios(), "Odds ratios with 95% intervals")),
               ("Statistics", _statistics(obj))]
        if obj.probabilities is not None:
            out.append(("Probabilities", _frame_table(obj.probabilities.to_frame(), "Fitted choice probabilities")))
        if obj.kind == "choice":
            try:
                out.append(("Marginal effects", obj.marginal_effects().summary()))
            except Exception as exc:  # noqa: BLE001 - a convenience: say why there are none
                out.append(("Marginal effects", Table(["note"], [[str(exc)]])))
        return [*out, *_provenance(obj)]

    @register_view(ConditionalLogitEffects)
    def _(obj: ConditionalLogitEffects) -> list[Section]:
        out: list[Section] = [("Summary", obj.summary()), *_figure(obj.plot)]
        how = "averaged over the cases" if obj.at == "average" else "at the means"
        for v in (*obj.regressors, *obj.case_variables):
            what = "column's attribute" if v in obj.regressors else "characteristic"
            out.append((f"Effects: {v}", _frame_table(obj.table(v), f"dP(row)/d{v} ({what}), {how}")))
            out.append((f"Elasticities: {v}", _frame_table(obj.table(v, "elasticity"), f"Elasticities of P(row) in {v} ({what}), {how}")))
            if v in obj.discrete:
                out.append((f"Discrete changes: {v}", _frame_table(obj.table(v, "discrete change"),
                                                                  f"Change in P(row) when {v} ({what}) goes from {obj.discrete[v]}, {how}")))
        every = obj.frame.copy()
        every.index = [" ".join(str(p) for p in i if p != "") for i in every.index]
        out.append(("All", _frame_table(every, "Every effect (variable, measure, probability, change in) with its delta-method standard error")))
        return [*out, *_provenance(obj)]


_register_micro()


# -- plain results of recorded calls (1 November 2026 dialogs: bond measures, Merton at a point, copula portfolio
# VaR): a dict of numbers as a table of its entries, a tuple of arrays as one table per item ----------------

def _plain_value(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    return value if isinstance(value, (bool, int, float, str)) or value is None else repr(value)


def _own_table(obj: Any) -> list[Section]:
    """A result's own table or summary (a named tuple's, say), as the default view finds them."""
    for attr in ("table", "summary"):
        value = getattr(obj, attr, None)
        value = value() if callable(value) else value
        if isinstance(value, Table):
            return [("Results", value)]
    return []


@sections.register(dict)
def _(obj: dict) -> list[Section]:
    """A dict of series by name (add factors by equation) as one table of the series by period; a dict of numbers
    (and arrays) as a table of its entries (and one table per array); any other dict as the default view shows it."""
    import numpy as np
    import pandas as pd

    own = _own_table(obj)
    if own:
        return own
    if obj and all(isinstance(v, pd.Series) for v in obj.values()):
        return [("Values", _frame_table(pd.DataFrame(obj), f"{len(obj)} series, by period"))]
    if not all(isinstance(v, (bool, int, float, str, np.generic, np.ndarray)) or v is None for v in obj.values()):
        return sections.dispatch(object)(obj)
    plain = {k: v for k, v in obj.items() if not isinstance(v, np.ndarray) or v.ndim == 0}
    out: list[Section] = [("Values", Table(["", "value"], [[str(k), _plain_value(v)] for k, v in plain.items()]))] if plain else []
    for k, v in obj.items():
        if k not in plain:
            out += [(str(k), table) for _, table in _matrix_sections(v)]
    return out or [("Value", Table(["value"], [[repr(obj)]]))]


@sections.register(tuple)
def _(obj: tuple) -> list[Section]:
    import numpy as np
    import pandas as pd

    own = _own_table(obj)
    if own:
        return own
    if obj and all(isinstance(v, (np.ndarray, pd.DataFrame, pd.Series)) for v in obj):
        return [(f"Item {i}", table) for i, v in enumerate(obj) for _, table in _matrix_sections(v)]
    return [("Value", Table(["value"], [[repr(obj)]]))]
