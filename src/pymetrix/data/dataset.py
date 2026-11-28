"""Dataset: the one data class.

A long-format pandas DataFrame whose index levels are the dataset's named
axes, plus per-variable metadata. Time series, cross-sections, panels and
multi-dimensional data are all Datasets; ``kind`` says which.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from .._api import api, renamed_arguments
from ..errors import DataError, NotFoundError
from ..expr import ExpressionError, evaluate, parse_expression

CALENDAR_FIELDS = ("quarter", "month", "seasonal()")  # derived from the time axis by Dataset.values_of


def _freq_base(freq: str) -> str:
    """The unit of a pandas period frequency: "Q-JUN" -> "Q", "2M" -> "M", "Y-DEC" / "A" -> "Y"."""
    base = str(freq).split("-")[0].lstrip("0123456789").upper()
    return {"A": "Y", "QE": "Q", "ME": "M", "YE": "Y"}.get(base, base[:1] if base else base)
from ..metadata import DatasetMeta, Measure, Variable
from ..output import Figure, Table
from .axes import Axis, EntityAxis, TimeAxis

if TYPE_CHECKING:
    from ..stats import TestResult

Span = tuple[Any, Any]

# Default aggregation by measure. Over time, stocks take the period's last value;
# across entities, stocks add up.
_OVER_TIME = {
    Measure.FLOW: "sum",
    Measure.COUNT: "sum",
    Measure.STOCK: "last",
    Measure.RATE: "mean",
    Measure.INDEX: "mean",
    Measure.RATIO: "mean",
}
_ACROSS = {**_OVER_TIME, Measure.STOCK: "sum"}


def _latest_vintage(variables: Mapping[str, Variable], inputs: Any) -> Any:
    """A derived series is as recent as its most recently released input."""
    vintages = [variables[n].vintage for n in inputs if n in variables and variables[n].vintage is not None]
    return max(vintages, key=lambda v: v.released) if vintages else None


_CALENDAR_LABELS = {
    "td_mon": "Mondays less Sundays", "td_tue": "Tuesdays less Sundays", "td_wed": "Wednesdays less Sundays",
    "td_thu": "Thursdays less Sundays", "td_fri": "Fridays less Sundays", "td_sat": "Saturdays less Sundays",
    "td_weekdays": "Weekdays less 5/2 of weekend days", "easter": "Share of the Easter window", "leap_year": "Leap-year February",
}


def _ident(value: Any) -> str:
    """An entity value as part of a series name: ``United States`` -> ``United_States``."""
    import re

    text = re.sub(r"\W+", "_", str(value)).strip("_")
    return text if text and not text[0].isdigit() else f"_{text}"


def _flat(names: tuple[Any, ...]) -> tuple[str, ...]:
    """Series names given one by one, as a list, or as a Selection: ``keep("c", "i")``, ``keep(["c", "i"])`` and
    ``keep(data.series("c", "i"))`` are the same."""
    out: list[str] = []
    for n in names:
        if isinstance(n, str):
            out.append(n)
        elif isinstance(n, (list, tuple, set, frozenset, pd.Index)) or (hasattr(n, "names") and hasattr(n, "__iter__")):
            out.extend(str(x) for x in n)
        else:
            out.append(n)
    return tuple(out)


def _consecutive(idx: pd.PeriodIndex) -> bool:
    """Each period once, in order, with none missing between the first and the last."""
    ordinals = idx.asi8
    return len(ordinals) > 0 and ordinals[-1] - ordinals[0] == len(ordinals) - 1 and idx.is_monotonic_increasing and idx.is_unique


def to_periods(values: pd.Series | pd.Index, freq: str | None) -> pd.PeriodIndex:
    """Convert a column of dates, years or period strings to a PeriodIndex."""
    if isinstance(values.dtype, pd.PeriodDtype):
        idx = pd.PeriodIndex(values)
        return idx if freq is None else idx.asfreq(freq)
    if freq is None:
        raise DataError("freq= is required to convert the time column to periods")
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        return pd.DatetimeIndex(values).to_period(freq)
    return pd.PeriodIndex(pd.Index(values).astype(str), freq=freq)


@dataclass(frozen=True, eq=False)
class Dataset:
    frame: pd.DataFrame
    axes: tuple[Axis, ...]
    variables: Mapping[str, Variable] = field(default_factory=dict)
    flags: pd.DataFrame | None = None
    meta: DatasetMeta = field(default_factory=DatasetMeta)

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        names = [a.name for a in axes]
        if list(self.frame.index.names) != names:
            raise DataError(
                f"index levels {list(self.frame.index.names)} do not match axes {names}"
            )
        if sum(isinstance(a, TimeAxis) for a in axes) > 1:
            raise DataError("a Dataset has at most one time axis")
        if not self.frame.index.is_unique:
            raise DataError("each combination of axis values must appear once")
        variables = {
            str(c): self.variables.get(str(c), Variable(str(c))) for c in self.frame.columns
        }
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "variables", variables)

    # construction

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        *,
        time: str | None = None,
        entity: str | Sequence[str] | None = None,
        freq: str | None = None,
        variables: Mapping[str, Variable] | None = None,
        meta: DatasetMeta | None = None,
        source: str | None = None,
    ) -> Dataset:
        """Build a Dataset from a DataFrame, naming its time and entity columns.

        A frame that already has a PeriodIndex needs no arguments. ``source`` (e.g. ``"user:pandas"``) is recorded on
        every series that does not already say where it came from.
        """
        if source is not None:
            built = cls.from_frame(df, time=time, entity=entity, freq=freq, variables=variables, meta=meta)
            stamped = {n: v if v.source else replace(v, source=source) for n, v in built.variables.items()}
            return cls(built.frame, built.axes, stamped, built.flags, built.meta)
        entities = [entity] if isinstance(entity, str) else list(entity or [])
        variables = dict(variables or {})
        meta = meta or DatasetMeta()
        if time is None and not entities:
            if isinstance(df.index, pd.PeriodIndex):
                out = df.copy()
                name = df.index.name or "time"
                out.index = df.index.rename(name)
                return cls(out.sort_index(), (TimeAxis(name, df.index.freqstr),), variables, meta=meta)
            raise DataError("give time= and/or entity= column names, or a frame with a PeriodIndex")
        unnamed = not isinstance(df.index, pd.MultiIndex) and df.index.name is None  # e.g. a filtered frame's row numbers
        out = df.reset_index(drop=True) if unnamed else df.reset_index()
        axes: list[Axis] = [EntityAxis(e) for e in entities]
        keys = list(entities)
        if time is not None:
            periods = to_periods(out[time], freq)
            out[time] = periods
            axes.append(TimeAxis(time, periods.freqstr))
            keys.append(time)
        out = out.set_index(keys).sort_index()
        return cls(out, tuple(axes), variables, meta=meta)

    @classmethod
    @api("pymetrix:dataset.example", show=lambda arguments: {"name"})  # always say which example
    def example(cls, name: str = "sample_macro") -> Dataset:
        """An example dataset shipped with pymetrix (see ``pymetrix.datasets.available()``)."""
        from .._version import __version__
        from ..datasets import load

        data = load(name)
        return Dataset(data.frame, data.axes, data.variables, data.flags,
                       replace(data.meta, vintage={"policy": "example", "name": name, "package_version": __version__}))

    @classmethod
    @api("pymetrix:dataset.empty")
    def empty(cls, freq: str, start: str, end: str) -> Dataset:
        """A time series dataset with no series yet: every period of frequency ``freq`` (``"Y"``, ``"Q"``, ``"M"``,
        ``"W"``, ``"D"``; ``"Y-JUN"`` for a July-June year; ``"A"`` is read as ``"Y"``) from ``start`` to ``end``, both
        included, on a time axis named ``time`` (EViews' ``wfcreate q 1990q1 2030q4``). Add series with ``with_series``:
        ``Dataset.empty("Q", "1990Q1", "2030Q4").with_series(t="1", y=None)``."""
        code = str(freq).strip().upper()
        code = "Y" + code[1:] if code[:1] == "A" else code
        try:
            pd.Period("2000", freq=code)
        except (ValueError, TypeError, AttributeError):
            raise DataError(f"{freq!r} is not a frequency: give Y, Q, M, W or D (Y-JUN for a July-June year)") from None
        try:
            first, last = pd.Period(str(start), freq=code), pd.Period(str(end), freq=code)
        except (ValueError, TypeError, AttributeError):
            raise DataError(f"{start!r} to {end!r} are not periods of frequency {code}: e.g. 1990Q1 for quarters, "
                            "1990-01 for months, 1990 for years") from None
        if last < first:
            raise DataError(f"the dataset ends ({end}) before it starts ({start})")
        periods = pd.period_range(first, last, freq=first.freq).rename("time")
        return cls(pd.DataFrame(index=periods), (TimeAxis("time", periods.freqstr),))

    @classmethod
    @api("pymetrix:dataset.read", show=lambda arguments: {"vintage"} if "://" in str(arguments.get("source", "")) else set())
    def read(cls, source: str, vintage: str = "latest", *, columns: Sequence[str] | str | None = None, where: str | None = None,
             span: Span | str | None = None, embed: bool = True, **kw: Any) -> Dataset:
        """Read a file (by its extension) or a connector's address (``fred://GDPC1``).

        ``vintage`` is which release of a connector's data to ask for: ``"latest"`` (the default, written out in
        generated code) or a date (``"2020-01-30"``, the data as published then, where the provider keeps
        vintages). The request is kept in ``meta.vintage`` and in the provenance of every result from the data.

        For a Parquet file, or a folder of them (partitioned as ``year=2010/...``), ``columns``, ``where`` (a row
        condition, as in ``select``) and ``span`` (``(start, end)`` on the time axis) read only part of it: pyarrow
        skips the columns not named and the folders and row groups the condition rules out, and the condition is then
        applied exactly (``pymetrix.io.selection``). What was read is kept in ``meta.extra["loads"]``.

        ``embed=False`` asks projects to keep only a reference to the file (its path, the selection, and a fingerprint
        of its contents) instead of a copy of the data read; opening the project reads the file again, and refuses if
        the file has changed. For data too large to copy into every project."""
        from datetime import datetime
        from pathlib import Path

        from ..io import read

        remote = "://" in str(source)
        if vintage not in (None, "latest"):
            if not remote:
                raise DataError(f"vintage= is for connectors (fred://, sdmx://...); {source} is a file")
            kw["vintage"] = vintage
        selection = {k: v for k, v in (("columns", columns), ("where", where), ("span", span)) if v is not None}
        parquet = not remote and (Path(str(source)).is_dir() or str(source).lower().endswith(".parquet"))
        if selection and not parquet:
            raise DataError(f"columns=, where= and span= read part of a Parquet file or folder; {source} is not one: "
                            "read it whole, then use keep() and select()")
        if not embed and not parquet:
            raise DataError(f"embed=False keeps a reference to a Parquet file or folder in projects; {source} is not one")
        data = read(source, **selection, **kw)
        if remote:
            request = {"policy": "as_of", "date": str(vintage)} if vintage not in (None, "latest") else \
                {"policy": "latest", "retrieved": datetime.now().isoformat(timespec="seconds")}
        else:
            path = Path(str(source))
            modified = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") if path.exists() else None
            request = {"policy": "file", "source": str(source), "modified": modified}
            if not embed:
                from ..io.selection import fingerprint

                request.update(embed=False, path=str(path.resolve()), fingerprint=fingerprint(path))
        return Dataset(data.frame, data.axes, data.variables, data.flags, replace(data.meta, vintage=request))

    @renamed_arguments(target="path")
    def write(self, path: str, **kw: Any) -> None:
        from ..io import write

        write(self, path, **kw)

    # structure

    @property
    def time_axis(self) -> TimeAxis | None:
        return next((a for a in self.axes if isinstance(a, TimeAxis)), None)

    @property
    def entity_axes(self) -> tuple[EntityAxis, ...]:
        return tuple(a for a in self.axes if isinstance(a, EntityAxis))

    @property
    def kind(self) -> Literal["cross_section", "time_series", "panel", "multi"]:
        n_entity = len(self.entity_axes)
        if self.time_axis is None:
            return "cross_section" if n_entity == 1 else "multi"
        return {0: "time_series", 1: "panel"}.get(n_entity, "multi")  # type: ignore[return-value]

    @property
    def names(self) -> list[str]:
        return [str(c) for c in self.frame.columns]

    def __len__(self) -> int:
        return len(self.frame)

    def __contains__(self, name: object) -> bool:
        return name in self.frame.columns

    def __getitem__(self, name: str) -> pd.Series:
        if name not in self.frame.columns:
            raise NotFoundError(f"no series {name!r} in the dataset")
        return self.frame[name]

    def __repr__(self) -> str:
        shown = ", ".join(self.names[:8]) + (", ..." if len(self.names) > 8 else "")
        return f"<Dataset {self.kind}: {len(self)} rows, {len(self.names)} variables ({shown})>"

    def values_of(self, name: str) -> pd.Series:
        """A column, or the values of an index level, or a calendar field of the time axis (``quarter``, ``month``,
        and ``seasonal()``, the period of the year: the quarter of quarterly data, the month of monthly data), aligned
        to the rows. A column of that name comes first."""
        if name in self.frame.columns:
            return self.frame[name]
        if name in self.frame.index.names:
            return pd.Series(self.frame.index.get_level_values(name), index=self.frame.index, name=name)
        if name in CALENDAR_FIELDS:
            return self._calendar(name)
        raise NotFoundError(f"{name!r} is neither a series nor an axis of the dataset")

    def calendar_names(self) -> list[str]:
        """The calendar fields ``values_of`` derives from the time axis for this frequency (``quarter``, ``month``)."""
        return [n for n in ("quarter", "month") if n not in self.frame.columns and self._calendar_ok(n)]

    def _calendar_ok(self, name: str) -> bool:
        t = self.time_axis
        if t is None or t.name not in self.frame.index.names:
            return False
        base = _freq_base(t.freq)
        if name == "seasonal()":
            return base in ("Q", "M")
        return base in {"quarter": ("Q", "M", "W", "D", "B", "H"), "month": ("M", "W", "D", "B", "H")}[name]

    def _calendar(self, name: str) -> pd.Series:
        t = self.time_axis
        if t is None or t.name not in self.frame.index.names:
            raise NotFoundError(f"{name} needs a time axis, and this dataset has none")
        if not self._calendar_ok(name):
            what = "seasonal() needs quarterly or monthly data" if name == "seasonal()" else f"{t.freq} data have no {name}s"
            raise NotFoundError(f"{what} (the time axis is {t.freq})")
        periods = pd.PeriodIndex(self.frame.index.get_level_values(t.name))
        field = ("quarter" if _freq_base(t.freq) == "Q" else "month") if name == "seasonal()" else name
        label = f"seasonal({_freq_base(t.freq)})" if name == "seasonal()" else name  # dummies name the seasons by it
        out = pd.Series(np.asarray(getattr(periods, field), dtype=int), index=self.frame.index, name=label)
        out.attrs["calendar"] = True  # dummies order the seasons by number (2 before 10)
        return out

    # expressions

    def lagged(self, name: str, lag: int) -> pd.Series:
        """``name`` shifted by ``lag`` periods (negative = past), always within entity."""
        s = self[name]
        if lag == 0:
            return s
        t = self.time_axis
        if t is None:
            raise ExpressionError(f"{name}[{lag:+d}]: lags need a time axis, and this dataset has none")
        idx = s.index
        if isinstance(idx, pd.MultiIndex):
            level = idx.names.index(t.name)
            arrays = [
                idx.get_level_values(i) - lag if i == level else idx.get_level_values(i)
                for i in range(idx.nlevels)
            ]
            moved = pd.Series(s.to_numpy(), index=pd.MultiIndex.from_arrays(arrays, names=idx.names))
        else:
            if s.dtype == np.float64 and isinstance(idx, pd.PeriodIndex) and _consecutive(idx):
                # every period once, in order, none missing: the shift is by position (what the reindex below gives)
                values, n = s.to_numpy(), len(idx)
                out = np.full(n, np.nan)
                if abs(lag) < n:
                    if lag < 0:
                        out[-lag:] = values[:n + lag]
                    else:
                        out[:n - lag] = values[lag:]
                return pd.Series(out, index=idx, name=name)
            moved = pd.Series(s.to_numpy(), index=idx - lag)
        return moved.reindex(idx).rename(name)

    def eval(self, expr: str) -> pd.Series:
        node = parse_expression(expr)
        out = evaluate(node, self.lagged)
        if not isinstance(out, pd.Series):
            out = pd.Series(out, index=self.frame.index, dtype=float)
        return out.rename(expr)

    @api("pymetrix:dataset.with_series")
    def with_series(self, *, within: Any = None, **exprs: str | pd.Series | None) -> Dataset:
        """Add or replace series, from expressions or from aligned Series; ``None`` makes a series of missing values
        (EViews' ``series y``).

        The values are fixed (like EViews' ``genr``); use ``derive`` for series that
        follow their inputs. Derived series that depend on a replaced series are
        recomputed; a derived series replaced here becomes a plain one. A replaced series keeps
        its label, description and units, but not its source, vintage, retrieval time or external
        reference: those described the old values.

        ``within`` (a ``Sample``, a ``(start, end)`` span or ``"start:end"``) assigns only the rows it selects and
        keeps the others' values, missing for a new series: EViews' ``smpl 2000q1 2010q4`` then ``y = ...``. Lags
        still reach outside it. A formula that uses the series it assigns is then computed one period at a time,
        each period seeing the values just assigned before it, as EViews does: ``y="y[-1]*1.02"`` grows ``y`` by 2%
        a period through the sample. Without ``within`` every row is assigned and a formula sees the values as they
        were before the call. A series named ``within`` is added with ``add_series``.
        """
        mask, sample = None, None
        if within is not None:
            from .sample import Sample

            sample = within if isinstance(within, Sample) else Sample(within)
            mask = sample.mask(self)
            if not mask.any():
                raise DataError(f"the sample {sample} selects no rows of this dataset")
        frame = self.frame.copy()
        variables = dict(self.variables)
        for name, value in exprs.items():
            old = variables.get(name)
            kept = self._kept_values(name) if mask is not None else None
            if isinstance(value, str):
                if mask is not None and self.time_axis is not None and self._uses(value, name):
                    values = self._by_period(name, value, mask, kept)
                else:
                    values = self.eval(value).to_numpy()
            elif value is None:
                values = np.full(len(frame), np.nan)
            else:
                values = self._aligned(name, value)  # a Series, so its dtype stays (text, nullable integers)
            frame[name] = values if mask is None else np.where(mask, np.asarray(values, dtype=float), kept)
            if isinstance(value, str) and (mask is None or old is None):
                variables[name] = Variable(name, derived_from=value if mask is None else f"{value} within {sample}")
            elif old is not None:
                variables[name] = replace(old, recipe=None, derived_from=f"{value} within {sample}" if isinstance(value, str) else None,
                                          source="", source_key="", vintage=None, release_lag=None, retrieved="", external_ref="")
            else:
                variables[name] = Variable(name)
        out = Dataset(frame, self.axes, variables, self.flags, self.meta)
        return out._recompute(set(exprs))

    def _kept_values(self, name: str) -> np.ndarray:
        """A series' values as numbers, for the rows an assignment within a sample keeps (missing for a new one)."""
        if name not in self.frame.columns:
            return np.full(len(self.frame), np.nan)
        try:
            return self.frame[name].to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError):
            raise DataError(f"{name} is not numeric, so it cannot be assigned within a sample") from None

    @staticmethod
    def _uses(expr: str, name: str) -> bool:
        from ..expr.nodes import Name

        return any(isinstance(n, Name) and n.id == name for n in parse_expression(expr).walk())

    def _by_period(self, name: str, expr: str, mask: np.ndarray, start: np.ndarray) -> np.ndarray:
        """``expr`` for series ``name``, which it uses, one period of ``mask`` at a time, in order: each period sees
        the values assigned before it (EViews' assignment under ``smpl``). All of a panel's entities at once."""
        frame = self.frame.copy()
        frame[name] = start
        work = Dataset(frame, self.axes, self.variables, self.flags, self.meta)
        periods = np.asarray(frame.index.get_level_values(self.time_axis.name).asi8)
        for period in np.unique(periods[mask]):
            rows = mask & (periods == period)
            column = frame[name].to_numpy(dtype=float, copy=True)
            column[rows] = work.eval(expr).to_numpy(dtype=float)[rows]
            frame[name] = column
        return frame[name].to_numpy(dtype=float)

    def _aligned(self, name: str, value: pd.Series) -> pd.Series:
        """``value`` on this dataset's rows. A series of another frequency, or one none of whose periods are this
        dataset's, is an error rather than a column of missing values: a dataset has one frequency (``to_freq``)."""
        mine = self.frame.index.get_level_values(self.time_axis.name) if self.time_axis is not None else None
        theirs = value.index
        if isinstance(theirs, pd.MultiIndex) and mine is not None and self.time_axis.name in theirs.names:
            theirs = theirs.get_level_values(self.time_axis.name)
        if isinstance(mine, pd.PeriodIndex) and isinstance(theirs, pd.PeriodIndex) and mine.freqstr != theirs.freqstr:
            have, need = theirs.freqstr.split("-")[0], mine.freqstr.split("-")[0]
            raise DataError(f"{name} is {have}, but this dataset is {need}: convert it first "
                            f"(e.g. Dataset.from_frame(...).to_freq({need!r})) or keep it in a dataset of its own")
        out = value.reindex(self.frame.index)
        if len(self.frame) and value.notna().any() and out.isna().all():
            raise DataError(f"none of the periods (or entities) of {name} are in this dataset: check its index")
        return out

    # selections of series

    @api("pymetrix:dataset.series")
    def series(self, *patterns: str, where: Any = None, **meta: Any) -> Any:
        """A ``Selection`` of series: named, by pattern (``"*_cycle"``), by metadata (``units="USD bn"``,
        ``adjustment="sa"``, ``derived=True``) or by a test on each ``Variable`` (``where=``)."""
        patterns = _flat(patterns)
        from ..selection import select_series

        if "derived" in meta:
            meta["recipe"] = meta.pop("derived")
        return select_series(self, patterns, where, **meta)

    group = series  # the earlier name, so scripts written with it still run

    # derived series

    @api("pymetrix:dataset.derive")
    def derive(self, **recipes: Any) -> Dataset:
        """Add series that keep their recipe and follow their inputs.

        A value is an expression (``dly="dlog(y)"``) or a recipe from
        ``pymetrix.data.derived`` (``Filtered("y", HP(), "trend")``). When a
        series they use is replaced (``with_series``), they are recomputed.
        """
        from .derived import Formula

        variables = dict(self.variables)
        frame = self.frame.copy()
        for name, recipe in recipes.items():
            if isinstance(recipe, str):
                from ..expr.nodes import RANDOM, Call

                drawn = sorted({n.fn for n in parse_expression(recipe).walk() if isinstance(n, Call) and n.fn in RANDOM})
                if drawn:
                    raise DataError(f"{name}: {', '.join(f + '()' for f in drawn)} draws random numbers, which are values, not "
                                    "a recipe to follow: make the series with with_series (Generate series), not derive")
            recipe = Formula(recipe) if isinstance(recipe, str) else recipe
            missing = sorted(set(recipe.inputs(self)) - set(self.names) - set(recipes))
            if missing:
                raise DataError(f"{name}: no series {', '.join(missing)} in the dataset")
            old = variables.get(name, Variable(name))
            variables[name] = replace(old, recipe=recipe, derived_from=recipe.describe(self))
            if name not in frame.columns:
                frame[name] = np.nan
        out = Dataset(frame, self.axes, variables, self.flags, self.meta)
        return out._recompute(None, only=set(recipes))

    def with_each(self, series: Any, name: str, formula: str, derived: bool = False, replace: bool = False,
                  within: Any = None) -> Dataset:
        """A new series for each of ``series`` (names separated by spaces or commas, wildcards such as ``"ip_*"``, a
        list or a ``Selection``), named by the template ``name`` and computed by the template ``formula``:
        ``data.with_each("gdp cons inv", "{x}_g", "100*d(log({x}))")``. ``{x}`` is the series; ``{1}``, ``{2}``... the
        parts a wildcard's ``*`` or ``?`` matched (``"ip_*"`` with ``"{1}_growth"``). With ``derived`` they are derived
        series (``derive``); otherwise their values are computed once (``with_series``). Series that exist already
        are an error unless ``replace``. ``within`` assigns only a sample's rows, as ``with_series`` does (not with
        ``derived``).

        Recorded and scripted as the one ``with_series`` (or ``derive``) call it makes, with every new series written
        out, so a script replays exactly whatever series are added later. ``pymetrix.data.each.preview_each`` shows
        the rows without making them."""
        from ..errors import SpecificationError
        from .each import preview_each

        found = preview_each(self, series, name, formula, replace=replace, derived=derived)
        if not found.ok:
            raise SpecificationError("; ".join(found.errors()))
        if derived:
            if within is not None:
                raise DataError("a derived series follows its formula in every period: within= is for with_series")
            return self.derive(**found.formulas())
        return self.with_series(**({"within": within} if within is not None else {}), **found.formulas())

    @property
    def derived(self) -> dict[str, Any]:
        """Derived series and their recipes."""
        return {n: v.recipe for n, v in self.variables.items() if v.recipe is not None}

    def recompute(self, *names: str) -> Dataset:
        """Recompute derived series (all, or those named, with what depends on them)."""
        names = _flat(names)
        return self._recompute(None, only=set(names) or None)

    @api("pymetrix:dataset.freeze")
    def freeze(self, *names: str) -> Dataset:
        """Keep the values of derived series (all, or those named) but drop their recipes."""
        names = _flat(names)
        variables = dict(self.variables)
        for name in names or tuple(self.derived):
            if variables[name].recipe is not None:
                variables[name] = replace(variables[name], recipe=None)
        return Dataset(self.frame, self.axes, variables, self.flags, self.meta)

    def _recompute(self, changed: set[str] | None, only: set[str] | None = None) -> Dataset:
        """Recompute derived series downstream of ``changed`` (None: all), or ``only`` and what follows them."""
        from .derived import order

        recipes = self.derived
        if not recipes:
            return self
        everything = order(recipes, self)  # dependency order; raises on circular definitions
        if only is not None:
            wanted = set(only) | set(order(recipes, self, set(only)))
        elif changed is not None:
            wanted = set(order(recipes, self, changed))
        else:
            wanted = set(everything)
        todo = [n for n in everything if n in wanted]
        if not todo:
            return self
        frame = self.frame.copy()
        variables = dict(self.variables)
        current = self
        for name in todo:
            recipe = recipes[name]
            extra = variables[name].extra
            try:
                if hasattr(recipe, "compute_noted"):  # a recipe that reports on its run (X-13): provenance
                    values, extra = recipe.compute_noted(current, extra)
                else:
                    values = recipe.compute(current)
                frame[name] = values.reindex(frame.index).to_numpy()
            except (ValueError, ExpressionError) as exc:
                raise DataError(f"{name} ({recipe.describe(current)}) could not be recomputed: {exc}") from exc
            variables[name] = replace(variables[name], vintage=_latest_vintage(variables, recipe.inputs(current)), extra=extra)
            current = Dataset(frame, self.axes, variables, self.flags, self.meta)
        return current

    # editing: each returns a new dataset, so every change is a step that can be recorded and replayed

    def row_key(self, key: Any) -> Any:
        """An index label from a row key as written in code: ``"2020Q1"`` (time series),
        ``"FRA"`` (cross-section) or ``("FRA", "2020Q1")`` (panel, in axis order)."""
        parts = key if isinstance(key, tuple) else (key,)
        if len(parts) != len(self.axes):
            raise NotFoundError(f"row {key!r}: give one value per axis ({', '.join(a.name for a in self.axes)})")
        out = []
        for axis, part in zip(self.axes, parts):
            if isinstance(axis, TimeAxis):
                level = self.frame.index.get_level_values(axis.name)
                out.append(pd.Period(str(part), freq=level.freq))
            else:
                out.append(part)
        label = tuple(out) if len(out) > 1 else out[0]
        if label not in self.frame.index:
            raise NotFoundError(f"no row {key!r} in the dataset")
        return label

    def flag(self, name: str) -> pd.Series:
        """The status of each observation of ``name`` (``Status`` values; empty = normal)."""
        if self.flags is None or name not in self.flags.columns:
            return pd.Series("", index=self.frame.index, name=name, dtype=object)
        return self.flags[name].fillna("").rename(name)

    @api("pymetrix:dataset.edit")
    def edit(self, values: Mapping[str, Mapping[Any, Any]]) -> Dataset:
        """Change observations by hand: ``{"c": {"2020Q1": 1.2}}``; ``None`` makes a value missing.

        Edited cells are flagged ``edited``; the series' vintage stays as released. Derived
        series that depend on an edited series are recomputed. Derived series themselves
        cannot be edited: freeze them first.
        """
        from ..metadata import Status

        frame = self.frame.copy()
        flags = self.flags.copy() if self.flags is not None else pd.DataFrame(index=frame.index)
        for name, cells in values.items():
            if name not in frame.columns:
                raise NotFoundError(f"no series {name!r} in the dataset")
            if self.variables[name].recipe is not None:
                raise DataError(f"{name} is derived ({self.variables[name].derived_from}); freeze it to edit its values")
            if name not in flags.columns:
                flags[name] = pd.Series(None, index=frame.index, dtype=object)
            for key, value in cells.items():
                label = self.row_key(key)
                frame.loc[label, name] = np.nan if value is None else float(value)
                flags.loc[label, name] = Status.EDITED.value
        out = Dataset(frame, self.axes, self.variables, flags, self.meta)
        return out._recompute(set(values))

    @api("pymetrix:dataset.rename")
    def rename(self, **names: str) -> Dataset:
        """Rename series (``old="new"``); recipes that use them are updated to the new names."""
        from .derived import Filtered, Formula

        for old, new in names.items():
            if old not in self.frame.columns:
                raise NotFoundError(f"no series {old!r} in the dataset")
            if not new.isidentifier():
                raise DataError(f"{new!r} is not a valid series name")
            if new in self.frame.columns and new not in names:
                raise DataError(f"there is already a series {new!r}")

        def renamed(recipe: Any) -> Any:
            if isinstance(recipe, Formula):
                from ..expr import Name, parse_expression

                node = parse_expression(recipe.expr).map(lambda n: Name(names[n.id], n.lag) if isinstance(n, Name) and n.id in names else None)
                return Formula(str(node))
            if isinstance(recipe, Filtered) and recipe.of in names:
                return replace(recipe, of=names[recipe.of])
            return recipe

        variables = {}
        for name, var in self.variables.items():
            new_name = names.get(name, name)
            recipe = renamed(var.recipe) if var.recipe is not None else None
            variables[new_name] = replace(var, name=new_name, recipe=recipe,
                                          derived_from=recipe.describe(self) if recipe is not None else var.derived_from)
        flags = self.flags.rename(columns=names) if self.flags is not None else None
        return Dataset(self.frame.rename(columns=names), self.axes, variables, flags, self.meta)

    @api("pymetrix:dataset.keep")
    def keep(self, *names: str) -> Dataset:
        """Only these series, in this order (the others removed, as ``drop`` would)."""
        names = _flat(names)
        missing = [n for n in names if n not in self.frame.columns]
        if missing:
            raise NotFoundError(f"no series {', '.join(missing)} in the dataset")
        others = [n for n in self.names if n not in names]
        kept = self.drop(*others) if others else self
        return Dataset(kept.frame[list(names)], kept.axes, {n: kept.variables[n] for n in names if n in kept.variables}, None, kept.meta)

    @api("pymetrix:dataset.drop")
    def drop(self, *names: str) -> Dataset:
        """Remove series. A series that a remaining derived series uses cannot be removed."""
        names = _flat(names)
        missing = [n for n in names if n not in self.frame.columns]
        if missing:
            raise NotFoundError(f"no series {', '.join(missing)} in the dataset")
        users = sorted(n for n, r in self.derived.items() if n not in names and set(r.inputs(self)) & set(names))
        if users:
            raise DataError(f"{', '.join(users)} {'is' if len(users) == 1 else 'are'} derived from {', '.join(names)}; freeze or drop {'it' if len(users) == 1 else 'them'} first")
        variables = {n: v for n, v in self.variables.items() if n not in names}
        flags = self.flags.drop(columns=[n for n in names if n in self.flags.columns]) if self.flags is not None else None
        return Dataset(self.frame.drop(columns=list(names)), self.axes, variables, flags, self.meta)

    @api("pymetrix:dataset.with_info")
    def with_info(self, name: str, **info: Any) -> Dataset:
        """Change a series' description: ``label``, ``units``, ``description``, ``source``, ``measure``,
        ``adjustment``, ``prices``, ``scale``, ``symbol``..."""
        from ..metadata import Adjustment, Measure, Prices

        if name not in self.variables:
            raise NotFoundError(f"no series {name!r} in the dataset")
        allowed = {f for f in Variable.__dataclass_fields__ if f not in ("name", "recipe", "derived_from", "vintage", "extra")}
        unknown = sorted(set(info) - allowed)
        if unknown:
            raise DataError(f"cannot set {', '.join(unknown)}; settable: {', '.join(sorted(allowed))}")
        enums = {"measure": Measure, "adjustment": Adjustment, "prices": Prices}
        info = {k: (enums[k](v) if k in enums and v not in (None, "") else (None if k == "measure" and v in ("", None) else v)) for k, v in info.items()}
        variables = dict(self.variables)
        variables[name] = replace(variables[name], **info)
        return Dataset(self.frame, self.axes, variables, self.flags, self.meta)

    def _static(self) -> dict[str, Variable]:
        """Variables with recipes dropped: for subsets and conversions, whose values must not be recomputed."""
        return {n: replace(v, recipe=None) if v.recipe is not None else v for n, v in self.variables.items()}

    # selection

    @property
    def sample(self) -> Any:
        """The active sample (a ``Sample``), or None for all rows: what estimation, tests, statistics and graphs
        use when no other sample is given."""
        from .sample import Sample

        return Sample.from_dict(self.meta.sample) if self.meta.sample is not None else None

    @api("pymetrix:dataset.over")
    def over(self, sample: Any = None, **kw: Any) -> Dataset:
        """The same data over a sample, for chains: ``data.over("1990Q1:2019Q4").fit(Equation("c ~ y"))``.
        ``with_sample`` under a shorter name: the sample becomes the dataset's active one, so it is in the code and
        in every result's provenance."""
        return self.with_sample(sample, **kw)

    @api("pymetrix:dataset.fit")
    def fit(self, spec: Any, **options: Any) -> Any:
        """Estimate ``spec`` (an ``Equation``, ``VAR``, ``System``... anything with ``fit(data, ...)``) on this
        data: ``data.over("1990Q1:2019Q4").fit(Equation("c ~ y"), method=OLS(cov="hac"))`` is
        ``Equation("c ~ y").fit(data.over("1990Q1:2019Q4"), method=OLS(cov="hac"))``."""
        if not callable(getattr(spec, "fit", None)):
            raise TypeError(f"a {type(spec).__name__} cannot be estimated: give an Equation, VAR, System or other specification")
        return spec.fit(self, **options)

    @api("pymetrix:dataset.with_sample", show=lambda arguments: {"sample"})  # with_sample(None) clears it
    def with_sample(self, sample: Any = None, **kw: Any) -> Dataset:
        """The same data with an active sample (EViews' ``smpl``): a ``Sample``, a ``(start, end)`` span, or the
        arguments of one (``with_sample(end="2007Q4", country=["DEU"])``); ``None`` for all rows again. Lags and
        transformations still use all the data; the sample chooses the rows commands use."""
        from .sample import Sample

        if kw:
            if sample is not None:
                raise DataError("give a Sample or its arguments, not both")
            sample = Sample(**kw)
        elif sample is not None and not isinstance(sample, Sample):
            sample = Sample(sample)
        if sample is not None:
            if not sample.mask(self).any():
                raise DataError(f"the sample {sample} selects no rows of this dataset")
            if sample.everything:
                sample = None
        meta = replace(self.meta, sample=sample.to_dict() if sample is not None else None)
        return Dataset(self.frame, self.axes, self.variables, self.flags, meta)

    # to and from NumPy, pandas and statsmodels (see pymetrix.interop)

    def to_frame(self, names: Any = None, sample: Any = None, missing: str = "keep") -> pd.DataFrame:
        """The series ``names`` (default all) on the sample's rows (``None``: the active sample, as commands use;
        ``"all"``; or a span or ``Sample``), with this dataset's index; ``missing``: keep, drop or raise."""
        from ..interop import to_frame

        return to_frame(self, names, sample, missing)

    def to_pandas(self, names: Any = None, sample: Any = "all") -> pd.DataFrame:
        """The series as a plain pandas DataFrame (default: every series, every row), with this dataset's index and
        no metadata: the way out to pandas. ``sample=None`` uses the active sample; a span or ``Sample`` selects rows."""
        frame = self.to_frame(names, sample)
        out = frame.copy()
        out.attrs = {}
        return out

    def to_numpy(self, names: Any = None, sample: Any = None, missing: str = "keep") -> np.ndarray:
        """``to_frame(...)`` as a float array (rows by series)."""
        return self.to_frame(names, sample, missing).to_numpy(dtype=float)

    def design(self, y: str, x: Any = (), sample: Any = None, add_constant: bool = True, missing: str = "drop",
               where: str | None = None) -> tuple[pd.Series, pd.DataFrame]:
        """``(y, X)`` for statsmodels, SciPy or scikit-learn, labelled, on the rows pymetrix's own estimators would use;
        names or formula expressions (``log(c)``, ``c[-1]``). See ``pymetrix.interop.design``."""
        from ..interop import design

        return design(self, y, x, sample, add_constant, missing, where)

    def add_series(self, name: str, values: Any, source: str | None = None, overwrite: bool = False, label: str = "",
                   units: str = "") -> Dataset:
        """A copy with series ``name`` from a pandas Series or an array; an existing name only with ``overwrite``; the
        source (default ``user:pandas`` or ``user:numpy``) is recorded."""
        from ..interop import add_series

        return add_series(self, name, values, source, overwrite, label, units)

    def apply_np(self, name: str, func: Any, series: Any, *, source_expr: str | None = None, overwrite: bool = False) -> Dataset:
        """Add ``name = func(arrays of series...)``, recorded as user code (see ``pymetrix.interop.apply_np``)."""
        from ..interop import apply_np

        return apply_np(self, name, func, series, source_expr=source_expr, overwrite=overwrite)

    def apply_pd(self, name: str, func: Any, series: Any, *, source_expr: str | None = None, overwrite: bool = False) -> Dataset:
        """Add ``name = func(pandas series...)``, recorded as user code (see ``pymetrix.interop.apply_pd``)."""
        from ..interop import apply_pd

        return apply_pd(self, name, func, series, source_expr=source_expr, overwrite=overwrite)

    def transform_each(self, selection: Any, func: Any, prefix: str = "", suffix: str = "", source_expr: str | None = None,
                       overwrite: bool = False) -> Dataset:
        """Add ``prefix + s + suffix = func(s)`` for each series of ``selection``."""
        from ..interop import transform_each

        return transform_each(self, selection, func, prefix, suffix, source_expr, overwrite)

    def sample_record(self, span: Any = None, where: str | None = None, **levels: Any) -> tuple[str, dict[str, Any] | None]:
        """Which rule ``sample_mask`` would follow for these arguments, and that rule in full, for provenance:
        ``("explicit", {...})``, ``("active sample of the dataset", sample.to_dict())`` or ``("all observations", None)``.
        A result then records how its rows were chosen, not only its first and last period."""
        from .sample import Sample, as_span

        span = as_span(span)
        if isinstance(span, Sample):
            return "explicit", span.to_dict()
        if span is not None or where is not None or levels:
            rule: dict[str, Any] = {}
            if span is not None:
                rule["span"] = [None if p is None else str(p) for p in span]
            if where is not None:
                rule["where"] = str(where)
            if levels:
                rule["levels"] = {k: str(v) for k, v in levels.items()}
            return "explicit", rule
        active = self.sample
        if active is not None:
            return "active sample of the dataset", active.to_dict()
        return "all observations", None

    def sample_mask(self, span: Any = None, where: str | None = None, **levels: Any) -> np.ndarray:
        """The rows a command uses: those of ``span`` (a ``(start, end)`` pair or a ``Sample``), ``where`` and
        ``levels`` when any is given; otherwise the active sample's (all rows without one)."""
        from .sample import Sample, as_span

        span = as_span(span)
        if isinstance(span, Sample):
            mask = span.mask(self)
            span = None
        elif span is None and where is None and not levels:
            active = self.sample
            return active.mask(self) if active is not None else np.ones(len(self.frame), dtype=bool)
        else:
            mask = np.ones(len(self.frame), dtype=bool)
        if span is not None:
            t = self.time_axis
            if t is None:
                raise DataError("span= needs a time axis; use where= to select a cross-section")
            periods = self.frame.index.get_level_values(t.name)
            start, end = span
            if start is not None:
                mask &= np.asarray(periods >= pd.Period(str(start), freq=periods.freq))
            if end is not None:
                mask &= np.asarray(periods <= pd.Period(str(end), freq=periods.freq))
        if where is not None:
            from .conditions import condition_mask

            mask &= condition_mask(self.frame, where)  # checked before it runs: a project's conditions may be anyone's
        for level, values in levels.items():
            wanted = values if isinstance(values, (list, tuple, set)) else [values]
            mask &= np.asarray(self.values_of(level).isin(wanted))
        return mask

    def select(self, where: str | None = None, span: Span | None = None, **levels: Any) -> Dataset:
        """Some of the periods or rows: a condition (``where="year > 2000"``), a ``span``, or entity levels."""
        if where is not None and not isinstance(where, str):
            raise TypeError("select() chooses periods or rows (where= a condition, span=); to keep some series, use "
                            f"data.keep({', '.join(map(repr, list(where)[:3]))}) or data.drop(...)")
        mask = self.sample_mask(span, where, **levels)
        flags = self.flags[mask] if self.flags is not None else None
        return Dataset(self.frame[mask], self.axes, self._static(), flags, self.meta)

    # aggregation

    def _how(self, name: str, how: str | Mapping[str, str] | None, rules: Mapping[Measure, str]) -> str:
        if isinstance(how, str):
            return how
        if isinstance(how, Mapping) and name in how:
            return how[name]
        measure = self.variables[name].measure
        if measure is None:
            raise DataError(
                f"series {name!r} has no measure; set Variable.measure or pass how="
            )
        return rules[measure]

    @api("pymetrix:dataset.to_freq")
    def to_freq(self, freq: str, how: str | Mapping[str, str] | None = None) -> Dataset:
        """Convert to a lower frequency (``"Q"``, ``"Y"``, ``"Y-JUN"``...). Periods with missing sub-periods become NaN.

        ``how`` is sum, mean, first, last (or another pandas aggregation), for every series or as a mapping by
        series; series it does not cover follow their ``Variable.measure`` (flows and counts summed, stocks the last
        value, rates, indices and ratios averaged). To a higher frequency, use ``TemporalDisaggregation``."""
        t = self.time_axis
        if t is None:
            raise DataError("to_freq needs a time axis")
        old = self.frame.index.get_level_values(t.name)
        new = old.asfreq(freq)
        keys = [self.frame.index.get_level_values(e.name) for e in self.entity_axes]
        keys.append(pd.PeriodIndex(new, name=t.name))
        numeric = self.frame.select_dtypes("number")
        grouped = numeric.groupby(keys)
        out = grouped.agg({c: self._how(c, how, _OVER_TIME) for c in numeric.columns})
        counts = grouped.count()
        target = out.index.get_level_values(t.name)
        expected = np.array(
            [len(pd.period_range(p.start_time, p.end_time, freq=old.freq)) for p in target]
        )
        out = out.where(counts.to_numpy() >= expected[:, None])
        axes = tuple(TimeAxis(t.name, out.index.get_level_values(t.name).freqstr, t.calendar) if a is t else a for a in self.axes)
        return Dataset(out, axes, self._static(), meta=self.meta)

    @api("pymetrix:dataset.aggregate")
    def aggregate(self, by: str | Sequence[str], how: str | Mapping[str, str] | None = None) -> Dataset:
        """Aggregate over everything except ``by`` (axis names or columns): ``aggregate("year")`` adds up a panel's
        entities period by period. ``how`` as for ``to_freq``; by measure, stocks are summed across entities."""
        by = [by] if isinstance(by, str) else list(by)
        keys = [self.values_of(b).rename(b) for b in by]
        values = self.frame.drop(columns=[b for b in by if b in self.frame.columns])
        values = values.select_dtypes("number")
        out = values.groupby(keys).agg({c: self._how(c, how, _ACROSS) for c in values.columns})
        out.index.names = by
        t = self.time_axis
        axes = tuple(t if t is not None and b == t.name else EntityAxis(b) for b in by)
        return Dataset(out, axes, self._static(), meta=self.meta)

    @api("pymetrix:dataset.pivot")
    def pivot(self, rows: str | Sequence[str], columns: str | Sequence[str] | None = None,
              values: str | Sequence[str] | None = None, agg: str | Sequence[str] | Mapping[str, str] = "mean",
              margins: bool = False, fill: float | None = None, span: Span | None = None, where: str | None = None) -> Table:
        """A pivot table (pandas ``pivot_table``): ``values`` summarised by ``rows`` and ``columns``.

        Rows and columns are axes (``country``, ``time``) or series of categories (``region``);
        ``agg`` is one or more of mean, sum, count, min, max, median, std, first, last, or a
        mapping from series to aggregation. ``margins`` adds totals.
        """
        rows_ = [rows] if isinstance(rows, str) else list(rows)
        cols_ = [] if columns is None else [columns] if isinstance(columns, str) else list(columns)
        keys = rows_ + cols_
        if values is None:
            values_ = [str(n) for n in self.frame.select_dtypes("number").columns if n not in keys]
        else:
            values_ = [values] if isinstance(values, str) else list(values)
        known = set(self.names) | {str(n) for n in self.frame.index.names}
        unknown = [k for k in keys + values_ if k not in known]
        if unknown:
            raise NotFoundError(f"no axis or series {', '.join(map(repr, unknown))}; have {', '.join(sorted(known))}")
        df = self.frame[self.sample_mask(span, where)].reset_index()
        for k in keys:  # periods as text, so they label rows and columns plainly
            if isinstance(df[k].dtype, pd.PeriodDtype):
                df[k] = df[k].astype(str)
        aggfunc = dict(agg) if isinstance(agg, Mapping) else agg if isinstance(agg, str) else list(agg)
        table = df.pivot_table(index=rows_, columns=cols_ or None, values=values_, aggfunc=aggfunc, margins=margins,
                               margins_name="Total", fill_value=fill, observed=True)
        if isinstance(table, pd.Series):
            table = table.to_frame()
        if isinstance(table.columns, pd.MultiIndex):  # drop levels that do not vary, join the rest
            keep = [i for i in range(table.columns.nlevels) if table.columns.get_level_values(i).nunique() > 1]
            keep = keep or [table.columns.nlevels - 1]
            table.columns = [" | ".join(str(c[i]) for i in keep) for c in table.columns]
        else:
            table.columns = [str(c) for c in table.columns]
        name = ", ".join(rows_)
        table.index = [" | ".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in table.index]
        table.index.name = name
        what = ", ".join(values_) if len(values_) <= 4 else f"{len(values_)} series"
        how = agg if isinstance(agg, str) else ", ".join(agg.values() if isinstance(agg, Mapping) else agg)
        by = f" by {name}" + (f" and {', '.join(cols_)}" if cols_ else "")
        return Table.from_frame(table, title=f"{what}{by} ({how})")

    # panels: reshaping, slices, coverage

    def _entity(self, by: str | None) -> str:
        names = [a.name for a in self.entity_axes]
        if by is not None:
            if by not in names:
                raise NotFoundError(f"{by!r} is not an entity axis; entity axes: {names}")
            return by
        if len(names) != 1:
            raise DataError(f"give the entity axis (by=): the dataset has {len(names)} ({names})" if names
                             else "the dataset has no entity axis")
        return names[0]

    @api("pymetrix:dataset.wide")
    def wide(self, *names: str, by: str | None = None, sep: str = "_") -> Dataset:
        """One column per series and entity (``y_AUS``, ``y_NZL``), dropping the entity axis: a panel
        becomes a set of time series. The inverse of ``long``."""
        names = _flat(names)
        by = self._entity(by)
        names_ = list(names) or [str(c) for c in self.frame.select_dtypes("number").columns]
        missing = [n for n in names_ if n not in self.frame.columns]
        if missing:
            raise NotFoundError(f"no series {', '.join(missing)}")
        frame = self.frame[names_].unstack(by)
        pairs = list(frame.columns)
        new_names = [f"{n}{sep}{_ident(e)}" for n, e in pairs]
        if len(set(new_names)) != len(new_names):
            raise DataError("entity names collide once made into series names; rename the entities")
        static = self._static()
        variables = {new: replace(static[n], name=new, label=f"{static[n].label or n} ({e})") for new, (n, e) in zip(new_names, pairs)}
        frame.columns = new_names
        axes = tuple(a for a in self.axes if a.name != by)
        return Dataset(frame.sort_index(), axes, variables, meta=self.meta)

    @api("pymetrix:dataset.long")
    def long(self, entity: str = "entity", stubs: Sequence[str] | None = None, name: str | None = None,
             sep: str = "_") -> Dataset:
        """From one column per series and entity (``y_AUS``, ``c_AUS``, ``y_NZL``...) to a panel with
        an ``entity`` axis; the inverse of ``wide``. ``stubs`` names the series (default: what comes
        before the last ``sep``, when two or more columns share it); ``name`` instead treats every
        column as one entity of a single series (a sheet with a column per country). Columns that
        match no stub are kept, the same for every entity (e.g. a world price)."""
        if entity in self.frame.index.names:
            raise DataError(f"the dataset already has an axis {entity!r}")
        numeric = [str(c) for c in self.frame.select_dtypes("number").columns]
        groups: dict[str, dict[str, str]]
        common: list[str] = []
        if name is not None:
            groups = {name: {c: c for c in numeric}}
        else:
            if stubs is None:
                counts: dict[str, int] = {}
                for c in numeric:
                    if sep in c:
                        prefix = c.rsplit(sep, 1)[0]
                        counts[prefix] = counts.get(prefix, 0) + 1
                stubs = [p for p, k in counts.items() if k >= 2]
                if not stubs:
                    raise DataError(f"no column names share a prefix before {sep!r}; give stubs=[...] or name=")
            ordered = sorted(stubs, key=len, reverse=True)  # the longest stub that fits
            groups = {s: {} for s in stubs}
            for c in numeric:
                stub = next((s for s in ordered if c.startswith(s + sep) and len(c) > len(s) + len(sep)), None)
                if stub is None:
                    common.append(c)
                else:
                    groups[stub][c] = c[len(stub) + len(sep):]
        index_names = [str(n) for n in self.frame.index.names]
        flat = self.frame.reset_index()
        parts = []
        for stub, cols in groups.items():
            if not cols:
                continue
            melted = flat.melt(id_vars=index_names, value_vars=list(cols), var_name=entity, value_name=stub)
            melted[entity] = melted[entity].map(cols)
            parts.append(melted.set_index([entity, *index_names])[stub])
        if not parts:
            raise DataError("no columns to reshape")
        frame = pd.concat(parts, axis=1)
        for c in common:
            frame[c] = self.frame[c].reindex(frame.index.droplevel(entity)).to_numpy()
        static = self._static()
        variables = {stub: replace(static[next(iter(cols))], name=stub, label=None) for stub, cols in groups.items() if cols}
        variables.update({c: static[c] for c in common})
        return Dataset(frame.sort_index(), (EntityAxis(entity), *self.axes), variables, meta=self.meta)

    def xs(self, **levels: Any) -> Dataset:
        """One value of some axes, dropping them: ``xs(country="FJI")`` is Fiji's time series,
        ``xs(time="2020Q1")`` a cross-section."""
        if not levels:
            raise DataError("give axis=value, e.g. xs(country='FJI')")
        if len(levels) >= len(self.axes):
            raise DataError("at least one axis must remain")
        frame, flags = self.frame, self.flags
        for level, value in levels.items():
            axis = next((a for a in self.axes if a.name == level), None)
            if axis is None:
                raise NotFoundError(f"no axis {level!r}; axes: {[a.name for a in self.axes]}")
            if isinstance(axis, TimeAxis):
                value = pd.Period(str(value), freq=frame.index.get_level_values(level).freq)
            if value not in frame.index.get_level_values(level):
                raise NotFoundError(f"no {level} {value!s} in the dataset")
            frame = frame.xs(value, level=level, drop_level=True)
            flags = flags.xs(value, level=level, drop_level=True) if flags is not None else None
        axes = tuple(a for a in self.axes if a.name not in levels)
        return Dataset(frame.sort_index(), axes, self._static(), flags, self.meta)

    @api("pymetrix:dataset.entity")
    def entity(self, value: Any, by: str | None = None) -> Dataset:
        """One entity's data, without the entity axis (for a panel: its time series)."""
        return self.xs(**{self._entity(by): value})

    @api("pymetrix:dataset.cross_section")
    def cross_section(self, period: Any) -> Dataset:
        """All entities in one period, without the time axis."""
        t = self.time_axis
        if t is None:
            raise DataError("the dataset has no time axis")
        return self.xs(**{t.name: period})

    def _observed(self, names: Sequence[str]) -> pd.Series:
        names_ = list(names) or [str(c) for c in self.frame.select_dtypes("number").columns]
        missing = [n for n in names_ if n not in self.frame.columns]
        if missing:
            raise NotFoundError(f"no series {', '.join(missing)}")
        return self.frame[names_].notna().all(axis=1)

    @api("pymetrix:dataset.as_of")
    def as_of(self, when: Any, lags: Mapping[str, int] | None = None) -> Dataset:
        """The data as known on ``when``: values not yet published (period end plus each series' ``release_lag``
        days) removed. See ``pymetrix.nowcast``."""
        from ..nowcast import as_of

        return as_of(self, when, lags)

    @api("pymetrix:dataset.ragged_edge")
    def ragged_edge(self, when: Any = None, lags: Mapping[str, int] | None = None) -> Table:
        """Each series' last observation, publication lag and publication day. See ``pymetrix.nowcast``."""
        from ..nowcast import ragged_edge

        return ragged_edge(self, when, lags)

    @api("pymetrix:dataset.fill_edge")
    def fill_edge(self, lags: int = 2, through: Any = None, names: Sequence[str] | None = None) -> Dataset:
        """Missing values at the end of each series (or of ``names``) forecast by an AR(``lags``) of the series, up to
        ``through`` (default: the last period of any series). See ``pymetrix.nowcast.fill_edge``."""
        from ..nowcast import fill_edge

        return fill_edge(self, lags, through, names)

    @api("pymetrix:dataset.realign")
    def realign(self) -> Dataset:
        """Vertical realignment: each series shifted on so that its last value falls in the latest period any series
        has. See ``pymetrix.nowcast.realign``."""
        from ..nowcast import realign

        return realign(self)

    @api("pymetrix:dataset.coverage")
    def coverage(self, *names: str) -> Table:
        """For each entity: the first and last periods with data (on every series named; default all),
        the observations, and the gaps between them."""
        names = _flat(names)
        t = self.time_axis
        if t is None:
            raise DataError("coverage needs a time axis")
        observed = self._observed(names)
        others = [a.name for a in self.axes if a is not t]
        frame = pd.DataFrame({"period": self.frame.index.get_level_values(t.name), "observed": observed.to_numpy()})
        keys = [self.frame.index.get_level_values(o) for o in others] if others else np.zeros(len(frame))
        rows, index = [], []
        for key, g in frame.groupby(keys, sort=True):
            seen = g.loc[g["observed"], "period"]
            if seen.empty:
                rows.append(["", "", 0, 0])
            else:
                first, last = seen.min(), seen.max()
                expected = len(pd.period_range(first, last, freq=first.freq))
                rows.append([str(first), str(last), int(len(seen)), int(expected - len(seen))])
            index.append(" | ".join(map(str, key)) if isinstance(key, tuple) and len(key) > 1
                         else str(key[0] if isinstance(key, tuple) else key) if others else "all")
        balanced = len({r[0] for r in rows}) == 1 and len({r[1] for r in rows}) == 1 and all(r[3] == 0 for r in rows) and rows[0][0]
        unit = ", ".join(others) or "series"
        notes = [f"Balanced: every {unit} from {rows[0][0]} to {rows[0][1]}, without gaps" if balanced
                 else "Not balanced: first or last periods differ, or there are gaps"]
        what = ", ".join(names) if names else "all series"
        return Table(["first", "last", "observations", "gaps"], rows, index=index, title=f"Coverage of {what}",
                     notes=notes, index_name=", ".join(others))

    @api("pymetrix:dataset.balance")
    def balance(self, *names: str, keep: Literal["periods", "entities"] = "periods", by: str | None = None) -> Dataset:
        """A balanced panel: keep the ``periods`` in which every entity has data on the series named
        (default all), or the ``entities`` that have data in every period."""
        names = _flat(names)
        t = self.time_axis
        if t is None:
            raise DataError("balance needs a time axis")
        by = self._entity(by)
        observed = self._observed(names)
        if keep not in ("periods", "entities"):
            raise DataError("keep must be 'periods' or 'entities'")
        level = t.name if keep == "periods" else by
        observed = self.complete()._observed(names)  # an entity-period with no row is missing too
        per = observed.groupby(level=level).all()
        mask = np.asarray(self.frame.index.get_level_values(level).isin(per[per].index))
        if not mask.any():
            other = "entities" if keep == "periods" else "periods"
            raise DataError(f"no {keep} with complete data; try keep={other!r}")
        flags = self.flags[mask] if self.flags is not None else None
        return Dataset(self.frame[mask], self.axes, self._static(), flags, self.meta)

    def complete(self) -> Dataset:
        """Add the missing rows (as missing values) so that every entity has every period from the
        first to the last."""
        levels = []
        for a in self.axes:
            values = self.frame.index.get_level_values(a.name)
            if isinstance(a, TimeAxis):
                levels.append(pd.period_range(values.min(), values.max(), freq=values.freq, name=a.name))
            else:
                levels.append(pd.Index(values.unique(), name=a.name).sort_values())
        full = pd.MultiIndex.from_product(levels) if len(levels) > 1 else levels[0]
        flags = self.flags.reindex(full) if self.flags is not None else None
        return Dataset(self.frame.reindex(full), self.axes, self.variables, flags, self.meta)

    @api("pymetrix:dataset.describe_panel")
    def describe_panel(self, *names: str, by: str | None = None) -> Table:
        """Overall, between-entity and within-entity variation of each series: mean, standard
        deviation, minimum and maximum, with observations (N), entities (n) and average periods (T)."""
        names = _flat(names)
        by = self._entity(by)
        names_ = list(names) or [str(c) for c in self.frame.select_dtypes("number").columns]
        rows, index = [], []
        frame = self.in_sample()
        for name in names_:
            x = frame[name].astype(float).dropna()
            groups = x.groupby(level=by)
            means = groups.mean()
            within = x - groups.transform("mean") + x.mean()
            rows += [[x.mean(), x.std(), x.min(), x.max(), f"N = {len(x)}"],
                     [np.nan, means.std(), means.min(), means.max(), f"n = {len(means)}"],
                     [np.nan, within.std(), within.min(), within.max(), f"T = {len(x) / max(len(means), 1):.1f}"]]
            index += [f"{name} overall", f"{name} between", f"{name} within"]
        notes = [f"Between: the {by} means; within: deviations from them, plus the overall mean.", *self._sample_note()]
        return Table(["mean", "sd", "min", "max", "observations"], rows, index=index,
                     title=f"Panel statistics by {by}", notes=notes)

    # statistics and display

    @api("pymetrix:dataset.validate")
    def validate(self, strict: bool = False, outlier: float = 6.0) -> Any:
        """Check the data before using them: duplicate rows, missing periods, infinite values, text among numbers
        (errors); missing values inside a series, constant series, outliers (warnings); missing units or vintages and
        ragged edges (notes). A ``Validation``; ``strict`` raises ``DataError`` if there are errors."""
        from .validate import validate

        return validate(self, strict, outlier)

    def in_sample(self) -> pd.DataFrame:
        """The rows of the active sample (all rows without one): what tables and graphs show by default."""
        return self.frame if self.meta.sample is None else self.frame[self.sample_mask()]

    def _sample_note(self) -> list[str]:
        return [f"Sample: {self.sample} ({self.sample.describe(self)})"] if self.meta.sample is not None else []

    def _numeric(self, names: str | Sequence[str] | None) -> pd.DataFrame:
        """The named series (default every numeric one) over the active sample, as floats."""
        frame = self.in_sample()
        if isinstance(names, str):
            names = [names]
        cols = list(names) if names else [str(c) for c in frame.select_dtypes("number").columns]
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise DataError(f"no series named {', '.join(map(repr, missing))}")
        return frame[cols].astype(float)

    @api("pymetrix:dataset.describe")
    def describe(self, names: str | Sequence[str] | None = None, percentiles: Sequence[float] = (0.25, 0.5, 0.75)) -> Table:
        """Count, mean, standard deviation, minimum, ``percentiles`` (default the quartiles; the median is always
        shown), maximum, skewness and excess kurtosis of each series (default every numeric one), over the active
        sample. Skewness and kurtosis are the bias-corrected sample statistics (pandas' ``skew`` and ``kurt``,
        scipy's ``skew`` and ``kurtosis`` with ``bias=False``); kurtosis is in excess of the normal's 3."""
        frame = self._numeric(names)
        shares = sorted({float(q) for q in percentiles} | {0.5})
        if any(not 0 <= q <= 1 for q in shares):
            raise DataError("percentiles are shares between 0 and 1, e.g. (0.05, 0.5, 0.95)")
        desc = frame.describe(percentiles=shares).T
        desc["skewness"] = frame.skew()
        desc["kurtosis"] = frame.kurt()
        notes = [*self._sample_note(),
                 "Skewness and kurtosis are bias-corrected sample statistics; kurtosis is excess kurtosis (0 for a normal)"]
        return Table.from_frame(desc, title="Descriptive statistics", notes=notes)

    @api("pymetrix:dataset.corr")
    def corr(self, names: Sequence[str] | None = None, method: Literal["pearson", "spearman", "kendall"] = "pearson") -> Table:
        """Correlations between series (default every numeric one) over the active sample, each pair over the
        periods both have: ``pearson`` (linear), ``spearman`` (of ranks) or ``kendall`` (tau-b, of concordant pairs)."""
        if method not in ("pearson", "spearman", "kendall"):
            raise DataError("method is 'pearson', 'spearman' or 'kendall'")
        title = {"pearson": "Correlations", "spearman": "Spearman rank correlations", "kendall": "Kendall rank correlations (tau-b)"}[method]
        return Table.from_frame(self._numeric(names).corr(method=method), title=title, notes=self._sample_note())

    @api("pymetrix:dataset.cov")
    def cov(self, names: Sequence[str] | None = None) -> Table:
        """Covariances between series (default every numeric one) over the active sample, each pair over the periods
        both have, with the n - 1 denominator."""
        notes = ["Each covariance uses the periods both series have, divided by their number less one", *self._sample_note()]
        return Table.from_frame(self._numeric(names).cov(), title="Covariances", notes=notes)

    @api("pymetrix:dataset.partial_corr")
    def partial_corr(self, names: Sequence[str], given: str | Sequence[str] = ()) -> Table:
        """Partial correlations between ``names`` holding the series ``given`` constant: the correlations of their
        residuals from least-squares regressions on a constant and ``given``, over the periods all of them have."""
        given = [given] if isinstance(given, str) else list(given)
        names = [names] if isinstance(names, str) else list(names)
        if len(names) < 2:
            raise DataError("partial correlations need two or more series")
        if set(names) & set(given):
            raise DataError("a series cannot be both correlated and held constant")
        frame = self._numeric(names + given).dropna()
        if len(frame) <= len(given) + 2:
            raise DataError(f"{len(frame)} complete periods are too few to hold {len(given)} series constant")
        z = np.column_stack([np.ones(len(frame)), frame[given].to_numpy()]) if given else np.ones((len(frame), 1))
        y = frame[names].to_numpy()
        resid = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
        out = pd.DataFrame(np.corrcoef(resid, rowvar=False), index=names, columns=names)
        held = ", ".join(given) if given else "nothing but a constant"
        notes = [f"Holding constant {held}; {len(frame)} periods with every series", *self._sample_note()]
        return Table.from_frame(out, title="Partial correlations", notes=notes)

    def _deviations(self, names: str | Sequence[str] | None, standardized: bool, least: int) -> tuple[list[str], np.ndarray, int]:
        """The series' deviations from their means over the periods every one has (in the active sample), divided by
        their standard deviations (with the T denominator) when ``standardized``."""
        frame = self._numeric(names).dropna()
        cols = [str(c) for c in frame.columns]
        if not cols:
            raise DataError("no numeric series to compute co-moments of")
        n = len(frame)
        if n < least:
            raise DataError(f"{n} periods with every series are too few for co-moments (at least {least})")
        d = frame.to_numpy() - frame.to_numpy().mean(axis=0)
        if standardized:
            sd = np.sqrt((d ** 2).mean(axis=0))
            if np.any(sd == 0):
                raise DataError(f"{', '.join(c for c, s in zip(cols, sd) if s == 0)} does not vary: it has no standardized co-moments")
            d = d / sd
        return cols, d, n

    def _comoment(self, order: int, names: str | Sequence[str] | None, standardized: bool, form: str) -> Table:
        if form not in ("matrix", "pairs"):
            raise DataError("form is 'matrix' (every co-moment, one row a series) or 'pairs' (one number for each pair)")
        cols, d, n = self._deviations(names, standardized, 3)
        what = {3: "skewness", 4: "kurtosis"}[order]
        k = len(cols)
        if form == "pairs":  # E[e_i^(order-1) e_j]: row i, column j
            out = pd.DataFrame((d ** (order - 1)).T @ d / n, index=cols, columns=cols)
            first = "squared" if order == 3 else "cubed"
            notes = [f"Row i, column j: the mean of ({'standardized ' if standardized else ''}deviation of i) {first} times "
                     f"(deviation of j); the diagonal is each series' {what}"]
        else:  # M3 = E[e (e kron e)'], n x n^2; M4 = E[e (e kron e kron e)'], n x n^3
            kron = d
            for _ in range(order - 2):
                kron = (kron[:, :, None] * d[:, None, :]).reshape(n, -1)
            labels = [",".join(c) for c in itertools.product(cols, repeat=order - 1)]
            out = pd.DataFrame(d.T @ kron / n, index=cols, columns=labels)
            notes = [f"Row i, column {'j,k' if order == 3 else 'j,k,l'}: the mean of the product of the "
                     f"{'standardized ' if standardized else ''}deviations of i, {'j and k' if order == 3 else 'j, k and l'} "
                     f"(the {k} x {k ** (order - 1)} matrix M{order}); unchanged by any order of the indices"]
        diagonal = ("the sample skewness with the T denominator (scipy's skew, bias=True)" if order == 3 else
                    "the sample kurtosis with the T denominator, not in excess (3 for a normal; scipy's kurtosis, "
                    "fisher=False, bias=True)")
        notes += [f"Moments divide by T, the {n} periods with every series" + (", and deviations by standard deviations "
                  f"with T: the diagonal is {diagonal}; describe() shows bias-corrected {'skewness' if order == 3 else 'excess kurtosis'}"
                  if standardized else ": not standardized (co-moments in the series' units)"), *self._sample_note()]
        title = ("Co-skewness" if order == 3 else "Co-kurtosis") + ("" if standardized else " (unstandardized co-moments)")
        return Table.from_frame(out, title=title, notes=notes)

    @api("pymetrix:dataset.coskewness")
    def coskewness(self, names: str | Sequence[str] | None = None, standardized: bool = True,
                   form: Literal["matrix", "pairs"] = "matrix") -> Table:
        """Co-skewness of series (default every numeric one) over the periods every one has in the active sample.

        With ``e`` the deviations from the means divided by the standard deviations (both with the T denominator),
        the co-skewness of i, j and k is S_ijk = mean(e_i e_j e_k). ``form="matrix"`` gives every S_ijk as the
        n x n² matrix M3 = E[e (e ⊗ e)'] (row i, column "j,k"), as in portfolio choice with higher moments;
        ``form="pairs"`` the n x n table of S_iij = mean(e_i² e_j) (row i, column j), the co-skewness usually reported
        for two assets. The diagonal is each series' skewness with the T denominator (scipy's ``skew``); ``describe``
        shows the bias-corrected skewness, G1 = g1 √(T(T-1))/(T-2). ``standardized=False`` gives the co-moments of
        the deviations themselves, mean((x_i - x̄_i)(x_j - x̄_j)(x_k - x̄_k))."""
        return self._comoment(3, names, standardized, form)

    @api("pymetrix:dataset.cokurtosis")
    def cokurtosis(self, names: str | Sequence[str] | None = None, standardized: bool = True,
                   form: Literal["matrix", "pairs"] = "matrix") -> Table:
        """Co-kurtosis of series (default every numeric one) over the periods every one has in the active sample.

        With ``e`` the standardized deviations (T denominator), the co-kurtosis of i, j, k and l is
        K_ijkl = mean(e_i e_j e_k e_l). ``form="matrix"`` gives the n x n³ matrix M4 = E[e (e ⊗ e ⊗ e)'] (row i,
        column "j,k,l"); ``form="pairs"`` the n x n table of K_iiij = mean(e_i³ e_j) (row i, column j). The diagonal is
        each series' kurtosis with the T denominator and not in excess (3 for a normal; scipy's ``kurtosis`` with
        ``fisher=False``); ``describe`` shows bias-corrected excess kurtosis, G2 = ((T+1)(b2 - 3) + 6)(T-1)/((T-2)(T-3)) for b2 this kurtosis.
        ``standardized=False`` gives the fourth co-moments of the deviations themselves."""
        return self._comoment(4, names, standardized, form)

    @api("pymetrix:dataset.systematic_comoments")
    def systematic_comoments(self, names: str | Sequence[str], market: str) -> Table:
        """Each series' beta, co-skewness and co-kurtosis with a market series, over the periods all have in the
        active sample (moments with the T denominator; d the deviations from the means):

        - beta = mean(d_i d_m) / mean(d_m²);
        - co-skewness = mean(d_i d_m²) / (σ_i σ_m²) and co-kurtosis = mean(d_i d_m³) / (σ_i σ_m³), standardized;
        - co-skewness beta = mean(d_i d_m²) / mean(d_m³) and co-kurtosis beta = mean(d_i d_m³) / mean(d_m⁴), the
          series' share of the market's skewness and kurtosis (Kraus and Litzenberger 1976; Fang and Lai 1997).

        A negative co-skewness means the series does badly when the market makes large moves; investors averse to
        that demand a higher return (Harvey and Siddique 2000)."""
        names = [names] if isinstance(names, str) else list(names)
        if not names:
            raise DataError("choose the series to relate to the market")
        if market in names:
            raise DataError(f"{market} is the market series: relate other series to it")
        cols, d, n = self._deviations([*names, market], False, 3)
        m = d[:, -1]
        s2, m3, m4 = (m ** 2).mean(), (m ** 3).mean(), (m ** 4).mean()
        if s2 == 0:
            raise DataError(f"the market series {market} does not vary")
        rows = []
        for j, name in enumerate(names):
            x = d[:, j]
            sx = np.sqrt((x ** 2).mean())
            cs, ck = (x * m ** 2).mean(), (x * m ** 3).mean()
            rows.append([(x * m).mean() / s2, cs / (sx * s2) if sx > 0 else np.nan, ck / (sx * s2 ** 1.5) if sx > 0 else np.nan,
                         cs / m3 if m3 != 0 else np.nan, ck / m4])
        out = pd.DataFrame(rows, index=names, columns=["beta", "coskewness", "cokurtosis", "coskewness beta", "cokurtosis beta"])
        notes = [f"Market: {market}; {n} periods with every series; moments divide by T",
                 "coskewness = mean(d_i d_m²)/(σ_i σ_m²); cokurtosis = mean(d_i d_m³)/(σ_i σ_m³); "
                 "coskewness beta = mean(d_i d_m²)/mean(d_m³); cokurtosis beta = mean(d_i d_m³)/mean(d_m⁴)", *self._sample_note()]
        return Table.from_frame(out, title=f"Systematic co-moments with {market}", notes=notes)

    def _two_series(self, x: str, y: str, what: str) -> tuple[np.ndarray, np.ndarray]:
        if self.kind != "time_series":
            raise DataError(f"a {what} needs a single time series; select one entity first")
        both = self._numeric([x, y] if x != y else [x]).dropna()
        a = both[x].to_numpy()
        return a, both[y].to_numpy() if x != y else a

    @api("pymetrix:dataset.crosscorrelogram")
    def crosscorrelogram(self, x: str, y: str, lags: int = 12) -> Table:
        """Cross-correlations corr(x(t), y(t+k)) for k = -lags ... lags, over the periods both series have: a large
        value at k > 0 means x moves ahead of y. Each uses the full-sample means and standard deviations, with the
        sum divided by T (as the autocorrelations of a correlogram are), so the table is a proper correlation
        function; the 95% band for no correlation is ±1.96/√T."""
        a, b = self._two_series(x, y, "cross-correlogram")
        n = len(a)
        if not 0 < lags < n:
            raise DataError(f"{lags} lags need more than {lags} observations (there are {n})")
        da, db = a - a.mean(), b - b.mean()
        scale = n * a.std() * b.std()
        rows = []
        for k in range(-lags, lags + 1):
            c = np.dot(da[:n - k], db[k:]) if k >= 0 else np.dot(da[-k:], db[:n + k])
            rows.append([k, float(c / scale)])
        band = 1.96 / np.sqrt(n)
        notes = [f"{n} observations; approximate 95% band for no correlation: ±{band:.3f}",
                 f"Lag k: the correlation of {x} at t with {y} at t+k", *self._sample_note()]
        return Table(["lag", "correlation"], rows, title=f"Cross-correlogram of {x} and {y}", notes=notes)

    @api("pymetrix:dataset.autocov")
    def autocov(self, name: str, lags: int = 20) -> Table:
        """Autocovariances of one series at lags 0 ... ``lags`` (the sum divided by T, as for the autocorrelations of
        a correlogram), with the autocorrelations they give."""
        a, _ = self._two_series(name, name, "autocovariance")
        n = len(a)
        if not 0 < lags < n:
            raise DataError(f"{lags} lags need more than {lags} observations (there are {n})")
        d = a - a.mean()
        gamma = [float(np.dot(d[:n - k], d[k:]) / n) for k in range(lags + 1)]
        rows = [[k, g, g / gamma[0]] for k, g in enumerate(gamma)]
        notes = [f"{n} observations; each autocovariance divided by T", *self._sample_note()]
        return Table(["lag", "autocovariance", "autocorrelation"], rows, title=f"Autocovariances of {name}", notes=notes)

    @api("pymetrix:dataset.filter")
    def filter(self, names: str | Sequence[str], method: Any, derived: bool = True) -> Dataset:
        """Add ``{name}_trend`` and ``{name}_cycle`` from a trend-cycle filter (HP, BaxterKing, ...).

        By default they are derived series, recomputed when ``name`` changes;
        ``derived=False`` stores plain values.
        """
        from ..filters import apply

        return apply(self, list(names) if not isinstance(names, str) else names, method, derived)

    @api("pymetrix:dataset.seasonal_adjust")
    def seasonal_adjust(self, names: str | Sequence[str], method: Any = None, derived: bool = True) -> Dataset:
        """Add ``{name}_sa`` (seasonally adjusted) and ``{name}_seasonal`` (the seasonal factor, or
        component when additive), by ``method``: ``STL()`` (the default), ``X13()``, ``TramoSeats()`` or ``Classical()``.
        By default they are derived series, recomputed when ``name`` changes."""
        from ..seasonal import STL, apply

        return apply(self, list(names) if not isinstance(names, str) else names, method if method is not None else STL(), derived)

    @api("pymetrix:dataset.calendar")
    def calendar(self, trading_days: str = "six", easter: int | None = 8, leap_year: bool = True) -> Dataset:
        """Add calendar regressors for monthly or quarterly data: trading days (``"six"`` contrasts
        td_mon ... td_sat, ``"one"`` weekday contrast td_weekdays, or ``"none"``), ``easter_{w}``
        and ``leap_year``, as defined by X-13ARIMA-SEATS."""
        from ..seasonal import calendar_effects

        t = self.time_axis
        if t is None:
            raise DataError("calendar regressors need a time axis")
        periods = pd.PeriodIndex(self.frame.index.get_level_values(t.name))
        unique = pd.PeriodIndex(periods.unique()).sort_values()
        table = calendar_effects(unique, trading_days, easter, leap_year)  # type: ignore[arg-type]
        frame = self.frame.copy()
        variables = dict(self.variables)
        for col in table.columns:
            frame[col] = table[col].reindex(periods).to_numpy()
            variables[col] = Variable(col, label=_CALENDAR_LABELS.get(col.rstrip("0123456789").rstrip("_"), col),
                                      derived_from="calendar regressor")
        return Dataset(frame, self.axes, variables, self.flags, self.meta)

    @api()
    def correlogram(self, name: str, lags: int = 20) -> Table:
        """Autocorrelations, partial autocorrelations and Ljung-Box Q statistics of one series."""
        from ..stats.timeseries import correlogram_table

        if self.kind != "time_series":
            raise DataError("a correlogram needs a single time series; select one entity first")
        return correlogram_table(self.in_sample()[name].dropna().to_numpy(float), lags, f"Correlogram of {name}")

    @api()
    def select_order(self, name: str, max_ar: int = 4, max_ma: int = 2, exog: Sequence[str] = (), intercept: bool = True, span: Span | None = None) -> Any:
        """Compare ARMA(p, q) models for ``name`` by AIC, BIC and Hannan-Quinn."""
        from ..stats.timeseries import select_order

        return select_order(self, name, max_ar, max_ma, exog, intercept, span)

    @api("pymetrix:dataset.johansen")
    def johansen(self, variables: Sequence[str], lags: int = 1, det_order: Literal[-1, 0, 1] = 0, signif: float = 0.05, span: Span | None = None) -> Any:
        """Johansen cointegration tests (trace and maximum eigenvalue) for ``variables``."""
        from ..vecm import johansen

        return johansen(self, variables, lags, det_order, signif, span)

    @api()
    def test(self, name: str, *args: Any, **opts: Any) -> TestResult:
        from ..stats import run_test

        return run_test(self, name, *args, **opts)

    def tests(self) -> list[str]:
        from ..stats import available_tests

        return available_tests(self)

    @api("pymetrix:dataset.plot")
    def plot(self, names: str | Sequence[str] | None = None, kind: str = "line", layout: str = "one", span: Span | None = None,
             entity: Any = None, title: str = "", stacked: bool = False, right: str | Sequence[str] = (),
             transform: str = "none", shade: Any = (), total: str | None = None, coverages: Sequence[float] | None = None,
             bins: int | None = None, **options: Any) -> Figure:
        """A figure of series.

        ``kind``: line, step, bar (side by side), barh (horizontal bars), area, scatter (x first, then y), histogram,
        box, contributions (the series as stacked parts; ``total`` names the series they add up to, else their sum is
        drawn), fan (each series with its bands from ``{name}_lo90``/``{name}_hi90`` or ``{name}_se`` columns, at
        ``coverages`` such as (0.3, 0.6, 0.9)), vintages (each series a release of the same data, oldest first),
        correlation (a heatmap of the series' correlations), ecdf (each series' empirical distribution, in one panel),
        hist2d (a two-dimensional histogram of y against x, x first), violin (each series' density, as box plots),
        hexbin (hexagonal bins of y against x, x first) or surface (the series side by side along x, the dates along y,
        the values as heights, in 3-D: a yield curve over time). ``bins``: for histogram and hist2d, or the hexagons
        across for hexbin. ``layout``: one panel, or each series in its own. ``stacked`` stacks bars or areas (negative values below
        zero, as contributions do). ``right`` names series to draw against a right-hand axis. ``transform``: none,
        index (first period shown = 100), change, growth (% on the previous period), yoy (% on a year earlier) or log.
        ``shade`` shades periods: ``"2008Q3:2009Q2, 2020Q1:2020Q2"`` (a label after ``=``), or the name of a 0/1 series
        such as recession dates. Other keywords are the figure's options (see ``Figure.with_options``): ``events``
        (``"2016Q2=Brexit vote"``), ``reference`` (``"2=inflation target"``), ``log``, ``ylim``, ``sharey``,
        ``bar_labels``, ``subtitle``, ``footnote``, ``theme`` and others. For a panel dataset, choose an ``entity``, or plot one series with a line per entity; a cross-section's
        entities label its bars. Forecasts (with intervals) are drawn as such."""
        from ..figures import forecast_figure, series_figure

        if names is None and kind == "line" and ("forecast" in self.frame.columns or any(f"{c}_lower" in self.frame.columns for c in self.names)):
            figure = forecast_figure(self)
        else:
            names = [names] if isinstance(names, str) else names
            figure = series_figure(self, names, kind, layout, span, entity, title, stacked, right, transform, shade,  # type: ignore[arg-type]
                                   total, coverages, bins)
        return figure.with_options(**options) if options else figure

    def __sub__(self, other: Dataset) -> Dataset:
        common = [c for c in self.frame.columns if c in other.frame.columns]
        frame = self.frame[common] - other.frame[common]
        variables = {
            c: self.variables[c] if self.variables[c] == other.variables.get(c) else Variable(c, derived_from="difference")
            for c in common
        }
        return Dataset(frame, self.axes, variables, meta=self.meta)
