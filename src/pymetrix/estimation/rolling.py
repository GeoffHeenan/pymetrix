"""Rolling (moving-window) and recursive (expanding-window) estimation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._api import api
from ..errors import SpecificationError
from ..output import Figure, Table

if TYPE_CHECKING:
    from ..data import Dataset
    from .equation import Equation
    from .fitted import FittedEquation


@dataclass(frozen=True, eq=False)
class RollingFit:
    """One fit per window, keyed by the window's last period."""

    equation: Equation
    method: Any
    window: int | None  # None: expanding (recursive) window
    fits: Mapping[pd.Period, FittedEquation]
    failed: Mapping[pd.Period, str]

    def _frame(self, values: Any) -> pd.DataFrame:
        ends = list(self.fits)
        rows = [values(f) for f in self.fits.values()]
        return pd.DataFrame(rows, index=pd.PeriodIndex(ends, name="end"))

    def _dataset(self, frame: pd.DataFrame) -> Dataset:
        from ..data import Dataset

        return Dataset.from_frame(frame)

    @property
    def params(self) -> Dataset:
        return self._dataset(self._frame(lambda f: f.params.as_dict()))

    @property
    def se(self) -> Dataset:
        return self._dataset(self._frame(lambda f: dict(zip(f.params.names, f.params.se))))

    @property
    def stats(self) -> Dataset:
        keep = ("nobs", "r2", "r2_adj", "se_regression", "loglike", "aic", "bic")
        return self._dataset(self._frame(lambda f: {k: f.stats[k] for k in keep if k in f.stats}))

    def plot(self, names: Sequence[str] | None = None) -> Figure:
        """Coefficient paths with bands of two standard errors."""
        from ..figures import coefficient_paths_figure

        return coefficient_paths_figure(self, names)

    @api()
    def forecast(self, data: Dataset, horizon: int = 1, dynamic: bool = False) -> Dataset:
        """Forecast of period end + horizon from each window's estimates (pseudo out-of-sample)."""
        from ..data import Dataset

        rows, index = [], []
        for end, fit in self.fits.items():
            target = end + horizon
            span = (str(end + 1), str(target)) if dynamic else (str(target), str(target))
            try:
                fc = fit.forecast(data, span=span, dynamic=dynamic).frame
            except (ValueError, KeyError):
                continue
            if fc.empty or not np.isfinite(fc["forecast"].iloc[-1]):  # target outside the data
                continue
            row = fc.iloc[-1]
            rows.append({"origin": str(end), "forecast": row["forecast"], "se": row["se"], "actual": row["actual"]})
            index.append(target)
        frame = pd.DataFrame(rows, index=pd.PeriodIndex(index, name=data.time_axis.name))  # type: ignore[union-attr]
        frame["error"] = frame["actual"] - frame["forecast"]
        return Dataset(frame, data.axes)

    @property
    def provenance(self) -> Any:
        """The windows' provenance gathered (as a backtest gathers its re-estimates'): the sample is the windows' last
        periods, with the first and the last window's estimation samples; observations are the periods any window
        used; warnings are each window's, once, with the windows where they arose. None if no window was estimated."""
        from dataclasses import replace

        from ..forecasting import _reestimates

        provs = {str(end): f.provenance for end, f in self.fits.items() if f.provenance is not None}
        if not provs:
            return None
        first, last = next(iter(provs.values())), list(provs.values())[-1]
        where: dict[str, list[str]] = {}
        for end, p in provs.items():
            for text in p.warnings:
                where.setdefault(text, []).append(end)
        used = set().union(*(f.resid.index for f in self.fits.values() if f.resid is not None))
        converged = [p.converged for p in provs.values() if p.converged is not None]
        ends = list(self.fits)
        return replace(last, spec={**dict(first.spec), "re-estimates": _reestimates("windows ending", self.fits, failed=len(self.failed))},
                       method={"type": f"{self._kind().lower()} estimation", "refit": dict(first.method)},
                       sample=(str(ends[0]), str(ends[-1])), nobs=len(used) or last.nobs, dropped=0, rows=None, presample=(), dropped_rows=None,
                       data_fingerprint=last.data_fingerprint if self.window is None else "",  # recursive: the last window holds every row
                       warnings=tuple(f"{t} (window{'s' if len(e) > 1 else ''} ending {', '.join(e)})" for t, e in where.items()),
                       converged=all(converged) if converged else None)

    def _kind(self) -> str:
        return "Recursive" if self.window is None else f"Rolling ({self.window}-period)"

    def summary(self, last: int = 12) -> Table:
        params = self.params.frame.tail(last)
        notes = [f"{len(self.fits)} windows ending {min(self.fits)} to {max(self.fits)}"]
        if self.failed:
            notes.append(f"{len(self.failed)} windows could not be estimated (e.g. {next(iter(self.failed.values()))})")
        return Table.from_frame(params, title=f"{self._kind()} estimates: {self.equation.text} (last {len(params)})", notes=notes)

    def __repr__(self) -> str:
        return f"<RollingFit {self._kind().lower()}, {len(self.fits)} windows: {self.equation.text!r}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


def fit_windows(
    eq: Equation,
    data: Dataset,
    window: int | None,
    min_obs: int | None,
    span: tuple[Any, Any] | None,
    method: Any,
    step: int,
) -> RollingFit:
    t = data.time_axis
    if t is None:
        raise SpecificationError("rolling and recursive estimation need a time axis")
    full = eq.fit(data, span=span, method=method)
    method = full.method
    periods = pd.PeriodIndex(sorted(set(full.resid.index.get_level_values(t.name))))  # type: ignore[union-attr]
    k = len(full.params)
    size = window if window is not None else (min_obs or k + 1)
    if size <= k:
        raise SpecificationError(f"windows need more than {k} periods (the number of parameters)")
    if size > len(periods):
        raise SpecificationError(f"the sample has only {len(periods)} periods")
    fits, failed = {}, {}
    from ..progress import report

    ends = range(size - 1, len(periods), step)
    for k, i in enumerate(ends):
        report(k / max(len(ends), 1), f"window ending {periods[i]} ({k + 1} of {len(ends)})")  # outside the try: Cancel stops
        first = periods[i - size + 1] if window is not None else periods[0]
        end = periods[i]
        try:
            fits[end] = eq.fit(data, span=(str(first), str(end)), method=method)
        except Exception as exc:  # a window that cannot be estimated is recorded, not fatal
            failed[end] = str(exc)
    return RollingFit(eq, method, window, fits, failed)
