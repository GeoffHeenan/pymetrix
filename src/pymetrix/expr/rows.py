"""Functions of a dataset's rows: the calendar of the time axis and seeded random draws.

``year()``, ``quarter()``, ``month()`` and ``day()`` are the year, quarter, month and day of each period's first day
(the period's own year and quarter for calendar years and quarters; ``month()`` of a quarter is its first month, 1, 4,
7 or 10); ``obsnum()`` numbers the observations from 1, within each entity of a panel, in time order (the row number
of data with no time axis). ``normrnd(mu = 0, sigma = 1, seed=s)`` and ``unifrnd(a = 0, b = 1, seed=s)`` make one
draw a row, in the dataset's row order, from ``pymetrix._random.generator(s)``: the bit generator
``set_random_generator`` chose, seeded, so the same formula gives the same draws each time it is computed and when a
session is replayed (each step is replayed with the generator it ran with). They are
``mu + sigma * standard_normal(n)`` and ``a + (b - a) * random(n)`` (NumPy's ``Generator.normal`` and ``uniform``
compute the same numbers).

They need the dataset (``data.eval``, Generate series, regressors written in an equation): a model has no rows, so
models refuse them, as they refuse expanding windows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .nodes import RANDOM, Call, ExpressionError

__all__ = ["calendar", "dataset_of", "draws", "round_half_away"]


def dataset_of(resolve: Any) -> Any:
    """The dataset a resolver reads from (``Dataset.lagged``, bound), or None."""
    owner = getattr(resolve, "__self__", None)
    return owner if hasattr(owner, "frame") and hasattr(owner, "time_axis") else None


def _need_data(fn: str) -> ExpressionError:
    what = "draws one number for each row" if fn in RANDOM else "comes from the data's time axis"
    return ExpressionError(f"{fn}() {what}: it works on a dataset (data.eval, Generate series, regressors), not inside "
                           "estimation or model solvers; make it a series first")


def calendar(fn: str, data: Any) -> pd.Series:
    """``year()``, ``quarter()``, ``month()``, ``day()`` or ``obsnum()`` for each row of ``data``."""
    if data is None:
        raise _need_data(fn)
    frame = data.frame
    index = frame.index
    t = data.time_axis
    if fn == "obsnum":
        if t is None or t.name not in (index.names or []):
            return pd.Series(np.arange(1, len(index) + 1, dtype=float), index=index)
        keys = [n for n in index.names if n != t.name] if isinstance(index, pd.MultiIndex) else []
        order = pd.DataFrame({"time": np.asarray(index.get_level_values(t.name))})
        for k in keys:
            order[k] = np.asarray(index.get_level_values(k))
        ranks = order.groupby(keys, sort=False)["time"].rank(method="first") if keys else order["time"].rank(method="first")
        return pd.Series(ranks.to_numpy(float), index=index)
    if t is None or t.name not in (index.names or []):
        raise ExpressionError(f"{fn}() needs a time axis, and this dataset has none")
    periods = index.get_level_values(t.name)
    starts = pd.DatetimeIndex(periods.start_time if isinstance(periods, pd.PeriodIndex) else periods)
    values = {"year": starts.year, "quarter": starts.quarter, "month": starts.month, "day": starts.day}[fn]
    return pd.Series(np.asarray(values, dtype=float), index=index)


def draws(node: Call, args: Sequence[Any], data: Any) -> pd.Series:
    """One random draw a row of ``data`` for ``normrnd`` or ``unifrnd``; ``args`` are the two parameters' values
    (numbers or Series), the seed is the node's last argument."""
    from .._random import generator

    if data is None:
        raise _need_data(node.fn)
    seed = int(node.args[-1].value)  # type: ignore[attr-defined]
    index = data.frame.index
    rng = generator(seed)
    first, second = (np.asarray(a.reindex(index) if isinstance(a, pd.Series) else a, dtype=float) for a in args)
    with np.errstate(all="ignore"):
        if node.fn == "normrnd":
            out = first + second * rng.standard_normal(len(index))
            out = np.where(second >= 0, out, np.nan)  # a negative standard deviation: missing, as normpdf gives
        else:
            out = first + (second - first) * rng.random(len(index))
    return pd.Series(np.broadcast_to(out, (len(index),)).astype(float), index=index)


def round_half_away(x: Any, n: Any = 0.0) -> Any:
    """x rounded to n decimals, halves away from zero (EViews's, MATLAB's, Stata's and Excel's rule; NumPy's
    ``round`` takes halves to the even neighbour). A Series stays a Series."""
    scale = np.power(10.0, np.asarray(n, dtype=float))
    values = np.asarray(x, dtype=float)
    with np.errstate(all="ignore"):
        out = np.sign(values) * np.floor(np.abs(values) * scale + 0.5) / scale
    if isinstance(x, pd.Series):
        return pd.Series(out, index=x.index)
    return float(out) if np.ndim(out) == 0 else out

