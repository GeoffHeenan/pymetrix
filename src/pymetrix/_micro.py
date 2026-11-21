"""What the applied-microeconometrics estimators share (double machine learning, shift-share IV, interactive fixed
effects, conditional logit): reading columns from a dataset, the coefficient table and provenance."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

COLUMNS = ["estimate", "std err", "z", "p-value", "[0.025", "0.975]"]


def interval_columns(alpha: float = 0.05) -> list[str]:
    """``COLUMNS`` with the interval's ends labelled for ``alpha`` (``COLUMNS`` itself at 0.05)."""
    return [*COLUMNS[:4], f"[{alpha / 2:g}", f"{1 - alpha / 2:g}]"]


def names(value: str | Sequence[str] | None) -> list[str]:
    """A list of series names from a list, or from text separated by commas or plus signs."""
    if value is None:
        return []
    if isinstance(value, str):
        return [s.strip() for s in value.replace("+", ",").split(",") if s.strip()]
    return [str(v) for v in value]


def columns(data: Dataset, wanted: Sequence[str], what: str) -> pd.DataFrame:
    """The columns ``wanted`` of ``data`` (series, or axes of its index), rows with any missing dropped; the
    dataset's index is kept."""
    frame = data.frame
    levels = list(frame.index.names)
    missing = [n for n in wanted if n not in frame.columns and n not in levels]
    if missing:
        raise SpecificationError(f"{what}: no series {', '.join(map(repr, missing))} in the dataset")
    out = pd.DataFrame(index=frame.index)
    for n in dict.fromkeys(wanted):
        out[n] = frame[n] if n in frame.columns else frame.index.get_level_values(n)
    numeric = [n for n in out.columns if n not in levels]
    return out.dropna(subset=numeric)


def numeric(frame: pd.DataFrame, cols: Sequence[str], what: str) -> np.ndarray:
    try:
        return frame[list(cols)].to_numpy(dtype=float)
    except (TypeError, ValueError):
        raise SpecificationError(f"{what}: {', '.join(cols)} must be numbers") from None


def estimate_rows(values: Sequence[float], ses: Sequence[float], alpha: float = 0.05) -> list[list[float]]:
    q = sps.norm.ppf(1 - alpha / 2)
    rows = []
    for b, s in zip(values, ses):
        z = b / s if s > 0 else np.nan
        rows.append([float(b), float(s), float(z), float(2 * sps.norm.sf(abs(z))), float(b - q * s), float(b + q * s)])
    return rows


def coefficient_table(names_: Sequence[str], values: Sequence[float], ses: Sequence[float], title: str, notes: Sequence[str] = (),
                      index_name: str = "parameter") -> Table:
    return Table(COLUMNS, estimate_rows(values, ses), index=list(names_), index_name=index_name, title=title, notes=tuple(notes))


def provenance(spec: dict[str, Any], data: Dataset, used: pd.DataFrame, variables: Sequence[str], caught: list[str], method: dict[str, Any],
               *, converged: bool | None = None, seed: int | None = None, resolved: dict[str, Any] | None = None) -> Any:
    from ._capture import provenance_for

    cols = [c for c in variables if c in used.columns]
    return provenance_for(spec, data, list(variables), used.index, used[cols], caught, method, converged=converged, seed=seed,
                          resolved=resolved)
