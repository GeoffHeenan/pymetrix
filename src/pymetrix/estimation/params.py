"""Estimated parameters: point estimates or draws, behind one interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pymetrix._lazy import stats

from .._random import generator as make_rng
from ..errors import SpecificationError


@dataclass(frozen=True, eq=False)
class PointEstimate:
    names: tuple[str, ...]
    values: np.ndarray
    cov: np.ndarray | None = None
    aliases: Mapping[str, str] | None = None  # names used before a rename (old -> new): still found, with a warning

    def __getitem__(self, name: str) -> float:
        if name not in self.names and self.aliases and name in self.aliases:
            import warnings

            warnings.warn(f"parameter {name!r} is now called {self.aliases[name]!r}", DeprecationWarning, stacklevel=2)
            name = self.aliases[name]
        return float(self.values[self.names.index(name)])

    def __len__(self) -> int:
        return len(self.names)

    def as_dict(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.values)}

    @property
    def se(self) -> np.ndarray:
        if self.cov is None:
            return np.full(len(self.names), np.nan)
        return np.sqrt(np.diag(self.cov))

    def interval(self, alpha: float = 0.05) -> pd.DataFrame:
        z = stats.norm.ppf(1 - alpha / 2)
        return pd.DataFrame(
            {"lower": self.values - z * self.se, "upper": self.values + z * self.se},
            index=list(self.names),
        )

    def sample(self, n: int, seed: int | None = None) -> Draws:
        if self.cov is None:
            raise SpecificationError("cannot sample parameters without a covariance matrix")
        rng = make_rng(seed)
        return Draws(self.names, rng.multivariate_normal(self.values, self.cov, size=n))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"value": self.values, "se": self.se}, index=list(self.names))


@dataclass(frozen=True, eq=False)
class Draws:
    """Posterior or bootstrap draws; rows are draws, columns parameters."""

    names: tuple[str, ...]
    draws: np.ndarray

    @property
    def values(self) -> np.ndarray:
        return self.draws.mean(axis=0)

    @property
    def cov(self) -> np.ndarray:
        return np.atleast_2d(np.cov(self.draws, rowvar=False))

    @property
    def se(self) -> np.ndarray:
        return self.draws.std(axis=0, ddof=1)

    def __getitem__(self, name: str) -> float:
        return float(self.values[self.names.index(name)])

    def __len__(self) -> int:
        return len(self.names)

    def as_dict(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.values)}

    def interval(self, alpha: float = 0.05) -> pd.DataFrame:
        lo, hi = np.quantile(self.draws, [alpha / 2, 1 - alpha / 2], axis=0)
        return pd.DataFrame({"lower": lo, "upper": hi}, index=list(self.names))

    def sample(self, n: int, seed: int | None = None) -> Draws:
        rng = make_rng(seed)
        return Draws(self.names, self.draws[rng.integers(0, len(self.draws), size=n)])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"value": self.values, "se": self.se}, index=list(self.names))


Params = PointEstimate | Draws


def point(names: Sequence[str], values: Sequence[float], cov: np.ndarray | None = None, aliases: Mapping[str, str] | None = None) -> PointEstimate:
    return PointEstimate(tuple(names), np.asarray(values, dtype=float), cov, dict(aliases) if aliases else None)
