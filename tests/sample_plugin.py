"""A minimal third-party plugin, written as an outside package would write it.

Its entry points (simulated in test_conformance.py) would be::

    [project.entry-points."pymetrix.filters"]
    demean = "sample_plugin:Demean"

    [project.entry-points."pymetrix.tests"]
    mean_zero = "sample_plugin:MeanZero"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from scipy import stats

from pymetrix.data import Dataset
from pymetrix.forms import Widget
from pymetrix.plugins import PluginInfo
from pymetrix.stats import Test, TestResult

PLUGIN = PluginInfo(
    name="pymetrix-sample",
    version="0.1.0",
    requires={"scipy": ">=1.11"},
    license="MIT",
    maintainer="A. Author",
    description="Example filter and test",
)


@dataclass(frozen=True)
class Demean:
    """Deviation from the (scaled) sample mean: the trend is a constant."""

    label: ClassVar[str] = "Demean"
    topics: ClassVar[tuple[str, ...]] = ("trend-cycle-filters",)

    scale: Annotated[float, Widget("number", min=0, max=10, help="multiplies the mean used as the trend")] = 1.0

    def describe(self, f: int) -> str:
        return f"Demean(scale={self.scale:g})"

    def trend(self, y: np.ndarray, f: int) -> np.ndarray:
        return np.full_like(y, self.scale * np.nanmean(y))


@dataclass(frozen=True)
class MeanZero(Test):
    """t test that a series has mean zero."""

    variable: Annotated[str, Widget("text", help="the series to test")]

    name = "mean_zero"
    null = "the mean is zero"
    applies_to = (Dataset,)
    topics = ("diagnostic-tests",)

    def run(self, data: Dataset) -> TestResult:
        x = data[self.variable].dropna().to_numpy(float)
        res = stats.ttest_1samp(x, 0.0)
        return TestResult(f"Mean zero ({self.variable})", self.null, float(res.statistic), len(x) - 1, float(res.pvalue), "t")
