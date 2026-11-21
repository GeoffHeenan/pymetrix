"""The bundled example datasets: each loads, says what it holds and where it comes from, and the simulated ones
have the effects their descriptions state."""

import numpy as np
import pytest

from pymetrix import Dataset, Equation
from pymetrix.datasets import available


def _downloaded(name: str) -> bool:
    """A plugin's example given as a path-like that fetches its data on first use (not a file it ships)."""
    import os
    from pathlib import Path

    from pymetrix.datasets import _plugins

    entry = _plugins().get(name)
    return entry is not None and isinstance(entry.path, os.PathLike) and not isinstance(entry.path, Path)


@pytest.mark.parametrize("name", sorted(available()))
def test_each_example_loads_with_its_source(name):
    if _downloaded(name):
        pytest.skip("downloaded on first use: the plugin's own tests load it from a stand-in, without the network")
    data = Dataset.example(name)
    assert len(data.frame) > 0 and data.meta.title and data.meta.license  # what it is, and where it comes from
    if name not in ("sample_macro", "sample_monthly"):
        assert data.meta.description  # what it is for


def test_the_shapes_are_as_described():
    assert Dataset.example("mroz").kind == "cross_section" and len(Dataset.example("mroz").frame) == 753
    assert Dataset.example("wage_panel").kind == "panel" and len(Dataset.example("wage_panel").frame) == 4360
    assert Dataset.example("nile").time_axis is not None and len(Dataset.example("nile").frame) == 100
    assert Dataset.example("mroz").variables["educ"].label == "years of schooling"


def test_the_simulated_effects_are_there():
    rd = Dataset.example("rd_sample")
    fit = Equation("y ~ score + score^2 + above").fit(rd)
    se = dict(zip(fit.params.names, fit.params.se))
    assert abs(fit.params["above"] - 1.5) < 3 * se["above"]  # the true jump, within sampling error
    did = Equation("y ~ d + x + C(state) + C(year)").fit(Dataset.example("policy_panel"))
    se = dict(zip(did.params.names, did.params.se))
    assert abs(did.params["d"] - 2.0) < 3 * se["d"]
    synth = Dataset.example("synth_panel").frame["y"].unstack("region")
    assert np.isfinite(synth.to_numpy()).all() and synth.shape == (26, 20)
