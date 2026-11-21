"""OccBin against Dynare 6.0 on pymetrix's New Keynesian template with a zero lower bound (the floor at -0.75 in
deviations), a demand shock of -2.5 in each of periods 1-4: Dynare's own ``occbin_solver`` with the shocks as
surprises (its piecewise-linear and linear paths), and Dynare's ``perfect_foresight_solver`` with the shocks known
from period 1, with the floor written as ``max`` and without it. The reference (tests/fixtures/dynare-occbin-zlb.json)
comes from tests/fixtures/dynare/nk_zlb.mod and nk_zlb_pf.mod, run by the dynare-reference workflow (case
occbin_zlb)."""

import json
from pathlib import Path

import numpy as np
import pytest

from pymetrix import occbin, templates

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "dynare-occbin-zlb.json").read_text(encoding="utf-8"))
SHOCKS = {"ey": {1: -2.5, 2: -2.5, 3: -2.5, 4: -2.5}}


@pytest.fixture(scope="module")
def nk_zlb():
    return templates.new_keynesian(lower_bound=0.0)


def _same(frame, key):  # Dynare's paths start in period 1, after the steady state
    expected = np.array(REFERENCE[key])
    assert np.allclose(frame[REFERENCE["variables"]].to_numpy(), expected, atol=1e-9), key


def test_surprises_equal_dynare_occbin(nk_zlb):
    path = occbin(nk_zlb, SHOCKS, periods=20)
    _same(path.deviations, "occbin_surprise")
    _same(path.linear, "linear_surprise")  # the template is in deviations: its levels are too
    assert not path.binding.any().any()  # as surprises the rate never reaches the floor


def test_anticipated_shocks_equal_dynare_perfect_foresight(nk_zlb):
    """The demo's zero-lower-bound episode: with the shocks known in advance the floor binds in periods 1-3, and the
    path without it (dashed in the graph) is Dynare's perfect-foresight path of the model without the floor, which
    goes below it."""
    path = occbin(nk_zlb, SHOCKS, periods=20, anticipated=True)
    _same(path.deviations, "perfect_foresight_floor")
    _same(path.linear, "perfect_foresight_linear")
    assert list(path.binding.index[path.binding["policy (max)"]]) == [1, 2, 3]
    assert np.array(REFERENCE["perfect_foresight_linear"])[:3, 2].max() < -0.75
