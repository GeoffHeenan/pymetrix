"""Bai and Ng's (2002) criteria for the number of factors (Elliott and Timmermann 2016, section 10.3): V(k) against the
residuals of regressing every series on the first k components, the penalties by hand, and the true number of
factors recovered from simulated data."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import PCA, Dataset
from pymetrix.errors import SpecificationError


def _panel(n=40, t=200, k=3, seed=0):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(t, k))
    loadings = rng.normal(size=(k, n))
    x = factors @ loadings + rng.normal(scale=1.0, size=(t, n))
    idx = pd.period_range("1970Q1", periods=t, freq="Q", name="time")
    return Dataset.from_frame(pd.DataFrame(x, index=idx, columns=[f"x{i}" for i in range(n)]))


def test_v_is_the_residual_variance_after_k_components_and_the_penalties():
    data = _panel()
    names = list(data.names)
    fit = PCA(names).fit(data)
    table = fit.factor_criteria(kmax=6)
    frame = data.frame[names]
    z = ((frame - frame.mean()) / frame.std(ddof=1)).to_numpy()
    N, T = z.shape[1], z.shape[0]
    scores = fit.scores.to_numpy()
    for k in (1, 3, 6):
        f = scores[:, :k]
        resid = z - f @ np.linalg.lstsq(f, z, rcond=None)[0]  # each series on the first k components
        v = float(np.sum(resid**2) / (N * T))
        row = table.rows[k - 1]
        assert row[0] == pytest.approx(v, rel=1e-10)
        assert row[1] == pytest.approx(np.log(v) + k * (N + T) / (N * T) * np.log(N * T / (N + T)), rel=1e-12)
        assert row[2] == pytest.approx(np.log(v) + k * (N + T) / (N * T) * np.log(min(N, T)), rel=1e-12)
        assert row[3] == pytest.approx(np.log(v) + k * np.log(min(N, T)) / min(N, T), rel=1e-12)


@pytest.mark.parametrize("k", [2, 4])
def test_the_criteria_find_the_number_of_factors(k):
    data = _panel(n=50, t=200, k=k, seed=k)
    table = PCA(list(data.names)).fit(data).factor_criteria(kmax=8)
    assert f"IC_p2 {k}" in table.notes[0] and f"IC_p1 {k}" in table.notes[0]


def test_kmax_is_checked():
    data = _panel(n=5, t=50)
    with pytest.raises(SpecificationError, match="kmax"):
        PCA(list(data.names)).fit(data).factor_criteria(kmax=5)
