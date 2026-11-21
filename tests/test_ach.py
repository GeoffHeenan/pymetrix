"""The autoregressive conditional hazard, checked independently: its Bernoulli likelihood by a plain loop over periods
(psi updated at each event); on independent Bernoulli(p) events, no duration dependence (a near zero) and a hazard near
p; the parameters of a simulated ACH recovered; refusals."""

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, ach
from pymetrix.errors import SpecificationError


def _ds(x):
    return Dataset.from_frame(pd.DataFrame({"e": x}, index=pd.period_range("1900-01", periods=len(x), freq="M")))


def _simulate(T, omega, a, b, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    psi, last = omega / (1 - a - b), -1
    for t in range(T):
        x[t] = rng.random() < 1 / max(psi, 1.0)
        if x[t]:
            psi = omega + a * (t - last) + b * psi
            last = t
    return x


def test_the_likelihood_by_a_loop_and_recovery():
    x = _simulate(6000, 2.0, 0.3, 0.5, 0)
    fit = ach(_ds(x), "e")
    omega, a, b = fit.values
    psi, last, ll = len(x) / x.sum(), -1, 0.0
    for t in range(len(x)):
        h = 1 / max(psi, 1 + 1e-9)
        ll += np.log(h) if x[t] else np.log(1 - h)
        if x[t]:
            psi = omega + a * (t - last) + b * psi
            last = t
    assert fit.loglike == pytest.approx(ll, rel=1e-10)
    for v, true, s in zip(fit.values, (2.0, 0.3, 0.5), fit.se):
        assert v == pytest.approx(true, abs=3.5 * s)


def test_independent_events_have_no_duration_dependence():
    x = (np.random.default_rng(1).random(4000) < 0.15).astype(float)
    fit = ach(_ds(x), "e")
    assert fit.values[1] == pytest.approx(0, abs=3 * fit.se[1] + 0.02)
    assert fit.fitted.mean() == pytest.approx(x.mean(), abs=0.01)
    with pytest.raises(SpecificationError, match="0 or 1"):
        ach(_ds(x * 2), "e")
