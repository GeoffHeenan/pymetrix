"""BLAS threads during estimation and sampling (pymetrix.threads): one by default, restored afterwards.

threadpoolctl sees OpenBLAS (Linux, Windows) but not Apple's Accelerate, which NumPy and SciPy use on a Mac; there the
limit goes through Accelerate's own switch (macOS 15 or later), and these tests check that switch as well."""

import sys

import numpy as np
import pytest

import pymetrix
from pymetrix import SpecificationError, templates
from pymetrix import threads

threadpoolctl = pytest.importorskip("threadpoolctl")

SINGLE, MULTI = threads.ACCELERATE_SINGLE_THREADED, threads.ACCELERATE_MULTI_THREADED


def _blas_threads() -> list[int]:
    return [i["num_threads"] for i in threadpoolctl.threadpool_info() if i["user_api"] == "blas"]


def _state() -> tuple[set[int], int | None]:
    """The threads of each BLAS threadpoolctl sees, and Accelerate's threading model (None: not a Mac)."""
    return set(_blas_threads()), threads.accelerate_threading()


def _one_thread() -> tuple[set[int], int | None]:
    """What pymetrix's limit of one thread gives: one thread in each BLAS seen, Accelerate single-threaded."""
    return ({1} if _blas_threads() else set()), (None if threads.accelerate_threading() is None else SINGLE)


@pytest.fixture(autouse=True)
def _restore():
    before = threads.get_blas_threads()
    # two threads outside pymetrix's limit, whatever the test run chose (tests/conftest.py asks for one), so that the
    # limit inside can be told from the setting outside; on a Mac, Accelerate's default (it decides) outside
    with threadpoolctl.threadpool_limits(limits=2, user_api="blas"):
        if sys.platform == "darwin":
            assert threads.accelerate_threading() == MULTI and set(_blas_threads()) <= {2}
        else:
            assert set(_blas_threads()) == {2}
        yield
    threads.set_blas_threads(before)


def test_numpys_blas_is_one_pymetrix_can_limit():
    if sys.platform == "darwin":  # Accelerate (threadpoolctl finds none) or OpenBLAS: either is limited
        assert _blas_threads() or threads.accelerate_threading() is not None, threadpoolctl.threadpool_info()
    else:
        assert _blas_threads() and threads.accelerate_threading() is None


@pytest.mark.skipif(sys.platform != "darwin", reason="Accelerate's switch is macOS's")
def test_accelerates_switch_has_the_values_pymetrix_sets():
    get, put = threads._accelerate()
    assert get() == MULTI  # the default: Accelerate decides
    assert put(SINGLE) == 0 and get() == SINGLE
    assert put(MULTI) == 0 and get() == MULTI


def test_the_limit_holds_inside_and_is_restored_after():
    outside = _state()
    with threads.blas_threads():
        assert _state() == _one_thread()
    assert _state() == outside
    threads.set_blas_threads(None)
    with threads.blas_threads():
        assert _state() == outside
    threads.set_blas_threads(2)  # more than one: threadpoolctl's number, and Accelerate decides (it takes no number)
    with threads.blas_threads():
        assert _state() == outside
    with pytest.raises(SpecificationError):
        pymetrix.set_blas_threads(0)


def test_estimation_runs_with_the_limit(monkeypatch):
    seen = []
    nk = templates.new_keynesian()
    original = nk.model.__class__.linearize

    def spy(self, *args, **kwargs):
        seen.append(_state())
        return original(self, *args, **kwargs)

    import pandas as pd

    from pymetrix import Dataset

    truth = nk.solve().simulate(60, nk.shock_sd, seed=2)
    truth.index = pd.period_range("2000Q1", periods=60, freq="Q", name="time")
    data = nk.measurement.apply(Dataset.from_frame(truth))
    monkeypatch.setattr(nk.model.__class__, "linearize", spy)
    outside = _state()
    nk.estimate(data, estimate=["rho"])
    assert seen and all(s == _one_thread() for s in seen)
    assert _state() == outside and np.all(np.array(_blas_threads()) >= 1)
