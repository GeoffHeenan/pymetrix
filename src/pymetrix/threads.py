"""Threads of the linear-algebra library (BLAS) during pymetrix's long, repeated calculations.

Estimation, Metropolis-Hastings, identification and sweeps solve the same medium-sized problems thousands of times.
At these sizes (tens to a few hundred variables) a multithreaded BLAS spends more time coordinating threads than it
saves, and on a busy laptop far more: a 300 x 300 solve measured 2-6 ms on one thread and 40-190 ms on eight. So
pymetrix runs those calculations with one BLAS thread by default, through ``threadpoolctl``, and restores the setting
afterwards; everything else (and your own code) keeps the library's default. Parallel Metropolis-Hastings chains get
their speed from processes instead, one thread each.

::

    pymetrix.set_blas_threads(4)      # e.g. for very large models on a quiet workstation
    pymetrix.set_blas_threads(None)   # leave the library's own setting alone

On a Mac, NumPy and SciPy use Apple's Accelerate, which ``threadpoolctl`` cannot see. From macOS 15 Accelerate has
its own switch (``BLASSetThreading``): one thread, or as many as Accelerate decides. pymetrix uses it as well, so one
thread means one thread there too, and more than one means Accelerate's own choice (it takes no number). Before macOS
15 there is no such switch and the limit does nothing on Accelerate (docs/known-gaps.md).
"""

from __future__ import annotations

import ctypes
import functools
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

from .errors import SpecificationError

__all__ = ["blas_threads", "get_blas_threads", "set_blas_threads"]

_BLAS_THREADS: int | None = 1


def get_blas_threads() -> int | None:
    """BLAS threads used during estimation, sampling, identification and sweeps (None: the library's own setting)."""
    return _BLAS_THREADS


def set_blas_threads(n: int | None) -> None:
    """Set the BLAS threads used during estimation, sampling, identification and sweeps; None leaves them alone."""
    global _BLAS_THREADS
    if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 1):
        raise SpecificationError(f"BLAS threads must be a positive whole number or None, not {n!r}")
    _BLAS_THREADS = n


# Accelerate's threading models (vecLib/thread_api.h): BLAS_THREADING_MULTI_THREADED (its default: Accelerate decides)
# and BLAS_THREADING_SINGLE_THREADED. tests/test_threads.py checks these values against Accelerate on macOS.
ACCELERATE_MULTI_THREADED = 0
ACCELERATE_SINGLE_THREADED = 1
_ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Accelerate"


@functools.lru_cache(maxsize=1)
def _accelerate() -> tuple[object, object] | None:
    """Accelerate's (BLASGetThreading, BLASSetThreading) on macOS 15 or later, else None (other systems, or an
    older macOS, which has neither)."""
    if sys.platform != "darwin":
        return None
    try:
        library = ctypes.CDLL(_ACCELERATE)
        get, put = library.BLASGetThreading, library.BLASSetThreading
    except (OSError, AttributeError):
        return None
    get.argtypes, get.restype = [], ctypes.c_uint32
    put.argtypes, put.restype = [ctypes.c_uint32], ctypes.c_int
    return get, put


def accelerate_threading() -> int | None:
    """Accelerate's threading model on the calling thread (``ACCELERATE_SINGLE_THREADED`` or
    ``ACCELERATE_MULTI_THREADED``), or None where pymetrix cannot set it (not a Mac, or macOS before 15)."""
    found = _accelerate()
    return None if found is None else int(found[0]())


@contextmanager
def _accelerate_limit(n: int) -> Iterator[None]:
    """Accelerate's own switch inside the block: one thread for ``n`` 1, Accelerate's choice for more. The setting
    belongs to the calling thread (Accelerate keeps it per thread), as the block's calculation does."""
    found = _accelerate()
    if found is None:
        yield
        return
    get, put = found
    before = int(get())
    wanted = ACCELERATE_SINGLE_THREADED if n == 1 else ACCELERATE_MULTI_THREADED
    if wanted == before or put(wanted) != 0:
        yield
        return
    try:
        yield
    finally:
        put(before)


@contextmanager
def blas_threads() -> Iterator[None]:
    """Limit BLAS to ``get_blas_threads()`` threads inside the block (nothing changes when that is None). The limit
    goes through ``threadpoolctl`` (OpenBLAS, MKL, BLIS; nothing when it is not installed) and, on macOS 15 or later,
    Accelerate's own switch."""
    if _BLAS_THREADS is None:
        yield
        return
    with ExitStack() as stack:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            pass
        else:
            stack.enter_context(threadpool_limits(limits=_BLAS_THREADS, user_api="blas"))
        stack.enter_context(_accelerate_limit(_BLAS_THREADS))
        yield
