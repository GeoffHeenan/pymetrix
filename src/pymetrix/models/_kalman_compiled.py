"""The Kalman filter's sequential loops compiled by numba, when it is installed (``pip install pymetrix[fast]``).

Never required: without numba (or with ``COMPILED = "off"``, or the environment variable ``PYMETRIX_NUMBA=0``) the
same recursions run in NumPy (``pymetrix.models.kalman_score``) and statsmodels' compiled filter gives the plain log
likelihood (``pymetrix.statespace.loglike``). The compiled loops do the same arithmetic in the same order up to
rounding (the products by loops or BLAS, the Cholesky factor by its textbook recursion), so the likelihood and its
gradient agree with the NumPy path to about 1e-12 relative (the tests ask 1e-10).

What is compiled is what Python cannot vectorise, the period-by-period recursions:

* ``variance_pass``: the predicted variance ``P``, the gain's pieces and the innovation variance's inverse Cholesky
  factor, period by period until the variance converges (Dynare's ``riccati_tol``), as ``kalman_score._filter``;
* ``means``, ``backward_means``, ``backward_variance``: the means' recursion forwards and the adjoint's two
  backwards (``kalman_score._adjoint``);
* ``loglike``: the whole filter for the log likelihood alone (variance and means in one pass, the converged
  shortcut, missing observations), started from a known mean and variance: what estimation and Metropolis-Hastings
  evaluate many times. statsmodels' filter, which it replaces where the start is not diffuse, spends most of its time
  setting up; this costs a few microseconds a period.

Each kernel is compiled on first use (about ten seconds, once) and cached on disk (numba's ``cache=True``, beside the
module or in numba's per-user cache), so later sessions and chains in other processes start at once.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["COMPILED", "available", "kernels"]

COMPILED = "auto"  # "auto": numba's kernels when numba is installed; "off": NumPy and statsmodels alone

_KERNELS: Any = None
_TRIED = False


def available() -> bool:
    """Whether the compiled kernels are in use: numba installed, and not switched off."""
    if COMPILED == "off" or os.environ.get("PYMETRIX_NUMBA", "") == "0":
        return False
    return kernels() is not None


def kernels() -> Any:
    """The compiled kernels (``pymetrix.models._kalman_kernels``), or None without numba."""
    global _KERNELS, _TRIED
    if not _TRIED:
        _TRIED = True
        try:
            _KERNELS = _build()
        except ImportError:  # no numba
            _KERNELS = None
    return _KERNELS


def _build() -> Any:
    from . import _kalman_kernels

    return _kalman_kernels
