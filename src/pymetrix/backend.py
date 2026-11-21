"""Numerical back end selection.

Expressions are evaluated with NumPy today. The design calls for compiling
expression trees to NumPy or JAX functions; this module is where that choice
will live.
"""

from __future__ import annotations

from typing import Literal

from .errors import SpecificationError

_BACKEND: Literal["numpy", "jax"] = "numpy"


def get_backend() -> str:
    return _BACKEND


def set_backend(name: Literal["numpy", "jax"]) -> None:
    global _BACKEND
    if name == "jax":
        raise NotImplementedError("the JAX back end is planned for build stage 5")
    if name != "numpy":
        raise SpecificationError(f"unknown back end {name!r}")
    _BACKEND = name
