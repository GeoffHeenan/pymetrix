"""The random number generator behind every simulation, bootstrap and sampler in pymetrix.

Every random draw pymetrix makes comes from a NumPy ``Generator`` built here, from a seed, by ``generator(seed)``;
streams for parallel chains, particle blocks and Monte Carlo replications are spawned from ``seed_sequence(seed)``
and each turned into a Generator the same way. The *bit generator* under it is a package setting:

    pymetrix.set_random_generator("mt19937")      # the Mersenne Twister from now on
    pymetrix.get_random_generator()               # 'mt19937'
    with pymetrix.random_generator("philox"):     # only inside the block
        ...

- ``"pcg64"`` (the default): NumPy's own default, ``np.random.default_rng(seed)``. Results are exactly those of
  earlier versions of pymetrix.
- ``"mt19937"``: the Mersenne Twister (``np.random.Generator(np.random.MT19937(seed))``), the generator of MATLAB,
  Stata, R and Python's ``random`` module.
- ``"philox"``: a counter-based generator (Random123), popular on GPUs and for very many parallel streams.
- ``"sfc64"``: Chris Doty-Humphrey's small fast chaotic generator.

The same seed gives the same draws for a given generator, on any machine and in any number of processes (the
setting travels to the processes that run chains and replications). Results record the generator with the seed
(``provenance.random_generator``), and a project replays each step with the generator it was run with.

**Not MATLAB's draws.** Choosing ``"mt19937"`` does not reproduce MATLAB's, Dynare's or Stata's random numbers, even
with the same seed: NumPy seeds the Mersenne Twister through ``SeedSequence`` (not MATLAB's ``rng(seed)``
initialisation), and turns its bits into normal draws with its own ziggurat (MATLAB's ``randn`` uses another
ziggurat, Stata's ``rnormal`` the inverse CDF). Simulations agree in distribution, not draw for draw.

**Already the Mersenne Twister.** A few paths go through a library's legacy ``np.random.RandomState`` (itself an
MT19937): the ``arch`` package's bootstraps, statsmodels' Markov-switching random starts and scikit-learn's
``random_state``. They are seeded from the same seed and are unaffected by this setting.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

import numpy as np

from .errors import SpecificationError

__all__ = ["GENERATORS", "generator", "get_random_generator", "random_generator", "seed_sequence", "set_random_generator"]

# name -> NumPy bit generator class
GENERATORS: dict[str, type] = {
    "pcg64": np.random.PCG64,
    "mt19937": np.random.MT19937,
    "philox": np.random.Philox,
    "sfc64": np.random.SFC64,
}
DEFAULT = "pcg64"
ENVIRONMENT = "PYMETRIX_RANDOM_GENERATOR"  # how processes that run chains and replications inherit the setting


def _checked(name: Any) -> str:
    key = str(name).strip().lower() if isinstance(name, str) else name
    if key not in GENERATORS:
        raise SpecificationError(f"the random generator is one of {', '.join(repr(g) for g in GENERATORS)}, not {name!r}")
    return key


def _initial() -> str:
    try:
        return _checked(os.environ.get(ENVIRONMENT, DEFAULT) or DEFAULT)
    except SpecificationError:
        return DEFAULT


_CURRENT = _initial()


def get_random_generator() -> str:
    """The bit generator pymetrix draws with: ``"pcg64"`` (the default), ``"mt19937"``, ``"philox"`` or ``"sfc64"``."""
    return _CURRENT


def set_random_generator(name: str) -> str:
    """Choose the bit generator for every random draw pymetrix makes from now on; returns the previous one.

    ``"pcg64"`` (the default, NumPy's), ``"mt19937"`` (Mersenne Twister), ``"philox"`` or ``"sfc64"``. Even the
    Mersenne Twister does not reproduce MATLAB's or Dynare's draws (they seed it and make normal draws differently)."""
    global _CURRENT
    previous, _CURRENT = _CURRENT, _checked(name)
    os.environ[ENVIRONMENT] = _CURRENT  # processes started from here draw the same way
    return previous


@contextlib.contextmanager
def random_generator(name: str) -> Iterator[str]:
    """Use another bit generator inside the block, then go back to the one before."""
    previous = set_random_generator(name)
    try:
        yield _CURRENT
    finally:
        set_random_generator(previous)


def generator(seed: Any = None, name: str | None = None) -> np.random.Generator:
    """A NumPy ``Generator`` from ``seed`` (an integer, a sequence of them, a ``SeedSequence`` or None), with the bit
    generator chosen by ``set_random_generator`` (or ``name``). A Generator is returned as it is. With the default
    ``"pcg64"`` this is exactly ``np.random.default_rng(seed)``."""
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, np.random.BitGenerator):
        return np.random.Generator(seed)
    key = _CURRENT if name is None else _checked(name)
    if key == "pcg64":
        return np.random.default_rng(seed)
    return np.random.Generator(GENERATORS[key](seed))


def seed_sequence(seed: Any = None) -> np.random.SeedSequence:
    """The ``SeedSequence`` that independent streams (chains, particle blocks, replications) are spawned from; each
    child becomes a Generator through ``generator(child)``, so the chosen bit generator applies to every stream."""
    if isinstance(seed, np.random.SeedSequence):
        return seed
    return np.random.SeedSequence(seed)
