"""Distributions of the dependent variable (part of an equation's specification).

Each family maps the linear index ``eta = X @ beta`` to the mean of the
dependent variable (``mean``). ``ML`` estimates them all: Logit, Probit and Poisson here (``estimation.ml``),
and the limited dependent variable models, whose likelihoods have more than the index (``estimation.limited``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, ndtr


@dataclass(frozen=True)
class Gaussian:
    def mean(self, eta: np.ndarray) -> np.ndarray:
        return eta


@dataclass(frozen=True)
class Logit:
    def mean(self, eta: np.ndarray) -> np.ndarray:
        return expit(eta)


@dataclass(frozen=True)
class Probit:
    def mean(self, eta: np.ndarray) -> np.ndarray:
        return ndtr(eta)


@dataclass(frozen=True)
class Poisson:
    def mean(self, eta: np.ndarray) -> np.ndarray:
        return np.exp(eta)


@dataclass(frozen=True)
class NegativeBinomial:
    """Counts with more variance than the Poisson allows: NB2, Var(y) = mu + alpha mu^2."""

    def mean(self, eta: np.ndarray) -> np.ndarray:
        return np.exp(eta)


@dataclass(frozen=True)
class ZeroInflatedPoisson:
    """Counts with extra zeros: a structural zero with probability logistic(inflate terms), else a Poisson count.
    ``inflate`` lists the regressors of that probability ("" = a constant only)."""

    inflate: str = ""


@dataclass(frozen=True)
class OrderedLogit:
    """Ordered outcomes (ratings, grades): P(y <= j) = logistic(cut_j - x b), with cut points in place of a constant."""


@dataclass(frozen=True)
class OrderedProbit:
    """Ordered outcomes: P(y <= j) = Phi(cut_j - x b)."""


@dataclass(frozen=True)
class Tobit:
    """A regression censored at ``left`` and/or ``right`` (None: no limit): values at a limit stand for any beyond it."""

    left: float | None = 0.0
    right: float | None = None


@dataclass(frozen=True)
class Multinomial:
    """Unordered outcomes (multinomial logit): coefficients for each outcome against the ``base`` (default: the first)."""

    base: float | None = None


Family = Gaussian | Logit | Probit | Poisson | NegativeBinomial | ZeroInflatedPoisson | OrderedLogit | OrderedProbit | Tobit | Multinomial
