"""Prior distributions, for Bayesian estimation and posterior-mode estimation of state-space models.

Parameterised by mean and standard deviation, as in Dynare, so priors read
the way economists usually state them. ``logpdf(x)`` is the log density and
``support`` the interval where it is positive (estimation keeps parameters inside it); ``rvs(rng)`` is a draw
(estimation's extra starting points).

The two inverse gammas are Dynare's: ``InvGamma1`` (Dynare's ``inv_gamma_pdf`` and
``inv_gamma1_pdf``) is the usual prior on a standard deviation, s/sigma^2 being chi-squared;
its standard deviation may be infinite (2 degrees of freedom, as fs2000 uses).
``InvGamma`` (also ``InvGamma2``, Dynare's ``inv_gamma2_pdf``) is the inverse gamma of the
value itself, the usual prior on a variance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import optimize, special

from pymetrix._lazy import stats

from ..errors import SpecificationError


@dataclass(frozen=True)
class Normal:
    mean: float
    sd: float

    support = (-np.inf, np.inf)

    def logpdf(self, x: float) -> float:  # closed forms here: estimation evaluates every prior at every trial value
        z = (x - self.mean) / self.sd
        return float(-0.5 * z * z - math.log(self.sd) - 0.5 * math.log(2 * math.pi))

    def rvs(self, rng: np.random.Generator) -> float:
        return float(rng.normal(self.mean, self.sd))


@dataclass(frozen=True)
class Beta:
    """A beta prior on (0, 1), given by its mean and standard deviation (as Dynare's beta_pdf)."""

    mean: float
    sd: float

    support = (0.0, 1.0)

    def logpdf(self, x: float) -> float:
        common = self.mean * (1 - self.mean) / self.sd**2 - 1
        if common <= 0:
            raise SpecificationError("Beta prior: the standard deviation is too large for the mean")
        a, b = self.mean * common, (1 - self.mean) * common
        if not 0 < x < 1:
            return float(stats.beta.logpdf(x, a, b))  # the edges as SciPy has them
        return float((a - 1) * math.log(x) + (b - 1) * math.log1p(-x) - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)))

    def rvs(self, rng: np.random.Generator) -> float:
        common = self.mean * (1 - self.mean) / self.sd**2 - 1
        return float(rng.beta(self.mean * common, (1 - self.mean) * common))


@dataclass(frozen=True)
class Gamma:
    """A gamma prior on (0, inf), given by its mean and standard deviation (as Dynare's gamma_pdf)."""

    mean: float
    sd: float

    support = (0.0, np.inf)

    def logpdf(self, x: float) -> float:
        shape = (self.mean / self.sd) ** 2
        scale = self.sd**2 / self.mean
        if x <= 0:
            return float(stats.gamma.logpdf(x, shape, scale=scale))
        return float((shape - 1) * math.log(x) - x / scale - math.lgamma(shape) - shape * math.log(scale))

    def rvs(self, rng: np.random.Generator) -> float:
        return float(rng.gamma((self.mean / self.sd) ** 2, self.sd**2 / self.mean))


@dataclass(frozen=True)
class InvGamma:
    """Inverse gamma of the value itself (Dynare's ``inv_gamma2_pdf``), with this mean and standard deviation."""

    mean: float
    sd: float

    support = (0.0, np.inf)

    def logpdf(self, x: float) -> float:
        a = (self.mean / self.sd) ** 2 + 2  # matches the mean and standard deviation
        scale = self.mean * (a - 1)
        if x <= 0:
            return float(stats.invgamma.logpdf(x, a, scale=scale))
        return float(a * math.log(scale) - math.lgamma(a) - (a + 1) * math.log(x) - scale / x)

    def rvs(self, rng: np.random.Generator) -> float:
        a = (self.mean / self.sd) ** 2 + 2
        return float(self.mean * (a - 1) / rng.gamma(a))


InvGamma2 = InvGamma


@lru_cache(maxsize=1024)
def _inverse_gamma_specification(mu: float, sd: float) -> tuple[float, float]:
    """InvGamma1's nu and s for this mean and standard deviation (see ``InvGamma1.nu_s``)."""
    if math.isinf(sd):
        return 2.0, 2 * mu**2 / math.pi
    second = sd**2 + mu**2

    def gap(log_excess: float) -> float:  # nu = 2 + exp(log_excess); s = (nu - 2) * (sd^2 + mu^2)
        nu = 2.0 + math.exp(log_excess)
        log_half_s = log_excess + math.log(second) - math.log(2.0)
        return 0.5 * log_half_s + special.gammaln((nu - 1) / 2) - special.gammaln(nu / 2) - math.log(mu)

    # gap rises from very negative towards 0.5 log(1 + (sd / mu)^2) > 0 as nu grows; far out its gammaln terms lose
    # every digit (at log_excess = 40 it reads exactly 0, which brentq took for a root): bracket where it is accurate
    upper = 20.0
    while gap(upper) <= 0 and upper < 34:
        upper += 2.0
    log_excess = optimize.brentq(gap, -40.0, upper, xtol=1e-14)
    nu = 2.0 + math.exp(log_excess)
    return nu, (nu - 2.0) * second


@dataclass(frozen=True)
class InvGamma1:
    """Dynare's inverse gamma of type 1 (``inv_gamma_pdf``), the usual prior on a standard deviation sigma:
    density proportional to sigma^-(nu+1) exp(-s / (2 sigma^2)), with nu and s matching the mean and standard
    deviation (an infinite standard deviation gives nu = 2)."""

    mean: float
    sd: float

    support = (0.0, np.inf)

    def __post_init__(self) -> None:
        if self.mean <= 0 or self.sd <= 0:
            raise SpecificationError("InvGamma1 prior: the mean and standard deviation must be positive")

    @property
    def nu_s(self) -> tuple[float, float]:
        """The degrees of freedom nu and scale s (Dynare's ``inverse_gamma_specification``), found once for each mean and
        standard deviation (a root search: each log density needs them)."""
        return _inverse_gamma_specification(float(self.mean), float(self.sd))

    def logpdf(self, x: float) -> float:
        if x <= 0:
            return -np.inf
        nu, s = self.nu_s
        return float(math.log(2) - special.gammaln(nu / 2) + (nu / 2) * math.log(s / 2) - (nu + 1) * math.log(x) - s / (2 * x * x))

    def rvs(self, rng: np.random.Generator) -> float:
        nu, s = self.nu_s  # s / sigma^2 is chi-squared with nu degrees of freedom
        return float(math.sqrt(s / rng.chisquare(nu)))


@dataclass(frozen=True)
class Uniform:
    low: float
    high: float

    @property
    def support(self) -> tuple[float, float]:
        return (self.low, self.high)

    def logpdf(self, x: float) -> float:
        return float(-np.log(self.high - self.low)) if self.low <= x <= self.high else -np.inf

    def rvs(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


Distribution = Normal | Beta | Gamma | InvGamma | InvGamma1 | Uniform


_NAMES = ("Normal", "Beta", "Gamma", "InvGamma", "InvGamma1", "InvGamma2", "Uniform")


def specs(text: str) -> dict[str, tuple[str, list[float]]]:
    """``kappa=Gamma(0.3, 0.1); rho=Beta(0.8, 0.1)`` -> {name: (distribution, [numbers])}, each prior checked."""
    import re

    out = {}
    for item in [t.strip() for t in re.split(r"[;\n]", text) if t.strip()]:
        m = re.fullmatch(r"(.+?)\s*=\s*(\w+)\s*\(([^()]*)\)", item)  # a name may be dlog(y) or y[-1]
        if not m or m.group(2) not in _NAMES:
            raise SpecificationError(f"give priors as name=Distribution(mean, sd), with Normal, Beta, Gamma, InvGamma1, InvGamma2 "
                                     f"(= InvGamma) or Uniform(low, high): {item!r}")
        try:
            values = [float(v) for v in m.group(3).split(",") if v.strip()]
        except ValueError:
            raise SpecificationError(f"{m.group(2)} takes two numbers: {item!r}") from None
        if len(values) != 2:
            raise SpecificationError(f"{m.group(2)} takes two numbers: {item!r}")
        globals()[m.group(2)](*values)
        out[m.group(1)] = (m.group(2), values)
    return out


def parse(text: str) -> dict[str, object]:
    """Priors typed as text (``x=Normal(0.5, 0.1); sigma=InvGamma1(1, 1)``), by name: what the dialogs give."""
    return {n: globals()[kind](*values) for n, (kind, values) in specs(text).items()}
