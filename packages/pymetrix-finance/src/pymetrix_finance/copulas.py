"""Copulas: the dependence between returns apart from their margins, its tails, and portfolio risk by simulation.

    cop = copula(data, ["a", "b", "c"], family="t", garch=True)       # t copula of GARCH-filtered returns
    cop.summary(); cop.tail_dependence()
    cop.portfolio_var([0.5, 0.3, 0.2], level=0.99)                   # VaR and ES by simulation
    copula(data, ["a", "b"], family="bb1", margins="skewt")           # parametric margins, IFM
    copula(data, ["a", "b"], family="clayton", gof=True)             # with a goodness-of-fit test
    copula(data, ["a", "b"], family="tawn1", gof=True, bootstrap="multiplier", n_boot=1000)   # asymmetric, fast test
    compare_copulas(data, ["a", "b"])                                # every family on the same data: AIC, BIC

**Families.** ``gaussian`` and ``t`` for any number of series; the Archimedean ``clayton``, ``gumbel``, ``frank`` and
``joe`` for any number (for more than two they are exchangeable, one parameter shared by every pair); the
two-parameter ``bb1`` (Clayton-Gumbel) and ``bb7`` (Joe-Clayton), the survival (180 degree) versions ``clayton180``,
``gumbel180``, ``joe180`` and the 90 and 270 degree rotations (``clayton90``, ..., negative dependence) for two; also
for two ``bb6`` (Joe-Gumbel), ``bb8`` (Joe-Frank), Tawn's asymmetric extreme-value copulas ``tawn1`` and ``tawn2``, and
the rotations of the two-parameter families, named with an underscore (``bb1_180``, ``bb7_90``, ``tawn1_270``, ...; as
VineCopula's 17-40, 104-134 and 204-234). Frank takes negative dependence (theta < 0) for two series. Each family has
its density, CDF, h-functions (conditional CDFs) and their inverses, Kendall's tau as a function of the parameter and
back, and its tail dependence:

    family   parameter              Kendall's tau                         lower tail       upper tail
    clayton  theta > 0              theta / (theta + 2)                   2^(-1/theta)     0
    gumbel   theta >= 1             1 - 1/theta                           0                2 - 2^(1/theta)
    frank    theta != 0             1 - 4/theta (1 - D1(theta)), Debye    0                0
    joe      theta >= 1             1 - 4 sum_k 1/(k (theta k + 2)(theta (k-1) + 2))   0         2 - 2^(1/theta)
    bb1      theta > 0, delta >= 1  1 - 2 / (delta (theta + 2))          2^(-1/(theta delta))  2 - 2^(1/delta)
    bb7      theta >= 1, delta > 0  1 + 4 int phi/phi' (numerically)     2^(-1/delta)     2 - 2^(1/theta)
    bb6      theta, delta >= 1      1 + 4 int phi/phi' (numerically)     0                2 - 2^(1/(theta delta))
    bb8      theta >= 1, 0<delta<=1 1 + 4 int phi/phi' (numerically)     0                0 (2 - 2^(1/theta) at delta 1)
    tawn     theta >= 1, 0<psi<=1   int t(1-t) A''(t)/A(t) dt             0                psi + 1 - (psi^theta + 1)^(1/theta)

BB6's generator is phi(u) = (-log(1 - (1 - u)^theta))^delta, BB8's -log((1 - (1 - delta u)^theta) / (1 - (1 -
delta)^theta)); both are worked in log phi so their densities hold at large parameters near the corners. Tawn's
copula is exp(-(x + y) A(y / (x + y))), x = -log u, y = -log v, with Pickands' function A(t) = (1 - psi1) t +
(1 - psi2)(1 - t) + ((psi1 t)^theta + (psi2 (1 - t))^theta)^(1/theta): type 1 has psi1 = 1, type 2 psi2 = 1 (each the
other with its arguments swapped); its h-function is C l_x / u, l = -log C. The survival copula swaps the tails; the 90
and 270 degree rotations negate tau and put the tail dependence in the off-diagonal corners (for Tawn's, as in
VineCopula, the density turned by 90 or 270 degrees: the other type reflected). The Archimedean densities in d dimensions are psi^(d)(sum phi(u_i)) prod phi'(u_i), with the
generator's derivatives in closed form (Hofert, Maechler and McNeil 2012: Gumbel's by a recursion whose terms are
all positive, Frank's by the polylogarithm of negative order, Joe's by Stirling numbers of the second kind).

**Simulation** is Marshall-Olkin for the Archimedean families: U_i = psi(E_i / V) with the frailty V whose Laplace
transform is the generator psi: gamma (Clayton), positive stable (Gumbel, Chambers-Mallows-Stuck), logarithmic series
(Frank), Sibuya (Joe), gamma to the power delta times a positive stable (BB1), gamma with a Sibuya shape (BB7). The
bivariate Frank (either sign) is drawn by conditional inversion of its h-function.

**Margins.** Each series' returns, or with ``garch=True`` its GARCH(1,1) standardised residuals, are made uniform
either by their ranks, u = rank / (T + 1) (``margins="empirical"``, canonical maximum likelihood, Genest, Ghoudi and
Rivest 1995), or by a fitted distribution (``"normal"``, ``"t"``, ``"skewt"``: Hansen's 1994 skewed t, standardised
to mean 0 and variance 1 and placed by a location and scale). With GARCH, ``"normal"`` and ``"t"`` are the GARCH's own
errors (estimated with it, the t's degrees of freedom included); ``"skewt"`` is fitted to the standardised residuals.

**Estimation.** ``method="ml"``: maximum likelihood of the copula on the uniforms, by inference functions for margins
(IFM, Joe and Xu 1996) when the margins are parametric (margins first, then the copula on their probability
transforms); ``"itau"``: the parameter from Kendall's tau (one-parameter families; the Gaussian and t correlations
sin(pi tau / 2)); ``"joint"``: full maximum likelihood of the margins and the copula together, from the IFM values
(parametric margins, without GARCH). The t copula's correlations are from Kendall's tau under ``ml`` too, its degrees
of freedom by maximum likelihood. Standard errors are from the observed information (a numerical Hessian) of the
parameters estimated by maximum likelihood: under IFM or with rank margins they treat the uniforms as known and so
ignore the margins' estimation (no Godambe sandwich); under ``joint`` they are from the joint Hessian of margins and
copula. Every fit has its log likelihood, AIC and BIC (of the copula; with parametric margins also the total).

**Goodness of fit**: the Cramer-von Mises distance between the empirical copula and the fitted one at the
pseudo-observations, S_n = sum_i (C_n(U_i) - C_theta(U_i))^2, with a parametric bootstrap p-value (Genest, Remillard
and Beaudoin 2009): samples from the fitted copula, each turned into pseudo-observations and refitted the same way,
p = (#{S_b >= S_n} + 1/2) / (B + 1). It is on the ranks whatever the margins. The bivariate Gaussian and t CDFs are by
Gauss-Legendre quadrature of the h-function (split where it turns, nodes clustered at the ends: about 1e-8); for more
than two series by Genz's (1992) separation of variables averaged over a fixed scrambled Sobol sample of
``QMC_POINTS`` (about 1e-5), so the test is the same parametric bootstrap in any dimension. ``bootstrap="multiplier"``
is Kojadinovic and Yan's (2011) multiplier bootstrap instead, as the copula package's gofCopula(..., simulation =
"mult"): no refitting, the empirical copula process replicated with standard normal multipliers, the empirical
copula's partial derivatives by finite differences and the estimator's influence (maximum pseudo-likelihood's with
its rank correction, or Kendall's tau's for two series) by numerical derivatives (``multiplier_test``); not for the t
copula, whose correlations are from Kendall's tau and degrees of freedom by maximum likelihood.

Vine copulas (pair-copula constructions of these bivariate families) are in ``vines``; Patton's time-varying
copulas and the DCC copula of several series in ``tvcopulas``.

**Portfolio risk.** Draws from the copula are mapped to returns through each margin: the fitted margins, the
empirical quantiles of the returns or residuals (``margins="empirical"``) or a normal fitted to them (``"normal"``);
with GARCH margins the residuals are scaled by each series' variance forecast for the next period. VaR and ES are of
the weighted sum.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import integrate, optimize
from scipy import stats as sps
from scipy.special import digamma, gammaln, logsumexp, ndtr, ndtri

from pymetrix.extension import api
from pymetrix.errors import SpecificationError
from pymetrix.output import Table
from pymetrix.extension import plugin_context
from . import PLUGIN
from pymetrix.extension import random_number_generator as make_rng

if TYPE_CHECKING:
    from pymetrix.data import Dataset
    from pymetrix.figures import Figure

ARCHIMEDEAN = ("clayton", "gumbel", "frank", "joe")  # any number of series (exchangeable)
ROTATABLE = ("clayton", "gumbel", "joe")
ROTATABLE_2 = ("bb1", "bb6", "bb7", "bb8", "tawn1", "tawn2")  # two-parameter families, rotated as name_180 etc.
EXTRA_2 = ("bb6", "bb8", "tawn1", "tawn2")  # added later: not among the families compared by default


def rotated_name(family: str, angle: int) -> str:
    """clayton180, ..., and bb1_180, tawn1_90, ... (an underscore where the family's name ends in a digit)."""
    return f"{family}{'_' if family[-1].isdigit() else ''}{angle}"


FAMILIES = ("gaussian", "t", "clayton", "gumbel", "frank", "joe", "bb1", "bb7",
            *(f"{f}{a}" for a in (180, 90, 270) for f in ROTATABLE), *EXTRA_2,
            *(rotated_name(f, a) for a in (180, 90, 270) for f in ROTATABLE_2))
MARGINS = ("empirical", "normal", "t", "skewt")
METHODS = ("ml", "itau", "joint")
Family = Literal["gaussian", "t", "clayton", "gumbel", "frank", "joe", "bb1", "bb7", "clayton180", "gumbel180", "joe180",
                 "clayton90", "gumbel90", "joe90", "clayton270", "gumbel270", "joe270", "bb6", "bb8", "tawn1", "tawn2",
                 "bb1_180", "bb6_180", "bb7_180", "bb8_180", "tawn1_180", "tawn2_180", "bb1_90", "bb6_90", "bb7_90", "bb8_90",
                 "tawn1_90", "tawn2_90", "bb1_270", "bb6_270", "bb7_270", "bb8_270", "tawn1_270", "tawn2_270"]
_EPS = 1e-12


def pseudo_observations(x: np.ndarray) -> np.ndarray:
    """Each column's ranks over (T + 1): uniform margins without assuming their shape."""
    return np.column_stack([sps.rankdata(c) / (len(c) + 1) for c in np.asarray(x, float).T])


# -- the elliptical copulas --------------------------------------------------------------------------------------------

def gaussian_logpdf(u: np.ndarray, R: np.ndarray) -> np.ndarray:
    z = sps.norm.ppf(u)
    sign, logdet = np.linalg.slogdet(R)
    q = np.einsum("ij,jk,ik->i", z, np.linalg.inv(R) - np.eye(len(R)), z)
    return -0.5 * logdet - 0.5 * q


def t_logpdf(u: np.ndarray, R: np.ndarray, nu: float) -> np.ndarray:
    d = R.shape[0]
    x = sps.t.ppf(u, nu)
    _, logdet = np.linalg.slogdet(R)
    q = np.einsum("ij,jk,ik->i", x, np.linalg.inv(R), x)
    joint = gammaln((nu + d) / 2) - gammaln(nu / 2) - d / 2 * np.log(nu * np.pi) - 0.5 * logdet - (nu + d) / 2 * np.log1p(q / nu)
    return joint - sps.t.logpdf(x, nu).sum(axis=1)


def _elliptical_h(u: np.ndarray, v: np.ndarray, rho: float, nu: float | None) -> np.ndarray:
    """P(V <= v | U = u) for the bivariate Gaussian (``nu`` None) or t copula."""
    if nu is None:
        return sps.norm.cdf((sps.norm.ppf(v) - rho * sps.norm.ppf(u)) / np.sqrt(1 - rho**2))
    xu, xv = sps.t.ppf(u, nu), sps.t.ppf(v, nu)
    return sps.t.cdf((xv - rho * xu) / np.sqrt((nu + xu**2) * (1 - rho**2) / (nu + 1)), nu + 1)


def _elliptical_hinv(u: np.ndarray, w: np.ndarray, rho: float, nu: float | None) -> np.ndarray:
    if nu is None:
        return sps.norm.cdf(sps.norm.ppf(w) * np.sqrt(1 - rho**2) + rho * sps.norm.ppf(u))
    xu = sps.t.ppf(u, nu)
    return sps.t.cdf(sps.t.ppf(w, nu + 1) * np.sqrt((nu + xu**2) * (1 - rho**2) / (nu + 1)) + rho * xu, nu)


@functools.lru_cache(maxsize=None)
def _nodes(n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    s = (x + 1) / 2
    return (1 - np.cos(np.pi * s)) / 2, w / 2 * np.pi / 2 * np.sin(np.pi * s)  # nodes clustered at both ends


def _elliptical_cdf(u: np.ndarray, v: np.ndarray, rho: float, nu: float | None) -> np.ndarray:
    """C(u, v) = int_0^u h(v | s) ds by Gauss-Legendre, split where h turns (s* = F(F^-1(v) / rho))."""
    u, v = np.clip(u, _EPS, 1 - _EPS), np.clip(v, _EPS, 1 - _EPS)
    if abs(rho) < 1e-12:
        return u * v
    xv = sps.norm.ppf(v) if nu is None else sps.t.ppf(v, nu)
    mid = sps.norm.cdf(xv / rho) if nu is None else sps.t.cdf(xv / rho, nu)
    mid = np.minimum(np.maximum(mid, 0.0), u)
    t, w = _nodes()
    total = np.zeros_like(u)
    for a, b in ((np.zeros_like(u), mid), (mid, u)):
        p = np.clip(a[:, None] + (b - a)[:, None] * t[None, :], 1e-300, 1 - _EPS)
        total += (b - a) * (_elliptical_h(p, np.broadcast_to(v[:, None], p.shape), rho, nu) * w).sum(axis=1)
    return total


QMC_POINTS = 1024  # the quasi-Monte Carlo sample of the Gaussian and t copula CDFs in more than two dimensions


@functools.lru_cache(maxsize=16)
def _qmc(n: int, dim: int) -> np.ndarray:
    """A scrambled Sobol sample, fixed (seeded) so the CDF is a smooth deterministic function of its arguments."""
    from scipy.stats import qmc

    return qmc.Sobol(dim, scramble=True, seed=20261031).random(n)


def elliptical_cdf_d(u: np.ndarray, R: np.ndarray, nu: float | None = None, n_points: int | None = None) -> np.ndarray:
    """The Gaussian (``nu`` None) or t copula's CDF at the rows of ``u`` in any dimension: Genz's (1992) separation
    of variables, P(X <= b) = E[e_1 e_2 ... e_d] with e_i = Phi((b_i - sum_j<i L_ij y_j) / L_ii) and
    y_j = Phi^-1(w_j e_j), averaged over a scrambled Sobol sample of w (the t as a normal scale mixture, its
    chi-square one more dimension of the sample)."""
    u = np.clip(np.asarray(u, float), _EPS, 1 - _EPS)
    n, d = u.shape
    L = np.linalg.cholesky(R)
    b = sps.norm.ppf(u) if nu is None else sps.t.ppf(u, nu)
    N = int(n_points or QMC_POINTS)
    W = _qmc(N, d - 1 + (nu is not None))
    s = np.ones(N) if nu is None else np.sqrt(sps.chi2.ppf(np.clip(W[:, -1], 1e-15, 1 - 1e-15), nu) / nu)
    out = np.empty(n)
    step = max(1, 2_000_000 // (N * d))
    for lo in range(0, n, step):
        bs = b[lo:lo + step, None, :] * s[None, :, None]
        e = ndtr(bs[..., 0] / L[0, 0])
        f = e.copy()
        acc = [np.zeros(bs.shape[:2]) for _ in range(d)]  # sum_j<i L_ij y_j, built as each y_j comes
        for i in range(1, d):
            y = ndtri(np.clip(W[None, :, i - 1] * e, 1e-300, 1 - 1e-16))
            for k in range(i, d):
                acc[k] += L[k, i - 1] * y
            e = ndtr((bs[..., i] - acc[i]) / L[i, i])
            f *= e
        out[lo:lo + step] = f.mean(axis=1)
    return out


def _kendall_corr(u: np.ndarray) -> np.ndarray:
    d = u.shape[1]
    R = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            R[i, j] = R[j, i] = np.sin(np.pi / 2 * sps.kendalltau(u[:, i], u[:, j])[0])
    w, V = np.linalg.eigh(R)  # the nearest correlation matrix if the pairwise values are not one
    if w.min() <= 1e-8:
        R = V @ np.diag(np.maximum(w, 1e-6)) @ V.T
        s = np.sqrt(np.diag(R))
        R = R / np.outer(s, s)
    return R


# -- the Archimedean copulas -------------------------------------------------------------------------------------------

def clayton_logpdf(u: np.ndarray, theta: float) -> np.ndarray:
    a, b = u[:, 0], u[:, 1]
    s = a ** -theta + b ** -theta - 1
    return np.log1p(theta) - (theta + 1) * (np.log(a) + np.log(b)) - (2 + 1 / theta) * np.log(s)


def gumbel_logpdf(u: np.ndarray, theta: float) -> np.ndarray:
    x, y = -np.log(u[:, 0]), -np.log(u[:, 1])
    A = (x**theta + y**theta) ** (1 / theta)
    return (-A + (theta - 1) * (np.log(x) + np.log(y)) + x + y + (1 - 2 * theta) / theta * np.log(x**theta + y**theta)
            + np.log(A + theta - 1))


@functools.lru_cache(maxsize=None)
def _log_stirling2(n: int) -> np.ndarray:
    """log S(n, k), k = 0..n, Stirling numbers of the second kind (exact integers, then logs; -inf for zero)."""
    row = [1]
    for m in range(1, n + 1):
        row = [0] + [k * (row[k] if k < len(row) else 0) + row[k - 1] for k in range(1, m + 1)]
    return np.array([math.log(s) if s > 0 else -np.inf for s in row])


def _positive_stable(a: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Positive stable draws with Laplace transform exp(-t^a), 0 < a <= 1 (Chambers-Mallows-Stuck)."""
    w, e = rng.uniform(0, np.pi, (n, 1)), rng.exponential(1.0, (n, 1))
    return (np.sin(a * w) / np.sin(w) ** (1 / a)) * (np.sin((1 - a) * w) / e) ** ((1 - a) / a) if a < 1 else np.ones((n, 1))


def _sibuya(a: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sibuya draws, P(V = k) = (-1)^(k+1) binom(a, k), by inverting P(V > k) = Gamma(k + 1 - a) / (k! Gamma(1 - a))
    from the approximation ((1 - U) Gamma(1 - a))^(-1/a) (Hofert 2011), corrected exactly below a million (above,
    the approximation's ceiling: its error there is beneath the log-gamma function's rounding)."""
    u = rng.uniform(size=n)
    out = np.ones(n)
    big = u > a
    if a >= 1 or not big.any():
        return out[:, None]
    tail = 1 - u[big]
    g = (tail * math.gamma(1 - a)) ** (-1 / a)
    k = np.where(g < 1e6, np.maximum(np.floor(g), 1.0), np.ceil(np.minimum(g, 1e300)))
    lt = np.log(tail)

    def log_sf(k: np.ndarray) -> np.ndarray:
        return gammaln(k + 1 - a) - gammaln(k + 1) - gammaln(1 - a)

    active = np.flatnonzero(g < 1e6)
    for _ in range(100):  # the smallest k with P(V > k) <= 1 - U
        if not len(active):
            break
        ka, la = k[active], lt[active]
        up = log_sf(ka) > la
        down = (ka > 1) & (log_sf(np.maximum(ka - 1, 1)) <= la)
        k[active] = ka + up - down
        active = active[up | down]
    out[big] = k
    return out[:, None]


class Archimedean:
    """An Archimedean copula C(u) = psi(sum phi(u_i)): psi the generator (a Laplace transform), phi its inverse."""

    name = ""
    names: tuple[str, ...] = ("theta",)
    any_dimension = True

    def bounds(self, d: int) -> list[tuple[float, float]]:
        raise NotImplementedError

    # the generator, its inverse, log |phi'| and log (-1)^k psi^(k)
    def psi(self, t: np.ndarray, p: Sequence[float]) -> np.ndarray:
        raise NotImplementedError

    def phi(self, u: np.ndarray, p: Sequence[float]) -> np.ndarray:
        raise NotImplementedError

    def log_dphi(self, u: np.ndarray, p: Sequence[float]) -> np.ndarray:
        raise NotImplementedError

    def log_dpsi(self, t: np.ndarray, p: Sequence[float], k: int) -> np.ndarray:
        raise NotImplementedError

    def frailty(self, n: int, p: Sequence[float], rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def tau(self, p: Sequence[float]) -> float:
        """Kendall's tau, 1 + 4 int_0^1 phi(u) / phi'(u) du (numerically; families override with closed forms)."""
        def ratio(u: float) -> float:
            a = np.array([u])
            return float(-self.phi(a, p)[0] / np.exp(self.log_dphi(a, p)[0]))

        return float(1 + 4 * integrate.quad(ratio, 0, 1, limit=200, epsabs=1e-11)[0])

    def tail(self, p: Sequence[float]) -> tuple[float, float]:
        raise NotImplementedError

    def logpdf(self, u: np.ndarray, p: Sequence[float]) -> np.ndarray:
        return self.log_dpsi(self.phi(u, p).sum(axis=1), p, u.shape[1]) + self.log_dphi(u, p).sum(axis=1)

    def cdf(self, u: np.ndarray, p: Sequence[float]) -> np.ndarray:
        return self.psi(self.phi(u, p).sum(axis=1), p)

    def h(self, u: np.ndarray, v: np.ndarray, p: Sequence[float]) -> np.ndarray:
        """P(V <= v | U = u) = psi'(phi(u) + phi(v)) / psi'(phi(u))."""
        u, v = np.clip(u, _EPS, 1 - _EPS), np.clip(v, _EPS, 1 - _EPS)
        a = self.phi(u[:, None], p)[:, 0]
        b = self.phi(v[:, None], p)[:, 0]
        return np.exp(self.log_dpsi(a + b, p, 1) - self.log_dpsi(a, p, 1))

    def hinv(self, u: np.ndarray, w: np.ndarray, p: Sequence[float]) -> np.ndarray:
        """The v with h(u, v) = w, by bisection (h rises in v)."""
        lo, hi = np.zeros_like(w, dtype=float), np.ones_like(w, dtype=float)
        for _ in range(60):
            mid = (lo + hi) / 2
            below = self.h(u, mid, p) < w
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        return (lo + hi) / 2

    def sample(self, n: int, d: int, p: Sequence[float], rng: np.random.Generator) -> np.ndarray:
        v = self.frailty(n, p, rng)
        return self.psi(rng.exponential(1.0, (n, d)) / v, p)


class Clayton(Archimedean):
    name = "clayton"

    def bounds(self, d: int) -> list[tuple[float, float]]:
        return [(1e-4, 50.0)]

    def psi(self, t, p):
        return (1 + t) ** (-1 / p[0])

    def phi(self, u, p):
        return np.expm1(-p[0] * np.log(u))

    def log_dphi(self, u, p):
        return np.log(p[0]) - (p[0] + 1) * np.log(u)

    def log_dpsi(self, t, p, k):
        a = 1 / p[0]
        return sum(np.log(a + j) for j in range(k)) - (a + k) * np.log1p(t)

    def logpdf(self, u, p):
        return clayton_logpdf(u, p[0]) if u.shape[1] == 2 else super().logpdf(u, p)

    def frailty(self, n, p, rng):
        return rng.gamma(1 / p[0], 1.0, (n, 1))

    def tau(self, p):
        return p[0] / (p[0] + 2)

    def tail(self, p):
        return 2 ** (-1 / p[0]), 0.0


class Gumbel(Archimedean):
    name = "gumbel"

    def bounds(self, d):
        return [(1.0 + 1e-6, 50.0)]

    def psi(self, t, p):
        return np.exp(-t ** (1 / p[0]))

    def phi(self, u, p):
        return (-np.log(u)) ** p[0]

    def log_dphi(self, u, p):
        x = -np.log(u)
        return np.log(p[0]) + (p[0] - 1) * np.log(x) - np.log(u)

    def log_dpsi(self, t, p, k):
        # (-1)^k psi^(k)(t) = exp(-t^a) sum_j c_kj t^(a j - k): c_(k+1),j = a c_k,(j-1) + (k - a j) c_k,j, all >= 0
        a = 1 / p[0]
        c = [1.0]
        for m in range(k):
            c = [(a * c[j - 1] if j >= 1 else 0.0) + ((m - a * j) * c[j] if j < len(c) else 0.0) for j in range(m + 2)]
        lt = np.log(t)
        terms = np.stack([np.log(c[j]) + a * j * lt for j in range(1, k + 1) if c[j] > 0])
        return -t**a - k * lt + logsumexp(terms, axis=0)

    def logpdf(self, u, p):
        return gumbel_logpdf(u, p[0]) if u.shape[1] == 2 else super().logpdf(u, p)

    def frailty(self, n, p, rng):
        return _positive_stable(1 / p[0], n, rng)

    def tau(self, p):
        return 1 - 1 / p[0]

    def tail(self, p):
        return 0.0, 2 - 2 ** (1 / p[0])


def debye1(x: float) -> float:
    """The Debye function D1(x) = (1/x) int_0^x t / (e^t - 1) dt."""
    if abs(x) < 1e-10:
        return 1.0
    return float(integrate.quad(lambda t: t / np.expm1(t) if t != 0 else 1.0, 0, x, epsabs=1e-13, epsrel=1e-12)[0] / x)


class Frank(Archimedean):
    name = "frank"

    def bounds(self, d):
        return [(-50.0, 50.0)] if d == 2 else [(1e-4, 35.0)]  # the logarithmic-series frailty needs 1 - e^-theta < 1

    def psi(self, t, p):
        return -np.log1p(np.expm1(-p[0]) * np.exp(-t)) / p[0]

    def phi(self, u, p):
        return -np.log(np.expm1(-p[0] * u) / np.expm1(-p[0]))

    def log_dphi(self, u, p):
        return np.log(abs(p[0])) - np.log(np.abs(np.expm1(p[0] * u)))

    def log_dpsi(self, t, p, k):
        # (-1)^k psi^(k)(t) = Li_(1-k)(z) / theta, z = (1 - e^-theta) e^-t; Li_-n(z) = sum_j j! S(n+1, j+1) (z/(1-z))^(j+1)
        th = p[0]
        if th <= 0:
            raise SpecificationError("Frank's generator derivatives here are for theta > 0")
        lz = np.log(-np.expm1(-th)) - t
        lx = lz - np.log1p(-np.exp(lz))
        ls = _log_stirling2(k)
        terms = np.stack([gammaln(j + 1) + ls[j + 1] + (j + 1) * lx for j in range(k)])
        return -np.log(th) + logsumexp(terms, axis=0)

    # the bivariate density, CDF and h-function in closed form, for either sign of theta
    def logpdf(self, u, p):
        if u.shape[1] != 2:
            return super().logpdf(u, p)
        th = p[0]
        a, au, av = -np.expm1(-th), -np.expm1(-th * u[:, 0]), -np.expm1(-th * u[:, 1])
        return np.log(th * a) - th * (u[:, 0] + u[:, 1]) - 2 * np.log(np.abs(a - au * av))

    def cdf(self, u, p):
        if u.shape[1] != 2:
            return super().cdf(u, p)
        th = p[0]
        return -np.log1p(np.expm1(-th * u[:, 0]) * np.expm1(-th * u[:, 1]) / np.expm1(-th)) / th

    def h(self, u, v, p):
        th = p[0]
        return np.exp(-th * u) * np.expm1(-th * v) / (np.expm1(-th) + np.expm1(-th * u) * np.expm1(-th * v))

    def hinv(self, u, w, p):
        th = p[0]
        return -np.log1p(w * np.expm1(-th) / (w + (1 - w) * np.exp(-th * u))) / th

    def frailty(self, n, p, rng):
        return rng.logseries(-np.expm1(-p[0]), (n, 1)).astype(float)

    def sample(self, n, d, p, rng):
        if d == 2:  # conditional inversion: either sign
            u, w = rng.uniform(size=n), rng.uniform(size=n)
            return np.column_stack([u, self.hinv(u, w, p)])
        return super().sample(n, d, p, rng)

    def tau(self, p):
        th = p[0]
        return 0.0 if th == 0 else 1 - 4 / th * (1 - debye1(th))

    def tail(self, p):
        return 0.0, 0.0


class Joe(Archimedean):
    name = "joe"

    def bounds(self, d):
        return [(1.0 + 1e-6, 50.0)]

    def psi(self, t, p):
        return -np.expm1(np.log(-np.expm1(-t)) / p[0])

    def phi(self, u, p):
        return -np.log1p(-np.exp(p[0] * np.log1p(-u)))

    def log_dphi(self, u, p):
        return np.log(p[0]) + (p[0] - 1) * np.log1p(-u) - np.log1p(-np.exp(p[0] * np.log1p(-u)))

    def log_dpsi(self, t, p, k):
        # (-1)^k psi^(k)(t) = a (1 - z)^a sum_j S(k, j) Gamma(j - a) / Gamma(1 - a) (z / (1 - z))^j, z = e^-t, a = 1/theta
        a = 1 / p[0]
        l1z = np.log(-np.expm1(-t))
        lx = -t - l1z
        ls = _log_stirling2(k)
        terms = np.stack([ls[j] + gammaln(j - a) - gammaln(1 - a) + j * lx for j in range(1, k + 1)])
        return np.log(a) + a * l1z + logsumexp(terms, axis=0)

    def frailty(self, n, p, rng):
        return _sibuya(1 / p[0], n, rng)

    def tau(self, p):
        th = p[0]
        if abs(th - 2) < 1e-6:
            return 0.5 * (self.tau([2 - 1e-5]) + self.tau([2 + 1e-5]))
        return float(1 + 2 / (2 - th) * (digamma(2) - digamma(2 / th + 1)))

    def tail(self, p):
        return 0.0, 2 - 2 ** (1 / p[0])


class BB1(Archimedean):
    """Clayton-Gumbel: psi(t) = (1 + t^(1/delta))^(-1/theta)."""

    name = "bb1"
    names = ("theta", "delta")
    any_dimension = False

    def bounds(self, d):
        return [(1e-4, 20.0), (1.0, 20.0)]

    def psi(self, t, p):
        return (1 + t ** (1 / p[1])) ** (-1 / p[0])

    def phi(self, u, p):
        return np.expm1(-p[0] * np.log(u)) ** p[1]

    def log_dphi(self, u, p):
        th, de = p
        return np.log(th) + np.log(de) + (de - 1) * np.log(np.expm1(-th * np.log(u))) - (th + 1) * np.log(u)

    def log_dpsi(self, t, p, k):
        th, de = p
        s = t ** (1 / de)
        lead = -np.log(th) - (1 / th + 1) * np.log1p(s)
        if k == 1:
            return lead - np.log(de) + (1 / de - 1) * np.log(t)
        if k == 2:
            first = (1 / th + 1) * np.exp(2 * ((1 / de - 1) * np.log(t) - np.log(de)) - np.log1p(s))
            second = (1 / de) * (1 - 1 / de) * t ** (1 / de - 2)
            return lead + np.log(first + second)
        raise SpecificationError("BB1 is for two series")

    def frailty(self, n, p, rng):
        th, de = p
        return rng.gamma(1 / th, 1.0, (n, 1)) ** de * _positive_stable(1 / de, n, rng)

    def tau(self, p):
        return 1 - 2 / (p[1] * (p[0] + 2))

    def tail(self, p):
        return 2 ** (-1 / (p[0] * p[1])), 2 - 2 ** (1 / p[1])


class BB7(Archimedean):
    """Joe-Clayton: psi(t) = 1 - (1 - (1 + t)^(-1/delta))^(1/theta)."""

    name = "bb7"
    names = ("theta", "delta")
    any_dimension = False

    def bounds(self, d):
        return [(1.0, 20.0), (1e-4, 20.0)]

    def psi(self, t, p):
        th, de = p
        return -np.expm1(np.log(-np.expm1(-np.log1p(t) / de)) / th)

    def phi(self, u, p):
        th, de = p
        return np.expm1(-de * np.log1p(-np.exp(th * np.log1p(-u))))

    def log_dphi(self, u, p):
        th, de = p
        return np.log(de) + np.log(th) - (de + 1) * np.log1p(-np.exp(th * np.log1p(-u))) + (th - 1) * np.log1p(-u)

    def log_dpsi(self, t, p, k):
        th, de = p
        a = 1 / th
        lz = -np.log1p(t) / de
        l1z = np.log(-np.expm1(lz))
        x = np.exp(lz - l1z)
        lead = np.log(a) + a * l1z - np.log(de) - np.log1p(t)
        if k == 1:
            return lead + np.log(x)
        if k == 2:
            return lead - np.log(de) - np.log1p(t) + np.log(x + (1 - a) * x**2 + de * x)
        raise SpecificationError("BB7 is for two series")

    def frailty(self, n, p, rng):
        th, de = p
        return rng.gamma(_sibuya(1 / th, n, rng) / de, 1.0)

    def tail(self, p):
        return 2 ** (-1 / p[1]), 2 - 2 ** (1 / p[0])


class _ConditionalSample:
    """Bivariate draws by conditional inversion: u uniform, v = h^-1(u, w) for w uniform."""

    def sample(self, n, d, p, rng):
        u, w = rng.uniform(size=n), rng.uniform(size=n)
        return np.column_stack([u, self.hinv(u, w, p)])


class _LogGenerator(Archimedean):
    """Bivariate Archimedean copulas worked in log phi and log t (t = phi(u) + phi(v)), which underflow otherwise at
    large parameters near the corners: the density, CDF and h-function from log_phi and log_dpsi_lt."""

    any_dimension = False

    def log_phi(self, u: np.ndarray, p: Sequence[float]) -> np.ndarray:
        raise NotImplementedError

    def log_dpsi_lt(self, lt: np.ndarray, p: Sequence[float], k: int) -> np.ndarray:
        raise NotImplementedError

    def psi_lt(self, lt: np.ndarray, p: Sequence[float]) -> np.ndarray:
        raise NotImplementedError

    def phi(self, u, p):
        return np.exp(self.log_phi(u, p))

    def psi(self, t, p):
        with np.errstate(divide="ignore"):
            return self.psi_lt(np.log(t), p)

    def log_dpsi(self, t, p, k):
        with np.errstate(divide="ignore"):
            return self.log_dpsi_lt(np.log(t), p, k)

    def logpdf(self, u, p):
        lt = np.logaddexp(self.log_phi(u[:, 0], p), self.log_phi(u[:, 1], p))
        return self.log_dpsi_lt(lt, p, 2) + self.log_dphi(u[:, 0], p) + self.log_dphi(u[:, 1], p)

    def cdf(self, u, p):
        return self.psi_lt(np.logaddexp(self.log_phi(u[:, 0], p), self.log_phi(u[:, 1], p)), p)

    def h(self, u, v, p):
        u, v = np.clip(u, _EPS, 1 - _EPS), np.clip(v, _EPS, 1 - _EPS)
        a = self.log_phi(u, p)
        return np.exp(self.log_dpsi_lt(np.logaddexp(a, self.log_phi(v, p)), p, 1) - self.log_dpsi_lt(a, p, 1))


def _log_one_minus_exp_minus(t: np.ndarray, lt: np.ndarray) -> np.ndarray:
    """log(1 - e^-t) from t and log t, accurate as t -> 0 (then log t - t/2)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t > 1e-8, np.log(-np.expm1(-np.maximum(t, 1e-300))), lt - t / 2)


class BB6(_ConditionalSample, _LogGenerator):
    """Joe-Gumbel: psi(t) = 1 - (1 - exp(-t^(1/delta)))^(1/theta), phi(u) = (-log(1 - (1 - u)^theta))^delta."""

    name = "bb6"
    names = ("theta", "delta")

    def bounds(self, d):
        return [(1.0, 20.0), (1.0, 20.0)]

    @staticmethod
    def _lL(u, th):
        """log L, L = -log(1 - (1 - u)^theta)."""
        x = th * np.log1p(-u)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(x > -30, np.log(-np.log1p(-np.exp(np.minimum(x, -1e-300)))), x + np.exp(x) / 2)

    def log_phi(self, u, p):
        return p[1] * self._lL(u, p[0])

    def log_dphi(self, u, p):
        th, de = p
        return (np.log(th) + np.log(de) + (de - 1) * self._lL(u, th) + (th - 1) * np.log1p(-u)
                - np.log1p(-np.exp(th * np.log1p(-u))))

    def psi_lt(self, lt, p):
        th, de = p
        s = np.exp(lt / de)
        return -np.expm1(_log_one_minus_exp_minus(s, lt / de) / th)

    def log_dpsi_lt(self, lt, p, k):
        # psi = 1 - w^a, w = 1 - e^-s, s = t^(1/delta), a = 1/theta: -psi' = a w^(a-1) e^-s s',
        # psi'' = a w^(a-1) e^-s s'^2 (1 + (1 - a) e^-s / w + (delta - 1) / s), s' = t^(1/delta - 1) / delta
        th, de = p
        a = 1 / th
        ls = lt / de
        s = np.exp(ls)
        lw = _log_one_minus_exp_minus(s, ls)
        ls1 = -np.log(de) + (1 / de - 1) * lt
        lead = np.log(a) + (a - 1) * lw - s + ls1
        if k == 1:
            return lead
        if k == 2:
            with np.errstate(divide="ignore"):
                extra = np.logaddexp(np.logaddexp(0.0, np.log(1 - a) - s - lw), np.log(de - 1) - ls)
            return lead + ls1 + extra
        raise SpecificationError("BB6 is for two series")

    def tail(self, p):
        return 0.0, 2 - 2 ** (1 / (p[0] * p[1]))


class BB8(_ConditionalSample, _LogGenerator):
    """Joe-Frank: psi(t) = (1 - (1 - eta e^-t)^(1/theta)) / delta, eta = 1 - (1 - delta)^theta;
    phi(u) = -log((1 - (1 - delta u)^theta) / eta)."""

    name = "bb8"
    names = ("theta", "delta")

    def bounds(self, d):
        return [(1.0, 20.0), (1e-4, 1.0)]

    @staticmethod
    def _log_eta(th: float, de: float) -> float:
        return float(np.log(-np.expm1(th * np.log1p(-de)))) if de < 1 else 0.0

    def log_phi(self, u, p):
        # phi = -log(1 - D / eta), D = (1 - delta u)^theta - (1 - delta)^theta, without the cancellation
        th, de = p
        with np.errstate(divide="ignore", invalid="ignore"):
            if de < 1:
                lD = th * np.log1p(-de) + np.log(np.expm1(th * (np.log1p(-de * u) - np.log1p(-de))))
            else:
                lD = th * np.log1p(-u)
            lr = lD - self._log_eta(th, de)
            r = np.exp(lr)
            return np.where(r > 1e-10, np.log(-np.log1p(-np.minimum(r, 1 - 1e-16))), lr + r / 2)

    def log_dphi(self, u, p):
        th, de = p
        return np.log(th) + np.log(de) + (th - 1) * np.log1p(-de * u) - np.log(-np.expm1(th * np.log1p(-de * u)))

    def _lw(self, lt, p):
        """log w, w = 1 - eta e^-t = (1 - delta)^theta + eta (1 - e^-t)."""
        th, de = p
        with np.errstate(divide="ignore"):
            first = th * np.log1p(-de) if de < 1 else -np.inf
        return np.logaddexp(first, self._log_eta(th, de) + _log_one_minus_exp_minus(np.exp(lt), lt))

    def psi_lt(self, lt, p):
        th, de = p
        return -np.expm1(self._lw(lt, p) / th) / de

    def log_dpsi_lt(self, lt, p, k):
        # z = eta e^-t, w = 1 - z, a = 1/theta: -psi' = (a/delta) w^(a-1) z, psi'' = (a/delta) w^(a-1) z (1 + (1-a) z/w)
        th, de = p
        a = 1 / th
        lz = self._log_eta(th, de) - np.exp(lt)
        lw = self._lw(lt, p)
        lead = np.log(a / de) + (a - 1) * lw + lz
        if k == 1:
            return lead
        if k == 2:
            return lead + np.log1p((1 - a) * np.exp(lz - lw))
        raise SpecificationError("BB8 is for two series")

    def tail(self, p):
        return 0.0, (2 - 2 ** (1 / p[0])) if p[1] >= 1 else 0.0


class Tawn:
    """Tawn's (1988) asymmetric extreme-value copula, C(u, v) = exp(-l(x, y)), x = -log u, y = -log v, with
    l(x, y) = (1 - psi1) y + (1 - psi2) x + ((psi1 y)^theta + (psi2 x)^theta)^(1/theta) (Pickands' A(t) =
    (1 - psi1) t + (1 - psi2)(1 - t) + ((psi1 t)^theta + (psi2 (1 - t))^theta)^(1/theta), t = y / (x + y)): type 1
    has psi1 = 1 and psi2 = psi, type 2 psi2 = 1 and psi1 = psi (VineCopula's families 104 and 204). Each is the
    other with its arguments swapped."""

    names = ("theta", "psi")
    any_dimension = False

    def __init__(self, kind: int) -> None:
        self.kind = kind
        self.name = f"tawn{kind}"

    def bounds(self, d):
        return [(1.0, 20.0), (1e-4, 1.0)]

    def _psis(self, p) -> tuple[float, float]:
        return (1.0, p[1]) if self.kind == 1 else (p[1], 1.0)

    def _parts(self, u, v, p):
        th = p[0]
        p1, p2 = self._psis(p)
        x, y = -np.log(u), -np.log(v)
        ls = np.logaddexp(th * (np.log(p1) + np.log(y)), th * (np.log(p2) + np.log(x))) / th  # log S
        S = np.exp(ls)
        ell = (1 - p1) * y + (1 - p2) * x + S
        lx = (1 - p2) + np.exp(th * np.log(p2) + (th - 1) * (np.log(x) - ls))
        ly = (1 - p1) + np.exp(th * np.log(p1) + (th - 1) * (np.log(y) - ls))
        mlxy = (th - 1) * np.exp(th * (np.log(p1) + np.log(p2)) + (th - 1) * (np.log(x) + np.log(y)) + (1 - 2 * th) * ls)
        return x, y, ell, lx, ly, mlxy

    def logpdf(self, u, p):
        u = np.clip(u, _EPS, 1 - _EPS)
        x, y, ell, lx, ly, mlxy = self._parts(u[:, 0], u[:, 1], p)
        return -ell + x + y + np.log(lx * ly + mlxy)

    def cdf(self, u, p):
        u = np.clip(u, _EPS, 1 - _EPS)
        return np.exp(-self._parts(u[:, 0], u[:, 1], p)[2])

    def h(self, u, v, p):
        """P(V <= v | U = u) = dC/du = C(u, v) l_x / u."""
        u, v = np.clip(u, _EPS, 1 - _EPS), np.clip(v, _EPS, 1 - _EPS)
        x, _, ell, lx, _, _ = self._parts(u, v, p)
        return np.exp(-ell + x) * lx

    hinv = Archimedean.hinv
    sample = _ConditionalSample.sample

    def A(self, t, p):
        th = p[0]
        p1, p2 = self._psis(p)
        return (1 - p1) * t + (1 - p2) * (1 - t) + ((p1 * t) ** th + (p2 * (1 - t)) ** th) ** (1 / th)

    def tau(self, p):
        """int_0^1 t (1 - t) A''(t) / A(t) dt, A'' = (theta - 1) (psi1 psi2)^theta (t (1 - t))^(theta - 2) B^(1/theta - 2)."""
        th = p[0]
        if th <= 1:
            return 0.0
        p1, p2 = self._psis(p)

        def f(t: float) -> float:
            B = (p1 * t) ** th + (p2 * (1 - t)) ** th
            return (th - 1) * (p1 * p2) ** th * (t * (1 - t)) ** (th - 1) * B ** (1 / th - 2) / float(self.A(t, p))

        return float(integrate.quad(f, 0, 1, limit=200, epsabs=1e-13, epsrel=1e-12)[0])

    def tail(self, p):
        th = p[0]
        p1, p2 = self._psis(p)
        return 0.0, p1 + p2 - (p1**th + p2**th) ** (1 / th)


class Rotated:
    """A bivariate copula rotated by 90, 180 (the survival copula) or 270 degrees: the density c(1 - u, 1 - v),
    c(1 - u, v) or c(u, 1 - v)."""

    any_dimension = False

    def __init__(self, base: Any, angle: int, name: str | None = None) -> None:
        self.base, self.angle = base, angle
        self.name, self.names = name or rotated_name(base.name, angle), base.names

    def bounds(self, d):
        return self.base.bounds(d)

    def _flip(self, u: np.ndarray) -> np.ndarray:
        a, b = u[:, 0], u[:, 1]
        return np.column_stack({180: (1 - a, 1 - b), 90: (1 - a, b), 270: (a, 1 - b)}[self.angle])

    def logpdf(self, u, p):
        return self.base.logpdf(self._flip(u), p)

    def cdf(self, u, p):
        a, b = u[:, 0], u[:, 1]
        c = self.base.cdf(self._flip(u), p)
        return {180: a + b - 1 + c, 90: b - c, 270: a - c}[self.angle]

    def h(self, u, v, p):
        g = self.base.h
        return {180: lambda: 1 - g(1 - u, 1 - v, p), 90: lambda: g(1 - u, v, p), 270: lambda: 1 - g(u, 1 - v, p)}[self.angle]()

    def hinv(self, u, w, p):
        g = self.base.hinv
        return {180: lambda: 1 - g(1 - u, 1 - w, p), 90: lambda: g(1 - u, w, p), 270: lambda: 1 - g(u, 1 - w, p)}[self.angle]()

    def sample(self, n, d, p, rng):
        return self._flip(self.base.sample(n, 2, p, rng))

    def tau(self, p):
        return self.base.tau(p) * (1 if self.angle == 180 else -1)

    def tail(self, p):
        lo, up = self.base.tail(p)
        return (up, lo) if self.angle == 180 else (0.0, 0.0)

    def off_tail(self, p) -> dict[str, float]:
        """The 90 and 270 degree rotations' tail dependence, in the corners where one is low and the other high."""
        lo, up = self.base.tail(p)
        if self.angle == 90:  # (1 - X, Y): the base's (low, low) is (high, low); its (high, high) is (low, high)
            return {"upper_lower": lo, "lower_upper": up}
        return {"lower_upper": lo, "upper_lower": up}


_BASE = {c.name: c() for c in (Clayton, Gumbel, Frank, Joe, BB1, BB7, BB6, BB8)}
_BASE.update({t.name: t for t in (Tawn(1), Tawn(2))})
ARCH: dict[str, Any] = {**_BASE, **{rotated_name(f, a): Rotated(_BASE[f], a) for a in (180, 90, 270) for f in ROTATABLE + ROTATABLE_2}}
# Tawn's 90 and 270 degree copulas turn the density (as VineCopula's 124, 134, 224 and 234 do): for an asymmetric
# copula that is the other type with one argument reflected, c_tawn1_90(u, v) = c_tawn2(1 - u, v)
for _k in (1, 2):
    for _a in (90, 270):
        ARCH[f"tawn{_k}_{_a}"] = Rotated(_BASE[f"tawn{3 - _k}"], _a, f"tawn{_k}_{_a}")


def _p(family: str, params: dict[str, Any]) -> tuple[float, ...]:
    return tuple(float(params[k]) for k in ARCH[family].names)


def tau_to_parameter(family: str, tau: float, d: int = 2) -> float:
    """The one-parameter family's parameter with Kendall's tau ``tau`` (inverting the closed form or numerically)."""
    fam = ARCH[family]
    if len(fam.names) != 1:
        raise SpecificationError(f"{family} has two parameters: Kendall's tau does not identify them (use method='ml')")
    lo, hi = fam.bounds(d)[0]
    sign = -1 if family.endswith(("90", "270")) else 1
    base = fam.base if isinstance(fam, Rotated) else fam
    t = sign * tau
    if family.startswith("frank") and d == 2 and t < 0:
        lo, hi = -hi, -1e-4
    elif family.startswith("frank"):
        lo = max(lo, 1e-4)
    f_lo, f_hi = base.tau([lo]), base.tau([hi])
    if t <= f_lo:
        return lo
    if t >= f_hi:
        return hi
    return float(optimize.brentq(lambda th: base.tau([th]) - t, lo, hi, xtol=1e-12))


def _quiet(f: Any) -> Any:
    """Over- and underflow at the edges of a search are expected: the log likelihood there is -inf, and refused."""
    @functools.wraps(f)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with np.errstate(all="ignore"):
            return f(*args, **kwargs)

    return wrapped


# -- the families together ---------------------------------------------------------------------------------------------

@_quiet
def copula_logpdf(family: str, params: dict[str, Any], u: np.ndarray) -> np.ndarray:
    if family == "gaussian":
        return gaussian_logpdf(u, params["R"])
    if family == "t":
        return t_logpdf(u, params["R"], params["nu"])
    return ARCH[family].logpdf(u, _p(family, params))


@_quiet
def copula_cdf(family: str, params: dict[str, Any], u: np.ndarray) -> np.ndarray:
    """The copula's CDF at the rows of ``u``: closed form for the Archimedean families, quadrature for the bivariate
    Gaussian and t, Genz's separation of variables by quasi-Monte Carlo for more than two series."""
    if family in ("gaussian", "t"):
        if u.shape[1] != 2:
            return elliptical_cdf_d(u, params["R"], params.get("nu"))
        return _elliptical_cdf(u[:, 0], u[:, 1], float(params["R"][0, 1]), params.get("nu"))
    return ARCH[family].cdf(u, _p(family, params))


@_quiet
def copula_h(family: str, params: dict[str, Any], u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The h-function P(V <= v | U = u) of a bivariate copula."""
    if family in ("gaussian", "t"):
        return _elliptical_h(u, v, float(params["R"][0, 1]), params.get("nu"))
    return ARCH[family].h(u, v, _p(family, params))


@_quiet
def copula_hinv(family: str, params: dict[str, Any], u: np.ndarray, w: np.ndarray) -> np.ndarray:
    if family in ("gaussian", "t"):
        return _elliptical_hinv(u, w, float(params["R"][0, 1]), params.get("nu"))
    return ARCH[family].hinv(u, w, _p(family, params))


def copula_tau(family: str, params: dict[str, Any]) -> float:
    """Kendall's tau implied by the parameters (of the first pair for the Gaussian and t)."""
    if family in ("gaussian", "t"):
        return float(2 / np.pi * np.arcsin(params["R"][0, 1]))
    return float(ARCH[family].tau(_p(family, params)))


def _check_dimension(family: str, d: int) -> None:
    if family in ("gaussian", "t"):
        return
    if family not in ARCH:
        raise SpecificationError(f"family is one of {', '.join(FAMILIES)}")
    if d != 2 and not ARCH[family].any_dimension:
        what = "the rotated copulas" if isinstance(ARCH[family], Rotated) else f"the {family} copula"
        raise SpecificationError(f"{what} is for two series here: use the gaussian, t, clayton, gumbel, frank or joe copula for more")


def _mean_tau(u: np.ndarray) -> float:
    d = u.shape[1]
    return float(np.mean([sps.kendalltau(u[:, i], u[:, j])[0] for i in range(d) for j in range(i + 1, d)]))


def _safe(values: np.ndarray) -> float:
    total = float(np.sum(values))
    return total if np.isfinite(total) else -1e300


def _bb_start(family: str, tau: float) -> list[tuple[float, float]]:
    tau = min(max(tau, 0.02), 0.95)
    starts = []
    if family == "bb1":
        for de in (1.05, 1.3, 2.0):
            th = 2 / (de * (1 - tau)) - 2
            starts.append((th, de) if th > 0.01 else (0.05, max(1.0, 1 / (1 - tau) / 1.03)))
    else:
        fam = ARCH["bb7"]
        for th in (1.05, 1.3, 2.0):
            g = lambda de: fam.tau((th, de)) - tau  # noqa: E731
            de = optimize.brentq(g, 1e-3, 20.0) if g(1e-3) < 0 < g(20.0) else 0.5
            starts.append((th, de))
    return starts


_GRID_2 = {"bb1": ((0.2, 0.6, 1.5, 3.0), (1.05, 1.3, 1.8, 3.0)), "bb7": ((1.05, 1.4, 2.0, 3.5), (0.1, 0.4, 1.0, 2.5)),
           "bb6": ((1.05, 1.5, 2.5, 4.0), (1.05, 1.5, 2.5, 4.0)), "bb8": ((1.5, 3.0, 6.0, 10.0), (0.3, 0.6, 0.85, 0.99)),
           "tawn1": ((1.3, 2.0, 3.5, 6.0), (0.2, 0.5, 0.8, 0.98)), "tawn2": ((1.3, 2.0, 3.5, 6.0), (0.2, 0.5, 0.8, 0.98))}


def _two_starts(family: str, u: np.ndarray, tau: float) -> list[tuple[float, float]]:
    """Starting values for the two-parameter families added after BB1 and BB7 (and the rotated ones): for rotated BB1
    and BB7 their own starts at the base's Kendall's tau, else the best two points of a small grid by likelihood."""
    fam = ARCH[family]
    base = fam.base.name if isinstance(fam, Rotated) else family
    if base in ("bb1", "bb7"):
        return _bb_start(base, -tau if family.endswith(("_90", "_270")) else tau)
    if base.startswith("tawn"):
        base = "tawn1"
    grid = [(a, b) for a in _GRID_2[base][0] for b in _GRID_2[base][1]]
    ll = [_safe(fam.logpdf(u, g)) for g in grid]
    return [grid[i] for i in np.argsort(ll)[::-1][:2]]


@_quiet
def fit_copula(u: np.ndarray, family: str, method: str = "ml") -> dict[str, Any]:
    """The copula's parameters and log likelihood on uniforms ``u`` (pseudo-observations or the margins' probability
    transforms): by maximum likelihood (``ml``) or from Kendall's tau (``itau``)."""
    d = u.shape[1]
    _check_dimension(family, d)
    if family == "gaussian":
        R = _kendall_corr(u) if method == "itau" else np.corrcoef(sps.norm.ppf(u).T)
        return {"R": R, "loglike": float(gaussian_logpdf(u, R).sum())}
    if family == "t":
        R = _kendall_corr(u)
        found = optimize.minimize_scalar(lambda v: -t_logpdf(u, R, v).sum(), bounds=(2.05, 200.0), method="bounded",
                                         options={"xatol": 1e-6})
        return {"R": R, "nu": float(found.x), "loglike": float(-found.fun)}
    fam = ARCH[family]
    tau = sps.kendalltau(u[:, 0], u[:, 1])[0] if d == 2 else _mean_tau(u)
    if method == "itau":
        th = tau_to_parameter(family, tau, d)
        return {"theta": th, "loglike": float(fam.logpdf(u, (th,)).sum())}
    if family == "clayton" and d == 2:  # as before the other families were added
        start = max(2 * tau / (1 - tau), 0.05)
        found = optimize.minimize_scalar(lambda th: -clayton_logpdf(u, th).sum(), bounds=(1e-4, max(50.0, 4 * start)), method="bounded",
                                         options={"xatol": 1e-8})
        return {"theta": float(found.x), "loglike": float(-found.fun)}
    if family == "gumbel" and d == 2:
        start = max(1 / (1 - tau), 1.01) if tau < 1 else 50.0
        found = optimize.minimize_scalar(lambda th: -gumbel_logpdf(u, th).sum(), bounds=(1.0 + 1e-6, max(50.0, 4 * start)), method="bounded",
                                         options={"xatol": 1e-8})
        return {"theta": float(found.x), "loglike": float(-found.fun)}
    if len(fam.names) == 1:
        lo, hi = fam.bounds(d)[0]
        start = tau_to_parameter(family, tau, d)  # Kendall's tau's value, for the search interval
        if family == "frank" and d == 2:
            lo, hi = ((1e-4, max(hi, 4 * start)) if start > 0 else (min(-hi, 4 * start), -1e-4))
        elif not family.startswith("frank"):
            hi = max(hi, 4 * start)
        found = optimize.minimize_scalar(lambda th: -_safe(fam.logpdf(u, (th,))), bounds=(lo, hi), method="bounded",
                                         options={"xatol": 1e-8})
        return {"theta": float(found.x), "loglike": float(-found.fun)}
    best = None
    t_eff = tau
    for start in (_bb_start(family, t_eff) if family in ("bb1", "bb7") else _two_starts(family, u, tau)):
        found = optimize.minimize(lambda p: -_safe(fam.logpdf(u, tuple(p))), np.clip(start, *zip(*fam.bounds(d))),
                                  method="L-BFGS-B", bounds=fam.bounds(d))
        if best is None or found.fun < best.fun:
            best = found
    return {**{k: float(v) for k, v in zip(fam.names, best.x)}, "loglike": float(-best.fun)}


@_quiet
def sample_copula(family: str, params: dict[str, Any], n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` draws of the copula's uniforms."""
    if family == "gaussian":
        return sps.norm.cdf(rng.standard_normal((n, d)) @ np.linalg.cholesky(params["R"]).T)
    if family == "t":
        nu = params["nu"]
        z = rng.standard_normal((n, d)) @ np.linalg.cholesky(params["R"]).T
        return sps.t.cdf(z / np.sqrt(rng.chisquare(nu, (n, 1)) / nu), nu)
    _check_dimension(family, d)
    return ARCH[family].sample(n, d, _p(family, params), rng)


@_quiet
def sample_conditional(family: str, params: dict[str, Any], n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` draws of a bivariate copula by conditional inversion: u uniform, v = h^-1(u, w) for w uniform."""
    u, w = rng.uniform(size=n), rng.uniform(size=n)
    return np.column_stack([u, copula_hinv(family, params, u, w)])


def tail_dependence(family: str, params: dict[str, Any]) -> dict[str, Any]:
    if family == "gaussian":
        return {"lower": 0.0, "upper": 0.0}
    if family == "t":
        nu, R = params["nu"], params["R"]
        lam = 2 * sps.t.cdf(-np.sqrt((nu + 1) * (1 - R) / (1 + R + 1e-300)), nu + 1)
        np.fill_diagonal(lam, 1.0)
        return {"lower": lam, "upper": lam}
    fam = ARCH[family]
    lo, up = fam.tail(_p(family, params))
    out: dict[str, Any] = {"lower": float(lo), "upper": float(up)}
    if isinstance(fam, Rotated) and fam.angle != 180:
        out.update({k: float(v) for k, v in fam.off_tail(_p(family, params)).items()})
    return out


def n_parameters(family: str, d: int) -> int:
    if family == "gaussian":
        return d * (d - 1) // 2
    if family == "t":
        return d * (d - 1) // 2 + 1
    return len(ARCH[family].names)


# -- standard errors ---------------------------------------------------------------------------------------------------

def _hessian(f: Any, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    k = len(x)
    h = 1e-4 * np.maximum(np.abs(x), 1.0)
    H = np.empty((k, k))
    f0 = f(x)
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = h[i]
        H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h[i] ** 2
        for j in range(i + 1, k):
            ej = np.zeros(k)
            ej[j] = h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


def _ml_names(family: str, d: int, method: str) -> list[str]:
    """The copula parameters estimated by maximum likelihood (those with standard errors)."""
    if method == "itau":
        return []
    if family == "gaussian":
        return [f"rho[{i},{j}]" for i in range(d) for j in range(i + 1, d)]
    if family == "t":
        return ([f"rho[{i},{j}]" for i in range(d) for j in range(i + 1, d)] if method == "joint" else []) + ["nu"]
    return list(ARCH[family].names)


def _with_values(params: dict[str, Any], names: list[str], x: np.ndarray) -> dict[str, Any]:
    out = dict(params)
    if "R" in out:
        out["R"] = np.array(out["R"], float, copy=True)
    for name, value in zip(names, x):
        if name.startswith("rho["):
            i, j = map(int, name[4:-1].split(","))
            out["R"][i, j] = out["R"][j, i] = value
        else:
            out[name] = value
    return out


def _values(params: dict[str, Any], names: list[str]) -> np.ndarray:
    out = []
    for name in names:
        if name.startswith("rho["):
            i, j = map(int, name[4:-1].split(","))
            out.append(params["R"][i, j])
        else:
            out.append(params[name])
    return np.array(out, float)


@_quiet
def _copula_se(u: np.ndarray, family: str, params: dict[str, Any], method: str) -> dict[str, float]:
    names = _ml_names(family, u.shape[1], method)
    if not names:
        return {}
    x = _values(params, names)

    def negll(v: np.ndarray) -> float:
        try:
            return -float(np.sum(copula_logpdf(family, _with_values(params, names, v), u)))
        except (np.linalg.LinAlgError, ValueError, SpecificationError):
            return np.nan

    return _se_from(negll, x, names)


def _se_from(negll: Any, x: np.ndarray, names: list[str], keep: list[str] | None = None) -> dict[str, float]:
    with np.errstate(all="ignore"):
        H = _hessian(negll, x)
        try:
            cov = np.linalg.inv(H)
            se = np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))
        except np.linalg.LinAlgError:
            se = np.full(len(x), np.nan)
    return {n: float(s) for n, s in zip(names, se) if keep is None or n in keep}


# -- parametric margins ------------------------------------------------------------------------------------------------

def _skewt_consts(nu: float, lam: float) -> tuple[float, float, float]:
    c = math.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / math.sqrt(np.pi * (nu - 2))
    a = 4 * lam * c * (nu - 2) / (nu - 1)
    b = math.sqrt(1 + 3 * lam**2 - a**2)
    return a, b, c


def skewt_logpdf(z: Any, nu: float, lam: float) -> np.ndarray:
    """Hansen's (1994) skewed t, standardised to mean 0 and variance 1: ``nu`` > 2 degrees of freedom, skewness
    -1 < ``lam`` < 1 (negative: a longer left tail)."""
    z = np.asarray(z, float)
    a, b, c = _skewt_consts(nu, lam)
    s = np.where(b * z + a < 0, 1 - lam, 1 + lam)
    return np.log(b * c) - (nu + 1) / 2 * np.log1p(((b * z + a) / s) ** 2 / (nu - 2))


def skewt_cdf(z: Any, nu: float, lam: float) -> np.ndarray:
    z = np.asarray(z, float)
    a, b, _ = _skewt_consts(nu, lam)
    y = b * z + a
    k = np.sqrt(nu / (nu - 2))
    return np.where(y < 0, (1 - lam) * sps.t.cdf(k * y / (1 - lam), nu),
                    (1 - lam) / 2 + (1 + lam) * (sps.t.cdf(k * y / (1 + lam), nu) - 0.5))


def skewt_ppf(q: Any, nu: float, lam: float) -> np.ndarray:
    q = np.asarray(q, float)
    a, b, _ = _skewt_consts(nu, lam)
    k = np.sqrt((nu - 2) / nu)
    low = q < (1 - lam) / 2
    w = np.where(low, sps.t.ppf(np.clip(q / (1 - lam), 0, 1), nu),
                 sps.t.ppf(np.clip(0.5 + (q - (1 - lam) / 2) / (1 + lam), 0, 1), nu))
    y = w * k * np.where(low, 1 - lam, 1 + lam)
    return (y - a) / b


_MARGIN_NAMES = {"normal": ("mu", "sigma"), "t": ("mu", "sigma", "nu"), "skewt": ("mu", "sigma", "nu", "lambda")}
_MARGIN_BOUNDS = {"mu": (None, None), "sigma": (1e-6, None), "nu": (2.05, 200.0), "lambda": (-0.99, 0.99)}


@dataclass(frozen=True)
class Margin:
    """A fitted marginal distribution: ``normal`` (mu, sigma), ``t`` (location mu, scale sigma, nu) or ``skewt``
    (Hansen's skewed t of mean mu and standard deviation sigma)."""

    kind: str
    params: dict[str, float]

    def _z(self, x: Any) -> np.ndarray:
        return (np.asarray(x, float) - self.params["mu"]) / self.params["sigma"]

    def logpdf(self, x: Any) -> np.ndarray:
        z, s = self._z(x), self.params["sigma"]
        if self.kind == "normal":
            return sps.norm.logpdf(z) - np.log(s)
        if self.kind == "t":
            return sps.t.logpdf(z, self.params["nu"]) - np.log(s)
        return skewt_logpdf(z, self.params["nu"], self.params["lambda"]) - np.log(s)

    def cdf(self, x: Any) -> np.ndarray:
        z = self._z(x)
        if self.kind == "normal":
            return sps.norm.cdf(z)
        if self.kind == "t":
            return sps.t.cdf(z, self.params["nu"])
        return skewt_cdf(z, self.params["nu"], self.params["lambda"])

    def ppf(self, q: Any) -> np.ndarray:
        if self.kind == "normal":
            z = sps.norm.ppf(q)
        elif self.kind == "t":
            z = sps.t.ppf(q, self.params["nu"])
        else:
            z = skewt_ppf(q, self.params["nu"], self.params["lambda"])
        return self.params["mu"] + self.params["sigma"] * z

    def scaled(self, m: float, s: float) -> Margin:
        """The margin of m + s X."""
        return Margin(self.kind, {k: float(v) for k, v in {**self.params, "mu": m + s * self.params["mu"], "sigma": s * self.params["sigma"]}.items()})


@_quiet
def fit_margin(x: np.ndarray, kind: str, free: Sequence[str] | None = None, fixed: dict[str, float] | None = None) -> Margin:
    """A margin by maximum likelihood (on the standardised series, scaled back). ``free`` limits the parameters
    estimated; the rest come from ``fixed``."""
    x = np.asarray(x, float)
    m, s = float(np.mean(x)), float(np.std(x)) or 1.0
    z = (x - m) / s
    names = _MARGIN_NAMES[kind]
    free = list(names if free is None else free)
    base = {"mu": 0.0, "sigma": 1.0, "nu": 8.0, "lambda": 0.0}
    names_all = _MARGIN_NAMES[kind]
    if fixed:
        base.update({k: (v - m) / s if k == "mu" else v / s if k == "sigma" else v for k, v in fixed.items()})
    if kind == "normal" and free == ["mu", "sigma"]:
        return Margin("normal", {"mu": m, "sigma": s})
    if kind == "t" and "sigma" in free:
        base["sigma"] = 0.8

    def negll(v: np.ndarray) -> float:
        out = -_safe(Margin(kind, {**base, **dict(zip(free, v))}).logpdf(z))
        return out if np.isfinite(out) else 1e300

    best = None
    for nu0 in (8.0, 4.0):
        x0 = [nu0 if n == "nu" else base[n] for n in free]
        found = optimize.minimize(negll, x0, method="L-BFGS-B", bounds=[_MARGIN_BOUNDS[n] for n in free])
        if best is None or found.fun < best.fun:
            best = found
    final = {**base, **dict(zip(free, map(float, best.x)))}
    return Margin(kind, {k: final[k] for k in names_all}).scaled(m, s)


# -- the fit -----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FittedCopula:
    """A copula of several returns, with their margins, for dependence and portfolio risk."""

    series: tuple[str, ...]
    family: str
    params: dict[str, Any]
    u: np.ndarray = field(repr=False)  # the uniforms the copula was fitted to (ranks, or the margins' transforms)
    residuals: np.ndarray = field(repr=False)  # standardised (GARCH) or raw returns, by column
    scale_next: np.ndarray = field(repr=False)  # next period's sd per series (1 without GARCH)
    mean_next: np.ndarray = field(repr=False)
    garch: bool = False
    index: Any = None
    margins: str = "empirical"
    method: str = "ml"
    margin_fits: tuple[Margin, ...] = ()
    se: dict[str, float] = field(default_factory=dict)
    gof_result: dict[str, float] | None = None
    margin_loglike: float | None = None  # the margins' log likelihood (parametric margins)
    gof_bootstrap: str = "parametric"  # how gof_result's p-value was bootstrapped

    stage = "finance"

    @property
    def loglike(self) -> float:
        """The copula's log likelihood."""
        return float(self.params["loglike"])

    @property
    def k(self) -> int:
        return n_parameters(self.family, len(self.series))

    @property
    def aic(self) -> float:
        return 2 * self.k - 2 * self.loglike

    @property
    def bic(self) -> float:
        return self.k * np.log(len(self.u)) - 2 * self.loglike

    @property
    def total_loglike(self) -> float | None:
        """Margins and copula together, with parametric margins."""
        return None if self.margin_loglike is None else self.loglike + self.margin_loglike

    @property
    def tau(self) -> float:
        """Kendall's tau implied by the copula (of the first two series)."""
        return copula_tau(self.family, self.params)

    def tail_dependence(self) -> Any:
        return tail_dependence(self.family, self.params)

    def simulate(self, n: int = 10_000, seed: int | None = 0) -> np.ndarray:
        """``n`` draws of the copula's uniforms, one column a series."""
        return sample_copula(self.family, self.params, int(n), len(self.series), make_rng(seed))

    def goodness_of_fit(self, n_boot: int = 100, seed: int | None = 0, bootstrap: str = "parametric") -> dict[str, float]:
        """The Cramer-von Mises test of the family against the empirical copula, its p-value by parametric bootstrap
        (Genest, Remillard and Beaudoin 2009) or with ``bootstrap="multiplier"`` Kojadinovic and Yan's (2011)
        multiplier bootstrap, on the pseudo-observations whatever the margins."""
        return gof_test(pseudo_observations(self.residuals), self.family, "itau" if self.method == "itau" else "ml", n_boot, seed,
                        bootstrap)

    def portfolio_var(self, weights: Sequence[float], level: float = 0.99, n: int = 100_000, margins: str | None = None,
                      seed: int | None = 0) -> dict[str, float]:
        """Next period's VaR and ES of a portfolio by simulation from the copula and the margins: the fitted margins
        (by default), ``"empirical"`` quantiles or a ``"normal"`` fitted to the returns or residuals."""
        if np.asarray(weights, float).shape != (len(self.series),):
            raise SpecificationError(f"give a weight for each of {', '.join(self.series)}")
        return _portfolio_from_uniforms(self, self.simulate(n, seed), weights, level, margins)

    def summary(self) -> Table:
        rows, names = [], []

        def add(name: str, value: float, key: str | None = None) -> None:
            names.append(name)
            rows.append([float(value), float(self.se.get(key, np.nan)) if key else np.nan])

        if "R" in self.params:
            R = self.params["R"]
            for i in range(len(self.series)):
                for j in range(i + 1, len(self.series)):
                    add(f"rho {self.series[i]}, {self.series[j]}", R[i, j], f"rho[{i},{j}]")
        for key, label in (("nu", "degrees of freedom"), ("theta", "theta"), ("delta", "delta"), ("psi", "psi")):
            if key in self.params:
                add(label, self.params[key], key)
        if self.family not in ("gaussian", "t"):
            add("Kendall's tau", self.tau)
        td = self.tail_dependence()
        lo, hi = (td["lower"][0, 1], td["upper"][0, 1]) if isinstance(td["lower"], np.ndarray) else (td["lower"], td["upper"])
        add("lower tail dependence", lo)
        add("upper tail dependence", hi)
        for key, label in (("lower_upper", "tail dependence, first low and second high"),
                           ("upper_lower", "tail dependence, first high and second low")):
            if key in td:
                add(label, td[key])
        for s, mf in zip(self.series, self.margin_fits):
            for key, value in mf.params.items():
                add(f"{s}: {key}", value)
        if self.gof_result is not None:
            add("Cramer-von Mises S_n", self.gof_result["statistic"])
            add("goodness of fit p-value", self.gof_result["pvalue"])
        T = len(self.u)
        what = "GARCH(1,1) standardised residuals" if self.garch else "the returns"
        how = {"empirical": "their ranks", "normal": "fitted normal margins", "t": "fitted t margins",
               "skewt": "fitted skewed-t margins (Hansen)"}[self.margins]
        est = {"ml": "maximum likelihood" + (" (IFM)" if self.margin_fits else ""), "itau": "inversion of Kendall's tau",
               "joint": "joint maximum likelihood of margins and copula"}[self.method]
        notes = [f"{self.family.capitalize()} copula of {what} by {how}, {T} periods; {est}.",
                 f"Log likelihood (copula): {self.loglike:.4f}", f"Akaike info criterion: {self.aic:.4f}",
                 f"Schwarz criterion: {self.bic:.4f}",
                 *([f"Log likelihood (with the margins): {self.total_loglike:.4f}"] if self.total_loglike is not None else [])]
        if self.se:
            notes.append("Standard errors from the observed information" + (" of margins and copula together." if self.method == "joint"
                         else ", treating the uniforms as known (they ignore the margins' estimation)."))
        if len(self.series) > 2:
            notes.append("Tail dependence of the first two series." if self.family in ("gaussian", "t") else
                         "Exchangeable: one parameter for every pair.")
        if self.gof_result is not None:
            notes.append(f"Goodness of fit: {self.gof_bootstrap} bootstrap with {int(self.gof_result['n_boot'])} samples"
                         + (" (Kojadinovic and Yan 2011)." if self.gof_bootstrap == "multiplier" else "."))
        cols = ["value", "std err"] if self.se else ["value"]
        return Table(cols, [r[: len(cols)] for r in rows], index=names, index_name="parameter", title="Copula", notes=tuple(notes))

    def plot(self) -> Figure:
        from pymetrix.figures import Figure, Panel, Scatter

        sim = self.simulate(len(self.u), seed=1)
        a, b = self.series[0], self.series[1]
        return Figure((Panel((Scatter(pd.Series(self.u[:, 0]), pd.Series(self.u[:, 1]), "data"),), title=f"Pseudo-observations: {a} and {b}",
                             xlabel=a, ylabel=b),
                       Panel((Scatter(pd.Series(sim[:, 0]), pd.Series(sim[:, 1]), "simulated"),), title=f"From the fitted {self.family} copula",
                             xlabel=a, ylabel=b)))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return pd.DataFrame(self.u, columns=list(self.series), index=self.index)

    def __repr__(self) -> str:
        return f"<FittedCopula {self.family} of {', '.join(self.series)}>"


def _portfolio_from_uniforms(fit: Any, U: np.ndarray, weights: Sequence[float], level: float, margins: str | None,
                             mean: np.ndarray | None = None, scale: np.ndarray | None = None) -> dict[str, float]:
    """VaR and ES of the weighted sum of returns made from the copula draws ``U`` through ``fit``'s margins: the
    fitted ones, the ``"empirical"`` quantiles of the returns or residuals, or a ``"normal"`` fitted to them; scaled
    by next period's mean and standard deviation (GARCH)."""
    w = np.asarray(weights, float)
    if w.shape != (len(fit.series),):
        raise SpecificationError(f"give a weight for each of {', '.join(fit.series)}")
    margins = margins or ("fitted" if fit.margin_fits else "empirical")
    if margins == "empirical":
        Z = np.column_stack([np.quantile(fit.residuals[:, j], U[:, j]) for j in range(U.shape[1])])
    elif margins == "normal":
        m, s = fit.residuals.mean(axis=0), fit.residuals.std(axis=0, ddof=1)
        Z = m + s * sps.norm.ppf(U)
    elif margins == "fitted" and fit.margin_fits:
        Z = np.column_stack([mf.ppf(U[:, j]) for j, mf in enumerate(fit.margin_fits)])
    else:
        raise SpecificationError("margins is 'fitted' (with parametric margins), 'empirical' or 'normal'")
    R = (fit.mean_next if mean is None else mean) + (fit.scale_next if scale is None else scale) * Z
    from .risk import historical

    v, e = historical(R @ w, 1 - level)
    return {"var": float(v), "es": float(e), "draws": float(len(U))}


# -- goodness of fit ---------------------------------------------------------------------------------------------------

def empirical_copula(U: np.ndarray, at: np.ndarray | None = None) -> np.ndarray:
    """C_n at the rows of ``at`` (default ``U``): the share of the rows of ``U`` below each, coordinate by coordinate."""
    at = U if at is None else at
    out = np.empty(len(at))
    step = max(1, 4_000_000 // max(len(U) * U.shape[1], 1))
    for i in range(0, len(at), step):
        out[i:i + step] = np.all(U[None, :, :] <= at[i:i + step, None, :], axis=2).mean(axis=1)
    return out


def cramer_von_mises(U: np.ndarray, family: str, params: dict[str, Any]) -> float:
    """S_n = sum_i (C_n(U_i) - C_theta(U_i))^2 at the pseudo-observations."""
    return float(np.sum((empirical_copula(U) - copula_cdf(family, params, U)) ** 2))


BOOTSTRAPS = ("parametric", "multiplier")


@_quiet
def gof_test(U: np.ndarray, family: str, method: str = "ml", n_boot: int = 100, seed: int | None = 0,
             bootstrap: str = "parametric") -> dict[str, float]:
    """Genest, Remillard and Beaudoin's (2009) Cramer-von Mises test on pseudo-observations ``U`` with a parametric
    bootstrap p-value: each bootstrap sample is drawn from the fitted copula, made pseudo-observations and refitted.
    In more than two dimensions the Gaussian and t CDFs are by quasi-Monte Carlo (``elliptical_cdf_d``).
    ``bootstrap="multiplier"``: Kojadinovic and Yan's (2011) multiplier p-value instead (``multiplier_test``)."""
    if bootstrap not in BOOTSTRAPS:
        raise SpecificationError(f"bootstrap is one of {', '.join(BOOTSTRAPS)}")
    if n_boot < 1:
        raise SpecificationError("the goodness-of-fit bootstrap needs at least one sample")
    if bootstrap == "multiplier":
        return multiplier_test(U, family, method, n_boot, seed)
    n, d = U.shape
    params = fit_copula(U, family, method)
    stat = cramer_von_mises(U, family, params)
    rng = make_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        Ub = pseudo_observations(sample_copula(family, params, n, d, rng))
        boot[b] = cramer_von_mises(Ub, family, fit_copula(Ub, family, method))
    return {"statistic": stat, "pvalue": float((np.sum(boot >= stat) + 0.5) / (n_boot + 1)), "n_boot": float(n_boot)}


def _multiplier_names(family: str, d: int, method: str) -> list[str]:
    """The parameters whose influence the multiplier bootstrap needs, or an error where it is not available."""
    if family == "t":
        raise SpecificationError("the multiplier bootstrap needs every parameter by maximum likelihood or Kendall's tau: "
                                 "the t copula's correlations are from Kendall's tau and its degrees of freedom by maximum "
                                 "likelihood; use bootstrap='parametric'")
    if method == "itau":
        if d != 2:
            raise SpecificationError("the multiplier bootstrap with method='itau' is for two series (as in the copula package)")
        return ["rho[0,1]"] if family == "gaussian" else ["theta"]
    return _ml_names(family, d, "ml")


def _num_grad(f: Any, x: np.ndarray, lo: np.ndarray, hi: np.ndarray, rel: float = 1e-6) -> list[np.ndarray]:
    """Central differences of the vector function f in each coordinate of x (one-sided at a bound)."""
    out = []
    for k in range(len(x)):
        h = rel * max(abs(x[k]), 1.0)
        up, dn = x.copy(), x.copy()
        up[k] = min(x[k] + h, hi[k])
        dn[k] = max(x[k] - h, lo[k])
        out.append((f(up) - f(dn)) / (up[k] - dn[k]))
    return out


def _parameter_bounds(family: str, d: int, names: list[str], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if family == "gaussian":
        return np.full(len(names), -0.9999), np.full(len(names), 0.9999)
    b = ARCH[family].bounds(d)
    if family == "frank" and d == 2:
        b = [(1e-4, np.inf)] if x[0] > 0 else [(-np.inf, -1e-4)]
    return np.array([lo for lo, _ in b], float), np.array([np.inf if family.startswith("frank") else hi for _, hi in b], float)


def multiplier_influence(U: np.ndarray, family: str, params: dict[str, Any], method: str = "ml") -> tuple[np.ndarray, np.ndarray]:
    """dC_theta/dtheta at the pseudo-observations (n, p) and the estimator's influence J (n, p), as in the copula
    package's Jscore: maximum pseudo-likelihood (Kojadinovic and Yan 2011, p. 7) J = Sigma^-1 (score - S), S the rank
    correction sum_j [(1/n) sum_{k: U_kj > U_ij} score_k d_j log c_k - mean_k(score_k d_j log c_k U_kj)], Sigma
    the scores' outer product; inversion of Kendall's tau (Kojadinovic, Yan and Holmes 2011)
    J = 4 / tau'(theta) (2 C(U) - U_1 - U_2 + (1 - tau) / 2). Derivatives by central differences."""
    n, d = U.shape
    names = _multiplier_names(family, d, method)
    x = _values(params, names)
    lo, hi = _parameter_bounds(family, d, names, x)
    dC = np.column_stack(_num_grad(lambda v: copula_cdf(family, _with_values(params, names, v), U), x, lo, hi))
    if method == "itau":
        tau = copula_tau(family, params)
        dtau = _num_grad(lambda v: np.array([copula_tau(family, _with_values(params, names, v))]), x, lo, hi)[0][0]
        J = 4 / dtau * (2 * copula_cdf(family, params, U) - U[:, 0] - U[:, 1] + (1 - tau) / 2)
        return dC, J[:, None]
    score = np.column_stack(_num_grad(lambda v: copula_logpdf(family, _with_values(params, names, v), U), x, lo, hi))
    S = np.zeros_like(score)
    for j in range(d):
        def logc_j(z: np.ndarray, j: int = j) -> np.ndarray:
            V = U.copy()
            V[:, j] = z
            return copula_logpdf(family, params, V)

        h = 1e-6
        dlogc = (logc_j(U[:, j] + h) - logc_j(U[:, j] - h)) / (2 * h)
        Ij = score * dlogc[:, None]
        order = np.argsort(-U[:, j], kind="stable")
        csum = np.vstack([np.zeros(len(names)), np.cumsum(Ij[order], axis=0)])
        below = sps.rankdata(U[:, j], method="max").astype(int)  # #{k: U_kj <= U_ij}
        S += csum[n - below] / n - (Ij * U[:, [j]]).mean(axis=0)
    Sigma = score.T @ score / n
    return dC, np.linalg.solve(Sigma, (score - S).T).T


@_quiet
def multiplier_test(U: np.ndarray, family: str, method: str = "ml", n_boot: int = 1000, seed: int | None = 0) -> dict[str, float]:
    """The Cramer-von Mises test S_n of ``gof_test`` with Kojadinovic and Yan's (2011) multiplier bootstrap p-value
    (as the copula package's gofCopula(..., simulation="mult")): no refitting; each replicate is
    S_n^(b) = (1/n) sum_i G_b(U_i)^2 with G_b(u) = n^-1/2 sum_k [(Z_k - Zbar)(1{U_k <= u} - sum_j D_j(u) 1{U_kj <= u_j})
    - Z_k dC_theta(u)/dtheta . J(U_k)], Z_k standard normal multipliers, D_j the empirical copula's partial
    derivatives by finite differences of width 2 / sqrt(n), J the estimator's influence (``multiplier_influence``)."""
    if n_boot < 1:
        raise SpecificationError("the goodness-of-fit bootstrap needs at least one sample")
    n, d = U.shape
    params = fit_copula(U, family, method)
    stat = cramer_von_mises(U, family, params)
    dC, J = multiplier_influence(U, family, params, method)
    b = 1 / np.sqrt(n)
    der = np.empty((n, d))
    for j in range(d):
        up, dn = U.copy(), U.copy()
        up[:, j] += b
        dn[:, j] -= b
        der[:, j] = (empirical_copula(U, up) - empirical_copula(U, dn)) / (2 * b)
    rng = make_rng(seed)
    Z = rng.standard_normal((int(n_boot), n))
    Zc = Z - Z.mean(axis=1, keepdims=True)
    G = -((Z @ J) @ dC.T)
    step = max(1, 2_000_000 // max(n * d, 1))
    for lo in range(0, n, step):
        at = U[lo:lo + step]
        below = U[:, None, :] <= at[None, :, :]  # (n, block, d): 1{U_kj <= u_ij}
        M = below.all(axis=2) - np.einsum("kbj,bj->kb", below, der[lo:lo + step])
        G[:, lo:lo + step] += Zc @ M
    boot = (G**2).sum(axis=1) / n**2
    return {"statistic": stat, "pvalue": float((np.sum(boot >= stat) + 0.5) / (n_boot + 1)), "n_boot": float(n_boot)}


# -- estimation with the margins ---------------------------------------------------------------------------------------

@dataclass
class _Prepared:
    frame: pd.DataFrame
    resid: np.ndarray
    scale: np.ndarray
    mean: np.ndarray
    margin_fits: tuple[Margin, ...]
    u: np.ndarray
    margin_loglike: float | None


def _prepare(data: Dataset, series: Sequence[str], garch: bool, dist: str | None, margins: str) -> _Prepared:
    if margins not in MARGINS:
        raise SpecificationError(f"margins is one of {', '.join(MARGINS)}")
    if len(series) < 2:
        raise SpecificationError("a copula needs at least two series")
    missing = [s for s in series if s not in data.names]
    if missing:
        raise SpecificationError(f"no series {', '.join(map(repr, missing))} in the dataset")
    frame = data.frame[list(series)].dropna()
    if len(frame) < 50:
        raise SpecificationError(f"{len(frame)} periods are too few for a copula")
    fits: tuple[Margin, ...] = ()
    if garch:
        from .risk import _fit_garch

        if margins in ("normal", "t"):
            if dist not in (None, margins):
                raise SpecificationError(f"with GARCH, margins={margins!r} are the GARCH's own errors: leave dist out or make it {margins!r}")
            dist = margins
        g = [_fit_garch(frame[s], dist=dist or "normal") for s in series]
        resid = np.column_stack([f.z for f in g])
        scale = np.array([np.sqrt(f.var_next) for f in g])
        mean = np.array([f.mu for f in g])
        if margins == "normal":
            fits = tuple(Margin("normal", {"mu": 0.0, "sigma": 1.0}) for _ in g)
        elif margins == "t":
            fits = tuple(Margin("t", {"mu": 0.0, "sigma": float(np.sqrt((f.nu - 2) / f.nu)), "nu": float(f.nu)}) for f in g)
        elif margins == "skewt":
            fits = tuple(fit_margin(resid[:, j], "skewt", free=("nu", "lambda"), fixed={"mu": 0.0, "sigma": 1.0}) for j in range(len(g)))
    else:
        resid, scale, mean = frame.to_numpy(float), np.ones(len(series)), np.zeros(len(series))
        if margins != "empirical":
            fits = tuple(fit_margin(resid[:, j], margins) for j in range(len(series)))
    if fits:
        u = np.clip(np.column_stack([mf.cdf(resid[:, j]) for j, mf in enumerate(fits)]), 1e-10, 1 - 1e-10)
        mll = float(sum(np.sum(mf.logpdf(resid[:, j])) for j, mf in enumerate(fits)))
    else:
        u, mll = pseudo_observations(resid), None
    return _Prepared(frame, resid, scale, mean, fits, u, mll)


def _chol_params(R: np.ndarray) -> np.ndarray:
    L = np.linalg.cholesky(R)
    L = L / np.diag(L)[:, None]
    return L[np.tril_indices(len(R), -1)]


def _chol_matrix(z: np.ndarray, d: int) -> np.ndarray:
    L = np.eye(d)
    L[np.tril_indices(d, -1)] = z
    S = L @ L.T
    s = np.sqrt(np.diag(S))
    return S / np.outer(s, s)


@_quiet
def _fit_joint(x: np.ndarray, family: str, margins: tuple[Margin, ...], start: dict[str, Any]) -> tuple[dict[str, Any], tuple[Margin, ...], dict[str, float], float]:
    """Full maximum likelihood of the margins and the copula together, from the IFM estimates. The margins are worked
    on standardised series (scaled back after); the correlation matrix through a Cholesky factor with unit diagonal."""
    d = x.shape[1]
    m, s = x.mean(axis=0), x.std(axis=0)
    z = (x - m) / s
    std = [mf.scaled(-m[j] / s[j], 1 / s[j]) for j, mf in enumerate(margins)]
    kinds = [mf.kind for mf in std]
    mnames = [list(_MARGIN_NAMES[k]) for k in kinds]
    lens = [len(n) for n in mnames]
    fam = None if family in ("gaussian", "t") else ARCH[family]

    def unpack(v: np.ndarray) -> tuple[list[Margin], dict[str, Any]]:
        out, pos = [], 0
        for k, names in zip(kinds, mnames):
            out.append(Margin(k, dict(zip(names, v[pos:pos + len(names)]))))
            pos += len(names)
        rest = v[pos:]
        if family == "gaussian":
            return out, {"R": _chol_matrix(rest, d)}
        if family == "t":
            return out, {"R": _chol_matrix(rest[:-1], d), "nu": rest[-1]}
        return out, dict(zip(fam.names, rest))

    def total(margs: list[Margin], cp: dict[str, Any], data: np.ndarray) -> float:
        u = np.clip(np.column_stack([mf.cdf(data[:, j]) for j, mf in enumerate(margs)]), 1e-12, 1 - 1e-12)
        return sum(_safe(mf.logpdf(data[:, j])) for j, mf in enumerate(margs)) + _safe(copula_logpdf(family, cp, u))

    def negll(v: np.ndarray) -> float:
        out = -total(*unpack(v), z)
        return out if np.isfinite(out) else 1e300

    x0 = [mf.params[n] for mf, names in zip(std, mnames) for n in names]
    bounds = [_MARGIN_BOUNDS[n] for names in mnames for n in names]
    if family in ("gaussian", "t"):
        cz = _chol_params(start["R"])
        x0 += list(cz)
        bounds += [(None, None)] * len(cz)
        if family == "t":
            x0.append(start["nu"])
            bounds.append((2.05, 200.0))
    else:
        x0 += [start[n] for n in fam.names]
        b = fam.bounds(d)
        if family == "frank" and d == 2:
            b = [(1e-4, 50.0)] if start["theta"] > 0 else [(-50.0, -1e-4)]
        bounds += b
    found = optimize.minimize(negll, np.array(x0, float), method="L-BFGS-B", bounds=bounds)
    if found.fun > negll(np.array(x0, float)):
        found.x = np.array(x0, float)
    margs, cp = unpack(found.x)
    # standard errors from the joint Hessian, in the natural parameters (correlations rather than the Cholesky factor)
    cnames = _ml_names(family, d, "joint")
    nat0 = np.r_[[mf.params[n] for mf, names in zip(margs, mnames) for n in names], _values(cp, cnames)]
    nm = sum(lens)

    def negll_natural(v: np.ndarray) -> float:
        ms, pos = [], 0
        for k, names in zip(kinds, mnames):
            ms.append(Margin(k, dict(zip(names, v[pos:pos + len(names)]))))
            pos += len(names)
        try:
            return -total(ms, _with_values(cp, cnames, v[nm:]), z)
        except (np.linalg.LinAlgError, ValueError):
            return np.nan

    se = _se_from(negll_natural, nat0, [f"m{i}" for i in range(nm)] + cnames, keep=cnames)
    fitted = tuple(mf.scaled(m[j], s[j]) for j, mf in enumerate(margs))
    u = np.clip(np.column_stack([mf.cdf(x[:, j]) for j, mf in enumerate(fitted)]), 1e-10, 1 - 1e-10)
    params = {**{k: v for k, v in cp.items()}, "loglike": float(np.sum(copula_logpdf(family, cp, u)))}
    for key in ("nu", "theta", "delta", "psi"):
        if key in params:
            params[key] = float(params[key])
    mll = float(sum(np.sum(mf.logpdf(x[:, j])) for j, mf in enumerate(fitted)))
    return params, fitted, se, mll


def _estimate(prep: _Prepared, family: str, method: str, garch: bool, margins: str) -> tuple[dict[str, Any], tuple[Margin, ...], dict[str, float], np.ndarray, float | None]:
    if method not in METHODS:
        raise SpecificationError(f"method is one of {', '.join(METHODS)}")
    if method == "itau" and family not in ("gaussian", "t") and len(ARCH[family].names) != 1:
        raise SpecificationError(f"{family} has two parameters: Kendall's tau does not identify them (use method='ml')")
    params = fit_copula(prep.u, family, "itau" if method == "itau" else "ml")
    if method != "joint":
        return params, prep.margin_fits, _copula_se(prep.u, family, params, method), prep.u, prep.margin_loglike
    if margins == "empirical" or garch:
        raise SpecificationError("method='joint' needs parametric margins (normal, t or skewt) and no GARCH")
    params, fits, se, mll = _fit_joint(prep.resid, family, prep.margin_fits, params)
    u = np.clip(np.column_stack([mf.cdf(prep.resid[:, j]) for j, mf in enumerate(fits)]), 1e-10, 1 - 1e-10)
    return params, fits, se, u, mll


def default_families(d: int, tau: float) -> list[str]:
    """The families compared by default: all that fit ``d`` series, those for negative dependence when tau < 0."""
    if d > 2:
        return ["gaussian", "t", *ARCHIMEDEAN]
    if tau >= 0:
        return ["gaussian", "t", "clayton", "gumbel", "frank", "joe", "bb1", "bb7", "clayton180", "gumbel180", "joe180"]
    return ["gaussian", "t", "frank", "clayton90", "gumbel90", "joe90", "clayton270", "gumbel270", "joe270"]


with plugin_context(PLUGIN):
    @api("finance:copula")
    def copula(data: Dataset, series: Sequence[str], family: Family = "t", garch: bool = False, dist: Literal["normal", "t"] | None = None,
               margins: Literal["empirical", "normal", "t", "skewt"] = "empirical", method: Literal["ml", "itau", "joint"] = "ml",
               gof: bool = False, n_boot: int = 100, seed: int = 0,
               bootstrap: Literal["parametric", "multiplier"] = "parametric") -> FittedCopula:
        """A copula of the returns ``series`` (Gaussian, t, Clayton, Gumbel, Frank or Joe for any number; BB1, BB7 and
        the rotated families for two), of the returns or, with ``garch``, of each one's GARCH(1,1) standardised
        residuals; ``margins`` by ranks or fitted (normal, t, Hansen's skewed t), ``method`` maximum likelihood (IFM
        with fitted margins), Kendall's tau or joint; ``gof`` adds the Cramer-von Mises test (``n_boot`` samples of the
        parametric bootstrap, or of Kojadinovic and Yan's multiplier bootstrap with ``bootstrap="multiplier"``)."""
        series = [series] if isinstance(series, str) else list(series)
        if family not in FAMILIES:
            raise SpecificationError(f"family is one of {', '.join(FAMILIES)}")
        _check_dimension(family, max(len(series), 2))
        prep = _prepare(data, series, garch, dist, margins)
        params, fits, se, u, mll = _estimate(prep, family, method, garch, margins)
        fit = FittedCopula(tuple(series), family, params, u, prep.resid, prep.scale, prep.mean, bool(garch), prep.frame.index,
                           margins, method, fits, se, None, mll)
        if bootstrap not in BOOTSTRAPS:
            raise SpecificationError(f"bootstrap is one of {', '.join(BOOTSTRAPS)}")
        if gof:
            fit = replace(fit, gof_result=fit.goodness_of_fit(int(n_boot), seed, bootstrap), gof_bootstrap=bootstrap)
        return fit

    @api("finance:compare_copulas")
    def compare_copulas(data: Dataset, series: Sequence[str], families: Sequence[str] | None = None, garch: bool = False,
                        dist: Literal["normal", "t"] | None = None, margins: Literal["empirical", "normal", "t", "skewt"] = "empirical",
                        method: Literal["ml", "itau", "joint"] = "ml", gof: bool = False, n_boot: int = 100, seed: int = 0,
                        bootstrap: Literal["parametric", "multiplier"] = "parametric") -> Table:
        """Copula families fitted to the same returns and margins, side by side: log likelihood, AIC, BIC, Kendall's
        tau, tail dependence and, with ``gof``, the Cramer-von Mises test (parametric or ``bootstrap="multiplier"``).
        ``families`` defaults to every family that fits the number of series (those for negative dependence when
        Kendall's tau is negative; BB6, BB8, Tawn and the rotated two-parameter families only when listed)."""
        series = [series] if isinstance(series, str) else list(series)
        if bootstrap not in BOOTSTRAPS:
            raise SpecificationError(f"bootstrap is one of {', '.join(BOOTSTRAPS)}")
        prep = _prepare(data, series, garch, dist, margins)
        d = len(series)
        if families is None:
            families = default_families(d, _mean_tau(prep.u))
            if method == "itau":
                families = [f for f in families if f in ("gaussian", "t") or len(ARCH[f].names) == 1]
        families = [families] if isinstance(families, str) else list(families)
        bad = [f for f in families if f not in FAMILIES]
        if bad:
            raise SpecificationError(f"unknown copula families {', '.join(map(repr, bad))}: they are {', '.join(FAMILIES)}")
        for f in families:
            _check_dimension(f, d)
        rows, notes = [], []
        for f in families:
            params, fits, se, u, mll = _estimate(prep, f, method, garch, margins)
            fit = FittedCopula(tuple(series), f, params, u, prep.resid, prep.scale, prep.mean, bool(garch), prep.frame.index,
                               margins, method, fits, se, None, mll)
            td = fit.tail_dependence()
            lo, hi = (td["lower"][0, 1], td["upper"][0, 1]) if isinstance(td["lower"], np.ndarray) else (td["lower"], td["upper"])
            shown = ", ".join(f"{k} {params[k]:.4g}" for k in ("nu", "theta", "delta", "psi") if k in params)
            if "R" in params:
                shown = ", ".join(filter(None, [f"rho {params['R'][0, 1]:.4g}" + (" ..." if d > 2 else ""), shown]))
            row = [shown, fit.k, fit.loglike, fit.aic, fit.bic, fit.tau, float(lo), float(hi)]
            if gof:
                g = fit.goodness_of_fit(int(n_boot), seed, bootstrap)
                row += [g["statistic"], g["pvalue"]]
            rows.append(row)
        aic = [r[3] for r in rows]
        bic = [r[4] for r in rows]
        notes.append(f"Lowest AIC: {families[int(np.argmin(aic))]}; lowest BIC: {families[int(np.argmin(bic))]}.")
        what = "GARCH(1,1) standardised residuals" if garch else "the returns"
        notes.append(f"Copulas of {what} of {', '.join(series)}, {len(prep.u)} periods, margins {margins}, method {method}; "
                     "log likelihood, AIC and BIC of the copula. Tail dependence and Kendall's tau of the first two series.")
        if gof:
            notes.append(f"Goodness of fit: Cramer-von Mises statistic and {bootstrap} bootstrap p-value ({n_boot} samples, seed {seed}).")
        cols = ["parameters", "k", "log likelihood", "AIC", "BIC", "Kendall's tau", "lower tail", "upper tail"] + (["CvM", "p-value"] if gof else [])
        return Table(cols, rows, index=list(families), index_name="family", title="Copula families compared", notes=tuple(notes))


with plugin_context(PLUGIN):  # a recorded call: the Copula portfolio VaR dialog (app.py) builds it
    FittedCopula.portfolio_var = api("finance:fitted_copula.portfolio_var")(FittedCopula.portfolio_var)  # type: ignore[method-assign]
