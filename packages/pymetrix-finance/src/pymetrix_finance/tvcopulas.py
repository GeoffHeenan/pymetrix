"""Time-varying copulas (Patton 2006): a bivariate copula whose dependence parameter follows an ARMA-type recursion.

    tv = tv_copula(data, ["a", "b"], family="gaussian")           # the correlation's path
    tv.summary(); tv.path                                         # parameters with standard errors; rho_t by period
    tv.forecast()                                                 # next period's correlation and tail dependence
    tv.simulate(horizon=10, n=5000)                               # forward paths of the uniforms and the parameter
    tv_copula(data, ["a", "b"], family="sjc")                     # symmetrised Joe-Clayton, both tails moving
    dcc = dcc_copula(data, ["a", "b", "c"], family="t")           # several series: DCC correlations of a t copula
    dcc.path; dcc.forecast(10); dcc.portfolio_var([0.4, 0.3, 0.3])

**The Gaussian and t** (the t with constant degrees of freedom nu): the correlation follows

    rho_t = L(omega + beta rho_{t-1} + alpha (1/m) sum_{j=1..m} q(u_{t-j}) q(v_{t-j})),

L(x) = (1 - e^-x) / (1 + e^-x) Patton's modified logistic, which keeps rho in (-1, 1), and q the standard normal
quantile (the t's with nu degrees of freedom for the t copula), over m = ``lags`` periods (10 in Patton).

**Clayton, Gumbel and the symmetrised Joe-Clayton (SJC)**: the tail dependence follows

    lambda_t = Lambda(omega + beta lambda_{t-1} + alpha (1/m) sum_{j=1..m} |u_{t-j} - v_{t-j}|),

Lambda the logistic, 1 / (1 + e^-x), into (0, 1): Clayton's lower tail (theta = -1 / log2 lambda), Gumbel's upper
(theta = 1 / log2(2 - lambda)), and the SJC's two, each with its own (omega, alpha, beta). The SJC copula is
(C_JC(u, v | lambda_U, lambda_L) + C_JC(1 - u, 1 - v | lambda_L, lambda_U) + u + v - 1) / 2, C_JC the Joe-Clayton
(the plugin's BB7, theta = 1 / log2(2 - lambda_U), delta = -1 / log2 lambda_L). The tails are kept in
[1e-6, 0.99].

**Start-up**: before the sample the forcing variable is its sample mean and the parameter the recursion's fixed
point there, p_0 = L(omega + beta p_0 + alpha mean); so with alpha = beta = 0 every period has L(omega) and the
model is the constant copula. **Estimation** is maximum likelihood on the uniforms (ranks, fitted or GARCH margins,
as for ``copula``), from the constant copula's value and a few other starts; standard errors from the observed
information (a numerical Hessian), treating the uniforms as known. The filtered path is a series by period, with
the one-step-ahead forecast; ``simulate`` draws forward, each period's uniforms from the copula at its parameter
and the parameter updated from them.

**The DCC copula** (``dcc_copula``) is a Gaussian or t copula of two or more series whose correlation matrix follows
Engle's (2002) DCC recursion on the copula's scores z_t (the normal or t(nu) quantiles of the uniforms):
Q_t = (1 - a - b) Qbar + a z_(t-1) z_(t-1)' + b Q_(t-1), Q_1 = Qbar = z'z / T, R_t = Q_t scaled to unit diagonal; the
recursion is ``mgarch``'s (``dcc_path_linear``, the same as its ``dcc_path`` run for every element at once), here on
the copula's scores rather than GARCH residuals, so the margins stay free. a, b (a, b >= 0, a + b < 1) and the t's nu
by maximum likelihood (Nelder-Mead from several starts), standard errors from the observed information with the
uniforms and Qbar taken as known; with a = b = 0 it is the constant copula with correlation Qbar scaled (for ranks,
the Gaussian copula fit's exactly), whose likelihood is reported beside. Forecasts are the recursion's next
correlation, then Engle and Sheppard's R_(T+h) = Rbar + (a + b)^(h-1) (R_(T+1) - Rbar); ``simulate`` draws forward,
updating Q from the simulated scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats as sps
from scipy.special import ndtri, stdtrit

from pymetrix.errors import SpecificationError
from pymetrix.extension import api, plugin_context, random_number_generator
from pymetrix.output import Table
from . import PLUGIN
from .copulas import ARCH, MARGINS, Margin, _portfolio_from_uniforms, _prepare, _quiet, _se_from, gumbel_logpdf

if TYPE_CHECKING:
    from pymetrix.data import Dataset
    from pymetrix.figures import Figure

TV_FAMILIES = ("gaussian", "t", "clayton", "gumbel", "sjc")
_TAIL_LO, _TAIL_HI = 1e-6, 0.99


def modified_logistic(x: Any) -> Any:
    """Patton's L(x) = (1 - e^-x) / (1 + e^-x) = tanh(x / 2), into (-1, 1)."""
    return np.tanh(np.asarray(x, float) / 2)


def logistic(x: Any) -> Any:
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def param_names(family: str) -> tuple[str, ...]:
    if family == "t":
        return ("omega", "alpha", "beta", "nu")
    if family == "sjc":
        return ("omega_U", "alpha_U", "beta_U", "omega_L", "alpha_L", "beta_L")
    return ("omega", "alpha", "beta")


def _forcing(u: np.ndarray, v: np.ndarray, family: str, nu: float | None) -> np.ndarray:
    if family == "gaussian":
        return ndtri(u) * ndtri(v)
    if family == "t":
        return stdtrit(nu, u) * stdtrit(nu, v)
    return np.abs(u - v)


def _lagged_mean(f: np.ndarray, m: int, fill: float) -> np.ndarray:
    """(1/m) sum_{j=1..m} f_{t-j} for t = 0..T (the last the next period's), pre-sample values ``fill``."""
    padded = np.r_[np.full(m, fill), f]
    c = np.r_[0.0, np.cumsum(padded)]
    return (c[m:] - c[:-m]) / m  # T + 1 values: windows ending before 0..T


def _fixed_point(omega: float, alpha: float, beta: float, fbar: float, link: Any, lo: float, hi: float) -> float:
    g = lambda p: float(link(omega + beta * p + alpha * fbar)) - p  # noqa: E731
    if g(lo) <= 0:
        return lo
    if g(hi) >= 0:
        return hi
    return float(optimize.brentq(g, lo, hi, xtol=1e-15, rtol=1e-15))


def _recursion(omega: float, alpha: float, beta: float, xbar: np.ndarray, link: Any, p0: float, lo: float, hi: float) -> np.ndarray:
    """p_t = link(omega + beta p_{t-1} + alpha xbar_t), kept in [lo, hi] (scalar arithmetic: the loop is the cost)."""
    out = [0.0] * len(xbar)
    prev = p0
    a = (alpha * np.asarray(xbar, float) + omega).tolist()
    tanh, exp = math.tanh, math.exp
    elliptical = link is modified_logistic
    for t, z in enumerate(a):
        x = z + beta * prev
        if elliptical:
            prev = tanh(x / 2)
        else:
            prev = 1 / (1 + exp(-x)) if x > -700 else 0.0
        prev = lo if prev < lo else hi if prev > hi else prev
        out[t] = prev
    return np.array(out)


def tv_filter(u: np.ndarray, v: np.ndarray, family: str, params: Sequence[float], lags: int = 10) -> dict[str, np.ndarray]:
    """The parameter's path for t = 1..T and the next period's (T + 1 values): ``rho`` (Gaussian, t), ``lower``
    and/or ``upper`` tail dependence (Clayton, Gumbel, SJC)."""
    p = [float(x) for x in params]
    nu = p[3] if family == "t" else None
    f = _forcing(u, v, family, nu)
    fbar = float(np.mean(f))
    xbar = _lagged_mean(f, int(lags), fbar)
    if family in ("gaussian", "t"):
        lo, hi = -0.9999, 0.9999
        p0 = _fixed_point(p[0], p[1], p[2], fbar, modified_logistic, lo, hi)
        return {"rho": _recursion(p[0], p[1], p[2], xbar, modified_logistic, p0, lo, hi)}
    out = {}
    blocks = {"clayton": [("lower", 0)], "gumbel": [("upper", 0)], "sjc": [("upper", 0), ("lower", 3)]}[family]
    for name, k in blocks:
        p0 = _fixed_point(p[k], p[k + 1], p[k + 2], fbar, logistic, _TAIL_LO, _TAIL_HI)
        out[name] = _recursion(p[k], p[k + 1], p[k + 2], xbar, logistic, p0, _TAIL_LO, _TAIL_HI)
    return out


def _bb7_from_tails(upper: Any, lower: Any) -> tuple[Any, Any]:
    return 1 / np.log2(2 - upper), -1 / np.log2(lower)


@_quiet
def tv_logpdf(u: np.ndarray, v: np.ndarray, family: str, path: dict[str, np.ndarray], nu: float | None = None) -> np.ndarray:
    """log c(u_t, v_t | parameter_t), the path's first T values."""
    T = len(u)
    if family == "gaussian":
        r = path["rho"][:T]
        x, y = ndtri(u), ndtri(v)
        return -0.5 * np.log1p(-r * r) - (r * r * (x * x + y * y) - 2 * r * x * y) / (2 * (1 - r * r))
    if family == "t":
        r = path["rho"][:T]
        x, y = stdtrit(nu, u), stdtrit(nu, v)
        from scipy.special import gammaln

        q = (x * x + y * y - 2 * r * x * y) / (nu * (1 - r * r))
        return (gammaln((nu + 2) / 2) + gammaln(nu / 2) - 2 * gammaln((nu + 1) / 2) - 0.5 * np.log1p(-r * r)
                - (nu + 2) / 2 * np.log1p(q) + (nu + 1) / 2 * (np.log1p(x * x / nu) + np.log1p(y * y / nu)))
    if family == "clayton":
        th = -1 / np.log2(path["lower"][:T])
        s = u ** -th + v ** -th - 1
        return np.log1p(th) - (th + 1) * (np.log(u) + np.log(v)) - (2 + 1 / th) * np.log(s)
    if family == "gumbel":
        th = 1 / np.log2(2 - path["upper"][:T])
        return gumbel_logpdf(np.column_stack([u, v]), th)
    lu, ll = path["upper"][:T], path["lower"][:T]
    a = _bb7_logpdf(u, v, _bb7_from_tails(lu, ll))
    b = _bb7_logpdf(1 - u, 1 - v, _bb7_from_tails(ll, lu))
    return np.logaddexp(a, b) - np.log(2)


def _bb7_logpdf(u: np.ndarray, v: np.ndarray, p: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """The Joe-Clayton (BB7) log density with a parameter for each observation."""
    bb7 = ARCH["bb7"]
    return bb7.log_dpsi(bb7.phi(u, p) + bb7.phi(v, p), p, 2) + bb7.log_dphi(u, p) + bb7.log_dphi(v, p)


def _bb7_h(u: np.ndarray, v: np.ndarray, p: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    bb7 = ARCH["bb7"]
    u, v = np.clip(u, 1e-12, 1 - 1e-12), np.clip(v, 1e-12, 1 - 1e-12)
    a = bb7.phi(u, p)
    return np.exp(bb7.log_dpsi(a + bb7.phi(v, p), p, 1) - bb7.log_dpsi(a, p, 1))


def tv_loglike(u: np.ndarray, v: np.ndarray, family: str, params: Sequence[float], lags: int = 10) -> float:
    path = tv_filter(u, v, family, params, lags)
    return float(np.sum(tv_logpdf(u, v, family, path, params[3] if family == "t" else None)))


def _bounds(family: str) -> list[tuple[float, float]]:
    block = [(-10.0, 10.0), (-25.0, 25.0), (-5.0, 5.0)]
    if family == "t":
        return block + [(2.05, 100.0)]
    if family == "sjc":
        return block + block
    return block


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def _starts(u: np.ndarray, v: np.ndarray, family: str) -> list[list[float]]:
    tau = float(sps.kendalltau(u, v)[0])
    if family in ("gaussian", "t"):
        r = float(np.clip(np.sin(np.pi / 2 * tau), -0.95, 0.95))
        w = 2 * np.arctanh(r)
        extra = [8.0] if family == "t" else []
        return [[w, 0.0, 0.0, *extra], [0.3 * w, 0.1, 1.2, *extra], [0.1 * w, 0.05, 1.8, *extra], [w, -0.2, 0.5, *extra]]
    lower = 2 ** (-1 / max(2 * tau / (1 - tau), 0.05)) if tau > 0 else 0.01
    upper = 2 - 2 ** (1 - max(tau, 0.02)) if tau > 0 else 0.01
    out = []
    for p in {"clayton": [lower], "gumbel": [upper], "sjc": [upper, lower]}[family]:
        z = _logit(p)
        out.append([[z, 0.0, 0.0], [z + 1.0, -2.0, 0.5], [z, -4.0, 1.0]])
    if family == "sjc":
        return [a + b for a, b in zip(out[0], out[1])]
    return out[0]


@_quiet
def fit_tv_copula(u: np.ndarray, v: np.ndarray, family: str = "gaussian", lags: int = 10) -> dict[str, Any]:
    """The recursion's parameters by maximum likelihood, the log likelihood and standard errors."""
    if family not in TV_FAMILIES:
        raise SpecificationError(f"family is one of {', '.join(TV_FAMILIES)}")
    if int(lags) < 1:
        raise SpecificationError("lags is at least 1")
    u, v = np.clip(np.asarray(u, float), 1e-10, 1 - 1e-10), np.clip(np.asarray(v, float), 1e-10, 1 - 1e-10)
    bounds = _bounds(family)

    def negll(x: np.ndarray) -> float:
        out = -tv_loglike(u, v, family, x, lags)
        return out if np.isfinite(out) else 1e300

    best = None
    for x0 in _starts(u, v, family):
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        found = optimize.minimize(negll, x0, method="L-BFGS-B", bounds=bounds)
        if best is None or found.fun < best.fun:
            best = found
    names = list(param_names(family))
    x = np.array(best.x, float)
    se = _se_from(negll, x, names)
    return {"params": dict(zip(names, map(float, x))), "loglike": float(-best.fun), "se": se}


# -- simulation --------------------------------------------------------------------------------------------------------

def _hinv_bisect(h: Any, u: np.ndarray, w: np.ndarray) -> np.ndarray:
    lo, hi = np.zeros_like(w), np.ones_like(w)
    for _ in range(55):
        mid = (lo + hi) / 2
        below = h(u, mid) < w
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return (lo + hi) / 2


@_quiet
def _draw(family: str, par: dict[str, np.ndarray], nu: float | None, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One draw of (u, v) for each path, at the paths' own parameters."""
    n = len(next(iter(par.values())))
    if family in ("gaussian", "t"):
        r = par["rho"]
        z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
        x, y = z1, r * z1 + np.sqrt(1 - r * r) * z2
        if family == "gaussian":
            return sps.norm.cdf(x), sps.norm.cdf(y)
        s = np.sqrt(rng.chisquare(nu, n) / nu)
        return sps.t.cdf(x / s, nu), sps.t.cdf(y / s, nu)
    u, w = rng.uniform(size=n), rng.uniform(size=n)
    if family == "clayton":
        th = -1 / np.log2(par["lower"])
        return u, ((w ** (-th / (1 + th)) - 1) * u ** (-th) + 1) ** (-1 / th)
    if family == "gumbel":
        th = 1 / np.log2(2 - par["upper"])

        def h(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # dC/du: C(u, v) A^(1 - theta) x^(theta - 1) / u
            b = np.clip(b, 1e-300, 1.0)
            x, y = -np.log(a), -np.log(b)
            s = x**th + y**th
            return np.exp(-s ** (1 / th)) * s ** (1 / th - 1) * x ** (th - 1) / a

        return u, _hinv_bisect(h, u, w)
    lu, ll = par["upper"], par["lower"]
    pick = rng.uniform(size=n) < 0.5  # the SJC is an equal mixture of the Joe-Clayton and its survival copula
    th1, de1 = _bb7_from_tails(lu, ll)
    th2, de2 = _bb7_from_tails(ll, lu)
    th, de = np.where(pick, th1, th2), np.where(pick, de1, de2)
    a = np.where(pick, u, 1 - u)
    b = _hinv_bisect(lambda x, y: _bb7_h(x, y, (th, de)), a, w)
    return np.where(pick, a, 1 - a), np.where(pick, b, 1 - b)


# -- the fit -----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FittedTVCopula:
    """A time-varying bivariate copula (Patton 2006): the recursion's parameters and the filtered path."""

    series: tuple[str, ...]
    family: str
    params: dict[str, float]
    se: dict[str, float]
    loglike: float
    u: np.ndarray = field(repr=False)
    residuals: np.ndarray = field(repr=False)
    scale_next: np.ndarray = field(repr=False)
    mean_next: np.ndarray = field(repr=False)
    lags: int = 10
    garch: bool = False
    index: Any = None
    margins: str = "empirical"
    margin_fits: tuple[Margin, ...] = ()
    constant_loglike: float | None = None

    stage = "finance"

    @property
    def _x(self) -> list[float]:
        return [self.params[k] for k in param_names(self.family)]

    @property
    def nu(self) -> float | None:
        return self.params.get("nu")

    def _filtered(self) -> dict[str, np.ndarray]:
        return tv_filter(self.u[:, 0], self.u[:, 1], self.family, self._x, self.lags)

    @property
    def path(self) -> pd.DataFrame:
        """The filtered parameter by period: the correlation (and the t's tail dependence) or the tail dependence."""
        f = self._filtered()
        T = len(self.u)
        cols = {k: v[:T] for k, v in f.items()}
        if self.family == "t":
            r, nu = cols["rho"], self.nu
            cols["tail"] = 2 * sps.t.cdf(-np.sqrt((nu + 1) * (1 - r) / (1 + r)), nu + 1)
        if self.family == "clayton":
            cols["theta"] = -1 / np.log2(cols["lower"])
        if self.family == "gumbel":
            cols["theta"] = 1 / np.log2(2 - cols["upper"])
        return pd.DataFrame(cols, index=self.index)

    def forecast(self) -> dict[str, float]:
        """Next period's parameter (and tail dependence), from the recursion at the end of the sample."""
        f = {k: float(v[-1]) for k, v in self._filtered().items()}
        if self.family == "t":
            r, nu = f["rho"], self.nu
            f["tail"] = float(2 * sps.t.cdf(-np.sqrt((nu + 1) * (1 - r) / (1 + r)), nu + 1))
        return f

    @property
    def k(self) -> int:
        return len(self.params)

    @property
    def aic(self) -> float:
        return 2 * self.k - 2 * self.loglike

    @property
    def bic(self) -> float:
        return self.k * np.log(len(self.u)) - 2 * self.loglike

    def simulate(self, horizon: int = 1, n: int = 10_000, seed: int | None = 0) -> dict[str, np.ndarray]:
        """``n`` paths ``horizon`` periods forward: ``u`` (n, horizon, 2) and the parameter(s) each period (n, horizon),
        the first period's from the forecast, each later one's from the recursion on the simulated uniforms."""
        rng = random_number_generator(seed)
        h, n, m = int(horizon), int(n), self.lags
        x = self._x
        nu = self.nu
        fam = self.family
        f_hist = _forcing(self.u[:, 0], self.u[:, 1], fam, nu)
        window = np.tile(np.r_[np.full(max(0, m - len(f_hist)), np.mean(f_hist)), f_hist[-m:]], (n, 1))
        current = {k: np.full(n, v) for k, v in self.forecast().items() if k != "tail"}
        U = np.empty((n, h, 2))
        paths = {k: np.empty((n, h)) for k in current}
        for step in range(h):
            for k in current:
                paths[k][:, step] = current[k]
            a, b = _draw(fam, current, nu, rng)
            U[:, step, 0], U[:, step, 1] = a, b
            window = np.column_stack([window[:, 1:], _forcing(np.clip(a, 1e-12, 1 - 1e-12), np.clip(b, 1e-12, 1 - 1e-12), fam, nu)])
            xbar = window.mean(axis=1)
            if fam in ("gaussian", "t"):
                current = {"rho": np.clip(modified_logistic(x[0] + x[2] * current["rho"] + x[1] * xbar), -0.9999, 0.9999)}
            else:
                blocks = {"clayton": [("lower", 0)], "gumbel": [("upper", 0)], "sjc": [("upper", 0), ("lower", 3)]}[fam]
                current = {name: np.clip(logistic(x[k] + x[k + 2] * current[name] + x[k + 1] * xbar), _TAIL_LO, _TAIL_HI)
                           for name, k in blocks}
        return {"u": U, **paths}

    def portfolio_var(self, weights: Sequence[float], level: float = 0.99, n: int = 100_000, margins: str | None = None,
                      seed: int | None = 0) -> dict[str, float]:
        """Next period's VaR and ES of a portfolio, the copula at its forecast parameter (margins as for ``copula``)."""
        return _portfolio_from_uniforms(self, self.simulate(1, n, seed)["u"][:, 0, :], weights, level, margins)

    def summary(self) -> Table:
        rows, names = [], []
        for k in param_names(self.family):
            names.append(k)
            rows.append([self.params[k], self.se.get(k, np.nan)])
        path = self.path
        for col in path.columns:
            names.append(f"{col}: mean of the path")
            rows.append([float(path[col].mean()), np.nan])
        for key, value in self.forecast().items():
            names.append(f"{key}: next period")
            rows.append([value, np.nan])
        what = "GARCH(1,1) standardised residuals" if self.garch else "the returns"
        how = {"empirical": "their ranks", "normal": "fitted normal margins", "t": "fitted t margins",
               "skewt": "fitted skewed-t margins (Hansen)"}[self.margins]
        driver = ("rho_t = L(omega + beta rho_{t-1} + alpha mean of q(u) q(v) over the last "
                  f"{self.lags} periods), L the modified logistic" if self.family in ("gaussian", "t") else
                  f"tail dependence = logistic(omega + beta lambda_(t-1) + alpha mean |u - v| over the last {self.lags} periods)")
        name = {"gaussian": "Gaussian", "t": "t", "clayton": "Clayton", "gumbel": "Gumbel", "sjc": "Symmetrised Joe-Clayton"}[self.family]
        notes = [f"Time-varying {name} copula (Patton 2006) of {what} of {', '.join(self.series)} by {how}, {len(self.u)} periods: {driver}.",
                 f"Log likelihood: {self.loglike:.4f}", f"Akaike info criterion: {self.aic:.4f}",
                 f"Schwarz criterion: {self.bic:.4f}",
                 *([f"Log likelihood (constant copula): {self.constant_loglike:.4f}"] if self.constant_loglike is not None else []),
                 "Standard errors from the observed information, treating the uniforms as known."]
        if self.family not in ("gaussian", "t"):
            path = self.path
            at = float(np.mean([((path[c] <= _TAIL_LO * 1.0001) | (path[c] >= _TAIL_HI * 0.9999)).mean() for c in ("lower", "upper") if c in path]))
            if at > 0:
                notes.append(f"The tail dependence is at its bounds ({_TAIL_LO:g} or {_TAIL_HI:g}) in {100 * at:.1f}% of the periods: "
                             "the likelihood has kinks there and the standard errors are unreliable.")
        return Table(["value", "std err"], rows, index=names, index_name="parameter", title="Time-varying copula", notes=tuple(notes))

    def plot(self) -> Figure:
        from pymetrix.figures import Figure, Line, Panel

        path = self.path
        lines = tuple(Line(path[c], c) for c in path.columns if c != "theta")
        return Figure((Panel(lines, title=f"Time-varying {self.family} copula: {', '.join(self.series)}", ylabel="dependence"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return self.path

    def __repr__(self) -> str:
        return f"<FittedTVCopula {self.family} of {', '.join(self.series)}>"


# -- DCC copulas: the Gaussian or t copula of several series with a DCC correlation ------------------------------------

DCC_FAMILIES = ("gaussian", "t")


def copula_scores(u: np.ndarray, family: str, nu: float | None = None) -> np.ndarray:
    """The copula's quantile scores: the standard normal (Gaussian) or t(nu) quantiles of the uniforms."""
    return ndtri(u) if family == "gaussian" else stdtrit(nu, u)


def dcc_copula_filter(u: np.ndarray, family: str, a: float, b: float, nu: float | None = None) -> dict[str, np.ndarray]:
    """The DCC recursion (Engle 2002, ``mgarch``'s, by its ``dcc_path_linear``) on the copula's scores z_t:
    Q_t = (1 - a - b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1}, Q_1 = Qbar = z'z / T, R_t = Q_t scaled to unit diagonal.
    ``R`` (T, d, d), next period's ``Q_next`` and ``R_next``, and ``Qbar``."""
    from .mgarch import _scale_to_unit, dcc_path_linear

    z = copula_scores(u, family, nu)
    Qbar = z.T @ z / len(z)
    R, Q_next = dcc_path_linear(z, a, b, Qbar)
    return {"R": R, "Q_next": Q_next, "R_next": _scale_to_unit(Q_next), "Qbar": Qbar, "z": z}


def dcc_copula_logpdf(z: np.ndarray, R: np.ndarray, family: str, nu: float | None = None) -> np.ndarray:
    """log c(u_t | R_t) period by period from the scores z (T, d) and correlations R (T, d, d)."""
    sign, logdet = np.linalg.slogdet(R)
    q = np.einsum("ti,tij,tj->t", z, np.linalg.inv(R), z)
    if family == "gaussian":
        return -0.5 * logdet - 0.5 * (q - np.sum(z * z, axis=1))
    from scipy.special import gammaln

    d = z.shape[1]
    joint = gammaln((nu + d) / 2) - gammaln(nu / 2) - d / 2 * np.log(nu * np.pi) - 0.5 * logdet - (nu + d) / 2 * np.log1p(q / nu)
    return joint - sps.t.logpdf(z, nu).sum(axis=1)


def _dcc_negll(x: np.ndarray, u: np.ndarray, family: str) -> float:
    a, b = float(x[0]), float(x[1])
    nu = float(x[2]) if family == "t" else None
    if a < 0 or b < 0 or a + b >= 1 or (nu is not None and not 2.05 <= nu <= 200.0):
        return 1e12
    f = dcc_copula_filter(u, family, a, b, nu)
    if np.any(np.linalg.slogdet(f["R"])[0] <= 0):
        return 1e12
    out = -float(np.sum(dcc_copula_logpdf(f["z"], f["R"], family, nu)))
    return out if np.isfinite(out) else 1e12


@_quiet
def fit_dcc_copula(u: np.ndarray, family: str = "gaussian") -> dict[str, Any]:
    """The DCC copula's a, b (and the t's nu) by maximum likelihood on the uniforms (Nelder-Mead from several starts,
    a, b >= 0, a + b < 1), its log likelihood, standard errors (observed information, the uniforms taken as known)
    and the constant copula's log likelihood (a = b = 0: the correlation matrix Qbar scaled, the t's nu refitted)."""
    if family not in DCC_FAMILIES:
        raise SpecificationError(f"family is one of {', '.join(DCC_FAMILIES)}")
    u = np.clip(np.asarray(u, float), 1e-10, 1 - 1e-10)
    nus = [8.0] if family == "t" else []
    best = None
    for start in ([0.03, 0.95], [0.05, 0.85], [0.02, 0.5], [0.005, 0.0]):
        found = optimize.minimize(_dcc_negll, np.array(start + nus), args=(u, family), method="Nelder-Mead",
                                  options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 8000})
        if best is None or found.fun < best.fun:
            best = found
    names = ["a", "b"] + (["nu"] if family == "t" else [])
    x = np.array(best.x, float)
    se = _se_from(lambda v: _dcc_negll(v, u, family), x, names)
    if family == "t":
        const = optimize.minimize_scalar(lambda v: _dcc_negll(np.array([0.0, 0.0, v]), u, family), bounds=(2.05, 200.0),
                                         method="bounded", options={"xatol": 1e-7})
        constant = float(-const.fun)
    else:
        constant = -_dcc_negll(np.array([0.0, 0.0]), u, family)
    return {"params": dict(zip(names, map(float, x))), "loglike": float(-best.fun), "se": se, "constant_loglike": constant}


@dataclass(frozen=True, eq=False)
class FittedDCCCopula:
    """A Gaussian or t copula of several series whose correlation matrix follows a DCC recursion on the copula's
    scores: the parameters, the filtered correlations, forecasts and simulation."""

    series: tuple[str, ...]
    family: str
    params: dict[str, float]
    se: dict[str, float]
    loglike: float
    u: np.ndarray = field(repr=False)
    residuals: np.ndarray = field(repr=False)
    scale_next: np.ndarray = field(repr=False)
    mean_next: np.ndarray = field(repr=False)
    garch: bool = False
    index: Any = None
    margins: str = "empirical"
    margin_fits: tuple[Margin, ...] = ()
    constant_loglike: float | None = None

    stage = "finance"

    @property
    def nu(self) -> float | None:
        return self.params.get("nu")

    def _filtered(self) -> dict[str, np.ndarray]:
        return dcc_copula_filter(np.clip(self.u, 1e-10, 1 - 1e-10), self.family, self.params["a"], self.params["b"], self.nu)

    def correlation_matrices(self) -> np.ndarray:
        """R_t for every period, (T, d, d)."""
        return self._filtered()["R"]

    @property
    def path(self) -> pd.DataFrame:
        """The filtered correlation of every pair by period."""
        R = self.correlation_matrices()
        d = len(self.series)
        cols = {f"{self.series[i]}, {self.series[j]}": R[:, i, j] for i in range(d) for j in range(i + 1, d)}
        return pd.DataFrame(cols, index=self.index)

    def forecast(self, horizon: int = 10) -> np.ndarray:
        """The correlation matrices 1 to ``horizon`` periods after the sample, (horizon, d, d): the recursion's next
        one, then Engle and Sheppard's approximation R_{T+h} = Rbar + (a + b)^(h-1) (R_{T+1} - Rbar), as ``mgarch``."""
        from .mgarch import _scale_to_unit

        f = self._filtered()
        Rbar, R1 = _scale_to_unit(f["Qbar"]), f["R_next"]
        p = self.params["a"] + self.params["b"]
        return np.stack([Rbar + p**h * (R1 - Rbar) for h in range(int(horizon))])

    @property
    def k(self) -> int:
        d = len(self.series)
        return len(self.params) + d * (d - 1) // 2  # with the correlations targeted (Qbar)

    @property
    def aic(self) -> float:
        return 2 * self.k - 2 * self.loglike

    @property
    def bic(self) -> float:
        return self.k * np.log(len(self.u)) - 2 * self.loglike

    def simulate(self, horizon: int = 1, n: int = 10_000, seed: int | None = 0) -> dict[str, np.ndarray]:
        """``n`` paths ``horizon`` periods forward: ``u`` (n, horizon, d) and ``R`` (n, horizon, d, d), the first
        period's correlation the forecast, each later one's from the recursion on the simulated scores."""
        rng = random_number_generator(seed)
        f = self._filtered()
        a, b, nu = self.params["a"], self.params["b"], self.nu
        h, n, d = int(horizon), int(n), len(self.series)
        Qbar = f["Qbar"]
        Q = np.broadcast_to(f["Q_next"], (n, d, d)).copy()
        U = np.empty((n, h, d))
        Rs = np.empty((n, h, d, d))
        for step in range(h):
            s = 1 / np.sqrt(np.einsum("nii->ni", Q))
            R = Q * s[:, :, None] * s[:, None, :]
            Rs[:, step] = R
            x = np.einsum("nij,nj->ni", np.linalg.cholesky(R), rng.standard_normal((n, d)))
            if self.family == "t":
                x = x / np.sqrt(rng.chisquare(nu, (n, 1)) / nu)
                U[:, step] = sps.t.cdf(x, nu)
            else:
                U[:, step] = sps.norm.cdf(x)
            Q = (1 - a - b) * Qbar + a * x[:, :, None] * x[:, None, :] + b * Q
        return {"u": U, "R": Rs}

    def portfolio_var(self, weights: Sequence[float], level: float = 0.99, n: int = 100_000, margins: str | None = None,
                      seed: int | None = 0) -> dict[str, float]:
        """Next period's VaR and ES of a portfolio, the copula at its forecast correlation (margins as for ``copula``)."""
        return _portfolio_from_uniforms(self, self.simulate(1, n, seed)["u"][:, 0, :], weights, level, margins)

    def summary(self) -> Table:
        names, rows = [], []
        for k, v in self.params.items():
            names.append(k)
            rows.append([v, self.se.get(k, np.nan)])
        path = self.path
        R1 = self.forecast(1)[0]
        d = len(self.series)
        for i in range(d):
            for j in range(i + 1, d):
                col = f"{self.series[i]}, {self.series[j]}"
                names += [f"rho {col}: mean of the path", f"rho {col}: next period"]
                rows += [[float(path[col].mean()), np.nan], [float(R1[i, j]), np.nan]]
        what = "GARCH(1,1) standardised residuals" if self.garch else "the returns"
        how = {"empirical": "their ranks", "normal": "fitted normal margins", "t": "fitted t margins",
               "skewt": "fitted skewed-t margins (Hansen)"}[self.margins]
        name = "Gaussian" if self.family == "gaussian" else "t"
        lr = 2 * (self.loglike - self.constant_loglike) if self.constant_loglike is not None else None
        notes = [f"DCC {name} copula of {what} of {', '.join(self.series)} by {how}, {len(self.u)} periods: "
                 "Q_t = (1 - a - b) Qbar + a z_(t-1) z_(t-1)' + b Q_(t-1), z the copula's "
                 + ("normal" if self.family == "gaussian" else "t") + " scores, Qbar = z'z / T, R_t = Q_t scaled to unit diagonal.",
                 f"Log likelihood: {self.loglike:.4f}", f"Akaike info criterion: {self.aic:.4f}",
                 f"Schwarz criterion: {self.bic:.4f}",
                 *([f"Log likelihood (constant copula, a = b = 0): {self.constant_loglike:.4f}",
                    f"LR statistic: {lr:.4f}"] if lr is not None else []),
                 "Standard errors from the observed information, treating the uniforms (and Qbar) as known."]
        return Table(["value", "std err"], rows, index=names, index_name="parameter", title="DCC copula", notes=tuple(notes))

    def plot(self) -> Figure:
        from pymetrix.figures import Figure, Line, Panel

        path = self.path
        return Figure((Panel(tuple(Line(path[c], c) for c in path.columns),
                             title=f"DCC {self.family} copula: correlations", ylabel="correlation"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return self.path

    def __repr__(self) -> str:
        return f"<FittedDCCCopula {self.family} of {', '.join(self.series)}>"


with plugin_context(PLUGIN):
    @api("finance:dcc_copula")
    def dcc_copula(data: Dataset, series: Sequence[str], family: Literal["gaussian", "t"] = "gaussian", garch: bool = False,
                   dist: Literal["normal", "t"] | None = None,
                   margins: Literal["empirical", "normal", "t", "skewt"] = "empirical") -> FittedDCCCopula:
        """A time-varying Gaussian or t copula of two or more returns whose correlation matrix follows Engle's (2002)
        DCC recursion on the copula's scores (the normal or t quantiles of the uniforms), by maximum likelihood on the
        margins' uniforms (ranks, fitted, or with ``garch`` the standardised residuals), with the filtered
        correlations, forecasts, forward simulation and portfolio VaR."""
        series = [series] if isinstance(series, str) else list(series)
        if family not in DCC_FAMILIES:
            raise SpecificationError(f"family is one of {', '.join(DCC_FAMILIES)}")
        if margins not in MARGINS:
            raise SpecificationError(f"margins is one of {', '.join(MARGINS)}")
        prep = _prepare(data, series, garch, dist, margins)
        fit = fit_dcc_copula(prep.u, family)
        return FittedDCCCopula(tuple(series), family, fit["params"], fit["se"], fit["loglike"], prep.u, prep.resid, prep.scale,
                               prep.mean, bool(garch), prep.frame.index, margins, prep.margin_fits, fit["constant_loglike"])

    @api("finance:tv_copula")
    def tv_copula(data: Dataset, series: Sequence[str], family: Literal["gaussian", "t", "clayton", "gumbel", "sjc"] = "gaussian",
                  lags: int = 10, garch: bool = False, dist: Literal["normal", "t"] | None = None,
                  margins: Literal["empirical", "normal", "t", "skewt"] = "empirical") -> FittedTVCopula:
        """A time-varying copula of two returns (Patton 2006): the Gaussian or t correlation, or the Clayton, Gumbel
        or symmetrised Joe-Clayton tail dependence, following an ARMA-type recursion on the last ``lags`` periods;
        by maximum likelihood on the margins' uniforms (ranks, fitted, or with ``garch`` the standardised residuals),
        with the filtered path, next period's forecast and forward simulation."""
        series = [series] if isinstance(series, str) else list(series)
        if len(series) != 2:
            raise SpecificationError("a time-varying copula here is of two series")
        if family not in TV_FAMILIES:
            raise SpecificationError(f"family is one of {', '.join(TV_FAMILIES)}")
        if margins not in MARGINS:
            raise SpecificationError(f"margins is one of {', '.join(MARGINS)}")
        prep = _prepare(data, series, garch, dist, margins)
        fit = fit_tv_copula(prep.u[:, 0], prep.u[:, 1], family, int(lags))
        const = {k: v for k, v in fit["params"].items()}
        x = list(const.values())
        x[1:3] = [0.0, 0.0]
        if family == "sjc":
            x[4:6] = [0.0, 0.0]
        # the constant copula's likelihood: the same model with alpha = beta = 0, omega by maximum likelihood
        names = list(param_names(family))
        free = [i for i, nme in enumerate(names) if nme.startswith("omega") or nme == "nu"]

        def negll(z: np.ndarray) -> float:
            y = list(x)
            for i, val in zip(free, z):
                y[i] = val
            out = -tv_loglike(prep.u[:, 0], prep.u[:, 1], family, y, int(lags))
            return out if np.isfinite(out) else 1e300

        b = _bounds(family)
        found = None
        for start in ([x[i] for i in free], [_starts(prep.u[:, 0], prep.u[:, 1], family)[0][i] for i in free]):
            got = optimize.minimize(negll, np.clip(start, [b[i][0] for i in free], [b[i][1] for i in free]), method="L-BFGS-B",
                                    bounds=[b[i] for i in free])
            if found is None or got.fun < found.fun:
                found = got
        return FittedTVCopula(tuple(series), family, fit["params"], fit["se"], fit["loglike"], prep.u, prep.resid, prep.scale,
                              prep.mean, int(lags), bool(garch), prep.frame.index, margins, prep.margin_fits, float(-found.fun))


__all__ = ["FittedTVCopula", "FittedDCCCopula", "tv_copula", "dcc_copula", "fit_tv_copula", "fit_dcc_copula", "tv_filter"]
