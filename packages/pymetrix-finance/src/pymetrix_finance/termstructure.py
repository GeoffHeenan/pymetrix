"""Forward rates, discount factors and bonds from a fitted yield curve, and the dynamic Nelson-Siegel model.

    forwards(yc, [1, 2, 5, 10]); discount(yc, [1, 5, 10])            # from the last period's fitted curve
    bond(yc, coupon=4, maturity=10)                                   # price, yield, duration, convexity
    dns = dynamic_nelson_siegel(data, series, maturities)            # one step: the Kalman filter and maximum likelihood
    dns.summary(); dns.factors; dns.forecast(12)

**Curves.** A Nelson-Siegel (or Svensson) curve gives zero-coupon yields, read here as continuously compounded:
the instantaneous forward rate is f(m) = b0 + b1 e^(-l m) + b2 l m e^(-l m) (+ b3 l2 m e^(-l2 m)), whose average over
0 to m is the yield; the discount factor is exp(-y(m) m). Yields in percent (``percent=True``, as FRED's) are
divided by 100 first. Treasury constant-maturity yields are par yields, so reading them as zero rates is an
approximation, close for short maturities and modest coupons.

**Bonds.** A bond paying ``coupon`` (percent of face a year) ``freq`` times a year to ``maturity`` years is priced
from the curve's discount factors, per 100 of face. Its yield is the continuously compounded rate that discounts the
cash flows to that price; duration is Macaulay's at that yield (equal to the modified duration under continuous
compounding) and convexity the matching second moment; the Fisher-Weil duration is the price's sensitivity to a
parallel shift of the whole curve.

**Dynamic Nelson-Siegel** (Diebold, Rudebusch and Aruoba 2006): the factors f_t follow a VAR(1), f_t = mu + A (f_(t-1)
- mu) + e_t with e_t ~ N(0, Q), and the yields are y_t = Z(l) f_t + u_t with independent errors of a variance for
each maturity. The decay, mu, A (diagonal, or ``full``), Q and the error variances are estimated together by maximum
likelihood, from the two-step estimates; missing yields are handled by the filter. The factors are the smoothed states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import optimize

from pymetrix.extension import api
from pymetrix.errors import SpecificationError
from pymetrix.output import Table
from pymetrix.extension import plugin_context
from . import PLUGIN
from .yields import loadings

if TYPE_CHECKING:
    from pymetrix.data import Dataset
    from pymetrix.figures import Figure
    from .yields import YieldCurve


def _row(yc: Any, period: Any) -> np.ndarray:
    f = yc.factors.dropna()
    if period is None:
        return f.iloc[-1].to_numpy()
    key = pd.Period(period, f.index.freq) if isinstance(f.index, pd.PeriodIndex) else period
    return f.loc[key].to_numpy()


def forward_loadings(m: Any, lam: float, lam2: float | None = None) -> np.ndarray:
    """The instantaneous forward rate's loadings on level, slope, curvature (and the second curvature)."""
    m = np.asarray(m, float)
    cols = [np.ones_like(m), np.exp(-lam * m), lam * m * np.exp(-lam * m)]
    if lam2 is not None:
        cols.append(lam2 * m * np.exp(-lam2 * m))
    return np.column_stack(cols)


def zero_rates(yc: Any, maturities: Any, period: Any = None, percent: bool = True) -> np.ndarray:
    """The fitted zero-coupon yields at ``maturities``, as fractions."""
    y = loadings(maturities, yc.lam, yc.lam2) @ _row(yc, period)
    return y / 100 if percent else y


def forwards(yc: Any, maturities: Any, period: Any = None, percent: bool = True) -> pd.Series:
    """Instantaneous forward rates at ``maturities`` (in the yields' units) from one period's fitted curve."""
    m = np.asarray(maturities, float)
    f = forward_loadings(m, yc.lam, yc.lam2) @ _row(yc, period)
    return pd.Series(f, index=pd.Index(m, name="maturity"), name="forward")


def discount(yc: Any, maturities: Any, period: Any = None, percent: bool = True) -> pd.Series:
    """Discount factors exp(-y(m) m) at ``maturities`` from one period's fitted curve."""
    m = np.asarray(maturities, float)
    return pd.Series(np.exp(-zero_rates(yc, m, period, percent) * m), index=pd.Index(m, name="maturity"), name="discount")


def _cash_flows(coupon: float, maturity: float, freq: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(np.ceil(maturity * freq - 1e-9))
    times = maturity - np.arange(n)[::-1] / freq
    cf = np.full(n, coupon / freq)
    cf[-1] += 100.0
    return times, cf


def bond_measures(times: np.ndarray, cf: np.ndarray, zeros: np.ndarray) -> dict[str, float]:
    """Price, continuous yield, Macaulay duration and convexity, and the Fisher-Weil duration, of cash flows at
    ``times`` discounted at the zero rates ``zeros`` (fractions); arrays or lists, one entry per cash flow."""
    times, cf, zeros = (np.asarray(v, float) for v in (times, cf, zeros))
    if not times.shape == cf.shape == zeros.shape or times.ndim != 1:
        raise SpecificationError("give one time, cash flow and zero rate for each payment")
    pv = cf * np.exp(-zeros * times)
    price = float(pv.sum())
    ytm = float(optimize.brentq(lambda y: np.sum(cf * np.exp(-y * times)) - price, -0.5, 2.0, xtol=1e-14))
    w = cf * np.exp(-ytm * times) / price
    return {"price": price, "yield": ytm, "duration": float(w @ times), "convexity": float(w @ times**2),
            "Fisher-Weil duration": float(pv @ times / price)}


def bond(yc: Any, coupon: float, maturity: float, freq: int = 2, period: Any = None, percent: bool = True) -> dict[str, float]:
    """A coupon bond's price per 100 of face, yield, duration and convexity, from one period's fitted curve."""
    if maturity <= 0 or freq < 1:
        raise SpecificationError("the maturity must be positive and the coupons at least once a year")
    times, cf = _cash_flows(float(coupon), float(maturity), int(freq))
    return bond_measures(times, cf, zero_rates(yc, times, period, percent))


def par_coupon(yc: Any, maturity: float, freq: int = 2, period: Any = None, percent: bool = True) -> float:
    """The coupon (percent a year) at which a bond of ``maturity`` prices at par from the curve."""
    times, _ = _cash_flows(0.0, float(maturity), int(freq))
    P = np.exp(-zero_rates(yc, times, period, percent) * times)
    return float(100 * freq * (1 - P[-1]) / P.sum())


# --- the dynamic Nelson-Siegel model ----------------------------------------------------------------------------


def _unpack(theta: np.ndarray, k: int, p: int, full: bool) -> dict[str, Any]:
    i = 0
    lam = float(np.exp(theta[i])); i += 1
    mu = theta[i:i + k]; i += k
    if full:
        A = theta[i:i + k * k].reshape(k, k); i += k * k
    else:
        A = np.diag(np.tanh(theta[i:i + k])); i += k
    L = np.zeros((k, k))
    L[np.tril_indices(k)] = theta[i:i + k * (k + 1) // 2]; i += k * (k + 1) // 2
    L[np.diag_indices(k)] = np.exp(np.diag(L))
    h = np.exp(2 * theta[i:i + p])
    return {"lam": lam, "mu": mu, "A": A, "Q": L @ L.T, "H": np.diag(h)}


def dns_system(parts: dict[str, Any], maturities: np.ndarray, series: Sequence[str]) -> Any:
    """The state-space form: states the factors' deviations from mu, ``y = mu-loadings + Z a + u``, ``a' = A a + e``."""
    from pymetrix.statespace import SystemMatrices

    k = len(parts["mu"])
    Z = loadings(maturities, parts["lam"])[:, :k]
    names = ("level", "slope", "curvature")[:k]
    return SystemMatrices(names, tuple(series), tuple(f"e_{n}" for n in names), parts["A"], np.zeros(k), np.eye(k), parts["Q"], Z,
                          Z @ parts["mu"], parts["H"], (True,) * k)


def dns_loglike(theta: np.ndarray, Y: np.ndarray, maturities: np.ndarray, series: Sequence[str], full: bool) -> float:
    from pymetrix.statespace import kalman_filter

    parts = _unpack(theta, 3, Y.shape[1], full)
    if np.max(np.abs(np.linalg.eigvals(parts["A"]))) >= 0.9999:
        return -1e12
    m = dns_system(parts, maturities, series)
    a0, P0, _ = m.start("stationary")
    try:
        return kalman_filter(Y, m, a0, P0).loglike
    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        return -1e12


@dataclass(frozen=True, eq=False)
class FittedDNS:
    """The dynamic Nelson-Siegel model, estimated in one step by the Kalman filter."""

    series: tuple[str, ...]
    maturities: np.ndarray
    lam: float
    mu: np.ndarray
    A: np.ndarray
    Q: np.ndarray
    H: np.ndarray
    loglike: float
    factors: pd.DataFrame  # smoothed
    filtered: pd.DataFrame = field(repr=False)
    yields: pd.DataFrame = field(repr=False)
    full: bool = False
    converged: bool = True

    stage = "finance"

    def fitted(self) -> pd.DataFrame:
        Z = loadings(self.maturities, self.lam)
        return pd.DataFrame(self.factors.to_numpy() @ Z.T, index=self.factors.index, columns=list(self.series))

    def forecast(self, h: int = 12) -> pd.DataFrame:
        """The yields forecast 1 to ``h`` periods ahead from the last filtered factors and the VAR(1)."""
        f = self.filtered.iloc[-1].to_numpy()
        out = []
        for _ in range(h):
            f = self.mu + self.A @ (f - self.mu)
            out.append(loadings(self.maturities, self.lam) @ f)
        last = self.filtered.index[-1]
        index = pd.period_range(last + 1, periods=h, freq=last.freq) if isinstance(last, pd.Period) else pd.RangeIndex(1, h + 1)
        return pd.DataFrame(out, index=index, columns=list(self.series))

    def summary(self) -> Table:
        names = list(self.factors.columns)
        rows = [["decay", self.lam]] + [[f"mean {n}", float(v)] for n, v in zip(names, self.mu)]
        rows += [[f"A {a} on {b}", float(self.A[i, j])] for i, a in enumerate(names) for j, b in enumerate(names) if self.full or i == j]
        rows += [[f"sd shock {n}", float(np.sqrt(self.Q[i, i]))] for i, n in enumerate(names)]
        rows += [[f"sd error {s}", float(np.sqrt(self.H[i, i]))] for i, s in enumerate(self.series)]
        notes = (f"Log likelihood: {self.loglike:.4f}", f"Periods: {len(self.yields)}", f"Maturities: {len(self.series)}",
                 f"Factors: {'a full' if self.full else 'a diagonal'} VAR(1)")
        if not self.converged:
            notes += ("The optimiser stopped before converging: treat the estimates with care.",)
        return Table(["value"], [[float(v)] for _, v in rows], index=[r[0] for r in rows], index_name="parameter",
                     title="Dynamic Nelson-Siegel", notes=notes)

    def plot(self) -> Figure:
        from pymetrix.figures import Figure, Line, Panel

        return Figure((Panel(tuple(Line(self.factors[c], c) for c in self.factors.columns), title="Smoothed factors"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return self.factors

    def __repr__(self) -> str:
        return f"<FittedDNS of {len(self.series)} maturities, decay {self.lam:.4f}>"


with plugin_context(PLUGIN):
    @api("finance:dynamic_nelson_siegel")
    def dynamic_nelson_siegel(data: Dataset, series: Sequence[str], maturities: Sequence[float], var: Literal["diagonal", "full"] = "diagonal",
                              maxiter: int = 2000) -> FittedDNS:
        """The dynamic Nelson-Siegel model of the yields ``series`` at ``maturities`` (years): the decay, the factors'
        VAR(1) and the measurement errors by maximum likelihood with the Kalman filter, starting from the two-step fit."""
        from pymetrix.statespace import kalman_filter, kalman_smoother
        from .yields import yield_curve

        if var not in ("diagonal", "full"):
            raise SpecificationError("var is 'diagonal' or 'full'")
        full = var == "full"
        two = yield_curve(data, series, maturities)
        series = list(two.series)
        m = two.maturities
        Y = two.yields.to_numpy(float)
        F = two.factors.dropna().to_numpy()
        mu0 = F.mean(axis=0)
        dev = F - mu0
        if full:
            A0 = np.linalg.lstsq(dev[:-1], dev[1:], rcond=None)[0].T
        else:
            A0 = np.diag([np.clip(np.sum(dev[1:, j] * dev[:-1, j]) / np.sum(dev[:-1, j] ** 2), -0.99, 0.99) for j in range(3)])
        e = dev[1:] - dev[:-1] @ A0.T
        L0 = np.linalg.cholesky(np.cov(e.T) + 1e-10 * np.eye(3))
        Lp = L0.copy()
        Lp[np.diag_indices(3)] = np.log(np.diag(L0))
        sd = np.nanstd(two.residuals.to_numpy(float), axis=0)
        floor = 0.05 * float(np.median(sd))  # an error sd can go no lower: at zero the likelihood is unbounded
        h0 = np.log(np.maximum(sd, 2 * floor))
        theta0 = np.concatenate([[np.log(two.lam)], mu0, A0.ravel() if full else np.arctanh(np.diag(A0)), Lp[np.tril_indices(3)], h0])
        bounds = [(None, None)] * (len(theta0) - len(series)) + [(np.log(floor), None)] * len(series)
        found = optimize.minimize(lambda th: -dns_loglike(th, Y, m, series, full), theta0, method="L-BFGS-B", bounds=bounds,
                                  options={"maxiter": int(maxiter), "maxfun": int(maxiter) * 50})
        parts = _unpack(found.x, 3, len(series), full)
        system = dns_system(parts, m, series)
        a0, P0, _ = system.start("stationary")
        kf = kalman_filter(Y, system, a0, P0)
        smooth, _ = kalman_smoother(kf, system)
        names = ["level", "slope", "curvature"]
        idx = two.yields.index
        return FittedDNS(tuple(series), m, parts["lam"], parts["mu"], parts["A"], parts["Q"], parts["H"], float(kf.loglike),
                         pd.DataFrame(smooth + parts["mu"], index=idx, columns=names),
                         pd.DataFrame(kf.a_filt + parts["mu"], index=idx, columns=names), two.yields, full, bool(found.success))


with plugin_context(PLUGIN):  # a recorded call: the Bond measures dialog (app.py) builds it
    bond_measures = api("finance:bond_measures")(bond_measures)
