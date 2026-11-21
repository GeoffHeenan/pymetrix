"""Long memory and the frequency domain: the periodogram, smoothed spectra, cross-spectra and coherence; the GPH and
local Whittle estimators of the fractional difference d; and ARFIMA models.

    periodogram(data, "y"); spectrum(data, "y", bandwidth=5).plot()
    cross_spectrum(data, "y", "x", bandwidth=5)        # coherence, phase and gain by frequency
    gph(data, "y"); local_whittle(data, "y")           # d: 0 short memory, (0, 0.5) stationary long memory
    arfima(data, "y", p=1, q=0)                        # (1 - L)^d phi(L)(y - mu) = theta(L) e

The periodogram at the Fourier frequencies lambda_j = 2 pi j / T is I(lambda_j) = |sum_t (y_t - ybar) e^{-i t lambda_j}|^2 /
(2 pi T), so that it estimates the spectral density f with var(y) = the integral of f over (-pi, pi). Smoothed
spectra average 2m + 1 neighbouring ordinates (a Daniell window, ``bandwidth`` = m), the cross-spectrum likewise; the
squared coherence is |f_xy|^2 / (f_xx f_yy), the phase arg f_xy and the gain |f_xy| / f_xx.

GPH (Geweke and Porter-Hudak 1983): least squares of log I(lambda_j) on -2 log(2 sin(lambda_j / 2)) for j = 1..m, m = T^power;
the slope is d, with the asymptotic standard error pi / sqrt(24 m). Local Whittle (Robinson 1995): d minimises
log(mean_j lambda_j^{2d} I_j) - 2d mean_j log lambda_j, standard error 1 / (2 sqrt(m)). ARFIMA(p, d, q) by conditional
sum of squares: the series demeaned, fractionally differenced by the expansion of (1 - L)^d up to each period, then
the ARMA residuals; d, phi and theta minimise their sum of squares, standard errors from the Gauss-Newton information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import optimize

from pymetrix._lazy import signal
from pymetrix._lazy import stats as sps

from ._api import api
from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["CrossSpectrum", "Spectrum", "arfima", "cross_spectrum", "fractional_difference", "gph", "local_whittle", "periodogram", "spectrum"]


def _series(data: Dataset, name: str, span: Any = None) -> pd.Series:
    if data.kind != "time_series":
        raise SpecificationError("spectral analysis needs a single time series")
    if name not in data:
        raise SpecificationError(f"{name!r} is not a series of the data")
    s = data[name].astype(float)
    if span is not None:
        s = s[data.sample_mask(span)]
    s = s.dropna()
    if len(s) < 16:
        raise SpecificationError("spectral analysis needs at least 16 observations")
    return s


def _dft(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(Fourier frequencies lambda_j, j = 1..floor(T/2), the demeaned series' transform at them)."""
    T = len(x)
    z = np.fft.fft(x - x.mean())
    j = np.arange(1, T // 2 + 1)
    return 2 * np.pi * j / T, z[j]


def _daniell(v: np.ndarray, m: int) -> np.ndarray:
    """The average of 2m + 1 neighbouring values, reflected at the ends."""
    if m <= 0:
        return v
    padded = np.r_[v[m - 1::-1] if m <= len(v) else v[::-1], v, v[:-m - 1:-1]]
    return np.convolve(padded, np.ones(2 * m + 1) / (2 * m + 1), mode="valid")


@dataclass(frozen=True, eq=False)
class Spectrum:
    """A periodogram or smoothed spectral density by frequency."""

    variable: str
    frequency: np.ndarray
    density: np.ndarray
    bandwidth: int
    nobs: int

    stage = "estimate"

    @property
    def period(self) -> np.ndarray:
        return 2 * np.pi / self.frequency

    def summary(self) -> Table:
        top = np.argsort(self.density)[::-1][:5]
        rows = [[float(self.frequency[i]), float(self.period[i]), float(self.density[i])] for i in top]
        notes = (f"{'Periodogram' if not self.bandwidth else f'Daniell-smoothed spectrum (m = {self.bandwidth})'}; {self.nobs} observations; "
                 "the largest ordinates. Frequencies in radians per period; periods in observations.",)
        return Table(["frequency", "period", "density"], rows, title=f"Spectrum: {self.variable}", notes=notes)

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return pd.DataFrame({"period": self.period, "density": self.density}, index=pd.Index(self.frequency, name="frequency"))

    def plot(self) -> Any:
        from .figures import Figure, Line, Panel

        idx = pd.Index(self.frequency, name="frequency")
        return Figure((Panel((Line(pd.Series(np.log(self.density), index=idx), "log density"),), title=f"Spectrum of {self.variable}",
                             xlabel="frequency (radians)"),))

    def __repr__(self) -> str:
        return f"<Spectrum {self.variable}: {len(self.frequency)} frequencies, bandwidth {self.bandwidth}>"


@api("pymetrix:periodogram")
def periodogram(data: Dataset, variable: str, span: Any = None) -> Spectrum:
    """The periodogram of ``variable`` at the Fourier frequencies."""
    return spectrum(data, variable, 0, span)


@api("pymetrix:spectrum")
def spectrum(data: Dataset, variable: str, bandwidth: int = 5, span: Any = None) -> Spectrum:
    """The spectral density of ``variable``: the periodogram smoothed by a Daniell window of 2 ``bandwidth`` + 1 ordinates."""
    x = _series(data, variable, span).to_numpy()
    lam, z = _dft(x)
    I = np.abs(z) ** 2 / (2 * np.pi * len(x))
    return Spectrum(variable, lam, _daniell(I, int(bandwidth)), int(bandwidth), len(x))


@dataclass(frozen=True, eq=False)
class CrossSpectrum:
    """The cross-spectrum of two series: coherence, phase and gain by frequency."""

    x: str
    y: str
    frequency: np.ndarray
    fxx: np.ndarray
    fyy: np.ndarray
    fxy: np.ndarray
    bandwidth: int

    stage = "estimate"

    @property
    def coherence(self) -> np.ndarray:
        return np.abs(self.fxy) ** 2 / (self.fxx * self.fyy)

    @property
    def phase(self) -> np.ndarray:
        return np.angle(self.fxy)

    @property
    def gain(self) -> np.ndarray:
        return np.abs(self.fxy) / self.fxx

    def summary(self) -> Table:
        bands = [("long run (periods > 32)", self.frequency < 2 * np.pi / 32), ("business cycle (6 to 32)", (self.frequency >= 2 * np.pi / 32) & (self.frequency <= 2 * np.pi / 6)),
                 ("short run (< 6)", self.frequency > 2 * np.pi / 6)]
        rows = [[name, float(self.coherence[m].mean()) if m.any() else np.nan, float(self.gain[m].mean()) if m.any() else np.nan] for name, m in bands]
        notes = (f"Daniell-smoothed cross-spectrum (m = {self.bandwidth}); squared coherence and gain of {self.y} on {self.x}, averaged by band "
                 "(periods in observations: for quarterly data the business cycle is 6 to 32).",)
        return Table(["band", "squared coherence", "gain"], rows, title=f"Cross-spectrum: {self.y} and {self.x}", notes=notes)

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return pd.DataFrame({"coherence": self.coherence, "phase": self.phase, "gain": self.gain}, index=pd.Index(self.frequency, name="frequency"))

    def plot(self) -> Any:
        from .figures import Figure, Line, Panel

        idx = pd.Index(self.frequency, name="frequency")
        return Figure((Panel((Line(pd.Series(self.coherence, index=idx), "squared coherence"),), title="Squared coherence", xlabel="frequency"),
                       Panel((Line(pd.Series(self.phase, index=idx), "phase"),), title="Phase", xlabel="frequency")))

    def __repr__(self) -> str:
        return f"<CrossSpectrum {self.y} and {self.x}>"


@api("pymetrix:cross_spectrum")
def cross_spectrum(data: Dataset, y: str, x: str, bandwidth: int = 5, span: Any = None) -> CrossSpectrum:
    """The cross-spectrum of ``y`` and ``x`` on their common sample, Daniell-smoothed."""
    if bandwidth < 1:
        raise SpecificationError("the cross-spectrum needs smoothing: bandwidth >= 1 (with none, the coherence is always 1)")
    a, b = _series(data, y, span), _series(data, x, span)
    common = a.index.intersection(b.index)
    ya, xb = a.loc[common].to_numpy(), b.loc[common].to_numpy()
    lam, zy = _dft(ya)
    _, zx = _dft(xb)
    T = len(common)
    fxx = _daniell(np.abs(zx) ** 2, bandwidth) / (2 * np.pi * T)
    fyy = _daniell(np.abs(zy) ** 2, bandwidth) / (2 * np.pi * T)
    cross = zx * np.conj(zy)
    fxy = (_daniell(cross.real, bandwidth) + 1j * _daniell(cross.imag, bandwidth)) / (2 * np.pi * T)
    return CrossSpectrum(x, y, lam, fxx, fyy, fxy, int(bandwidth))


# long memory


def _m(T: int, power: float) -> int:
    m = int(np.floor(T**power))
    if m < 4:
        raise SpecificationError("too few frequencies: raise power or use more observations")
    return m


@api("pymetrix:gph")
def gph(data: Dataset, variable: str, power: float = 0.5, span: Any = None) -> Any:
    """Geweke and Porter-Hudak's log-periodogram estimate of d, from the first T^power Fourier frequencies."""
    from .stats.base import TestResult

    x = _series(data, variable, span).to_numpy()
    T = len(x)
    m = _m(T, power)
    lam, z = _dft(x)
    I = np.abs(z[:m]) ** 2 / (2 * np.pi * T)
    reg = -2 * np.log(2 * np.sin(lam[:m] / 2))
    X = np.column_stack([np.ones(m), reg])
    b = np.linalg.lstsq(X, np.log(I), rcond=None)[0]
    se = np.pi / np.sqrt(24 * m)
    d = float(b[1])
    return TestResult(f"GPH estimate of d ({variable})", "d = 0 (short memory)", d, None, float(2 * sps.norm.sf(abs(d / se))), "normal",
                      details={"d": d, "std err": float(se), "frequencies": m, "observations": T})


@api("pymetrix:local_whittle")
def local_whittle(data: Dataset, variable: str, power: float = 0.65, span: Any = None) -> Any:
    """Robinson's local Whittle estimate of d, from the first T^power Fourier frequencies."""
    from .stats.base import TestResult

    x = _series(data, variable, span).to_numpy()
    T = len(x)
    m = _m(T, power)
    lam, z = _dft(x)
    I = np.abs(z[:m]) ** 2 / (2 * np.pi * T)
    ll = np.log(lam[:m])

    def R(d: float) -> float:
        return float(np.log(np.mean(lam[:m] ** (2 * d) * I)) - 2 * d * ll.mean())

    d = float(optimize.minimize_scalar(R, bounds=(-0.99, 1.99), method="bounded", options={"xatol": 1e-10}).x)
    se = 1 / (2 * np.sqrt(m))
    return TestResult(f"Local Whittle estimate of d ({variable})", "d = 0 (short memory)", d, None, float(2 * sps.norm.sf(abs(d / se))),
                      "normal", details={"d": d, "std err": float(se), "frequencies": m, "observations": T})


def fractional_difference(x: np.ndarray, d: float) -> np.ndarray:
    """(1 - L)^d x by the binomial expansion, truncated at the start of the sample: pi_0 = 1, pi_k = pi_{k-1} (k - 1 - d) / k."""
    T = len(x)
    k = np.arange(1, T)
    pi = np.r_[1.0, np.cumprod((k - 1 - d) / k)]
    return np.convolve(x, pi)[:T]


def _arfima_resid(theta: np.ndarray, x: np.ndarray, p: int, q: int) -> np.ndarray:
    d, ar, ma = theta[0], theta[1:1 + p], theta[1 + p:1 + p + q]
    w = fractional_difference(x, d)
    return signal.lfilter(np.r_[1.0, -ar], np.r_[1.0, ma], w)  # theta(L)^-1 phi(L) w


@dataclass(frozen=True, eq=False)
class FittedARFIMA:
    variable: str
    p: int
    q: int
    params: np.ndarray
    cov: np.ndarray = field(repr=False)
    mean: float = 0.0
    sigma2: float = 0.0
    nobs: int = 0
    resid: pd.Series = field(default=None, repr=False)

    stage = "estimate"

    @property
    def names(self) -> tuple[str, ...]:
        return ("d", *[f"ar{i + 1}" for i in range(self.p)], *[f"ma{i + 1}" for i in range(self.q)])

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def summary(self) -> Table:
        z = self.params / self.se
        rows = [[float(b), float(s), float(t), float(2 * sps.norm.sf(abs(t)))] for b, s, t in zip(self.params, self.se, z)]
        notes = (f"ARFIMA({self.p}, d, {self.q}) by conditional sum of squares; mean {self.mean:.6g}; residual variance {self.sigma2:.6g}; "
                 f"{self.nobs} observations. d in (0, 0.5): stationary long memory; (0.5, 1): nonstationary but mean-reverting.",)
        return Table(["estimate", "std err", "z", "p-value"], rows, index=list(self.names), index_name="parameter",
                     title=f"ARFIMA: {self.variable}", notes=notes)

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return pd.DataFrame({"estimate": self.params, "se": self.se}, index=list(self.names))

    def __repr__(self) -> str:
        return f"<FittedARFIMA({self.p}, d, {self.q}) {self.variable}: d = {self.params[0]:.4f}>"


@api("pymetrix:arfima")
def arfima(data: Dataset, variable: str, p: int = 0, q: int = 0, span: Any = None) -> FittedARFIMA:
    """ARFIMA(p, d, q) of ``variable`` by conditional sum of squares, the mean removed first."""
    s = _series(data, variable, span)
    y = s.to_numpy()
    mu = float(y.mean())
    x = y - mu
    start_d = float(local_whittle(data, variable, span=span).statistic)
    theta0 = np.r_[np.clip(start_d, -0.45, 0.95), np.zeros(p + q)]
    found = optimize.least_squares(lambda th: _arfima_resid(th, x, p, q), theta0, method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14)
    theta = found.x
    e = _arfima_resid(theta, x, p, q)
    n = len(e)
    s2 = float(e @ e / n)
    cov = s2 * np.linalg.pinv(found.jac.T @ found.jac)
    return FittedARFIMA(variable, p, q, theta, cov, mu, s2, n, pd.Series(e, index=s.index, name="resid"))
