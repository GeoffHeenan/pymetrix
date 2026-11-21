"""Nonparametric estimation: kernel densities, kernel regression (Nadaraya-Watson and local linear, with any number of
regressors), Robinson's partially linear model, nonparametric autoregression, and a GARCH whose shock density is
estimated by kernel (Engle and Gonzalez-Rivera 1991).

    kde(data, "r")                                     # a Gaussian kernel density; bandwidth by Silverman's rule
    kde(data, "r", bandwidth="cv")                     # least-squares cross-validation
    kernel_regression(data, "y", ["x1", "x2"])         # local linear, bandwidths by leave-one-out cross-validation
    kernel_regression(data, "y", ["x"], kind="lc")     # Nadaraya-Watson (local constant)
    partially_linear(data, "y", linear=["x"], nonparametric=["z"])   # y = x'b + g(z) + e (Robinson 1988)
    nonparametric_autoregression(data, "y", lags=2)    # E[y_t | y_{t-1}, y_{t-2}] by kernel regression
    semiparametric_garch(data, "r")                    # GARCH(1, 1) with the shocks' density by kernel

Kernels are Gaussian (densities may also use the Epanechnikov kernel); multivariate regressions use the product
kernel with one bandwidth for each regressor. Cross-validation minimises the leave-one-out squared error over the log
bandwidths, from the normal reference rule. The kernel-regression R^2 is the squared correlation of fitted and
actual values.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import optimize

from pymetrix._lazy import stats as sps

from ._api import api
from ._random import generator as make_rng
from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["KernelDensity", "KernelRegression", "PartiallyLinear", "SemiparametricGARCH", "kde", "kernel_regression",
           "nonparametric_autoregression", "partially_linear", "semiparametric_garch"]

_SQRT2PI = np.sqrt(2 * np.pi)


def _column(data: Dataset, name: str) -> pd.Series:
    if name not in data:
        raise SpecificationError(f"{name!r} is not a series of the data")
    return data[name].astype(float)


# densities


def bandwidth_rule(x: np.ndarray, rule: str) -> float:
    """Silverman's (0.9) or Scott's (1.059) rule: c min(sd, IQR / 1.349) n^(-1/5)."""
    n = len(x)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    a = min(np.std(x, ddof=1), iqr / 1.349) if iqr > 0 else np.std(x, ddof=1)
    c = {"silverman": 0.9, "scott": 1.059}[rule]
    return float(c * a * n ** -0.2)


def _lscv(x: np.ndarray, h: float) -> float:
    """Least-squares cross-validation for the Gaussian kernel: int f^2 - 2 mean f_{-i}(x_i), in closed form."""
    n = len(x)
    conv_sum = k_sum = 0.0
    for i in range(0, n, 1000):  # in blocks of rows: memory near 1000 x n
        d = (x[i:i + 1000, None] - x[None, :]) / h
        conv_sum += float(np.sum(np.exp(-d**2 / 4))) / (2 * np.sqrt(np.pi))  # the kernel convolved with itself: N(0, 2)
        k_sum += float(np.sum(np.exp(-d**2 / 2))) / _SQRT2PI
    return float(conv_sum / (n**2 * h) - 2 * (k_sum - n / _SQRT2PI) / (n * (n - 1) * h))


def _density_at(points: np.ndarray, x: np.ndarray, h: float, kernel: str) -> np.ndarray:
    u = (points[:, None] - x[None, :]) / h
    if kernel == "gaussian":
        K = np.exp(-u**2 / 2) / _SQRT2PI
    else:  # Epanechnikov
        K = np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
    return K.mean(axis=1) / h


@dataclass(frozen=True, eq=False)
class KernelDensity:
    """A kernel density estimate on a grid."""

    variable: str
    grid: np.ndarray
    density: np.ndarray
    bandwidth: float
    kernel: str
    rule: str
    x: np.ndarray = field(repr=False)

    stage = "estimate"

    def __call__(self, points: Any) -> np.ndarray:
        """The density at ``points``."""
        return _density_at(np.atleast_1d(np.asarray(points, float)), self.x, self.bandwidth, self.kernel)

    def summary(self) -> Table:
        rows = [["observations", len(self.x)], ["bandwidth", self.bandwidth], ["bandwidth rule", self.rule], ["kernel", self.kernel],
                ["mean", float(np.mean(self.x))], ["standard deviation", float(np.std(self.x, ddof=1))],
                ["mode (on the grid)", float(self.grid[np.argmax(self.density)])]]
        return Table(["", "value"], rows, title=f"Kernel density: {self.variable}")

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return pd.DataFrame({"density": self.density}, index=pd.Index(self.grid, name=self.variable))

    def plot(self, normal: bool = True) -> Any:
        """The density, with the normal of the same mean and variance for comparison."""
        from .figures import Figure, Line, Panel

        idx = pd.Index(self.grid, name=self.variable)
        layers = [Line(pd.Series(self.density, index=idx), "kernel density")]
        if normal:
            layers.append(Line(pd.Series(sps.norm.pdf(self.grid, np.mean(self.x), np.std(self.x, ddof=1)), index=idx), "normal", style="--"))
        return Figure((Panel(tuple(layers), title=f"Density of {self.variable}", xlabel=self.variable),))

    def __repr__(self) -> str:
        return f"<KernelDensity {self.variable}: bandwidth {self.bandwidth:.4g} ({self.rule})>"


@api("pymetrix:kde")
def kde(data: Dataset, variable: str, bandwidth: float | Literal["silverman", "scott", "cv"] = "silverman",
        kernel: Literal["gaussian", "epanechnikov"] = "gaussian", points: int = 256, cut: float = 3.0, span: Any = None) -> KernelDensity:
    """A kernel density estimate of ``variable`` on ``points`` points reaching ``cut`` bandwidths beyond the data."""
    s = _column(data, variable)
    if span is not None:
        s = s[data.sample_mask(span)]
    x = s.dropna().to_numpy(float)
    if len(x) < 5:
        raise SpecificationError("a kernel density needs at least 5 observations")
    if kernel not in ("gaussian", "epanechnikov"):
        raise SpecificationError("kernel is 'gaussian' or 'epanechnikov'")
    if isinstance(bandwidth, str):
        if bandwidth == "cv":
            if kernel != "gaussian":
                raise SpecificationError("cross-validated bandwidths here use the Gaussian kernel")
            h0 = bandwidth_rule(x, "silverman")
            found = optimize.minimize_scalar(lambda lh: _lscv(x, np.exp(lh)), bounds=(np.log(h0) - 3, np.log(h0) + 3), method="bounded",
                                             options={"xatol": 1e-8})
            h, rule = float(np.exp(found.x)), "least-squares cross-validation"
        elif bandwidth in ("silverman", "scott"):
            h, rule = bandwidth_rule(x, bandwidth), f"{bandwidth.capitalize()}'s rule"
        else:
            raise SpecificationError("bandwidth is a number, 'silverman', 'scott' or 'cv'")
    else:
        h, rule = float(bandwidth), "given"
        if h <= 0:
            raise SpecificationError("the bandwidth must be positive")
    grid = np.linspace(x.min() - cut * h, x.max() + cut * h, int(points))
    return KernelDensity(variable, grid, _density_at(grid, x, h, kernel), h, kernel, rule, x)


# kernel regression


def _weights(X: np.ndarray, at: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Product Gaussian kernel weights, (points, observations), summed over the regressors one at a time."""
    z = np.zeros((len(at), len(X)))
    for j in range(X.shape[1]):
        z += ((at[:, None, j] - X[None, :, j]) / h[j]) ** 2
    return np.exp(-0.5 * z) / np.prod(h * _SQRT2PI)


def _fit_at(X: np.ndarray, y: np.ndarray, at: np.ndarray, h: np.ndarray, kind: str, leave_out: bool = False,
            block: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """The conditional mean (and, local linear, its gradient) at ``at``, in blocks of ``block`` points."""
    if len(at) > block:
        parts = [_fit_at(X, y, at[i:i + block], h, kind, False) for i in range(0, len(at), block)] if not leave_out else [
            _fit_block_leave_out(X, y, h, kind, i, min(i + block, len(at))) for i in range(0, len(at), block)]
        return np.concatenate([p[0] for p in parts]), np.vstack([p[1] for p in parts])
    W = _weights(X, at, h)
    if leave_out:
        np.fill_diagonal(W, 0.0)
    q = X.shape[1]
    if kind == "lc":
        s = W.sum(axis=1)
        return W @ y / np.where(s > 0, s, np.nan), np.full((len(at), q), np.nan)
    mean, grad = np.empty(len(at)), np.empty((len(at), q))
    for i in range(len(at)):
        D = np.column_stack([np.ones(len(X)), X - at[i]])
        A = D.T @ (D * W[i][:, None])
        b = np.linalg.lstsq(A, D.T @ (W[i] * y), rcond=None)[0]
        mean[i], grad[i] = b[0], b[1:]
    return mean, grad


def _fit_block_leave_out(X: np.ndarray, y: np.ndarray, h: np.ndarray, kind: str, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out fits at the data points start..stop - 1 (each point's own weight zeroed)."""
    W = _weights(X, X[start:stop], h)
    W[np.arange(stop - start), np.arange(start, stop)] = 0.0
    q = X.shape[1]
    if kind == "lc":
        s = W.sum(axis=1)
        return W @ y / np.where(s > 0, s, np.nan), np.full((stop - start, q), np.nan)
    mean, grad = np.empty(stop - start), np.empty((stop - start, q))
    for i in range(stop - start):
        D = np.column_stack([np.ones(len(X)), X - X[start + i]])
        b = np.linalg.lstsq(D.T @ (D * W[i][:, None]), D.T @ (W[i] * y), rcond=None)[0]
        mean[i], grad[i] = b[0], b[1:]
    return mean, grad


def _cv_bandwidths(X: np.ndarray, y: np.ndarray, kind: str) -> np.ndarray:
    n, q = X.shape
    h0 = 1.06 * X.std(axis=0, ddof=1) * n ** (-1 / (4 + q))

    def loss(lh: np.ndarray) -> float:
        m, _ = _fit_at(X, y, X, np.exp(lh), kind, leave_out=True)
        r = y - m
        return float(np.mean(r**2)) if np.all(np.isfinite(r)) else 1e300

    found = optimize.minimize(loss, np.log(h0), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 2000})
    return np.exp(found.x)


def _r2(y: np.ndarray, m: np.ndarray) -> float:
    yc, mc = y - m.mean(), m - m.mean()  # centred on the fitted values' mean, as statsmodels' KernelReg
    return float((yc @ mc) ** 2 / ((yc @ yc) * (mc @ mc)))


@dataclass(frozen=True, eq=False)
class KernelRegression:
    """A kernel regression: fitted conditional means (and local-linear gradients) at the data."""

    y_name: str
    x_names: tuple[str, ...]
    kind: str
    bandwidth: np.ndarray
    fitted: pd.Series
    gradient: pd.DataFrame
    resid: pd.Series
    r2: float
    bandwidth_rule: str
    X: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)

    stage = "estimate"

    def predict(self, points: Any) -> pd.DataFrame:
        """The conditional mean (and gradient) at ``points``: an array or frame with a column for each regressor."""
        P = np.atleast_2d(np.asarray(points, float))
        if P.shape[1] != len(self.x_names) and P.shape[0] == len(self.x_names):
            P = P.T
        if P.shape[1] != len(self.x_names):
            P = P.reshape(-1, len(self.x_names))
        m, g = _fit_at(self.X, self.y, P, self.bandwidth, self.kind)
        out = pd.DataFrame(P, columns=list(self.x_names))
        out["mean"] = m
        if self.kind == "ll":
            for j, name in enumerate(self.x_names):
                out[f"d/d{name}"] = g[:, j]
        return out

    def summary(self) -> Table:
        label = {"lc": "Nadaraya-Watson (local constant)", "ll": "local linear"}[self.kind]
        rows = [[name, float(h), float(np.nanmean(self.gradient[name])) if self.kind == "ll" else np.nan]
                for name, h in zip(self.x_names, self.bandwidth)]
        notes = (f"{label} kernel regression of {self.y_name}; Gaussian product kernel; bandwidths by {self.bandwidth_rule}; "
                 f"{len(self.y)} observations; R^2 = {self.r2:.4f}.",)
        return Table(["regressor", "bandwidth", "mean gradient"], rows, title=f"Kernel regression: {self.y_name}", notes=notes)

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        out = pd.DataFrame({self.y_name: self.y, "fitted": self.fitted.to_numpy(), "resid": self.resid.to_numpy()}, index=self.fitted.index)
        return pd.concat([out, self.gradient.add_prefix("d/d")], axis=1) if self.kind == "ll" else out

    def plot(self, regressor: str | None = None, points: int = 100) -> Any:
        """The data and the fitted curve against one regressor (the others held at their medians)."""
        from .figures import Figure, Line, Panel, Scatter

        name = regressor or self.x_names[0]
        if name not in self.x_names:
            raise SpecificationError(f"{name!r} is not a regressor: {', '.join(self.x_names)}")
        j = self.x_names.index(name)
        grid = np.linspace(self.X[:, j].min(), self.X[:, j].max(), points)
        P = np.tile(np.median(self.X, axis=0), (points, 1))
        P[:, j] = grid
        m, _ = _fit_at(self.X, self.y, P, self.bandwidth, self.kind)
        idx = pd.Index(grid, name=name)
        return Figure((Panel((Scatter(pd.Series(self.X[:, j]), pd.Series(self.y), "data"), Line(pd.Series(m, index=idx), "kernel regression")),
                             title=f"{self.y_name} on {name}", xlabel=name, ylabel=self.y_name),))

    def __repr__(self) -> str:
        return f"<KernelRegression {self.y_name} on {', '.join(self.x_names)} ({self.kind}): R^2 {self.r2:.3f}>"


def _regression(frame: pd.DataFrame, y_name: str, x_names: Sequence[str], kind: str, bandwidth: Any) -> KernelRegression:
    frame = frame.dropna()
    if len(frame) < 10:
        raise SpecificationError("kernel regression needs at least 10 complete observations")
    if kind not in ("lc", "ll"):
        raise SpecificationError("kind is 'lc' (Nadaraya-Watson) or 'll' (local linear)")
    y = frame[y_name].to_numpy(float)
    X = frame[list(x_names)].to_numpy(float)
    q = X.shape[1]
    if isinstance(bandwidth, str):
        if bandwidth == "cv":
            h, rule = _cv_bandwidths(X, y, kind), "leave-one-out cross-validation"
        elif bandwidth == "normal":
            h, rule = 1.06 * X.std(axis=0, ddof=1) * len(y) ** (-1 / (4 + q)), "the normal reference rule"
        else:
            raise SpecificationError("bandwidth is 'cv', 'normal', or a number for each regressor")
    else:
        h, rule = np.broadcast_to(np.asarray(bandwidth, float), (q,)).copy(), "given"
        if np.any(h <= 0):
            raise SpecificationError("bandwidths must be positive")
    m, g = _fit_at(X, y, X, h, kind)
    index = frame.index
    return KernelRegression(y_name, tuple(x_names), kind, np.asarray(h, float), pd.Series(m, index=index, name="fitted"),
                            pd.DataFrame(g, index=index, columns=list(x_names)), pd.Series(y - m, index=index, name="resid"),
                            _r2(y, m), rule, X, y)


@api("pymetrix:kernel_regression")
def kernel_regression(data: Dataset, y: str, x: Sequence[str], kind: Literal["ll", "lc"] = "ll",
                      bandwidth: Any = "cv", span: Any = None) -> KernelRegression:
    """Kernel regression of ``y`` on the ``x`` series: local linear (``kind="ll"``) or Nadaraya-Watson (``"lc"``), with
    bandwidths by cross-validation, the normal reference rule, or given."""
    x = [x] if isinstance(x, str) else list(x)
    frame = pd.DataFrame({name: _column(data, name) for name in [y, *x]})
    if span is not None:
        frame = frame[data.sample_mask(span)]
    return _regression(frame, y, x, kind, bandwidth)


@api("pymetrix:nonparametric_autoregression")
def nonparametric_autoregression(data: Dataset, variable: str, lags: int = 1, kind: Literal["ll", "lc"] = "ll",
                                 bandwidth: Any = "cv", span: Any = None) -> KernelRegression:
    """E[y_t | y_{t-1}, ..., y_{t-lags}] by kernel regression: a nonlinear autoregression with no functional form."""
    if data.kind != "time_series":
        raise SpecificationError("a nonparametric autoregression needs a single time series")
    if lags < 1:
        raise SpecificationError("lags must be at least 1")
    s = _column(data, variable)
    frame = pd.DataFrame({variable: s, **{f"{variable}[-{j}]": s.shift(j) for j in range(1, lags + 1)}})
    if span is not None:
        frame = frame[data.sample_mask(span)]
    return _regression(frame, variable, [f"{variable}[-{j}]" for j in range(1, lags + 1)], kind, bandwidth)


# partially linear


@dataclass(frozen=True, eq=False)
class PartiallyLinear:
    """Robinson's partially linear model y = x'b + g(z) + e."""

    y_name: str
    linear: tuple[str, ...]
    nonparametric: tuple[str, ...]
    params: np.ndarray
    cov: np.ndarray = field(repr=False)
    bandwidth: np.ndarray = field(repr=False)
    g: pd.Series = field(repr=False)
    resid: pd.Series = field(repr=False)
    Z: np.ndarray = field(repr=False)

    stage = "estimate"

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def summary(self) -> Table:
        z = self.params / self.se
        rows = [[float(b), float(s), float(t), float(2 * sps.norm.sf(abs(t)))] for b, s, t in zip(self.params, self.se, z)]
        notes = (f"Robinson (1988): y and x net of kernel regressions on {', '.join(self.nonparametric)} (bandwidths "
                 + ", ".join(f"{h:.4g}" for h in self.bandwidth) + f"); heteroskedasticity-robust standard errors; {len(self.resid)} observations.",)
        return Table(["estimate", "std err", "z", "p-value"], rows, index=list(self.linear), index_name="parameter",
                     title=f"Partially linear model: {self.y_name}", notes=notes)

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return pd.DataFrame({"estimate": self.params, "se": self.se}, index=list(self.linear))

    def plot(self) -> Any:
        """The estimated g against the first nonparametric variable."""
        from .figures import Figure, Line, Panel

        order = np.argsort(self.Z[:, 0])
        idx = pd.Index(self.Z[order, 0], name=self.nonparametric[0])
        return Figure((Panel((Line(pd.Series(self.g.to_numpy()[order], index=idx), "g"),), title=f"The nonparametric part, g({self.nonparametric[0]})",
                             xlabel=self.nonparametric[0]),))

    def __repr__(self) -> str:
        return f"<PartiallyLinear {self.y_name}: {dict(zip(self.linear, np.round(self.params, 4)))}>"


@api("pymetrix:partially_linear")
def partially_linear(data: Dataset, y: str, linear: Sequence[str], nonparametric: Sequence[str], bandwidth: Any = "cv",
                     span: Any = None) -> PartiallyLinear:
    """Robinson's (1988) estimator of y = x'b + g(z) + e: b by least squares of y - E[y|z] on x - E[x|z], the conditional
    means by Nadaraya-Watson (bandwidths by cross-validation on y, or given); g by kernel regression of y - x'b on z."""
    linear = [linear] if isinstance(linear, str) else list(linear)
    nonparametric = [nonparametric] if isinstance(nonparametric, str) else list(nonparametric)
    frame = pd.DataFrame({name: _column(data, name) for name in [y, *linear, *nonparametric]})
    if span is not None:
        frame = frame[data.sample_mask(span)]
    frame = frame.dropna()
    Y = frame[y].to_numpy(float)
    X = frame[linear].to_numpy(float)
    Z = frame[nonparametric].to_numpy(float)
    q = Z.shape[1]
    if isinstance(bandwidth, str):
        if bandwidth != "cv":
            raise SpecificationError("bandwidth is 'cv' or a number for each nonparametric variable")
        h = _cv_bandwidths(Z, Y, "lc")
    else:
        h = np.broadcast_to(np.asarray(bandwidth, float), (q,)).copy()
    my, _ = _fit_at(Z, Y, Z, h, "lc")
    mX = np.column_stack([_fit_at(Z, X[:, j], Z, h, "lc")[0] for j in range(X.shape[1])])
    ey, eX = Y - my, X - mX
    b = np.linalg.lstsq(eX, ey, rcond=None)[0]
    e = ey - eX @ b
    n, k = eX.shape
    bread = np.linalg.inv(eX.T @ eX)
    cov = bread @ (eX * (e**2)[:, None]).T @ eX @ bread * n / (n - k)
    g, _ = _fit_at(Z, Y - X @ b, Z, h, "lc")
    return PartiallyLinear(y, tuple(linear), tuple(nonparametric), b, cov, h, pd.Series(g, index=frame.index, name="g"),
                           pd.Series(Y - X @ b - g, index=frame.index, name="resid"), Z)


# semiparametric GARCH


def _garch_variance(e: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """h_t = omega + alpha e_{t-1}^2 + beta h_{t-1}, from h_0 = the sample variance (a linear filter)."""
    from scipy.signal import lfilter

    x = np.empty(len(e))
    x[0] = np.var(e)
    x[1:] = omega + alpha * e[:-1] ** 2
    return lfilter([1.0], [1.0, -beta], x)


def _garch_params(theta: np.ndarray) -> tuple[float, float, float, float]:
    """(mu, omega, alpha, beta) from unconstrained theta: omega > 0, alpha, beta > 0, alpha + beta < 1."""
    mu = theta[0]
    omega = np.exp(theta[1])
    a, b = np.exp(theta[2]), np.exp(theta[3])
    total = a + b
    scale = 1 / (1 + total)  # (a, b) / (1 + a + b): positive and summing below one
    return float(mu), float(omega), float(a * scale), float(b * scale)


def _theta_from(mu: float, omega: float, alpha: float, beta: float) -> np.ndarray:
    rest = 1 - alpha - beta
    return np.array([mu, np.log(omega), np.log(alpha / rest), np.log(beta / rest)])


@dataclass(frozen=True, eq=False)
class SemiparametricGARCH:
    """GARCH(1, 1) estimated with the kernel density of its standardised shocks, beside the Gaussian QML fit."""

    variable: str
    params: np.ndarray
    cov: np.ndarray = field(repr=False)
    qml: np.ndarray = field(repr=False)
    qml_se: np.ndarray = field(repr=False)
    density: KernelDensity = field(repr=False)
    volatility: pd.Series = field(repr=False)
    loglike: float = 0.0
    rounds: int = 1
    draws: int = 0

    stage = "estimate"
    names = ("mu", "omega", "alpha", "beta")

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def summary(self) -> Table:
        rows = [[float(v), float(s), float(q), float(qs)] for v, s, q, qs in zip(self.params, self.se, self.qml, self.qml_se)]
        notes = (f"Engle and Gonzalez-Rivera (1991): the GARCH(1, 1) likelihood with the shocks' density estimated by kernel "
                 f"(bandwidth {self.density.bandwidth:.4g}; interpolated on a fine grid), {self.rounds} round(s) from Gaussian QML; "
                 + (f"standard errors from {self.draws} bootstrap draws (resampled standardised shocks). " if self.draws else
                    "no standard errors: give draws= for a bootstrap. ") + "QML standard errors are robust (sandwich).",)
        return Table(["estimate", "std err", "QML", "QML std err"], rows, index=list(self.names), index_name="parameter",
                     title=f"Semiparametric GARCH: {self.variable}", notes=notes)

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return pd.DataFrame({"estimate": self.params, "se": self.se, "qml": self.qml, "qml_se": self.qml_se}, index=list(self.names))

    def plot(self) -> Any:
        return self.density.plot()

    def __repr__(self) -> str:
        return f"<SemiparametricGARCH {self.variable}: {dict(zip(self.names, np.round(self.params, 4)))}>"


def _hessian(f: Any, x: np.ndarray) -> np.ndarray:
    n = len(x)
    H = np.empty((n, n))
    h = 1e-4 * np.maximum(np.abs(x), 1e-2)
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i], ej[j] = h[i], h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


def _delta(theta: np.ndarray, cov_theta: np.ndarray) -> np.ndarray:
    J = np.empty((4, 4))
    for j in range(4):
        h = 1e-6
        up, dn = theta.copy(), theta.copy()
        up[j] += h
        dn[j] -= h
        J[:, j] = (np.array(_garch_params(up)) - np.array(_garch_params(dn))) / (2 * h)
    return J @ cov_theta @ J.T


def _semi_fit(r: np.ndarray, index: pd.Index, bandwidth: Any, rounds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, KernelDensity, float]:
    """(theta, the QML theta, its sandwich covariance in (mu, omega, alpha, beta), the shocks' density, log likelihood)."""
    from .data import Dataset as _Dataset

    def qml_negll(theta: np.ndarray) -> np.ndarray:
        mu, om, a, b = _garch_params(theta)
        e = r - mu
        h = _garch_variance(e, om, a, b)
        return 0.5 * (np.log(2 * np.pi) + np.log(h) + e**2 / h)

    v = np.var(r)
    start = _theta_from(float(np.mean(r)), 0.05 * v, 0.05, 0.9)
    theta = optimize.minimize(lambda th: qml_negll(th).sum(), start, method="BFGS", options={"gtol": 1e-8, "maxiter": 5000}).x
    qml_theta = theta.copy()
    H = _hessian(lambda th: -qml_negll(th).sum(), theta)
    G = np.empty((len(r), 4))
    for j in range(4):  # per-period scores, for the sandwich
        step = 1e-6
        up, dn = theta.copy(), theta.copy()
        up[j] += step
        dn[j] -= step
        G[:, j] = -(qml_negll(up) - qml_negll(dn)) / (2 * step)
    Hinv = np.linalg.pinv(-H)
    qml_cov = _delta(theta, Hinv @ (G.T @ G) @ Hinv)
    density, ll = None, np.nan
    for _ in range(max(int(rounds), 1)):
        mu, om, a, b = _garch_params(theta)
        e = r - mu
        z = e / np.sqrt(_garch_variance(e, om, a, b))
        z = (z - z.mean()) / z.std()
        density = kde(_Dataset.from_frame(pd.DataFrame({"z": z}, index=index)), "z", bandwidth=bandwidth, points=4096, cut=6)
        grid, logf = density.grid, np.log(np.maximum(density.density, 1e-300))  # the log density, interpolated on a fine grid

        def negll(th: np.ndarray, grid: np.ndarray = grid, logf: np.ndarray = logf) -> float:
            mu, om, a, b = _garch_params(th)
            e = r - mu
            h = _garch_variance(e, om, a, b)
            return float(-np.sum(np.interp(e / np.sqrt(h), grid, logf, left=logf[0], right=logf[-1]) - 0.5 * np.log(h)))

        found = optimize.minimize(negll, theta, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000})
        theta, ll = found.x, -found.fun
    return theta, qml_theta, qml_cov, density, float(ll)


@api("pymetrix:semiparametric_garch")
def semiparametric_garch(data: Dataset, variable: str, bandwidth: Any = "silverman", rounds: int = 2, draws: int = 0,
                         seed: int = 0, span: Any = None) -> SemiparametricGARCH:
    """GARCH(1, 1) for ``variable`` (a constant mean) by Gaussian QML, then re-estimated ``rounds`` times by maximising
    the likelihood with the standardised shocks' kernel density in place of the normal (Engle and Gonzalez-Rivera 1991).
    Standard errors need ``draws``: a bootstrap simulating the fitted GARCH with resampled standardised shocks (the
    Hessian with the density held fixed understates them badly)."""
    s = _column(data, variable)
    if span is not None:
        s = s[data.sample_mask(span)]
    s = s.dropna()
    r = s.to_numpy(float)
    if len(r) < 100:
        raise SpecificationError("a semiparametric GARCH needs at least 100 observations")
    theta, qml_theta, qml_cov, density, ll = _semi_fit(r, s.index, bandwidth, rounds)
    mu, om, a, b = _garch_params(theta)
    cov = np.full((4, 4), np.nan)
    if draws:
        rng = make_rng(seed)
        e = r - mu
        h = _garch_variance(e, om, a, b)
        z = e / np.sqrt(h)
        sims = []
        for _ in range(int(draws)):
            shocks = rng.choice(z, size=len(r), replace=True)
            hs, rs = np.empty(len(r)), np.empty(len(r))
            hs[0] = h[0]
            for t in range(len(r)):
                if t:
                    hs[t] = om + a * (rs[t - 1] - mu) ** 2 + b * hs[t - 1]
                rs[t] = mu + np.sqrt(hs[t]) * shocks[t]
            sims.append(_garch_params(_semi_fit(rs, s.index, bandwidth, rounds)[0]))
        cov = np.cov(np.array(sims).T)
    vol = pd.Series(np.sqrt(_garch_variance(r - mu, om, a, b)), index=s.index, name="volatility")
    return SemiparametricGARCH(variable, np.array([mu, om, a, b]), cov, np.array(_garch_params(qml_theta)), np.sqrt(np.diag(qml_cov)),
                               density, vol, ll, int(rounds), int(draws))
