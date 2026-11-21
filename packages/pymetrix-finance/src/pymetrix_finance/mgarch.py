"""Multivariate GARCH: conditional covariances and correlations of several returns.

    fit = mgarch(data, ["r1", "r2", "r3"], model="dcc")
    fit.summary(); fit.plot()                      # parameters; conditional correlations and volatilities
    fit.correlations                               # each pair's correlation by period
    fit.forecast(10)                               # covariances and correlations up to 10 periods ahead
    fit.portfolio_var([0.5, 0.3, 0.2], level=0.99) # next period's VaR and ES of a portfolio

The models, with H_t the covariance of the returns' surprises e_t given the past:

- ``ccc``, constant conditional correlation (Bollerslev 1990): a GARCH(1,1) for each series, H_t = D_t R D_t with
  D_t their conditional standard deviations and R the correlation of the standardised residuals z_t.
- ``dcc``, dynamic conditional correlation (Engle 2002): the same first stage, then
  Q_t = (1 - a - b) Qbar + a z_t-1 z_t-1' + b Q_t-1 and R_t = Q_t scaled to unit diagonal, with Qbar the covariance of
  z and (a, b) by quasi-maximum likelihood given the first stage. Their standard errors come from the second stage
  alone and so understate the uncertainty a little.
- ``bekk``, scalar BEKK with variance targeting (Engle and Kroner 1995): H_t = (1 - a - b) S + a e_t-1 e_t-1' + b H_t-1,
  S the sample covariance; ``diagonal_bekk``: H_ij,t = (1 - a_i a_j - b_i b_j) S_ij + a_i a_j e_i e_j + b_i b_j H_ij,t-1.
  Both by Gaussian quasi-maximum likelihood on the demeaned returns.
- ``deco``, dynamic equicorrelation (Engle and Kelly 2012): DCC's Q_t, but every pair given the same correlation at
  t, rho_t, the mean of DCC's correlations; the equicorrelation matrix's determinant and inverse are in closed form,
  so it suits many series. With two series it is DCC.
- ``diagonal_vech`` (Bollerslev, Engle and Wooldridge 1988, with variance targeting): H_ij,t = (1 - a_ij - b_ij) S_ij +
  a_ij e_i e_j + b_ij H_ij,t-1, a free (a_ij, b_ij) for each pair; it nests the diagonal and scalar BEKK, but H_t is
  not guaranteed positive definite, and periods where it is not are refused by the likelihood.
- ``vech``, the full VECH (Bollerslev, Engle and Wooldridge 1988) with variance targeting: vech(H_t) = (I - A - B) vech(S)
  + A vech(e_t-1 e_t-1') + B vech(H_t-1), A and B unrestricted m x m (m = N(N + 1) / 2), so every variance and
  covariance can respond to every square and cross-product. With 2 m^2 parameters it is for two or three series; it
  starts from the diagonal VECH it nests.
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
from .risk import _fit_garch, normal

if TYPE_CHECKING:
    from pymetrix.data import Dataset
    from pymetrix.figures import Figure

MODELS = ("ccc", "dcc", "adcc", "deco", "bekk", "diagonal_bekk", "diagonal_vech", "vech")


def _scale_to_unit(Q: np.ndarray) -> np.ndarray:
    d = 1 / np.sqrt(np.diag(Q))
    return Q * np.outer(d, d)


def dcc_step(Q: np.ndarray, z: np.ndarray, a: float, b: float, Qbar: np.ndarray, g: float = 0.0, Nbar: Any = None) -> np.ndarray:
    """The next Q of DCC (g = 0) or asymmetric DCC: (1 - a - b) Qbar - g Nbar + a z z' + g n n' + b Q, n = min(z, 0)."""
    Q = (1 - a - b) * Qbar + a * np.outer(z, z) + b * Q
    if g:
        n = np.minimum(z, 0.0)
        Q = Q + g * (np.outer(n, n) - Nbar)
    return Q


def adcc_bound(Qbar: np.ndarray, Nbar: np.ndarray) -> float:
    """delta, the largest eigenvalue of Qbar^-1/2 Nbar Qbar^-1/2: asymmetric DCC is stationary if a + b + delta g < 1."""
    w, V = np.linalg.eigh(Qbar)
    root = V @ np.diag(w**-0.5) @ V.T
    return float(np.max(np.linalg.eigvalsh(root @ Nbar @ root)))


def dcc_path(z: np.ndarray, a: float, b: float, Qbar: np.ndarray, g: float = 0.0, Nbar: Any = None) -> np.ndarray:
    """The (asymmetric) DCC correlations R_t (T, N, N) of standardised residuals z (T, N), starting from Q_1 = Qbar:
    ``dcc_step`` period by period, run as a linear filter for every element at once (``dcc_path_linear``)."""
    return dcc_path_linear(z, a, b, Qbar, g, Nbar)[0]


def dcc_path_linear(z: np.ndarray, a: float, b: float, Qbar: np.ndarray, g: float = 0.0,
                    Nbar: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """``dcc_step`` run by a linear filter: each element of Q_t is the scalar recursion
    q_t = (1 - a - b) qbar - g nbar + a z_(t-1),i z_(t-1),j + g n_(t-1),i n_(t-1),j + b q_(t-1), q_1 = qbar
    (n = min(z, 0); g = 0 for DCC), run for all of them at once (scipy's lfilter) rather than period by period.
    The correlations R_t (T, N, N) and the next period's Q."""
    from scipy.signal import lfilter

    T, N = z.shape
    iu = np.triu_indices(N)
    x = (1 - a - b) * Qbar[iu] + a * (z[:, iu[0]] * z[:, iu[1]])  # x_t drives Q_(t+1), t = 0..T-1
    if g:
        n = np.minimum(z, 0.0)
        x = x + g * (n[:, iu[0]] * n[:, iu[1]] - np.asarray(Nbar)[iu])
    q = lfilter([1.0], [1.0, -b], x, axis=0, zi=(b * Qbar[iu])[None, :])[0]  # Q_2 .. Q_(T+1)
    Q = np.empty((T + 1, N, N))
    Q[0] = Qbar
    Q[1:, iu[0], iu[1]] = q
    Q[1:, iu[1], iu[0]] = q
    s = 1 / np.sqrt(np.einsum("tii->ti", Q[:T]))
    return Q[:T] * s[:, :, None] * s[:, None, :], Q[T]


def dcc_objective(params: np.ndarray, z: np.ndarray, Qbar: np.ndarray, Nbar: Any = None) -> float:
    """Minus the second-stage log likelihood of (asymmetric) DCC: half the sum of log|R_t| + z_t' R_t^-1 z_t - z_t'z_t."""
    a, b = params[0], params[1]
    g = params[2] if len(params) > 2 else 0.0
    if a < 0 or b < 0 or g < 0 or a + b >= 1:
        return 1e12
    if g and a + b + adcc_bound(Qbar, Nbar) * g >= 1:
        return 1e12
    R = dcc_path(z, a, b, Qbar, g, Nbar)
    sign, logdet = np.linalg.slogdet(R)
    if np.any(sign <= 0):
        return 1e12
    quad = np.einsum("ti,tij,tj->t", z, np.linalg.inv(R), z)
    return float(0.5 * np.sum(logdet + quad - np.sum(z**2, axis=1)))


def equicorrelation(R: np.ndarray) -> np.ndarray:
    """rho_t, the mean off-diagonal correlation of each R_t (T, N, N)."""
    N = R.shape[1]
    return (R.sum(axis=(1, 2)) - N) / (N * (N - 1))


def deco_objective(params: np.ndarray, z: np.ndarray, Qbar: np.ndarray) -> float:
    """Minus DECO's second-stage log likelihood, in closed form: |R| = (1 - rho)^(N-1) (1 + (N - 1) rho) and
    R^-1 = (I - rho / (1 + (N - 1) rho) J) / (1 - rho)."""
    a, b = params
    if a < 0 or b < 0 or a + b >= 1:
        return 1e12
    N = z.shape[1]
    rho = equicorrelation(dcc_path(z, a, b, Qbar))
    if np.any(rho <= -1 / (N - 1)) or np.any(rho >= 1):
        return 1e12
    logdet = (N - 1) * np.log(1 - rho) + np.log(1 + (N - 1) * rho)
    s, ss = z.sum(axis=1), np.sum(z**2, axis=1)
    quad = (ss - rho / (1 + (N - 1) * rho) * s**2) / (1 - rho)
    return float(0.5 * np.sum(logdet + quad - ss))


def _equi_matrix(rho: np.ndarray, N: int) -> np.ndarray:
    return (1 - rho)[:, None, None] * np.eye(N) + rho[:, None, None] * np.ones((N, N))


def bekk_path(e: np.ndarray, S: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Conditional covariances (T, N, N) of a variance-targeted BEKK with H_ij = (1 - A_ij - B_ij) S_ij + A_ij e_i e_j +
    B_ij H_ij,t-1 (A = a a' and B = b b' for the diagonal model, constants for the scalar), starting from S.

    Each element is a scalar first-order recursion, run over every period at once by a linear filter (scipy's lfilter):
    the same arithmetic as the loop over periods, which is what took the time of a fit."""
    from scipy.signal import lfilter

    T, N = e.shape
    A, B = np.broadcast_to(A, (N, N)), np.broadcast_to(B, (N, N))
    H = np.empty((T, N, N))
    H[0] = S
    if T > 1:
        x = (1 - A - B) * S + A * (e[:-1, :, None] * e[:-1, None, :])  # x_t drives H_(t+1)
        for i in range(N):
            for j in range(N):
                b = float(B[i, j])
                H[1:, i, j] = lfilter([1.0], [1.0, -b], x[:, i, j], zi=[b * S[i, j]])[0]
    return H


def vech_path(e: np.ndarray, S: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Conditional covariances (T, N, N) of the full VECH with variance targeting, starting from S.

    The terms that do not depend on H, (I - A - B) vech(S) + A vech(e_t e_t'), are formed for every period at once;
    only vech(H_t+1) = that + B vech(H_t) is left to the loop over periods."""
    T, N = e.shape
    iu = np.triu_indices(N)
    s = S[iu]
    c = (np.eye(len(s)) - A - B) @ s
    u = c + (e[:, iu[0]] * e[:, iu[1]]) @ A.T  # u_t drives vech(H_t+1)
    h = np.empty((T, len(s)))
    if T:
        h[0] = s
    for t in range(T - 1):
        h[t + 1] = u[t] + B @ h[t]
    H = np.empty((T, N, N))
    H[:, iu[0], iu[1]] = h
    H[:, iu[1], iu[0]] = h
    return H


def _vech_fit(e: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    T, N = e.shape
    m = N * (N + 1) // 2
    S = e.T @ e / T
    diag, _, _ = _bekk_fit(e, "diagonal_vech")

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return theta[:m * m].reshape(m, m), theta[m * m:].reshape(m, m)

    def nll(theta: np.ndarray) -> float:
        Am, Bm = unpack(theta)
        if np.max(np.abs(np.linalg.eigvals(Am + Bm))) >= 0.9999:
            return 1e12
        H = vech_path(e, S, Am, Bm)
        ll = gaussian_loglike(e, H)
        return 1e12 if ll <= -1e299 else -ll

    start = np.r_[np.diag(diag[:m]).ravel(), np.diag(diag[m:]).ravel()]
    found = optimize.minimize(nll, start, method="L-BFGS-B", options={"maxiter": 2000})
    found = optimize.minimize(nll, found.x, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-9, "fatol": 1e-10})
    theta = found.x if found.fun <= nll(start) else start
    return theta, -nll(theta), S


def gaussian_loglike(e: np.ndarray, H: np.ndarray) -> float:
    N = e.shape[1]
    sign, logdet = np.linalg.slogdet(H)
    if np.any(sign <= 0):
        return -1e300
    quad = np.einsum("ti,tij,tj->t", e, np.linalg.inv(H), e)
    return float(-0.5 * np.sum(N * np.log(2 * np.pi) + logdet + quad))


def _bekk_matrices(theta: np.ndarray, model: str, N: int) -> tuple[np.ndarray, np.ndarray]:
    if model == "bekk":
        return np.full((N, N), theta[0]), np.full((N, N), theta[1])
    if model == "diagonal_vech":  # theta: the upper triangles of A, then of B
        iu = np.triu_indices(N)
        m = len(iu[0])
        A, B = np.zeros((N, N)), np.zeros((N, N))
        A[iu], B[iu] = theta[:m], theta[m:]
        return A + np.triu(A, 1).T, B + np.triu(B, 1).T
    a, b = theta[:N], theta[N:]
    return np.outer(a, a), np.outer(b, b)


def _bekk_fit(e: np.ndarray, model: str) -> tuple[np.ndarray, float, np.ndarray]:
    T, N = e.shape
    S = e.T @ e / T

    def nll(theta: np.ndarray) -> float:
        if model == "bekk":
            a, b = theta
            if a < 0 or b < 0 or a + b >= 0.9999:
                return 1e12
        elif model == "diagonal_vech":
            m = len(theta) // 2
            if np.any(theta < 0) or np.any(theta[:m] + theta[m:] >= 0.9999):
                return 1e12
        else:
            if np.any(theta[:N] < 0) or np.any(theta[N:] < 0) or np.any(theta[:N] ** 2 + theta[N:] ** 2 >= 0.9999):
                return 1e12
        A, B = _bekk_matrices(theta, model, N)
        intercept = (1 - A - B) * S
        if np.linalg.eigvalsh(intercept).min() <= 0:  # the targeted constant must be a covariance
            return 1e12
        return -gaussian_loglike(e, bekk_path(e, S, A, B))

    if model == "bekk":
        start = np.array([0.05, 0.9])
    elif model == "diagonal_vech":  # from the scalar BEKK: the model it nests
        scalar, _, _ = _bekk_fit(e, "bekk")
        m = N * (N + 1) // 2
        start = np.r_[np.full(m, scalar[0]), np.full(m, scalar[1])]
    else:
        start = np.r_[np.full(N, np.sqrt(0.05)), np.full(N, np.sqrt(0.9))]
    found = optimize.minimize(nll, start, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-10})
    found = optimize.minimize(nll, found.x, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-12})
    return found.x, -found.fun, S


def _hessian_se(f: Any, x: np.ndarray) -> np.ndarray:
    from pymetrix.extension import numerical_hessian as hessian

    try:
        H = hessian(f, x)
        return np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))
    except np.linalg.LinAlgError:
        return np.full(len(x), np.nan)


@dataclass(frozen=True, eq=False)
class FittedMGARCH:
    """An estimated multivariate GARCH: parameters, conditional covariances and correlations, and forecasts."""

    model: str
    variables: tuple[str, ...]
    index: pd.Index = field(repr=False)
    returns: np.ndarray = field(repr=False)  # (T, N)
    mu: np.ndarray = field(repr=False)  # each series' mean
    H: np.ndarray = field(repr=False)  # (T, N, N): the covariance of period t's surprise given the past
    names: tuple[str, ...] = ()
    values: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    se: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    loglike: float = float("nan")
    state: dict[str, Any] = field(default_factory=dict, repr=False)  # what forecasts need

    stage = "finance"

    @property
    def params(self) -> Any:
        from pymetrix.extension import point_estimate as point

        cov = np.diag(self.se**2) if len(self.se) and np.all(np.isfinite(self.se)) else None
        return point(list(self.names), self.values, cov)

    @property
    def stats(self) -> dict[str, float]:
        T, N = self.returns.shape
        k = len(self.values)
        return {"nobs": float(T), "series": float(N), "k": float(k), "loglike": self.loglike,
                "aic": -2 * self.loglike + 2 * k, "bic": -2 * self.loglike + k * np.log(T)}

    @property
    def correlations(self) -> pd.DataFrame:
        """Each pair's conditional correlation by period."""
        d = np.sqrt(np.einsum("tii->ti", self.H))
        out = {}
        for i in range(len(self.variables)):
            for j in range(i + 1, len(self.variables)):
                out[f"{self.variables[i]} · {self.variables[j]}"] = self.H[:, i, j] / (d[:, i] * d[:, j])
        return pd.DataFrame(out, index=self.index)

    @property
    def volatilities(self) -> pd.DataFrame:
        """Each series' conditional standard deviation by period."""
        return pd.DataFrame(np.sqrt(np.einsum("tii->ti", self.H)), index=self.index, columns=list(self.variables))

    def _next(self) -> np.ndarray:
        """The covariance forecast for the period after the sample."""
        return self.forecast(1)[0]

    def forecast(self, horizon: int = 10) -> np.ndarray:
        """The covariance matrices forecast 1 to ``horizon`` periods after the sample, (horizon, N, N). DCC's
        correlations use Engle and Sheppard's approximation E[z z'] = R."""
        s = self.state
        N = len(self.variables)
        out = np.empty((horizon, N, N))
        if self.model in ("ccc", "dcc", "adcc", "deco"):
            omega, alpha, beta = s["omega"], s["alpha"], s["beta"]
            var = s["var_next"].copy()
            persist = alpha + s.get("gamma", 0.0) / 2 + beta
            long_var = omega / np.maximum(1 - persist, 1e-12)
            if self.model in ("dcc", "adcc", "deco"):
                a, b, Qbar = s["a"], s["b"], s["Qbar"]
                R1 = _scale_to_unit(s["Q_next"])
                Rbar = _scale_to_unit(Qbar)
            for h in range(horizon):
                v = long_var + persist**h * (var - long_var)
                if self.model == "ccc":
                    R = s["R"]
                else:
                    R = Rbar + (a + b) ** h * (R1 - Rbar)
                    if self.model == "deco":
                        R = _equi_matrix(equicorrelation(R[None]), N)[0]
                d = np.sqrt(v)
                out[h] = R * np.outer(d, d)
            return out
        if self.model == "vech":
            iu = np.triu_indices(N)
            D = s["vech_A"] + s["vech_B"]
            gap = (s["H_next"] - s["S"])[iu]
            for h in range(horizon):
                v = s["S"][iu] + np.linalg.matrix_power(D, h) @ gap
                Hh = np.zeros((N, N))
                Hh[iu] = v
                out[h] = Hh + np.triu(Hh, 1).T
            return out
        A, B, S, h_next = s["A"], s["B"], s["S"], s["H_next"]
        for h in range(horizon):
            out[h] = S + (A + B) ** h * (h_next - S)
        return out

    def portfolio_var(self, weights: Sequence[float], level: float = 0.99) -> dict[str, float]:
        """Next period's VaR and ES (normal) of a portfolio with these weights, from the forecast covariance."""
        w = np.asarray(weights, float)
        if w.shape != (len(self.variables),):
            raise SpecificationError(f"give a weight for each of {', '.join(self.variables)}")
        H = self._next()
        sd = float(np.sqrt(w @ H @ w))
        v, e = normal(float(w @ self.mu), sd, 1 - level)
        return {"var": v, "es": e, "sd": sd, "mean": float(w @ self.mu)}

    def summary(self) -> Table:
        rows = [[float(v), float(s)] for v, s in zip(self.values, self.se)]
        notes = [f"Periods: {len(self.index)}", f"Series: {len(self.variables)}", f"Log likelihood: {self.loglike:.4f}"]
        if self.model in ("dcc", "adcc", "deco"):
            notes.append(f"First stage: a {'GJR-' if self.state.get('asymmetric') else ''}GARCH(1,1) for each series; the "
                         f"{self.model.upper()} parameters' standard errors are from the second stage alone.")
        if self.model == "adcc":
            notes.append("Asymmetric DCC (Cappiello, Engle and Sheppard 2006): g is the extra response of correlations to joint falls; "
                         "forecasts use DCC's approximation.")
        if self.model == "deco":
            notes.append("Dynamic equicorrelation (Engle and Kelly 2012): every pair's correlation at t is the mean of DCC's.")
        if self.model == "vech":
            notes.append("Full VECH: standard errors are not reported (2 m^2 parameters); compare the log likelihood with the diagonal VECH's.")
        if self.model in ("bekk", "diagonal_bekk", "diagonal_vech", "vech"):
            notes.append("Variance targeting: the constant is set by the sample covariance of the demeaned returns.")
        if self.model == "ccc":
            notes.append("The correlation of the standardised residuals is in the Correlations tab.")
        return Table(["estimate", "std err"], rows, index=list(self.names), index_name="parameter",
                     title=f"{self.model.upper().replace('_', ' ')} GARCH: {', '.join(self.variables)}", notes=tuple(notes))

    def covariance_table(self) -> Table:
        H = self._next()
        return Table(list(self.variables), H.tolist(), index=list(self.variables), index_name="",
                     title="Covariance forecast for the next period")

    def plot(self) -> Figure:
        from pymetrix.figures import Figure, Line, Panel

        corr = self.correlations
        vol = self.volatilities
        top = Panel(tuple(Line(corr[c], c) for c in corr.columns), title="Conditional correlations")
        bottom = Panel(tuple(Line(vol[c], c) for c in vol.columns), title="Conditional volatilities")
        return Figure((top, bottom))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return pd.concat([self.volatilities.add_prefix("sd "), self.correlations.add_prefix("corr ")], axis=1)

    def __repr__(self) -> str:
        return f"<FittedMGARCH {self.model} of {', '.join(self.variables)}: {len(self.index)} periods>"


with plugin_context(PLUGIN):
    @api("finance:mgarch")
    def mgarch(data: Dataset, variables: Sequence[str],
               model: Literal["ccc", "dcc", "adcc", "deco", "bekk", "diagonal_bekk", "diagonal_vech", "vech"] = "dcc",
               asymmetric: bool = False) -> FittedMGARCH:
        """A multivariate GARCH of the returns in ``variables``: constant (ccc), dynamic (dcc) or asymmetric dynamic
        (adcc) conditional correlation or dynamic equicorrelation (deco) over GARCH(1,1) volatilities (GJR with
        ``asymmetric``), or a scalar or diagonal BEKK or diagonal VECH (see the module notes)."""
        from pymetrix.extension import report_progress as report

        variables = [variables] if isinstance(variables, str) else list(variables)
        if model not in MODELS:
            raise SpecificationError(f"model is one of {', '.join(MODELS)}")
        if len(variables) < 2:
            raise SpecificationError("a multivariate GARCH needs at least two series of returns")
        if data.time_axis is None or data.kind != "time_series":
            raise SpecificationError("a multivariate GARCH needs returns over time")
        missing = [v for v in variables if v not in data.names]
        if missing:
            raise SpecificationError(f"no series {', '.join(map(repr, missing))} in the dataset")
        frame = data.frame[variables].dropna()
        if len(frame) < 100:
            raise SpecificationError("at least 100 periods of returns are needed")
        r = frame.to_numpy(float)
        T, N = r.shape
        if asymmetric and model in ("bekk", "diagonal_bekk", "diagonal_vech", "vech"):
            raise SpecificationError("asymmetric (GJR) volatilities are for the ccc, dcc and adcc models")
        if model in ("ccc", "dcc", "adcc", "deco"):
            fits = []
            for i, v in enumerate(variables):
                report(i / (N + 1), f"GARCH for {v}")
                fits.append(_fit_garch(frame[v], asymmetric=asymmetric))
            mu = np.array([f.mu for f in fits])
            z = np.column_stack([f.z for f in fits])
            sigma = np.column_stack([f.sigma for f in fits])
            parts = ("mean", "omega", "alpha", "gamma", "beta") if asymmetric else ("mean", "omega", "alpha", "beta")
            names = [f"{v}: {p}" for v, f in zip(variables, fits) for p in parts]
            values = [x for f in fits for x in ((f.mu, f.omega, f.alpha, f.gamma, f.beta) if asymmetric else (f.mu, f.omega, f.alpha, f.beta))]
            se = [np.nan] * len(values)
            state: dict[str, Any] = {"omega": np.array([f.omega for f in fits]), "alpha": np.array([f.alpha for f in fits]),
                                     "beta": np.array([f.beta for f in fits]), "var_next": np.array([f.var_next for f in fits]),
                                     "gamma": np.array([f.gamma for f in fits]), "asymmetric": bool(asymmetric)}
            loglike_vol = float(np.sum(-0.5 * (np.log(2 * np.pi) + 2 * np.log(sigma) + z**2)))
            if model == "ccc":
                R = np.corrcoef(z.T)
                Rt = np.broadcast_to(R, (T, N, N))
                state["R"] = R
                corr_ll = -0.5 * np.sum(np.linalg.slogdet(R)[1] + np.einsum("ti,ij,tj->t", z, np.linalg.inv(R), z) - np.sum(z**2, axis=1))
            elif model == "deco":
                report(N / (N + 1), "DECO correlations")
                Qbar = z.T @ z / T
                found = optimize.minimize(deco_objective, np.array([0.03, 0.95]), args=(z, Qbar), method="Nelder-Mead",
                                          options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 8000})
                a, b = found.x
                Rt = _equi_matrix(equicorrelation(dcc_path(z, a, b, Qbar)), N)
                Q_next = dcc_step(_q_at(z, a, b, Qbar), z[-1], a, b, Qbar)
                state.update(a=a, b=b, g=0.0, Qbar=Qbar, Nbar=None, Q_next=Q_next)
                names += ["deco a", "deco b"]
                values += list(found.x)
                se += list(_hessian_se(lambda p: deco_objective(p, z, Qbar), found.x))
                corr_ll = -found.fun
            else:
                report(N / (N + 1), f"{model.upper()} correlations")
                Qbar = z.T @ z / T
                neg = np.minimum(z, 0.0)
                Nbar = neg.T @ neg / T if model == "adcc" else None
                start = np.array([0.03, 0.95]) if model == "dcc" else np.array([0.02, 0.94, 0.01])
                found = optimize.minimize(dcc_objective, start, args=(z, Qbar, Nbar), method="Nelder-Mead",
                                          options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 8000})
                a, b = found.x[:2]
                g = float(found.x[2]) if model == "adcc" else 0.0
                Rt = dcc_path(z, a, b, Qbar, g, Nbar)
                Q_next = dcc_step(_q_at(z, a, b, Qbar, g, Nbar), z[-1], a, b, Qbar, g, Nbar)
                state.update(a=a, b=b, g=g, Qbar=Qbar, Nbar=Nbar, Q_next=Q_next)
                ab_se = _hessian_se(lambda p: dcc_objective(p, z, Qbar, Nbar), found.x)
                names += [f"{model} a", f"{model} b"] + ([f"{model} g"] if model == "adcc" else [])
                values += list(found.x)
                se += list(ab_se)
                corr_ll = -found.fun
            H = Rt * np.einsum("ti,tj->tij", sigma, sigma)
            loglike = loglike_vol + float(corr_ll)
        elif model == "vech":
            if N > 3:
                raise SpecificationError("the full VECH has 2 m^2 parameters: use it for two or three series (diagonal_vech for more)")
            mu = r.mean(axis=0)
            e = r - mu
            report(0.1, "full VECH by maximum likelihood")
            theta, loglike, S = _vech_fit(e)
            m = N * (N + 1) // 2
            Av, Bv = theta[:m * m].reshape(m, m), theta[m * m:].reshape(m, m)
            H = vech_path(e, S, Av, Bv)
            iu = np.triu_indices(N)
            h_next = (np.eye(m) - Av - Bv) @ S[iu] + Av @ np.outer(e[-1], e[-1])[iu] + Bv @ H[-1][iu]
            Hn = np.zeros((N, N))
            Hn[iu] = h_next
            state = {"vech_A": Av, "vech_B": Bv, "S": S, "H_next": Hn + np.triu(Hn, 1).T}
            pairs = [f"{variables[i]},{variables[j]}" for i, j in zip(*iu)]
            names = [f"A[{p}; {q}]" for p in pairs for q in pairs] + [f"B[{p}; {q}]" for p in pairs for q in pairs]
            values = list(theta)
            se = [np.nan] * len(values)
        else:
            mu = r.mean(axis=0)
            e = r - mu
            report(0.1, f"{model.replace('_', ' ')} by maximum likelihood")
            theta, loglike, S = _bekk_fit(e, model)
            A, B = _bekk_matrices(theta, model, N)
            H = bekk_path(e, S, A, B)
            H_next = (1 - A - B) * S + A * np.outer(e[-1], e[-1]) + B * H[-1]
            state = {"A": A, "B": B, "S": S, "H_next": H_next}
            if model == "bekk":
                names, values = ["a", "b"], list(theta)
            elif model == "diagonal_vech":
                pairs = [f"{variables[i]}, {variables[j]}" for i, j in zip(*np.triu_indices(N))]
                names, values = [f"a {p}" for p in pairs] + [f"b {p}" for p in pairs], list(theta)
            else:
                names = [f"a {v}" for v in variables] + [f"b {v}" for v in variables]
                values = list(theta)
            se = list(_hessian_se(lambda th: -gaussian_loglike(e, bekk_path(e, S, *_bekk_matrices(th, model, N))), theta))
        return FittedMGARCH(model, tuple(variables), frame.index, r, mu, H, tuple(names), np.asarray(values, float),
                            np.asarray(se, float), float(loglike), state)


def _q_at(z: np.ndarray, a: float, b: float, Qbar: np.ndarray, g: float = 0.0, Nbar: Any = None) -> np.ndarray:
    """The (asymmetric) DCC Q of the last period of the sample."""
    Q = Qbar.copy()
    for t in range(len(z) - 1):
        Q = dcc_step(Q, z[t], a, b, Qbar, g, Nbar)
    return Q
