"""Bayesian VARs with stochastic volatility (Cogley and Sargent 2005; Carriero, Clark and Marcellino 2019).

    fit = BVARSV(["gy", "pi", "r"], lags=4).fit(data, draws=2000, burn=1000)
    fit.volatilities(); fit.plot(); fit.irf(20); fit.irf(20, period="2008Q4")
    fit.fevd(20, period="2008Q4"); fit.historical_decomposition()     # with bands over the posterior draws

y_t = B'x_t + A^-1 D_t e_t, x_t the constant and p lags, A lower triangular with ones on its diagonal, D_t diagonal
with log standard deviations h_i,t following random walks, h_i,t = h_i,t-1 + eta_i,t, var(eta_i) = w_i. Unlike the
TVP-VAR the coefficients are constant, so a larger VAR stays estimable; the volatilities still change.

Priors: B ~ the Minnesota prior in its independent normal form (own first lag own_mean, the coefficient on lag l of
variable j in equation i with standard deviation tightness / l^decay sqrt(psi_i / psi_j), constants loose), psi from
univariate autoregressions; the free elements of A ~ N(0, 10); h_i,0 ~ N(log sqrt(psi_i), 1); w_i ~ inverse gamma
with scale 4 kw^2 and 4 degrees of freedom. Gibbs sampling: B by generalised least squares given A and the
volatilities (all equations together), each row of A by weighted least squares, the volatilities by forward filtering
and backward sampling through Kim, Shephard and Chib's mixture (the indicators then drawn given them, Del Negro and
Primiceri's order), and w_i. Responses are identified recursively (A^-1 D_t), at the last period unless a date is given.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from ._api import api
from ._random import generator as make_rng
from .errors import SpecificationError
from .output import Table
from .tvpvar import KSC_MEANS, KSC_VARIANCES, KSC_WEIGHTS, ffbs

if TYPE_CHECKING:
    from .data import Dataset
    from .figures import Figure

__all__ = ["BVARSV", "FittedBVARSV", "coefficient_posterior"]


def coefficient_posterior(Y: np.ndarray, X: np.ndarray, sigma_inv: np.ndarray, b0: np.ndarray, V0_inv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mean, covariance) of vec(B) (equations stacked) given each period's error precision sigma_inv (T, K, K):
    precision V0^-1 + sum_t sigma_inv_t (x) x_t x_t', mean from V0^-1 b0 + sum_t (sigma_inv_t y_t) (x) x_t."""
    T, k = X.shape
    K = Y.shape[1]
    blocks = np.einsum("tij,ta,tb->iajb", sigma_inv, X, X).reshape(K * k, K * k)
    rhs = np.einsum("tij,tj,ta->ia", sigma_inv, Y, X).reshape(K * k)
    P = V0_inv + blocks
    cov = np.linalg.inv(P)
    cov = (cov + cov.T) / 2
    return cov @ (V0_inv @ b0 + rhs), cov


def _design(Y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    T = len(Y)
    X = np.column_stack([np.ones(T - p)] + [Y[p - l:T - l] for l in range(1, p + 1)])
    return Y[p:], X


@dataclass(frozen=True, eq=False)
class BVARSV:
    """A Bayesian VAR with stochastic volatility: constant coefficients, drifting shock volatilities."""

    variables: Sequence[str]
    lags: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", tuple(self.variables))
        if len(self.variables) < 1 or self.lags < 1:
            raise SpecificationError("a BVAR-SV needs variables and at least one lag")

    @api("pymetrix:bvar_sv.fit")
    def fit(self, data: Dataset, draws: int = 2000, burn: int = 1000, tightness: float = 0.2, decay: float = 1.0,
            own_mean: float = 1.0, kw: float = 0.1, seed: int = 0, span: Any = None) -> FittedBVARSV:
        from .progress import report

        if data.kind != "time_series":
            raise SpecificationError("a BVAR-SV needs a single time series")
        missing = [v for v in self.variables if v not in data]
        if missing:
            raise SpecificationError(f"{', '.join(missing)} are not series of the data")
        frame = data.frame[list(self.variables)].astype(float)
        if span is not None:
            frame = frame[data.sample_mask(span)]
        frame = frame.dropna()
        Yall = frame.to_numpy()
        p, K = self.lags, len(self.variables)
        Y, X = _design(Yall, p)
        T, k = X.shape
        if T < 3 * k:
            raise SpecificationError("too few observations for this many lags")
        # psi: residual variances of univariate AR(p)s
        psi = np.empty(K)
        for i in range(K):
            yi, Xi = _design(Yall[:, i:i + 1], p)
            e = yi[:, 0] - Xi @ np.linalg.lstsq(Xi, yi[:, 0], rcond=None)[0]
            psi[i] = e @ e / len(e)
        b0 = np.zeros(K * k)
        v0 = np.empty(K * k)
        for i in range(K):
            for r in range(k):
                pos = i * k + r
                if r == 0:
                    v0[pos] = 100 * psi[i]
                else:
                    l, j = (r - 1) // K + 1, (r - 1) % K
                    v0[pos] = (tightness / l**decay) ** 2 * psi[i] / psi[j]
                    if l == 1 and j == i:
                        b0[pos] = own_mean
        V0_inv = np.diag(1 / v0)
        rng = make_rng(seed)
        Bv = np.linalg.lstsq(X, Y, rcond=None)[0].T.reshape(-1)  # start: least squares
        A = np.eye(K)
        h = np.tile(0.5 * np.log(psi), (T, 1))
        h0 = 0.5 * np.log(psi)
        w = np.full(K, 0.01)
        cbar = 1e-6
        mix = np.full((T, K), 4)
        keep_B, keep_A, keep_h, keep_w = [], [], [], []
        total = burn + draws
        for it in range(total):
            if it % 100 == 0:
                report(it / total, f"Gibbs sampling: {it} of {total}")
            # (1) the volatilities given the indicators
            U = Y - X @ Bv.reshape(K, k).T
            ystar = U @ A.T
            for i in range(K):
                lz = np.log(ystar[:, i] ** 2 + cbar)
                s = mix[:, i]
                h[:, i] = ffbs(lz[:, None], np.full((T, 1, 1), 2.0), KSC_VARIANCES[s][:, None, None], np.array([[w[i]]]),
                               np.array([h0[i]]), np.array([[1.0]]), rng, c=KSC_MEANS[s][:, None])[:, 0]
            # (2) the coefficients, given A and the volatilities
            Ainv = np.linalg.inv(A)
            prec = np.einsum("ji,tj,jk->tik", A, np.exp(-2 * h), A)  # A' D_t^-2 A
            mean, cov = coefficient_posterior(Y, X, prec, b0, V0_inv)
            Bv = mean + np.linalg.cholesky(cov) @ rng.standard_normal(K * k)
            U = Y - X @ Bv.reshape(K, k).T
            # (3) A by rows: u_i = -a_i' u_<i + D_i e_i
            for i in range(1, K):
                wts = np.exp(-2 * h[:, i])
                Z = -U[:, :i]
                Pa = np.eye(i) / 10 + (Z * wts[:, None]).T @ Z
                ca = np.linalg.inv(Pa)
                ma = ca @ ((Z * wts[:, None]).T @ U[:, i])
                A[i, :i] = ma + np.linalg.cholesky((ca + ca.T) / 2) @ rng.standard_normal(i)
            # (4) the indicators, given everything
            ystar = U @ A.T
            for i in range(K):
                lz = np.log(ystar[:, i] ** 2 + cbar)
                dens = KSC_WEIGHTS * sps.norm.pdf(lz[:, None], 2 * h[:, i][:, None] + KSC_MEANS, np.sqrt(KSC_VARIANCES))
                cum = np.cumsum(dens / dens.sum(axis=1, keepdims=True), axis=1)
                mix[:, i] = np.minimum((rng.random(T)[:, None] > cum).sum(axis=1), 6)
            # (5) the volatilities' innovation variances
            dh = np.diff(np.vstack([h0, h]), axis=0)
            for i in range(K):
                w[i] = 1 / rng.gamma((4 + T) / 2, 2 / (4 * kw**2 + dh[:, i] @ dh[:, i]))
            if it >= burn:
                keep_B.append(Bv.reshape(K, k).copy())
                keep_A.append(A.copy())
                keep_h.append(h.copy())
                keep_w.append(w.copy())
            del Ainv
        report(1.0, "done")
        return FittedBVARSV(self, np.array(keep_B), np.array(keep_A), np.array(keep_h), np.array(keep_w),
                            frame.index[p:], X, Y)


@dataclass(frozen=True, eq=False)
class FittedBVARSV:
    """Posterior draws of a BVAR with stochastic volatility."""

    spec: BVARSV
    B: np.ndarray = field(repr=False)  # (draws, K, 1 + K p): each equation's constant then lags
    A: np.ndarray = field(repr=False)  # (draws, K, K)
    h: np.ndarray = field(repr=False)  # (draws, T, K): log standard deviations
    w: np.ndarray = field(repr=False)
    index: pd.Index = field(repr=False)
    X: np.ndarray = field(repr=False)
    Y: np.ndarray = field(repr=False)

    stage = "estimate"

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self.spec.variables)

    @property
    def coefs(self) -> np.ndarray:
        """The posterior mean lag matrices, (lags, K, K)."""
        K, p = len(self.variables), self.spec.lags
        Bm = self.B.mean(axis=0)
        return np.stack([Bm[:, 1 + K * l:1 + K * (l + 1)] for l in range(p)])

    def volatilities(self, coverage: float = 0.68) -> pd.DataFrame:
        """Each shock's standard deviation by period: the posterior median and band."""
        sd = np.exp(self.h)
        lo, med, hi = np.quantile(sd, [(1 - coverage) / 2, 0.5, (1 + coverage) / 2], axis=0)
        cols = {}
        for i, v in enumerate(self.variables):
            cols[v], cols[f"{v}_lower"], cols[f"{v}_upper"] = med[:, i], lo[:, i], hi[:, i]
        return pd.DataFrame(cols, index=self.index)

    @api()
    def irf(self, periods: int = 20, period: Any = None, coverage: float = 0.68) -> Any:
        """Recursively identified responses with that period's volatilities (default: the last), median and bands
        across the draws."""
        from .var import IRF, _ma

        t = self._period(period)
        K, p = len(self.variables), self.spec.lags
        out = []
        for d in range(len(self.B)):
            lagm = np.stack([self.B[d][:, 1 + K * l:1 + K * (l + 1)] for l in range(p)])
            impact = np.linalg.inv(self.A[d]) * np.exp(self.h[d, t])
            out.append(_ma(lagm, periods) @ impact)
        out = np.array(out)
        lo, med, hi = np.quantile(out, [(1 - coverage) / 2, 0.5, (1 + coverage) / 2], axis=0)
        return IRF(med, self.variables, self.variables, lo, hi, False, f"recursive (BVAR-SV, volatilities of {self.index[t]})", coverage)

    def _period(self, period: Any) -> int:
        if period is None:
            return len(self.index) - 1
        try:
            return int(self.index.get_loc(pd.Period(str(period), freq=self.index.freq)))
        except (KeyError, ValueError) as exc:
            raise SpecificationError(f"{period!r} is not a period of the sample ({self.index[0]} to {self.index[-1]})") from exc

    def _lag_matrices(self) -> np.ndarray:
        """Each draw's lag matrices, (draws, lags, K, K)."""
        K, p = len(self.variables), self.spec.lags
        return np.stack([self.B[:, :, 1 + K * l:1 + K * (l + 1)] for l in range(p)], axis=1)

    @api()
    def fevd(self, periods: int = 20, period: Any = None, coverage: float = 0.68) -> Any:
        """Forecast-error variance decomposition with that period's volatilities (default: the last), as for ``irf``:
        for each draw the shares of the recursively identified shocks (impact A^-1 D_t), the volatilities held at
        their values in that period over the horizon. The point shares are the posterior means (so each row adds up
        to one), the bands pointwise quantiles over the draws."""
        from .var import FEVD, _ma
        from .var_decompositions import fevd_shares

        if not 0 < coverage < 1:
            raise SpecificationError(f"coverage is between 0 and 1, e.g. 0.68; got {coverage}")
        t = self._period(period)
        lagm = self._lag_matrices()
        shares = np.stack([fevd_shares(_ma(lagm[d], periods - 1) @ (np.linalg.inv(self.A[d]) * np.exp(self.h[d, t])))
                           for d in range(len(self.B))])
        lo, hi = np.quantile(shares, [(1 - coverage) / 2, (1 + coverage) / 2], axis=0)
        return FEVD(shares.mean(axis=0), self.variables, self.variables, f"recursive (BVAR-SV, volatilities of {self.index[t]}), posterior mean",
                    lo, hi, coverage, f"{len(shares)} posterior draws")

    @api()
    def historical_decomposition(self, coverage: float = 0.68, draws: int | None = None) -> Any:
        """Each variable over the sample, split into the contributions of the recursively identified shocks and the
        baseline (the path from the first observations and the constants, with no shocks), for each posterior draw
        (all, or the first ``draws``): the residuals u_t = y_t - B'x_t, the structural shocks e_t = D_t^-1 A u_t
        through that period's impact matrix A^-1 D_t, and their contributions through the draw's lag matrices. Each
        draw's parts add up to the data exactly; the point decomposition is their mean over draws (which does too),
        the bands pointwise quantiles. A shock's contribution in a period, A^-1[:, j] (A u_t)_j, does not depend on
        the volatility (it scales the shock and its impact in opposite directions); the standardised shocks do."""
        from .var_decompositions import decomposition, tv_contributions

        if not 0 < coverage < 1:
            raise SpecificationError(f"coverage is between 0 and 1, e.g. 0.68; got {coverage}")
        n = len(self.B) if not draws else min(int(draws), len(self.B))
        K, p = len(self.variables), self.spec.lags
        T = len(self.Y)
        B, A, h = self.B[:n], self.A[:n], self.h[:n]
        U = self.Y[None] - np.einsum("tk,nik->nti", self.X, B)
        impact = np.linalg.inv(A)[:, None] * np.exp(h)[:, :, None, :]  # (n, T, K, K): A^-1 D_t
        lagm = np.broadcast_to(self._lag_matrices()[:n, None], (n, T, p, K, K))
        const = np.broadcast_to(B[:, None, :, 0], (n, T, K))
        history = np.broadcast_to(np.stack([self.X[0, 1 + K * (l - 1):1 + K * l] for l in range(p, 0, -1)]), (n, p, K))
        contrib, base, e = tv_contributions(lagm, impact, U, const, history)
        return decomposition(contrib, base, e, self.Y, self.index, self.variables, "recursive, BVAR-SV, posterior mean",
                             "relative to the baseline: constants and initial conditions", coverage, f"{n} posterior draws")

    def summary(self) -> Table:
        K, p = len(self.variables), self.spec.lags
        names = ["const"] + [f"{v}[-{l}]" for l in range(1, p + 1) for v in self.variables]
        Bm, Bs = self.B.mean(axis=0), self.B.std(axis=0)
        rows = [[f"{eq}: {n}", float(Bm[i, r]), float(Bs[i, r])] for i, eq in enumerate(self.variables) for r, n in enumerate(names)]
        vol = np.exp(self.h).mean(axis=0)
        notes = (f"{len(self.B)} posterior draws, {len(self.index)} periods ({self.index[0]} to {self.index[-1]}).",
                 "Shock standard deviations (posterior mean), first and last period: "
                 + ", ".join(f"{v} {vol[0, i]:.4g} -> {vol[-1, i]:.4g}" for i, v in enumerate(self.variables)))
        return Table(["", "posterior mean", "posterior sd"], rows, title=f"BVAR with stochastic volatility: {', '.join(self.variables)}", notes=notes)

    def plot(self) -> Figure:
        from .figures import Band, Figure, Line, Panel

        v = self.volatilities()
        return Figure(tuple(Panel((Band(v[f"{n}_lower"], v[f"{n}_upper"], "68%"), Line(v[n], "median")), title=f"{n}: shock standard deviation")
                            for n in self.variables), columns=min(len(self.variables), 2))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.volatilities()

    def __repr__(self) -> str:
        return f"<FittedBVARSV {', '.join(self.variables)}: {self.spec.lags} lags, {len(self.B)} draws>"
