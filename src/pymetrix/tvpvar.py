"""Time-varying-parameter VARs with stochastic volatility (Primiceri 2005; Del Negro and Primiceri 2015).

    tvp = TVPVAR(["infl", "unemp", "rate"], lags=2).fit(data, draws=2000, burn=1000)
    tvp.volatilities(); tvp.irf("1975Q1", 20); tvp.irf("2000Q1", 20); tvp.plot()

y_t = X_t beta_t + u_t, u_t ~ N(0, A_t^-1 Sigma_t Sigma_t' A_t^-1'), with X_t = I_K (x) (1, y_(t-1)', ..., y_(t-p)'), A_t
lower triangular with ones on its diagonal, and Sigma_t diagonal. The coefficients beta_t, the free elements of A_t
(equation by equation) and the log standard deviations log sigma_t follow random walks with innovation covariances Q,
S (block diagonal by equation) and W (diagonal here). The model is estimated by Gibbs sampling, in Del Negro and
Primiceri's corrected order: the volatilities given the mixture indicators, the coefficients and A, then the
indicators given them. Each path is drawn by forward filtering and backward sampling (Carter and Kohn 1994); the
volatilities through Kim, Shephard and Chib's (1998) seven-normal mixture for log chi-squared(1).

Priors follow Primiceri, from least squares on a ``training`` sample before the estimation sample: beta_0 ~
N(b_OLS, 4 V_OLS), the free elements of A_0 ~ N(a_OLS, 4 |a_OLS| + 0.01 per element), log sigma_0 ~ N(log s_OLS, 1);
Q ~ IW(kq^2 tau V_OLS, tau), S_i ~ IW(ks^2 (i + 1) diag, i + 1), W_i ~ IG with scale kw^2 4 and 4 degrees of freedom.
The impulse responses at a date hold that date's coefficients and covariance fixed, identified recursively by A_t
(the VAR's Cholesky order), for every kept draw.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from ._api import api
from ._random import generator as make_rng
from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset
    from .figures import Figure

__all__ = ["FittedTVPVAR", "TVPVAR", "ffbs", "KSC_WEIGHTS", "KSC_MEANS", "KSC_VARIANCES"]

# Kim, Shephard and Chib (1998), Table 4: log chi-squared(1) as a mixture of seven normals (means already shifted by
# -1.2704, the mean of log chi-squared(1))
KSC_WEIGHTS = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750])
KSC_MEANS = np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819]) - 1.2704
KSC_VARIANCES = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])


def ffbs(y: np.ndarray, Z: np.ndarray, H: np.ndarray, Q: np.ndarray, a0: np.ndarray, P0: np.ndarray, rng: np.random.Generator,
         c: np.ndarray | None = None) -> np.ndarray:
    """A draw of the states of y_t = c_t + Z_t s_t + e_t, e_t ~ N(0, H_t), s_t = s_(t-1) + v_t, v_t ~ N(0, Q), s_0 ~ N(a0, P0)
    (forward filtering, backward sampling). y (T, n), Z (T, n, k), H (T, n, n); returns (T, k)."""
    T, k = len(y), len(a0)
    a_f, P_f = np.empty((T, k)), np.empty((T, k, k))
    a, P = a0.astype(float), P0.astype(float)
    for t in range(T):
        P = P + Q  # predicted
        Zt = Z[t]
        F = Zt @ P @ Zt.T + H[t]
        Kg = np.linalg.solve(F, Zt @ P).T
        resid = y[t] - Zt @ a - (0.0 if c is None else c[t])
        a = a + Kg @ resid
        P = P - Kg @ Zt @ P
        P = (P + P.T) / 2
        a_f[t], P_f[t] = a, P
    out = np.empty((T, k))
    out[-1] = rng.multivariate_normal(a_f[-1], P_f[-1], method="cholesky") if k > 1 else a_f[-1] + np.sqrt(max(P_f[-1][0, 0], 0)) * rng.standard_normal(1)
    for t in range(T - 2, -1, -1):
        Pp = P_f[t] + Q
        G = np.linalg.solve(Pp, P_f[t]).T  # P_f[t] Pp^-1 (both symmetric)
        mean = a_f[t] + G @ (out[t + 1] - a_f[t])
        cov = P_f[t] - G @ P_f[t]
        cov = (cov + cov.T) / 2 + 1e-12 * np.eye(k)
        out[t] = mean + np.linalg.cholesky(cov) @ rng.standard_normal(k)
    return out


def _iw(scale: np.ndarray, dof: float, rng: np.random.Generator) -> np.ndarray:
    return np.atleast_2d(sps.invwishart.rvs(df=dof, scale=scale, random_state=rng))


@dataclass(frozen=True)
class TVPVAR:
    """A time-varying-parameter VAR with stochastic volatility, estimated by Gibbs sampling (Primiceri 2005)."""

    code_positional: ClassVar[int] = 1  # generated code gives the variables by position: TVPVAR(['a', 'b'])

    variables: Sequence[str]
    lags: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", tuple(self.variables))
        if len(self.variables) < 1 or self.lags < 1:
            raise SpecificationError("a TVP-VAR needs one or more variables and at least one lag")

    @api("pymetrix:tvp_var.fit")
    def fit(self, data: Dataset, draws: int = 2000, burn: int = 1000, training: int = 40, kq: float = 0.01, ks: float = 0.1,
            kw: float = 0.01, thin: int = 1, seed: int = 0) -> FittedTVPVAR:
        """Gibbs sampling: ``burn`` draws discarded, then ``draws`` kept (every ``thin``-th); priors from least squares on
        the first ``training`` periods (see the module notes)."""
        from .progress import report

        if data.kind != "time_series":
            raise SpecificationError("a TVP-VAR needs a time series")
        missing = [v for v in self.variables if v not in data.names]
        if missing:
            raise SpecificationError(f"no series {', '.join(map(repr, missing))} in the dataset")
        frame = data.frame[list(self.variables)].dropna()
        K, p = len(self.variables), self.lags
        Yall = frame.to_numpy(float)
        rows = np.arange(p, len(Yall))
        Xall = np.column_stack([np.ones(len(rows))] + [Yall[rows - l] for l in range(1, p + 1)])
        Yall = Yall[rows]
        index = frame.index[rows]
        m = Xall.shape[1]
        if training < m + 5 or training >= len(Yall) - 20:
            raise SpecificationError(f"the training sample ({training} periods) must exceed the regressors ({m}) and leave a sample to estimate")
        # priors from least squares on the training sample
        Xtr, Ytr = Xall[:training], Yall[:training]
        Btr = np.linalg.lstsq(Xtr, Ytr, rcond=None)[0]
        Utr = Ytr - Xtr @ Btr
        Str = Utr.T @ Utr / (training - m)
        b0 = Btr.T.reshape(-1)  # equation by equation
        V0 = np.kron(Str, np.linalg.inv(Xtr.T @ Xtr))
        C = np.linalg.cholesky(Str)
        Dc = np.diag(np.diag(C))
        Ainv0 = C @ np.linalg.inv(Dc)  # unit lower triangular: Str = Ainv0 D^2 Ainv0'
        A0 = np.linalg.inv(Ainv0)
        a0_blocks = [A0[i, :i] for i in range(1, K)]
        h0 = np.log(np.diag(Dc))
        Y, X = Yall[training:], Xall[training:]
        index = index[training:]
        T = len(Y)
        nb = K * m
        Zb = np.zeros((T, K, nb))
        for i in range(K):
            Zb[:, i, i * m:(i + 1) * m] = X
        rng = make_rng(seed)
        # initial states: constant at the training values
        beta = np.tile(b0, (T, 1))
        a_blocks = [np.tile(a, (T, 1)) for a in a0_blocks]
        h = np.tile(h0, (T, 1))
        Q = kq**2 * training * V0
        S = [ks**2 * (i + 1) * np.eye(i) for i in range(1, K)]
        Wd = np.full(K, kw**2)
        mix = np.full((T, K), 4)
        total = burn + draws * thin
        keep_beta, keep_A, keep_h = [], [], []
        cbar = 1e-3
        for it in range(total):
            if it % 25 == 0:
                report(it / total, f"Gibbs draw {it + 1} of {total}")
            # (1) the log volatilities, given the indicators (Del Negro and Primiceri's order)
            Ahat = self._A_paths(a_blocks, T, K)
            resid = Y - np.einsum("tij,tj->ti", Zb, beta)
            ystar = np.einsum("tij,tj->ti", Ahat, resid)
            for i in range(K):
                lz = np.log(ystar[:, i] ** 2 + cbar)
                s = mix[:, i]
                draw = ffbs(lz[:, None], np.full((T, 1, 1), 2.0), KSC_VARIANCES[s][:, None, None], np.array([[Wd[i]]]),
                            np.array([h0[i]]), np.array([[1.0]]), rng, c=KSC_MEANS[s][:, None])
                h[:, i] = draw[:, 0]
            # (2) the coefficients, given A and the volatilities
            Ainv = np.linalg.inv(Ahat)
            Hs = np.einsum("tij,tj,tkj->tik", Ainv, np.exp(2 * h), Ainv)
            beta = ffbs(Y, Zb, Hs, Q, b0, 4 * V0, rng)
            # (3) the contemporaneous relations, equation by equation
            resid = Y - np.einsum("tij,tj->ti", Zb, beta)
            for i in range(1, K):
                Zi = -resid[:, :i][:, None, :]
                Hi = np.exp(2 * h[:, i])[:, None, None]
                a_blocks[i - 1] = ffbs(resid[:, i][:, None], Zi, Hi, S[i - 1], a0_blocks[i - 1], np.diag(4 * np.abs(a0_blocks[i - 1]) + 0.01), rng)
            # (4) the mixture indicators, given everything
            Ahat = self._A_paths(a_blocks, T, K)
            ystar = np.einsum("tij,tj->ti", Ahat, resid)
            for i in range(K):
                lz = np.log(ystar[:, i] ** 2 + cbar)
                dens = KSC_WEIGHTS * sps.norm.pdf(lz[:, None], 2 * h[:, i][:, None] + KSC_MEANS, np.sqrt(KSC_VARIANCES))
                cum = np.cumsum(dens / dens.sum(axis=1, keepdims=True), axis=1)
                mix[:, i] = (rng.random(T)[:, None] > cum).sum(axis=1)
            # (5) the innovation covariances
            db = np.diff(np.vstack([b0, beta]), axis=0)
            Q = _iw(kq**2 * training * V0 + db.T @ db, training + T, rng)
            for i in range(1, K):
                da = np.diff(np.vstack([a0_blocks[i - 1], a_blocks[i - 1]]), axis=0)
                S[i - 1] = _iw(ks**2 * (i + 1) * np.eye(i) + da.T @ da, i + 1 + T, rng)
            dh = np.diff(np.vstack([h0, h]), axis=0)
            Wd = 1 / rng.gamma((4 + T) / 2, 1 / ((kw**2 * 4 + (dh**2).sum(axis=0)) / 2))
            if it >= burn and (it - burn) % thin == 0:
                keep_beta.append(beta.copy())
                keep_A.append(np.hstack(a_blocks) if K > 1 else np.zeros((T, 0)))
                keep_h.append(h.copy())
        return FittedTVPVAR(self, index, np.array(keep_beta), np.array(keep_A), np.array(keep_h), int(training), m,
                            {"kq": kq, "ks": ks, "kw": kw, "burn": burn, "thin": thin})

    @staticmethod
    def _A_paths(a_blocks: list[np.ndarray], T: int, K: int) -> np.ndarray:
        A = np.tile(np.eye(K), (T, 1, 1))
        for i in range(1, K):
            A[:, i, :i] = a_blocks[i - 1]
        return A


@dataclass(frozen=True, eq=False)
class FittedTVPVAR:
    """Posterior draws of a TVP-VAR-SV's paths: coefficients, contemporaneous relations and volatilities."""

    spec: TVPVAR
    index: pd.Index
    beta: np.ndarray = field(repr=False)  # (draws, T, K m)
    a: np.ndarray = field(repr=False)  # (draws, T, K (K - 1) / 2)
    h: np.ndarray = field(repr=False)  # (draws, T, K): log standard deviations
    training: int = 40
    m: int = 1
    settings: dict[str, Any] = field(default_factory=dict)

    stage = "estimate"

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self.spec.variables)

    def volatilities(self, coverage: float = 0.68) -> pd.DataFrame:
        """Each shock's standard deviation over time: posterior median and bands."""
        s = np.exp(self.h)
        lo, med, hi = np.quantile(s, [(1 - coverage) / 2, 0.5, (1 + coverage) / 2], axis=0)
        cols = {}
        for i, v in enumerate(self.variables):
            cols[f"{v}"], cols[f"{v} lower"], cols[f"{v} upper"] = med[:, i], lo[:, i], hi[:, i]
        return pd.DataFrame(cols, index=self.index)

    def coefficient_paths(self, coverage: float = 0.68) -> pd.DataFrame:
        """Each coefficient's posterior median over time (columns: equation, regressor)."""
        names = ["const"] + [f"{v}[-{l}]" for l in range(1, self.spec.lags + 1) for v in self.variables]
        med = np.median(self.beta, axis=0)
        cols = pd.MultiIndex.from_tuples([(eq, r) for eq in self.variables for r in names], names=["equation", "regressor"])
        return pd.DataFrame(med, index=self.index, columns=cols)

    def _at(self, period: Any) -> int:
        if period is None:
            return len(self.index) - 1
        key = pd.Period(str(period), self.index.freq)
        if key not in self.index:
            raise SpecificationError(f"{period} is not in the estimation sample ({self.index[0]} to {self.index[-1]})")
        return int(self.index.get_loc(key))

    def irf(self, period: Any = None, periods: int = 20, coverage: float = 0.68) -> Any:
        """Responses to one-standard-deviation recursive shocks at ``period`` (default: the last), from every draw's
        coefficients and covariance at that date: the posterior median, with bands."""
        from .var import IRF, _lag_matrices, _ma

        t = self._at(period)
        K, p, m = len(self.variables), self.spec.lags, self.m
        sims = []
        for d in range(len(self.beta)):
            B = self.beta[d, t].reshape(K, m).T  # (m, K): regressors by equation
            A = _lag_matrices(B, 1, K, p)
            Am = np.eye(K)
            Am[np.tril_indices(K, -1)] = self.a[d, t]
            impact = np.linalg.inv(Am) @ np.diag(np.exp(self.h[d, t]))
            sims.append(_ma(A, periods) @ impact)
        sims = np.stack(sims)
        lo, med, hi = np.quantile(sims, [(1 - coverage) / 2, 0.5, (1 + coverage) / 2], axis=0)
        return IRF(med, self.variables, self.variables, lo, hi, False, f"recursive, at {self.index[t]}", coverage)

    def irf_paths(self, periods: int = 20, draws: int = 200) -> np.ndarray:
        """The posterior median responses at every date: (dates, horizons 0 to ``periods``, responses, shocks), as
        ``irf`` gives them at one date, from ``draws`` of the kept draws evenly spaced (all of them if fewer)."""
        if periods < 1 or draws < 1:
            raise SpecificationError("periods and draws are whole numbers of at least 1")
        K, p, m = len(self.variables), self.spec.lags, self.m
        pick = np.unique(np.linspace(0, len(self.beta) - 1, min(len(self.beta), int(draws))).round().astype(int))
        n = len(pick)
        lower = np.tril_indices(K, -1)
        out = np.empty((len(self.index), periods + 1, K, K))
        for t in range(len(self.index)):
            B = self.beta[pick, t].reshape(n, K, m)  # by draw: equations by regressors (the constant first)
            A = np.stack([B[:, :, 1 + l * K: 1 + (l + 1) * K] for l in range(p)], axis=1)  # (draws, lag, K, K)
            Am = np.tile(np.eye(K), (n, 1, 1))
            Am[:, lower[0], lower[1]] = self.a[pick, t]
            impact = np.linalg.inv(Am) * np.exp(self.h[pick, t])[:, None, :]  # A_t^-1 Sigma_t
            phi = np.zeros((n, periods + 1, K, K))
            phi[:, 0] = np.eye(K)
            for i in range(1, periods + 1):
                for j in range(1, min(i, p) + 1):
                    phi[:, i] += phi[:, i - j] @ A[:, j - 1]
            out[t] = np.median(phi @ impact[:, None], axis=0)
        return out

    def irf_over_time(self, response: str, shock: str, periods: int = 20, draws: int = 200) -> pd.DataFrame:
        """How ``response`` answers ``shock`` at each date: posterior medians, the dates down the rows and the
        horizons (0 to ``periods``) across (see ``irf_paths``)."""
        for name in (response, shock):
            if name not in self.variables:
                raise SpecificationError(f"no variable {name!r} in this TVP-VAR: {', '.join(self.variables)}")
        paths = self.irf_paths(periods, draws)
        i, k = self.variables.index(response), self.variables.index(shock)
        return pd.DataFrame(paths[:, :, i, k], index=self.index.rename("date"), columns=pd.RangeIndex(periods + 1, name="horizon"))

    def irf_surface(self, periods: int = 20, draws: int = 200, responses: Sequence[str] | None = None,
                    shocks: Sequence[str] | None = None) -> Figure:
        """The responses as they changed over time: a surface for each response and shock, horizons along x, dates
        along y, the posterior median response as height (``irf_over_time``)."""
        from .figures import Figure, Panel, Surface

        responses, shocks = list(responses or self.variables), list(shocks or self.variables)
        missing = [n for n in responses + shocks if n not in self.variables]
        if missing:
            raise SpecificationError(f"no variable {', '.join(map(repr, missing))} in this TVP-VAR: {', '.join(self.variables)}")
        paths = self.irf_paths(periods, draws)
        index, horizons = self.index.rename("date"), pd.RangeIndex(periods + 1, name="horizon")
        panels = []
        for r in responses:
            for s in shocks:
                i, k = self.variables.index(r), self.variables.index(s)
                values = pd.DataFrame(paths[:, :, i, k], index=index, columns=horizons)
                panels.append(Panel((Surface(values, label="response"),), title=f"{r}, shock to {s}", legend=False))
        return Figure(tuple(panels), title="Responses over time (posterior medians, recursive identification)", columns=len(shocks),
                      width=max(7.5, 3.6 * len(shocks)))

    def summary(self) -> Table:
        vol = np.exp(self.h)
        rows = []
        for i, v in enumerate(self.variables):
            med = np.median(vol[:, :, i], axis=0)
            rows.append([float(med[0]), float(med.min()), float(med.max()), float(med[-1])])
        return Table(["first", "lowest", "highest", "last"], rows, index=list(self.variables), index_name="shock s.d.",
                     title=f"TVP-VAR with stochastic volatility: {', '.join(self.variables)}",
                     notes=(f"Posterior medians of each shock's standard deviation over {self.index[0]} to {self.index[-1]}; "
                            f"{len(self.beta)} kept Gibbs draws after {self.settings.get('burn', 0)}; priors from {self.training} "
                            f"training periods (kq {self.settings.get('kq')}, ks {self.settings.get('ks')}, kw {self.settings.get('kw')}).",
                            "Responses at a date: irf(period); over time: irf_over_time(response, shock), irf_surface(); "
                            "paths: volatilities(), coefficient_paths()."))

    def plot(self) -> Figure:
        from .figures import Band, Figure, Line, Panel

        v = self.volatilities()
        return Figure(tuple(Panel((Band(v[f"{n} lower"], v[f"{n} upper"], "68%"), Line(v[n], "median")), title=f"{n}: shock standard deviation")
                            for n in self.variables))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.volatilities()

    def __repr__(self) -> str:
        return f"<FittedTVPVAR of {', '.join(self.variables)}: {len(self.beta)} draws>"
