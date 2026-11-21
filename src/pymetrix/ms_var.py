"""Markov-switching VARs (Hamilton 1989; Krolzig 1997), by the EM algorithm.

    ms = MSVAR(["gy", "pi"], lags=1, regimes=2, switching="intercept_variance").fit(data)
    ms.summary(); ms.probabilities(smoothed=True); ms.plot(); ms.irf(regime=0, periods=20)
    ms.fevd(regime=1, periods=20); ms.historical_decomposition()   # along the most likely regime path
    ms.fevd(regime=1, draws=500, coverage=0.68)                      # bands from the estimates' asymptotic distribution

y_t = c_s + A_1,s y_t-1 + ... + A_p,s y_t-p + u_t, u_t ~ N(0, Sigma_s), s_t a Markov chain with transition matrix P
(P[i, j] = prob(s_t+1 = j | s_t = i)). ``switching`` says what depends on the regime: "intercept" (MSI), "intercept_variance"
(MSIH: the intercepts and the covariance) or "all" (MSIAH). The likelihood conditions on the first p observations, the
first regime drawn from the chain's ergodic distribution. EM: the Hamilton filter and Kim's smoother give each period's
regime probabilities and the transitions'; the M-step updates P from them and the coefficients by generalised least
squares weighted by the smoothed probabilities, then the covariances (conditional maximisation). Since the first
regime is drawn from P's ergodic distribution, a final polish (BFGS) maximises the exact likelihood over all the
parameters, so the estimates are maximum likelihood (statsmodels' Markov regression's, for one equation).

The likelihood has local maxima, and EM from regimes that start alike keeps them alike, so the starts are regime
paths read off the data (quantile splits of each variable and of its rolling mean, rolling variances of the linear
VAR's residuals, k-means, one-variable Markov-switching fits) besides ``starts - 1`` random soft splits; each is
screened by a short EM run, collapsed ones (identical regimes, an empty regime) are restarted, and the best few run to
convergence (``pymetrix.ms_var_em``, which also has the filter and smoother vectorised over time). Regimes are
ordered by the first variable's intercept.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from ._api import api
from .errors import SpecificationError
from .output import Table
from ._random import generator as make_rng
from .ms_var_bands import band_source, check_coverage, quantile_bands

if TYPE_CHECKING:
    from .data import Dataset
    from .figures import Figure

__all__ = ["FittedMSVAR", "MSVAR", "hamilton_filter"]


def _ergodic(P: np.ndarray) -> np.ndarray:
    M = len(P)
    A = np.vstack([P.T - np.eye(M), np.ones(M)])
    return np.linalg.lstsq(A, np.r_[np.zeros(M), 1.0], rcond=None)[0]


def _densities(Y: np.ndarray, means: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """(T, M) normal densities of y_t under each regime; means (M, T, K), sigmas (M, K, K)."""
    T, K = Y.shape
    out = np.empty((T, len(sigmas)))
    for j, S in enumerate(sigmas):
        L = np.linalg.cholesky(S)
        z = np.linalg.solve(L, (Y - means[j]).T)
        out[:, j] = np.exp(-0.5 * np.sum(z**2, axis=0) - np.log(np.diag(L)).sum() - K / 2 * np.log(2 * np.pi))
    return out


def hamilton_filter(dens: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(filtered probabilities, predicted probabilities, log likelihood), starting from the ergodic distribution."""
    T, M = dens.shape
    filt, pred = np.empty((T, M)), np.empty((T, M))
    xi = _ergodic(P)
    ll = 0.0
    for t in range(T):
        pred[t] = xi if t == 0 else filt[t - 1] @ P
        joint = pred[t] * dens[t]
        lik = joint.sum()
        ll += np.log(max(lik, 1e-300))
        filt[t] = joint / max(lik, 1e-300)
    return filt, pred, float(ll)


def _smooth(filt: np.ndarray, pred: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kim's smoother: (smoothed probabilities (T, M), the transitions' smoothed probabilities summed over t (M, M))."""
    T, M = filt.shape
    sm = np.empty_like(filt)
    sm[-1] = filt[-1]
    trans = np.zeros((M, M))
    for t in range(T - 2, -1, -1):
        ratio = sm[t + 1] / np.maximum(pred[t + 1], 1e-300)
        joint = filt[t][:, None] * P * ratio[None, :]  # prob(s_t = i, s_t+1 = j | all)
        sm[t] = joint.sum(axis=1)
        trans += joint
    return sm, trans


def _design(Y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    T = len(Y)
    return Y[p:], np.column_stack([Y[p - l:T - l] for l in range(1, p + 1)])


@dataclass(frozen=True, eq=False)
class MSVAR:
    """A Markov-switching VAR."""

    variables: Sequence[str]
    lags: int = 1
    regimes: int = 2
    switching: Literal["intercept", "intercept_variance", "all"] = "intercept_variance"

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", tuple(self.variables))
        if self.regimes < 2 or self.lags < 0:
            raise SpecificationError("a Markov-switching VAR needs at least two regimes and lags >= 0")
        if self.switching not in ("intercept", "intercept_variance", "all"):
            raise SpecificationError("switching is 'intercept', 'intercept_variance' or 'all'")

    @api("pymetrix:msvar.fit")
    def fit(self, data: Dataset, starts: int = 10, max_iter: int = 5000, tol: float = 1e-12, seed: int = 0, span: Any = None) -> FittedMSVAR:
        """Maximum likelihood by EM and a polish of the exact likelihood. ``starts``: the random soft splits are
        ``starts - 1`` of the starting regime paths, after those read off the data (``pymetrix.ms_var_em``);
        ``max_iter``: EM iterations from each start; ``tol``: EM stops when the log likelihood changes by less than
        this, relative (no tighter than 1e-8: the polish finishes); ``seed``: the random splits and k-means."""
        if data.kind != "time_series":
            raise SpecificationError("a Markov-switching VAR needs a single time series")
        missing = [v for v in self.variables if v not in data]
        if missing:
            raise SpecificationError(f"{', '.join(missing)} are not series of the data")
        frame = data.frame[list(self.variables)].astype(float)
        if span is not None:
            frame = frame[data.sample_mask(span)]
        frame = frame.dropna()
        Yall = frame.to_numpy()
        Y, Xl = _design(Yall, self.lags) if self.lags else (Yall, np.zeros((len(Yall), 0)))
        T, K = Y.shape
        M = self.regimes
        from .ms_var_em import screen_and_fit

        best = screen_and_fit(self, Y, Xl, starts, max_iter, tol, make_rng(seed))
        if best is None:
            raise SpecificationError("EM failed from every start: try fewer regimes or switching parameters")
        order = np.argsort([c[0] for c in best["C"]])  # regimes by the first variable's intercept
        best = {**best, "C": best["C"][order], "B": best["B"][order], "S": best["S"][order], "P": best["P"][np.ix_(order, order)],
                "filt": best["filt"][:, order], "smooth": best["smooth"][:, order]}
        return FittedMSVAR(self, best["C"], best["B"], best["S"], best["P"], best["ll"], best["iterations"],
                           pd.DataFrame(best["filt"], index=frame.index[self.lags:], columns=[f"regime {j}" for j in range(M)]),
                           pd.DataFrame(best["smooth"], index=frame.index[self.lags:], columns=[f"regime {j}" for j in range(M)]),
                           T, frame)

    def _mstep(self, Y: np.ndarray, Xl: np.ndarray, w: np.ndarray, S: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        T, K = Y.shape
        M = w.shape[1]
        kx = Xl.shape[1]
        if self.switching == "all":
            C, B, Sn = np.empty((M, K)), np.empty((M, K, kx)), np.empty((M, K, K))
            Z = np.column_stack([np.ones(T), Xl])
            for j in range(M):
                sw = np.sqrt(w[:, j])
                theta = np.linalg.lstsq(Z * sw[:, None], Y * sw[:, None], rcond=None)[0]
                C[j], B[j] = theta[0], theta[1:].T
                U = Y - Z @ theta
                Sn[j] = (U * w[:, j][:, None]).T @ U / w[:, j].sum()
            return C, B, Sn
        # intercepts switch, lags common: GLS over the stacked (t, regime) rows
        Sinv = np.linalg.inv(S) if S is not None else np.tile(np.eye(K), (M, 1, 1))
        q = M + kx
        lhs, rhs = np.zeros((K * q, K * q)), np.zeros(K * q)
        for j in range(M):
            Z = np.column_stack([np.tile(np.eye(M)[j], (T, 1)), Xl])  # (T, q)
            Zw = Z * w[:, j][:, None]
            lhs += np.kron(Sinv[j], Zw.T @ Z)
            rhs += (Zw.T @ Y @ Sinv[j]).T.reshape(-1)
        theta = np.linalg.solve(lhs, rhs).reshape(K, q).T  # (q, K)
        C = theta[:M]
        B = np.tile(theta[M:].T, (M, 1, 1))
        Sn = np.empty((M, K, K))
        resid = [Y - C[j] - Xl @ theta[M:] for j in range(M)]
        if self.switching == "intercept_variance":
            for j in range(M):
                Sn[j] = (resid[j] * w[:, j][:, None]).T @ resid[j] / w[:, j].sum()
        else:
            pooled = sum((resid[j] * w[:, j][:, None]).T @ resid[j] for j in range(M)) / T
            Sn[:] = pooled
        return C, B, Sn

    def _em(self, Y: np.ndarray, Xl: np.ndarray, w: np.ndarray, max_iter: int, tol: float) -> dict[str, Any] | None:
        """EM from the smoothed weights ``w`` (T, M), then the polish on the exact likelihood (``ms_var_em``)."""
        from .ms_var_em import EM_TOL, em, polish

        r = em(self, Y, Xl, w, max_iter, max(tol, EM_TOL))
        return None if r is None else polish(self, r, Y, Xl)


@dataclass(frozen=True, eq=False)
class FittedMSVAR:
    """An estimated Markov-switching VAR."""

    spec: MSVAR
    intercepts: np.ndarray  # (M, K)
    coefs: np.ndarray = field(repr=False)  # (M, K, K p): lags 1..p side by side
    sigmas: np.ndarray = field(repr=False)  # (M, K, K)
    transition: np.ndarray = field(repr=False)  # P[i, j] = prob(next j | now i)
    loglike: float = 0.0
    iterations: int = 0
    filtered: pd.DataFrame = field(default=None, repr=False)
    smoothed: pd.DataFrame = field(default=None, repr=False)
    nobs: int = 0
    sample_rows: pd.DataFrame | None = field(default=None, repr=False)  # the data fitted, with the first lags observations (not `observations`: that is the tidy-table verb)

    stage = "estimate"

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self.spec.variables)

    @property
    def durations(self) -> np.ndarray:
        """Each regime's expected duration, 1 / (1 - P[j, j])."""
        return 1 / np.maximum(1 - np.diag(self.transition), 1e-12)

    def probabilities(self, smoothed: bool = True) -> pd.DataFrame:
        return self.smoothed if smoothed else self.filtered

    @api()
    def irf(self, regime: int = 0, periods: int = 20, draws: int = 0, coverage: float = 0.68, seed: int = 0) -> Any:
        """Responses within a regime (as if it lasted), recursively identified with that regime's covariance. With
        ``draws``, ``coverage`` bands: pointwise quantiles over draws of the parameters from the estimates' asymptotic
        normal distribution (``parameter_draws``)."""
        from .var import IRF, _ma

        regime = self._regime(regime)
        values = _ma(self._lag_matrices(regime), periods) @ np.linalg.cholesky(self.sigmas[regime])
        label = f"Cholesky, regime {regime}"
        if not draws:
            return IRF(values, self.variables, self.variables, None, None, False, label)
        sims = np.stack([_ma(A, periods) @ np.linalg.cholesky(S) for A, S in self._regime_draws(regime, draws, coverage, seed)])
        lo, hi = quantile_bands(sims, coverage)
        return IRF(values, self.variables, self.variables, lo, hi, False, f"{label}; bands from {band_source(self, draws)}", coverage)

    def parameter_covariance(self) -> np.ndarray:
        """The estimates' asymptotic covariance, on the scale ``parameter_draws`` draws on (intercepts; lag
        coefficients; each covariance's Cholesky factor, log diagonal; each transition row's logits against its last
        entry): the inverse of minus the Hessian of the exact log likelihood. Computed once."""
        from .ms_var_bands import parameter_covariance

        if "_covariance" not in self.__dict__:
            self.__dict__["_covariance"] = parameter_covariance(self)
        return self.__dict__["_covariance"]

    def parameter_draws(self, draws: int = 500, seed: int = 0) -> dict[str, np.ndarray]:
        """Draws of the parameters from the estimates' asymptotic normal distribution, on a scale where every draw is
        a valid set of covariance and transition matrices (see ``pymetrix.ms_var_bands``): ``intercepts`` (N, M, K),
        ``coefs`` (N, M, K, K p), ``sigmas`` (N, M, K, K) and ``transition`` (N, M, M)."""
        from .ms_var_bands import parameter_draws

        return parameter_draws(self, draws, seed)

    def _regime_draws(self, regime: int, draws: int, coverage: float, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Each draw's lag matrices (p, K, K) and covariance in ``regime``."""
        check_coverage(coverage)
        d = self.parameter_draws(draws, seed)
        return [(self._lag_matrices(regime, coefs), S) for coefs, S in zip(d["coefs"][:, regime], d["sigmas"][:, regime])]

    def _regime(self, regime: int) -> int:
        if not 0 <= int(regime) < self.spec.regimes:
            raise SpecificationError(f"regime is 0 to {self.spec.regimes - 1}; got {regime}")
        return int(regime)

    def _lag_matrices(self, regime: int, coefs: np.ndarray | None = None) -> np.ndarray:
        K, p = len(self.variables), self.spec.lags
        coefs = self.coefs[regime] if coefs is None else coefs
        return np.stack([coefs[:, K * l:K * (l + 1)] for l in range(p)]) if p else np.zeros((1, K, K))

    @api()
    def fevd(self, regime: int = 0, periods: int = 20, draws: int = 0, coverage: float = 0.68, seed: int = 0) -> Any:
        """Forecast-error variance decomposition within a regime (held fixed over the horizon, as ``irf``), the shocks
        recursively identified with that regime's covariance. With ``draws``, ``coverage`` bands: pointwise quantiles
        of the shares over draws of the parameters from the estimates' asymptotic normal distribution (each draw's
        shares add up to one; the point shares are those at the estimates)."""
        from .var import FEVD, _ma
        from .var_decompositions import fevd_shares

        regime = self._regime(regime)
        theta = _ma(self._lag_matrices(regime), periods - 1) @ np.linalg.cholesky(self.sigmas[regime])
        label = f"Cholesky, regime {regime} held fixed"
        if not draws:
            return FEVD(fevd_shares(theta), self.variables, self.variables, label)
        shares = np.stack([fevd_shares(_ma(A, periods - 1) @ np.linalg.cholesky(S)) for A, S in self._regime_draws(regime, draws, coverage, seed)])
        lo, hi = quantile_bands(shares, coverage)
        return FEVD(fevd_shares(theta), self.variables, self.variables, label, lo, hi, coverage, band_source(self, draws))

    def regime_path(self) -> pd.Series:
        """The most likely regime each period, by the smoothed probabilities."""
        return pd.Series(np.argmax(self.smoothed.to_numpy(), axis=1), index=self.smoothed.index, name="regime")

    @api()
    def historical_decomposition(self, draws: int = 0, coverage: float = 0.68, seed: int = 0) -> Any:
        """Each variable over the sample, split into the contributions of the recursively identified shocks and the
        baseline, along the most likely regime path: in each period the regime with the highest smoothed probability
        (``regime_path()``) supplies the intercepts, lag matrices and impact matrix (the Cholesky factor of its
        covariance). The residuals u_t = y_t - c_s(t) - sum_l A_l,s(t) y_t-l give the shocks e_t = P_s(t)^-1 u_t; each
        shock's contribution follows x_j,t = sum_l A_l,s(t) x_j,t-l + P_s(t)[:, j] e_j,t from zero, and the baseline
        b_t = c_s(t) + sum_l A_l,s(t) b_t-l from the first observations, so the regime's intercepts (and so any shift
        of the mean between regimes) are in the baseline. The parts add up to the data exactly. (Weighting the
        regimes' parameters by their probabilities would also add up, but the weighted system is no regime of the
        model; the most likely path is a history the model can produce.)

        With ``draws``, ``coverage`` bands for each contribution: pointwise quantiles over the decompositions under
        draws of the parameters from the estimates' asymptotic normal distribution, the regime path held at the most
        likely one (so the bands leave out the uncertainty about the regimes). Every draw's parts add up to the data;
        the point decomposition is the one at the estimates."""
        from .var_decompositions import decomposition, tv_contributions

        if self.sample_rows is None:
            raise SpecificationError("this Markov-switching VAR was fitted before historical decompositions: fit it again")
        K, p = len(self.variables), self.spec.lags
        Yall = self.sample_rows.to_numpy(dtype=float)
        Y = Yall[p:]
        T = len(Y)
        s = self.regime_path().to_numpy()
        if draws:
            check_coverage(coverage)

        def parts(intercepts: np.ndarray, coefs: np.ndarray, sigmas: np.ndarray) -> tuple[np.ndarray, ...]:
            lagm = np.stack([self._lag_matrices(j, coefs[j]) if p else np.zeros((0, K, K)) for j in range(self.spec.regimes)])[s]
            const = intercepts[s]
            fitted = const + sum(np.einsum("tij,tj->ti", lagm[:, l], Yall[p - 1 - l:p - 1 - l + T]) for l in range(p))
            return lagm, np.linalg.cholesky(sigmas)[s], Y - fitted, const

        lagm, impact, U, const = parts(self.intercepts, self.coefs, self.sigmas)
        contrib, base, e = tv_contributions(lagm[None], impact[None], U[None], const[None], Yall[None, :p])
        switches = int((np.diff(s) != 0).sum())
        result = decomposition(contrib, base, e, Y, self.smoothed.index, self.variables,
                               f"Cholesky, most likely regime each period ({switches} switches)",
                               "relative to the baseline: the regimes' intercepts along the most likely regime path and initial conditions")
        if not draws:
            return result
        d = self.parameter_draws(draws, seed)
        stacked = [np.stack(x) for x in zip(*(parts(*z) for z in zip(d["intercepts"], d["coefs"], d["sigmas"])))]
        sims, _, _ = tv_contributions(*stacked, np.broadcast_to(Yall[None, :p], (len(stacked[0]), p, K)))
        lo, med, hi = np.quantile(sims, [(1 - coverage) / 2, 0.5, (1 + coverage) / 2], axis=0)
        names = list(self.variables)

        def frames(x: np.ndarray) -> dict[str, pd.DataFrame]:
            return {v: pd.DataFrame(x[:, i, :], index=self.smoothed.index, columns=names) for i, v in enumerate(names)}

        return replace(result, lower=frames(lo), median=frames(med), upper=frames(hi), coverage=coverage,
                       band_source=f"{band_source(self, draws)}, the regime path held at the most likely one")

    def summary(self) -> Table:
        K, p, M = len(self.variables), self.spec.lags, self.spec.regimes
        rows = []
        for j in range(M):
            rows += [[f"regime {j}: const {v}", float(self.intercepts[j, i])] for i, v in enumerate(self.variables)]
            if self.spec.switching == "all" or j == 0:
                label = f"regime {j}" if self.spec.switching == "all" else "all regimes"
                rows += [[f"{label}: {v} <- {w}[-{l + 1}]", float(self.coefs[j, i, K * l + m])]
                         for l in range(p) for i, v in enumerate(self.variables) for m, w in enumerate(self.variables)]
            rows += [[f"regime {j}: sd {v}", float(np.sqrt(self.sigmas[j, i, i]))] for i, v in enumerate(self.variables)]
        rows += [[f"P[{i} -> {j}]", float(self.transition[i, j])] for i in range(M) for j in range(M)]
        notes = (f"Switching: {self.spec.switching.replace('_', ' and ')}", f"Observations: {self.nobs}",
                 f"Log likelihood: {self.loglike:.4f}", f"EM iterations: {self.iterations}",
                 "Expected durations: " + ", ".join(f"regime {j} {d:.3g} periods" for j, d in enumerate(self.durations)))
        return Table(["", "estimate"], rows, title=f"Markov-switching VAR({p}), {M} regimes: {', '.join(self.variables)}", notes=notes)

    def plot(self) -> Figure:
        from .figures import Figure, Line, Panel

        sm = self.smoothed
        return Figure((Panel(tuple(Line(sm[c], c) for c in sm.columns), title="Smoothed regime probabilities"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.smoothed

    def __repr__(self) -> str:
        return f"<FittedMSVAR {self.spec.regimes} regimes, {', '.join(self.variables)}: log likelihood {self.loglike:.4f}>"
