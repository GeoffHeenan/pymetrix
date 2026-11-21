"""Panel VARs: one VAR for many entities, with fixed effects removed (Holtz-Eakin, Newey and Rosen 1988; Love and
Zicchino 2006; Abrigo and Love 2016).

    pvar = VAR(["inv", "sales", "cash"], lags=2, by="firm").fit(panel)          # GMM on forward orthogonal deviations
    pvar.summary(); pvar.granger(); pvar.irf(10).plot(); pvar.fevd(10)
    VAR([...], lags=2, by="firm").fit(panel, transform="within")                 # least squares on demeaned data

The model is y_it = A_1 y_i,t-1 + ... + A_p y_i,t-p + mu_i + e_it, with the same A's for every entity i and a fixed
effect mu_i. Removing mu_i by demeaning makes the lags correlated with the error when T is short (Nickell 1981), so
by default each variable, and each of its lags, is transformed by **forward orthogonal deviations** (Arellano and Bover
1995): x*_t = sqrt((T_i - t) / (T_i - t + 1)) (x_t - the mean of x over the entity's later periods), which keeps the
errors uncorrelated, and the transformed lags are instrumented by the lags' untransformed levels, which the future
means do not touch. With ``instrument_lags`` = p (the default) the instruments just identify the coefficients;
with more, the two-step efficient GMM estimate is used and Hansen's J tests the over-identifying restrictions.
``transform="within"`` is least squares on entity-demeaned data, consistent only as T grows.

Standard errors are robust and clustered by entity. Impulse responses and variance decompositions follow the VAR's
(Cholesky by default; ``identify`` as for a VAR); their bands are quantiles of responses computed from draws of the
coefficients from their estimated distribution, as Love and Zicchino do.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from ._api import api
from ._random import generator as make_rng
from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset
    from .var import FEVD, IRF, VAR


def forward_orthogonal(x: np.ndarray) -> np.ndarray:
    """Forward orthogonal deviations of each column over the rows (one entity's periods, in order); the last row,
    which has no later periods, is dropped."""
    x = np.asarray(x, float)
    T = len(x)
    out = np.empty((T - 1, x.shape[1]))
    later_sum = np.cumsum(x[::-1], axis=0)[::-1]  # sum of rows t..T-1
    for t in range(T - 1):
        n = T - t - 1  # later periods
        out[t] = np.sqrt(n / (n + 1)) * (x[t] - later_sum[t + 1] / n)
    return out


def _entity_blocks(spec: VAR, data: Dataset, p: int, q: int, transform: str, span: Any) -> tuple[list[tuple[Any, np.ndarray, np.ndarray, np.ndarray]], pd.DataFrame]:
    """For each entity: the transformed y (rows, K), the transformed lags (rows, K p) and the instruments (rows, K q);
    and the rows used, in levels (the variables and their lags)."""
    entity = spec.by
    names = [a.name for a in data.axes]
    if data.kind != "panel" or entity not in names:
        raise SpecificationError(f"a panel VAR needs a panel dataset with an entity axis {entity!r}; this one has {', '.join(names)}")
    time = data.time_axis.name
    K = len(spec.variables)
    depth = max(p, q)
    cols = {v: data[v].astype(float) for v in spec.variables}
    for lag in range(1, depth + 1):
        for v in spec.variables:
            cols[f"{v}[-{lag}]"] = data.lagged(v, -lag).astype(float)
    frame = pd.DataFrame(cols)[data.sample_mask(span)]
    frame = frame[np.isfinite(frame.to_numpy(float)).all(axis=1)]
    if frame.empty:
        raise SpecificationError("no complete observations for the panel VAR")
    lag_cols = [f"{v}[-{lag}]" for lag in range(1, p + 1) for v in spec.variables]
    inst_cols = [f"{v}[-{lag}]" for lag in range(1, q + 1) for v in spec.variables]
    # each entity's rows in time order, by numpy (the frame is split once, not by pandas entity by entity)
    codes, keys = pd.factorize(frame.index.get_level_values(entity), sort=True)
    keys = keys.tolist()  # Python scalars, as pandas' groupby gives
    ordinals = pd.PeriodIndex(frame.index.get_level_values(time)).asi8
    order = np.lexsort((ordinals, codes))
    codes, ordinals = codes[order], ordinals[order]
    position = {c: i for i, c in enumerate(frame.columns)}
    values = frame.to_numpy(float)[order]
    Y_all, X_all, Z_all = (values[:, [position[c] for c in cols]] for cols in (list(spec.variables), lag_cols, inst_cols))
    cuts = np.flatnonzero(np.diff(codes)) + 1
    blocks = []
    for start, stop in zip(np.r_[0, cuts], np.r_[cuts, len(codes)]):
        key = keys[codes[start]]
        if ordinals[stop - 1] - ordinals[start] + 1 != stop - start:  # a missing (or repeated) period
            raise SpecificationError(f"entity {key} has gaps in its sample; a panel VAR needs consecutive periods")
        if stop - start < 2:
            continue
        Y, X, Z = Y_all[start:stop], X_all[start:stop], Z_all[start:stop]
        if transform == "fod":
            Y, X, Z = forward_orthogonal(Y), forward_orthogonal(X), Z[:-1]  # instruments: the lags' levels
        else:
            Y, X = Y - Y.mean(axis=0), X - X.mean(axis=0)
            Z = X
        blocks.append((key, Y, X, Z))
    if not blocks:
        raise SpecificationError("no entity has enough periods for the panel VAR")
    return blocks, frame


def _gmm(blocks: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray]], K: int, two_step: bool) -> dict[str, Any]:
    """System GMM with the same instruments in each equation; clustered (entity) covariance."""
    X = np.vstack([b[2] for b in blocks])
    Z = np.vstack([b[3] for b in blocks])
    Y = np.vstack([b[1] for b in blocks])
    m, kz = X.shape[1], Z.shape[1]
    if np.linalg.matrix_rank(Z.T @ X) < m:
        raise SpecificationError("the panel VAR's lags are not identified by its instruments")
    ZX, ZY = Z.T @ X, Z.T @ Y
    G = np.kron(np.eye(K), ZX)  # moments vec(Z'(Y - X B)) in b = vec(B), equation by equation
    gy = ZY.T.reshape(-1)

    def solve(W: np.ndarray) -> np.ndarray:
        return np.linalg.solve(G.T @ W @ G, G.T @ W @ gy)

    def score(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        B = b.reshape(K, m).T
        S = np.zeros((K * kz, K * kz))
        for _, yi, xi, zi in blocks:
            gi = (zi.T @ (yi - xi @ B)).T.reshape(-1)
            S += np.outer(gi, gi)
        return B, S

    W1 = np.kron(np.eye(K), np.linalg.inv(Z.T @ Z))
    b = solve(W1)
    B, S = score(b)
    if two_step and kz > m:
        W2 = np.linalg.pinv(S)
        b = solve(W2)
        B, S = score(b)
        cov = np.linalg.inv(G.T @ np.linalg.pinv(S) @ G)
        g = (Z.T @ (Y - X @ B)).T.reshape(-1)
        J = float(g @ np.linalg.pinv(S) @ g)
        Jdf = K * (kz - m)
    else:
        bread = np.linalg.inv(G.T @ W1 @ G)
        cov = bread @ G.T @ W1 @ S @ W1 @ G @ bread
        J, Jdf = None, 0
    U = Y - X @ B
    return {"B": B, "cov": cov, "U": U, "J": J, "Jdf": Jdf, "nobs": len(Y)}


@dataclass(frozen=True, eq=False)
class FittedPanelVAR:
    """A panel VAR's coefficients with clustered standard errors, its tests, impulse responses and decompositions."""

    spec: Any
    lags: int
    transform: str
    instrument_lags: int
    B: np.ndarray  # (K p, K): rows are the lags, lag-major; columns the equations
    cov: np.ndarray = field(repr=False)  # of vec(B) (equation by equation)
    resid: np.ndarray = field(repr=False)
    entities: int = 0
    nobs: int = 0
    J: float | None = None
    Jdf: int = 0
    identification: Any = None
    index: Any = field(default=None, repr=False)
    levels: Any = field(default=None, repr=False)  # the rows used, in levels: each variable and its lags (historical decompositions)

    stage = "estimate"

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self.spec.variables)

    @property
    def regressors(self) -> list[str]:
        return [f"{v}[-{lag}]" for lag in range(1, self.lags + 1) for v in self.variables]

    @property
    def coefs(self) -> np.ndarray:
        """Lag matrices, shape (lags, K, K): coefs[l][i, j] is variable j's effect at lag l + 1 in equation i."""
        from .var import _lag_matrices

        return _lag_matrices(self.B, 0, len(self.variables), self.lags)

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov)).reshape(len(self.variables), -1).T

    @property
    def sigma(self) -> np.ndarray:
        return self.resid.T @ self.resid / len(self.resid)

    @property
    def roots(self) -> np.ndarray:
        K, p = len(self.variables), self.lags
        companion = np.zeros((K * p, K * p))
        companion[:K, :] = np.hstack(list(self.coefs))
        companion[K:, :-K] = np.eye(K * (p - 1))
        return np.sort(np.abs(np.linalg.eigvals(companion)))[::-1]

    @property
    def is_stable(self) -> bool:
        return bool(self.roots[0] < 1)

    @property
    def stats(self) -> dict[str, float]:
        out = {"nobs": float(self.nobs), "entities": float(self.entities), "lags": float(self.lags)}
        if self.J is not None:
            out.update(J=self.J, J_df=float(self.Jdf), J_p=float(sps.chi2.sf(self.J, self.Jdf)))
        return out

    def summary(self) -> Table:
        cols = [c for v in self.variables for c in (v, "s.e.")]
        se = self.se
        rows = [[x for j in range(len(self.variables)) for x in (float(self.B[i, j]), float(se[i, j]))] for i in range(len(self.regressors))]
        how = ("GMM on forward orthogonal deviations, instrumented by the lags' levels "
               f"({self.instrument_lags} lag{'s' if self.instrument_lags != 1 else ''})" if self.transform == "fod"
               else "least squares on entity-demeaned data (within)")
        notes = [f"{self.nobs} observations, {self.entities} entities; {how}; standard errors clustered by entity.",
                 f"{'Stable' if self.is_stable else 'Not stable'}: the largest companion root is {self.roots[0]:.4f}."]
        if self.J is not None:
            notes.append(f"Hansen's J = {self.J:.4f}, chi-squared({self.Jdf}), p = {sps.chi2.sf(self.J, self.Jdf):.4f}.")
        return Table(cols, rows, index=self.regressors, index_name="lag", title=f"Panel VAR: {', '.join(self.variables)}", notes=tuple(notes))

    def granger(self) -> Table:
        """Wald tests that each variable's lags do not help to predict each other variable (and all the others' together)."""
        K, p = len(self.variables), self.lags
        rows, index = [], []
        for i, eq in enumerate(self.variables):
            offset = i * K * p
            for j, cause in enumerate(self.variables):
                if j == i:
                    continue
                pos = [offset + lag * K + j for lag in range(p)]
                rows.append(self._wald(pos))
                index.append(f"{cause} -> {eq}")
            pos = [offset + lag * K + j for lag in range(p) for j in range(K) if j != i]
            if K > 2:
                rows.append(self._wald(pos))
                index.append(f"all others -> {eq}")
        return Table(["chi-squared", "df", "p-value"], rows, index=index, index_name="excluded -> equation",
                     title="Panel VAR Granger causality", notes=("Wald tests with the clustered covariance.",))

    def _wald(self, pos: list[int]) -> list[float]:
        b = self.B.T.reshape(-1)[pos]
        V = self.cov[np.ix_(pos, pos)]
        stat = float(b @ np.linalg.solve(V, b))
        return [stat, float(len(pos)), float(sps.chi2.sf(stat, len(pos)))]

    @api()
    def identify(self, scheme: Any) -> FittedPanelVAR:
        from .var import _impact

        _impact(scheme, self.sigma, self.coefs, self.variables)
        return replace(self, identification=scheme)

    def impact(self) -> np.ndarray:
        from .var import Cholesky, _impact

        return _impact(self.identification or Cholesky(), self.sigma, self.coefs, self.variables)

    def irf(self, periods: int = 10, orthogonal: bool = True, cumulative: bool = False, draws: int = 500, alpha: float = 0.05,
            seed: int = 0) -> IRF:
        """Impulse responses, with bands from ``draws`` draws of the coefficients from their estimated distribution."""
        from .var import IRF, Cholesky, _impact, _lag_matrices, _ma, _scheme_label

        scheme = self.identification or Cholesky()
        K, p = len(self.variables), self.lags

        def responses(B: np.ndarray) -> np.ndarray:
            A = _lag_matrices(B, 0, K, p)
            phi = _ma(A, periods)
            out = phi @ _impact(scheme, self.sigma, A, self.variables) if orthogonal else phi
            return np.cumsum(out, axis=0) if cumulative else out

        values = responses(self.B)
        lower = upper = None
        if draws:
            rng = make_rng(seed)
            chol = np.linalg.cholesky(self.cov + 1e-14 * np.eye(len(self.cov)))
            b = self.B.T.reshape(-1)
            sims = np.stack([responses((b + chol @ rng.standard_normal(len(b))).reshape(K, -1).T) for _ in range(draws)])
            lower, upper = np.quantile(sims, [alpha / 2, 1 - alpha / 2], axis=0)
        label = _scheme_label(scheme) if orthogonal else "reduced-form (unit) shocks"
        return IRF(values, self.variables, self.variables, lower, upper, cumulative, label, 1 - alpha)

    def fevd(self, periods: int = 10) -> FEVD:
        from .var import FEVD, Cholesky, _ma, _scheme_label

        theta = _ma(self.coefs, periods - 1) @ self.impact()
        contributions = np.cumsum(theta**2, axis=0)
        return FEVD(contributions / contributions.sum(axis=2, keepdims=True), self.variables, list(self.variables),
                    _scheme_label(self.identification or Cholesky()))

    @api()
    def historical_decomposition(self, draws: int = 0, alpha: float = 0.05, seed: int = 0, entity: Any = None) -> Any:
        """Each entity's data over its estimation sample, split into the contributions of the identified structural
        shocks and the baseline: the path from the entity's first lags and its fixed effect, with no shocks. The parts
        add up to the data exactly, entity by entity.

        The fixed effect is recovered as the entity's mean of y(t) - A_1 y(t-1) - ... - A_p y(t-p) (the constant of
        within least squares; the same for GMM on forward orthogonal deviations, whose transform removed it); the
        residuals are the data less it and the lags; the shocks are ``P^-1 u(t)`` with the impact matrix of the
        current identification (Cholesky by default), as for a VAR.

        With ``draws``, ``1 - alpha`` bands for each contribution: pointwise quantiles over the decompositions under
        draws of the coefficients from their estimated distribution (the draws the panel VAR's impulse-response bands
        use, same ``seed``), each with its own fixed effects, residuals, baseline and impact matrix. Returns a
        ``PanelDecomposition``: one ``Decomposition`` per entity; ``entity`` chooses the one its views show first."""
        from .historical_extra import panel_historical

        return panel_historical(self, draws, alpha, seed, entity)

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return pd.DataFrame(self.B, index=self.regressors, columns=list(self.variables))

    def __repr__(self) -> str:
        return f"<FittedPanelVAR of {', '.join(self.variables)}: {self.lags} lags, {self.entities} entities>"


PANEL_CRITERIA = {"bic": "MBIC", "aic": "MAIC", "hq": "MQIC"}


def select_order(spec: VAR, data: Dataset, max_lags: int, span: Any = None, transform: Literal["fod", "within"] = "fod") -> Any:
    """Andrews and Lu's (2001) moment and model selection criteria for lags 1 ... ``max_lags``, as Abrigo and Love's
    pvarsoc reports them: every lag length estimated by GMM with the same instruments (the levels' lags 1 ... max_lags + 1)
    on the same rows, and with q instrument lags and p lags, K variables and n observations,

        MBIC = J - (q - p) K^2 ln n,  MAIC = J - 2 (q - p) K^2,  MQIC = J - 2.1 (q - p) K^2 ln ln n,

    smaller being better; and the overall coefficient of determination CD = 1 - det(Sigma_e) / det(Sigma_y) of the
    transformed data. With ``transform="within"`` (least squares, no J), Lutkepohl's AIC, BIC and HQ of ln det(Sigma_e)
    on the common sample instead. ``best`` maps "bic", "aic" and "hq" to the lags each picks."""
    from .var import LagSelection

    if transform not in ("fod", "within"):
        raise SpecificationError("transform is 'fod' (forward orthogonal deviations, GMM) or 'within' (least squares)")
    max_lags = int(max_lags)
    if max_lags < 1:
        raise SpecificationError("max_lags must be at least 1")
    K, q = len(spec.variables), max_lags + 1
    rows = []
    longest = _entity_blocks(spec, data, max_lags, q, transform, span)[0] if transform == "within" else None
    for p in range(1, max_lags + 1):
        if transform == "fod":  # the same instruments, so the same rows, for every p
            blocks = _entity_blocks(spec, data, p, q, transform, span)[0]
            est = _gmm(blocks, K, two_step=True)
        else:  # the same rows for every p: those the longest lags allow (lag-major columns: the first K p are lags 1 ... p)
            blocks = [(key, yi, xi[:, : K * p], xi[:, : K * p]) for key, yi, xi, _ in longest]  # type: ignore[union-attr]
            est = _gmm(blocks, K, two_step=False)
        Y = np.vstack([b[1] for b in blocks])
        n = est["nobs"]
        U = est["U"]
        cd = 1 - np.linalg.det(U.T @ U / n) / np.linalg.det(np.cov(Y.T, bias=True))
        if transform == "fod":
            J, extra = est["J"], (q - p) * K * K
            rows.append([cd, J, float(sps.chi2.sf(J, est["Jdf"])), J - extra * np.log(n), J - 2 * extra, J - 2.1 * extra * np.log(np.log(n)), n])
        else:
            logdet, free = np.linalg.slogdet(U.T @ U / n)[1], K * K * p
            rows.append([cd, logdet + np.log(n) * free / n, logdet + 2 * free / n, logdet + 2 * np.log(np.log(n)) * free / n, n])
    if transform == "fod":
        columns = ["CD", "J", "J p-value", "MBIC", "MAIC", "MQIC", "nobs"]
        labels = PANEL_CRITERIA
        how = (f"GMM on forward orthogonal deviations, instrumented by the levels' lags 1 to {q} for every p (Andrews and Lu 2001; "
               "Abrigo and Love's pvarsoc)")
    else:
        columns, labels = ["CD", "BIC", "AIC", "HQ", "nobs"], {"bic": "BIC", "aic": "AIC", "hq": "HQ"}
        how = "Least squares on entity-demeaned data; Lutkepohl's criteria of ln det(Sigma) on the rows the longest lags allow"
    frame = pd.DataFrame(rows, index=pd.Index(range(1, max_lags + 1), name="lags"), columns=columns)
    best = {k: int(frame[label].idxmin()) for k, label in labels.items()}
    notes = [how, f"{int(frame['nobs'].iloc[0])} observations for every lag length; smaller is better",
             *[f"Best by {labels[k]}: {p} lag{'s' * (p != 1)}" for k, p in best.items()]]
    return LagSelection(Table.from_frame(frame, title=f"Panel VAR lag selection: {', '.join(spec.variables)}", notes=notes), best)


def fit_panel(spec: VAR, data: Dataset, span: Any, p: int, transform: Literal["fod", "within"] = "fod",
              instrument_lags: int | None = None) -> FittedPanelVAR:
    if transform not in ("fod", "within"):
        raise SpecificationError("transform is 'fod' (forward orthogonal deviations, GMM) or 'within' (least squares)")
    if spec.exog or spec.trend == "ct":
        raise SpecificationError("a panel VAR here has entity fixed effects only: no exogenous variables or trend yet")
    q = p if instrument_lags is None or transform == "within" else int(instrument_lags)
    if q < p:
        raise SpecificationError("instrument_lags must be at least the number of lags")
    blocks, frame = _entity_blocks(spec, data, p, q, transform, span)
    K = len(spec.variables)
    est = _gmm(blocks, K, two_step=True)
    used = frame.loc[frame.index.get_level_values(spec.by).isin([b[0] for b in blocks])]  # entities with two periods or more
    levels = used[[*spec.variables, *[f"{v}[-{lag}]" for lag in range(1, p + 1) for v in spec.variables]]]
    return FittedPanelVAR(spec, p, transform, q, est["B"], est["cov"], est["U"], len(blocks), est["nobs"], est["J"], est["Jdf"],
                          None, frame.index, levels)
