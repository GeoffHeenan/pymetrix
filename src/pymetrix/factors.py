"""Principal components, and factor models.

    pca = PCA(["gr", "gip", "gemp", "pmi"], n=2).fit(monthly)
    pca.scores                   # the components, period by period (a DataFrame: pc1, pc2)
    pca.loadings                 # each series' weight in each component
    pca.explained                # the share of the variance each component explains
    pca.transform(newer)         # the same components for other data (e.g. a bridge equation on factors)

Components are the eigenvectors of the correlation matrix (``standardize=True``: series scaled to unit variance)
or of the covariance matrix, on the periods where every series is observed. Signs are fixed so that each
component's largest loading is positive. For nowcasting with missing values and mixed frequencies, use
``pymetrix.nowcast.DynamicFactor``.

    fa = FactorModel(["educ", "motheduc", "fatheduc", "age", "exper"], factors=2, rotation="varimax").fit(data)
    fa.loadings; fa.uniquenesses; fa.scores("bartlett"); fa.test     # k factors are enough: chi-squared

``FactorModel`` is maximum-likelihood factor analysis of the correlation matrix, as R's factanal computes it: the
uniquenesses maximise the likelihood (bounded below by 0.005, so a variable explained entirely by the factors shows
as a Heywood case at the bound), the loadings follow from them, then are rotated: varimax (orthogonal, Kaiser
normalised) or promax (oblique, power 4, from varimax), with the factors' correlations. Factors are ordered by their
sums of squared loadings and signed so that each column of loadings sums to a positive number. Regression (Thomson)
scores use the factors' correlations under an oblique rotation (R 4.3 leaves them out); Bartlett scores are weighted
least squares.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ._api import api
from .errors import NotFoundError, SpecificationError
from .output import Table

__all__ = ["PCA", "FittedPCA", "FactorModel", "FittedFactorModel"]


@dataclass(frozen=True)
class PCA:
    """Principal components of a set of series: the weighted averages that explain most of their variance.

    ``n`` keeps the first n components (all by default); ``standardize`` (the default) works with the series scaled
    to unit variance, so that each counts equally whatever its units."""

    variables: Sequence[str]
    n: int | None = None  # components kept; all by default
    standardize: bool = True

    @api("pymetrix:pca.fit")
    def fit(self, data: Any, span: Any = None) -> FittedPCA:
        """The components of the series over the periods of ``span`` (first, last) where none is missing."""
        names = list(self.variables)
        missing = [v for v in names if v not in data.names]
        if missing:
            raise NotFoundError(f"no series {', '.join(missing)} in the data")
        frame = data.frame[data.sample_mask(span)][names].astype(float).dropna()
        if len(frame) <= len(names):
            raise SpecificationError(f"{len(frame)} complete periods for {len(names)} series: too few for principal components")
        mean = frame.mean()
        scale = frame.std(ddof=1) if self.standardize else pd.Series(1.0, index=names)
        z = (frame - mean) / scale
        values, vectors = np.linalg.eigh(np.cov(z.to_numpy(), rowvar=False))
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        flip = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(vectors.shape[1])])
        vectors = vectors * np.where(flip == 0, 1.0, flip)
        k = len(names) if self.n is None else int(self.n)
        if not 1 <= k <= len(names):
            raise SpecificationError(f"n must be between 1 and {len(names)}")
        columns = [f"pc{j + 1}" for j in range(k)]
        loadings = pd.DataFrame(vectors[:, :k], index=names, columns=columns)
        from ._capture import provenance_for

        prov = provenance_for(self, data, names, frame.index, frame, [], {"type": "principal components",
                                                                         "standardize": self.standardize})
        return FittedPCA(self, mean, scale, loadings, pd.Series(values, index=[f"pc{j + 1}" for j in range(len(values))]),
                         z.to_numpy() @ vectors[:, :k], frame.index, columns, prov)


@dataclass(frozen=True, eq=False)
class FittedPCA:
    spec: PCA
    mean: pd.Series
    scale: pd.Series
    loadings: pd.DataFrame
    eigenvalues: pd.Series  # of every component, largest first
    _scores: np.ndarray
    _index: Any
    _columns: list[str]
    provenance: Any = None  # data vintages, sample, method, fingerprint

    @property
    def scores(self) -> pd.DataFrame:
        """The components in each period of the sample: one column per component (pc1, pc2, ...)."""
        return pd.DataFrame(self._scores, index=self._index, columns=self._columns)

    @property
    def explained(self) -> pd.Series:
        """The share of the total variance each component explains."""
        return (self.eigenvalues / self.eigenvalues.sum()).rename("share of variance")

    def transform(self, data: Any) -> pd.DataFrame:
        """The components for (other) data, with this fit's means, scales and loadings."""
        frame = data.frame[list(self.loadings.index)].astype(float)
        return ((frame - self.mean) / self.scale) @ self.loadings

    def factor_criteria(self, kmax: int | None = None) -> Table:
        """Bai and Ng's (2002) information criteria for the number of factors: IC_p1, IC_p2 and IC_p3 for k = 1 to
        ``kmax`` (default min(8, N - 1)), with the k each picks (the smallest value).

        V(k) is the average squared residual of the series (as the components see them: standardized or demeaned)
        after their projection on the first k components; IC = ln V(k) + k g(N, T) with the penalties
        (N + T)/(NT) ln(NT/(N + T)), (N + T)/(NT) ln C^2 and ln(C^2)/C^2, C^2 = min(N, T)."""
        N, T = len(self.loadings), len(self._index)
        kmax = min(8, N - 1) if kmax is None else int(kmax)
        if not 1 <= kmax < N:
            raise SpecificationError(f"kmax must be between 1 and {N - 1}")
        lam = self.eigenvalues.to_numpy(float) * (T - 1) / T  # eigenvalues of Z'Z / T
        c2 = min(N, T)
        penalties = {"IC_p1": (N + T) / (N * T) * np.log(N * T / (N + T)), "IC_p2": (N + T) / (N * T) * np.log(c2),
                     "IC_p3": np.log(c2) / c2}
        ks = np.arange(1, kmax + 1)
        V = np.array([lam[k:].sum() / N for k in ks])
        values = {name: np.log(V) + ks * pen for name, pen in penalties.items()}
        rows = [[float(V[i])] + [float(values[name][i]) for name in penalties] for i in range(len(ks))]
        chosen = {name: int(ks[np.argmin(v)]) for name, v in values.items()}
        return Table(["V(k)", *penalties], rows, index=[str(k) for k in ks], index_name="factors",
                     title="Number of factors (Bai and Ng 2002)",
                     notes=("Factors chosen: " + ", ".join(f"{name} {k}" for name, k in chosen.items()) + f"; N = {N} series, T = {T} periods.",
                            "IC_p2 is the usual choice; the criteria need N and T both reasonably large."))

    def summary(self) -> Table:
        share = self.explained
        notes = [f"{'Correlation' if self.spec.standardize else 'Covariance'} matrix; {len(self._index)} complete periods",
                 "Variance explained: " + ", ".join(f"{c} {share[c]:.1%}" for c in self._columns)]
        return Table.from_frame(self.loadings, title="Principal components: loadings", notes=notes)

    def __repr__(self) -> str:
        return f"<FittedPCA: {len(self._columns)} of {len(self.loadings)} components, {self.explained.iloc[:len(self._columns)].sum():.1%} of the variance>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()


def _fa_objective(psi: np.ndarray, S: np.ndarray, q: int) -> tuple[float, np.ndarray]:
    """factanal's FAfn and FAgr: the concentrated discrepancy at uniquenesses psi, and its gradient."""
    sc = 1 / np.sqrt(psi)
    values, vectors = np.linalg.eigh(S * np.outer(sc, sc))
    values, vectors = values[::-1], vectors[:, ::-1]
    e = values[q:]
    load = np.sqrt(psi)[:, None] * vectors[:, :q] * np.sqrt(np.maximum(values[:q] - 1, 0))
    g = load @ load.T + np.diag(psi) - S
    return float(-(np.sum(np.log(e) - e) - q + len(S))), np.diag(g) / psi**2


def _fa_loadings(psi: np.ndarray, S: np.ndarray, q: int) -> np.ndarray:
    sc = 1 / np.sqrt(psi)
    values, vectors = np.linalg.eigh(S * np.outer(sc, sc))
    values, vectors = values[::-1], vectors[:, ::-1]
    return np.sqrt(psi)[:, None] * vectors[:, :q] * np.sqrt(np.maximum(values[:q] - 1, 0))


def varimax(L: np.ndarray, normalize: bool = True, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Kaiser's varimax rotation, as R's stats::varimax: (rotated loadings, rotation matrix)."""
    p, k = L.shape
    if k < 2:
        return L, np.eye(k)
    sc = np.sqrt(np.sum(L**2, axis=1)) if normalize else np.ones(p)
    x = L / sc[:, None]
    T, d = np.eye(k), 0.0
    for _ in range(1000):
        z = x @ T
        B = x.T @ (z**3 - z * np.sum(z**2, axis=0) / p)
        u, sv, vt = np.linalg.svd(B)
        T = u @ vt
        d, past = float(np.sum(sv)), d
        if d < past * (1 + eps):
            break
    return (x @ T) * sc[:, None], T


def promax(L: np.ndarray, m: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Hendrickson and White's promax rotation, as R's stats::promax: (rotated loadings, rotation matrix R), the
    loadings L R and the factors' correlations inv(R'R)."""
    k = L.shape[1]
    if k < 2:
        return L, np.eye(k)
    x, T = varimax(L)
    Q = x * np.abs(x) ** (m - 1)
    U = np.linalg.lstsq(x, Q, rcond=None)[0]
    U = U * np.sqrt(np.diag(np.linalg.inv(U.T @ U)))[None, :]
    return x @ U, T @ U


@dataclass(frozen=True)
class FactorModel:
    """Maximum-likelihood factor analysis: each series is a combination of ``factors`` common factors plus a part of
    its own (its uniqueness); ``rotation`` makes the loadings easier to read (none, varimax, promax).
    ``dynamics`` above 0 (factors following a VAR) is the dynamic factor model: use pymetrix.nowcast.DynamicFactor."""

    variables: Sequence[str]
    factors: int
    dynamics: int = 0  # 0: static; p: factors follow a VAR(p)
    rotation: Literal["none", "varimax", "promax"] = "none"

    @api("pymetrix:factor_model.fit")
    def fit(self, data: Any, span: Any = None, method: Any = None) -> FittedFactorModel:
        """The factor model on the periods (or rows) of ``span`` where every series is observed."""
        if self.dynamics:
            raise SpecificationError("dynamic factor models are pymetrix.nowcast.DynamicFactor; dynamics=0 is the static model")
        if self.rotation not in ("none", "varimax", "promax"):
            raise SpecificationError("rotation is none, varimax or promax")
        names = list(self.variables)
        missing = [v for v in names if v not in data.names]
        if missing:
            raise NotFoundError(f"no series {', '.join(missing)} in the data")
        p, q = len(names), int(self.factors)
        if p < 3:
            raise SpecificationError("factor analysis needs at least three series")
        dof = 0.5 * ((p - q) ** 2 - p - q)
        if q < 1 or dof < 0:
            raise SpecificationError(f"{q} factors are too many for {p} series (at most those with (p - k)^2 >= p + k)")
        frame = data.frame[data.sample_mask(span)][names].astype(float).dropna()
        n = len(frame)
        if n <= p:
            raise SpecificationError(f"{n} complete observations for {p} series: too few for factor analysis")
        S = np.corrcoef(frame.to_numpy(), rowvar=False)
        from scipy import optimize

        start = (1 - 0.5 * q / p) / np.diag(np.linalg.inv(S))
        scale = 0.01  # factanal's parscale: the optimiser works on psi / 0.01
        def scaled(u: np.ndarray) -> tuple[float, np.ndarray]:
            value, grad = _fa_objective(u * scale, S, q)
            return value, grad * scale

        found = optimize.minimize(scaled, start / scale, jac=True, method="L-BFGS-B", bounds=[(0.005 / scale, 1 / scale)] * p,
                                  options={"ftol": 1e7 * np.finfo(float).eps, "gtol": 0.0, "maxcor": 5, "maxiter": 1000})
        psi = found.x * scale
        objective = _fa_objective(psi, S, q)[0]
        L = _fa_loadings(psi, S, q)
        phi = None
        rotmat = np.eye(q)
        if self.rotation == "varimax":
            L, rotmat = varimax(L)
        elif self.rotation == "promax":
            L, rotmat = promax(L)
            phi = np.linalg.inv(rotmat.T @ rotmat)
        order = np.argsort(-np.sum(L**2, axis=0), kind="stable")  # factanal's sortLoadings
        L = L[:, order]
        sign = np.where(L.sum(axis=0) < 0, -1.0, 1.0)
        L = L * sign
        if phi is not None:
            phi = sign[:, None] * phi[np.ix_(order, order)] * sign[None, :]
        columns = [f"factor{j + 1}" for j in range(q)]
        statistic = (n - 1 - (2 * p + 5) / 6 - 2 * q / 3) * objective
        from scipy import stats
        from ._capture import provenance_for

        prov = provenance_for(self, data, names, frame.index, frame, [], {"type": "maximum-likelihood factor analysis",
                                                                         "rotation": self.rotation})
        return FittedFactorModel(
            self, pd.DataFrame(L, index=names, columns=columns), pd.Series(psi, index=names, name="uniqueness"),
            None if phi is None else pd.DataFrame(phi, index=columns, columns=columns), S, frame,
            {"statistic": float(statistic), "df": float(dof), "p_value": float(stats.chi2.sf(statistic, dof)) if dof > 0 else np.nan,
             "objective": objective, "converged": bool(found.success), "heywood": [v for v, u in zip(names, psi) if u <= 0.005 + 1e-9]},
            prov)


@dataclass(frozen=True, eq=False)
class FittedFactorModel:
    """A fitted factor model: loadings, uniquenesses, the factors' correlations (promax), scores and the test that the
    number of factors is enough."""

    spec: FactorModel
    loadings: pd.DataFrame
    uniquenesses: pd.Series
    factor_correlation: pd.DataFrame | None
    correlation: np.ndarray
    frame: pd.DataFrame
    stats: dict[str, Any]
    provenance: Any = None

    @property
    def communalities(self) -> pd.Series:
        """The share of each series' variance the factors explain: 1 - uniqueness."""
        return (1 - self.uniquenesses).rename("communality")

    @property
    def explained(self) -> pd.DataFrame:
        """Each factor's sum of squared loadings, and its share of the total variance."""
        ss = (self.loadings**2).sum()
        return pd.DataFrame({"SS loadings": ss, "proportion": ss / len(self.loadings), "cumulative": (ss / len(self.loadings)).cumsum()})

    @property
    def test(self) -> Any:
        """The likelihood-ratio test that this many factors are enough (Bartlett's correction), as factanal reports it."""
        from .stats.base import TestResult

        st = self.stats
        return TestResult(f"{self.loadings.shape[1]} factors are sufficient", f"{self.loadings.shape[1]} factors are sufficient",
                          st["statistic"], int(st["df"]), None if np.isnan(st["p_value"]) else st["p_value"], "chi2")

    def scores(self, method: Literal["regression", "bartlett"] = "regression") -> pd.DataFrame:
        """Factor scores for each observation used: regression (Thomson) or Bartlett (weighted least squares)."""
        z = ((self.frame - self.frame.mean()) / self.frame.std(ddof=1)).to_numpy()
        L = self.loadings.to_numpy()
        if method == "regression":
            sc = z @ np.linalg.solve(self.correlation, L)
            if self.factor_correlation is not None:
                sc = sc @ self.factor_correlation.to_numpy()
        elif method == "bartlett":
            tmp = (L / self.uniquenesses.to_numpy()[:, None]).T
            sc = np.linalg.solve(tmp @ L, tmp @ z.T).T
        else:
            raise SpecificationError("scores are regression or bartlett")
        return pd.DataFrame(sc, index=self.frame.index, columns=self.loadings.columns)

    def summary(self) -> Table:
        frame = self.loadings.copy()
        frame["uniqueness"] = self.uniquenesses
        st = self.stats
        notes = [f"Maximum likelihood on the correlation matrix of {len(self.frame)} observations; rotation {self.spec.rotation}",
                 f"{len(self.loadings.columns)} factors are sufficient: chi-squared {st['statistic']:.3f} on {st['df']:g} df"
                 + (f", p-value {st['p_value']:.4g}" if not np.isnan(st["p_value"]) else " (no degrees of freedom left)"),
                 "Variance explained: " + ", ".join(f"{c} {v:.1%}" for c, v in self.explained["proportion"].items())]
        if self.factor_correlation is not None:
            fc = self.factor_correlation.to_numpy()
            notes.append("Factor correlations: " + ", ".join(f"{self.loadings.columns[i]}-{self.loadings.columns[j]} {fc[i, j]:.3f}"
                                                              for i in range(len(fc)) for j in range(i + 1, len(fc))))
        if st["heywood"]:
            notes.append(f"Heywood case: the uniqueness of {', '.join(st['heywood'])} is at its lower bound (0.005)")
        if not st["converged"]:
            notes.append("The optimiser did not converge")
        return Table.from_frame(frame, title=f"Factor analysis: {len(self.loadings.columns)} factors", notes=notes)

    def __repr__(self) -> str:
        return f"<FittedFactorModel: {self.loadings.shape[1]} factors of {len(self.loadings)} series, rotation {self.spec.rotation}>"

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()
