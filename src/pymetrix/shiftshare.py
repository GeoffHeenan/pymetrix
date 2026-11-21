"""Shift-share (Bartik) instruments: 2SLS with an instrument built from regions' exposure shares and industry shocks.

    ss = shift_share(regions, "dy", "dx", shares=["s_agr", "s_man", ...], shocks={"s_agr": 0.02, "s_man": -0.05, ...})
    ss.summary()          # the 2SLS estimate with robust, AKM, AKM0 and shock-level (BHJ) standard errors
    ss.rotemberg()        # Goldsmith-Pinkham, Sorkin and Swift's weights: which industries drive the estimate
    ss.shock_level()      # Borusyak, Hull and Jaravel's equivalent regression across industries

With shares s_lk of region l in industry k and shocks g_k, the instrument is B_l = sum_k s_lk g_k. The outcome y and
the endogenous x are regressed on the controls W (a constant and ``controls``; with regression ``weights`` e_l), giving
y~ and x~, and beta = sum_l e_l B_l y~_l / sum_l e_l B_l x~_l (2SLS). The residual eps = y~ - beta x~.

Standard errors:

- robust: heteroskedasticity-robust across regions, sum_l e_l^2 B~_l^2 eps_l^2 / D^2, D = sum_l e_l B~_l x~_l;
- AKM (Adao, Kolesar and Morales 2019): sum_k Xhat_k^2 R_k^2 / D^2, with R_k = sum_l e_l s_lk eps_l and Xhat the
  coefficients of the residualized instrument B~ on the shares (weighted least squares); with ``shock_clusters`` the
  products Xhat_k R_k are summed within each cluster of industries first;
- AKM0: the same variance with the residuals under the null, eps(beta0) = y~ - beta0 x~, inverted into a
  confidence set (a quadratic inequality in beta0; it can be unbounded when the instrument is weak);
- BHJ (Borusyak, Hull and Jaravel 2022): the shock-level IV of ybar_k on xbar_k instrumented by g_k with weights
  s_k = sum_l e_l s_lk, ybar_k = sum_l e_l s_lk y~_l / s_k. Its coefficient equals beta exactly; its robust
  standard error is the exposure-robust one, sum_k g_k^2 R_k^2 / D^2 (clustered by ``shock_clusters`` if given).

Rotemberg weights (Goldsmith-Pinkham, Sorkin and Swift 2020): alpha_k = g_k Z_k' e x~ / sum_j g_j Z_j' e x~, with
beta_k = Z_k' e y~ / Z_k' e x~ the just-identified estimate with share k alone as instrument; sum_k alpha_k = 1 and
sum_k alpha_k beta_k = beta.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from pymetrix._lazy import stats as sps

from . import _micro
from ._api import api
from .errors import SpecificationError
from .output import Table

if TYPE_CHECKING:
    from .data import Dataset

__all__ = ["FittedShiftShare", "shift_share"]


def _wls_resid(W: np.ndarray, e: np.ndarray, *vs: np.ndarray) -> list[np.ndarray]:
    """Residuals of each v on W by least squares weighted by e."""
    sw = np.sqrt(e)
    Q = np.linalg.pinv(W * sw[:, None])
    return [v - W @ (Q @ (sw * v)) for v in vs]


def _cluster_sum(values: np.ndarray, clusters: np.ndarray | None) -> np.ndarray:
    if clusters is None:
        return values
    return pd.Series(values).groupby(clusters).sum().to_numpy()


@dataclass(frozen=True, eq=False)
class FittedShiftShare:
    """A shift-share IV estimate, with standard errors robust to shock-level correlation and the instrument's pieces."""

    outcome: str
    treatment: str
    controls: tuple[str, ...]
    industries: tuple[str, ...]
    beta: float
    se_robust: float
    se_akm: float
    se_bhj: float
    akm0_interval: tuple[float, float]  # may be (-inf, inf), or the complement of an interval (then lower > upper)
    first_stage: float  # the instrument's coefficient in the first stage (x~ on B~)
    first_stage_f: float  # robust first-stage F
    se: str  # which standard error params use
    shocks: pd.Series = field(repr=False)
    instrument: pd.Series = field(repr=False)  # B_l, by region
    shares: pd.DataFrame = field(repr=False)  # s_lk
    exposure: pd.Series = field(repr=False)  # s_k = sum_l e_l s_lk
    rotemberg_weights: pd.DataFrame = field(repr=False)
    shock_frame: pd.DataFrame = field(repr=False)  # the shock-level data: s_k, g_k, ybar_k, xbar_k
    nobs: int = 0
    shock_clusters: int | None = None
    alpha: float = 0.05
    provenance: Any = field(default=None, repr=False)

    stage = "estimate"

    @property
    def params(self) -> Any:
        from .estimation.params import PointEstimate

        s = {"akm": self.se_akm, "bhj": self.se_bhj, "robust": self.se_robust}[self.se]
        return PointEstimate((self.treatment,), np.array([self.beta]), np.array([[s**2]]))

    @property
    def stats(self) -> dict[str, float]:
        return {"nobs": float(self.nobs), "industries": float(len(self.industries)), "first_stage_f": self.first_stage_f}

    def summary(self) -> Table:
        rows, index = [], []
        for label, s in (("robust (regions)", self.se_robust), ("AKM", self.se_akm), ("BHJ shock level", self.se_bhj)):
            rows.append(_micro.estimate_rows([self.beta], [s], self.alpha)[0])
            index.append(label)
        lo, hi = self.akm0_interval
        rows.append([self.beta, np.nan, np.nan, np.nan, lo, hi])
        index.append("AKM0 (null imposed)")
        notes = [f"2SLS of {self.outcome} on {self.treatment}, instrument sum_k share_k x shock_k over {len(self.industries)} industries; "
                 f"{self.nobs} regions" + (f"; controls {', '.join(self.controls)}" if self.controls else "") + ".",
                 f"First stage: coefficient {self.first_stage:.4g}, robust F {self.first_stage_f:.2f}.",
                 "AKM and BHJ allow regions with similar shares to have correlated errors"
                 + (f"; industries clustered into {self.shock_clusters} groups." if self.shock_clusters else "."),
                 "BHJ's shock-level IV reproduces the coefficient exactly (ss.shock_level())."]
        if lo > hi:
            notes.append(f"The AKM0 set is everything outside ({hi:.4g}, {lo:.4g}): the instrument is weak.")
        return Table(_micro.interval_columns(self.alpha), rows, index=index, index_name="standard error", title=f"Shift-share IV: effect of {self.treatment} on {self.outcome}",
                     notes=tuple(notes))

    def in_akm0_set(self, value: float) -> bool:
        """Whether ``value`` is in the AKM0 confidence set (an interval, the line, or the line outside an interval)."""
        lo, hi = self.akm0_interval
        return bool(lo <= value <= hi) if lo <= hi else bool(value >= lo or value <= hi)

    def rotemberg(self) -> pd.DataFrame:
        """Each industry's Rotemberg weight alpha_k, shock g_k and just-identified estimate beta_k, largest weights first."""
        return self.rotemberg_weights.reindex(self.rotemberg_weights["weight"].abs().sort_values(ascending=False).index)

    def rotemberg_table(self, top: int = 10) -> Table:
        r = self.rotemberg()
        neg = r.loc[r["weight"] < 0, "weight"].sum()
        notes = (f"Weights sum to {r['weight'].sum():.6g}; negative weights sum to {neg:.4g}.",
                 "sum_k weight_k x beta_k = the 2SLS estimate; beta_k uses share k alone as the instrument.")
        shown = r.head(top)
        return Table(["weight", "shock", "beta_k", "exposure"], shown.to_numpy().tolist(), index=[str(i) for i in shown.index], index_name="industry",
                     title="Rotemberg weights (Goldsmith-Pinkham, Sorkin and Swift)", notes=notes)

    def shock_level(self) -> Table:
        """Borusyak, Hull and Jaravel's shock-level IV: the same coefficient, with exposure-robust errors."""
        f = self.shock_frame
        num = float((f["exposure"] * f["shock"] * f["ybar"]).sum())
        den = float((f["exposure"] * f["shock"] * f["xbar"]).sum())
        return Table(_micro.interval_columns(self.alpha), _micro.estimate_rows([num / den], [self.se_bhj], self.alpha), index=[self.treatment], index_name="parameter",
                     title="Shock-level IV (Borusyak, Hull and Jaravel)",
                     notes=(f"{len(f)} industries, weighted by their exposure s_k; outcome and treatment are exposure-weighted averages of the "
                            "residualized regional values; instrument: the shock.",))

    def plot(self) -> Any:
        from .figures import Bars, Figure, Panel, Rule, Scatter

        r = self.rotemberg().head(15)
        bars = Bars(pd.Series(r["weight"].to_numpy(), index=[str(i) for i in r.index]), "Rotemberg weight", horizontal=True)
        full = self.rotemberg_weights
        scatter = Scatter(pd.Series(full["weight"].to_numpy(), name="Rotemberg weight"), pd.Series(full["beta_k"].to_numpy(), name="beta_k"), "industries")
        return Figure((Panel((bars, Rule(0.0, vertical=True)), title="Largest Rotemberg weights"),
                       Panel((scatter, Rule(self.beta, style="--")), title="Just-identified estimates against their weights",
                             xlabel="Rotemberg weight", ylabel="beta_k")))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.summary().to_frame()

    def __repr__(self) -> str:
        return f"<FittedShiftShare {self.treatment} on {self.outcome}: {self.beta:.4g} (AKM s.e. {self.se_akm:.3g})>"


def _shock_vector(shocks: Any, industries: list[str]) -> np.ndarray:
    if isinstance(shocks, (Mapping, pd.Series)):
        missing = [k for k in industries if k not in shocks]
        if missing:
            raise SpecificationError(f"no shock for {', '.join(missing)}")
        return np.array([float(shocks[k]) for k in industries])
    values = np.asarray(list(shocks), dtype=float)
    if values.shape != (len(industries),):
        raise SpecificationError(f"{len(values)} shocks for {len(industries)} shares")
    return values


@api("pymetrix:shift_share")
def shift_share(data: Dataset, outcome: str, treatment: str, shares: Sequence[str], shocks: Any, controls: Sequence[str] = (),
                weights: str | None = None, shock_clusters: Mapping[str, Any] | None = None, se: Literal["akm", "bhj", "robust"] = "akm",
                alpha: float = 0.05) -> FittedShiftShare:
    """Shift-share IV: ``outcome`` on the endogenous ``treatment``, instrumented by sum_k share_k shock_k, with a
    constant and ``controls``. ``shares`` are the series of exposure shares (one per industry), ``shocks`` one per
    share (a mapping from share name, or a list in the same order); ``weights`` a series of regression weights;
    ``shock_clusters`` maps each share to a cluster of industries for AKM and BHJ errors; ``se`` chooses the
    standard error the coefficient carries."""
    from ._capture import capture

    industries, controls = _micro.names(shares), _micro.names(controls)
    if len(industries) < 2:
        raise SpecificationError("a shift-share instrument needs at least two shares")
    if se not in ("akm", "bhj", "robust"):
        raise SpecificationError(f"se is 'akm', 'bhj' or 'robust', not {se!r}")
    g = _shock_vector(shocks, industries)
    wanted = [outcome, treatment, *industries, *controls, *([weights] if weights else [])]
    frame = _micro.columns(data, wanted, "shift-share IV")
    y, x = _micro.numeric(frame, [outcome], "shift-share")[:, 0], _micro.numeric(frame, [treatment], "shift-share")[:, 0]
    S = _micro.numeric(frame, industries, "shift-share")
    e = _micro.numeric(frame, [weights], "shift-share")[:, 0] if weights else np.ones(len(y))
    if (e <= 0).any():
        raise SpecificationError("regression weights must be positive")
    W = np.column_stack([np.ones(len(y)), *([_micro.numeric(frame, controls, "shift-share")] if controls else [])])
    n = len(y)
    if n <= W.shape[1] + 1:
        raise SpecificationError(f"{n} regions: too few")
    clusters = None
    if shock_clusters is not None:
        missing = [k for k in industries if k not in shock_clusters]
        if missing:
            raise SpecificationError(f"shock_clusters has no cluster for {', '.join(missing)}")
        clusters = np.array([str(shock_clusters[k]) for k in industries])
    with capture() as caught:
        B = S @ g
        yt, xt, Bt = _wls_resid(W, e, y, x, B)
        D = float(np.sum(e * Bt * xt))
        if abs(D) < 1e-12 * max(1.0, float(np.sum(e * Bt**2))):
            raise SpecificationError("the instrument is uncorrelated with the treatment given the controls")
        beta = float(np.sum(e * Bt * yt)) / D
        eps = yt - beta * xt
        se_robust = float(np.sqrt(np.sum((e * Bt * eps) ** 2)) / abs(D))
        # AKM: the residualized instrument on the shares (weighted), and the share-weighted residuals
        sw = np.sqrt(e)
        Xhat = np.linalg.lstsq(S * sw[:, None], Bt * sw, rcond=None)[0]
        R = S.T @ (e * eps)
        se_akm = float(np.sqrt(np.sum(_cluster_sum(Xhat * R, clusters) ** 2)) / abs(D))
        se_bhj = float(np.sqrt(np.sum(_cluster_sum(g * R, clusters) ** 2)) / abs(D))
        # AKM0: invert the null-imposed test
        a, b = S.T @ (e * yt), S.T @ (e * xt)
        z2 = sps.norm.ppf(1 - alpha / 2) ** 2
        c = z2 / D**2
        pa, pb = _cluster_sum(Xhat * a, clusters), _cluster_sum(Xhat * b, clusters)
        A2, A1, A0 = float(pb @ pb), float(pa @ pb), float(pa @ pa)
        qa, qb, qc = 1 - c * A2, -2 * (beta - c * A1), beta**2 - c * A0
        disc = qb**2 - 4 * qa * qc
        if qa > 0:
            if disc < 0:
                interval = (np.nan, np.nan)
            else:
                r = np.sqrt(disc)
                interval = ((-qb - r) / (2 * qa), (-qb + r) / (2 * qa))
        elif disc <= 0:
            interval = (-np.inf, np.inf)
        else:  # the set outside two roots: reported as (upper root, lower root)
            r = np.sqrt(disc)
            lo, hi = sorted(((-qb - r) / (2 * qa), (-qb + r) / (2 * qa)))
            interval = (hi, lo)
        # first stage
        pi = float(np.sum(e * Bt * xt) / np.sum(e * Bt**2))
        u = xt - pi * Bt
        se_pi = np.sqrt(np.sum((e * Bt * u) ** 2)) / np.sum(e * Bt**2)
        # shock level and Rotemberg weights
        sk = S.T @ e
        with np.errstate(invalid="ignore", divide="ignore"):
            ybar, xbar = (S.T @ (e * yt)) / sk, (S.T @ (e * xt)) / sk
            zx = S.T @ (e * xt)
            alpha_k = g * zx / np.sum(g * zx)
            beta_k = (S.T @ (e * yt)) / zx
    index = frame.index
    shock_frame = pd.DataFrame({"exposure": sk, "shock": g, "ybar": ybar, "xbar": xbar}, index=pd.Index(industries, name="industry"))
    rot = pd.DataFrame({"weight": alpha_k, "shock": g, "beta_k": beta_k, "exposure": sk}, index=pd.Index(industries, name="industry"))
    spec = {"outcome": outcome, "treatment": treatment, "shares": industries, "shocks": dict(zip(industries, g.tolist())), "controls": controls,
            "weights": weights, "se": se}
    prov = _micro.provenance(spec, data, frame, [outcome, treatment, *industries, *controls], caught,
                             {"type": "shift-share IV", "estimation": "2SLS", "standard errors": ["robust", "AKM", "AKM0", "BHJ"]})
    return FittedShiftShare(outcome, treatment, tuple(controls), tuple(industries), beta, se_robust, se_akm, se_bhj,
                            (float(interval[0]), float(interval[1])), pi, float((pi / se_pi) ** 2), se, pd.Series(g, index=industries, name="shock"),
                            pd.Series(B, index=index, name="instrument"), pd.DataFrame(S, index=index, columns=industries), pd.Series(sk, index=industries, name="exposure"),
                            rot, shock_frame, n, len(np.unique(clusters)) if clusters is not None else None, alpha, prov)
