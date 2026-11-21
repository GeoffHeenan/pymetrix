"""Synthetic difference-in-differences (Arkhangelsky, Athey, Hirshberg, Imbens and Wager 2021).

    sdid = synthetic_did(data, "packs", "treated")      # a 0/1 treatment, all treated units adopting together
    sdid.summary(); sdid.plot(); sdid.unit_weights; sdid.time_weights

Unit weights omega (on the simplex, with an intercept) make the controls' pre-treatment path parallel to the treated
units' average, with a ridge penalty zeta^2 T_pre ||omega||^2, zeta = (N_treated T_post)^(1/4) sigma, sigma the standard
deviation of the controls' first differences before treatment. Time weights lambda (on the simplex, with an
intercept, almost unpenalised) make the pre-treatment periods predict the controls' post-treatment average. The
estimate is the weighted double difference

    tau = (Ybar_tr,post - sum_t lambda_t Ybar_tr,t) - sum_i omega_i (Ybar_i,post - sum_t lambda_t Y_it),

which is also two-way fixed effects weighted by omega_i lambda_t (treated units and post periods weighted uniformly).
Difference-in-differences (uniform weights) and the synthetic control (omega without intercept or penalty, no time
weights) are reported beside it. Standard errors: the jackknife over units with several treated units, placebo
assignments among the controls with one (Algorithm 4 of the paper).
"""

from __future__ import annotations

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

__all__ = ["SyntheticDiD", "synthetic_did"]


def simplex_weights(A: np.ndarray, b: np.ndarray, penalty: float, intercept: bool) -> np.ndarray:
    """argmin over the simplex of ||w0 + A w - b||^2 + penalty ||w||^2 (w0 free, concentrated out, if ``intercept``)."""
    if intercept:
        A = A - A.mean(axis=0)
        b = b - b.mean()
    J = A.shape[1]
    P = A.T @ A + penalty * np.eye(J)
    q = A.T @ b

    def loss(w: np.ndarray) -> float:
        return float(w @ P @ w - 2 * q @ w)

    def grad(w: np.ndarray) -> np.ndarray:
        return 2 * (P @ w - q)

    found = optimize.minimize(loss, np.full(J, 1 / J), jac=grad, method="SLSQP", bounds=[(0.0, 1.0)] * J,
                              constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1, "jac": lambda w: np.ones(J)}],
                              options={"maxiter": 2000, "ftol": 1e-15})
    w = np.clip(found.x, 0, None)
    return w / w.sum()


def _estimates(Y: np.ndarray, n_co: int, t_pre: int) -> dict[str, Any]:
    """SDID, DiD and SC on a (units, periods) array: controls first, then treated; pre periods first."""
    Yco, Ytr = Y[:n_co], Y[n_co:]
    n_tr, T = Ytr.shape
    t_post = T - t_pre
    sigma = np.std(np.diff(Yco[:, :t_pre], axis=1), ddof=1)
    zeta = (n_tr * t_post) ** 0.25 * sigma
    tr_pre, tr_post = Ytr[:, :t_pre].mean(axis=0), Ytr[:, t_pre:].mean()
    omega = simplex_weights(Yco[:, :t_pre].T, tr_pre, zeta**2 * t_pre, intercept=True)
    lam = simplex_weights(Yco[:, :t_pre], Yco[:, t_pre:].mean(axis=1), (1e-6 * sigma) ** 2 * n_co, intercept=True)
    co_post = Yco[:, t_pre:].mean(axis=1)

    def double_difference(w: np.ndarray, l: np.ndarray) -> float:
        return float((tr_post - tr_pre @ l) - w @ (co_post - Yco[:, :t_pre] @ l))

    tau = double_difference(omega, lam)
    did = double_difference(np.full(n_co, 1 / n_co), np.full(t_pre, 1 / t_pre))
    sc_w = simplex_weights(Yco[:, :t_pre].T, tr_pre, (1e-6 * sigma) ** 2 * t_pre, intercept=False)
    sc = float(tr_post - sc_w @ co_post)
    return {"tau": tau, "did": did, "sc": sc, "omega": omega, "lambda": lam, "sc_weights": sc_w, "zeta": float(zeta)}


@dataclass(frozen=True, eq=False)
class SyntheticDiD:
    """A synthetic difference-in-differences estimate, with DiD and the synthetic control beside it."""

    outcome: str
    att: float
    se: float
    did: float
    sc: float
    unit_weights: pd.Series
    time_weights: pd.Series
    se_method: str
    treated_units: tuple[Any, ...]
    start: Any
    zeta: float
    paths: pd.DataFrame = field(repr=False)  # the treated average and the omega-weighted controls by period

    stage = "estimate"

    def summary(self) -> Table:
        q = sps.norm.ppf(0.975)
        z = self.att / self.se if self.se > 0 else np.nan
        rows = [["synthetic DiD", self.att, self.se, z, float(2 * sps.norm.sf(abs(z))), self.att - q * self.se, self.att + q * self.se],
                ["difference-in-differences", self.did, np.nan, np.nan, np.nan, np.nan, np.nan],
                ["synthetic control", self.sc, np.nan, np.nan, np.nan, np.nan, np.nan]]
        top = self.unit_weights.sort_values(ascending=False).head(5)
        notes = (f"{len(self.treated_units)} treated unit(s) from {self.start}; {int((self.unit_weights > 0).sum())} controls with weight; "
                 f"standard error by {self.se_method}; zeta = {self.zeta:.4g}.",
                 "Largest unit weights: " + ", ".join(f"{k} {v:.3f}" for k, v in top.items()))
        return Table(["", "estimate", "std err", "z", "p-value", "2.5%", "97.5%"], [[r[0], *map(float, r[1:])] for r in rows],
                     title=f"Synthetic difference-in-differences: {self.outcome}", notes=notes)

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from .views import _views_to_frame

            return _views_to_frame(self, view)
        return self.paths

    def plot(self) -> Any:
        """The treated units' average against the weighted controls (shifted by the pre-period gap), with the start marked."""
        from .figures import Figure, Line, Panel

        p = self.paths
        return Figure((Panel((Line(p["treated"], "treated"), Line(p["synthetic"], "synthetic control (weighted, shifted)", style="--")),
                             title=f"{self.outcome}: synthetic difference-in-differences"),))

    def __repr__(self) -> str:
        return f"<SyntheticDiD {self.outcome}: {self.att:.4f} ({self.se:.4f})>"


@api("pymetrix:synthetic_did")
def synthetic_did(data: Dataset, outcome: str, treatment: str, se: Literal["auto", "jackknife", "placebo"] = "auto",
                  placebos: int = 200, seed: int = 0) -> SyntheticDiD:
    """Synthetic difference-in-differences of ``outcome`` for the units ``treatment`` turns on, all at the same period
    and for good (a block design); a balanced panel."""
    from .treatment import _check, _panel_axes

    unit, time = _panel_axes(data, "synthetic difference-in-differences")
    _check(data, [outcome, treatment])
    Y = data.frame[outcome].unstack(unit).T  # units by periods
    D = data.frame[treatment].unstack(unit).T
    if Y.isna().to_numpy().any() or D.isna().to_numpy().any():
        raise SpecificationError("synthetic DiD needs a balanced panel with no missing outcomes or treatments")
    Dv = D.to_numpy(float)
    if not np.isin(Dv, (0.0, 1.0)).all():
        raise SpecificationError(f"{treatment} must be 0 or 1")
    treated = Dv.max(axis=1) == 1
    if not treated.any() or treated.all():
        raise SpecificationError("synthetic DiD needs treated and never-treated units")
    first = np.argmax(Dv[treated], axis=1)
    if len(set(first)) != 1 or not np.all(Dv[treated][:, first[0]:] == 1):
        raise SpecificationError("synthetic DiD here needs every treated unit to adopt at the same period, for good; "
                                 "use staggered_did for staggered adoption")
    t_pre = int(first[0])
    if t_pre < 2:
        raise SpecificationError("synthetic DiD needs at least two periods before treatment")
    Yv = Y.to_numpy(float)
    order = np.r_[np.flatnonzero(~treated), np.flatnonzero(treated)]
    Ys = Yv[order]
    n_co = int((~treated).sum())
    est = _estimates(Ys, n_co, t_pre)
    method = ("jackknife" if treated.sum() > 1 else "placebo") if se == "auto" else se
    if method == "jackknife":
        if treated.sum() < 2:
            raise SpecificationError("the jackknife needs at least two treated units; use se='placebo'")
        n = len(Ys)
        taus = []
        for i in range(n):  # the weights held fixed, unit i dropped (Algorithm 3)
            keep = np.delete(np.arange(n), i)
            Yk = Ys[keep]
            nk = n_co - (i < n_co)
            om = np.delete(est["omega"], i) if i < n_co else est["omega"]
            if om.sum() <= 0:
                continue
            om = om / om.sum()
            lam = est["lambda"]
            tr = Yk[nk:]
            co = Yk[:nk]
            taus.append(float((tr[:, t_pre:].mean() - tr[:, :t_pre].mean(axis=0) @ lam) - om @ (co[:, t_pre:].mean(axis=1) - co[:, :t_pre] @ lam)))
        taus = np.array(taus)
        se_value = float(np.sqrt((len(taus) - 1) / len(taus) * np.sum((taus - taus.mean()) ** 2)))
    elif method == "placebo":
        rng = make_rng(seed)
        n_tr = int(treated.sum())
        if n_co <= n_tr:
            raise SpecificationError("placebo standard errors need more controls than treated units")
        taus = []
        Yco = Ys[:n_co]
        for _ in range(int(placebos)):
            pick = rng.permutation(n_co)
            taus.append(_estimates(np.vstack([Yco[pick[n_tr:]], Yco[pick[:n_tr]]]), n_co - n_tr, t_pre)["tau"])
        se_value = float(np.std(taus, ddof=1))
    else:
        raise SpecificationError("se is 'auto', 'jackknife' or 'placebo'")
    units = Y.index[order]
    periods = Y.columns
    unit_weights = pd.Series(est["omega"], index=units[:n_co], name="omega")
    time_weights = pd.Series(est["lambda"], index=periods[:t_pre], name="lambda")
    tr_path = Ys[n_co:].mean(axis=0)
    co_path = est["omega"] @ Ys[:n_co]
    shift = (tr_path[:t_pre] - co_path[:t_pre]) @ est["lambda"]
    paths = pd.DataFrame({"treated": tr_path, "synthetic": co_path + shift}, index=periods)
    return SyntheticDiD(outcome, est["tau"], se_value, est["did"], est["sc"], unit_weights, time_weights, method,
                        tuple(units[n_co:]), periods[t_pre], est["zeta"], paths)
