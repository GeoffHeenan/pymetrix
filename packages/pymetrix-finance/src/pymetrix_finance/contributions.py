"""Each asset's share of a portfolio's value at risk and expected shortfall: marginal and component VaR and ES.

    rc = risk_contributions(data, ["a", "b", "c"], [0.5, 0.3, 0.2], level=0.99)            # normal, sample covariance
    rc = risk_contributions(data, ["a", "b"], [0.6, 0.4], cov=mgarch_fit)                  # next period's GARCH covariance
    rc = risk_contributions(data, ["a", "b", "c"], [0.5, 0.3, 0.2], method="historical")
    rc.table(); rc.plot()

VaR and ES are losses, positive numbers, of the portfolio's return per period. The marginal VaR of an asset is how
VaR changes with its weight; its component VaR is its weight times that, and the components add up to VaR (Euler's
theorem: VaR is homogeneous of degree one in the weights). Under the normal, with mean m and covariance S,
    marginal VaR_i = -m_i + z (S w)_i / sd,   marginal ES_i = -m_i + phi(z) / p (S w)_i / sd,   sd = sqrt(w' S w).
Historically, component ES_i is -w_i times asset i's average return in the periods the portfolio lost more than VaR,
which add up to the historical ES exactly; component VaR_i is -w_i times its average return in the periods nearest
the VaR (the closest 1% of the sample, and at least five), scaled to add up to VaR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from pymetrix.extension import api
from pymetrix.errors import SpecificationError
from pymetrix.output import Table
from pymetrix.extension import plugin_context
from . import PLUGIN
from .risk import historical, normal

if TYPE_CHECKING:
    from pymetrix.data import Dataset
    from pymetrix.figures import Figure


def normal_contributions(w: np.ndarray, mu: np.ndarray, S: np.ndarray, p: float) -> dict[str, np.ndarray]:
    """Marginal and component VaR and ES under the normal (``p`` the tail probability, 0.01 for 99%)."""
    sd = float(np.sqrt(w @ S @ w))
    z = sps.norm.ppf(p)
    beta = S @ w / sd
    mvar, mes = -mu - z * beta, -mu + sps.norm.pdf(z) / p * beta
    return {"marginal VaR": mvar, "component VaR": w * mvar, "marginal ES": mes, "component ES": w * mes}


def historical_contributions(w: np.ndarray, R: np.ndarray, p: float) -> dict[str, np.ndarray]:
    """Component VaR and ES from the periods near and beyond the portfolio's historical VaR."""
    rp = R @ w
    var, es = historical(rp, p)
    tail = rp <= -var
    ces = -w * R[tail].mean(axis=0)
    k = max(5, int(round(0.01 * len(rp))))
    near = np.argsort(np.abs(rp + var))[:k]
    cvar = -w * R[near].mean(axis=0)
    cvar = cvar * var / cvar.sum() if cvar.sum() != 0 else cvar
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"marginal VaR": np.where(w != 0, cvar / w, np.nan), "component VaR": cvar,
                "marginal ES": np.where(w != 0, ces / w, np.nan), "component ES": ces}


@dataclass(frozen=True, eq=False)
class RiskContributions:
    """Each asset's marginal and component VaR and ES in a portfolio."""

    assets: tuple[str, ...]
    weights: np.ndarray
    level: float
    method: str
    var: float
    es: float
    parts: dict[str, np.ndarray] = field(repr=False)
    source: str = "sample"

    stage = "finance"

    def frame(self) -> pd.DataFrame:
        f = pd.DataFrame({"weight": self.weights, **self.parts}, index=list(self.assets))
        f["share of VaR"] = f["component VaR"] / self.var
        f["share of ES"] = f["component ES"] / self.es
        return f[["weight", "marginal VaR", "component VaR", "share of VaR", "marginal ES", "component ES", "share of ES"]]

    def table(self) -> Table:
        f = self.frame()
        how = ("normal, with the " + self.source + " covariance") if self.method == "normal" else "historical simulation"
        return Table(list(f.columns), f.to_numpy(float).tolist(), index=list(f.index), index_name="asset",
                     title=f"Risk contributions at {self.level:.1%}",
                     notes=(f"Portfolio VaR {self.var:.6g} and ES {self.es:.6g} per period ({how}).",
                            "Components add up to the portfolio's VaR and ES; marginal values are per unit of weight."))

    def summary(self) -> Table:
        return self.table()

    def plot(self) -> Figure:
        from pymetrix.figures import Bars, Figure, Panel

        f = self.frame()
        return Figure((Panel((Bars(f["share of VaR"], "VaR"), Bars(f["share of ES"], "ES")), title="Share of the portfolio's risk"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return self.frame()

    def __repr__(self) -> str:
        return f"<RiskContributions of {', '.join(self.assets)}: VaR {self.var:.4g}, ES {self.es:.4g}>"


with plugin_context(PLUGIN):
    @api("finance:risk_contributions")
    def risk_contributions(data: Dataset, assets: Sequence[str], weights: Sequence[float], level: float = 0.99,
                           method: Literal["normal", "historical"] = "normal", cov: Any = None) -> RiskContributions:
        """Marginal and component VaR and ES of each asset in a portfolio with ``weights``, normal (with the sample
        covariance, a given one, or a fitted multivariate GARCH's next-period forecast) or historical."""
        assets = [assets] if isinstance(assets, str) else list(assets)
        w = np.asarray(weights, float)
        missing = [a for a in assets if a not in data.names]
        if missing:
            raise SpecificationError(f"no series {', '.join(map(repr, missing))} in the dataset")
        if w.shape != (len(assets),):
            raise SpecificationError(f"give one weight for each of the {len(assets)} assets")
        if not 0.5 <= level < 1:
            raise SpecificationError("level is the confidence, from 0.5 to below 1: 0.99 for 99% VaR")
        if method not in ("normal", "historical"):
            raise SpecificationError("method is 'normal' or 'historical'")
        R = data.frame[assets].dropna().to_numpy(float)
        p = 1 - level
        source = "sample"
        if method == "historical":
            if cov is not None:
                raise SpecificationError("a covariance is used only by the normal method")
            if p * len(R) < 1:
                raise SpecificationError(f"{len(R)} periods are too few for a historical {level:.1%} VaR")
            var, es = historical(R @ w, p)
            parts = historical_contributions(w, R, p)
        else:
            mu = R.mean(axis=0)
            if cov is None:
                S = np.cov(R.T, ddof=1)
            elif hasattr(cov, "forecast") and hasattr(cov, "variables"):
                if list(cov.variables) != assets:
                    raise SpecificationError(f"the multivariate GARCH is of {', '.join(cov.variables)}; the assets must match, in order")
                S, mu, source = cov.forecast(1)[0], np.asarray(cov.mu, float), f"{cov.model} forecast"
            else:
                S, source = np.asarray(getattr(cov, "values", cov), float), "given"
            S = np.atleast_2d(S)
            if S.shape != (len(assets), len(assets)):
                raise SpecificationError(f"the covariance must be {len(assets)} x {len(assets)}")
            var, es = normal(float(w @ mu), float(np.sqrt(w @ S @ w)), p)
            parts = normal_contributions(w, mu, S, p)
        return RiskContributions(tuple(assets), w, float(level), method, float(var), float(es), parts, source)
