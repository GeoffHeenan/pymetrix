"""Credit risk: Merton's distance to default from equity prices, and default intensities from CDS spreads.

    m = merton(data, equity="mcap", debt="debt", rate="rf")             # assets, their volatility, DD and PD by period
    m.summary(); m.frame; m.plot()
    merton_point(equity=30, equity_vol=0.5, debt=80, rate=0.03)         # one firm at one date
    cds = cds_curve([1, 3, 5, 7, 10], [0.008, 0.011, 0.014, 0.015, 0.016], recovery=0.4, rate=0.03)

**Merton (1974).** Equity is a call on the firm's assets V with strike the debt D due at horizon T:
E = V N(d1) - D e^(-rT) N(d2), and equity volatility sigma_E E = N(d1) sigma_V V. Given E, sigma_E, D and r, the two
equations give V and sigma_V (``merton_point``). Over time (``merton``), sigma_V is found as in KMV and Vassalou and
Xing (2004): start from the equity volatility, solve each period's V from the first equation, set sigma_V to the
annualised volatility of the log changes of V, and repeat until it settles. The distance to default is
DD = (ln(V / D) + (mu - sigma_V^2 / 2) T) / (sigma_V sqrt(T)), with mu the assets' drift (the rate, risk-neutral, unless
``drift="assets"``, their mean log change), and the probability of default N(-DD). The default point is often taken
as short-term debt plus half the long-term: give that series as ``debt``.

**CDS** (``cds_curve``). A piecewise-constant default intensity is bootstrapped so that each maturity's CDS, paying
its spread each quarter (with the premium accrued to default) for protection of 1 - recovery, has zero value; the
survival probability to t is exp(-integral of the intensity). With a flat curve the intensity is close to the
"credit triangle", spread / (1 - recovery).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

from pymetrix.extension import api
from pymetrix.errors import SpecificationError
from pymetrix.output import Table
from pymetrix.extension import plugin_context
from . import PLUGIN
from .options import black_scholes

if TYPE_CHECKING:
    from pymetrix.data import Dataset
    from pymetrix.figures import Figure


def _assets_from_equity(E: float, D: float, r: float, sv: float, T: float) -> float:
    """The asset value whose call at strike D is worth E."""
    hi = E + D * 2 + 1.0
    return float(optimize.brentq(lambda v: black_scholes(v, D, T, r, sv) - E, E, hi * 10, xtol=1e-12 * max(E, 1.0)))


def merton_solve(E: float, sigma_e: float, D: float, r: float, T: float = 1.0) -> tuple[float, float]:
    """Asset value and volatility from equity value and volatility (Merton's two equations)."""
    def eqs(x: np.ndarray) -> list[float]:
        V, sv = np.exp(x)
        d1 = (np.log(V / D) + (r + sv**2 / 2) * T) / (sv * np.sqrt(T))
        return [black_scholes(V, D, T, r, sv) / E - 1, norm.cdf(d1) * sv * V / (sigma_e * E) - 1]

    start = np.log([E + D * np.exp(-r * T), sigma_e * E / (E + D * np.exp(-r * T))])
    sol = optimize.root(eqs, start, method="hybr", options={"xtol": 1e-13})
    if not sol.success:
        raise SpecificationError(f"Merton's equations have no solution here: {sol.message}")
    V, sv = np.exp(sol.x)
    return float(V), float(sv)


def distance_to_default(V: Any, D: Any, sigma_v: float, mu: Any, T: float = 1.0) -> Any:
    return (np.log(np.asarray(V) / np.asarray(D)) + (np.asarray(mu) - sigma_v**2 / 2) * T) / (sigma_v * np.sqrt(T))


def merton_point(equity: float, equity_vol: float, debt: float, rate: float, horizon: float = 1.0) -> dict[str, float]:
    """One firm at one date: asset value and volatility, distance to default and probability of default (risk-neutral)."""
    if min(equity, equity_vol, debt, horizon) <= 0:
        raise SpecificationError("equity, its volatility, debt and the horizon must be positive")
    V, sv = merton_solve(float(equity), float(equity_vol), float(debt), float(rate), float(horizon))
    dd = float(distance_to_default(V, debt, sv, rate, horizon))
    return {"assets": V, "asset vol": sv, "leverage": float(debt) / V, "DD": dd, "PD": float(norm.cdf(-dd))}


@dataclass(frozen=True, eq=False)
class MertonModel:
    """Asset values, asset volatility, distances to default and default probabilities by period."""

    frame: pd.DataFrame  # equity, debt, assets, DD, PD
    asset_vol: float
    iterations: int
    horizon: float
    drift: str
    per_year: int

    stage = "finance"

    def summary(self) -> Table:
        f = self.frame
        last = f.iloc[-1]
        rows = [["asset volatility", self.asset_vol], ["latest assets", last["assets"]], ["latest leverage", last["debt"] / last["assets"]],
                ["latest DD", last["DD"]], ["latest PD", last["PD"]], ["lowest DD", f["DD"].min()], ["highest PD", f["PD"].max()]]
        return Table(["value"], [[float(v)] for _, v in rows], index=[r[0] for r in rows], index_name="", title="Merton model",
                     notes=(f"Asset volatility by the KMV iteration ({self.iterations} rounds); horizon {self.horizon:g} years; drift: "
                            f"{'the risk-free rate (risk-neutral)' if self.drift == 'rate' else 'the assets mean log change'}.",
                            f"{len(f)} periods, {self.per_year} a year; PD = N(-DD)."))

    def plot(self) -> Figure:
        from pymetrix.figures import Figure, Line, Panel

        f = self.frame
        return Figure((Panel((Line(f["DD"], "distance to default"),), title="Distance to default"),
                       Panel((Line(f["PD"], "probability of default"),), title=f"Probability of default within {self.horizon:g} year(s)")))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return self.frame

    def __repr__(self) -> str:
        return f"<MertonModel: asset volatility {self.asset_vol:.4f}, latest PD {self.frame['PD'].iloc[-1]:.4g}>"


def _premium_and_protection(times: np.ndarray, hazard_at: Any, r: float, R: float) -> tuple[np.ndarray, np.ndarray]:
    """For each maturity in ``times`` (multiples of a quarter), the premium leg per unit spread (with accrual) and the
    protection leg, on a monthly grid."""
    grid = np.arange(1, int(round(times.max() * 12)) + 1) / 12
    lam = np.array([hazard_at(t - 1 / 24) for t in grid])
    surv = np.exp(-np.cumsum(lam / 12))
    prev = np.concatenate([[1.0], surv[:-1]])
    disc = np.exp(-r * grid)
    default = prev - surv
    protection = np.cumsum((1 - R) * disc * default)
    # premium: quarterly payments to survivors, plus half a month's accrual (on average) on default
    pay = np.where(np.isclose((grid * 4) % 1, 0) | np.isclose((grid * 4) % 1, 1), 0.25 * disc * surv, 0.0)
    accrual = disc * default * ((grid * 12 - 1) % 3 + 0.5) / 12
    premium = np.cumsum(pay + accrual)
    k = np.rint(times * 12).astype(int) - 1
    return premium[k], protection[k]


def bootstrap_hazards(maturities: Sequence[float], spreads: Sequence[float], recovery: float = 0.4, rate: float = 0.0) -> np.ndarray:
    """Piecewise-constant default intensities, one per maturity band, that price each CDS at zero."""
    m = np.asarray(maturities, float)
    s = np.asarray(spreads, float)
    lam: list[float] = []

    def hazard_at(t: float, lam_list: list[float], last: float) -> float:
        for j, mj in enumerate(m[:len(lam_list)]):
            if t < mj:
                return lam_list[j]
        return last

    for i in range(len(m)):
        def value(x: float) -> float:
            prem, prot = _premium_and_protection(m[:i + 1], lambda t: hazard_at(t, lam, x), rate, recovery)
            return float(s[i] * prem[-1] - prot[-1])

        lam.append(float(optimize.brentq(value, 1e-10, 10.0, xtol=1e-14)))
    return np.array(lam)


@dataclass(frozen=True, eq=False)
class CDSCurve:
    """Default intensities bootstrapped from a CDS curve, and the survival and default probabilities they imply."""

    maturities: np.ndarray
    spreads: np.ndarray
    hazards: np.ndarray
    recovery: float
    rate: float

    stage = "finance"

    def hazard(self, t: Any) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, float))
        idx = np.minimum(np.searchsorted(self.maturities, t, side="right"), len(self.maturities) - 1)
        return self.hazards[idx]

    def survival(self, t: Any) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, float))
        edges = np.concatenate([[0.0], self.maturities])
        out = []
        for x in t:
            total, start = 0.0, 0.0
            for j, end in enumerate(edges[1:]):
                total += self.hazards[j] * max(0.0, min(x, end) - start)
                start = end
                if x <= end:
                    break
            if x > edges[-1]:
                total += self.hazards[-1] * (x - edges[-1])
            out.append(np.exp(-total))
        return np.array(out)

    def table(self) -> Table:
        rows = [[float(m), float(s) * 1e4, float(h), float(s) / (1 - self.recovery), float(1 - self.survival(m)[0])]
                for m, s, h in zip(self.maturities, self.spreads, self.hazards)]
        return Table(["maturity", "spread (bp)", "intensity", "credit triangle", "default probability"], rows,
                     index=[f"{m:g}y" for m in self.maturities], index_name="CDS", title="CDS curve",
                     notes=(f"Intensities constant between maturities, bootstrapped with recovery {self.recovery:g} and rate "
                            f"{self.rate:g}; quarterly premiums with accrual on default.",))

    def summary(self) -> Table:
        return self.table()

    def plot(self) -> Figure:
        from pymetrix.figures import Figure, Line, Panel

        t = np.linspace(0.01, self.maturities.max(), 200)
        return Figure((Panel((Line(pd.Series(1 - self.survival(t), index=pd.Index(t, name="years")), "default probability"),),
                             title="Cumulative default probability", xlabel="years"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return self.table().to_frame()

    def __repr__(self) -> str:
        return f"<CDSCurve of {len(self.maturities)} maturities>"


with plugin_context(PLUGIN):
    @api("finance:merton")
    def merton(data: Dataset, equity: str, debt: str, rate: str | float = 0.0, horizon: float = 1.0,
               drift: Literal["rate", "assets"] = "rate", periods_per_year: int | None = None, tol: float = 1e-6,
               max_iter: int = 200) -> MertonModel:
        """Merton's model over time: asset values and volatility (the KMV iteration) from the equity value ``equity``
        and the default point ``debt``, with distances to default and default probabilities over ``horizon`` years."""
        from .returns import annual_periods

        names = [equity, debt, *([rate] if isinstance(rate, str) else [])]
        missing = [s for s in names if s not in data.names]
        if missing:
            raise SpecificationError(f"no series {', '.join(map(repr, missing))} in the dataset")
        if data.time_axis is None or data.kind != "time_series":
            raise SpecificationError("the Merton model over time needs equity and debt over time")
        frame = data.frame[list(dict.fromkeys(names))].dropna()
        if len(frame) < 20:
            raise SpecificationError("too few periods to estimate the assets' volatility")
        E, D = frame[equity].to_numpy(float), frame[debt].to_numpy(float)
        if (E <= 0).any() or (D <= 0).any():
            raise SpecificationError("equity and debt must be positive")
        r = frame[rate].to_numpy(float) if isinstance(rate, str) else np.full(len(E), float(rate))
        q = int(periods_per_year) if periods_per_year else annual_periods(frame.index)
        sv = float(np.std(np.diff(np.log(E)), ddof=1) * np.sqrt(q)) * float(np.mean(E / (E + D)))
        sv = max(sv, 1e-3)
        for it in range(1, int(max_iter) + 1):
            V = np.array([_assets_from_equity(E[t], D[t], r[t], sv, horizon) for t in range(len(E))])
            new = float(np.std(np.diff(np.log(V)), ddof=1) * np.sqrt(q))
            if abs(new - sv) < tol:
                sv = new
                break
            sv = new
        V = np.array([_assets_from_equity(E[t], D[t], r[t], sv, horizon) for t in range(len(E))])
        mu = r if drift == "rate" else np.full(len(E), float(np.mean(np.diff(np.log(V))) * q + sv**2 / 2))
        dd = distance_to_default(V, D, sv, mu, horizon)
        out = pd.DataFrame({"equity": E, "debt": D, "assets": V, "DD": dd, "PD": norm.cdf(-dd)}, index=frame.index)
        return MertonModel(out, sv, it, float(horizon), drift, q)

    @api("finance:cds_curve")
    def cds_curve(maturities: Sequence[float], spreads: Sequence[float], recovery: float = 0.4, rate: float = 0.0) -> CDSCurve:
        """Default intensities bootstrapped from CDS ``spreads`` (fractions a year: 0.01 for 100 basis points) at
        ``maturities`` (years, in quarters), with ``recovery`` and a flat continuously compounded ``rate``."""
        m, s = np.asarray(maturities, float), np.asarray(spreads, float)
        if m.shape != s.shape or len(m) == 0 or np.any(np.diff(m) <= 0) or np.any(m <= 0):
            raise SpecificationError("give rising maturities, each with a spread")
        if not np.allclose(m * 4, np.round(m * 4)):
            raise SpecificationError("maturities are whole quarters (0.25, 0.5, 1, ...)")
        if np.any(s <= 0) or not 0 <= recovery < 1:
            raise SpecificationError("spreads must be positive and the recovery from 0 to below 1")
        return CDSCurve(m, s, bootstrap_hazards(m, s, recovery, rate), float(recovery), float(rate))


with plugin_context(PLUGIN):  # a recorded call: the Merton at a point dialog (app.py) builds it
    merton_point = api("finance:merton_point")(merton_point)
