"""Volatility surfaces: SVI and SABR smiles for each expiry, and the Heston model for the whole surface.

    surf = vol_surface(chain, "price", "strike", "expiry", spot=100, rate=0.03, dividend=0.01, kind="cp", model="svi")
    surf.summary(); surf.arbitrage(); surf.vol(105, 0.5); surf.plot()
    heston_price(100, 105, 0.5, 0.03, 0.01, v0=0.04, kappa=2, theta=0.04, sigma=0.5, rho=-0.7)

The quotes' implied volatilities (Black-Scholes) are fitted, with k = ln(K / F) the log forward moneyness and
F = S e^((r - q) T):
    - **SVI** (Gatheral 2004), for each expiry, total variance w(k) = sigma_imp^2 T = a + b (rho (k - m) + sqrt((k - m)^2
      + s^2)), by least squares, with b >= 0, |rho| < 1, s > 0 and a + b s sqrt(1 - rho^2) >= 0 (w never negative).
      ``arbitrage()`` checks Gatheral's butterfly condition g(k) >= 0 over the quotes' range, and that total variance rises with
      expiry at every k (no calendar arbitrage).
    - **SABR** (Hagan et al. 2002), for each expiry, alpha, rho and nu by least squares on Hagan's lognormal implied
      volatility, with ``beta`` fixed (1 by default, as for equity; 0.5 is common for rates).
    - **Heston** (1993), one set of v0, kappa, theta, sigma (the volatility of variance) and rho for every expiry, by
      least squares on price errors divided by each quote's Black-Scholes vega (close to implied-volatility errors, without
      solving for them at every step), from several starting points; prices by Lewis's (2000) single integral of the characteristic function
      (in Albrecher et al.'s 2007 form, which avoids the branch-cut trap).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import optimize

from pymetrix.extension import api
from pymetrix.errors import SpecificationError
from pymetrix.output import Table
from pymetrix.extension import plugin_context
from . import PLUGIN
from .options import black_scholes, implied_vol

if TYPE_CHECKING:
    from pymetrix.data import Dataset
    from pymetrix.figures import Figure

# -- SVI -----------------------------------------------------------------------------------------------------------


def svi_total_variance(k: Any, a: float, b: float, rho: float, m: float, s: float) -> np.ndarray:
    k = np.asarray(k, float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))


def svi_g(k: Any, a: float, b: float, rho: float, m: float, s: float) -> np.ndarray:
    """Gatheral's butterfly density factor: g(k) >= 0 everywhere means no butterfly arbitrage."""
    k = np.asarray(k, float)
    root = np.sqrt((k - m) ** 2 + s**2)
    w = a + b * (rho * (k - m) + root)
    w1 = b * (rho + (k - m) / root)
    w2 = b * s**2 / root**3
    return (1 - k * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2


def fit_svi(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    """SVI parameters (a, b, rho, m, s) fitted to total variances ``w`` at log moneyness ``k``."""
    def resid(p: np.ndarray) -> np.ndarray:
        return svi_total_variance(k, *p) - w

    def floor(p: np.ndarray) -> float:  # the smallest total variance, at its minimum, is not negative
        a, b, rho, _, s = p
        return a + b * s * np.sqrt(max(1 - rho**2, 0.0))

    best = None
    for m0 in (0.0, float(k[np.argmin(w)])):
        for s0 in (0.05, 0.2):
            p0 = np.array([max(w.min() / 2, 1e-6), 0.1, -0.3, m0, s0])
            found = optimize.minimize(lambda p: float(resid(p) @ resid(p)), p0, method="SLSQP",
                                      bounds=[(-1.0, max(w.max(), 1e-4) * 2), (1e-8, 10.0), (-0.999, 0.999), (-2.0, 2.0), (1e-4, 5.0)],
                                      constraints=[{"type": "ineq", "fun": floor}], options={"maxiter": 2000, "ftol": 1e-16})
            if best is None or found.fun < best.fun:
                best = found
    return best.x


# -- SABR ----------------------------------------------------------------------------------------------------------


def sabr_vol(K: Any, f: float, T: float, alpha: float, beta: float, rho: float, nu: float) -> np.ndarray:
    """Hagan's lognormal implied volatility of the SABR model."""
    K = np.asarray(K, float)
    fk = f * K
    lfk = np.log(f / K)
    one = 1 - beta
    lead = alpha / (fk ** (one / 2) * (1 + one**2 / 24 * lfk**2 + one**4 / 1920 * lfk**4))
    z = nu / alpha * fk ** (one / 2) * lfk
    with np.errstate(divide="ignore", invalid="ignore"):
        xz = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
        ratio = np.where(np.abs(z) > 1e-10, z / xz, 1.0 - rho * z / 2)
    tail = 1 + (one**2 * alpha**2 / (24 * fk**one) + rho * beta * nu * alpha / (4 * fk ** (one / 2)) + (2 - 3 * rho**2) * nu**2 / 24) * T
    return lead * ratio * tail


def fit_sabr(K: np.ndarray, vols: np.ndarray, f: float, T: float, beta: float) -> np.ndarray:
    """SABR (alpha, rho, nu) fitted to implied volatilities at strikes ``K``, with ``beta`` fixed."""
    atm = float(np.interp(f, np.sort(K), vols[np.argsort(K)]))

    def resid(p: np.ndarray) -> np.ndarray:
        return sabr_vol(K, f, T, p[0], beta, p[1], p[2]) - vols

    found = optimize.least_squares(resid, [atm * f ** (1 - beta), -0.3, 0.5], bounds=([1e-6, -0.999, 1e-6], [10.0 * max(f, 1) ** (1 - beta), 0.999, 10.0]),
                                   xtol=1e-14, ftol=1e-14)
    return found.x


# -- Heston --------------------------------------------------------------------------------------------------------

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(200)


def heston_cf(u: np.ndarray, T: float, v0: float, kappa: float, theta: float, sigma: float, rho: float) -> np.ndarray:
    """The characteristic function of ln(S_T / F) under Heston (Albrecher et al.'s form)."""
    iu = 1j * u
    b = kappa - rho * sigma * iu
    d = np.sqrt(b**2 + sigma**2 * (iu + u**2))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = kappa * theta / sigma**2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (b - d) / sigma**2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * v0)


def heston_price(S: float, K: Any, T: float, r: float, q: float, v0: float, kappa: float, theta: float, sigma: float, rho: float,
                 kind: str = "call", upper: float = 200.0) -> np.ndarray:
    """Heston prices of European options at strikes ``K`` by Lewis's single integral (Gauss-Legendre on [0, upper])."""
    K = np.atleast_1d(np.asarray(K, float))
    u = (_NODES + 1) * upper / 2
    wts = _WEIGHTS * upper / 2
    F = S * np.exp((r - q) * T)
    k = np.log(F / K)  # log forward moneyness, per strike
    phi = heston_cf(u - 0.5j, T, v0, kappa, theta, sigma, rho)
    integrand = np.real(np.exp(1j * np.outer(k, u)) * phi) / (u**2 + 0.25)
    call = S * np.exp(-q * T) - np.sqrt(F * K) * np.exp(-r * T) / np.pi * (integrand @ wts)
    if kind == "call":
        return call
    return call - S * np.exp(-q * T) + K * np.exp(-r * T)


# -- the fitted surface --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class FittedVolSurface:
    """A model of the implied volatility surface fitted to option quotes."""

    model: str
    params: pd.DataFrame  # by expiry (SVI, SABR) or one row (Heston)
    quotes: pd.DataFrame = field(repr=False)  # strike, expiry, k, market vol, fitted vol
    spot: float = 100.0
    rate: float = 0.0
    dividend: float = 0.0
    beta: float = 1.0
    at_bounds: tuple[str, ...] = ()  # parameters the fit left at a bound

    stage = "finance"

    def _forward(self, T: Any) -> Any:
        return self.spot * np.exp((self.rate - self.dividend) * np.asarray(T, float))

    def vol(self, strike: Any, expiry: float) -> np.ndarray:
        """The model's implied volatility at ``strike`` and ``expiry`` (SVI and SABR: total variance interpolated
        linearly in expiry between the fitted ones, and flat beyond them)."""
        K = np.atleast_1d(np.asarray(strike, float))
        T = float(expiry)
        if self.model == "heston":
            p = self.params.iloc[0]
            prices = heston_price(self.spot, K, T, self.rate, self.dividend, p["v0"], p["kappa"], p["theta"], p["sigma"], p["rho"])
            return np.array([implied_vol(c, self.spot, k_, T, self.rate, self.dividend, "call") for c, k_ in zip(prices, K)])
        ts = self.params.index.to_numpy(float)
        tot = np.column_stack([self._total_at(K, t) for t in ts])
        w = np.array([np.interp(T, ts, row) if T <= ts[-1] else row[-1] * T / ts[-1] for row in tot])
        return np.sqrt(np.maximum(w, 0) / T)

    def _total_at(self, K: np.ndarray, T: float) -> np.ndarray:
        p = self.params.loc[T]
        F = float(self._forward(T))
        if self.model == "svi":
            return svi_total_variance(np.log(K / F), p["a"], p["b"], p["rho"], p["m"], p["s"])
        return sabr_vol(K, F, T, p["alpha"], self.beta, p["rho"], p["nu"]) ** 2 * T

    def arbitrage(self) -> Table:
        """SVI: the butterfly condition's minimum on each expiry's smile, and calendar violations between expiries."""
        if self.model != "svi":
            raise SpecificationError("the arbitrage checks are for SVI fits")
        ks = self.quotes["k"]
        grid = np.linspace(ks.min() - 0.1, ks.max() + 0.1, 301)  # where the quotes are: SVI's wings beyond are extrapolation
        rows, prev, names = [], None, []
        for T, p in self.params.iterrows():
            g = svi_g(grid, p["a"], p["b"], p["rho"], p["m"], p["s"])
            w = svi_total_variance(grid, p["a"], p["b"], p["rho"], p["m"], p["s"])
            calendar = float(np.min(w - prev)) if prev is not None else np.nan
            rows.append([float(g.min()), calendar, "yes" if g.min() >= 0 and (prev is None or calendar >= -1e-10) else "no"])
            names.append(f"{T:g}")
            prev = w
        return Table(["min g(k)", "min rise in total variance", "free of arbitrage"], rows, index=names, index_name="expiry",
                     title="SVI arbitrage checks", notes=(f"Butterfly: g(k) >= 0 for k from {grid[0]:.2f} to {grid[-1]:.2f} (the quotes' range and 0.1 beyond); calendar: total variance "
                                                          "at each k no lower than at the expiry before.",))

    def summary(self) -> Table:
        q = self.quotes
        err = (q["fitted vol"] - q["market vol"]).abs()
        p = self.params
        return Table(list(p.columns), p.to_numpy(float).tolist(), index=[f"{i:g}" if isinstance(i, float) else str(i) for i in p.index],
                     index_name="expiry" if self.model != "heston" else "", title=f"Volatility surface: {self.model.upper() if self.model != 'heston' else 'Heston'}",
                     notes=(f"{len(q)} quotes; implied volatility errors: mean {err.mean():.5f}, largest {err.max():.5f}.",
                            *(("SABR beta fixed at " + f"{self.beta:g}.",) if self.model == "sabr" else ()),
                            *((f"At their bounds: {', '.join(self.at_bounds)}; the model may not fit these quotes.",) if self.at_bounds else ())))

    def plot(self) -> Figure:
        from pymetrix.figures import Figure, Line, Panel, Scatter

        q = self.quotes
        layers: list[Any] = []
        for T, g in q.groupby("expiry"):
            g = g.sort_values("strike")
            layers.append(Scatter(pd.Series(g["strike"].to_numpy()), pd.Series(g["market vol"].to_numpy()), f"market {T:g}"))
            Ks = np.linspace(g["strike"].min(), g["strike"].max(), 60)
            layers.append(Line(pd.Series(self.vol(Ks, T), index=pd.Index(Ks, name="strike")), f"fitted {T:g}"))
        return Figure((Panel(tuple(layers), title="Implied volatility: market and model", xlabel="strike", ylabel="implied volatility"),))

    def grid(self, x: Literal["strike", "moneyness"] = "strike", strikes: int = 25, expiries: int = 12) -> pd.DataFrame:
        """The model's implied volatility over a grid: expiries down the rows (from the first quoted to the last;
        SVI and SABR interpolate total variance between the fitted ones), and across, ``strikes`` strikes over the
        quotes' range, or with ``x="moneyness"`` that many log forward moneyness values k = ln(K / F) over theirs (the
        strike at each expiry K = F e^k)."""
        if x not in ("strike", "moneyness"):
            raise SpecificationError("x is 'strike' or 'moneyness'")
        q = self.quotes
        quoted = np.sort(q["expiry"].unique().astype(float))
        if len(quoted) < 2:
            raise SpecificationError("a volatility surface needs quotes at two or more expiries")
        Ts = np.linspace(quoted[0], quoted[-1], max(2, int(expiries)))
        across = np.linspace(q["strike"].min(), q["strike"].max(), max(2, int(strikes))) if x == "strike" else \
            np.linspace(q["k"].min(), q["k"].max(), max(2, int(strikes)))
        rows = [self.vol(across if x == "strike" else self._forward(T) * np.exp(across), T) for T in Ts]
        return pd.DataFrame(rows, index=pd.Index(Ts, name="expiry"), columns=pd.Index(across, name="strike" if x == "strike" else "log moneyness"))

    def plot_surface(self, x: Literal["strike", "moneyness"] = "strike", strikes: int = 25, expiries: int = 12) -> Figure:
        """The fitted surface in 3-D: implied volatility by strike (or log moneyness) and expiry (``grid``)."""
        from pymetrix.figures import Figure, Panel, Surface

        values = self.grid(x, strikes, expiries)
        name = "Heston" if self.model == "heston" else self.model.upper()
        return Figure((Panel((Surface(values, label="implied volatility"),), title=f"Implied volatility surface ({name})",
                             xlabel="strike" if x == "strike" else "log moneyness, ln(K / F)", ylabel="expiry (years)", legend=False),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return self.quotes

    def __repr__(self) -> str:
        return f"<FittedVolSurface {self.model} of {len(self.quotes)} quotes>"


with plugin_context(PLUGIN):
    @api("finance:vol_surface")
    def vol_surface(data: Dataset, price: str, strike: str, expiry: str, spot: float, rate: float = 0.0, dividend: float = 0.0,
                    kind: str = "call", model: Literal["svi", "sabr", "heston"] = "svi", beta: float = 1.0) -> FittedVolSurface:
        """An SVI, SABR (each expiry) or Heston (the whole surface) model fitted to option quotes' implied volatilities."""
        from .options import implied_volatility

        if model not in ("svi", "sabr", "heston"):
            raise SpecificationError("model is 'svi', 'sabr' or 'heston'")
        smile = implied_volatility(data, price, strike, expiry, spot=spot, rate=rate, dividend=dividend, kind=kind)
        q = smile.frame.dropna(subset=["implied vol"]).copy()
        if len(q) < 5:
            raise SpecificationError("too few quotes with an implied volatility")
        S, r, d = float(spot), float(rate), float(dividend)
        q["k"] = np.log(q["strike"] / (S * np.exp((r - d) * q["expiry"])))
        q = q.rename(columns={"implied vol": "market vol"})
        rows, fitted, at_bounds = {}, pd.Series(np.nan, index=q.index), ()
        if model in ("svi", "sabr"):
            for T, g in q.groupby("expiry"):
                if len(g) < (5 if model == "svi" else 3):
                    raise SpecificationError(f"expiry {T:g} has too few quotes for {model.upper()}")
                if model == "svi":
                    p = fit_svi(g["k"].to_numpy(), g["market vol"].to_numpy() ** 2 * T)
                    rows[float(T)] = dict(zip(("a", "b", "rho", "m", "s"), p))
                    fitted[g.index] = np.sqrt(np.maximum(svi_total_variance(g["k"].to_numpy(), *p), 0) / T)
                else:
                    F = S * np.exp((r - d) * T)
                    p = fit_sabr(g["strike"].to_numpy(), g["market vol"].to_numpy(), F, float(T), float(beta))
                    rows[float(T)] = dict(zip(("alpha", "rho", "nu"), p))
                    fitted[g.index] = sabr_vol(g["strike"].to_numpy(), F, float(T), p[0], beta, p[1], p[2])
            params = pd.DataFrame.from_dict(rows, orient="index")
            params.index.name = "expiry"
        else:
            K, T, mv = q["strike"].to_numpy(), q["expiry"].to_numpy(), q["market vol"].to_numpy()
            expiries = np.unique(T)

            def model_vols(p: np.ndarray) -> np.ndarray:
                out = np.empty(len(K))
                for t in expiries:
                    sel = T == t
                    c = heston_price(S, K[sel], t, r, d, *p)
                    out[sel] = [implied_vol(ci, S, ki, t, r, d, "call") for ci, ki in zip(c, K[sel])]
                return out

            from scipy.stats import norm

            calls = black_scholes(S, K, T, r, mv, d)  # the quotes as calls (parity), and their vegas
            F = S * np.exp((r - d) * T)
            d1 = (np.log(F / K) + mv**2 / 2 * T) / (mv * np.sqrt(T))
            vega = S * np.exp(-d * T) * norm.pdf(d1) * np.sqrt(T)

            def resid(p: np.ndarray) -> np.ndarray:
                out = np.empty(len(K))
                for t in expiries:
                    sel = T == t
                    out[sel] = heston_price(S, K[sel], t, r, d, *p)
                return (out - calls) / np.maximum(vega, 1e-8)

            atm = float(np.median(mv)) ** 2
            found = None
            for kappa0, sigma0 in ((1.0, 0.3), (3.0, 0.6), (1.0, 1.0), (5.0, 0.4)):
                trial = optimize.least_squares(resid, [atm, kappa0, atm, sigma0, -0.5], bounds=([1e-5, 1e-3, 1e-5, 1e-3, -0.999], [2.0, 20.0, 2.0, 5.0, 0.999]),
                                               xtol=1e-12, ftol=1e-12, max_nfev=300)
                if found is None or trial.cost < found.cost:
                    found = trial
            params = pd.DataFrame([dict(zip(("v0", "kappa", "theta", "sigma", "rho"), found.x))], index=["Heston"])
            lo, hi = np.array([1e-5, 1e-3, 1e-5, 1e-3, -0.999]), np.array([2.0, 20.0, 2.0, 5.0, 0.999])
            edge = (np.abs(found.x - lo) < 1e-6 * (hi - lo)) | (np.abs(found.x - hi) < 1e-6 * (hi - lo)) | (found.x < lo * 20)
            at_bounds = tuple(n for n, e in zip(params.columns, edge) if e)
            fitted[:] = model_vols(found.x)
        q["fitted vol"] = fitted
        return FittedVolSurface(model, params, q[["strike", "expiry", "k", "kind", "price", "market vol", "fitted vol"]], S, r, d, float(beta),
                                at_bounds)


with plugin_context(PLUGIN):  # a recorded call: the Heston option price dialog (app.py) builds it
    heston_price = api("finance:heston_price")(heston_price)
