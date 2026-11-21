"""Option prices, their Greeks, implied volatilities and volatility smiles.

    option_price(spot=100, strike=95, expiry=0.5, rate=0.03, vol=0.2)              # Black-Scholes, with the Greeks
    option_price(spot=100, strike=95, expiry=0.5, rate=0.03, vol=0.2, kind="put", american=True)   # a binomial tree
    smile = implied_volatility(quotes, "price", "strike", "expiry", spot=100, rate=0.03)        # one per quote
    smile.plot()                                                                   # by strike, one line per expiry

European options are priced by Black, Scholes and Merton with a continuous dividend yield ``dividend``;
American ones by a Cox-Ross-Rubinstein tree of ``steps`` steps. ``expiry`` is in years, ``rate`` and ``dividend``
continuously compounded per year, ``vol`` the annual volatility. The Greeks are per unit: vega and rho for a change of
1 (100 percentage points) in volatility or the rate, theta per year. Implied volatility solves for the volatility
that gives the quoted price (Brent's method); a price outside the no-arbitrage bounds has none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

from pymetrix.extension import api
from pymetrix.errors import SpecificationError
from pymetrix.output import Table
from pymetrix.extension import plugin_context
from . import PLUGIN

if TYPE_CHECKING:
    from pymetrix.data import Dataset
    from pymetrix.figures import Figure


def _d1d2(S: Any, K: Any, T: Any, r: Any, sigma: Any, q: Any) -> tuple[Any, Any]:
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def black_scholes(S: Any, K: Any, T: Any, r: Any, sigma: Any, q: Any = 0.0, kind: str = "call") -> Any:
    """The Black-Scholes-Merton price of a European call or put (arrays broadcast)."""
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def greeks(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, kind: str = "call") -> dict[str, float]:
    """Delta, gamma, vega, theta (per year) and rho of a European option, in closed form."""
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    dq, dr, pdf = np.exp(-q * T), np.exp(-r * T), norm.pdf(d1)
    gamma = dq * pdf / (S * sigma * np.sqrt(T))
    vega = S * dq * pdf * np.sqrt(T)
    decay = -S * dq * pdf * sigma / (2 * np.sqrt(T))
    if kind == "call":
        delta = dq * norm.cdf(d1)
        theta = decay - r * K * dr * norm.cdf(d2) + q * S * dq * norm.cdf(d1)
        rho = K * T * dr * norm.cdf(d2)
    else:
        delta = -dq * norm.cdf(-d1)
        theta = decay + r * K * dr * norm.cdf(-d2) - q * S * dq * norm.cdf(-d1)
        rho = -K * T * dr * norm.cdf(-d2)
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega), "theta": float(theta), "rho": float(rho)}


def binomial(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, kind: str = "call",
             american: bool = False, steps: int = 500) -> float:
    """The Cox-Ross-Rubinstein tree price, American (exercise at any node) or European."""
    return _tree(S, K, T, r, sigma, q, kind, american, steps)[0]


def _tree(S: float, K: float, T: float, r: float, sigma: float, q: float, kind: str, american: bool,
          steps: int) -> tuple[float, float, float, float]:
    """The tree's price, and its delta, gamma and theta from the nodes of its first two steps (Hull's estimates)."""
    if steps < 3:
        raise SpecificationError("a binomial tree needs at least 3 steps")
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0 < p < 1:
        raise SpecificationError("too few steps for this volatility and rate: the tree's probabilities leave 0 to 1")
    disc = np.exp(-r * dt)
    prices = S * u ** np.arange(steps, -1, -1) * d ** np.arange(0, steps + 1)
    sign = 1.0 if kind == "call" else -1.0
    value = np.maximum(sign * (prices - K), 0.0)
    early: dict[int, np.ndarray] = {}
    for n in range(steps, 0, -1):
        value = disc * (p * value[:-1] + (1 - p) * value[1:])
        if american:
            prices = prices[:-1] / u
            value = np.maximum(value, sign * (prices - K))
        if n - 1 in (1, 2):
            early[n - 1] = value.copy()
    (f11, f10), (f22, f21, f20) = early[1], early[2]
    delta = (f11 - f10) / (S * u - S * d)
    gamma = ((f22 - f21) / (S * u * u - S) - (f21 - f20) / (S - S * d * d)) / (0.5 * (S * u * u - S * d * d))
    return float(value[0]), float(delta), float(gamma), float((f21 - value[0]) / (2 * dt))


def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float = 0.0, kind: str = "call") -> float:
    """The volatility at which Black-Scholes gives ``price`` (nan outside the no-arbitrage bounds)."""
    fwd_s, fwd_k = S * np.exp(-q * T), K * np.exp(-r * T)
    low, high = (max(fwd_s - fwd_k, 0.0), fwd_s) if kind == "call" else (max(fwd_k - fwd_s, 0.0), fwd_k)
    if not (np.isfinite(price) and T > 0 and low < price < high):
        return float("nan")
    return float(optimize.brentq(lambda v: black_scholes(S, K, T, r, v, q, kind) - price, 1e-6, 20.0, xtol=1e-12))


@dataclass(frozen=True, eq=False)
class OptionValuation:
    """An option's price and Greeks, with its inputs."""

    kind: str
    american: bool
    inputs: dict[str, float]
    price: float
    values: dict[str, float]  # the Greeks
    method: str

    stage = "finance"

    def table(self) -> Table:
        rows = [["price", self.price], *[[k, v] for k, v in self.values.items()]]
        style = "American" if self.american else "European"
        notes = (f"{style} {self.kind} by {self.method}; spot {self.inputs['spot']:g}, strike {self.inputs['strike']:g}, "
                 f"expiry {self.inputs['expiry']:g} years, rate {self.inputs['rate']:g}, dividend {self.inputs['dividend']:g}, "
                 f"volatility {self.inputs['vol']:g}.",
                 "Vega and rho per unit change in volatility or the rate; theta per year." if not self.american else
                 "Delta, gamma and theta from the tree's first nodes, vega and rho by re-pricing it; per unit change, theta per year.")
        return Table(["value"], [[float(v)] for _, v in rows], index=[k for k, _ in rows], index_name="", title="Option value",
                     notes=notes)

    def summary(self) -> Table:
        return self.table()

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return self.table().to_frame()

    def __repr__(self) -> str:
        return f"<OptionValuation {self.kind}: {self.price:.6g}>"


@dataclass(frozen=True, eq=False)
class VolatilitySmile:
    """Implied volatilities of option quotes, by strike and expiry."""

    frame: pd.DataFrame = field(repr=False)  # strike, expiry, kind, price, implied vol, moneyness
    spot: Any = None

    stage = "finance"

    def table(self) -> Table:
        f = self.frame
        missing = int(f["implied vol"].isna().sum())
        notes = ["Black-Scholes implied volatility of each quote."]
        if missing:
            notes.append(f"{missing} quote(s) outside the no-arbitrage bounds have none.")
        return Table(list(f.columns), f.to_numpy(object).tolist(), index=[str(i) for i in f.index], index_name="quote",
                     title="Implied volatility", notes=tuple(notes))

    def summary(self) -> Table:
        return self.table()

    def surface(self) -> pd.DataFrame:
        """Implied volatility by strike (rows) and expiry (columns)."""
        return self.frame.pivot_table(index="strike", columns="expiry", values="implied vol")

    def plot(self) -> Figure:
        from pymetrix.figures import Figure, Line, Panel

        s = self.surface()
        lines = tuple(Line(s[c].dropna().rename_axis("strike"), f"expiry {c:g}") for c in s.columns)
        return Figure((Panel(lines, title="Volatility smile", xlabel="strike", ylabel="implied volatility"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return self.frame

    def __repr__(self) -> str:
        return f"<VolatilitySmile of {len(self.frame)} quotes>"


def _tree_greeks(S: float, K: float, T: float, r: float, sigma: float, q: float, kind: str, steps: int) -> tuple[float, dict[str, float]]:
    """An American option's price and Greeks: delta, gamma and theta from the tree's nodes; vega and rho by central
    differences of whole trees (bumps of 0.01 and 0.001, wide enough to rise above the tree's jaggedness)."""
    price, delta, gamma, theta = _tree(S, K, T, r, sigma, q, kind, True, steps)

    def at(r_: float, s_: float) -> float:
        return _tree(S, K, T, r_, s_, q, kind, True, steps)[0]

    vega = (at(r, sigma + 0.01) - at(r, sigma - 0.01)) / 0.02
    rho = (at(r + 0.001, sigma) - at(r - 0.001, sigma)) / 0.002
    return price, {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def _value(data: Dataset, value: Any, what: str) -> Any:
    if isinstance(value, str):
        if value not in data.names:
            raise SpecificationError(f"no series {value!r} in the dataset (for the {what})")
        return data[value].to_numpy(float)
    return float(value)


with plugin_context(PLUGIN):
    @api("finance:option_price")
    def option_price(spot: float, strike: float, expiry: float, rate: float, vol: float, dividend: float = 0.0,
                     kind: Literal["call", "put"] = "call", american: bool = False, steps: int = 500) -> OptionValuation:
        """The price and Greeks of a European option (Black-Scholes-Merton) or an American one (a binomial tree of
        ``steps`` steps)."""
        if kind not in ("call", "put"):
            raise SpecificationError("kind is 'call' or 'put'")
        if spot <= 0 or strike <= 0 or expiry <= 0 or vol <= 0:
            raise SpecificationError("spot, strike, expiry and volatility must be positive")
        S, K, T, r, s, q = map(float, (spot, strike, expiry, rate, vol, dividend))
        inputs = {"spot": S, "strike": K, "expiry": T, "rate": r, "dividend": q, "vol": s}
        if american:
            price, values = _tree_greeks(S, K, T, r, s, q, kind, int(steps))
            return OptionValuation(kind, True, inputs, price, values, f"a {int(steps)}-step binomial tree")
        return OptionValuation(kind, False, inputs, float(black_scholes(S, K, T, r, s, q, kind)), greeks(S, K, T, r, s, q, kind),
                               "Black-Scholes-Merton")

    @api("finance:implied_volatility")
    def implied_volatility(data: Dataset, price: str, strike: str, expiry: str, spot: Any, rate: Any = 0.0,
                           dividend: Any = 0.0, kind: str = "call") -> VolatilitySmile:
        """The Black-Scholes implied volatility of each option quote: ``price``, ``strike`` and ``expiry`` (years) are
        series; ``spot``, ``rate`` and ``dividend`` numbers or series; ``kind`` 'call', 'put', or a series of 1 for
        calls and -1 for puts."""
        P, K, T = (_value(data, v, w) for v, w in ((price, "price"), (strike, "strike"), (expiry, "expiry")))
        n = len(P)
        S, r, q = (np.broadcast_to(_value(data, v, w), (n,)) for v, w in ((spot, "spot"), (rate, "rate"), (dividend, "dividend")))
        if kind in ("call", "put"):
            kinds = np.array([kind] * n, dtype=object)
        else:
            flag = _value(data, kind, "kind")
            if not np.all(np.isin(flag[~np.isnan(flag)], (1, -1))):
                raise SpecificationError(f"{kind} must hold 1 for calls and -1 for puts")
            kinds = np.where(flag == 1, "call", "put").astype(object)
        iv = [implied_vol(P[i], S[i], K[i], T[i], r[i], q[i], kinds[i]) if np.isfinite(P[i] + K[i] + T[i] + S[i]) else np.nan
              for i in range(n)]
        frame = pd.DataFrame({"strike": K, "expiry": T, "kind": kinds, "price": P, "implied vol": iv, "moneyness": K / S},
                             index=data.frame.index)
        return VolatilitySmile(frame, spot)
