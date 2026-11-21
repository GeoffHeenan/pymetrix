"""Build the finance plugin's example datasets in src/pymetrix_finance/datasets (run from packages/pymetrix-finance).

    python scripts/make_datasets.py

US Treasury yields come from FRED (public domain; fetched through pymetrix's fred:// reader, so this needs the
network); the others are simulated with fixed seeds and their true values stated. The results are committed. The
Fama-French returns are not built here: Kenneth R. French's data library states no licence to redistribute them, so
``Dataset.example("french")`` downloads them on first use (``pymetrix_finance/datasets/french.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pymetrix import Dataset
from pymetrix.metadata import DatasetMeta, Variable

OUT = Path(__file__).resolve().parents[1] / "src" / "pymetrix_finance" / "datasets"


def _save(name: str, frame: pd.DataFrame, title: str, description: str, license: str, labels: dict[str, str],
          entity: str | None = None, time: str | None = None, freq: str | None = None) -> None:
    variables = {c: Variable(c, label=labels.get(c, "")) for c in frame.columns if c not in (entity, time)}
    data = Dataset.from_frame(frame, entity=entity, time=time, freq=freq, variables=variables,
                              meta=DatasetMeta(title=title, description=description, license=license))
    data.write(str(OUT / f"{name}.parquet"))
    print(f"{name}: {len(frame)} rows, {len(frame.columns)} columns")

def finance_sets() -> None:
    """The finance plugin's examples: Treasury yields (real) and simulated yields, options, intraday returns and
    events with their true parameters stated."""
    from pymetrix_finance.options import black_scholes
    from pymetrix_finance.yields import loadings

    # US Treasury yields: FRED's monthly constant-maturity series (the Board of Governors' H.15, public domain), fetched
    # through pymetrix's own fred:// reader, so this needs the network; the result is committed
    ids = ["GS3M", "GS6M", "GS1", "GS2", "GS3", "GS5", "GS7", "GS10", "GS20", "GS30"]
    cols = ["m3", "m6", "y1", "y2", "y3", "y5", "y7", "y10", "y20", "y30"]
    frame = Dataset.read(f"fred://{','.join(ids)}").frame.loc["1982-01":"2026-09"]
    frame.columns = cols
    _save("treasury_yields", frame, "US Treasury constant-maturity yields, monthly, 1982-2026",
          "yields in percent, monthly averages, from 3 months (m3) to 30 years (y30); the 20-year is missing from 1987 to "
          "1993 and the 30-year from 2002 to 2006, when the Treasury did not issue them",
          "public domain (Board of Governors of the Federal Reserve System, H.15, via FRED, Federal Reserve Bank of St. Louis)",
          {c: f"FRED {i}: {c[1:]}-{'month' if c[0] == 'm' else 'year'} Treasury constant-maturity yield, percent" for c, i in zip(cols, ids)})

    rng = np.random.default_rng(20261026)
    # yield curves: Diebold-Li factors as AR(1)s, decay 0.7308 a year (0.0609 a month), 10 maturities, in percent
    T, lam = 420, 0.7308
    mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    mean, phi, sd = np.array([5.0, -1.5, 0.0]), np.array([0.99, 0.95, 0.8]), np.array([0.15, 0.25, 0.5])
    F = np.zeros((T, 3))
    F[0] = mean
    for t in range(1, T):
        F[t] = mean + phi * (F[t - 1] - mean) + rng.normal(0, sd)
    cols = ["m3", "m6", "y1", "y2", "y3", "y5", "y7", "y10", "y20", "y30"]
    Y = F @ loadings(mats, lam).T + rng.normal(0, 0.05, (T, len(mats)))
    frame = pd.DataFrame(Y, columns=cols, index=pd.period_range("1990-01", periods=T, freq="M", name="time"))
    _save("yield_curves", frame, "Government bond yields at ten maturities, monthly (simulated)",
          "Nelson-Siegel yields in percent from 3 months (m3) to 30 years (y30), decay 0.7308 a year: level, slope and "
          "curvature AR(1)s with means 5, -1.5, 0, coefficients 0.99, 0.95, 0.8; measurement noise sd 0.05",
          "simulated by pymetrix (BSD-3)", {c: f"yield at {m:g} years, percent" for c, m in zip(cols, mats)})

    # an option chain: out-of-the-money puts and calls with a skewed smile
    S, r, q = 100.0, 0.03, 0.01
    rows = []
    for T_ in (1 / 12, 0.25, 0.5, 1.0, 2.0):
        for K in np.arange(60.0, 141.0, 5.0):
            k = np.log(K / (S * np.exp((r - q) * T_)))
            vol = 0.20 - 0.12 * k + 0.25 * k**2 + 0.01 * T_
            cp = 1 if K >= S else -1
            price = float(black_scholes(S, K, T_, r, vol, q, "call" if cp == 1 else "put"))
            if price >= 0.01:
                rows.append({"quote": f"{'C' if cp == 1 else 'P'}{K:g}-{T_:.3f}", "strike": K, "expiry": T_, "cp": cp,
                             "price": round(price, 4), "true_vol": vol})
    _save("option_chain", pd.DataFrame(rows), "An equity index option chain (simulated)",
          "out-of-the-money puts (cp = -1) and calls (cp = 1), spot 100, rate 0.03, dividend yield 0.01, five expiries in "
          "years; priced by Black-Scholes at true_vol = 0.20 - 0.12 k + 0.25 k^2 + 0.01 expiry, k the log forward moneyness",
          "simulated by pymetrix (BSD-3)", {"strike": "strike price", "expiry": "years to expiry", "cp": "1 for a call, -1 for a put",
                                            "price": "option price", "true_vol": "the volatility it was priced at"}, entity="quote")

    # intraday returns: 250 days of 78 five-minute returns; log variance an AR(1); jumps on ten days
    days, n = 250, 78
    lv = np.zeros(days)
    lv[0] = np.log(1e-4)
    for t in range(1, days):
        lv[t] = 0.05 * np.log(1e-4) + 0.95 * lv[t - 1] + rng.normal(0, 0.2)
    ret = rng.normal(0, 1, (days, n)) * np.sqrt(np.exp(lv)[:, None] / n)
    jump_days = np.sort(rng.choice(days, 10, replace=False))
    ret[jump_days, rng.integers(0, n, 10)] += rng.choice([-1, 1], 10) * 0.02
    stamps = pd.bdate_range("2023-01-02", periods=days)
    idx = pd.PeriodIndex([pd.Period(d + pd.Timedelta(minutes=570 + 5 * j), "5min") for d in stamps for j in range(n)], name="time")
    _save("intraday_returns", pd.DataFrame({"r": ret.ravel()}, index=idx), "Five-minute index returns, 250 days (simulated)",
          "log returns, 78 a day from 9:30; each day's variance exp(lv), lv an AR(1) with coefficient 0.95 around log(1e-4); "
          f"jumps of 0.02 on ten days: {', '.join(str(stamps[d].date()) for d in jump_days)}",
          "simulated by pymetrix (BSD-3)", {"r": "five-minute log return"})

    # event returns: 20 firms on a market model, each with one event raising its return by 0.02 on the day
    days = 750
    stamps = pd.bdate_range("2015-01-02", periods=days)
    mkt = rng.normal(0.0004, 0.01, days)
    frame = pd.DataFrame({"mkt": mkt}, index=pd.PeriodIndex(stamps, freq="D", name="time"))
    events = []
    for f in range(20):
        beta = rng.uniform(0.6, 1.4)
        ret = 0.0001 + beta * mkt + rng.normal(0, 0.015, days)
        day = int(rng.integers(300, 700))
        ret[day] += 0.02
        frame[f"f{f + 1:02d}"] = ret
        events.append(f"f{f + 1:02d} {stamps[day].date()}")
    _save("event_returns", frame, "Daily returns of 20 firms around their announcements (simulated)",
          "returns as fractions: the market mkt and firms f01-f20 on a market model with betas from 0.6 to 1.4; each firm's "
          f"event raises its return by 0.02 on the day: {'; '.join(events)}",
          "simulated by pymetrix (BSD-3)", {"mkt": "the market's daily return", **{f"f{f + 1:02d}": "a firm's daily return" for f in range(20)}})


if __name__ == "__main__":
    finance_sets()
