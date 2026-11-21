"""A simulated stand-in for Kenneth R. French's data library, for tests that need no network and no third-party data.

The plugin downloads ``Dataset.example("french")`` from the library on first use. The tests serve it these files
instead: the same five zipped CSV files, in the library's layout (notes, a percent table of monthly rows, then an
annual table), holding simulated returns. Everything here is pymetrix's own (BSD-3-Clause).

The simulation, 1949-01 to 2017-03 plus a year either side (so the window is cut from longer files, as the real ones
are): the market's excess return follows a GJR-GARCH(1,1) with standardised Student-t(4) errors (omega 0.6,
alpha 0.04, gamma 0.12, beta 0.84, in percent a month, mean 0.6); smb, hml and mom are independent normal factors
(means 0.2, 0.3, 0.7; s.d. 3, 3, 4); rf is 0.3 plus a small AR(1); each industry and sorted portfolio is rf plus a
fixed beta on the market, loadings on smb and hml (and mom for the momentum sorts) and its own normal noise.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from pymetrix_finance.datasets import french

MONTHS = pd.period_range("1948-01", "2018-06", freq="M")


def returns(seed: int = 20261120) -> pd.DataFrame:
    """The simulated library's monthly returns, in percent, with the library's own column names."""
    rng = np.random.default_rng(seed)
    n = len(MONTHS)
    omega, alpha, gamma, beta, nu = 0.6, 0.04, 0.12, 0.84, 4.0
    z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    var, eps = np.empty(n), np.empty(n)
    var[0] = omega / (1 - alpha - gamma / 2 - beta)
    for t in range(n):
        if t:
            var[t] = omega + (alpha + gamma * (eps[t - 1] < 0)) * eps[t - 1] ** 2 + beta * var[t - 1]
        eps[t] = np.sqrt(var[t]) * z[t]
    out = {"Mkt-RF": 0.6 + eps, "SMB": rng.normal(0.2, 3.0, n), "HML": rng.normal(0.3, 3.0, n), "Mom": rng.normal(0.7, 4.0, n)}
    rf = np.empty(n)
    rf[0] = 0.3
    for t in range(1, n):
        rf[t] = 0.3 + 0.95 * (rf[t - 1] - 0.3) + rng.normal(0, 0.02)
    out["RF"] = rf
    names = [c for stem, columns in french.FILES.items() if stem not in ("F-F_Research_Data_Factors", "F-F_Momentum_Factor") for c in columns]
    for name in names:
        b, s, h, m = rng.uniform(0.6, 1.4), rng.uniform(-0.5, 1.2), rng.uniform(-0.6, 0.8), rng.uniform(-0.4, 0.4)
        momentum = m if "PRIOR" in name else 0.0
        out[name] = (rf + b * out["Mkt-RF"] + s * out["SMB"] + h * out["HML"] + momentum * out["Mom"]
                     + rng.normal(0.1, rng.uniform(1.0, 3.0), n))
    return pd.DataFrame(out, index=MONTHS)


def library(seed: int = 20261120) -> dict[str, bytes]:
    """The five files as the library serves them: ``{url: zip bytes}``."""
    frame = returns(seed).round(2)
    files = {}
    for stem, columns in french.FILES.items():
        cols = list(columns)
        table = frame[cols]
        lines = ["This file was simulated for pymetrix's tests.", "", "Missing data are indicated by -99.99 or -999.", "",
                 "  Average Value Weighted Returns -- Monthly", "," + ",".join(cols)]
        lines += [f"{p.year}{p.month:02d}," + ",".join(f"{v:7.2f}" for v in row) for p, row in zip(table.index, table.to_numpy())]
        annual = table.groupby(table.index.year).sum()
        lines += ["", "", "  Average Value Weighted Returns -- Annual", "," + ",".join(cols)]
        lines += [f"{y}," + ",".join(f"{v:7.2f}" for v in row) for y, row in zip(annual.index, annual.to_numpy())]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(f"{stem}.csv", "\r\n".join(lines) + "\r\n")
        files[f"{french.BASE}{stem}_CSV.zip"] = buffer.getvalue()
    return files
