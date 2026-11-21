"""Write the example datasets: a quarterly toy economy and monthly indicators of the same economy.

``sample_macro``: quarterly, 2000Q1-2024Q4::

    c = 10 + 0.6*y + 0.2*c[-1] + e_c
    i = 5 + 0.1*y[-1] + e_i
    y = c + i + g

simulated from 1990 so that the first decade (a burn-in, not kept) takes the economy to its path before 2000.

``sample_monthly``: monthly, 2000-01 to 2025-03, indicators of the same economy for nowcasting: industrial
production ``ip`` (an index), retail sales ``retail``, employment ``emp`` and a survey ``pmi`` (a diffusion index
around 50). They follow a monthly activity path whose growth adds up, quarter by quarter, to GDP growth (including
2025Q1, which is not in ``sample_macro``: the quarter to nowcast). Each series carries its publication lag in days
(``Variable.release_lag``), so ``data.as_of(date)`` gives what was known on a day.

Both are written to examples/ and into the package (pymetrix/datasets), where ``Dataset.example(...)`` finds them.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pymetrix import Dataset, DatasetMeta, Measure, Variable

rng = np.random.default_rng(42)
burn = 40
periods = pd.period_range("1990Q1", "2025Q1", freq="Q", name="time")  # 2025Q1: simulated, kept back from sample_macro
n = len(periods)
g = 20 + 0.5 * (np.arange(n) - burn) + rng.normal(0, 1, n)
g = np.maximum(g, 5.0)
c, i, y = np.zeros(n), np.zeros(n), np.zeros(n)
i[0] = 5 + 0.1 * 30
c[0] = (10 + 0.6 * (i[0] + g[0])) / 0.2  # near the path: c = (10 + 0.6 (i + g) + 0.2 c) / 0.4 at rest
y[0] = c[0] + i[0] + g[0]
for t in range(1, n):
    i[t] = 5 + 0.1 * y[t - 1] + rng.normal(0, 0.5)
    e_c = rng.normal(0, 1)
    c[t] = (10 + 0.6 * (i[t] + g[t]) + 0.2 * c[t - 1] + e_c) / 0.4
    y[t] = c[t] + i[t] + g[t]

keep = (periods >= pd.Period("2000Q1", "Q")) & (periods <= pd.Period("2024Q4", "Q"))
variables = {
    "c": Variable("c", label="Consumption", units="currency units", scale=6, measure=Measure.FLOW, release_lag=60),
    "i": Variable("i", label="Investment", units="currency units", scale=6, measure=Measure.FLOW, release_lag=60),
    "g": Variable("g", label="Government spending", units="currency units", scale=6, measure=Measure.FLOW, release_lag=60),
    "y": Variable("y", label="GDP", units="currency units", scale=6, measure=Measure.FLOW, release_lag=30),
}
quarterly = Dataset.from_frame(
    pd.DataFrame({"c": c[keep], "i": i[keep], "g": g[keep], "y": y[keep]}, index=periods[keep]),
    variables=variables,
    meta=DatasetMeta(title="Toy economy (simulated)"),
)

# monthly activity: quarterly GDP growth spread over the quarter's months with noise that cancels within the quarter
months = pd.period_range("2000-01", "2025-03", freq="M", name="time")
qgrowth = pd.Series(np.diff(np.log(y)), index=periods[1:])  # log growth of each quarter
monthly_growth = np.empty(len(months))
for q in pd.period_range("2000Q1", "2025Q1", freq="Q"):
    noise = rng.normal(0, 0.002, 3)
    noise -= noise.mean()
    at = (months >= q.asfreq("M", "s")) & (months <= q.asfreq("M", "e"))
    monthly_growth[at] = qgrowth[q] / 3 + noise
activity = np.cumsum(monthly_growth)  # log activity, relative to January 2000


def ar1(rho: float, sd: float, size: int) -> np.ndarray:
    out = np.zeros(size)
    for t in range(1, size):
        out[t] = rho * out[t - 1] + rng.normal(0, sd)
    return out


m = len(months)
standard = (monthly_growth - monthly_growth.mean()) / monthly_growth.std()
monthly = pd.DataFrame({
    "ip": 100 * np.exp(1.4 * activity + ar1(0.6, 0.006, m)),
    "retail": 250 * np.exp(0.9 * activity + ar1(0.3, 0.004, m)),
    "emp": 5000 * np.exp(0.5 * pd.Series(activity).ewm(alpha=0.3).mean().to_numpy() + 0.0005 * np.arange(m) + ar1(0.8, 0.0008, m)),
    "pmi": 50 + 3.0 * pd.Series(standard).rolling(2, min_periods=1).mean().to_numpy() + ar1(0.5, 1.0, m),
}, index=months)
monthly_vars = {
    "ip": Variable("ip", label="Industrial production", units="index", measure=Measure.FLOW, release_lag=45),
    "retail": Variable("retail", label="Retail sales volume", units="currency units", scale=6, measure=Measure.FLOW, release_lag=15),
    "emp": Variable("emp", label="Employment", units="thousands of persons", scale=3, measure=Measure.STOCK, release_lag=5),
    "pmi": Variable("pmi", label="Purchasing managers' index", units="index (50 = no change)", measure=Measure.INDEX
                    if hasattr(Measure, "INDEX") else None, release_lag=1),
}
indicators = Dataset.from_frame(monthly, variables=monthly_vars, meta=DatasetMeta(title="Monthly indicators of the toy economy (simulated)"))

here = Path(__file__).parent
for name, data in (("sample_macro", quarterly), ("sample_monthly", indicators)):
    for out in (here / f"{name}.parquet", here.parent / "src" / "pymetrix" / "datasets" / f"{name}.parquet"):
        data.write(str(out))
        print(f"wrote {out}")
print("2025Q1 GDP (kept back, for checking nowcasts):", round(float(y[-1]), 3))
