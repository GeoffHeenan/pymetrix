"""The finance plugin's example datasets, registered so that ``Dataset.example("french")`` finds them.

Made by ``packages/pymetrix-finance/scripts/make_datasets.py``: US Treasury yields (real) and simulated sets with
their true values stated in their descriptions. The Fama-French returns are not shipped: ``french`` downloads them
from Kenneth R. French's data library on first use and keeps them in the user's cache (``french.py``).
"""

from __future__ import annotations

from pathlib import Path

from pymetrix.extension import ExampleDataset
from pymetrix.extension import register

from .. import PLUGIN
from .french import Download

_HERE = Path(__file__).parent
DESCRIPTIONS = {
    "french": "Fama-French monthly returns, 1949-2017 (Kenneth French's data library, downloaded on first use): mkt_rf, "
              "smb, hml, mom, rf, 12 industries and size-value and size-momentum portfolios; performance, factor models, "
              "portfolios",
    "treasury_yields": "US Treasury constant-maturity yields, monthly, 1982-2026 (FRED), 3 months to 30 years in percent; "
                       "yield curves",
    "yield_curves": "simulated: Nelson-Siegel yields at ten maturities, monthly, decay 0.7308 and AR(1) factors; yield curves",
    "option_chain": "simulated: 71 option quotes, spot 100, five expiries, priced with a skewed smile (true_vol); implied "
                    "volatility",
    "intraday_returns": "simulated: five-minute returns, 78 a day for 250 days, with jumps on ten days; realised volatility, HAR",
    "event_returns": "simulated: daily returns of 20 firms and the market, each firm with one event worth +0.02; event studies",
}

for _name, _description in DESCRIPTIONS.items():
    _path = Download() if _name == "french" else _HERE / f"{_name}.parquet"
    register("datasets", ExampleDataset(_name, _description, _path), name=_name, plugin=PLUGIN)
