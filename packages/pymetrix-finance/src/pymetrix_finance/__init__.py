"""First-party plugin: financial econometrics (``pymetrix-finance``, ``pip install pymetrix[finance]``).

Value at risk and expected shortfall with their backtests, multivariate GARCH, returns and performance measures,
factor models and portfolio optimisation, option prices and implied volatility, yield curves,
risk contributions, event studies of returns, realised volatility with the HAR model,
portfolio strategy backtests, and bonds and the dynamic Nelson-Siegel model. A package of its own, written against the extension API: it registers its
calls, dialogs, results, views, menu, workspace, report template and example datasets like any other plugin, and
needs nothing beyond pymetrix's own dependencies. ``from pymetrix_finance import value_at_risk``; the same names are
also importable from ``pymetrix`` when this package is installed.
"""

from __future__ import annotations

from ._version import __version__
from pymetrix.extension import PluginInfo

PLUGIN = PluginInfo(
    name="pymetrix-finance",
    version=__version__,
    license="BSD-3-Clause",
    maintainer="Geoffrey Heenan",
    description="Financial econometrics: value at risk and expected shortfall with backtests; multivariate GARCH; "
                "returns and performance; factor models and portfolios; options; yield curves; risk contributions; event studies; "
                "realised volatility and HAR; strategy backtests; bonds and dynamic Nelson-Siegel",
)

from .risk import RiskForecast, value_at_risk  # noqa: E402 - registers under PLUGIN
from .mgarch import FittedMGARCH, mgarch  # noqa: E402
from .returns import Performance, performance, returns  # noqa: E402
from .portfolio import FactorModel, Portfolio, factor_model, optimize_portfolio  # noqa: E402
from .options import OptionValuation, VolatilitySmile, implied_volatility, option_price  # noqa: E402
from .yields import YieldCurve, yield_curve  # noqa: E402
from .contributions import RiskContributions, risk_contributions  # noqa: E402
from .events import AbnormalReturns, abnormal_returns  # noqa: E402
from .realized import FittedHAR, har, realized_measures  # noqa: E402
from .strategies import StrategyBacktest, backtest_strategy, black_litterman, risk_parity_weights  # noqa: E402
from .blacklitterman import BlackLittermanPosterior  # noqa: E402
from .termstructure import FittedDNS, bond_measures, dynamic_nelson_siegel  # noqa: E402
from .copulas import FittedCopula, compare_copulas, copula  # noqa: E402
from .vines import FittedVine, vine_copula  # noqa: E402
from .tvcopulas import FittedDCCCopula, FittedTVCopula, dcc_copula, tv_copula  # noqa: E402
from .credit import CDSCurve, MertonModel, cds_curve, merton, merton_point  # noqa: E402
from .surface import FittedVolSurface, heston_price, vol_surface  # noqa: E402
from . import app  # noqa: E402,F401 - the dialogs, views, menu and workspace
from . import datasets  # noqa: E402,F401 - the example datasets


def register() -> None:
    """The entry point: importing the package has registered everything."""

__all__ = ["PLUGIN", "FittedVolSurface", "heston_price", "vol_surface", "CDSCurve", "MertonModel", "cds_curve", "merton", "merton_point", "FittedCopula", "compare_copulas", "copula", "FittedVine", "vine_copula", "FittedTVCopula", "tv_copula", "FittedDCCCopula", "dcc_copula", "FittedDNS", "bond_measures", "StrategyBacktest", "BlackLittermanPosterior", "backtest_strategy", "black_litterman", "dynamic_nelson_siegel",
           "risk_parity_weights", "FittedHAR", "har", "realized_measures", "AbnormalReturns", "RiskContributions", "abnormal_returns", "risk_contributions", "FactorModel", "FittedMGARCH", "OptionValuation", "Performance", "Portfolio", "RiskForecast", "VolatilitySmile",
           "YieldCurve", "factor_model", "implied_volatility", "mgarch", "optimize_portfolio", "option_price", "performance", "returns",
           "value_at_risk", "yield_curve"]
