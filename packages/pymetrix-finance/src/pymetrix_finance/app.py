"""The finance plugin in the app: its dialogs, its results' windows, the Finance menu and the Finance workspace,
registered through the extension API like any other plugin's."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pymetrix.extension import Command, Widget, command, migration
from pymetrix.extension import plugin_context, register
from pymetrix.extension import register_view
from . import PLUGIN
from .contributions import RiskContributions
from .events import AbnormalReturns
from .realized import FittedHAR
from .blacklitterman import BlackLittermanPosterior
from .strategies import StrategyBacktest
from .termstructure import FittedDNS
from .copulas import FittedCopula
from .tvcopulas import FittedDCCCopula, FittedTVCopula
from .vines import FittedVine
from .credit import CDSCurve, MertonModel
from .surface import FittedVolSurface
from .mgarch import FittedMGARCH
from .options import OptionValuation, VolatilitySmile
from .portfolio import FactorModel, Portfolio
from .returns import Performance
from .risk import RiskForecast
from .yields import YieldCurve

if TYPE_CHECKING:
    from pymetrix.commands import Session

with plugin_context(PLUGIN):
    @command(menu="Finance/Value at risk...")
    @dataclass(frozen=True)
    class ValueAtRisk(Command):
        """Value at risk and expected shortfall, forecast one period ahead and backtested.

        Each period's VaR and ES come from the returns before it only: the last window of them (historical
        simulation, normal, Student-t, Cornish-Fisher), or a GARCH(1,1) estimated on everything before (GARCH,
        filtered historical simulation). The window shows the backtests (Kupiec, Christoffersen, the dynamic
        quantile test, McNeil-Frey for shortfall, the Basel zone) and the returns against -VaR and -ES.

        VaR at 99% is the loss exceeded on about 1% of periods; ES is the average loss on those periods, so it
        also weighs how bad the tail is. Historical simulation and the window-based methods react slowly to a
        change in volatility, and a longer window reacts more slowly still; GARCH and filtered historical
        simulation follow the current volatility. Read the backtests together: Kupiec asks whether violations
        happen as often as the level says, Christoffersen also whether they cluster, which signals that the
        forecasts miss changes in volatility. At high levels violations are rare, so the tests have little power
        without a long sample.
        """

        topics: ClassVar[tuple[str, ...]] = ("value-at-risk",)

        series: Annotated[str, Widget("series", depends_on="data", help="returns, in the units they are (log or simple)")]
        data: Annotated[str, Widget("object", source="Dataset", help="the time-series dataset holding the returns; periods where they are missing are skipped")] = "data"
        level: Annotated[float, Widget("number", min=0.5, max=0.9999, step=0.005, help="the confidence: 0.99 for 99% VaR")] = 0.99
        method: Annotated[Literal["historical", "normal", "t", "cornish_fisher", "garch", "fhs"],
                          Widget("choice", help="historical simulation, normal, Student-t, Cornish-Fisher, GARCH, or filtered historical")] = "historical"
        window: Annotated[int, Widget("number", min=20, max=5000, help="returns used for each forecast (GARCH: from the start)")] = 250
        refit: Annotated[int, Widget("number", min=1, max=500, advanced=True, help="re-estimate the t or GARCH every this many periods")] = 20
        start: Annotated[str, Widget("period", help="the first period forecast (empty: once the window is full)")] = ""
        dist: Annotated[Literal["normal", "t"], Widget("choice", show_if=("method", ("garch", "fhs")),
                                                       help="the GARCH's errors: normal or Student-t (fat tails)")] = "normal"
        asymmetric: Annotated[bool, Widget("checkbox", show_if=("method", ("garch", "fhs")),
                                           help="GJR: volatility rises more after falls than after rises")] = False
        name: str = "risk"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref

            return make_call("finance:value_at_risk", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                             series=self.series, level=self.level, method=self.method, window=self.window, refit=self.refit,
                             start=self.start or None, **({"dist": self.dist, "asymmetric": self.asymmetric}
                                                          if self.method in ("garch", "fhs") else {}))

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Multivariate GARCH...")
    @dataclass(frozen=True)
    class FitMGARCH(Command):
        """Multivariate GARCH: the conditional covariances and correlations of several returns.

        CCC and DCC put a GARCH(1,1) on each series, then a constant or a moving correlation of their standardised
        residuals; scalar and diagonal BEKK model the whole covariance at once, with its long-run level set to the
        sample covariance. The window shows the parameters, the correlations and volatilities over time, and the
        covariance forecast for the next period.

        The models trade flexibility against parameters: CCC holds correlations fixed, DCC lets them move with a
        few parameters, and the full VECH and BEKK forms grow quickly with the number of series and become hard
        to estimate beyond a handful. Correlations that rise in falling markets, which DCC and its asymmetric
        form can show, are what matter for diversification. Use returns at a frequency where volatility clusters
        (daily or weekly); the next-period covariance can feed the Portfolio and Risk contributions dialogs.
        """

        topics: ClassVar[tuple[str, ...]] = ("multivariate-garch",)

        variables: Annotated[str, Widget("series", depends_on="data", multiple=True, help="two or more series of returns")]
        data: Annotated[str, Widget("object", source="Dataset", help="the time-series dataset of returns; periods missing any of the series are dropped")] = "data"
        model: Annotated[Literal["ccc", "dcc", "adcc", "deco", "bekk", "diagonal_bekk", "diagonal_vech", "vech"],
                         Widget("choice", help="constant, dynamic or asymmetric dynamic conditional correlation, dynamic equicorrelation, "
                                               "scalar or diagonal BEKK, or diagonal or full VECH")] = "dcc"
        asymmetric: Annotated[bool, Widget("checkbox", show_if=("model", ("ccc", "dcc", "adcc")),
                                           help="GJR volatilities: more response to falls than to rises")] = False
        name: str = "mgarch_fit"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref

            names = [v.strip() for v in self.variables.split(",") if v.strip()]
            return make_call("finance:mgarch", output=self.name, recorded_by=self.command_id, data=Ref(self.data), variables=names,
                             model=self.model, **({"asymmetric": True} if self.asymmetric else {}))

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    def _names(text: str) -> list[str]:
        return [v.strip() for v in text.split(",") if v.strip()]


    @command(menu="Finance/Returns...")
    @dataclass(frozen=True)
    class Returns(Command):
        """Returns from prices, added to the dataset: log (the change in the log price) or simple (p / p[-1] - 1).

        Each price gets a series named after it with the suffix (spx_ret from spx), over one period or several.
        Prices must be positive.

        Log returns add up over time, which suits modelling and multi-period sums; simple returns add up across
        assets in a portfolio, which suits portfolio weights and backtests. The two are close for small changes
        and differ for large ones. Multi-period returns overlap when computed every period, so successive values
        are correlated by construction: account for that (e.g. with Newey-West standard errors) in regressions
        on them.
        """

        topics: ClassVar[tuple[str, ...]] = ("returns-and-performance",)

        prices: Annotated[str, Widget("series", depends_on="data", multiple=True, help="positive prices or index levels, total-return where dividends matter; one return series each")]
        data: Annotated[str, Widget("object", source="Dataset", help="the dataset; the returns are added to it")] = "data"
        kind: Annotated[Literal["log", "simple"], Widget("choice", help="log returns, or simple (arithmetic) ones")] = "log"
        periods: Annotated[int, Widget("number", min=1, max=1000, help="the return over this many periods")] = 1
        suffix: Annotated[str, Widget("text", help="added to each price's name for its return")] = "_ret"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref

            return make_call("finance:returns", output=self.data, recorded_by=self.command_id, data=Ref(self.data),
                             prices=_names(self.prices), kind=self.kind, periods=self.periods, suffix=self.suffix)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Performance...")
    @dataclass(frozen=True)
    class MeasurePerformance(Command):
        """Performance measures of return series: return, volatility, Sharpe and Sortino ratios, drawdowns.

        Annualised by the data's frequency. With a risk-free rate the ratios use excess returns; with a benchmark,
        alpha, beta, tracking error and the information ratio are added. The window shows the measures, the growth
        of 1 invested with the drawdowns, and the cumulative returns.

        Mean returns are estimated imprecisely: a Sharpe ratio from a few years of data has a wide standard
        error (reported), so differences between assets' ratios are often not significant. The Sharpe ratio
        penalises upside and downside alike; Sortino counts only the downside, and the maximum drawdown measures
        the worst peak-to-trough loss in the sample, which depends on its length. Alpha and beta against a
        benchmark come from a regression of the excess returns on the benchmark's; a positive alpha is a
        risk-adjusted excess only if the benchmark is the right risk.
        """

        topics: ClassVar[tuple[str, ...]] = ("returns-and-performance",)

        series: Annotated[str, Widget("series", depends_on="data", multiple=True, help="returns, as fractions (0.01 for 1%)")]
        data: Annotated[str, Widget("object", source="Dataset", help="the time-series dataset of returns; its frequency sets the annualising; incomplete periods dropped")] = "data"
        rf: Annotated[str, Widget("series", depends_on="data", help="the risk-free rate per period (empty: none)")] = ""
        benchmark: Annotated[str, Widget("series", depends_on="data", help="returns to compare with (empty: none)")] = ""
        kind: Annotated[Literal["log", "simple"], Widget("choice", help="whether the returns are log or simple")] = "log"
        periods_per_year: Annotated[int, Widget("number", min=0, max=100000, advanced=True,
                                                help="periods in a year for annualising (0: from the frequency)")] = 0
        name: str = "perf"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref

            return make_call("finance:performance", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                             series=_names(self.series), rf=self.rf or None, benchmark=self.benchmark or None, kind=self.kind,
                             periods_per_year=self.periods_per_year or None)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Factor model...")
    @dataclass(frozen=True)
    class FactorRegression(Command):
        """Factor model: each asset's excess return on the factors (CAPM, Fama-French), and the GRS test.

        Alphas and betas by least squares with Newey-West standard errors; the Gibbons-Ross-Shanken test asks
        whether every alpha is zero at once. The window shows the coefficients and the rolling betas.

        A beta measures an asset's exposure to a factor; an alpha is the average return the factors do not
        explain. If the factors price the assets, every alpha is zero: the GRS test asks that jointly and takes
        account of the correlation between the assets' errors, so it can reject when no single alpha is
        significant. It assumes normal, serially uncorrelated errors and needs more periods than assets. Factors
        must be excess returns (or long-short returns) for the alphas to have that meaning; rolling betas show
        whether exposures are stable.
        """

        topics: ClassVar[tuple[str, ...]] = ("factor-models-and-portfolios",)

        assets: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the assets' (or portfolios') returns; the risk-free rate, if given, is subtracted before the regressions")]
        factors: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the factors: excess returns, such as mkt-rf, smb, hml")]
        data: Annotated[str, Widget("object", source="Dataset", help="the time-series dataset of returns, factors and rf; periods missing any are dropped")] = "data"
        rf: Annotated[str, Widget("series", depends_on="data", help="the risk-free rate taken from the assets' returns (empty: none)")] = ""
        name: str = "factors_fit"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref

            return make_call("finance:factor_model", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                             assets=_names(self.assets), factors=_names(self.factors), rf=self.rf or None)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Portfolio...")
    @dataclass(frozen=True)
    class OptimizePortfolio(Command):
        """Portfolio weights: minimum variance, maximum Sharpe ratio, or the least variance for a target return.

        Weights sum to one, long only or between -1 and 2; the mean and covariance are the sample's, or the
        next-period forecast of a fitted multivariate GARCH. The window shows the weights and the efficient frontier.

        Optimised weights take the estimated means and covariances as true, and they are only estimates: the
        mean returns especially are imprecise, so maximum-Sharpe and target-return weights are often extreme and
        unstable out of sample. Minimum variance needs only the covariance and is usually more robust. Long-only
        limits act as shrinkage and tame the extremes. Judge a rule out of sample with the Strategy backtest
        rather than by its in-sample frontier.
        """

        topics: ClassVar[tuple[str, ...]] = ("factor-models-and-portfolios",)

        assets: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the assets' returns per period; their mean and covariance (or a GARCH's) set the weights")]
        data: Annotated[str, Widget("object", source="Dataset", help="the time-series dataset of returns; periods missing any asset are dropped")] = "data"
        objective: Annotated[Literal["min_variance", "max_sharpe", "target_return"],
                             Widget("choice", help="minimum variance, maximum Sharpe ratio, or a target mean return")] = "min_variance"
        rf: Annotated[float, Widget("number", min=-1.0, max=1.0, step=0.0001, help="the risk-free rate per period, for the Sharpe ratio")] = 0.0
        target_return: Annotated[float, Widget("number", min=-1.0, max=1.0, step=0.0001,
                                               help="the mean return per period, for a target return")] = 0.0
        long_only: Annotated[bool, Widget("checkbox", help="weights between 0 and 1 (otherwise -1 to 2)")] = True
        covariance: Annotated[str, Widget("object", source="FittedMGARCH",
                                          help="a multivariate GARCH whose next-period covariance to use (empty: the sample's)")] = ""
        name: str = "portfolio"

        schema_version: ClassVar[int] = 2  # 2: target became target_return

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref

            extra: dict[str, Any] = {"target_return": self.target_return} if self.objective == "target_return" else {}
            if self.covariance:
                extra["cov"] = Ref(self.covariance)
            return make_call("finance:optimize_portfolio", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                             assets=_names(self.assets), objective=self.objective, rf=self.rf, long_only=self.long_only, **extra)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()

    @migration("finance:optimize_portfolio", 1)
    def _optimize_portfolio_target_return(fields: dict[str, Any]) -> dict[str, Any]:  # schema 2: target became target_return
        if "target" in fields:
            fields["target_return"] = fields.pop("target")
        return fields

    @command(menu="Finance/Option price...")
    @dataclass(frozen=True)
    class PriceOption(Command):
        """Option price and Greeks: Black-Scholes-Merton for European options, a binomial tree for American ones.

        Expiry is in years; the rate and dividend yield are continuously compounded per year, the volatility annual.
        The window shows the price with delta, gamma, vega, theta and rho.

        Black-Scholes-Merton assumes a constant volatility, a lognormal price and frictionless trading; the
        price is the same whatever the expected return, since the option can be hedged. The volatility is the
        input that matters most and the only one not observed: market prices imply a different one by strike and
        expiry (see Implied volatility). The Greeks are sensitivities: delta the change in price per unit of
        spot (the hedge ratio), gamma the change in delta, vega per unit of volatility, theta per unit of time
        and rho per unit of the rate. American options are worth more only when early exercise pays, e.g. puts
        deep in the money or calls before a dividend.
        """

        topics: ClassVar[tuple[str, ...]] = ("options",)

        spot: Annotated[float, Widget("number", min=0.0, max=1e9, help="the underlying's price now")] = 100.0
        strike: Annotated[float, Widget("number", min=0.0, max=1e9, help="the price at which the option can be exercised; against the spot it sets the moneyness")] = 100.0
        expiry: Annotated[float, Widget("number", min=0.0, max=100.0, step=0.05, help="time to expiry in years (0.25 for three months); more time adds time value")] = 1.0
        rate: Annotated[float, Widget("number", min=-1.0, max=1.0, step=0.001, help="the risk-free rate, continuously compounded")] = 0.03
        vol: Annotated[float, Widget("number", min=0.0, max=10.0, step=0.01, help="the annual volatility (0.2 for 20%)")] = 0.2
        dividend: Annotated[float, Widget("number", min=-1.0, max=1.0, step=0.001, help="the dividend yield, continuously compounded")] = 0.0
        kind: Annotated[Literal["call", "put"], Widget("choice", help="call: the right to buy at the strike; put: the right to sell at it")] = "call"
        american: Annotated[bool, Widget("checkbox", help="exercise at any time (priced by a binomial tree)")] = False
        steps: Annotated[int, Widget("number", min=3, max=20000, advanced=True, help="the binomial tree's steps")] = 500
        name: str = "option"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call

            extra: dict[str, Any] = {"steps": self.steps} if self.american else {}
            return make_call("finance:option_price", output=self.name, recorded_by=self.command_id, spot=self.spot, strike=self.strike,
                             expiry=self.expiry, rate=self.rate, vol=self.vol, dividend=self.dividend, kind=self.kind,
                             american=self.american, **extra)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Implied volatility...")
    @dataclass(frozen=True)
    class ImpliedVolatility(Command):
        """Implied volatility of option quotes, and the smile they make by strike and expiry.

        Each quote's price, strike and expiry (in years) are series of a dataset of quotes; the spot, rate and
        dividend yield are numbers here. The window shows each quote's implied volatility and the smile, one line per
        expiry.

        The implied volatility is the one at which Black-Scholes gives the market price: a way of quoting prices
        that removes the effect of strike and expiry. If Black-Scholes held it would be the same for every
        quote; a smile or skew (higher for low strikes, as in equity markets) shows the market pricing fatter
        tails or crash risk than the lognormal. Quotes far from the money, stale or with wide bid-ask spreads
        give unreliable volatilities; mid prices of liquid options are best.
        """

        topics: ClassVar[tuple[str, ...]] = ("options",)

        price: Annotated[str, Widget("series", depends_on="data", help="each quote's price; one outside the no-arbitrage bounds gets no volatility")]
        strike: Annotated[str, Widget("series", depends_on="data", help="each quote's strike, in the spot's units; strike / spot is the moneyness shown")]
        expiry: Annotated[str, Widget("series", depends_on="data", help="each quote's time to expiry in years (0.25 for three months); must be positive")]
        data: Annotated[str, Widget("object", source="Dataset", help="a dataset of option quotes taken at one time, one row per option")] = "data"
        spot: Annotated[float, Widget("number", min=0.0, max=1e9, help="the underlying's price when the quotes were taken")] = 100.0
        rate: Annotated[float, Widget("number", min=-1.0, max=1.0, step=0.001, help="the risk-free rate, continuously compounded")] = 0.0
        dividend: Annotated[float, Widget("number", min=-1.0, max=1.0, step=0.001, help="the dividend yield, continuously compounded")] = 0.0
        kind: Annotated[str, Widget("text", help="call, put, or a series holding 1 for calls and -1 for puts")] = "call"
        name: str = "smile"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref

            return make_call("finance:implied_volatility", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                             price=self.price, strike=self.strike, expiry=self.expiry, spot=self.spot, rate=self.rate,
                             dividend=self.dividend, kind=self.kind)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Yield curve...")
    @dataclass(frozen=True)
    class FitYieldCurve(Command):
        """Yield curve: Nelson-Siegel or Svensson level, slope and curvature, period by period (Diebold-Li).

        Give the yield series and their maturities in years, in the same order. The decay is estimated over the whole
        sample unless given. The window shows the fit at each maturity, the factors over time with the latest curve,
        and the factors as a table.

        Level, slope and curvature summarise the curve: the level moves all yields, the slope short against long
        ones, and the curvature the middle. The decay sets the maturity where the curvature loading peaks; with
        it fixed, each period's fit is linear least squares, and the factors can be modelled and forecast as
        time series (Diebold-Li). The curves are smooth fits, not arbitrage-free: they are suited to summarising
        and forecasting the curve rather than to pricing.
        """

        topics: ClassVar[tuple[str, ...]] = ("yield-curves",)

        series: Annotated[str, Widget("series", depends_on="data", multiple=True, help="yields, shortest maturity first")]
        maturities: Annotated[str, Widget("text", help="each series' maturity in years, in order: 0.25, 1, 2, 5, 10")]
        data: Annotated[str, Widget("object", source="Dataset", help="the dataset of yields, one row per period; the factors are in the yields' units")] = "data"
        model: Annotated[Literal["nelson_siegel", "svensson"], Widget("choice", help="Nelson-Siegel, or Svensson's second curvature")] = "nelson_siegel"
        lam: Annotated[float, Widget("number", min=0.0, max=100.0, step=0.01, advanced=True,
                                     help="the decay per year (0: estimate it); Diebold-Li's is 0.7308")] = 0.0
        name: str = "curve"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref
            from pymetrix.errors import SpecificationError

            try:
                maturities = [float(v) for v in self.maturities.replace(";", ",").split(",") if v.strip()]
            except ValueError:
                raise SpecificationError(f"the maturities must be numbers separated by commas, not {self.maturities!r}") from None
            extra: dict[str, Any] = {"lam": self.lam} if self.lam else {}
            return make_call("finance:yield_curve", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                             series=_names(self.series), maturities=maturities, model=self.model, **extra)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Risk contributions...")
    @dataclass(frozen=True)
    class DecomposeRisk(Command):
        """Risk contributions: each asset's marginal and component VaR and ES in a portfolio.

        The components add up to the portfolio's VaR and ES, so they say where its risk comes from. Normal, with the
        sample covariance or a multivariate GARCH's forecast for the next period, or historical simulation.

        An asset's component is its weight times its marginal contribution: how much the portfolio's VaR or ES
        would change with a small increase in that weight. A component can be negative when the asset hedges the
        rest. With the normal method the components follow from the covariance, so they inherit its estimation
        error and miss fat tails; historical simulation uses the sample's own losses, but rests on few
        observations in the tail.
        """

        topics: ClassVar[tuple[str, ...]] = ("risk-contributions",)

        assets: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the returns of the portfolio's assets, in the order of the weights")]
        weights: Annotated[str, Widget("text", help="the portfolio's weights, in the assets' order: 0.5, 0.3, 0.2")]
        data: Annotated[str, Widget("object", source="Dataset", help="the time-series dataset of returns; periods missing any asset are dropped")] = "data"
        level: Annotated[float, Widget("number", min=0.5, max=0.9999, step=0.005, help="the confidence: 0.99 for 99% VaR")] = 0.99
        method: Annotated[Literal["normal", "historical"], Widget("choice", help="normal, or historical simulation")] = "normal"
        covariance: Annotated[str, Widget("object", source="FittedMGARCH",
                                          help="a multivariate GARCH whose next-period covariance to use (empty: the sample's)")] = ""
        name: str = "risk_parts"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref
            from pymetrix.errors import SpecificationError

            try:
                weights = [float(v) for v in self.weights.replace(";", ",").split(",") if v.strip()]
            except ValueError:
                raise SpecificationError(f"the weights must be numbers separated by commas, not {self.weights!r}") from None
            extra: dict[str, Any] = {"cov": Ref(self.covariance)} if self.covariance else {}
            return make_call("finance:risk_contributions", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                             assets=_names(self.assets), weights=weights, level=self.level, method=self.method, **extra)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Event study of returns...")
    @dataclass(frozen=True)
    class StudyEvents(Command):
        """Event study: abnormal returns around events, and the tests that they are zero.

        Normal returns come from the market model (with a market series), a factor model or the mean return, estimated
        before each event. The window shows the cross-sectional, Patell, BMP and rank tests, the average and cumulative
        abnormal returns, and each event's abnormal returns.

        Abnormal returns measure the event's effect only if the normal-return model is right and the estimation
        window is unaffected by the event, which is why it ends before the event window. Leakage before the
        announcement shows up as abnormal returns before day 0; a wider window catches it but adds noise. Events
        that cluster in calendar time have correlated abnormal returns, which the standard tests assume away;
        the BMP test allows for event-induced changes in variance, and the rank test does not assume normality.
        """

        topics: ClassVar[tuple[str, ...]] = ("event-studies",)

        events: Annotated[str, Widget("text", help="asset: date, date; asset: date (e.g. ibm: 2015-03-02; ge: 2016-05-10)")]
        data: Annotated[str, Widget("object", source="Dataset", help="the assets' returns over time")] = "data"
        market: Annotated[str, Widget("series", depends_on="data", help="the market's returns, for the market model (empty: none)")] = ""
        factors: Annotated[str, Widget("series", depends_on="data", multiple=True, advanced=True,
                                       help="factors for a factor model instead (empty: none)")] = ""
        estimation_start: Annotated[int, Widget("number", min=-10000, max=-2, help="the estimation window's first period, from the event")] = -250
        estimation_end: Annotated[int, Widget("number", min=-10000, max=-1, help="its last period, before the event window")] = -11
        window_start: Annotated[int, Widget("number", min=-1000, max=0, help="the event window's first period")] = -10
        window_end: Annotated[int, Widget("number", min=0, max=1000, help="the event window's last period")] = 10
        test_start: Annotated[int, Widget("number", min=-1000, max=1000, help="the first period of the tested CAR")] = -1
        test_end: Annotated[int, Widget("number", min=-1000, max=1000, help="the last period of the tested CAR")] = 1
        name: str = "events"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref
            from pymetrix.errors import SpecificationError

            events: dict[str, list[str]] = {}
            for part in self.events.split(";"):
                if not part.strip():
                    continue
                asset, sep, dates = part.partition(":")
                if not sep or not asset.strip():
                    raise SpecificationError(f"write each asset's events as 'asset: date, date', not {part.strip()!r}")
                events.setdefault(asset.strip(), []).extend(d.strip() for d in dates.split(",") if d.strip())
            extra: dict[str, Any] = {"market": self.market} if self.market else {}
            if self.factors:
                extra["factors"] = _names(self.factors)
            return make_call("finance:abnormal_returns", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                             events=events, estimation=(self.estimation_start, self.estimation_end),
                             window=(self.window_start, self.window_end), car_window=(self.test_start, self.test_end), **extra)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Realised volatility...")
    @dataclass(frozen=True)
    class RealizedVolatility(Command):
        """Realised volatility: each day's realised variance, bipower variation and jump test, from intraday returns.

        The result is a daily dataset: rv (realised variance), bv (bipower variation, robust to jumps), tq, the
        Barndorff-Nielsen-Shephard jump statistic and p-value, the jump variation on days it rejects, the returns
        counted, and the annualised realised volatility rvol.
        """

        topics: ClassVar[tuple[str, ...]] = ("realized-volatility",)

        series: Annotated[str, Widget("series", depends_on="data", help="intraday returns (several a day)")]
        data: Annotated[str, Widget("object", source="Dataset", help="intraday returns over time")] = "data"
        jump_level: Annotated[float, Widget("number", min=0.5, max=0.99999, step=0.001, help="the confidence for flagging a jump day")] = 0.999
        annualize: Annotated[int, Widget("number", min=1, max=100000, advanced=True, help="trading days a year, for rvol")] = 252
        name: str = "daily"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref

            return make_call("finance:realized_measures", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                             series=self.series, jump_level=self.jump_level, annualize=self.annualize)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/HAR model...")
    @dataclass(frozen=True)
    class FitHAR(Command):
        """HAR model: tomorrow's realised variance on today's and its weekly and monthly averages (Corsi 2009).

        Least squares with Newey-West standard errors, in levels or logs. The window shows the coefficients, the fit
        and the forecast for the next 22 days.

        The daily, weekly and monthly averages stand for traders acting at different horizons, and together they
        mimic the slow decay of volatility's autocorrelation with only three slopes. Logs make the errors closer
        to normal and keep forecasts positive, but the forecast of the log is not the log of the forecast. The
        coefficients' sum near one means volatility is highly persistent; the Newey-West standard errors allow
        for heteroskedastic, autocorrelated errors.
        """

        topics: ClassVar[tuple[str, ...]] = ("realized-volatility",)

        series: Annotated[str, Widget("series", depends_on="data", help="daily realised variance, e.g. rv from Realised volatility; one value per trading day")] = "rv"
        data: Annotated[str, Widget("object", source="Dataset", help="the daily dataset Realised volatility makes, or any holding a daily realised variance")] = "daily"
        log: Annotated[bool, Widget("checkbox", help="model the log of realised variance")] = False
        name: str = "har_fit"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref

            return make_call("finance:har", output=self.name, recorded_by=self.command_id, data=Ref(self.data), series=self.series,
                             log=self.log)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Strategy backtest...")
    @dataclass(frozen=True)
    class BacktestStrategy(Command):
        """Strategy backtest: rebalanced portfolios out of sample, net of costs, compared.

        Every few periods each strategy sets its weights from the window before: equal weights, inverse volatility,
        minimum variance, maximum Sharpe or risk parity. The weights drift between rebalances; trading costs are
        charged on turnover. The window shows each strategy's performance, the growth of 1, and the weights.

        Each strategy uses only the returns before it, so the comparison is out of sample; this is where the
        estimation error of optimised portfolios shows, and the equal-weight portfolio is a hard benchmark to
        beat. Shorter windows adapt faster but estimate more noisily; more frequent rebalancing tracks the
        targets but costs more turnover. Trying many strategies and settings on one history and keeping the best
        overstates its future performance.
        """

        topics: ClassVar[tuple[str, ...]] = ("portfolio-strategies",)

        assets: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the assets' simple returns")]
        strategies: Annotated[str, Widget("text", help="equal_weight, inverse_volatility, min_variance, max_sharpe, risk_parity")] = "equal_weight, min_variance, risk_parity"
        data: Annotated[str, Widget("object", source="Dataset", help="the time-series dataset of returns (and rf); periods missing any are dropped")] = "data"
        window: Annotated[int, Widget("number", min=3, max=10000, help="periods each estimate uses")] = 60
        rebalance: Annotated[int, Widget("number", min=1, max=10000, help="rebalance every this many periods")] = 1
        cost: Annotated[float, Widget("number", min=0.0, max=0.1, step=0.0005, help="cost per unit of turnover (0.001 for 10 bp)")] = 0.0
        rf: Annotated[str, Widget("series", depends_on="data", help="the risk-free rate, for maximum Sharpe and the ratios (empty: 0)")] = ""
        long_only: Annotated[bool, Widget("checkbox", help="weights from 0 to 1 (otherwise -1 to 2)")] = True
        name: str = "strategy_results"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref

            extra: dict[str, Any] = {"rf": self.rf} if self.rf else {}
            return make_call("finance:backtest_strategy", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                             assets=_names(self.assets), strategies=_names(self.strategies), window=self.window, rebalance=self.rebalance,
                             cost=self.cost, long_only=self.long_only, **extra)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Dynamic Nelson-Siegel...")
    @dataclass(frozen=True)
    class FitDynamicNelsonSiegel(Command):
        """Dynamic Nelson-Siegel: the yield-curve factors and their dynamics estimated together, by the Kalman filter.

        The decay, the factors' VAR(1), their shocks and each maturity's measurement error by maximum likelihood,
        starting from the period-by-period fit. The window shows the estimates, the smoothed factors and forecasts.

        Estimating the factors and their dynamics together accounts for the measurement errors and gives
        forecasts of the whole curve, with their uncertainty. A diagonal VAR lets each factor follow its own
        past and is the safer choice for forecasting; the full VAR adds interactions between the factors at the
        cost of more parameters. Like the static fit, the model is not arbitrage-free.
        """

        topics: ClassVar[tuple[str, ...]] = ("yield-curves",)

        series: Annotated[str, Widget("series", depends_on="data", multiple=True, help="yields, shortest maturity first")]
        maturities: Annotated[str, Widget("text", help="each series' maturity in years, in order: 0.25, 1, 2, 5, 10")]
        data: Annotated[str, Widget("object", source="Dataset", help="the dataset of yields, one row per period; the factors are in the yields' units")] = "data"
        var: Annotated[Literal["diagonal", "full"], Widget("choice", help="each factor on its own lag, or on all three")] = "diagonal"
        name: str = "dns"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref
            from pymetrix.errors import SpecificationError

            try:
                maturities = [float(v) for v in self.maturities.replace(";", ",").split(",") if v.strip()]
            except ValueError:
                raise SpecificationError(f"the maturities must be numbers separated by commas, not {self.maturities!r}") from None
            return make_call("finance:dynamic_nelson_siegel", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                             series=_names(self.series), maturities=maturities, var=self.var)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Copula...")
    @dataclass(frozen=True)
    class FitCopula(Command):
        """Copula: how returns move together apart from their own distributions, and how often they crash together.

        Gaussian, t, Clayton, Gumbel, Frank or Joe for any number of series (the Archimedean ones exchangeable, one
        parameter for every pair); BB1, BB6, BB7, BB8, Tawn's asymmetric extreme-value copulas (type 1 and 2), the
        survival and the rotated copulas for two. Fitted to the returns' ranks or to fitted margins (normal, t, skewed t), or with GARCH to each series' standardised residuals; by
        maximum likelihood, from Kendall's tau or jointly with the margins. The window shows the parameters with their
        standard errors, the tail dependence, AIC and BIC, the goodness-of-fit test if asked for, and the data beside
        draws from the copula.

        A copula separates the dependence between returns from their marginal distributions. The Gaussian copula
        has no tail dependence: extreme falls are no more likely to coincide than its correlation implies, which
        can understate joint crash risk; the t copula adds symmetric tail dependence, Clayton, survival Gumbel and
        survival Joe lower-tail and Gumbel and Joe upper-tail dependence, BB1 and BB7 both, BB6 and Tawn's upper
        tail (Tawn's asymmetric: one series' extremes matter more), Frank and BB8 none. Returns with volatility
        clustering are not independent over time, which fitting to ranks assumes; filtering by GARCH first deals with
        that. The goodness-of-fit test (Cramer-von Mises) takes a few seconds by the parametric bootstrap, which
        refits each sample; the multiplier bootstrap (Kojadinovic and Yan) needs no refitting and is much faster,
        but not for the t copula.
        """

        topics: ClassVar[tuple[str, ...]] = ("copulas",)

        series: Annotated[str, Widget("series", depends_on="data", multiple=True, help="two or more series of returns")]
        data: Annotated[str, Widget("object", source="Dataset", help="the time-series dataset of returns; periods missing any of the series are dropped")] = "data"
        family: Annotated[Literal["gaussian", "t", "clayton", "gumbel", "frank", "joe", "bb1", "bb7", "clayton180", "gumbel180", "joe180",
                                  "clayton90", "gumbel90", "joe90", "clayton270", "gumbel270", "joe270", "bb6", "bb8", "tawn1",
                                  "tawn2", "bb1_180", "bb6_180", "bb7_180", "bb8_180", "tawn1_180", "tawn2_180", "bb1_90",
                                  "bb6_90", "bb7_90", "bb8_90", "tawn1_90", "tawn2_90", "bb1_270", "bb6_270", "bb7_270",
                                  "bb8_270", "tawn1_270", "tawn2_270"],
                          Widget("choice", help="the copula: BB1, BB6, BB7, BB8, Tawn's and the rotated ones (180 survival, 90 and 270 for negative dependence) take two series")] = "t"
        margins: Annotated[Literal["empirical", "normal", "t", "skewt"],
                           Widget("choice", help="each series made uniform by its ranks, or by a fitted normal, t or skewed t (with GARCH, normal and t are its errors)")] = "empirical"
        method: Annotated[Literal["ml", "itau", "joint"],
                          Widget("choice", help="maximum likelihood (IFM with fitted margins), Kendall's tau, or margins and copula jointly (fitted margins, no GARCH)")] = "ml"
        garch: Annotated[bool, Widget("checkbox", help="filter each series by a GARCH(1,1) first")] = False
        dist: Annotated[Literal["normal", "t"], Widget("choice", help="with GARCH and rank or skewed-t margins: its errors, normal or Student-t")] = "normal"
        gof: Annotated[bool, Widget("checkbox", help="test the family's fit: Cramer-von Mises with a parametric bootstrap (100 samples) or a multiplier one (1,000)")] = False
        bootstrap: Annotated[Literal["parametric", "multiplier"],
                             Widget("choice", advanced=True, help="the goodness-of-fit p-value: parametric bootstrap (refits each sample) or Kojadinovic and Yan's multiplier bootstrap (fast; not for the t)")] = "parametric"
        name: str = "copula_fit"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref

            return make_call("finance:copula", output=self.name, recorded_by=self.command_id, data=Ref(self.data), series=_names(self.series),
                             family=self.family, **_copula_options(self.garch, self.dist, self.margins, self.method, self.gof, self.bootstrap))

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    def _copula_options(garch: bool, dist: str, margins: str, method: str, gof: bool, bootstrap: str = "parametric") -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if garch:
            extra["garch"] = True
            if margins not in ("normal", "t"):
                extra["dist"] = dist
        if margins != "empirical":
            extra["margins"] = margins
        if method != "ml":
            extra["method"] = method
        if gof:
            extra["gof"] = True
            if bootstrap == "multiplier":
                extra.update(n_boot=1000, bootstrap="multiplier")
        return extra


    @command(menu="Finance/Compare copula families...")
    @dataclass(frozen=True)
    class CompareCopulas(Command):
        """Copula families side by side: which describes the dependence between these returns best.

        Every family that fits the number of series (those for negative dependence when Kendall's tau is negative),
        or those listed, fitted to the same margins: their parameters, log likelihood, AIC and BIC, Kendall's tau and
        tail dependence, and with the goodness-of-fit option the Cramer-von Mises test of each.

        The lowest AIC or BIC picks among the families; the goodness-of-fit test says whether the chosen one is
        itself compatible with the data. Families nest (BB1 includes Clayton and Gumbel), so a two-parameter family's
        gain in likelihood is weighed by the criteria' penalty.
        """

        topics: ClassVar[tuple[str, ...]] = ("copulas",)

        series: Annotated[str, Widget("series", depends_on="data", multiple=True, help="two or more series of returns")]
        data: Annotated[str, Widget("object", source="Dataset", help="the time-series dataset of returns; periods missing any of the series are dropped")] = "data"
        families: Annotated[str, Widget("text", help="families separated by commas (gaussian, t, clayton, gumbel, frank, joe, bb1, bb7, clayton180, ..., bb6, bb8, tawn1, tawn2, bb1_180, tawn1_90, ...); empty for all that fit (BB6, BB8, Tawn and the rotated BB only when listed)")] = ""
        margins: Annotated[Literal["empirical", "normal", "t", "skewt"],
                           Widget("choice", help="each series made uniform by its ranks, or by a fitted normal, t or skewed t")] = "empirical"
        method: Annotated[Literal["ml", "itau", "joint"], Widget("choice", help="maximum likelihood, Kendall's tau, or jointly with fitted margins")] = "ml"
        garch: Annotated[bool, Widget("checkbox", help="filter each series by a GARCH(1,1) first")] = False
        dist: Annotated[Literal["normal", "t"], Widget("choice", help="with GARCH and rank or skewed-t margins: its errors")] = "normal"
        gof: Annotated[bool, Widget("checkbox", help="add each family's Cramer-von Mises goodness-of-fit test (slower)")] = False
        bootstrap: Annotated[Literal["parametric", "multiplier"],
                             Widget("choice", advanced=True, help="the goodness-of-fit p-value: parametric bootstrap or the faster multiplier one (not for the t)")] = "parametric"
        name: str = "copula_comparison"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref

            extra = _copula_options(self.garch, self.dist, self.margins, self.method, self.gof, self.bootstrap)
            if _names(self.families):
                extra["families"] = _names(self.families)
            return make_call("finance:compare_copulas", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                             series=_names(self.series), **extra)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Vine copula...")
    @dataclass(frozen=True)
    class FitVineCopula(Command):
        """Vine copula: the dependence of several returns built pair by pair, each pair with its own copula family.

        An R-vine (or a C-vine or D-vine) of bivariate copulas: the first tree joins the series with the strongest
        dependence (the maximum spanning tree of |Kendall's tau|), each later tree the conditional distributions the
        one before leaves, each pair copula's family chosen by AIC or BIC among the Gaussian, t, Clayton, Gumbel,
        Frank, Joe, BB1, BB7, their rotations and independence (BB6, BB8, Tawn's and the rotated BB families as well
        when listed, or with "all"). Fitted to the returns' ranks, fitted margins or GARCH residuals, tree by tree or
        jointly. The window lists each edge's family, parameters, Kendall's tau and tail dependence, the
        goodness-of-fit test if asked for, and the data beside draws from the vine.

        A single multivariate copula gives every pair the same kind of dependence: the t its symmetric tails, an
        exchangeable Archimedean one parameter for all. A vine lets each pair differ, lower-tail dependence for one,
        none for another, at the cost of many parameters; truncating it after a few trees (independence above)
        keeps it parsimonious. Selection and estimation are sequential, so the later trees inherit the earlier ones'
        errors; the joint option re-estimates all the parameters together (slower). The goodness-of-fit test
        transforms the data by the vine's Rosenblatt transform, which should leave independent uniforms, and refits
        the vine on each of 100 bootstrap samples: it takes a while.
        """

        topics: ClassVar[tuple[str, ...]] = ("copulas",)

        series: Annotated[str, Widget("series", depends_on="data", multiple=True, help="two or more series of returns")]
        data: Annotated[str, Widget("object", source="Dataset", help="the time-series dataset of returns; periods missing any of the series are dropped")] = "data"
        structure: Annotated[Literal["rvine", "cvine", "dvine"], Widget("choice", help="a regular vine, a C-vine (a root in each tree) or a D-vine (a path)")] = "rvine"
        order: Annotated[str, Widget("text", advanced=True, help="a D-vine's path or a C-vine's roots, series separated by commas; empty to select them")] = ""
        families: Annotated[str, Widget("text", advanced=True, help="pair-copula families separated by commas (indep, gaussian, t, clayton, gumbel, frank, joe, bb1, bb7, clayton180, ..., bb6, bb8, tawn1, tawn2, bb1_180, tawn1_90, ...); empty for the usual set, all for every one")] = ""
        criterion: Annotated[Literal["aic", "bic"], Widget("choice", advanced=True, help="how each pair's family is chosen")] = "aic"
        trunc_level: Annotated[int, Widget("number", min=0, max=50, advanced=True, help="independence above this tree; 0 for none")] = 0
        method: Annotated[Literal["sequential", "joint"], Widget("choice", advanced=True, help="tree by tree, or all the parameters together from there (slower)")] = "sequential"
        margins: Annotated[Literal["empirical", "normal", "t", "skewt"],
                           Widget("choice", help="each series made uniform by its ranks, or by a fitted normal, t or skewed t (with GARCH, normal and t are its errors)")] = "empirical"
        garch: Annotated[bool, Widget("checkbox", help="filter each series by a GARCH(1,1) first")] = False
        dist: Annotated[Literal["normal", "t"], Widget("choice", help="with GARCH and rank or skewed-t margins: its errors, normal or Student-t")] = "normal"
        gof: Annotated[bool, Widget("checkbox", help="test the vine's fit: Rosenblatt transform, Cramer-von Mises, parametric bootstrap refitting the vine (100 samples; slow)")] = False
        name: str = "vine_fit"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref

            extra = _copula_options(self.garch, self.dist, self.margins, "ml", False)
            if self.structure != "rvine":
                extra["structure"] = self.structure
            if _names(self.order):
                extra["order"] = _names(self.order)
            if _names(self.families):
                extra["families"] = "all" if _names(self.families) == ["all"] else _names(self.families)
            if self.criterion != "aic":
                extra["criterion"] = self.criterion
            if self.trunc_level:
                extra["trunc_level"] = int(self.trunc_level)
            if self.method != "sequential":
                extra["method"] = self.method
            if self.gof:
                extra["gof"] = True
            return make_call("finance:vine_copula", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                             series=_names(self.series), **extra)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Time-varying copula...")
    @dataclass(frozen=True)
    class FitTVCopula(Command):
        """Time-varying copula: how the dependence between two returns changes over time (Patton 2006).

        The Gaussian or t copula's correlation, or the Clayton, Gumbel or symmetrised Joe-Clayton copula's tail
        dependence, follows an ARMA-type recursion driven by the last periods' uniforms: the product of their normal
        (or t) quantiles for the correlation, their absolute difference for the tails. The window shows the
        recursion's parameters with standard errors, the likelihood against the constant copula's, the filtered
        path by period and next period's forecast.

        Correlations rise in crises; a constant copula averages over calm and turbulent periods. The recursion is a
        filter, not a structural model: its parameters are often imprecise (the tail recursions especially), and the
        likelihood ratio against the constant copula says whether the variation is worth having. For several series'
        correlations together, the DCC copula (Finance -> Copulas -> DCC copula...) is the counterpart.
        """

        topics: ClassVar[tuple[str, ...]] = ("copulas",)

        series: Annotated[str, Widget("series", depends_on="data", multiple=True, help="two series of returns")]
        data: Annotated[str, Widget("object", source="Dataset", help="the time-series dataset of returns; periods missing either series are dropped")] = "data"
        family: Annotated[Literal["gaussian", "t", "clayton", "gumbel", "sjc"],
                          Widget("choice", help="the copula: Gaussian or t with a moving correlation, Clayton, Gumbel or symmetrised Joe-Clayton with moving tails")] = "gaussian"
        lags: Annotated[int, Widget("number", min=1, max=250, advanced=True, help="periods averaged in the recursion's forcing variable (10 in Patton)")] = 10
        margins: Annotated[Literal["empirical", "normal", "t", "skewt"],
                           Widget("choice", help="each series made uniform by its ranks, or by a fitted normal, t or skewed t (with GARCH, normal and t are its errors)")] = "empirical"
        garch: Annotated[bool, Widget("checkbox", help="filter each series by a GARCH(1,1) first")] = False
        dist: Annotated[Literal["normal", "t"], Widget("choice", help="with GARCH and rank or skewed-t margins: its errors, normal or Student-t")] = "normal"
        name: str = "tv_copula_fit"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref

            extra = _copula_options(self.garch, self.dist, self.margins, "ml", False)
            if self.family != "gaussian":
                extra["family"] = self.family
            if int(self.lags) != 10:
                extra["lags"] = int(self.lags)
            return make_call("finance:tv_copula", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                             series=_names(self.series), **extra)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/DCC copula...")
    @dataclass(frozen=True)
    class FitDCCCopula(Command):
        """DCC copula: how the correlations of several returns change over time, apart from their own distributions.

        A Gaussian or t copula whose correlation matrix follows Engle's dynamic conditional correlation recursion on
        the copula's scores (the normal or t quantiles of each series' uniforms): Q_t = (1 - a - b) Qbar + a z z' +
        b Q_(t-1). Fitted by maximum likelihood to the returns' ranks, fitted margins or GARCH residuals. The window
        shows a and b (and the t's degrees of freedom) with standard errors, the likelihood against the constant
        copula's, each pair's correlation by period and next period's.

        The DCC multivariate GARCH models the volatilities and correlations together with normal errors; the copula
        keeps the margins apart (ranks, a skewed t, GARCH residuals) and with the t adds tail dependence that rises
        with the correlation. a measures how much the last period's co-movement moves the correlations, a + b their
        persistence.
        """

        topics: ClassVar[tuple[str, ...]] = ("copulas",)

        series: Annotated[str, Widget("series", depends_on="data", multiple=True, help="two or more series of returns")]
        data: Annotated[str, Widget("object", source="Dataset", help="the time-series dataset of returns; periods missing any of the series are dropped")] = "data"
        family: Annotated[Literal["gaussian", "t"], Widget("choice", help="the copula: Gaussian, or t (tail dependence, one degrees-of-freedom parameter)")] = "gaussian"
        margins: Annotated[Literal["empirical", "normal", "t", "skewt"],
                           Widget("choice", help="each series made uniform by its ranks, or by a fitted normal, t or skewed t (with GARCH, normal and t are its errors)")] = "empirical"
        garch: Annotated[bool, Widget("checkbox", help="filter each series by a GARCH(1,1) first")] = False
        dist: Annotated[Literal["normal", "t"], Widget("choice", help="with GARCH and rank or skewed-t margins: its errors, normal or Student-t")] = "normal"
        name: str = "dcc_copula_fit"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref

            extra = _copula_options(self.garch, self.dist, self.margins, "ml", False)
            if self.family != "gaussian":
                extra["family"] = self.family
            return make_call("finance:dcc_copula", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                             series=_names(self.series), **extra)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Merton distance to default...")
    @dataclass(frozen=True)
    class MertonDistance(Command):
        """Merton model: a firm's asset value and volatility from its equity, and its distance to default.

        Equity is a call on the assets with the debt as strike. Asset volatility is found by the KMV iteration; the
        window shows the distance to default and the probability of default over the horizon, by period.

        The distance to default is the number of standard deviations by which the asset value exceeds the
        default point at the horizon. With the risk-free drift the probability of default is risk-neutral, and
        it tends to be higher than the real-world probability; with the assets' own drift it depends on a
        noisily estimated mean. The model's normal distribution makes its probabilities poorly calibrated in
        levels, so the distance to default is mostly used to rank firms or track one over time.
        """

        topics: ClassVar[tuple[str, ...]] = ("credit-risk",)

        equity: Annotated[str, Widget("series", depends_on="data", help="the market value of equity")]
        debt: Annotated[str, Widget("series", depends_on="data", help="the default point: short-term debt plus half the long-term, say")]
        data: Annotated[str, Widget("object", source="Dataset", help="equity and debt over time")] = "data"
        rate: Annotated[str, Widget("series", depends_on="data", help="the risk-free rate (empty: zero)")] = ""
        horizon: Annotated[float, Widget("number", min=0.05, max=30.0, step=0.25, help="years to the debt's maturity")] = 1.0
        drift: Annotated[Literal["rate", "assets"], Widget("choice", help="the assets' drift: the rate (risk-neutral) or their own mean")] = "rate"
        name: str = "merton_fit"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref

            extra: dict[str, Any] = {"rate": self.rate} if self.rate else {}
            return make_call("finance:merton", output=self.name, recorded_by=self.command_id, data=Ref(self.data), equity=self.equity,
                             debt=self.debt, horizon=self.horizon, drift=self.drift, **extra)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/CDS curve...")
    @dataclass(frozen=True)
    class BootstrapCDS(Command):
        """CDS curve: default intensities bootstrapped from CDS spreads, and the default probabilities they imply.

        Spreads are in basis points a year, at maturities in years; each maturity's CDS is priced at zero by an
        intensity constant since the maturity before.

        The intensity is the default rate per year given survival so far; its rise with maturity shows the
        market pricing more risk later on. Spreads approximately equal the intensity times the loss given
        default, so the recovery assumption scales the implied probabilities: a higher recovery means a higher
        probability for the same spread. The probabilities are risk-neutral and include compensation for bearing
        default risk.
        """

        topics: ClassVar[tuple[str, ...]] = ("credit-risk",)

        maturities: Annotated[str, Widget("text", help="years, rising: 1, 3, 5, 7, 10")] = "1, 3, 5, 7, 10"
        spreads: Annotated[str, Widget("text", help="basis points a year, one for each maturity: 80, 110, 140, 150, 160")] = ""
        recovery: Annotated[float, Widget("number", min=0.0, max=0.99, step=0.05, help="the share recovered at default")] = 0.4
        rate: Annotated[float, Widget("number", min=-0.5, max=1.0, step=0.001, help="the risk-free rate, continuously compounded")] = 0.0
        name: str = "cds"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.errors import SpecificationError

            try:
                m = [float(v) for v in self.maturities.replace(";", ",").split(",") if v.strip()]
                s = [float(v) / 1e4 for v in self.spreads.replace(";", ",").split(",") if v.strip()]
            except ValueError:
                raise SpecificationError("the maturities and spreads must be numbers separated by commas") from None
            return make_call("finance:cds_curve", output=self.name, recorded_by=self.command_id, maturities=m, spreads=s,
                             recovery=self.recovery, rate=self.rate)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Volatility surface...")
    @dataclass(frozen=True)
    class FitVolSurface(Command):
        """Volatility surface: SVI or SABR smiles for each expiry, or the Heston model for all of them.

        The quotes' Black-Scholes implied volatilities are fitted; SVI fits are checked for butterfly and calendar
        arbitrage. The window shows the parameters, the market and model smiles, and the quotes with both volatilities.

        SVI and SABR are flexible curves fitted expiry by expiry, so they interpolate the market smile closely
        but say little across expiries; Heston is one stochastic-volatility model for the whole surface, with
        fewer parameters and a worse fit, but consistent dynamics. A surface with butterfly or calendar
        arbitrage would give negative probabilities or prices that fall with expiry, so it should not be used
        for pricing. Fits to few or illiquid quotes per expiry are fragile.
        """

        topics: ClassVar[tuple[str, ...]] = ("volatility-surfaces",)

        price: Annotated[str, Widget("series", depends_on="data", help="each quote's price; one outside the no-arbitrage bounds gets no volatility")]
        strike: Annotated[str, Widget("series", depends_on="data", help="each quote's strike, in the spot's units")]
        expiry: Annotated[str, Widget("series", depends_on="data", help="each quote's time to expiry in years; SVI and SABR fit one smile per expiry")]
        data: Annotated[str, Widget("object", source="Dataset", help="a dataset of option quotes taken at one time, one row per option")] = "data"
        spot: Annotated[float, Widget("number", min=0.0, max=1e9, help="the underlying's price when the quotes were taken; with rate and dividend it sets the forward")] = 100.0
        rate: Annotated[float, Widget("number", min=-1.0, max=1.0, step=0.001, help="the risk-free rate, continuously compounded")] = 0.0
        dividend: Annotated[float, Widget("number", min=-1.0, max=1.0, step=0.001, help="the dividend yield, continuously compounded")] = 0.0
        kind: Annotated[str, Widget("text", help="call, put, or a series holding 1 for calls and -1 for puts")] = "call"
        model: Annotated[Literal["svi", "sabr", "heston"], Widget("choice", help="SVI or SABR for each expiry, or Heston for the surface")] = "svi"
        beta: Annotated[float, Widget("number", min=0.0, max=1.0, step=0.1, show_if=("model", ("sabr",)),
                                      help="SABR's beta, fixed: 1 for equity, 0.5 is common for rates")] = 1.0
        name: str = "surface"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call
            from pymetrix.extension import Ref

            extra: dict[str, Any] = {"beta": self.beta} if self.model == "sabr" else {}
            return make_call("finance:vol_surface", output=self.name, recorded_by=self.command_id, data=Ref(self.data), price=self.price,
                             strike=self.strike, expiry=self.expiry, spot=self.spot, rate=self.rate, dividend=self.dividend, kind=self.kind,
                             model=self.model, **extra)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


# -- dialogs added 1 November 2026: Black-Litterman, bond measures, a Heston price, Merton at a point, copula portfolio VaR --

def _floats(text: str, what: str) -> list[float]:
    from pymetrix.errors import SpecificationError

    try:
        return [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
    except ValueError:
        raise SpecificationError(f"{what}: give numbers separated by commas") from None


def _views(text: str, assets: list[str]) -> tuple[list[list[float]], list[float]]:
    """Black-Litterman views as written, ``"a - b = 0.02, c = 0.05"`` or ``"0.5 a + 0.5 b = 0.03"``: the rows of P (one
    weight per asset) and q."""
    import re

    from pymetrix.errors import SpecificationError

    rows, q = [], []
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        lhs, sep, rhs = part.partition("=")
        if not sep:
            raise SpecificationError(f"write each view as a combination of assets = a return, e.g. a - b = 0.02 (not {part!r})")
        try:
            value = float(rhs)
        except ValueError:
            raise SpecificationError(f"{rhs.strip()!r} is not a number (in {part!r})") from None
        row = [0.0] * len(assets)
        terms = re.findall(r"([+-]?)\s*(\d*\.?\d*(?:[eE][+-]?\d+)?)\s*\*?\s*([A-Za-z_][A-Za-z0-9_.]*)", lhs)
        if not terms or re.sub(r"[+-]?\s*\d*\.?\d*(?:[eE][+-]?\d+)?\s*\*?\s*[A-Za-z_][A-Za-z0-9_.]*", "", lhs).strip():
            raise SpecificationError(f"cannot read the view {part!r}: write it as weights and asset names, e.g. 0.5 a + 0.5 b = 0.03")
        for sign, weight, name in terms:
            if name not in assets:
                raise SpecificationError(f"{name!r} in the view {part!r} is not one of the assets ({', '.join(assets)})")
            row[assets.index(name)] += (-1.0 if sign == "-" else 1.0) * (float(weight) if weight else 1.0)
        rows.append(row)
        q.append(value)
    return rows, q


with plugin_context(PLUGIN):
    @command(menu="Finance/Black-Litterman...")
    @dataclass(frozen=True)
    class BlackLitterman(Command):
        """Black-Litterman expected returns: the market's implied returns, moved towards your views as far as you trust them.

        The prior is the return that makes the market weights optimal for an investor with risk aversion delta, given the
        assets' covariance (from their returns in the data): pi = delta S w. Views are linear combinations of the assets'
        returns, a - b = 0.02 (a beats b by 2% a period) or c = 0.05; each has an uncertainty, the diagonal of omega. The
        posterior mean blends the prior and the views by their precisions, and the posterior covariance adds the
        remaining uncertainty about the mean. The window shows, by asset, the market weights, the prior (implied) and
        posterior expected returns and the weights optimal for the posterior; the views with their uncertainties and
        their values under the prior and the posterior; a chart of prior against posterior; and the posterior covariance.
        Give the posterior mean to Portfolio for constrained weights.

        Tau scales the uncertainty of the prior mean (S tau): small values keep the posterior near the market's returns.
        With the confidences left empty each view's variance is that of the prior on it (He and Litterman), so tau
        cancels and every view counts as much as the prior; smaller variances pull the posterior further towards the views.
        """

        topics: ClassVar[tuple[str, ...]] = ("factor-models-and-portfolios",)
        results: ClassVar[tuple[tuple[str, str], ...]] = (
            ("Summary", "by asset: market weight, prior (implied) and posterior expected return, posterior sd, optimal weight"),
            ("Views", "each view's value, variance (omega) and its value under the prior and the posterior"),
            ("Graph", "prior against posterior expected returns"),
            ("Posterior covariance", "the posterior covariance of returns, by asset"),
        )

        assets: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the assets' returns per period; their covariance is used")]
        views: Annotated[str, Widget("text", help="views on the returns per period, separated by commas, e.g. a - b = 0.002, c = 0.005")]
        data: Annotated[str, Widget("object", source="Dataset", help="the time-series dataset of returns")] = "data"
        market_weights: Annotated[str, Widget("text", help="the market (or benchmark) weights in the assets' order; empty = equal")] = ""
        confidences: Annotated[str, Widget("text", help="each view's variance (the diagonal of omega): smaller = more confident; "
                                                        "empty = the prior's variance of the view")] = ""
        tau: Annotated[float, Widget("number", advanced=True, min=0.001, max=1.0, step=0.01, help="the uncertainty of the prior mean, relative to the covariance")] = 0.05
        delta: Annotated[float, Widget("number", advanced=True, min=0.0, max=100.0, step=0.5, help="the market's risk aversion, which scales the implied returns")] = 2.5
        name: str = "black_litterman_result"

        def _parts(self) -> dict[str, Any]:
            from pymetrix.errors import SpecificationError

            assets = _names(self.assets)
            if len(assets) < 2:
                raise SpecificationError("choose two or more assets")
            weights = _floats(self.market_weights, "market weights") or [1 / len(assets)] * len(assets)
            if len(weights) != len(assets):
                raise SpecificationError(f"give a market weight for each of the {len(assets)} assets")
            P, q = _views(self.views, assets)
            if not P:
                raise SpecificationError("give at least one view, e.g. a - b = 0.002")
            out: dict[str, Any] = {"market_weights": weights, "views": P, "q": q, "delta": self.delta, "tau": self.tau}
            conf = _floats(self.confidences, "confidences")
            if conf:
                if len(conf) != len(q) or min(conf) <= 0:
                    raise SpecificationError(f"give a positive variance for each of the {len(q)} views")
                out["omega"] = [[c if i == j else 0.0 for j in range(len(conf))] for i, c in enumerate(conf)]
            return out

        def validate(self, session: Session) -> list[str]:
            from pymetrix.errors import SpecificationError

            if self.data not in session.objects:
                return [f"no object named {self.data!r}"]
            try:
                self._parts()
            except SpecificationError as exc:
                return [str(exc)]
            return []

        def to_call(self) -> Any:
            from pymetrix.extension import Ref, make_call

            cov = make_call("pymetrix:dataset.cov", Ref(self.data), names=_names(self.assets))
            return make_call("finance:black_litterman", output=self.name, recorded_by=self.command_id, cov=cov, **self._parts())

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Bond measures...")
    @dataclass(frozen=True)
    class BondMeasures(Command):
        """A coupon bond's price per 100 of face, its yield, Macaulay duration, convexity and Fisher-Weil duration, from
        zero-coupon rates.

        The bond pays the coupon (a year, per 100 of face) in Payments a year until the maturity, and 100 then; each cash
        flow is discounted at the zero rate (continuously compounded) of its date: give one rate for a flat curve, or one
        for each payment. The yield is the single continuously compounded rate that gives the same price; duration is the
        cash flows' average time weighted by their present values at that yield, and convexity the average squared time,
        so that a small change dy in the yield changes the price by about -duration dy + convexity dy^2 / 2 of it. The
        Fisher-Weil duration weights by the zero curve instead (the sensitivity to a parallel shift of the curve). For
        the zero rates of a fitted curve, see Yield curve (its bond method).
        """

        topics: ClassVar[tuple[str, ...]] = ("yield-curves",)

        coupon: Annotated[float, Widget("number", min=0.0, max=100.0, step=0.25, help="the coupon a year, per 100 of face (5 for a 5% bond)")] = 5.0
        years: Annotated[float, Widget("number", min=0.01, max=100.0, step=0.5, help="years to maturity")] = 10.0
        freq: Annotated[int, Widget("number", min=1, max=12, help="payments a year (2: semi-annual coupons)")] = 2
        zeros: Annotated[str, Widget("text", help="zero rates, continuously compounded fractions: one (a flat curve) or one for each payment")] = "0.04"
        name: str = "bond"

        def _flows(self) -> dict[str, Any]:
            from pymetrix.errors import SpecificationError

            from .termstructure import _cash_flows

            if self.years <= 0 or self.freq < 1:
                raise SpecificationError("the maturity must be positive and the coupons at least once a year")
            times, cf = _cash_flows(float(self.coupon), float(self.years), int(self.freq))
            zeros = _floats(self.zeros, "zero rates")
            if len(zeros) == 1:
                zeros = zeros * len(times)
            if len(zeros) != len(times):
                raise SpecificationError(f"give one zero rate, or one for each of the {len(times)} payments")
            return {"times": [round(float(t), 12) for t in times], "cf": [float(c) for c in cf], "zeros": zeros}

        def validate(self, session: Session) -> list[str]:
            from pymetrix.errors import SpecificationError

            try:
                self._flows()
            except SpecificationError as exc:
                return [str(exc)]
            return []

        def to_call(self) -> Any:
            from pymetrix.extension import make_call

            return make_call("finance:bond_measures", output=self.name, recorded_by=self.command_id, **self._flows())

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Heston option price...")
    @dataclass(frozen=True)
    class HestonOptionPrice(Command):
        """The price of a European option in Heston's stochastic-volatility model.

        The variance starts at v0 and reverts to theta at the speed kappa, with volatility of variance sigma and
        correlation rho between its shocks and the price's; the price is Lewis's single integral of the characteristic
        function. A negative rho makes the implied volatility fall with the strike (the equity skew) and sigma makes the
        smile curve; with sigma near zero and v0 = theta the price approaches Black-Scholes at volatility sqrt(theta).
        Fitted parameters come from Volatility surface (Heston). Feller's condition 2 kappa theta > sigma^2 keeps the
        variance away from zero; prices are still defined when it fails.
        """

        topics: ClassVar[tuple[str, ...]] = ("volatility-surfaces", "options")

        spot: Annotated[float, Widget("number", min=0.0, max=1e9, help="the underlying's price now")] = 100.0
        strike: Annotated[float, Widget("number", min=0.0, max=1e9, help="the strike")] = 100.0
        expiry: Annotated[float, Widget("number", min=0.001, max=100.0, step=0.05, help="time to expiry in years")] = 1.0
        rate: Annotated[float, Widget("number", min=-1.0, max=1.0, step=0.001, help="the risk-free rate, continuously compounded")] = 0.03
        dividend: Annotated[float, Widget("number", min=-1.0, max=1.0, step=0.001, help="the dividend yield, continuously compounded")] = 0.0
        v0: Annotated[float, Widget("number", min=0.0, max=10.0, step=0.01, help="the variance now (0.04 for a volatility of 20%)")] = 0.04
        kappa: Annotated[float, Widget("number", min=0.0, max=100.0, step=0.1, help="the speed at which the variance reverts")] = 2.0
        theta: Annotated[float, Widget("number", min=0.0, max=10.0, step=0.01, help="the long-run variance")] = 0.04
        sigma: Annotated[float, Widget("number", min=0.001, max=10.0, step=0.05, help="the volatility of the variance")] = 0.3
        rho: Annotated[float, Widget("number", min=-0.999, max=0.999, step=0.05, help="the correlation of the variance's and the price's shocks")] = -0.7
        kind: Annotated[Literal["call", "put"], Widget("choice", help="call or put")] = "call"
        name: str = "heston_option"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call

            return make_call("finance:heston_price", output=self.name, recorded_by=self.command_id, S=self.spot, K=self.strike, T=self.expiry,
                             r=self.rate, q=self.dividend, v0=self.v0, kappa=self.kappa, theta=self.theta, sigma=self.sigma, rho=self.rho,
                             kind=self.kind)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Merton at a point...")
    @dataclass(frozen=True)
    class MertonPoint(Command):
        """Merton's model for one firm at one date: its asset value and volatility from its equity's value and volatility,
        its leverage, distance to default and (risk-neutral) probability of default.

        Equity is a call on the assets with the debt as strike, so the equity's value and volatility give two equations
        for the assets' value and volatility. The distance to default is how many standard deviations the assets are
        above the debt at the horizon, with the risk-free drift; the probability of default is N(-DD). For a firm over
        time, from its market value of equity, use Merton distance to default.
        """

        topics: ClassVar[tuple[str, ...]] = ("credit-risk",)

        equity: Annotated[float, Widget("number", min=0.0, max=1e12, help="the market value of equity")] = 50.0
        equity_vol: Annotated[float, Widget("number", min=0.001, max=10.0, step=0.01, help="the equity's annual volatility (0.4 for 40%)")] = 0.4
        debt: Annotated[float, Widget("number", min=0.0, max=1e12, help="the default point: short-term debt plus half the long-term, say")] = 80.0
        rate: Annotated[float, Widget("number", min=-1.0, max=1.0, step=0.001, help="the risk-free rate, continuously compounded")] = 0.03
        horizon: Annotated[float, Widget("number", min=0.05, max=30.0, step=0.25, help="years to the debt's maturity")] = 1.0
        name: str = "merton_firm"

        def to_call(self) -> Any:
            from pymetrix.extension import make_call

            return make_call("finance:merton_point", output=self.name, recorded_by=self.command_id, equity=self.equity,
                             equity_vol=self.equity_vol, debt=self.debt, rate=self.rate, horizon=self.horizon)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


    @command(menu="Finance/Copula portfolio VaR...")
    @dataclass(frozen=True)
    class CopulaPortfolioVaR(Command):
        """A portfolio's value at risk and expected shortfall for the next period, simulated from a fitted copula and its
        margins.

        Draws from the copula give the joint ranks of the series' returns; the margins turn them into returns (the
        copula's fitted margins, the empirical quantiles of the returns, or normals fitted to them), and the portfolio's
        return is their weighted sum. VaR is the loss exceeded with probability 1 - level, ES the average loss beyond it.
        Copulas with tail dependence (t, Clayton) give larger losses than the Gaussian at high levels for the same
        correlations. With a GARCH filter, the margins are of the standardised residuals.
        """

        topics: ClassVar[tuple[str, ...]] = ("copulas", "value-at-risk")

        copula: Annotated[str, Widget("object", source="FittedCopula", help="a copula fitted with Finance > Copula")]
        weights: Annotated[str, Widget("text", help="the portfolio's weight on each of the copula's series, in its order, e.g. 0.5, 0.5")]
        level: Annotated[float, Widget("number", min=0.5, max=0.9999, step=0.005, help="the VaR level, e.g. 0.99")] = 0.99
        margins: Annotated[Literal["fitted", "empirical", "normal"],
                           Widget("choice", help="the copula's fitted margins, the empirical quantiles of the returns, or normals")] = "fitted"
        name: str = "copula_var"
        draws: Annotated[int, Widget("number", min=1000, max=10_000_000, step=10_000, advanced=True, help="simulated periods")] = 100_000
        seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="random seed of the simulation")] = 0

        def validate(self, session: Session) -> list[str]:
            from pymetrix.errors import SpecificationError

            fit = session.objects.get(self.copula)
            if not isinstance(fit, FittedCopula):
                return [f"{self.copula!r} is not a fitted copula (Finance > Copula)"]
            try:
                w = _floats(self.weights, "weights")
            except SpecificationError as exc:
                return [str(exc)]
            return [] if len(w) == len(fit.series) else [f"give a weight for each of {', '.join(fit.series)}"]

        def to_call(self) -> Any:
            from pymetrix.extension import Ref, make_call

            return make_call("finance:fitted_copula.portfolio_var", Ref(self.copula), output=self.name, recorded_by=self.command_id,
                             weights=_floats(self.weights, "weights"), level=self.level, n=self.draws,
                             margins=None if self.margins == "fitted" else self.margins, seed=self.seed)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


FittedCopula.next_steps = (("Portfolio VaR…", "CopulaPortfolioVaR", "copula", "finance"),)  # type: ignore[attr-defined] - its window's Next


def _mgarch_views(obj: FittedMGARCH) -> list[Any]:
    from pymetrix.extension import figure_views, frame_table

    return [("Summary", obj.summary()), *figure_views(obj.plot), ("Covariance", obj.covariance_table()),
            ("Correlations", frame_table(obj.correlations, "Conditional correlations by period")),
            ("Volatilities", frame_table(obj.volatilities, "Conditional standard deviations by period"))]


register_view(FittedMGARCH, _mgarch_views, plugin=PLUGIN)


def _risk_views(obj: RiskForecast) -> list[Any]:
    from pymetrix.extension import figure_views, frame_table

    return [("Backtests", obj.backtests()), *figure_views(obj.plot),
            ("Forecasts", frame_table(obj.frame, f"{obj.level:.1%} VaR and ES, one period ahead"))]


register_view(RiskForecast, _risk_views, plugin=PLUGIN)


def _performance_views(obj: Performance) -> list[Any]:
    from pymetrix.extension import figure_views, frame_table

    return [("Measures", obj.table()), *figure_views(obj.plot), ("Cumulative", frame_table(obj.cumulative(), "Growth of 1 invested"))]


register_view(Performance, _performance_views, plugin=PLUGIN)


def _factor_views(obj: FactorModel) -> list[Any]:
    from pymetrix.extension import figure_views

    return [("Summary", obj.summary()), *figure_views(obj.plot)]


register_view(FactorModel, _factor_views, plugin=PLUGIN)


def _portfolio_views(obj: Portfolio) -> list[Any]:
    from pymetrix.extension import figure_views, frame_table

    return [("Weights", obj.weights_table()), *figure_views(obj.plot), ("Frontier", frame_table(obj.frontier(25), "The efficient frontier"))]


register_view(Portfolio, _portfolio_views, plugin=PLUGIN)


def _option_views(obj: OptionValuation) -> list[Any]:
    return [("Value", obj.table())]


register_view(OptionValuation, _option_views, plugin=PLUGIN)


def _smile_views(obj: VolatilitySmile) -> list[Any]:
    from pymetrix.extension import figure_views, frame_table

    return [("Quotes", obj.table()), *figure_views(obj.plot), ("Surface", frame_table(obj.surface(), "Implied volatility by strike and expiry"))]


register_view(VolatilitySmile, _smile_views, plugin=PLUGIN)


def _curve_views(obj: YieldCurve) -> list[Any]:
    from pymetrix.extension import figure_views, frame_table

    return [("Fit", obj.table()), *figure_views(obj.plot), ("Factors", frame_table(obj.factors, "Factors by period"))]


register_view(YieldCurve, _curve_views, plugin=PLUGIN)


def _contribution_views(obj: RiskContributions) -> list[Any]:
    from pymetrix.extension import figure_views

    return [("Contributions", obj.table()), *figure_views(obj.plot)]


register_view(RiskContributions, _contribution_views, plugin=PLUGIN)


def _black_litterman_views(obj: BlackLittermanPosterior) -> list[Any]:
    from pymetrix.extension import figure_views, frame_table

    out = [("Summary", obj.summary())]
    if obj.views is not None:
        out.append(("Views", obj.views_table()))
    out += [*figure_views(obj.plot), ("Posterior covariance", frame_table(obj.posterior_cov, "Posterior covariance of returns"))]
    if obj.views is not None:
        out.append(("View weights", frame_table(obj.views, "The views' weights on the assets (P)")))
    prov = obj.provenance
    if prov is not None:
        from pymetrix.output import Table

        rows = [["method", f"{prov.method['type']}: {prov.method['estimation']}"], ["omega", str(prov.method["omega"])],
                ["delta", float(prov.spec["delta"])], ["tau", float(prov.spec["tau"])], ["assets", ", ".join(prov.spec["assets"])],
                ["inputs fingerprint", prov.data_fingerprint], ["pymetrix", prov.package_version], ["created", str(prov.created)[:19]]]
        out.append(("Provenance", Table(["", "value"], rows, title="Provenance")))
    return out


register_view(BlackLittermanPosterior, _black_litterman_views, plugin=PLUGIN)


def _event_views(obj: AbnormalReturns) -> list[Any]:
    from pymetrix.extension import figure_views, frame_table

    return [("Tests", obj.tests()), *figure_views(obj.plot), ("Average", frame_table(obj.average(), "AAR and CAAR by period from the event")),
            ("Events", frame_table(obj.ar, "Abnormal returns by period from the event")), ("CAR", frame_table(obj.car(), "Each event's CAR"))]


register_view(AbnormalReturns, _event_views, plugin=PLUGIN)


def _har_views(obj: FittedHAR) -> list[Any]:
    from pymetrix.extension import figure_views, frame_table

    return [("Summary", obj.summary()), *figure_views(obj.plot), ("Forecast", frame_table(obj.forecast(22), "Realised variance forecast"))]


register_view(FittedHAR, _har_views, plugin=PLUGIN)


def _backtest_views(obj: StrategyBacktest) -> list[Any]:
    from pymetrix.extension import figure_views, frame_table

    return [("Performance", obj.performance()), *figure_views(obj.plot), ("Returns", frame_table(obj.returns, "Net returns by period")),
            ("Turnover", frame_table(obj.turnover, "Turnover by period"))]


register_view(StrategyBacktest, _backtest_views, plugin=PLUGIN)


def _dns_views(obj: FittedDNS) -> list[Any]:
    from pymetrix.extension import figure_views, frame_table

    return [("Summary", obj.summary()), *figure_views(obj.plot), ("Factors", frame_table(obj.factors, "Smoothed factors")),
            ("Forecast", frame_table(obj.forecast(12), "Yields forecast"))]


register_view(FittedDNS, _dns_views, plugin=PLUGIN)


def _copula_views(obj: FittedCopula) -> list[Any]:
    from pymetrix.extension import figure_views

    return [("Summary", obj.summary()), *figure_views(obj.plot)]


register_view(FittedCopula, _copula_views, plugin=PLUGIN)


def _vine_views(obj: FittedVine) -> list[Any]:
    from pymetrix.extension import figure_views

    return [("Summary", obj.summary()), *figure_views(obj.plot)]


register_view(FittedVine, _vine_views, plugin=PLUGIN)


def _tv_copula_views(obj: FittedTVCopula) -> list[Any]:
    from pymetrix.extension import figure_views, frame_table

    return [("Summary", obj.summary()), *figure_views(obj.plot), ("Path", frame_table(obj.path, "The dependence parameter by period"))]


register_view(FittedTVCopula, _tv_copula_views, plugin=PLUGIN)


def _dcc_copula_views(obj: FittedDCCCopula) -> list[Any]:
    from pymetrix.extension import figure_views, frame_table

    return [("Summary", obj.summary()), *figure_views(obj.plot), ("Path", frame_table(obj.path, "The correlations by period"))]


register_view(FittedDCCCopula, _dcc_copula_views, plugin=PLUGIN)


def _merton_views(obj: MertonModel) -> list[Any]:
    from pymetrix.extension import figure_views, frame_table

    return [("Summary", obj.summary()), *figure_views(obj.plot), ("By period", frame_table(obj.frame, "Assets, distance and probability of default"))]


register_view(MertonModel, _merton_views, plugin=PLUGIN)


def _cds_views(obj: CDSCurve) -> list[Any]:
    from pymetrix.extension import figure_views

    return [("Curve", obj.table()), *figure_views(obj.plot)]


register_view(CDSCurve, _cds_views, plugin=PLUGIN)


def _surface_views(obj: FittedVolSurface) -> list[Any]:
    from pymetrix.extension import figure_views, frame_table

    views = [("Summary", obj.summary()), *figure_views(obj.plot), *[("Surface", f) for _, f in figure_views(obj.plot_surface)],
             ("Quotes", frame_table(obj.quotes, "Market and model implied volatility"))]
    if obj.model == "svi":
        views.append(("Arbitrage", obj.arbitrage()))
    return views


register_view(FittedVolSurface, _surface_views, plugin=PLUGIN)


def _workspace_and_menu() -> None:
    from pymetrix.extension import MenuLayout, Workspace

    register("workspaces", Workspace("finance", "Finance", ("Finance", "Test"), ("Objects", "History", "Problems"),
                                     ("Returns", "MeasurePerformance", "ValueAtRisk", "OptimizePortfolio", "BacktestStrategy", "FitYieldCurve"),
                                     "returns, risk, portfolios, options and yield curves"), name="finance", plugin=PLUGIN)
    from pymetrix.extension import ReportTemplate

    register("report_templates", ReportTemplate(
        "finance_summary", "Finance results: performance, risk and its backtests, portfolios, factor tests and curves.",
        ("Measures", "Performance", "Backtests", "Weights", "Contributions", "Tests", "Summary", "Fit", "Curve", "Graph")),
        name="finance_summary", plugin=PLUGIN)
    # grouped by task in submenus: 28 items in one list ran off a laptop's screen
    register("menus", MenuLayout("Finance", ["Returns", "MeasurePerformance", None,
                                             ("Risk", ["ValueAtRisk", "DecomposeRisk", "FitMGARCH", None, "RealizedVolatility", "FitHAR"]),
                                             ("Copulas", ["FitCopula", "CompareCopulas", "FitVineCopula", "FitTVCopula", "FitDCCCopula", None,
                                                          "CopulaPortfolioVaR"]),
                                             ("Portfolios", ["FactorRegression", "OptimizePortfolio", "BlackLitterman", "BacktestStrategy",
                                                             "StudyEvents"]),
                                             ("Options", ["PriceOption", "HestonOptionPrice", "ImpliedVolatility", "FitVolSurface"]),
                                             ("Fixed income", ["FitYieldCurve", "FitDynamicNelsonSiegel", "BondMeasures"]),
                                             ("Credit", ["MertonDistance", "MertonPoint", "BootstrapCDS"])]), name="finance", plugin=PLUGIN)


_workspace_and_menu()


def _help() -> None:
    from pymetrix.extension import add_help_topic as add_topic

    add_topic("value-at-risk", "Value at risk and expected shortfall", """
        Value at risk (VaR) at 99% is the loss a period's return exceeds with probability 1%; expected shortfall (ES)
        is the average loss when it does. Forecasts are judged by their violations, the periods whose loss exceeded
        VaR: about 1% of them, at no predictable times (Kupiec, Christoffersen, the dynamic quantile test), and
        losses beyond VaR that average ES (McNeil-Frey). Historical simulation assumes the recent past repeats; the
        normal understates fat tails, which Student-t and Cornish-Fisher allow for; GARCH lets volatility cluster.
        """, ("Christoffersen, P. F. (1998). Evaluating interval forecasts. International Economic Review 39(4), 841-862. doi:10.2307/2527341",
              "Engle, R. F. and Manganelli, S. (2004). CAViaR. Journal of Business & Economic Statistics 22(4), 367-381. doi:10.1198/073500104000000370",
              "Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. Journal of Derivatives 3(2), 73-84. doi:10.3905/jod.1995.407942",
              "McNeil, A. J. and Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach. Journal of Empirical Finance 7(3-4), 271-300. doi:10.1016/s0927-5398(00)00012-8"))
    add_topic("multivariate-garch", "Multivariate GARCH", """
        Volatilities move together, and so do correlations: they rise in crises, just when diversification is wanted.
        A multivariate GARCH models the covariance of several returns given the past. CCC keeps each series' GARCH
        volatility and a constant correlation; DCC lets the correlation move, in two steps that scale to many series;
        BEKK models the whole covariance at once, positive definite by construction, with the long-run covariance set
        to the sample's (variance targeting). The covariance forecast gives a portfolio's risk for the next period.
        """, ("Bollerslev, T. (1990). Modelling the coherence in short-run nominal exchange rates: a multivariate generalized ARCH model. Review of Economics and Statistics 72(3), 498-505. doi:10.2307/2109358",
              "Engle, R. F. (2002). Dynamic conditional correlation. Journal of Business & Economic Statistics 20(3), 339-350. doi:10.1198/073500102288618487",
              "Engle, R. F. and Kroner, K. F. (1995). Multivariate simultaneous generalized ARCH. Econometric Theory 11(1), 122-150. doi:10.1017/s0266466600009063",
              "Engle, R. F. and Sheppard, K. (2001). Theoretical and empirical properties of dynamic conditional correlation multivariate GARCH. NBER Working Paper 8554. doi:10.3386/w8554"))

    add_topic("returns-and-performance", "Returns and performance", """
        Log returns add over time and simple returns add across a portfolio; for small returns they are close. A
        series' performance is judged by its annual return and volatility, the Sharpe ratio (mean excess return over
        volatility; Lo's standard error shows how uncertain it is), the Sortino ratio (which counts only losses), the
        maximum drawdown (the worst fall from a peak) and, against a benchmark, alpha, beta, tracking error and the
        information ratio.
        """, ("Lo, A. W. (2002). The statistics of Sharpe ratios. Financial Analysts Journal 58(4), 36-52. doi:10.2469/faj.v58.n4.2453",
              "Sharpe, W. F. (1994). The Sharpe ratio. Journal of Portfolio Management 21(1), 49-58. doi:10.3905/jpm.1994.409501",
              "Sortino, F. A. and Price, L. N. (1994). Performance measurement in a downside risk framework. Journal of Investing 3(3), 59-64. doi:10.3905/joi.3.3.59"))
    add_topic("factor-models-and-portfolios", "Factor models and portfolios", """
        A factor model regresses an asset's excess return on factor returns: the market (CAPM), or the market, size
        and value (Fama-French). Beta is the exposure to a factor; alpha is the return the factors do not explain.
        The GRS test asks whether all the assets' alphas are zero together. Mean-variance portfolios choose weights
        for the least variance, the highest Sharpe ratio, or the least variance at a target return; together the
        last trace the efficient frontier.
        """, ("Fama, E. F. and French, K. R. (1993). Common risk factors in the returns on stocks and bonds. Journal of Financial Economics 33(1), 3-56. doi:10.1016/0304-405x(93)90023-5",
              "Gibbons, M. R., Ross, S. A. and Shanken, J. (1989). A test of the efficiency of a given portfolio. Econometrica 57(5), 1121-1152. doi:10.2307/1913625",
              "Markowitz, H. (1952). Portfolio selection. Journal of Finance 7(1), 77-91. doi:10.1111/j.1540-6261.1952.tb01525.x"))

    add_topic("options", "Options", """
        A European option is exercised only at expiry and has Black, Scholes and Merton's price; an American one at any
        time, priced here by a binomial tree. The Greeks say how the price moves: delta and gamma with the underlying,
        vega with volatility, theta with time, rho with the rate. Implied volatility is the volatility that gives a
        quoted price; across strikes and expiries it is rarely constant, and the smile or skew it makes is how the
        market departs from Black-Scholes.
        """, ("Black, F. and Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy 81(3), 637-654. doi:10.1086/260062",
              "Cox, J. C., Ross, S. A. and Rubinstein, M. (1979). Option pricing: a simplified approach. Journal of Financial Economics 7(3), 229-263. doi:10.1016/0304-405x(79)90015-1",
              "Hull, J. C. (2018). Options, Futures, and Other Derivatives, 10th edition. Pearson.",
              "Merton, R. C. (1973). Theory of rational option pricing. Bell Journal of Economics 4(1), 141-183. doi:10.2307/3003143"))
    add_topic("yield-curves", "Yield curves", """
        Nelson and Siegel describe a yield curve by three factors: the level (long yields), the slope (short less long)
        and the curvature (a hump in the middle), with a decay saying where the hump sits; Svensson adds a second hump.
        Diebold and Li fit the factors period by period with the decay fixed, so that they become time series to model
        and forecast: here each by an AR(1), or with the VAR and ARIMA tools on the factors table.
        """, ("Diebold, F. X. and Li, C. (2006). Forecasting the term structure of government bond yields. Journal of Econometrics 130(2), 337-364. doi:10.1016/j.jeconom.2005.03.005",
              "Nelson, C. R. and Siegel, A. F. (1987). Parsimonious modeling of yield curves. Journal of Business 60(4), 473-489. doi:10.1086/296409",
              "Svensson, L. E. O. (1994). Estimating and interpreting forward interest rates: Sweden 1992-1994. NBER Working Paper 4871. doi:10.3386/w4871"))

    add_topic("risk-contributions", "Risk contributions", """
        A portfolio's VaR is not the sum of its assets' VaRs, because they diversify one another. Marginal VaR is how
        much the portfolio's VaR rises with a little more of an asset; component VaR, its weight times that, splits the
        portfolio's VaR into parts that add up exactly (Euler's theorem), and likewise for expected shortfall. A large
        component says where the risk is; a negative one marks a hedge.
        """, ("Jorion, P. (2007). Value at Risk, 3rd edition. McGraw-Hill.",
              "Tasche, D. (1999). Risk contributions and performance measurement. Working paper, Technische Universitat "
              "Munchen."))
    add_topic("event-studies", "Event studies of returns", """
        An event study asks whether returns were unusual around events such as announcements. Each asset's normal
        return is estimated before the event (the market model, a factor model or the mean); abnormal returns are the
        actual less the normal, averaged across events (AAR) and cumulated (CAAR). The cross-sectional t test is simple;
        Patell's standardises by each event's prediction error; Boehmer, Musumeci and Poulsen's also allows the event to
        change the variance; the rank test does not assume normal returns.
        """, ("Boehmer, E., Musumeci, J. and Poulsen, A. B. (1991). Event-study methodology under conditions of event-induced variance. Journal of Financial Economics 30(2), 253-272. doi:10.1016/0304-405x(91)90032-f",
              "Corrado, C. J. (1989). A nonparametric test for abnormal security-price performance in event studies. Journal of Financial Economics 23(2), 385-395. doi:10.1016/0304-405x(89)90064-0",
              "MacKinlay, A. C. (1997). Event studies in economics and finance. Journal of Economic Literature 35(1), 13-39. jstor:2729691",
              "Patell, J. M. (1976). Corporate forecasts of earnings per share and stock price behavior. Journal of Accounting Research 14(2), 246-276. doi:10.2307/2490543"))

    add_topic("realized-volatility", "Realised volatility", """
        With many returns a day, the day's variance can be measured rather than modelled: realised variance, the sum
        of squared intraday returns, measures it plus any jumps; bipower variation, from products of neighbouring
        absolute returns, measures it without them, and their difference tests for a jump (Barndorff-Nielsen and
        Shephard). The HAR model forecasts realised variance from its daily, weekly and monthly averages, a simple
        stand-in for long memory that is hard to beat.
        """, ("Andersen, T. G., Bollerslev, T., Diebold, F. X. and Labys, P. (2003). Modeling and forecasting realized volatility. Econometrica 71(2), 579-625. doi:10.1111/1468-0262.00418",
              "Barndorff-Nielsen, O. E. and Shephard, N. (2006). Econometrics of testing for jumps in financial economics using bipower variation. Journal of Financial Econometrics 4(1), 1-30. doi:10.1093/jjfinec/nbi022",
              "Corsi, F. (2009). A simple approximate long-memory model of realized volatility. Journal of Financial Econometrics 7(2), 174-196. doi:10.1093/jjfinec/nbp001",
              "Huang, X. and Tauchen, G. (2005). The relative contribution of jumps to total price variance. Journal of Financial Econometrics 3(4), 456-499. doi:10.1093/jjfinec/nbi025"))

    add_topic("portfolio-strategies", "Portfolio strategies", """
        A backtest asks how a way of choosing weights would have done: at each rebalance it estimates from the past
        only, then holds the weights, which drift with returns, until the next. Trading back to the targets costs
        money in proportion to turnover. Simple rules (equal weights, inverse volatility, risk parity, which gives each
        asset the same share of risk) often beat mean-variance optimisation out of sample, because estimated means are
        so noisy; Black-Litterman tempers them by starting from the returns implied by market weights. Its result
        shows, by asset, those implied (prior) returns beside the posterior ones, the views with their uncertainty and
        how far the posterior moved each, and the weights that the posterior makes optimal for an investor with the
        market's risk aversion; unpacked as mu, sigma it gives the posterior mean and covariance.
        """, ("Black, F. and Litterman, R. (1992). Global portfolio optimization. Financial Analysts Journal 48(5), 28-43. doi:10.2469/faj.v48.n5.28",
              "DeMiguel, V., Garlappi, L. and Uppal, R. (2009). Optimal versus naive diversification: how inefficient is the 1/N portfolio strategy? Review of Financial Studies 22(5), 1915-1953. doi:10.1093/rfs/hhm075",
              "Maillard, S., Roncalli, T. and Teiletche, J. (2010). The properties of equally weighted risk contribution portfolios. Journal of Portfolio Management 36(4), 60-70. doi:10.3905/jpm.2010.36.4.060",
              "Spinu, F. (2013). An algorithm for computing risk parity weights. SSRN 2297383. doi:10.2139/ssrn.2297383"))

    add_topic("copulas", "Copulas", """
        A copula is the dependence between variables once each has been made uniform by its own distribution, so the
        margins and the dependence can be modelled apart. The Gaussian copula has no tail dependence: joint crashes
        are as rare as the correlation alone implies. The t copula, Clayton (lower tail), Gumbel and Joe (upper tail),
        BB1 and BB7 (both tails, unequal), BB6 and Tawn's asymmetric extreme-value copulas (upper tail) let extremes
        come together; Frank and BB8 have none; the survival (180 degree) copulas swap the tails and the 90 and 270
        degree rotations describe negative dependence. The margins are the
        series' ranks or fitted distributions (normal, t, Hansen's skewed t), estimated first (IFM) or jointly with the
        copula. AIC and BIC compare families on the same data; the Cramer-von Mises test with a parametric (or the
        faster multiplier) bootstrap asks whether a family fits at all. Fitted to GARCH-filtered returns and simulated, a copula gives a
        portfolio's VaR and ES without assuming joint normality. A vine copula builds the dependence of many series
        from bivariate copulas, a family for each pair, tree by tree, and is tested through its Rosenblatt transform,
        which should leave independent uniforms; a time-varying copula lets the correlation or the tail dependence
        of two series move with the recent past (Patton's recursion), and the DCC copula the correlations of
        several (Engle's recursion on the copula's scores).
        """, ("Aas, K., Czado, C., Frigessi, A. and Bakken, H. (2009). Pair-copula constructions of multiple "
              "dependence. Insurance: Mathematics and Economics 44, 182-198.",
              "Bedford, T. and Cooke, R. M. (2002). Vines: a new graphical model for dependent random variables. "
              "Annals of Statistics 30, 1031-1068.",
              "Dissmann, J., Brechmann, E. C., Czado, C. and Kurowicka, D. (2013). Selecting and estimating regular vine "
              "copulae and application to financial returns. Computational Statistics and Data Analysis 59, 52-69.",
              "Engle, R. F. (2002). Dynamic conditional correlation. Journal of Business & Economic Statistics 20(3), 339-350. doi:10.1198/073500102288618487",
              "Genz, A. (1992). Numerical computation of multivariate normal probabilities. Journal of Computational and "
              "Graphical Statistics 1, 141-149.",
              "Genest, C., Ghoudi, K. and Rivest, L.-P. (1995). A semiparametric estimation procedure of dependence "
              "parameters in multivariate families of distributions. Biometrika 82(3), 543-552. doi:10.1093/biomet/82.3.543",
              "Genest, C., Remillard, B. and Beaudoin, D. (2009). Goodness-of-fit tests for copulas: a review and a "
              "power study. Insurance: Mathematics and Economics 44, 199-213.",
              "Hansen, B. E. (1994). Autoregressive conditional density estimation. International Economic Review 35, 705-730.",
              "Hofert, M., Maechler, M. and McNeil, A. J. (2012). Likelihood inference for Archimedean copulas in high "
              "dimensions under known margins. Journal of Multivariate Analysis 110, 133-150.",
              "Joe, H. (2014). Dependence Modeling with Copulas. Chapman and Hall/CRC.",
              "Joe, H. and Xu, J. J. (1996). The estimation method of inference functions for margins for multivariate "
              "models. Technical Report 166, Department of Statistics, University of British Columbia.",
              "Kojadinovic, I. and Yan, J. (2011). A goodness-of-fit test for multivariate multiparameter copulas based on "
              "multiplier central limit theorems. Statistics and Computing 21(1), 17-30. doi:10.1007/s11222-009-9142-y",
              "McNeil, A. J., Frey, R. and Embrechts, P. (2015). Quantitative Risk Management, revised edition. Princeton.",
              "Patton, A. J. (2006). Modelling asymmetric exchange rate dependence. International Economic Review 47(2), 527-556. doi:10.1111/j.1468-2354.2006.00387.x",
              "Rosenblatt, M. (1952). Remarks on a multivariate transformation. Annals of Mathematical Statistics 23(3), 470-472. doi:10.1214/aoms/1177729394",
              "Schepsmeier, U. (2019). A goodness-of-fit test for regular vine copula models. Econometric Reviews 38(1), 25-46.",
              "Tawn, J. A. (1988). Bivariate extreme value theory: models and estimation. Biometrika 75(3), 397-415. doi:10.1093/biomet/75.3.397"))

    add_topic("credit-risk", "Credit risk", """
        Merton's model sees equity as a call option on the firm's assets with the debt as strike: from the equity's
        value and volatility it backs out the assets', and the distance to default is how many standard deviations
        the assets are above the debt. Market-implied default intensities come instead from credit default swaps:
        the spread pays for protection, so spread / (1 - recovery) is roughly the yearly chance of default.
        """, ("Crosbie, P. and Bohn, J. (2003). Modeling default risk. Working paper, Moody's KMV.",
              "Hull, J. C. and White, A. (2000). Valuing credit default swaps I. Journal of Derivatives 8(1), 29-40. doi:10.3905/jod.2000.319115",
              "Merton, R. C. (1974). On the pricing of corporate debt: the risk structure of interest rates. Journal of Finance 29(2), 449-470. doi:10.1111/j.1540-6261.1974.tb03058.x",
              "Vassalou, M. and Xing, Y. (2004). Default risk in equity returns. Journal of Finance 59(2), 831-868. doi:10.1111/j.1540-6261.2004.00650.x"))

    add_topic("finance", "Finance: which tool for which question", """
        **How has it done?** Finance → Returns (from prices), then Performance: return, volatility, Sharpe, drawdowns.
        **What drives it?** Factor model: CAPM or Fama-French betas and alphas, with the GRS test.
        **What should I hold?** Portfolio (minimum variance, maximum Sharpe, a target), and Strategy backtest to see how
        rebalancing would have done, after costs.
        **How much could I lose?** Value at risk (with backtests), Multivariate GARCH and Copula for joint risk, Risk
        contributions for where the risk comes from; Realised volatility and HAR with intraday data.
        **Did the news move the price?** Event study of returns.
        **Rates and credit:** Yield curve, Dynamic Nelson-Siegel, and bond prices from a curve; Merton distance to
        default and CDS curve.
        **Options:** Option price and Greeks, Implied volatility, and Volatility surface (SVI, SABR, Heston).
        The finance tutorial (the package's docs/tutorial.md, and as a notebook) walks through them on example data, and
        Report → the finance_summary template collects the results.
        """, (("Campbell, J. Y., Lo, A. W. and MacKinlay, A. C. (1997). The Econometrics of Financial Markets. Princeton University Press. doi:10.2307/j.ctt7skm5"),
              "Tsay, R. S. (2010). Analysis of Financial Time Series, 3rd ed. Wiley. doi:10.1002/9780470644560",
              "Cochrane, J. H. (2005). Asset Pricing, revised ed. Princeton University Press."))

    add_topic("volatility-surfaces", "Volatility surfaces", """
        Implied volatility varies with strike (the smile or skew) and with expiry (the term structure). SVI describes
        each expiry's smile by five parameters and can be checked for arbitrage: butterfly (a negative density) and
        calendar (total variance falling with expiry). SABR is a stochastic-volatility model whose smile has Hagan's
        closed form, common for rates. Heston models the variance itself as mean-reverting, with one set of parameters
        for every expiry, priced through its characteristic function.
        """, ("Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives. Presentation, Global Derivatives & Risk Management, Madrid.",
              "Gatheral, J. and Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. Quantitative Finance 14(1), 59-71. doi:10.1080/14697688.2013.819986",
              "Hagan, P. S., Kumar, D., Lesniewski, A. S. and Woodward, D. E. (2002). Managing smile risk. Wilmott Magazine 1, 84-108.",
              "Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. Review of Financial Studies 6(2), 327-343. doi:10.1093/rfs/6.2.327",
              "Lewis, A. L. (2000). Option Valuation under Stochastic Volatility. Finance Press."))


_help()
