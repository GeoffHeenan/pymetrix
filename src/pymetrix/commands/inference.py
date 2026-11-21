"""Dialogs for single-equation tests that need the data as well as the fit, and for heterogeneous panels:

- Test > Weak-instrument tests: Anderson-Rubin (a test, or the confidence set by inversion), Stock-Wright S, and the C
  (difference-in-Sargan) test, on an equation estimated by IV or GMM;
- Test > Non-nested tests: Davidson-MacKinnon's J and the Cox test, of one fitted equation against another;
- Test > Gregory-Hansen cointegration test (beside Johansen cointegration);
- Estimate > Quantile process (beside Coefficient plot): a quantile regression's coefficients across quantiles;
- Estimate > Panel mean group and CCE: mean group, CCE mean group, CCE pooled and the pooled mean group (with its
  Hausman test in the result's window).

The tests take the equation, method and sample of the chosen fit, so the generated code calls the function with the
equation written out (``anderson_rubin(Equation('...'), data, ...)``) and replays without the fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from ..errors import SpecificationError
from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _numbers(text: str) -> list[float]:
    try:
        return [float(v) for v in _names(text.replace(";", ","))]
    except ValueError:
        raise SpecificationError(f"give numbers separated by commas, not {text!r}") from None


def _values(text: str) -> dict[str, float]:
    """``"educ=0.1, exper=0.05"`` -> {"educ": 0.1, "exper": 0.05}."""
    out = {}
    for part in _names(text):
        key, sep, value = part.partition("=")
        if not sep or not key.strip():
            raise SpecificationError(f"write each value as name=number, e.g. educ=0.1 (not {part!r})")
        try:
            out[key.strip()] = float(value)
        except ValueError:
            raise SpecificationError(f"{value.strip()!r} is not a number (in {part!r})") from None
    return out


def _fit_span(session: Session, fit: Any, data: str, start: str, end: str) -> Any:
    """The sample a test uses: Start and End if given, else the fit's own estimation sample when the data are a time
    series (so that the test sees the fit's observations), else the data's active sample (None)."""
    if start or end:
        return (start or None, end or None)
    sample = getattr(getattr(fit, "provenance", None), "sample", None)
    dataset = session.objects.get(data)
    if sample and getattr(dataset, "time_axis", None) is not None:
        return tuple(str(s) for s in sample)
    return None


def _fitted(session: Session, name: str, what: str) -> list[str]:
    from ..estimation import FittedEquation

    if name not in session.objects:
        return [f"no object named {name!r}"]
    return [] if isinstance(session.objects[name], FittedEquation) else [f"{name} is not a fitted equation: {what}"]


# -- weak instruments -----------------------------------------------------------------------------------------------

_IV_TESTS = Literal["anderson_rubin", "anderson_rubin_set", "stock_wright", "c_test"]
_AR = ("anderson_rubin", "anderson_rubin_set")


@command(menu="Test/Weak-instrument tests...")
@dataclass(frozen=True)
class InstrumentTests(Command):
    """Tests for instrumental-variable equations that stay valid however weak the instruments: Anderson-Rubin and
    Stock-Wright; and the C test of whether some of the instruments are valid.

    Choose an equation estimated by IV or GMM (Estimate > Equation, method IV, LIML or GMM) and the data it was fitted
    to; the test uses its equation, its instruments and its sample.

    Anderson-Rubin tests that the coefficient on one endogenous regressor takes a given value, by regressing
    y - value * regressor on the other regressors and the excluded instruments: an F test (or, with robust errors, a
    Wald test) that the instruments' coefficients are zero. Its size is right whatever the instruments' strength,
    where the 2SLS t test over-rejects with weak instruments. Inverting it over a grid of values gives the
    Anderson-Rubin confidence set: the values not rejected. With weak instruments it can be wide, unbounded or empty
    (empty means the over-identifying restrictions fail), which is the honest answer.

    Stock and Wright's S tests that every parameter of a GMM equation takes the given values: the continuously updated
    GMM objective there, chi-squared with as many degrees of freedom as instruments, again whatever their strength.

    The C (difference-in-Sargan) test asks whether the suspect instruments are valid, given that the others are: Hansen's
    J with all the instruments less J without the suspect ones (GMM fits), chi-squared with as many degrees of freedom as
    suspect instruments. A small p-value says the suspect ones are correlated with the error.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import GMM, IV, Dataset, Equation, anderson_rubin, anderson_rubin_set, c_test, stock_wright

        card = Dataset.example("card")
        eq = Equation("lwage ~ educ + exper + black + south")
        anderson_rubin(eq, card, "educ", ["nearc4", "nearc2"], 0.1)                  # is the return to schooling 10%?
        anderson_rubin_set(eq, card, "educ", ["nearc4", "nearc2"], [i / 100 for i in range(0, 41)])  # the values not rejected
        gmm = GMM(instruments=["nearc4", "nearc2", "exper", "black", "south"])
        c_test(eq, card, gmm, ["nearc2"])                                          # is nearc2 a valid instrument?
        stock_wright(eq, card, gmm, {"const": 4.0, "educ": 0.1, "exper": 0.04, "black": -0.2, "south": -0.1})
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("statistic, df, p-value", ("the test's statistic: F (or chi-squared with robust errors) for Anderson-Rubin, "
                                    "chi-squared for Stock-Wright and C")),
        ("value, in set", "the confidence set: each value of the grid, its Anderson-Rubin p-value and whether it is in the set"),
    )
    topics: ClassVar[tuple[str, ...]] = ("instrumental-variables", "diagnostic-tests")

    fit: Annotated[str, Widget("object", source="FittedEquation",
                               help="an equation estimated by IV or GMM: its equation, instruments and sample are used")]
    test: Annotated[_IV_TESTS, Widget("choice", help="Anderson-Rubin test or confidence set (IV or GMM fits); Stock-Wright S "
                                                     "and the C test (GMM fits)")] = "anderson_rubin"
    data: Annotated[str, Widget("object", source="Dataset", help="the data the equation was fitted to")] = "data"
    endog: Annotated[str, Widget("text", show_if=("test", _AR),
                                 help="the endogenous regressor whose coefficient is tested; empty = the fit's first")] = ""
    value: Annotated[float, Widget("number", min=-1e6, max=1e6, step=0.01, show_if=("test", ("anderson_rubin",)),
                                   help="the coefficient's value under the null")] = 0.0
    grid: Annotated[str, Widget("text", advanced=True, show_if=("test", ("anderson_rubin_set",)),
                                help="values to test: low, high, points (e.g. -1, 1, 81), or a list of more than three")] = "-1, 1, 81"
    alpha: Annotated[float, Widget("number", min=0.001, max=0.5, step=0.01, show_if=("test", ("anderson_rubin_set",)),
                                   help="significance level: the set has coverage 1 - alpha")] = 0.05
    cov: Annotated[Literal["classic", "white"], Widget("choice", show_if=("test", _AR),
                                                       help="classic: the exact F test; white: robust to heteroskedasticity")] = "classic"
    values: Annotated[str, Widget("text", show_if=("test", ("stock_wright",)),
                                  help="every parameter's value under the null, e.g. const=4, educ=0.1, exper=0.04")] = ""
    suspect: Annotated[str, Widget("text", show_if=("test", ("c_test",)),
                                   help="the instruments whose validity is tested, among the fit's, e.g. nearc2")] = ""
    name: str = "iv_test"
    instruments: Annotated[str, Widget("text", advanced=True,
                                       help="Anderson-Rubin: the excluded instruments; empty = the fit's")] = ""
    start: Annotated[str, Widget("period", advanced=True, help="first period; empty = the fit's sample")] = ""
    end: Annotated[str, Widget("period", advanced=True, help="last period; empty = the fit's sample")] = ""

    def _grid(self) -> list[float]:
        values = _numbers(self.grid)
        if len(values) == 3 and float(values[2]).is_integer() and values[2] >= 2:
            import numpy as np

            return [round(float(v), 12) for v in np.linspace(values[0], values[1], int(values[2]))]
        if len(values) < 2:
            raise SpecificationError("give the grid as low, high, points, or a list of values")
        return values

    def _excluded(self, fit: Any) -> list[str]:
        from ..estimation import GMM

        if _names(self.instruments):
            return _names(self.instruments)
        if isinstance(fit.method, GMM):  # GMM lists every instrument: the excluded ones are those not among the regressors
            return [z for z in fit.method.instruments if z not in set(fit.params.names)]
        return list(getattr(fit.method, "instruments", None) or [])

    def _endog(self, fit: Any) -> str:
        if self.endog:
            return self.endog
        from ..estimation import GMM

        endog = list(getattr(fit.method, "endog", None) or [])
        if isinstance(fit.method, GMM):  # GMM: the regressors that do not instrument themselves
            endog = [n for n in fit.params.names if n != "const" and n not in set(fit.method.instruments)]
        if not endog:
            raise SpecificationError("name the endogenous regressor to test (Endogenous regressor)")
        return endog[0]

    def validate(self, session: Session) -> list[str]:
        from ..estimation import GMM

        problems = _fitted(session, self.fit, "choose an equation estimated by IV or GMM")
        if self.data not in session.objects:
            problems.append(f"no object named {self.data!r}")
        if problems:
            return problems
        fit = session.objects[self.fit]
        try:
            if self.test in _AR:
                if not self._excluded(fit):
                    return [f"{self.fit} has no excluded instruments: fit it by IV or GMM, or list them under Instruments"]
                self._endog(fit)
                if self.test == "anderson_rubin_set":
                    self._grid()
            elif not isinstance(fit.method, GMM):
                return [f"the {'Stock-Wright' if self.test == 'stock_wright' else 'C'} test needs an equation estimated by GMM"]
            elif self.test == "stock_wright" and not _values(self.values):
                return ["give every parameter's value, e.g. const=4, educ=0.1"]
            elif self.test == "c_test" and not _names(self.suspect):
                return ["name the suspect instruments"]
        except SpecificationError as exc:
            return [str(exc)]
        return []

    def to_call(self, session: Session) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        fit = session[self.fit]
        common = {"output": self.name, "recorded_by": self.command_id, "equation": fit.equation, "data": Ref(self.data),
                  "span": _fit_span(session, fit, self.data, self.start, self.end)}
        if self.test == "anderson_rubin":
            return make_call("pymetrix:anderson_rubin", endog=self._endog(fit), instruments=self._excluded(fit), value=self.value,
                             cov=self.cov, **common)
        if self.test == "anderson_rubin_set":
            return make_call("pymetrix:anderson_rubin_set", endog=self._endog(fit), instruments=self._excluded(fit),
                             grid=self._grid(), alpha=self.alpha, cov=self.cov, **common)
        if self.test == "stock_wright":
            return make_call("pymetrix:stock_wright", method=fit.method, values=_values(self.values), **common)
        return make_call("pymetrix:c_test", method=fit.method, suspect=_names(self.suspect), **common)

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:  # the code needs the fit's equation and method: Session writes to_call(session).to_code()
        return f"# {self.test} on {self.fit}: see the recorded call"


# -- non-nested tests ------------------------------------------------------------------------------------------------


@command(menu="Test/Non-nested tests (J, Cox)...")
@dataclass(frozen=True)
class NonNestedTest(Command):
    """Test one linear equation against another that does not nest it, for the same dependent variable: Davidson and
    MacKinnon's J test or the Cox test (Pesaran's form).

    The null is that the first equation is right. The J test adds the second equation's fitted values to the first and
    tests their coefficient (a t test); the Cox test compares the two equations' error variances with what the first
    predicts for the second (standard normal; large negative values reject). Run the test both ways: both may reject
    (neither is adequate) or neither (the data cannot choose). Both equations use the observations they share.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation, cox_test, j_test

        data = Dataset.example("sample_macro")
        a, b = Equation("c ~ y + c[-1]"), Equation("c ~ y + y[-1]")
        j_test(a, b, data)        # is a right, against b?
        j_test(b, a, data)        # and the other way
        cox_test(a, b, data)
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("statistic, p-value", "J: the t statistic on the second equation's fitted values; Cox: a standard normal statistic"),
    )
    topics: ClassVar[tuple[str, ...]] = ("diagnostic-tests", "least-squares")

    first: Annotated[str, Widget("object", source="FittedEquation", help="the equation under the null: is it right?")]
    second: Annotated[str, Widget("object", source="FittedEquation",
                                  help="the alternative: a linear equation for the same dependent variable, not nested in the first")]
    test: Annotated[Literal["j", "cox"], Widget("choice", help="Davidson-MacKinnon's J (a t test) or the Cox test (Pesaran's form)")] = "j"
    data: Annotated[str, Widget("object", source="Dataset", help="the data both were fitted to")] = "data"
    name: str = "nonnested"
    start: Annotated[str, Widget("period", advanced=True, help="first period; empty = the first fit's sample")] = ""
    end: Annotated[str, Widget("period", advanced=True, help="last period; empty = the first fit's sample")] = ""

    def validate(self, session: Session) -> list[str]:
        problems = _fitted(session, self.first, "choose a fitted equation") + _fitted(session, self.second, "choose a fitted equation")
        problems += [] if self.data in session.objects else [f"no object named {self.data!r}"]
        if not problems and self.first == self.second:
            problems.append("choose two different equations")
        return problems

    def to_call(self, session: Session) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        first = session[self.first]
        return make_call(f"pymetrix:{self.test}_test", output=self.name, recorded_by=self.command_id, eq1=first.equation,
                         eq2=session[self.second].equation, data=Ref(self.data),
                         span=_fit_span(session, first, self.data, self.start, self.end))

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        return f"# {self.test}_test of {self.first} against {self.second}: see the recorded call"


# -- Gregory-Hansen ----------------------------------------------------------------------------------------------------


@command(menu="Test/Gregory-Hansen cointegration test...")
@dataclass(frozen=True)
class GregoryHansenTest(Command):
    """Gregory and Hansen's test of no cointegration against cointegration with one break at an unknown date.

    For each candidate break date (all but the trimmed ends of the sample) the cointegrating regression is fitted with
    a level shift, a level shift and a trend, or a regime shift (level and slopes), and the ADF statistic of its
    residuals computed; the test takes the smallest. Its p-value and critical values are simulated from independent
    random walks of the same length (Replications); the asymptotic ones of Gregory and Hansen (1996) are in the details
    for up to four regressors. The date of the smallest statistic is reported as the break, but the test is not a
    reliable way to date it. Rejecting does not show that there was a break: a relation without one rejects too, so run
    the usual Engle-Granger or Johansen test as well.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, gregory_hansen

        data = Dataset.example("sample_macro").derive(lc="log(c)", ly="log(y)")
        gregory_hansen(data, "lc", ["ly"], model="regime", lags=2)      # a shift in level and slope at an unknown date
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("statistic, p-value", "the smallest ADF statistic over the break dates, its simulated p-value"),
        ("critical values", "simulated 1%, 5% and 10% values; the asymptotic ones in the details"),
        ("break", "the date of the smallest statistic"),
    )
    topics: ClassVar[tuple[str, ...]] = ("cointegration",)

    y: Annotated[str, Widget("series", depends_on="data", help="the dependent series, in levels")]
    x: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the regressors, in levels (up to four for the asymptotic values)")]
    data: Annotated[str, Widget("object", source="Dataset", help="time series in levels, I(1)")] = "data"
    model: Annotated[Literal["level", "trend", "regime"],
                     Widget("choice", help="what breaks: the level, the level with a trend, or the level and the slopes")] = "level"
    lags: Annotated[int, Widget("number", min=0, max=24, help="lagged differences in the ADF regression of the residuals")] = 1
    name: str = "gregory_hansen_test"
    trim: Annotated[float, Widget("number", min=0.05, max=0.45, step=0.05, advanced=True,
                                  help="the share of the sample at each end where no break is considered")] = 0.15
    reps: Annotated[int, Widget("number", min=50, max=100_000, step=100, advanced=True,
                                help="random-walk replications for the p-value and critical values")] = 500
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="random seed of the replications")] = 0
    start: Annotated[str, Widget("period", advanced=True)] = ""
    end: Annotated[str, Widget("period", advanced=True)] = ""

    def validate(self, session: Session) -> list[str]:
        problems = [] if self.data in session.objects else [f"no object named {self.data!r}"]
        return problems + ([] if self.y and _names(self.x) else ["choose the dependent series and at least one regressor"])

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        span = (self.start or None, self.end or None) if self.start or self.end else None
        return make_call("pymetrix:gregory_hansen", output=self.name, recorded_by=self.command_id, data=Ref(self.data), y=self.y,
                         x=_names(self.x), model=self.model, lags=self.lags, trim=self.trim, reps=self.reps, seed=self.seed, span=span)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


# -- quantile process ------------------------------------------------------------------------------------------------


@command(menu="Estimate/Quantile process...")
@dataclass(frozen=True)
class QuantileProcessPlot(Command):
    """A quantile regression's coefficients across quantiles, each with its confidence band and the least-squares
    estimate as a dashed line.

    Choose a fitted equation (a quantile regression, or any linear one): its equation and sample are re-estimated by
    quantile regression at each of the quantiles listed, and by least squares for comparison. A coefficient that
    changes across quantiles means the regressor moves the spread or the shape of the distribution of the dependent
    variable, not only its centre; one whose band covers the least-squares line at every quantile is consistent with a
    pure location shift. The bands are pointwise, so do not read them as a joint test.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation, quantile_process

        card = Dataset.example("card")
        qp = quantile_process(Equation("lwage ~ educ + exper + black"), card, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        qp.table()       # coefficients by quantile, and least squares
        qp.plot(["educ"])
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Graph", "each coefficient across quantiles with its (1 - alpha) band, and the least-squares estimate dashed"),
        ("Table", "the coefficients at each quantile, and least squares"),
    )
    topics: ClassVar[tuple[str, ...]] = ("quantile-regression",)

    fit: Annotated[str, Widget("object", source="FittedEquation",
                               help="a quantile regression (or any linear equation): its equation and sample are used")]
    data: Annotated[str, Widget("object", source="Dataset", help="the data it was fitted to")] = "data"
    quantiles: Annotated[str, Widget("text", help="the quantiles, between 0 and 1, separated by commas")] = "0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9"
    vcov: Annotated[Literal["robust", "iid"], Widget("choice", help="standard errors: robust (kernel sandwich) or assuming iid errors")] = "robust"
    alpha: Annotated[float, Widget("number", min=0.01, max=0.5, step=0.05, help="the bands cover 1 - alpha")] = 0.1
    name: str = "quantile_process_result"
    start: Annotated[str, Widget("period", advanced=True, help="first period; empty = the fit's sample")] = ""
    end: Annotated[str, Widget("period", advanced=True, help="last period; empty = the fit's sample")] = ""

    def validate(self, session: Session) -> list[str]:
        problems = _fitted(session, self.fit, "choose a fitted equation") + ([] if self.data in session.objects else [f"no object named {self.data!r}"])
        try:
            qs = _numbers(self.quantiles)
        except SpecificationError as exc:
            return problems + [str(exc)]
        if not qs or any(not 0 < q < 1 for q in qs):
            problems.append("give quantiles between 0 and 1")
        return problems

    def to_call(self, session: Session) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        fit = session[self.fit]
        return make_call("pymetrix:quantile_process", output=self.name, recorded_by=self.command_id, eq=fit.equation,
                         data=Ref(self.data), quantiles=_numbers(self.quantiles), span=_fit_span(session, fit, self.data, self.start, self.end),
                         vcov=self.vcov, alpha=self.alpha)

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        return f"# quantile_process of {self.fit}: see the recorded call"


# -- heterogeneous panels --------------------------------------------------------------------------------------------


@command(menu="Estimate/Panel mean group and CCE...")
@dataclass(frozen=True)
class FitPanelMeanGroup(Command):
    """Heterogeneous panels with long time series: the mean group, common correlated effects (mean group or pooled)
    and the pooled mean group estimators.

    Mean group (Pesaran and Smith) fits least squares unit by unit and averages the coefficients, with standard errors
    from their spread across units: consistent when the slopes differ by unit, where pooled estimators are not. CCE
    (Pesaran 2006) adds the cross-section averages of the dependent variable and the regressors to each unit's
    regression, absorbing unobserved common factors whatever their loadings; its mean group averages the units'
    slopes, its pooled version pools them. The pooled mean group (Pesaran, Shin and Smith) estimates an error-correction
    ARDL with a common long run and unit-specific adjustment and short-run dynamics, by maximum likelihood; its window
    has the Hausman test of the common long run against the mean group of the units' own long runs (a small p-value
    rejects the common long run).

    Each unit needs more periods than coefficients. The data must be a panel (an entity axis and a time axis).
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        import numpy as np
        import pandas as pd
        from pymetrix import Dataset, cce, mean_group, pooled_mean_group

        rng = np.random.default_rng(0)
        rows = []
        for unit in range(8):                           # eight countries, 40 quarters each, slopes that differ
            x = np.cumsum(rng.normal(size=40))
            rows.append(pd.DataFrame({"country": f"c{unit}", "time": pd.period_range("2010Q1", periods=40, freq="Q"),
                                      "x": x, "y": 1 + (0.5 + 0.1 * unit) * x + rng.normal(size=40)}))
        panel = Dataset.from_frame(pd.concat(rows), entity="country", time="time")
        mean_group(panel, "y", ["x"])                   # each unit's least squares, averaged
        cce(panel, "y", ["x"])                          # CCE mean group; cce(..., pooled=True) for CCE pooled
        pmg = pooled_mean_group(panel, "y", ["x"], lags=1)
        pmg.hausman()                                   # the common long run against the mean group's
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", "the averaged (or pooled) coefficients with their standard errors"),
        ("Units", "each unit's coefficients (pooled mean group: each unit's adjustment and short run)"),
        ("Hausman", "pooled mean group: the common long run against the mean group of the units' own long runs"),
    )
    topics: ClassVar[tuple[str, ...]] = ("panel-data", "cointegration")

    y: Annotated[str, Widget("series", depends_on="data", help="the dependent series")]
    x: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the regressors")]
    data: Annotated[str, Widget("object", source="Dataset", help="a panel with long time series: an entity and a time axis")] = "data"
    method: Annotated[Literal["mean group", "cce mean group", "cce pooled", "pooled mean group"],
                      Widget("choice", help="unit-by-unit slopes averaged; with cross-section averages (CCE), averaged or pooled; "
                                            "or a common long run (pooled mean group)")] = "mean group"
    trend: Annotated[bool, Widget("checkbox", show_if=("method", ("mean group",)), help="a linear trend in each unit's regression")] = False
    lags: Annotated[int, Widget("number", min=1, max=12, show_if=("method", ("pooled mean group",)),
                                help="lags of the ARDL (the short run in differences has one fewer)")] = 1
    name: str = "panel_mg"

    def validate(self, session: Session) -> list[str]:
        data = session.objects.get(self.data)
        if data is None:
            return [f"no object named {self.data!r}"]
        problems = [] if self.y and _names(self.x) else ["choose the dependent series and at least one regressor"]
        if not getattr(data, "entity_axes", None):
            problems.append(f"{self.data} is not a panel: it needs an entity axis (Data > Panel > Wide to panel)")
        return problems

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        common = {"output": self.name, "recorded_by": self.command_id, "data": Ref(self.data), "y": self.y, "x": _names(self.x)}
        if self.method == "mean group":
            return make_call("pymetrix:mean_group", trend=self.trend, **common)
        if self.method == "pooled mean group":
            return make_call("pymetrix:pooled_mean_group", lags=self.lags, **common)
        return make_call("pymetrix:cce", pooled=self.method == "cce pooled", **common)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
