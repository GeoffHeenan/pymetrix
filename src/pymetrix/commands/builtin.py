"""Built-in commands for the forms UI."""

from __future__ import annotations

import re
from dataclasses import MISSING, dataclass, fields, is_dataclass
from typing import Annotated, Any, ClassVar, Literal

from ..errors import SpecificationError
from ..data import Dataset
from ..estimation import ARMA, OLS, Equation, Logit, Poisson, Probit
from ..estimation.registry import MAIN_FORM_FIELDS, labels_of
from ..forms import build, parse_options
from ..plugins import installed, registry
from ..var import VAR, Cholesky, LongRun
from ..models import Model
from ..stats import available_tests
from .registry import Command, MigrationError, Widget, command, existing_series, migration
from .session import Session


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _span(start: str, end: str) -> tuple[str | None, str | None] | None:
    """An empty box means no limit on that side (None), not a period called ""."""
    return (start or None, end or None) if start or end else None


def _span_code(start: str, end: str) -> str:
    return f", span={_span(start, end)!r}" if start or end else ""


def _params_code(params: str) -> str:
    return f", params={_names(params)!r}" if _names(params) else ""


_MAIN_METHOD_FIELDS = MAIN_FORM_FIELDS  # set on the Estimate form, not in Method settings


def series_used(cmd: Any) -> frozenset[str] | None:
    """The series of its dataset that a dialog step reads, from its formula and series fields
    (None: not known, so a change to any series of the dataset counts)."""
    from ..expr import ExpressionError, parse_expression

    if not hasattr(cmd, "data") or not any(hasattr(cmd, f) for f in ("formula", "variables", "variable")):
        return None
    used: set[str] = set()
    try:
        if getattr(cmd, "formula", ""):
            text = cmd.formula
            if "~" in text or "=" in text:
                used |= set(Equation(text, params=_names(getattr(cmd, "params", ""))).variables())
            else:
                used |= set(parse_expression(text).variables())
        for field_name in ("variables", "variable", "endog", "weights", "cluster"):
            used |= set(_names(str(getattr(cmd, field_name, "") or "")))
        for field_name in ("exog", "exog_coint", "instruments"):
            for expr in _names(str(getattr(cmd, field_name, "") or "")):
                used |= set(parse_expression(expr).variables())
    except (ExpressionError, ValueError):
        return None
    return frozenset(used)


def _code(obj: Any) -> str:
    """Python that rebuilds a configuration dataclass, showing only non-default fields."""
    from ..estimation import Cluster

    if isinstance(obj, Cluster):
        return f"Cluster({', '.join(map(repr, obj.levels))})"
    shown = []
    for f in fields(obj):
        value = getattr(obj, f.name)
        default = f.default if f.default is not MISSING else (f.default_factory() if f.default_factory is not MISSING else MISSING)
        if value != default:
            if is_dataclass(value):
                text = _code(value)
            else:
                text = repr(list(value) if isinstance(value, tuple) else value)
            shown.append(f"{f.name}={text}")
    return f"{type(obj).__name__}({', '.join(shown)})"


@command(menu="File/Load data...")
@dataclass(frozen=True)
class LoadData(Command):
    """Read a data file into the workspace as a dataset.

    Parquet files written by PyMetrix carry their own axes and metadata. For CSV and Excel files, name the column of
    dates or periods (Time) and, for panels and cross-sections, the column identifying entities (Entity), and choose
    the frequency.

    The time column and frequency matter beyond labels: they fix what a lag such as ``x[-1]`` means, the defaults
    that follow the frequency (the HP smoothing parameter, band-pass cycle lengths) and the periods forecasts are
    dated in. Check the dataset's overview after loading: a frequency read wrongly, or a fiscal year
    taken for a calendar year, shifts every date without raising an error.
    """

    path: Annotated[str, Widget("file", source="Data files (*.parquet *.csv *.xlsx *.xls);;All files (*)",
                                help="a Parquet, CSV or Excel file; PyMetrix's Parquet files keep their axes and metadata")]
    time: Annotated[str, Widget("column", depends_on="path", help="column of dates or periods (not needed for our Parquet files)")] = ""
    entity: Annotated[str, Widget("column", depends_on="path", help="column identifying countries, firms, households... (panels, cross-sections)")] = ""
    freq: Annotated[Literal["", "Y", "Y-JUN", "Q", "M", "W", "D"], Widget("choice", help="period frequency of the time column; Y-JUN = July-June fiscal year")] = ""
    name: str = "data"

    def _kw(self) -> dict[str, str]:
        return {k: v for k, v in (("time", self.time), ("entity", self.entity), ("freq", self.freq)) if v}

    def to_call(self) -> Any:
        from ..calls import make_call

        return make_call("pymetrix:dataset.read", output=self.name, recorded_by=self.command_id, source=self.path, **self._kw())

    def run(self, session: Session) -> Dataset:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="File/Load example data...")
@dataclass(frozen=True)
class LoadExample(Command):
    """Load an example dataset installed with PyMetrix, such as the simulated quarterly economy used in the course labs.

    Each example suits particular methods: ``sample_macro`` and ``sample_monthly`` for time series, VARs and
    nowcasting; ``mroz``, ``card``, ``randhie`` and ``anes96`` for limited-dependent-variable and IV models;
    ``wage_panel``, ``grunfeld`` and ``jobtraining`` for panels. The simulated ``policy_panel``, ``rd_sample`` and
    ``synth_panel`` have known true effects (2.0, 1.5 and -3.0), so you can check that an estimator recovers them.
    Plugins can add examples of their own; the dataset's notes say where real data come from.
    """

    topics: ClassVar[tuple[str, ...]] = ()

    dataset: Annotated[str, Widget("dataset", help="sample_macro (a quarterly economy), or a cross-section or panel "
                                                   "example; the simulated ones have known effects")] = "sample_macro"
    name: str = "data"

    def to_call(self) -> Any:
        from ..calls import make_call

        return make_call("pymetrix:dataset.example", output=self.name, recorded_by=self.command_id, name=self.dataset)

    def run(self, session: Session) -> Dataset:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="File/Open result...")
@dataclass(frozen=True)
class OpenResult(Command):
    """Open a result saved to a file (File → Save result…, or ``save_result`` in code): its tables, figures, summary
    and provenance as they were saved. It can be read and copied, not estimated again. Projects keep a copy of the file.

    The provenance records how the result was produced: the data and their vintages, the sample, the settings and the
    package version. When a re-run gives different numbers, compare the provenance first: a revised data vintage or a
    different sample is the usual reason.
    """

    topics: ClassVar[tuple[str, ...]] = ()

    path: Annotated[str, Widget("file", source="pymetrix results (*.pymxr);;All files (*)",
                                help="a .pymxr file written by File > Save result... or save_result in code")]
    name: str = "result"

    def run(self, session: Session) -> Any:
        from pathlib import Path

        from ..saved import result_from_bytes

        blob = session.opened.get(self.path)  # a project's copy, when it was opened from one
        if blob is None:
            blob = Path(self.path).read_bytes()
        result = result_from_bytes(blob, source=self.path, name=Path(self.path).stem)
        session.opened[self.path] = blob
        session.objects[self.name] = result
        return result

    def to_code(self) -> str:
        return f"{self.name} = load_result({self.path!r})"


@command(menu="Data/Add results to dataset...")
@dataclass(frozen=True)
class AddToDataset(Command):
    """Add residuals, fitted values, volatilities, forecast columns, components or factor scores to a dataset as new series.

    Choose the source object and the item to add. The dataset is replaced by an extended copy; give the new series a
    name or accept source_item.
    """

    source: Annotated[str, Widget("object", source="FittedEquation,FittedVAR,FittedVECM,FittedStateSpace,FittedPCA,FittedFactorModel,Dataset",
                                  help="a fit, a forecast, principal components or a factor analysis, or another dataset")]
    item: Annotated[str, Widget("item", depends_on="source", help="residuals, fitted values, volatility, or a column")]
    data: Annotated[str, Widget("object", source="Dataset", help="the dataset to add the series to (it is replaced by the extended one)")] = "data"
    series_name: Annotated[str, Widget("text", help="name of the new series; empty = the source's name and the item's")] = ""
    replace: Annotated[bool, Widget("checkbox", help="replace a series of the same name, if there is one")] = False

    schema_version: ClassVar[int] = 2  # 2: replace (schema-1 steps replaced silently)

    def _name(self) -> str:
        kind, _, key = self.item.partition(":")
        return self.series_name or (f"{self.source}_{key}" if kind in ("extra", "column") else f"{self.source}_{self.item.replace(':', '_')}")

    def validate(self, session: Session) -> list[str]:
        from .session import result_items

        if self.source not in session.objects or self.data not in session.objects:
            return ["choose a source object and a dataset"]
        if self.item not in result_items(session[self.source]):
            return [f"{self.item!r} is not available from {self.source!r}"]
        if not self._name().isidentifier():
            return [f"{self._name()!r} is not a valid series name"]
        return existing_series(session, self.data, [self._name()], self.replace)

    def run(self, session: Session) -> Any:
        from .session import result_series

        series = result_series(session[self.source], self.item)
        result = session[self.data].with_series(**{self._name(): series})
        session.objects[self.data] = result
        return result

    def to_code(self) -> str:
        from .session import result_code

        return f"{self.data} = {self.data}.with_series({self._name()}={result_code(self.source, self.item)})"


@command(menu="Model/Define equation...")
@dataclass(frozen=True)
class DefineEquation(Command):
    """Define an equation or identity without estimating it, for use in a model.

    Write `~` for a stochastic equation and `=` for an identity. Lags are written `x[-1]`. For an explicit
    (nonlinear) form, name every parameter under Params.
    """

    topics: ClassVar[tuple[str, ...]] = ('model-solution',)

    formula: Annotated[str, Widget("formula", help="e.g. c ~ y + c[-1]   (use = for identities)")]
    params: Annotated[str, Widget("text", help="comma-separated parameter names; empty for the linear form")] = ""
    name: str = "eq"
    label: Annotated[str, Widget("text", advanced=True, help="the equation's name in a model, e.g. IS; empty = none")] = ""
    starts: Annotated[str, Widget("text", advanced=True, help="the parameters' starting values, e.g. rho=0.5, beta=0.9")] = ""
    bounds: Annotated[str, Widget("text", advanced=True, help="the parameters' bounds, e.g. rho=(0, 1); beta=(0, 0.999)")] = ""
    priors: Annotated[str, Widget("text", advanced=True,
                                  help="priors for estimation in a model, e.g. rho=Beta(0.8, 0.1): mean and standard deviation")] = ""

    def validate(self, session: Session) -> list[str]:
        try:
            self._equation()
        except (ValueError, SpecificationError) as exc:
            return [str(exc)]
        return []

    def _equation(self) -> Equation:
        from ..metadata import Param

        names, starts, bounds = _names(self.params), _calibration(self.starts), _bounds(self.bounds)
        unknown = sorted((set(starts) | set(bounds)) - set(names))
        if unknown:
            raise SpecificationError(f"starting values and bounds are for the parameters named under Params, not {', '.join(unknown)}")
        params = [Param(n, **({"start": starts[n]} if n in starts else {}), **({"bounds": bounds[n]} if n in bounds else {}))
                  if n in starts or n in bounds else n for n in names]
        options: dict[str, Any] = {"name": self.label.strip()} if self.label.strip() else {}
        if self.priors.strip():
            options["priors"] = _priors(self.priors)
        return Equation(self.formula, params=params, **options)

    def to_call(self) -> Any:
        from ..calls import VALUE, make_call

        return make_call(VALUE, output=self.name, recorded_by=self.command_id, value=self._equation())

    def run(self, session: Session) -> Equation:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


_FAMILIES = {"logit": Logit, "probit": Probit, "poisson": Poisson}
_LIMITED_FAMILIES = ("negative binomial", "zero-inflated poisson", "ordered logit", "ordered probit", "multinomial logit", "tobit")


def _kind(*kinds: str):
    """Labels of the registered methods of these kinds, looked up when the form is shown."""
    return lambda: tuple(label for kind in kinds for label in labels_of(kind))


def _with_variance() -> list[str]:
    return registry("variances").labels()


def _message(exc: Exception) -> str:
    return str(exc.args[0]) if exc.args else str(exc)


@command(menu="Estimate/Equation...")
@dataclass(frozen=True)
class FitEquation(Command):
    """Estimate a single equation by least squares, instrumental variables or maximum likelihood.

    Write the equation with `~`, e.g. `dlog(c) ~ dlog(y) + c[-1]`; a constant is included unless you write `- 1`.
    Lags are `x[-1]`; `fe(id)` adds fixed effects and `C(region)` categorical dummies. `seasonal()` adds seasonal
    dummies (one per quarter or month, less one with a constant: `seasonal[Q2]` ...), and `C(quarter)` or `C(month)`
    takes the quarter or month from the dates. For a nonlinear equation write its explicit form and list its
    parameters under Params.

    Choose the Method, then its settings below it. Instrumental-variable methods ask for the endogenous regressors
    and the excluded instruments. Maximum likelihood (ML) covers logit, probit and Poisson families, ARMA errors and
    GARCH-type variance models; the variance model's own settings appear under Variance settings. Covariance sets
    how standard errors are computed; the sample is limited by Start and End.
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import OLS, Dataset, Equation

        data = Dataset.example("sample_macro")
        eq = Equation("dlog(c) ~ dlog(y) + dlog(c)[-1]")   # a constant is included
        fit = eq.fit(data)                                 # least squares, the dialog's default method
        print(fit.summary())       # coefficients, standard errors, t and p-values; R-squared, criteria, Durbin-Watson
        robust = eq.fit(data, method=OLS(cov="hac"))       # the same estimates, Newey-West (HAC) standard errors
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", ("coefficients with standard errors (as Covariance chose), t and p-values; the notes give the "
                     "method, sample, R-squared, criteria and Durbin-Watson")),
        ("Statistics", ("fit.stats: r2, se_regression, aic, bic, hq and the rest; the criteria are totals, not divided "
                        "by n as in EViews")),
        ("Diagnostics", ("Breusch-Godfrey, Breusch-Pagan, Jarque-Bera and RESET on the residuals: a small p-value "
                         "rejects the null in the last column")),
        ("Correlogram", ("for time series, the residuals' AC, PAC and Ljung-Box Q by lag; small p-values mean serial "
                         "correlation remains")),
        ("Graph, Residuals, Q-Q plot", ("actual and fitted with the residuals, the residual series, and the residuals "
                                        "against the normal distribution")),
        ("Other methods", ("IV, GMM, ML and the rest add their own statistics and tests (and logit, probit and counts a "
                           "Marginal effects tab): see the chosen method's page")),
    )

    topics: ClassVar[tuple[str, ...]] = ('least-squares', 'instrumental-variables', 'maximum-likelihood', 'garch')

    formula: Annotated[str, Widget("formula")]
    params: Annotated[str, Widget("text")] = ""
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    method: Annotated[str, Widget("extension", source="methods", help="estimation methods whose libraries are installed")] = "OLS"
    family: Annotated[Literal["gaussian", "logit", "probit", "poisson", "negative binomial", "zero-inflated poisson", "ordered logit",
                              "ordered probit", "multinomial logit", "tobit"], Widget("choice", show_if=("method", _kind("ml", "bayes")),
                      help="logit, probit for 0/1 outcomes; poisson, negative binomial, zero-inflated for counts; ordered, "
                           "multinomial for choices; tobit if censored")] = "gaussian"
    censor_left: Annotated[str, Widget("text", show_if=("family", ("tobit",)), help="the lower limit (empty: none)")] = "0"
    censor_right: Annotated[str, Widget("text", show_if=("family", ("tobit",)), help="the upper limit (empty: none)")] = ""
    inflate: Annotated[str, Widget("text", show_if=("family", ("zero-inflated poisson",)),
                                   help="regressors of the probability of a structural zero, e.g. age + kids (empty: a constant)")] = ""
    endog: Annotated[str, Widget("series", depends_on="data", multiple=True, show_if=("method", _kind("iv")), group="Instrumental variables")] = ""
    instruments: Annotated[str, Widget("text", show_if=("method", _kind("iv")), group="Instrumental variables")] = ""
    ar: Annotated[str, Widget("text", help="AR order, e.g. 2, or lags 1, 4", show_if=("method", _kind("ml")), group="Error model")] = ""
    ma: Annotated[str, Widget("text", help="MA order, e.g. 1, or lags 1, 4; overlapping observations (h-step changes) "
                                           "give MA(h-1) errors", show_if=("method", _kind("ml")), group="Error model")] = ""
    variance: Annotated[str, Widget("extension", source="variances", show_if=("method", _kind("ml")), group="Error model",
                                    help="conditional variance: constant, or a GARCH-type model")] = "constant"
    dist: Annotated[Literal["normal", "t", "skewt", "ged"], Widget("choice", show_if=("variance", _with_variance), group="Error model")] = "normal"
    variance_settings: Annotated[str, Widget("options", depends_on="variance", source="variances", exclude=("dist",),
                                             show_if=("variance", _with_variance), group="Variance settings")] = ""
    cov: Annotated[Literal["classic", "white", "hac", "cluster"], Widget("choice", group="Sample and covariance")] = "classic"
    cluster: Annotated[str, Widget("series", depends_on="data", multiple=True, show_if=("cov", ("cluster",)), group="Sample and covariance",
                                   help="one or two grouping series or axes")] = ""
    start: Annotated[str, Widget("period", group="Sample and covariance")] = ""
    end: Annotated[str, Widget("period", group="Sample and covariance")] = ""
    name: str = "fit"
    method_settings: Annotated[str, Widget("options", depends_on="method", source="methods", exclude=_MAIN_METHOD_FIELDS,
                                           group="Method settings")] = ""
    # advanced
    hac_lags: Annotated[int, Widget("number", min=0, max=100, advanced=True, group="Covariance", show_if=("cov", ("hac",)),
                                    help="Newey-West truncation lag; 0 = automatic, 4(n/100)^(2/9); more for persistent errors")] = 0
    hac_correction: Annotated[bool, Widget("checkbox", advanced=True, group="Covariance", show_if=("cov", ("hac",)),
                                           help="multiply HAC by n / (n - k), as EViews and Stata's newey; untick for "
                                                "statsmodels' default")] = True
    weights: Annotated[str, Widget("series", depends_on="data", advanced=True, group="Sample",
                                   help="weighted least squares: weights proportional to 1 / error variance; empty = none")] = ""
    missing: Annotated[Literal["drop", "raise"], Widget("choice", advanced=True, group="Sample")] = "drop"
    where: Annotated[str, Widget("text", advanced=True, group="Sample",
                                 help="a condition on the rows, e.g. recession == 0 or year >= 1990; empty = all rows")] = ""

    def validate(self, session: Session) -> list[str]:
        errors = [] if self.data in session.objects else [f"no dataset named {self.data!r}"]
        entry = registry("methods").find(self.method)
        if entry is None:
            return errors + [f"unknown method {self.method!r}"]
        if not entry.available:
            errors.append(f"{self.method}: {entry.problem}")
        try:
            variance = self._variance()
        except (KeyError, TypeError, ValueError) as exc:
            return errors + [_message(exc)]
        error_model = self._errors() is not None or variance is not None
        if entry.kind == "bayes":  # its own limits: four families, no error models, uncertainty from the posterior
            if self.family not in ("gaussian", "logit", "probit", "poisson"):
                errors.append(f"Bayes estimates gaussian, logit, probit and poisson equations, not {self.family}")
            if error_model:
                errors.append("Bayes does not take ARMA errors or GARCH variance: choose method ML")
            if self.cov != "classic":
                errors.append("Bayes gives the posterior's uncertainty: leave Standard errors conventional")
            try:
                self._method()
            except (ValueError, SpecificationError) as exc:
                errors.append(f"priors: {_message(exc)}")
            return errors
        ml = entry.kind == "ml"
        if entry.kind == "robust" and self.cov != "classic":
            errors.append(f"{self.method} gives Huber's H1 standard errors: leave Standard errors conventional")
        if entry.kind == "iv" and not (_names(self.endog) and _names(self.instruments)):
            errors.append(f"{self.method} needs endogenous regressors and instruments")
        if self.cov == "cluster" and not 1 <= len(_names(self.cluster)) <= 2:
            errors.append("choose one or two cluster variables")
        if self.family != "gaussian" and not ml:
            errors.append(f"family {self.family} needs a maximum-likelihood method")
        if error_model and not ml:
            errors.append("ARMA errors and GARCH variance are estimated by maximum likelihood: choose method ML")
        if ml and self.family == "gaussian" and not error_model:
            errors.append(f"{self.method} needs family logit, probit or poisson, ARMA errors or GARCH variance; use OLS otherwise")
        if self.dist != "normal" and variance is None:
            errors.append(f"{self.dist} errors are available with a volatility model")
        if variance is not None:
            v_entry = registry("variances").get(self.variance)
            missing = [lib for lib in getattr(variance, "libraries", v_entry.requires) if not installed(lib)]
            if missing:
                errors.append(f"{self.variance} with {self.dist} errors uses {', '.join(missing)}: {v_entry.install_hint}")
        return errors

    def _errors(self) -> ARMA | None:
        def lags(text: str) -> Any:
            values = [int(s) for s in _names(text)]
            return values[0] if len(values) == 1 else values

        if not (_names(self.ar) or _names(self.ma)):
            return None
        return ARMA(ar=lags(self.ar) if _names(self.ar) else 0, ma=lags(self.ma) if _names(self.ma) else 0)

    def _variance(self) -> Any:
        if self.variance == "constant":
            return None
        return build(registry("variances").get(self.variance), self.variance_settings, dist=self.dist)

    def _method(self) -> Any:
        """The chosen method's object: main-form settings plus its own settings sub-form."""
        from ..estimation import Cluster

        entry = registry("methods").get(self.method)
        _, kw = parse_options(self.method_settings)
        if isinstance(kw.get("start"), str):  # NLS starting values typed as "b0=0.5, b1=0.1"
            kw["start"] = {k: float(v) for k, v in parse_options(kw["start"])[1].items()} or None
        main: dict[str, Any] = {"endog": _names(self.endog), "instruments": _names(self.instruments),
                                "cov": Cluster(*_names(self.cluster)) if self.cov == "cluster" else self.cov}
        if self.hac_lags:
            main["hac_lags"] = self.hac_lags
        if not self.hac_correction:
            main["hac_correction"] = False
        names = {f.name for f in fields(entry.obj)}
        return entry.build(**{**{k: v for k, v in main.items() if k in names}, **kw})

    def _fit_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.weights:
            options["weights"] = self.weights
        if self.missing != "drop":
            options["missing"] = self.missing
        if self.where.strip():
            options["where"] = self.where.strip()
        return options

    def _family(self) -> Any:
        from ..estimation import Multinomial, NegativeBinomial, OrderedLogit, OrderedProbit, Tobit, ZeroInflatedPoisson

        if self.family in _FAMILIES:
            return _FAMILIES[self.family]()
        if self.family == "tobit":
            def limit(text: str) -> float | None:
                try:
                    return float(text) if text.strip() else None
                except ValueError:
                    raise SpecificationError(f"the Tobit limit {text!r} is not a number") from None

            return Tobit(left=limit(self.censor_left), right=limit(self.censor_right))
        return {"negative binomial": NegativeBinomial, "ordered logit": OrderedLogit, "ordered probit": OrderedProbit,
                "multinomial logit": Multinomial}.get(self.family, lambda: ZeroInflatedPoisson(self.inflate.strip()) if
                                                      self.family == "zero-inflated poisson" else None)()

    def _equation(self) -> tuple[Equation, str]:
        family = self._family()
        options = {"family": family, "errors": self._errors(), "variance": self._variance()}
        options = {k: v for k, v in options.items() if v is not None}
        eq = Equation(self.formula, params=_names(self.params), **options)
        code = f"Equation({self.formula!r}{_params_code(self.params)}" + "".join(f", {k}={_code(v)}" for k, v in options.items()) + ")"
        return eq, code

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:equation.fit", self._equation()[0], output=self.name, recorded_by=self.command_id,
                         data=Ref(self.data), span=_span(self.start, self.end), method=self._method(), **self._fit_options())

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Test/Correlogram...")
@dataclass(frozen=True)
class Correlogram(Command):
    """Autocorrelations, partial autocorrelations and Q-statistics of a series or of residuals.

    Choose a dataset and series, or a fitted equation to examine its residuals. Tick Squared to look for volatility
    clustering.

    Each Q-statistic (Ljung-Box) tests that all autocorrelations up to its lag are zero; a small p-value means
    serial correlation remains. For residuals of an equation with ARMA terms, the p-values take those terms off the
    degrees of freedom; after ARMA or GARCH the residuals are standardised. The band in the notes (1.96/sqrt(T))
    is a guide for single lags only: among 20 lags one may fall outside by chance. Autocorrelations that die out
    very slowly suggest a unit root (Test > Run test); a partial autocorrelation that cuts off after p lags suggests
    an AR(p). With Squared, significant Q-statistics point to ARCH effects.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation

        data = Dataset.example("sample_macro").derive(dly="dlog(y)")
        data.correlogram("dly", lags=12)           # AC, PAC, the Ljung-Box Q and its p-value at each lag
        fit = Equation("dlog(c) ~ dlog(y)").fit(data)
        fit.correlogram(lags=12)                   # of the residuals: small p-values mean serial correlation remains
        fit.correlogram(lags=12, squared=True)     # of the squared residuals, for ARCH effects
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("AC, PAC", ("the autocorrelation and partial autocorrelation at each lag; compare each with the band in the "
                     "notes, 1.96/sqrt(T)")),
        ("Q-stat, p-value", ("Ljung-Box statistic for all lags up to this one; a small p-value means autocorrelation "
                             "(serial correlation, or ARCH with Squared)")),
        ("notes", "the observations used and the approximate 95% band for AC and PAC"),
    )

    topics: ClassVar[tuple[str, ...]] = ('diagnostic-tests',)
    schema_version: ClassVar[int] = 2  # 2: target became source

    source: Annotated[str, Widget("object", source="Dataset,FittedEquation", help="a dataset, or a fitted equation for its residuals")]
    variable: Annotated[str, Widget("series", depends_on="source", help="for a dataset: the series to examine")] = ""
    lags: Annotated[int, Widget("number", min=1, max=500,
                                help="autocorrelations shown and tested jointly; about ln(T), or a few seasonal cycles for seasonal data")] = 20
    squared: bool = False
    name: str = "correlogram"

    def validate(self, session: Session) -> list[str]:
        obj = session.objects.get(self.source)
        if obj is None:
            return [f"no object named {self.source!r}"]
        if isinstance(obj, Dataset) and not self.variable:
            return ["choose the series to examine"]
        return []

    def to_call(self, session: Session) -> Any:
        from ..calls import make_call, method_call
        from ..codec import Ref

        obj = session[self.source]
        record = {"output": self.name, "recorded_by": self.command_id}
        if not isinstance(obj, Dataset):
            return method_call(type(obj), "correlogram", Ref(self.source), lags=self.lags, squared=self.squared, **record)
        if self.squared:
            squares = make_call("pymetrix:dataset.with_series", Ref(self.source), sq=f"{self.variable}^2")
            return method_call(Dataset, "correlogram", squares, name="sq", lags=self.lags, **record)
        return method_call(Dataset, "correlogram", Ref(self.source), name=self.variable, lags=self.lags, **record)

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        if self.variable:
            if self.squared:
                return f"{self.name} = {self.source}.with_series(sq={self.variable + '^2'!r}).correlogram('sq', lags={self.lags})"
            return f"{self.name} = {self.source}.correlogram({self.variable!r}, lags={self.lags})"
        return f"{self.name} = {self.source}.correlogram(lags={self.lags}, squared={self.squared})"


@migration("pymetrix:correlogram", 1)
def _correlogram_source(fields: dict[str, Any]) -> dict[str, Any]:  # schema 2: target became source
    if "target" in fields:
        fields["source"] = fields.pop("target")
    return fields


def _shares(text: str) -> tuple[float, ...]:
    """Percentiles typed as "5, 50, 95" (or as shares, "0.05, 0.5, 0.95") as shares."""
    values = [float(v) for v in _names(text)]
    return tuple(v / 100 if any(x > 1 for x in values) else v for v in values)


@command(menu="Data/Descriptive statistics...")
@dataclass(frozen=True)
class DescriptiveStatistics(Command):
    """Count, mean, standard deviation, minimum, percentiles, maximum, skewness and kurtosis of series.

    Tick the series (none = every numeric series). Percentiles are typed as percentages ("5, 50, 95"); the median is
    always shown. Statistics use the active sample, each series over the periods it has. Skewness and kurtosis are
    the bias-corrected sample statistics (as pandas, Excel's SKEW and KURT, and scipy with bias=False compute them);
    kurtosis is excess kurtosis, 0 for a normal distribution, so values well above 0 mean fat tails.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset

        data = Dataset.example("sample_macro").derive(dly="dlog(y)", dlc="dlog(c)")
        data.describe(["dly", "dlc"], percentiles=(0.05, 0.5, 0.95))
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("count, mean, std, min, max", "observations, mean, standard deviation (n - 1), smallest and largest values"),
        ("percentiles", "the values below which that share of the observations lie (25%, 50% and 75% by default)"),
        ("skewness, kurtosis", "bias-corrected skewness and excess kurtosis (0 for a normal distribution)"),
    )
    topics: ClassVar[tuple[str, ...]] = ("distributions-and-moments",)

    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    series: Annotated[str, Widget("series", depends_on="data", multiple=True, help="series to describe; empty = all")] = ""
    percentiles: Annotated[str, Widget("text", help="percentages, e.g. 5, 50, 95; the median is always shown")] = "25, 50, 75"
    name: str = "stats"

    def validate(self, session: Session) -> list[str]:
        if not isinstance(session.objects.get(self.data), Dataset):
            return [f"no dataset named {self.data!r}"]
        try:
            shares = _shares(self.percentiles)
        except ValueError:
            return ["percentiles are numbers separated by commas, e.g. 5, 50, 95"]
        return [] if all(0 <= q <= 1 for q in shares) else ["percentiles are between 0 and 100"]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        extra = {"percentiles": _shares(self.percentiles)} if _shares(self.percentiles) != (0.25, 0.5, 0.75) else {}
        return make_call("pymetrix:dataset.describe", Ref(self.data), output=self.name, recorded_by=self.command_id,
                         names=tuple(_names(self.series)) or None, **extra)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Data/Correlations...")
@dataclass(frozen=True)
class Correlations(Command):
    """Correlations (Pearson, Spearman or Kendall), covariances, partial correlations, co-skewness, co-kurtosis or
    systematic co-moments with a market series.

    Tick the series (none = every numeric series for correlations, covariances, co-skewness and co-kurtosis; partial
    correlations need two or more). Pearson correlations measure linear association; Spearman's are the correlations of ranks and Kendall's
    tau-b counts concordant pairs, so both are unchanged by monotonic transformations and less moved by outliers.
    Correlations and covariances use, for each pair, the periods both series have. Partial correlations hold the
    series in Given constant: they are the correlations of the residuals from regressions of each series on a
    constant and those series, over the periods every series has. Co-skewness and co-kurtosis are the means of
    products of three or four standardized deviations (moments divided by T), over the periods every series has: as
    the full matrix (M3, one row a series and one column a pair; M4, a column a triple) or one number for each pair
    (mean(e_i² e_j), mean(e_i³ e_j)); the diagonal is each series' skewness or (not excess) kurtosis with the T
    denominator. Systematic co-moments relate each series to the Market series: beta, co-skewness, co-kurtosis and
    the co-skewness and co-kurtosis betas. All use the active sample.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset

        data = Dataset.example("sample_macro").derive(dly="dlog(y)", dlc="dlog(c)", dli="dlog(i)")
        data.corr(["dly", "dlc", "dli"], method="spearman")
        data.cov(["dly", "dlc"])
        data.partial_corr(["dlc", "dli"], given=["dly"])
        data.coskewness(["dly", "dlc"], form="pairs")
        data.systematic_comoments(["dlc", "dli"], market="dly")
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("table", "a symmetric matrix: correlations (1 on the diagonal), covariances (variances on the diagonal) or "
                  "partial correlations; co-skewness or co-kurtosis (the matrix M3 or M4, or a table of pairs); or a row "
                  "of systematic co-moments for each series"),
        ("notes", "for partial correlations, what is held constant and the periods used; for co-moments, the definition "
                  "and the periods used"),
    )
    topics: ClassVar[tuple[str, ...]] = ("distributions-and-moments",)

    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    series: Annotated[str, Widget("series", depends_on="data", multiple=True, help="series to relate; empty = all")] = ""
    show: Annotated[Literal["correlation", "covariance", "partial", "coskewness", "cokurtosis", "systematic"],
                    Widget("choice", help="correlations, covariances, partial correlations holding series constant, co-skewness, "
                                          "co-kurtosis, or systematic co-moments with a market series")] = "correlation"
    method: Annotated[Literal["pearson", "spearman", "kendall"],
                      Widget("choice", show_if=("show", ("correlation",)), help="pearson: linear; spearman, kendall: of ranks")] = "pearson"
    given: Annotated[str, Widget("series", depends_on="data", multiple=True, show_if=("show", ("partial",)),
                                 help="series held constant")] = ""
    form: Annotated[Literal["matrix", "pairs"], Widget("choice", show_if=("show", ("coskewness", "cokurtosis")),
                                                       help="matrix: every co-moment (M3, M4); pairs: one number for each pair")] = "matrix"
    standardized: Annotated[bool, Widget("checkbox", show_if=("show", ("coskewness", "cokurtosis")),
                                         help="divide the deviations by the standard deviations (unticked: co-moments in the series' units)")] = True
    market: Annotated[str, Widget("series", depends_on="data", show_if=("show", ("systematic",)),
                                  help="the market series the others are related to")] = ""
    name: str = "correlations"

    def validate(self, session: Session) -> list[str]:
        if not isinstance(session.objects.get(self.data), Dataset):
            return [f"no dataset named {self.data!r}"]
        if self.show == "partial" and len(_names(self.series)) < 2:
            return ["choose two or more series for partial correlations"]
        if self.show == "partial" and set(_names(self.series)) & set(_names(self.given)):
            return ["a series cannot be both correlated and held constant"]
        if self.show == "systematic" and (not self.market.strip() or not _names(self.series)):
            return ["choose the series and the market series to relate them to"]
        if self.show == "systematic" and self.market.strip() in _names(self.series):
            return ["the market series cannot also be one of the series related to it"]
        return []

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        record = {"output": self.name, "recorded_by": self.command_id}
        names = tuple(_names(self.series)) or None
        if self.show == "covariance":
            return make_call("pymetrix:dataset.cov", Ref(self.data), names=names, **record)
        if self.show == "partial":
            return make_call("pymetrix:dataset.partial_corr", Ref(self.data), names=names, given=tuple(_names(self.given)), **record)
        if self.show in ("coskewness", "cokurtosis"):
            options = {**({"standardized": False} if not self.standardized else {}), **({"form": self.form} if self.form != "matrix" else {})}
            return make_call(f"pymetrix:dataset.{self.show}", Ref(self.data), names=names, **options, **record)
        if self.show == "systematic":
            return make_call("pymetrix:dataset.systematic_comoments", Ref(self.data), names=names, market=self.market.strip(), **record)
        return make_call("pymetrix:dataset.corr", Ref(self.data), names=names, method=self.method, **record)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Test/Cross-correlogram...")
@dataclass(frozen=True)
class CrossCorrelogram(Command):
    """Cross-correlations of two series at leads and lags, or the autocovariances of one.

    The correlation of x at t with y at t+k, for k from -lags to lags: a peak at k > 0 means x leads y by k periods,
    at k < 0 that it lags. Each uses the full-sample means and standard deviations, with the sums divided by T (as
    the autocorrelations of a correlogram are). The band in the notes, 1.96/sqrt(T), is a guide for a single lag
    of two unrelated series: series that are each autocorrelated (or trending) show spurious cross-correlations, so
    difference or prewhiten them first. Leave y empty for the autocovariances of x at lags 0 to lags. The dataset
    must be a single time series.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset

        data = Dataset.example("sample_macro").derive(dly="dlog(y)", dlc="dlog(c)")
        data.crosscorrelogram("dly", "dlc", lags=8)   # corr(dly(t), dlc(t+k)), k = -8 ... 8
        data.autocov("dly", lags=8)                   # autocovariances at lags 0 ... 8
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("lag, correlation", "the correlation of x at t with y at t+k; compare it with the band in the notes"),
        ("autocovariance", "with y empty: the autocovariances of x, and the autocorrelations they give"),
    )
    topics: ClassVar[tuple[str, ...]] = ("distributions-and-moments",)

    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    x: Annotated[str, Widget("series", depends_on="data", help="the first series (at t)")] = ""
    y: Annotated[str, Widget("series", depends_on="data", help="the second series (at t+k); empty: autocovariances of x")] = ""
    lags: Annotated[int, Widget("number", min=1, max=500, help="leads and lags shown")] = 12
    name: str = "crosscorrelogram"

    def validate(self, session: Session) -> list[str]:
        if not isinstance(session.objects.get(self.data), Dataset):
            return [f"no dataset named {self.data!r}"]
        return [] if self.x else ["choose the series x"]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        record = {"output": self.name, "recorded_by": self.command_id}
        if not self.y:
            return make_call("pymetrix:dataset.autocov", Ref(self.data), name=self.x, lags=self.lags, **record)
        return make_call("pymetrix:dataset.crosscorrelogram", Ref(self.data), x=self.x, y=self.y, lags=self.lags, **record)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Data/Filter...")
@dataclass(frozen=True)
class Filter(Command):
    """Split series into trend and cycle with a filter (HP, band-pass, Hamilton...).

    Tick the series and choose the filter; its settings appear below it, defaulting to values suited to the data's
    frequency. `<series>_trend` and `<series>_cycle` are added to the dataset (or to a new one, if you name it).

    With Keep updated (the default) they are derived series: when the filtered series changes, they are
    recomputed. A two-sided filter then revises its whole history, as it would with a new data release.

    The cycle is defined by the filter, not found in the data: HP removes a smooth trend whose smoothness the
    smoothing parameter sets; Baxter-King and Christiano-Fitzgerald keep cycles of 1.5 to 8 years by default; Hamilton
    takes the error in forecasting the series two years ahead. Two-sided filters use future values, so the last few
    years of the cycle are revised as data arrive and are unreliable in real time; Baxter-King loses points at
    both ends instead. Compare two filters before drawing conclusions about the latest output gap.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import HP, Dataset, Hamilton

        data = Dataset.example("sample_macro").derive(ly="100*log(y)")
        data = data.filter("ly", HP())        # adds ly_trend and ly_cycle, recomputed when ly changes
        data["ly_cycle"]                      # 100 times the log deviation from trend: per cent
        data.filter("ly", Hamilton())         # Hamilton's cycle: the error in forecasting ly two years ahead
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("{series}_trend", "the trend the filter defines, added to the dataset (or to the new one named in Name)"),
        ("{series}_cycle", ("the cyclical part (series = trend + cycle); for 100*log of a series, the per cent "
                            "deviation from trend")),
        ("Variables tab", ("derived_from names the filter and its setting, e.g. HP(lambda=1600) trend of ly; 'kept up "
                           "to date' is yes with Keep updated")),
        ("Values at the ends", ("see the chosen filter's page: Baxter-King loses values at both ends; two-sided filters "
                                "revise their last values")),
    )

    topics: ClassVar[tuple[str, ...]] = ('trend-cycle-filters', 'derived-series')

    variables: Annotated[str, Widget("series", depends_on="data", multiple=True, help="series to filter, e.g. log GDP")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    method: Annotated[str, Widget("extension", source="filters",
                                  help="HP, one-sided HP, band-pass (Baxter-King, Christiano-Fitzgerald), Hamilton or moving "
                                       "average; each defines the cycle differently")] = "HP"
    settings: Annotated[str, Widget("options", depends_on="method", source="filters", group="Filter settings")] = ""
    keep_updated: Annotated[bool, Widget("checkbox", help="recompute the trend and cycle whenever the series changes")] = True
    name: Annotated[str, Widget("text", help="a new dataset for the result; empty = add to the dataset itself")] = ""

    def validate(self, session: Session) -> list[str]:
        try:
            self._method()
        except (KeyError, TypeError, ValueError) as exc:
            return [_message(exc)]
        return [] if self.data in session.objects else [f"no dataset named {self.data!r}"]

    def _method(self) -> Any:
        return build(registry("filters").get(self.method), self.settings)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        names = _names(self.variables)
        return make_call("pymetrix:dataset.filter", Ref(self.data), output=self.name or self.data, recorded_by=self.command_id,
                         names=names[0] if len(names) == 1 else names, method=self._method(), derived=self.keep_updated)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Data/Derived series...")
@dataclass(frozen=True)
class DeriveSeries(Command):
    """Add a series defined by a formula that stays up to date, such as `dlog(y)` or `c / y * 100`.

    Unlike a plain series, it keeps its formula: when a series it uses changes, it is recomputed, and so is
    anything derived from it in turn. The formula uses the same expression language as equations (`x[-1]` for
    lags, `log`, `dlog`, `d` and other functions). Its values are fixed only if you freeze it in code
    (`data.freeze("name")`) or take a subset of the dataset.
    """

    topics: ClassVar[tuple[str, ...]] = ("derived-series",)

    series: Annotated[str, Widget("text", help="name of the new series, e.g. dly")]
    formula: Annotated[str, Widget("formula", help="e.g. dlog(y), c / y * 100, (y - y[-4]) / y[-4]")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    replace: Annotated[bool, Widget("checkbox", help="replace a series of the same name, if there is one")] = False

    schema_version: ClassVar[int] = 2  # 2: replace (schema-1 steps replaced silently)

    def validate(self, session: Session) -> list[str]:
        if not self.series.isidentifier():
            return [f"{self.series!r} is not a valid series name"]
        if self.data not in session.objects:
            return [f"no dataset named {self.data!r}"]
        return existing_series(session, self.data, [self.series], self.replace)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:dataset.derive", Ref(self.data), output=self.data, recorded_by=self.command_id, **{self.series: self.formula})

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/ARMA order selection...")
@dataclass(frozen=True)
class SelectOrder(Command):
    """Compare ARMA orders for a series by AIC, BIC and Hannan-Quinn.

    Every combination up to the largest AR and MA orders is estimated on the same sample; the table marks the best
    by each criterion.

    Each candidate is fitted by exact Gaussian maximum likelihood on the same observations, so the criteria are
    comparable. BIC penalises parameters most and picks the smallest models, AIC the largest; when they disagree,
    fit both and compare their residual correlograms. Differences of one or two units are small, and the more
    parsimonious model usually forecasts as well. Orders that fail to converge are left blank, often a sign of
    near-cancelling AR and MA roots. Make a trending series stationary first (difference it, or add a trend series
    under Exogenous series).
    """

    topics: ClassVar[tuple[str, ...]] = ('maximum-likelihood',)

    variable: Annotated[str, Widget("series", depends_on="data")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    max_ar: Annotated[int, Widget("number", min=0, max=12, help="largest AR order p tried; widen it if the best order is at the edge")] = 4
    max_ma: Annotated[int, Widget("number", min=0, max=12,
                                  help="largest MA order q tried; every p is crossed with every q, so the grid grows fast")] = 2
    exog: Annotated[str, Widget("series", depends_on="data", multiple=True,
                                help="regressors in every candidate: the orders then describe the regression's errors")] = ""
    name: str = "orders"
    intercept: Annotated[bool, Widget("checkbox", advanced=True, help="include a constant in every candidate")] = True

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:dataset.select_order", Ref(self.data), output=self.name, recorded_by=self.command_id,
                         name=self.variable, max_ar=self.max_ar, max_ma=self.max_ma, exog=tuple(_names(self.exog)),
                         intercept=self.intercept)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/Rolling or recursive...")
@dataclass(frozen=True)
class FitRolling(Command):
    """Estimate an equation over moving or expanding windows, with recursive residuals and CUSUM tests.

    Window 0 gives recursive (expanding-window) estimation, starting from Min obs periods; a positive window gives
    rolling estimates. Results include coefficient paths and pseudo out-of-sample forecasts.

    Read the coefficient paths against their two-standard-error bands: a coefficient that drifts well outside its
    earlier bands points to a structural break or an omitted variable, which the CUSUM tests (Test > Run test, on the
    full-sample fit) check formally. A rolling window tracks change faster but each estimate is noisier; a short
    window also widens the bands, so apparent movement may be sampling noise. The pseudo out-of-sample forecasts use
    only data up to each window's end, a fairer test of forecasting ability than the in-sample fit.
    """

    topics: ClassVar[tuple[str, ...]] = ('least-squares', 'diagnostic-tests')

    formula: Annotated[str, Widget("formula", help="the equation re-estimated in every window, e.g. c ~ y + c[-1]")]
    params: Annotated[str, Widget("text", help="comma-separated parameter names; empty for the linear form")] = ""
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    window: Annotated[int, Widget("number", help="moving-window length in periods; 0 = expanding (recursive)", min=0, max=10000)] = 0
    min_obs: Annotated[int, Widget("number", advanced=True, help="recursive: periods in the first window; 0 = parameters + 1", min=0, max=10000)] = 0
    cov: Annotated[Literal["classic", "white", "hac"], Widget("", help="classic, white (robust) or hac (Newey-West): the bands "
                                                                       "around the coefficient paths")] = "classic"
    start: Annotated[str, Widget("period")] = ""
    end: Annotated[str, Widget("period")] = ""
    name: str = "rolling"
    step: Annotated[int, Widget("number", min=1, max=1000, advanced=True, help="estimate every step-th window")] = 1
    hac_lags: Annotated[int, Widget("number", min=0, max=100, advanced=True, show_if=("cov", ("hac",)),
                                    help="Newey-West truncation lag; 0 = automatic, 4(n/100)^(2/9)")] = 0

    def validate(self, session: Session) -> list[str]:
        return [] if self.data in session.objects else [f"no dataset named {self.data!r}"]

    def _method(self) -> OLS:
        return OLS(cov=self.cov, hac_lags=self.hac_lags or None)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        eq = Equation(self.formula, params=_names(self.params))
        options = {"span": _span(self.start, self.end), "method": self._method(), "step": self.step}
        if self.window:
            return make_call("pymetrix:equation.fit_rolling", eq, output=self.name, recorded_by=self.command_id,
                             data=Ref(self.data), window=self.window, **options)
        return make_call("pymetrix:equation.fit_recursive", eq, output=self.name, recorded_by=self.command_id,
                         data=Ref(self.data), min_obs=self.min_obs or None, **options)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


def _var_spec(variables: str, lags: str, max_lags: int, exog: str, trend: str, by: str = "") -> tuple[VAR, str]:
    lag: Any = int(lags) if lags.strip().isdigit() else lags.strip().lower()
    spec = VAR(_names(variables), lags=lag, max_lags=max_lags, exog=_names(exog), trend=trend, by=by.strip() or None)  # type: ignore[arg-type]
    return spec, _code(spec).replace("VAR(variables=", "VAR(", 1)


@command(menu="Estimate/VAR...")
@dataclass(frozen=True)
class FitVAR(Command):
    """Estimate a vector autoregression, by least squares or with a Bayesian (Minnesota) prior.

    Tick the series: they enter in the list's order, which is the order used for Cholesky identification (Impulse
    responses can reorder them). Lags is a number, or a criterion (aic, bic, hq,
    fpe) to choose it up to Max lags. Tick Bayesian to shrink the coefficients towards random walks (own mean 1; 0 for
    growth rates): smaller tightness shrinks more, and the sum-of-coefficients and initial-observation priors pull
    towards unit roots. Choose hyperparameters lets the marginal likelihood pick them. Independent prior samples the
    independent normal-Wishart form by Gibbs, which can shrink other variables' lags harder (cross tightness).
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import VAR, Bayes, Dataset, Minnesota

        data = Dataset.example("sample_macro").derive(dy="400*dlog(y)", dc="400*dlog(c)", di="400*dlog(i)")
        var = VAR(["dy", "dc", "di"], lags=2).fit(data)    # least squares; the list's order is the Cholesky ordering
        print(var.summary())       # a column per equation (coefficients and standard errors), criteria, stability
        # a Bayesian VAR with a Minnesota prior (own_mean=0 suits growth rates): posterior means and sds
        bvar = VAR(["dy", "dc", "di"], lags=4).fit(data, method=Bayes(prior=Minnesota(own_mean=0), draws=500))
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", ("a column per equation with standard errors '(se)', or posterior standard deviations '(sd)' for a "
                     "Bayesian VAR")),
        ("AIC, BIC, HQ, FPE", ("per-observation criteria (Lütkepohl) for this lag length only; compare lag lengths with "
                               "VAR lag selection")),
        ("Roots, Roots graph", ("the companion-matrix roots: all moduli below 1 (inside the unit circle) means a stable "
                                "VAR whose responses die out")),
        ("Residual correlogram", ("auto- and cross-correlations of the residuals by lag; the notes count those outside "
                                  "the ±1.96/sqrt(T) band")),
        ("Test → Run test", ("on the VAR: portmanteau (no residual autocorrelation), normality and granger; small "
                             "p-values reject")),
        ("Bayesian notes", ("the prior, the log marginal likelihood (to compare priors) and the share of explosive "
                            "posterior draws")),
    )

    topics: ClassVar[tuple[str, ...]] = ('var', 'bayesian-var')

    variables: Annotated[str, Widget("series", depends_on="data", multiple=True,
                                     help="the VAR's series, in the list's order: the default Cholesky ordering")]
    lags: Annotated[str, Widget("text", help="a number, or aic / bic / hq / fpe to choose")] = "2"
    max_lags: Annotated[int, Widget("number", advanced=True, help="searched when lags is a criterion", min=1, max=24)] = 8
    exog: Annotated[str, Widget("text", help="exogenous expressions (optional)", advanced=True)] = ""
    trend: Literal["c", "ct", "n"] = "c"
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    start: Annotated[str, Widget("period")] = ""
    end: Annotated[str, Widget("period")] = ""
    name: str = "var"
    bayesian: Annotated[bool, Widget("checkbox", help="a Bayesian VAR with a Minnesota prior", group="Bayesian")] = False
    tightness: Annotated[float, Widget("number", min=0.0001, max=10, step=0.05, help="overall tightness; smaller shrinks more",
                                       show_if=("bayesian", (True,)), group="Bayesian")] = 0.2
    own_mean: Annotated[float, Widget("number", min=-2, max=2, step=0.1, help="prior mean of own first lag: 1 for levels, 0 for growth rates",
                                      show_if=("bayesian", (True,)), group="Bayesian")] = 1.0
    sum_of_coefficients: Annotated[str, Widget("text", advanced=True, help="sum-of-coefficients prior, e.g. 1; empty = none",
                                               show_if=("bayesian", (True,)), group="Bayesian")] = ""
    initial_observation: Annotated[str, Widget("text", advanced=True, help="single-unit-root prior, e.g. 1; empty = none",
                                               show_if=("bayesian", (True,)), group="Bayesian")] = ""
    optimize: Annotated[bool, Widget("checkbox", advanced=True, help="choose the hyperparameters by marginal likelihood",
                                     show_if=("bayesian", (True,)), group="Bayesian")] = False
    draws: Annotated[int, Widget("number", min=100, max=100_000, step=500, advanced=True, help="posterior draws (Bayesian)")] = 5000
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="random seed for posterior draws (Bayesian)")] = 0
    independent: Annotated[bool, Widget("checkbox", advanced=True, help="the independent normal-Wishart form, sampled by Gibbs (hyperparameter search "
                                                         "by Chib's marginal likelihood)",
                                        show_if=("bayesian", (True,)), group="Bayesian")] = False
    cross_tightness: Annotated[float, Widget("number", advanced=True, min=0.0001, max=1, step=0.05, help="other variables' lags relative to own (independent prior)",
                                             show_if=("independent", (True,)), group="Bayesian")] = 0.5
    zeros: Annotated[str, Widget("text", advanced=True, group="Restrictions",
                                 help="regressors left out of equations, e.g. pi: r; b: a[-2], const")] = ""
    block_exogenous: Annotated[str, Widget("text", advanced=True, group="Restrictions",
                                           help="variables whose equations leave out the others' lags, e.g. oil")] = ""
    iterate: Annotated[bool, Widget("checkbox", advanced=True, group="Restrictions",
                                    help="iterate the feasible GLS to maximum likelihood")] = False
    hierarchical: Annotated[bool, Widget("checkbox", advanced=True, show_if=("bayesian", (True,)), group="Bayesian",
                                         help="Giannone-Lenza-Primiceri: hyperpriors on the tightness, the dummy-observation priors "
                                              "and the scales, estimated (the tightness and dummy fields above are then not used)")] = False
    hyper_mcmc: Annotated[bool, Widget("checkbox", advanced=True, show_if=("hierarchical", (True,)), group="Bayesian",
                                       help="integrate over the hyperparameters by MCMC, rather than fix them at their posterior mode")] = False

    def _restricted(self) -> Any:
        if not (self.zeros.strip() or self.block_exogenous.strip()):
            return None
        from ..var import Restricted

        zeros: dict[str, list[str]] = {}
        for part in (p.strip() for p in self.zeros.split(";") if p.strip()):
            eq, colon, rest = part.partition(":")
            if not colon or not eq.strip():
                raise SpecificationError(f"{part!r}: write each equation as name: regressor, regressor")
            zeros[eq.strip()] = [r.strip() for r in rest.split(",") if r.strip()]
        return Restricted(zeros=zeros, block_exogenous=tuple(_names(self.block_exogenous)), iterate=self.iterate)

    def _method(self) -> Any:
        restricted = self._restricted()
        if restricted is not None:
            if self.bayesian:
                raise SpecificationError("restrictions are for least squares; untick Bayesian or clear the restrictions")
            return restricted
        if not self.bayesian:
            return None
        from ..bvar import Minnesota, NormalWishart
        from ..estimation import Bayes

        soc = float(self.sum_of_coefficients) if self.sum_of_coefficients.strip() else None
        dio = float(self.initial_observation) if self.initial_observation.strip() else None
        if self.hierarchical:
            if self.independent:
                raise SpecificationError("the hierarchical (GLP) prior is the conjugate one; untick independent or hierarchical")
            from ..bvar_hierarchical import GLP

            return Bayes(prior=GLP(mcmc=self.hyper_mcmc, own_mean=self.own_mean), draws=self.draws, seed=self.seed)
        if self.independent:
            prior: Any = NormalWishart(tightness=self.tightness, cross_tightness=self.cross_tightness, own_mean=self.own_mean,
                                       sum_of_coefficients=soc, initial_observation=dio, optimize=self.optimize)
        else:
            prior = Minnesota(tightness=self.tightness, own_mean=self.own_mean, sum_of_coefficients=soc,
                              initial_observation=dio, optimize=self.optimize)
        return Bayes(prior=prior, draws=self.draws, seed=self.seed)

    def validate(self, session: Session) -> list[str]:
        try:
            restricted = self._restricted()
        except ValueError as exc:
            return [f"restrictions: {exc}"]
        if restricted is not None and self.bayesian:
            return ["restrictions are for least squares; untick Bayesian or clear the restrictions"]
        try:
            self._method()
        except ValueError as exc:
            return [f"Bayesian prior: {exc}"]
        return []

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        spec, _ = _var_spec(self.variables, self.lags, self.max_lags, self.exog, self.trend)
        return make_call("pymetrix:var.fit", spec, output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                         span=_span(self.start, self.end), method=self._method())

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/Panel VAR...")
@dataclass(frozen=True)
class FitPanelVAR(Command):
    """Estimate a panel VAR: one VAR for many entities, with their fixed effects removed.

    By default each variable is transformed by forward orthogonal deviations and its lags are instrumented by their
    levels (GMM; more instrument lags over-identify, with Hansen's J); Within is least squares on demeaned data, biased
    when each entity has few periods. Standard errors are clustered by entity. The ticked series enter in the list's
    order, which is the order used for Cholesky identification of the impulse responses.

    The panel VAR assumes common slopes: every entity shares the same dynamics, and only the fixed effects differ.
    With many entities and few periods, prefer GMM: the within estimator's bias (Nickell) is of order 1/T. Hansen's J
    tests the over-identifying instruments; a small p-value questions their validity, and a very large one with many
    instruments may mean too many instruments rather than a good model.
    """

    topics: ClassVar[tuple[str, ...]] = ('var', 'panel-var')

    variables: Annotated[str, Widget("series", depends_on="data", multiple=True,
                                     help="the VAR's series, in the list's order: the default Cholesky ordering")]
    by: Annotated[str, Widget("text", help="the dataset's entity axis (firm, country...)")]
    lags: Annotated[int, Widget("number", min=1, max=12,
                                help="lags of every variable in every equation; each costs periods per entity, so keep it short")] = 1
    transform: Annotated[Literal["fod", "within"], Widget("choice", help="forward orthogonal deviations (GMM) or within (least squares)")] = "fod"
    instrument_lags: Annotated[int, Widget("number", min=0, max=24, advanced=True,
                                           help="lags of the levels as instruments (0: as many as the lags)")] = 0
    data: Annotated[str, Widget("object", source="Dataset", help="a panel dataset, with an entity axis and a time axis")] = "data"
    start: Annotated[str, Widget("period")] = ""
    end: Annotated[str, Widget("period")] = ""
    name: str = "pvar"
    choose_lags: Annotated[Literal["", "bic", "aic", "hq"], Widget("choice", advanced=True,
                                                                  help="choose the lags by Andrews and Lu's MBIC, MAIC or MQIC instead")] = ""
    max_lags: Annotated[int, Widget("number", min=1, max=12, advanced=True, help="most lags the criterion considers")] = 4

    def validate(self, session: Session) -> list[str]:
        data = session.objects.get(self.data)
        axes = [a.name for a in getattr(data, "axes", ())]
        if data is not None and self.by.strip() not in axes:
            return [f"{self.data} has no axis {self.by!r}: its axes are {', '.join(axes)}"]
        return []

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref
        from ..var import VAR

        names = [v.strip() for v in self.variables.split(",") if v.strip()]
        spec = (VAR(names, lags=self.choose_lags, max_lags=self.max_lags, by=self.by.strip()) if self.choose_lags
                else VAR(names, lags=int(self.lags), by=self.by.strip()))
        extra: dict[str, Any] = {"instrument_lags": self.instrument_lags} if self.instrument_lags else {}
        return make_call("pymetrix:var.fit", spec, output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                         span=_span(self.start, self.end), transform=self.transform, **extra)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/Proxy SVAR...", id="pymetrix:proxy_svar")
@dataclass(frozen=True)
class ProxySVARCommand(Command):
    """Identify one shock of a fitted VAR by an external instrument (a proxy SVAR).

    The instrument (a high-frequency surprise, a narrative series) must be correlated with the shock of the variable it
    normalises on and with no other. The window shows the shock's impact, its responses with moving-block bootstrap
    bands, and the first-stage F (below 10: a weak instrument).
    """

    topics: ClassVar[tuple[str, ...]] = ('var', 'proxy-svar')

    instrument: Annotated[str, Widget("series", depends_on="data", help="the external instrument, e.g. a high-frequency surprise")]
    var: Annotated[str, Widget("object", source="FittedVAR",
                               help="a VAR fitted without exogenous series; the instrument needs 20+ periods in its sample")] = "var"
    data: Annotated[str, Widget("object", source="Dataset", help="the dataset holding the instrument")] = "data"
    normalise_on: Annotated[str, Widget("text", help="the variable whose shock it instruments, moved by one by a unit shock "
                                                     "(empty: the VAR's first)")] = ""
    periods: Annotated[int, Widget("number", min=1, max=120, help="horizons of the responses")] = 20
    scale: Annotated[Literal["sd", "unit"], Widget("choice", help="one standard deviation, or a shock that moves the Normalise on variable by one")] = "sd"
    draws: Annotated[int, Widget("number", min=0, max=10_000, step=100, advanced=True, help="bootstrap draws for the bands (0: none)")] = 500
    name: str = "proxy"
    block: Annotated[int | None, Widget("number", min=1, max=100, advanced=True,
                                        help="moving-block bootstrap: block length; empty = automatic")] = None
    alpha: Annotated[float, Widget("number", min=0.01, max=0.5, step=0.01, advanced=True, help="1 - the bands' coverage, e.g. 0.05 for 95%")] = 0.05
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="random seed for the bootstrap")] = 0

    schema_version: ClassVar[int] = 2  # 2: target became normalise_on

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        extra: dict[str, Any] = {"normalise_on": self.normalise_on} if self.normalise_on.strip() else {}
        extra.update({k: getattr(self, k) for k, d in (("block", None), ("alpha", 0.05), ("seed", 0)) if getattr(self, k) != d})
        return make_call("pymetrix:fitted_var.proxy_svar", Ref(self.var), output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                         instrument=self.instrument, periods=self.periods, scale=self.scale, draws=self.draws, **extra)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@migration("pymetrix:proxy_svar", 1)
def _proxy_svar_normalise_on(fields: dict[str, Any]) -> dict[str, Any]:  # schema 2: target became normalise_on
    if "target" in fields:
        fields["normalise_on"] = fields.pop("target")
    return fields


def _matrix_text(text: str, what: str) -> list[list[Any]] | None:
    """'1 0 0; * 1 0; * * 1' -> [[1.0, 0.0, 0.0], ['*', 1.0, 0.0], ['*', '*', 1.0]]; empty -> None."""
    if not text.strip():
        return None
    rows = []
    for part in (p for p in text.replace("\n", ";").split(";") if p.strip()):
        row: list[Any] = []
        for item in part.replace(",", " ").split():
            if item == "*":
                row.append("*")
            else:
                try:
                    row.append(float(item))
                except ValueError:
                    raise SpecificationError(f"{what}: {item!r} is neither a number nor * (free)") from None
        rows.append(row)
    return rows


@command(menu="Estimate/Structural VAR (A-B)...", id="pymetrix:identify_ab")
@dataclass(frozen=True)
class IdentifyStructuralVAR(Command):
    """Identify a fitted VAR by restrictions on A and B (A u = B e), estimated by maximum likelihood.

    Write each matrix by rows, separated by semicolons, with * for a free entry: "1 0 0; * 1 0; * * 1". A empty is
    the identity; B empty is entirely free. Long run restricts the shocks' long-run effects (with A the identity),
    so short- and long-run restrictions combine. The window shows A and B with standard errors and the test of any
    over-identifying restrictions; impulse responses and decompositions of the result use this identification.

    A and B have 2K^2 entries but the residual covariance of K variables gives only K(K+1)/2 moments, so at least
    2K^2 - K(K+1)/2 entries must be fixed (long-run zeros count too). With exactly that many (and the rank condition met) the
    model is just identified and fits the covariance exactly; each further restriction is tested, and a small p-value rejects them.
    Several starts guard against a local maximum: if the estimates change with the seed, add starts.
    """

    topics: ClassVar[tuple[str, ...]] = ("var", "structural-var")

    var: Annotated[str, Widget("object", source="FittedVAR", help="the fitted VAR; A and B must reproduce its residual covariance")] = "var"
    A: Annotated[str, Widget("text", help="rows separated by ;, * free: 1 0 0; * 1 0; * * 1 (empty: the identity)")] = ""
    B: Annotated[str, Widget("text", help="rows separated by ;, * free: * 0 0; 0 * 0; 0 0 * (empty: all free)")] = ""
    long_run: Annotated[str, Widget("text", help="zeros of the long-run effects, * free (empty: none)")] = ""
    name: str = "svar"
    starts: Annotated[int, Widget("number", min=1, max=1000, advanced=True, help="starting points of the likelihood's maximisation")] = 10
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="random seed of the starting points")] = 0

    def validate(self, session: Session) -> list[str]:
        try:
            self._scheme()
        except SpecificationError as exc:
            return [str(exc)]
        return []

    def _scheme(self) -> Any:
        from ..structural import AB

        parts = {k: _matrix_text(getattr(self, k), k) for k in ("A", "B", "long_run")}
        parts.update({k: getattr(self, k) for k, d in (("starts", 10), ("seed", 0)) if getattr(self, k) != d})
        return AB(**{k: v for k, v in parts.items() if v is not None})

    def to_call(self) -> Any:
        from ..calls import method_call
        from ..codec import Ref
        from ..var import FittedVAR

        return method_call(FittedVAR, "identify", Ref(self.var), output=self.name, recorded_by=self.command_id, scheme=self._scheme())

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/TVP-VAR with stochastic volatility...", id="pymetrix:fit_tvp_var")
@dataclass(frozen=True)
class FitTVPVAR(Command):
    """Estimate a VAR whose coefficients, contemporaneous relations and shock volatilities drift over time (Primiceri).

    Estimated by Gibbs sampling, with priors from least squares on a training sample before the estimation sample.
    The window shows each shock's volatility over time; responses at any date come from tvp.irf(date). Long chains
    take minutes: start with a few hundred draws to see that it runs.

    The prior scales (k_Q, k_S, k_W, under Advanced) decide how much drift the data may show: larger values let
    coefficients and volatilities move more, and the results can depend on them, especially k_Q. The training
    sample is used only for the priors and is not part of the estimation sample. Primiceri found that most of the
    change in post-war US data was in the volatilities rather than the coefficients; check that a story of changing
    transmission survives other prior scales and a longer chain.
    """

    topics: ClassVar[tuple[str, ...]] = ('var', 'tvp-var')

    variables: Annotated[str, Widget("series", depends_on="data", multiple=True,
                                     help="in the list's order: the recursive ordering of the shocks (A_t lower triangular)")]
    data: Annotated[str, Widget("object", source="Dataset",
                                help="a single time series (not a panel); its first Training periods set the priors")] = "data"
    lags: Annotated[int, Widget("number", min=1, max=8,
                                help="lags of every variable; each adds drifting coefficients, so 2 (Primiceri's choice) is usual")] = 2
    draws: Annotated[int, Widget("number", advanced=True, min=10, max=100_000, step=500,
                                 help="Gibbs draws kept after the burn-in; thousands for stable bands, a few hundred to try a run")] = 2000
    burn: Annotated[int, Widget("number", advanced=True, min=0, max=100_000, step=500, help="Gibbs draws discarded first")] = 1000
    training: Annotated[int, Widget("number", min=10, max=400, help="periods before the sample used for the priors")] = 40
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True,
                                help="random seed of the sampler; if results move with the seed, the chain is too short")] = 0
    name: str = "tvp"
    kq: Annotated[float, Widget("number", min=1e-5, max=1.0, step=0.005, advanced=True,
                                help="prior scale of the coefficients' drift (Primiceri's k_Q)")] = 0.01
    ks: Annotated[float, Widget("number", min=1e-5, max=1.0, step=0.01, advanced=True,
                                help="prior scale of the contemporaneous relations' drift (k_S)")] = 0.1
    kw: Annotated[float, Widget("number", min=1e-5, max=1.0, step=0.005, advanced=True, help="prior scale of the volatilities' drift (k_W)")] = 0.01
    thin: Annotated[int, Widget("number", min=1, max=100, advanced=True, help="keep every this-many-th draw")] = 1

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref
        from ..tvpvar import TVPVAR

        names = [v.strip() for v in self.variables.split(",") if v.strip()]
        return make_call("pymetrix:tvp_var.fit", TVPVAR(names, self.lags), output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                         draws=self.draws, burn=self.burn, training=self.training, seed=self.seed,
                         **{k: getattr(self, k) for k, d in (("kq", 0.01), ("ks", 0.1), ("kw", 0.01), ("thin", 1)) if getattr(self, k) != d})

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/Factor-augmented VAR...")
@dataclass(frozen=True)
class FitFAVAR(Command):
    """Estimate a factor-augmented VAR (Bernanke, Boivin and Eliasz).

    Factors are the principal components of a panel of many series, in a VAR with the observed variables (ordered
    last, so a recursive ordering puts policy last). Factors can be a number or Bai and Ng's icp1, icp2 or icp3; lags
    a number or aic, bic, hq or fpe. Slow-moving series (output, prices) purge the factors of the observed variables.
    """

    topics: ClassVar[tuple[str, ...]] = ('large-and-switching-vars',)

    panel: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the panel the factors come from")]
    observed: Annotated[str, Widget("series", depends_on="data", multiple=True, help="observed variables in the VAR, e.g. the policy rate")]
    factors: Annotated[str, Widget("text", help="a number, or icp1, icp2, icp3 (Bai and Ng)")] = "3"
    lags: Annotated[str, Widget("text", help="a number, or aic, bic, hq, fpe")] = "2"
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    name: str = "fv"  # not "favar": the result would hide the function in the recorded script
    slow: Annotated[str, Widget("series", depends_on="data", multiple=True, advanced=True,
                                help="slow-moving series of the panel, for the purge (optional)")] = ""
    max_factors: Annotated[int, Widget("number", min=1, max=30, advanced=True, help="most factors the criterion considers")] = 8
    max_lags: Annotated[int, Widget("number", min=1, max=24, advanced=True, help="most lags the criterion considers")] = 8

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        def names(text: str) -> list[str]:
            return [v.strip() for v in text.split(",") if v.strip()]

        def number(text: str) -> Any:
            text = text.strip().lower()
            return int(text) if text.isdigit() else text

        options: dict[str, Any] = {}
        if names(self.slow):
            options["slow"] = names(self.slow)
        if isinstance(number(self.factors), str) and self.max_factors != 8:
            options["max_factors"] = self.max_factors
        if isinstance(number(self.lags), str) and self.max_lags != 8:
            options["max_lags"] = self.max_lags
        return make_call("pymetrix:favar", output=self.name, recorded_by=self.command_id, data=Ref(self.data), panel=names(self.panel),
                         observed=names(self.observed), factors=number(self.factors), lags=number(self.lags), **options)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/Equation lag search...")
@dataclass(frozen=True)
class EquationLagSearch(Command):
    """Choose how many lags an equation takes.

    Lags 1 to p of the variables searched (by default the left-hand side as written) are added for p = 0 up to the
    maximum, each fitted on the same sample, and compared by AIC, BIC and Hannan-Quinn.

    BIC is consistent and picks short lags; AIC picks longer ones and suits forecasting. Whichever you choose, check
    the chosen equation's residuals for remaining autocorrelation (Test > Correlogram): too few lags leave dynamics
    in the errors and bias the other coefficients when a lagged dependent variable is present. A criterion at its
    minimum at the largest lag tried suggests trying a larger maximum.
    """

    topics: ClassVar[tuple[str, ...]] = ('least-squares',)

    formula: Annotated[str, Widget("formula", help="the equation without the lags searched, e.g. dlog(c) ~ dlog(y)")]
    of: Annotated[str, Widget("text", help="the variables whose lags are added, comma-separated; empty = the left-hand side")] = ""
    max_lags: Annotated[int, Widget("number", min=1, max=24,
                                    help="longest lag tried; the common sample starts after it, so a large maximum costs observations")] = 8
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    name: str = "lag_search"

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref
        from ..estimation import Equation

        of = [v.strip() for v in self.of.split(",") if v.strip()]
        return make_call("pymetrix:equation.select_lags", Equation(self.formula), output=self.name, recorded_by=self.command_id,
                         data=Ref(self.data), max_lags=self.max_lags, **({"of": of} if of else {}))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/VAR lag selection...")
@dataclass(frozen=True)
class VARLagOrder(Command):
    """Information criteria for choosing a VAR's lag length.

    All lag lengths up to the maximum are compared on the same sample. For a panel VAR give its entity axis (under
    Advanced): Andrews and Lu's MBIC, MAIC and MQIC from Hansen's J, with the same instruments for every lag length.

    The criteria often disagree: BIC and HQ choose fewer lags than AIC and FPE, and in VARs with several variables
    a few extra lags cost many parameters. Short lags may leave serial correlation in the residuals (check it with a
    portmanteau test on the fitted VAR, Test > Run test); long lags make impulse responses noisy. For
    quarterly data, four lags is a common check on whatever the criteria choose.
    """

    topics: ClassVar[tuple[str, ...]] = ('var',)

    variables: Annotated[str, Widget("series", depends_on="data", multiple=True)]
    max_lags: Annotated[int, Widget("number", min=1, max=24,
                                    help="longest lag compared; all lengths use the sample this allows, so it costs observations")] = 8
    exog: Annotated[str, Widget("text", help="exogenous expressions (optional)", advanced=True)] = ""
    trend: Literal["c", "ct", "n"] = "c"
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    name: str = "lag_order"
    by: Annotated[str, Widget("text", advanced=True, help="a panel's entity axis, for a panel VAR (Andrews and Lu's criteria)")] = ""

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        spec, _ = _var_spec(self.variables, "aic", self.max_lags, self.exog, self.trend, self.by)
        return make_call("pymetrix:var.select_order", spec, output=self.name, recorded_by=self.command_id, data=Ref(self.data))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


def _identified_call(session: Session, cmd: Any) -> tuple[Any, type]:
    """What impulse responses and decompositions are computed on: the VAR or VECM by name, or a call identifying it;
    and its class."""
    from ..calls import method_call
    from ..codec import Ref

    _, identification = _identified(cmd)
    cls = type(session[cmd.var])
    return (Ref(cmd.var) if identification is None else method_call(cls, "identify", Ref(cmd.var), scheme=identification)), cls


def _sign_text(text: str) -> dict[str, dict[str, str]]:
    """'supply: gy+, pi-; demand: gy+, pi+' → {"supply": {"gy": "+", "pi": "-"}, "demand": {"gy": "+", "pi": "+"}}."""
    out: dict[str, dict[str, str]] = {}
    for part in (p.strip() for p in text.replace("\n", ";").split(";") if p.strip()):
        shock, colon, rest = part.partition(":")
        if not colon or not shock.strip():
            raise SpecificationError(f"{part!r}: write each shock as name: variable+, variable-")
        signs = {}
        for item in (i.strip() for i in rest.split(",") if i.strip()):
            if item[-1] not in "+-" or not item[:-1].strip():
                raise SpecificationError(f"{item!r}: end each variable with + or -")
            signs[item[:-1].strip()] = item[-1]
        out[shock.strip()] = signs
    if not out:
        raise SpecificationError("give at least one shock and the signs of its responses")
    return out


def _major_text(text: str) -> dict[str, dict[str, str]]:
    """'monetary: 1979M10 ff, 1980M5 ff' → {"monetary": {"1979M10": "ff", "1980M5": "ff"}}."""
    out: dict[str, dict[str, str]] = {}
    for part in (p.strip() for p in text.replace("\n", ";").split(";") if p.strip()):
        shock, colon, rest = part.partition(":")
        if not colon or not shock.strip():
            raise SpecificationError(f"{part!r}: write each shock as name: date variable, date variable")
        events = {}
        for item in (i.split() for i in rest.split(",") if i.strip()):
            if len(item) != 2:
                raise SpecificationError(f"{' '.join(item)!r}: give a date and the variable the shock moved most, e.g. 1979M10 ff")
            events[item[0]] = item[1]
        out[shock.strip()] = events
    return out


def _zero_text(text: str) -> dict[str, list[str]]:
    """'monetary: gy, pi; demand: r' → {"monetary": ["gy", "pi"], "demand": ["r"]}."""
    out: dict[str, list[str]] = {}
    for part in (p.strip() for p in text.replace("\n", ";").split(";") if p.strip()):
        shock, colon, rest = part.partition(":")
        if not colon or not shock.strip():
            raise SpecificationError(f"{part!r}: write each shock as name: variable, variable")
        names = [v.strip() for v in rest.split(",") if v.strip()]
        if not names:
            raise SpecificationError(f"{part!r}: name the variables the shock does not move")
        out[shock.strip()] = names
    return out


_SIGN_OPTIONS = (("rotations", "rotations", 1000), ("max_tries", "max_tries", 200_000), ("sign_seed", "seed", 0))


def _identified(cmd: Any) -> tuple[str, Any]:
    var = cmd.var
    if cmd.scheme == "LongRun":
        return f"{var}.identify(LongRun())", LongRun()
    if cmd.scheme == "SignRestrictions":
        from ..var import SignRestrictions

        zeros = {f: _zero_text(getattr(cmd, f, "")) for f in ("zeros", "long_run_zeros") if getattr(cmd, f, "").strip()}
        restrictions = _sign_text(cmd.signs) if cmd.signs.strip() or not zeros else {}
        options: dict[str, Any] = {"horizon": cmd.sign_horizon} if cmd.sign_horizon else {}
        options.update({arg: getattr(cmd, f) for f, arg, d in _SIGN_OPTIONS if getattr(cmd, f) != d})
        if cmd.narrative_signs.strip():
            options["narrative_signs"] = _sign_text(cmd.narrative_signs)
        if cmd.narrative_major.strip():
            options["narrative_major"] = _major_text(cmd.narrative_major)
        options.update(zeros)
        extra = "".join(f", {k}={v!r}" for k, v in options.items())
        return f"{var}.identify(SignRestrictions({restrictions!r}{extra}))", SignRestrictions(restrictions, **options)
    if cmd.scheme in ("MaxShare", "News"):
        from ..var import MaxShare

        scheme = MaxShare(cmd.share_variable.strip(), horizon=cmd.share_horizon, start=cmd.share_start, zero_impact=cmd.scheme == "News")
        extra = "".join(f", {k}={v!r}" for k, v, d in (("horizon", scheme.horizon, 40), ("start", scheme.start, 0),
                                                       ("zero_impact", scheme.zero_impact, False)) if v != d)
        return f"{var}.identify(MaxShare({scheme.variable!r}{extra}))", scheme
    if cmd.scheme in ("Heteroskedasticity", "Generalised"):  # not an identify(...) scheme: see _structural_call
        return var, None
    if _names(cmd.ordering):
        return f"{var}.identify(Cholesky({_names(cmd.ordering)!r}))", Cholesky(_names(cmd.ordering))
    return var, None


# --- more identification and more VAR types in the impulse-response and decomposition dialogs (dlg/var) ---

_SCHEME_HELP = ("Cholesky (recursive, by Ordering), LongRun (Blanchard-Quah), SignRestrictions (Sign restrictions, and any "
                "Zero restrictions), "
                "MaxShare (the shock explaining most of a variable's forecast-error variance), News (the same, with no "
                "effect on impact), Heteroskedasticity (Rigobon: shifts in the shocks' variances at Variance regime breaks)")
_SCHEME_HELP_HD = _SCHEME_HELP.rpartition(", Heteroskedasticity")[0]  # historical decompositions: no Rigobon scheme
_SHARE_SCHEMES = ("MaxShare", "News")
_RECURSIVE_ONLY = ("FittedBVARSV", "FittedMSVAR", "FittedQuantileVAR")  # their own recursive responses


def _scheme_problems(cmd: Any, session: Session, decomposition: str = "") -> list[str]:
    """What is wrong with the identification asked for, given the object it is asked of."""
    problems = _sign_problems(cmd)
    obj = session.objects.get(cmd.var)
    kind = type(obj).__name__
    if kind in _RECURSIVE_ONLY:
        if kind == "FittedQuantileVAR" and decomposition == "historical decomposition":
            return ["a quantile VAR has no historical decomposition (its quantile residuals are not centred, so there is no "
                    "clean split into a baseline and shocks): use its responses or variance decomposition"]
        if cmd.scheme != "Cholesky" or _names(cmd.ordering):
            return [f"a {kind[6:]}'s responses are recursive in its own order: choose Cholesky with Ordering empty"]
        return problems
    from ..var import FittedVAR

    if cmd.scheme in _SHARE_SCHEMES:
        if not cmd.share_variable.strip():
            problems.append("max share: name the variable whose forecast-error variance the shock explains, e.g. tfp")
        elif obj is not None and cmd.share_variable.strip() not in getattr(obj, "variables", ()):
            problems.append(f"max share: {cmd.share_variable.strip()!r} is not a variable of {cmd.var}")
        if cmd.share_start > cmd.share_horizon:
            problems.append("max share: the first horizon is after the last")
    if cmd.scheme == "Heteroskedasticity" and not _names(cmd.breaks):
        problems.append("heteroskedasticity: give the first period of each new variance regime, e.g. 1984Q1")
    if cmd.scheme in (*_SHARE_SCHEMES, "Heteroskedasticity", "Generalised") and obj is not None and not isinstance(obj, FittedVAR):
        problems.append(f"{cmd.scheme} identification is for VARs; a {kind} takes Cholesky")
    return problems


def _own_arguments(cmd: Any, kind: str, method: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """The arguments of a BVAR-SV's, MS-VAR's or quantile VAR's own irf, fevd or historical_decomposition: Period and
    the bands' coverage (BVAR-SV), Regime (MS-VAR); a BVAR-SV's historical decomposition uses every period's
    volatilities and the first Draws posterior draws (0: all), an MS-VAR's the most likely regime path. An MS-VAR's
    bands take Draws (0: none), Coverage and Seed: draws of its parameters from the estimates' asymptotic normal."""
    own = {"periods": arguments["periods"]} if "periods" in arguments else {}
    if kind == "FittedBVARSV":
        own["coverage"] = cmd.coverage
        if method == "historical_decomposition":
            own.update({"draws": cmd.draws} if cmd.draws else {})
        elif cmd.period.strip():
            own["period"] = cmd.period.strip()
    elif kind == "FittedMSVAR":
        if method != "historical_decomposition":
            own["regime"] = cmd.regime
        if getattr(cmd, "draws", 0):  # bands from draws of the parameters (asymptotic normal of the estimates)
            own.update(draws=cmd.draws, coverage=cmd.coverage, **({"seed": cmd.seed} if cmd.seed else {}))
    return own


def _structural_call(session: Session, cmd: Any, method: str, **arguments: Any) -> Any:
    """The call of ``method`` (irf, fevd, historical_decomposition) under the dialog's identification: on the VAR as
    identified, on its heteroskedasticity SVAR, its generalised responses, or a BVAR-SV's, MS-VAR's or quantile VAR's
    own responses."""
    from ..calls import method_call
    from ..codec import Ref

    obj = session[cmd.var]
    kind = type(obj).__name__
    output = {"output": cmd.name, "recorded_by": cmd.command_id}
    if kind in _RECURSIVE_ONLY:
        return method_call(type(obj), method, Ref(cmd.var), **output, **_own_arguments(cmd, kind, method, arguments))
    if cmd.scheme == "Heteroskedasticity":
        from ..identification_extra import HeteroskedasticitySVAR

        hsvar = method_call(type(obj), "heteroskedasticity_svar", Ref(cmd.var), breaks=_names(cmd.breaks))
        keep = {k: v for k, v in arguments.items() if k in ("periods", "cumulative")}
        return method_call(HeteroskedasticitySVAR, method, hsvar, **output, **keep)
    if cmd.scheme == "Generalised":
        return method_call(type(obj), "girf", Ref(cmd.var), **output, **arguments)
    receiver, cls = _identified_call(session, cmd)
    return method_call(cls, method, receiver, **output, **arguments)


@command(menu="Estimate/Impulse responses...")
@dataclass(frozen=True)
class ImpulseResponse(Command):
    """Impulse responses of a VAR or VECM, with bootstrap bands.

    Choose the identification scheme: Cholesky orders shocks as the variables are ordered; LongRun imposes
    Blanchard-Quah long-run restrictions; SignRestrictions names shocks by the signs of their responses
    (supply: gy+, pi-; demand: gy+, pi+), holding on impact and for the periods after given; Zeros adds the variables a
    shock does not move on impact (monetary: gy, pi), with the signs or alone (Arias, Rubio-Ramírez and Waggoner's
    zero and sign restrictions). MaxShare identifies one shock as the one explaining most of a variable's
    forecast-error variance up to a horizon (Uhlig; Francis et al.), News the same with no effect on that variable on
    impact (Barsky and Sims); Heteroskedasticity identifies every shock by the change in their variances at the
    Breaks (Rigobon; no bands); Generalised gives Pesaran and Shin's ordering-free responses. Bayesian VARs with
    stochastic volatility, Markov-switching and quantile VARs give their own recursive responses (Cholesky, in their
    order): at Period's volatilities, in Regime, or at the quantile fitted (a Markov-switching VAR's bands: Draws draws
    of its parameters from the estimates' asymptotic normal distribution). Tick Cumulative for variables in differences.

    Cholesky responses depend on the ordering: the first variable responds on impact to its own shock only, the last
    to all of them. Try other plausible orderings; conclusions that change with it rest on the ordering, not the
    data. Responses are to one-standard-deviation shocks unless Orthogonal is unticked. Bands are percentile bands:
    a parametric bootstrap for a VAR, a residual bootstrap for a VECM, posterior draws for a Bayesian VAR. Under
    sign restrictions the response shown is the pointwise median of the accepted rotations, which need not come
    from any single one, and the bands describe the identified set as well as sampling uncertainty. A stationary
    VAR's responses die out; a VECM's responses are of levels, and permanent shocks have lasting effects.
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import VAR, Cholesky, Dataset

        data = Dataset.example("sample_macro").derive(dy="400*dlog(y)", dc="400*dlog(c)", di="400*dlog(i)")
        var = VAR(["dy", "dc", "di"], lags=2).fit(data)
        irf = var.irf(12, draws=200)   # Cholesky in the VAR's order; bands from 200 bootstrap draws (dialog: 500)
        irf.to_frame()                 # a column "response <- shock" for each pair, with its lower and upper band
        # another ordering, and responses summed over horizons (for variables in differences)
        var.identify(Cholesky(["di", "dy", "dc"])).irf(12, cumulative=True, draws=200)
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Graph", ("a panel per response and shock with its band; where the band includes zero the response is not "
                   "clearly different from zero")),
        ("Selected horizons", ("the responses at selected horizons, a column 'response <- shock' for each pair; the "
                               "notes give the identification and the bands' source")),
        ("All horizons", "every horizon, with each response's lower and upper band"),
        ("Coverage", ("95% by default: pointwise percentile bands, not joint bands for the whole path; 68% bands are "
                      "narrower and also common")),
        ("Heatmap", "every response to every shock by horizon, coloured on a scale symmetric about zero"),
    )

    topics: ClassVar[tuple[str, ...]] = ('var', 'structural-var', 'large-and-switching-vars')

    var: Annotated[str, Widget("object", source="FittedVAR,FittedVECM,FittedBVARSV,FittedMSVAR,FittedQuantileVAR")]
    periods: Annotated[int, Widget("number", min=1, max=400,
                                   help="periods after the shock to trace; several years of the data's frequency to see responses die out")] = 20
    scheme: Annotated[Literal["Cholesky", "LongRun", "SignRestrictions", "MaxShare", "News", "Heteroskedasticity", "Generalised"],
                      Widget("", help=_SCHEME_HELP + "; or Generalised (Pesaran-Shin, ordering-free)")] = "Cholesky"
    ordering: Annotated[str, Widget("text", help="Cholesky order, most exogenous first; empty = VAR order", show_if=("scheme", ("Cholesky",)))] = ""
    signs: Annotated[str, Widget("text", help="each shock and the signs of its responses: supply: gy+, pi-; demand: gy+, pi+",
                                 show_if=("scheme", ("SignRestrictions",)))] = ""
    sign_horizon: Annotated[int, Widget("number", min=0, max=40, help="the signs hold on impact and for this many periods after",
                                        show_if=("scheme", ("SignRestrictions",)))] = 0
    zeros: Annotated[str, Widget("text", help="zero restrictions on impact: each shock and the variables it does not move, e.g. monetary: gy, pi",
                                 show_if=("scheme", ("SignRestrictions",)))] = ""
    long_run_zeros: Annotated[str, Widget("text", advanced=True, show_if=("scheme", ("SignRestrictions",)),
                                          help="zero restrictions in the long run (cumulated responses), e.g. demand: gy")] = ""
    share_variable: Annotated[str, Widget("text", show_if=("scheme", ("MaxShare", "News")),
                                          help="the variable whose forecast-error variance the shock explains most of, e.g. tfp")] = ""
    share_horizon: Annotated[int, Widget("number", min=0, max=400, show_if=("scheme", ("MaxShare", "News")),
                                         help="the share is of the forecast-error variance up to this horizon, e.g. 40 quarters")] = 40
    share_start: Annotated[int, Widget("number", min=0, max=400, advanced=True, show_if=("scheme", ("MaxShare", "News")),
                                       help="max share: the first horizon counted (0: impact)")] = 0
    breaks: Annotated[str, Widget("text", show_if=("scheme", ("Heteroskedasticity",)),
                                  help="the first period of each new variance regime, e.g. 1984Q1 (two dates: three regimes)")] = ""
    cumulative: bool = False
    draws: Annotated[int, Widget("number", advanced=True, help="bootstrap replications (or posterior draws for a Bayesian VAR) for bands; 0 = none", min=0, max=10000, step=100)] = 500
    name: str = "irf"
    orthogonal: Annotated[bool, Widget("checkbox", advanced=True, help="untick for responses to unit reduced-form shocks")] = True
    coverage: Annotated[float, Widget("number", min=0.5, max=0.999, step=0.05, advanced=True,
                                      help="probability the bands cover; 0.68 (one standard error) is also common for responses")] = 0.95
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True,
                                help="seed of the bootstrap or posterior draws; bands that move with it need more draws")] = 0
    rotations: Annotated[int, Widget("number", min=10, max=100_000, step=100, advanced=True, show_if=("scheme", ("SignRestrictions",)),
                                     help="sign restrictions: rotations accepted")] = 1000
    max_tries: Annotated[int, Widget("number", min=1000, max=10_000_000, step=10_000, advanced=True, show_if=("scheme", ("SignRestrictions",)),
                                     help="sign restrictions: rotations tried before giving up")] = 200_000
    sign_seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, show_if=("scheme", ("SignRestrictions",)),
                                     help="sign restrictions: random seed of the rotations")] = 0
    narrative_signs: Annotated[str, Widget("text", advanced=True, show_if=("scheme", ("SignRestrictions",)),
                                           help="narrative restrictions: the shock's sign in named periods, e.g. monetary: 1979M10+, 1990M12-")] = ""
    narrative_major: Annotated[str, Widget("text", advanced=True, show_if=("scheme", ("SignRestrictions",)),
                                           help="narrative restrictions: the variable the shock moved most, e.g. monetary: 1979M10 ff")] = ""
    period: Annotated[str, Widget("period", advanced=True, group="BVAR-SV and MS-VAR",
                                  help="BVAR with stochastic volatility: the period whose volatilities size the shocks (empty: the last)")] = ""
    regime: Annotated[int, Widget("number", min=0, max=9, advanced=True, group="BVAR-SV and MS-VAR",
                                  help="Markov-switching VAR: the regime whose dynamics and covariance give the responses (0: the first)")] = 0

    def _options(self, vecm: bool = False) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.cumulative:
            options["cumulative"] = True
        if not self.orthogonal:
            options["orthogonal"] = False
        if self.draws != 500 or (vecm and self.draws):  # a VECM's irf draws none unless asked
            options["draws"] = self.draws
        if self.draws and self.coverage != 0.95:
            options["alpha"] = round(1 - self.coverage, 6)
        if self.draws and self.seed:
            options["seed"] = self.seed
        return options

    def validate(self, session: Session) -> list[str]:
        return _scheme_problems(self, session)

    def to_call(self, session: Session) -> Any:
        from ..vecm import FittedVECM

        vecm = isinstance(session[self.var], FittedVECM)
        options = self._options(vecm=vecm)
        if vecm and not self.draws:
            options.pop("draws", None)  # its default
        if self.scheme == "Generalised":
            options.pop("orthogonal", None)
            return _structural_call(session, self, "girf", periods=self.periods, **options)
        return _structural_call(session, self, "irf", periods=self.periods, **options)

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        target, _ = _identified(self)
        opts = "".join(f", {k}={v!r}" for k, v in self._options().items())
        if self.scheme == "Heteroskedasticity":
            cumulative = ", cumulative=True" if self.cumulative else ""
            return f"{self.name} = {self.var}.heteroskedasticity_svar({_names(self.breaks)!r}).irf({self.periods}{cumulative})"
        if self.scheme == "Generalised":
            return f"{self.name} = {self.var}.girf({self.periods}{opts.replace(', orthogonal=False', '')})"
        return f"{self.name} = {target}.irf({self.periods}{opts})"


def _sign_problems(cmd: Any) -> list[str]:
    if cmd.scheme != "SignRestrictions":
        return []
    try:
        _identified(cmd)
    except (ValueError, SpecificationError) as exc:
        return [f"sign restrictions: {exc}"]
    return []


@command(menu="Estimate/Variance decomposition...")
@dataclass(frozen=True)
class VarianceDecomposition(Command):
    """Forecast-error variance decomposition of a VAR or VECM.

    The share of each variable's forecast-error variance due to each identified shock, by horizon.

    The shares, in per cent, add up to 100 for every variable at each horizon. Short horizons show what moves a
    variable now, long ones what drives its fluctuations; for a stationary VAR the shares settle as the horizon grows. The shares
    depend on the identification as much as the responses do: under Cholesky the first variable's own shock
    explains all of its variance on impact by construction. A shock that explains little of a variable's variance
    can still have a significant response; the two answer different questions.

    Under MaxShare or News the first shock is the one identified (its share of the chosen variable is as large as any
    shock's can be over those horizons) and the others only complete the decomposition; under Heteroskedasticity the
    shares are those of the first variance regime, where every shock has unit variance.

    A Bayesian VAR with stochastic volatility gives the shares with the volatilities of Period (empty: the last),
    posterior means with Coverage bands over the draws; a Markov-switching VAR those of Regime, held fixed over the
    horizon, with Coverage bands over Draws draws of its parameters from the estimates' asymptotic normal distribution
    (0: none); a quantile VAR those of its quantile system with the covariance of its quantile residuals (Ando,
    Greenwood-Nimmo and Shin's quantile connectedness, orthogonalised). All three are recursive in the VAR's order:
    choose Cholesky with Ordering empty.
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import VAR, Cholesky, Dataset

        data = Dataset.example("sample_macro").derive(dy="400*dlog(y)", dc="400*dlog(c)", di="400*dlog(i)")
        var = VAR(["dy", "dc", "di"], lags=2).fit(data)
        fevd = var.fevd(12)            # Cholesky in the VAR's order, horizons 1 to 12
        fevd.table("dy")               # % of dy's forecast-error variance due to each shock, by horizon; rows add to 100
        var.identify(Cholesky(["di", "dy", "dc"])).fevd(12).table("dy")   # the same under another ordering
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Decomposition", ("per cent of each variable's forecast-error variance due to each shock, columns 'variable: "
                           "shock', at horizons 1 (impact) onwards")),
        ("A tab per variable", "the same shares for that variable alone; each row adds to 100"),
        ("Heatmap", "the shares at the last horizon, every variable and shock at once"),
        ("fevd.table(variable)", "the variable's shares by horizon, for use in code"),
    )

    topics: ClassVar[tuple[str, ...]] = ('var', 'structural-var', 'large-and-switching-vars')

    var: Annotated[str, Widget("object", source="FittedVAR,FittedVECM,FittedBVARSV,FittedMSVAR,FittedQuantileVAR")]
    periods: Annotated[int, Widget("number", min=1, max=400, help="forecast horizons decomposed, 1 to this; long enough for the shares to settle")] = 20
    scheme: Annotated[Literal["Cholesky", "LongRun", "SignRestrictions", "MaxShare", "News", "Heteroskedasticity"],
                      Widget("", help=_SCHEME_HELP)] = "Cholesky"
    ordering: Annotated[str, Widget("text", help="Cholesky order, most exogenous first; empty = VAR order", show_if=("scheme", ("Cholesky",)))] = ""
    signs: Annotated[str, Widget("text", help="each shock and the signs of its responses: supply: gy+, pi-; demand: gy+, pi+",
                                 show_if=("scheme", ("SignRestrictions",)))] = ""
    sign_horizon: Annotated[int, Widget("number", min=0, max=40, help="the signs hold on impact and for this many periods after",
                                        show_if=("scheme", ("SignRestrictions",)))] = 0
    zeros: Annotated[str, Widget("text", help="zero restrictions on impact: each shock and the variables it does not move, e.g. monetary: gy, pi",
                                 show_if=("scheme", ("SignRestrictions",)))] = ""
    long_run_zeros: Annotated[str, Widget("text", advanced=True, show_if=("scheme", ("SignRestrictions",)),
                                          help="zero restrictions in the long run (cumulated responses), e.g. demand: gy")] = ""
    share_variable: Annotated[str, Widget("text", show_if=("scheme", ("MaxShare", "News")),
                                          help="the variable whose forecast-error variance the shock explains most of, e.g. tfp")] = ""
    share_horizon: Annotated[int, Widget("number", min=0, max=400, show_if=("scheme", ("MaxShare", "News")),
                                         help="the share is of the forecast-error variance up to this horizon, e.g. 40 quarters")] = 40
    share_start: Annotated[int, Widget("number", min=0, max=400, advanced=True, show_if=("scheme", ("MaxShare", "News")),
                                       help="max share: the first horizon counted (0: impact)")] = 0
    breaks: Annotated[str, Widget("text", show_if=("scheme", ("Heteroskedasticity",)),
                                  help="the first period of each new variance regime, e.g. 1984Q1 (two dates: three regimes)")] = ""
    name: str = "fevd"
    rotations: Annotated[int, Widget("number", min=10, max=100_000, step=100, advanced=True, show_if=("scheme", ("SignRestrictions",)),
                                     help="sign restrictions: rotations accepted")] = 1000
    max_tries: Annotated[int, Widget("number", min=1000, max=10_000_000, step=10_000, advanced=True, show_if=("scheme", ("SignRestrictions",)),
                                     help="sign restrictions: rotations tried before giving up")] = 200_000
    sign_seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, show_if=("scheme", ("SignRestrictions",)),
                                     help="sign restrictions: random seed of the rotations")] = 0
    narrative_signs: Annotated[str, Widget("text", advanced=True, show_if=("scheme", ("SignRestrictions",)),
                                           help="narrative restrictions: the shock's sign in named periods, e.g. monetary: 1979M10+, 1990M12-")] = ""
    narrative_major: Annotated[str, Widget("text", advanced=True, show_if=("scheme", ("SignRestrictions",)),
                                           help="narrative restrictions: the variable the shock moved most, e.g. monetary: 1979M10 ff")] = ""
    period: Annotated[str, Widget("period", advanced=True, group="BVAR-SV and MS-VAR",
                                  help="BVAR with stochastic volatility: the period whose volatilities size the shocks (empty: the last)")] = ""
    coverage: Annotated[float, Widget("number", min=0.5, max=0.999, step=0.05, advanced=True, group="BVAR-SV and MS-VAR",
                                      help="BVAR with stochastic volatility or Markov-switching VAR: probability the bands cover")] = 0.68
    regime: Annotated[int, Widget("number", min=0, max=9, advanced=True, group="BVAR-SV and MS-VAR",
                                  help="Markov-switching VAR: the regime held fixed over the horizon (0: the first)")] = 0
    draws: Annotated[int, Widget("number", min=0, max=10000, step=100, advanced=True, group="BVAR-SV and MS-VAR",
                                 help="Markov-switching VAR: draws of the parameters for bands; 0 = none")] = 0
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, group="BVAR-SV and MS-VAR",
                                help="Markov-switching VAR: seed of the parameter draws")] = 0

    def validate(self, session: Session) -> list[str]:
        return _scheme_problems(self, session, "variance decomposition")

    def to_call(self, session: Session) -> Any:
        return _structural_call(session, self, "fevd", periods=self.periods)

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        if self.scheme == "Heteroskedasticity":
            return f"{self.name} = {self.var}.heteroskedasticity_svar({_names(self.breaks)!r}).fevd({self.periods})"
        target, _ = _identified(self)
        return f"{self.name} = {target}.fevd({self.periods})"


@command(menu="Estimate/Historical decomposition...")
@dataclass(frozen=True)
class HistoricalDecomposition(Command):
    """Historical decomposition of a VAR, VECM, panel VAR or proxy SVAR.

    Each variable over the estimation sample, split into the contributions of the identified shocks and the
    baseline (deterministic terms and initial conditions); the parts add up to the data.

    It answers which shocks drove a variable away from its baseline in a given episode, such as how much of a
    recession was monetary. The split depends on the identification, as impulse responses do; under sign restrictions
    one member of the identified set is used. Early in the sample the baseline carries most of the movement,
    because few past shocks have accumulated; read the decomposition from a few years in.

    Draws adds bands to each contribution: quantiles over a Bayesian VAR's posterior draws, or over the identified set
    under sign (or zero and sign) restrictions; the point decomposition stays as it is. Under MaxShare or News only
    the first shock is identified: the others' contributions are one way of splitting the rest.

    A panel VAR is decomposed entity by entity, its baseline starting from the entity's first lags with its fixed
    effect; Entity picks the one the window shows (hd.select(entity) another), and its bands come from draws of the
    coefficients. A proxy SVAR splits each variable into its identified shock, the other shocks together and the
    baseline, with bands from its moving-block bootstrap; its shock is identified by the proxy, so leave the scheme at
    Cholesky with no ordering.

    A Bayesian VAR with stochastic volatility is decomposed draw by draw, each period's shocks through that period's
    impact matrix: the point is the mean over the posterior draws (the first Draws of them; 0: all), with Coverage
    bands. A Markov-switching VAR is decomposed along its most likely regime path (the regime with the highest
    smoothed probability each period supplies the intercepts, dynamics and impact), its intercepts in the baseline;
    Draws gives Coverage bands over draws of its parameters from the estimates' asymptotic normal distribution, the
    regime path held fixed (the bands leave out the uncertainty about the regimes).
    Both are recursive in the VAR's order (Cholesky, Ordering empty), and their parts add up to the data exactly.
    """

    topics: ClassVar[tuple[str, ...]] = ('var', 'structural-var', 'large-and-switching-vars')

    var: Annotated[str, Widget("object", source="FittedVAR,FittedVECM,FittedPanelVAR,ProxySVAR,FittedBVARSV,FittedMSVAR")]
    scheme: Annotated[Literal["Cholesky", "LongRun", "SignRestrictions", "MaxShare", "News"],
                      Widget("", help=_SCHEME_HELP_HD)] = "Cholesky"
    ordering: Annotated[str, Widget("text", help="Cholesky order, most exogenous first; empty = VAR order", show_if=("scheme", ("Cholesky",)))] = ""
    signs: Annotated[str, Widget("text", help="each shock and the signs of its responses: supply: gy+, pi-; demand: gy+, pi+",
                                 show_if=("scheme", ("SignRestrictions",)))] = ""
    sign_horizon: Annotated[int, Widget("number", min=0, max=40, help="the signs hold on impact and for this many periods after",
                                        show_if=("scheme", ("SignRestrictions",)))] = 0
    zeros: Annotated[str, Widget("text", help="zero restrictions on impact: each shock and the variables it does not move, e.g. monetary: gy, pi",
                                 show_if=("scheme", ("SignRestrictions",)))] = ""
    long_run_zeros: Annotated[str, Widget("text", advanced=True, show_if=("scheme", ("SignRestrictions",)),
                                          help="zero restrictions in the long run (cumulated responses), e.g. demand: gy")] = ""
    share_variable: Annotated[str, Widget("text", show_if=("scheme", ("MaxShare", "News")),
                                          help="the variable whose forecast-error variance the shock explains most of, e.g. tfp")] = ""
    share_horizon: Annotated[int, Widget("number", min=0, max=400, show_if=("scheme", ("MaxShare", "News")),
                                         help="the share is of the forecast-error variance up to this horizon, e.g. 40 quarters")] = 40
    share_start: Annotated[int, Widget("number", min=0, max=400, advanced=True, show_if=("scheme", ("MaxShare", "News")),
                                       help="max share: the first horizon counted (0: impact)")] = 0
    name: str = "hd"
    draws: Annotated[int, Widget("number", advanced=True, min=0, max=10000, step=100,
                                 help="bands: posterior draws of a Bayesian VAR, rotations of a sign-identified set, a panel VAR's "
                                      "coefficient draws, a proxy SVAR's block bootstrap or parameter draws of a "
                                      "Markov-switching VAR; 0 = none")] = 0
    entity: Annotated[str, Widget("text", help="panel VARs: the entity the window shows; empty = the first")] = ""
    coverage: Annotated[float, Widget("number", min=0.5, max=0.999, step=0.05, advanced=True,
                                      help="probability the bands cover; 0.68 is also common")] = 0.95
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True,
                                help="seed of the rotations drawn for the bands")] = 0
    rotations: Annotated[int, Widget("number", min=10, max=100_000, step=100, advanced=True, show_if=("scheme", ("SignRestrictions",)),
                                     help="sign restrictions: rotations accepted")] = 1000
    max_tries: Annotated[int, Widget("number", min=1000, max=10_000_000, step=10_000, advanced=True, show_if=("scheme", ("SignRestrictions",)),
                                     help="sign restrictions: rotations tried before giving up")] = 200_000
    sign_seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, show_if=("scheme", ("SignRestrictions",)),
                                     help="sign restrictions: random seed of the rotations")] = 0
    narrative_signs: Annotated[str, Widget("text", advanced=True, show_if=("scheme", ("SignRestrictions",)),
                                           help="narrative restrictions: the shock's sign in named periods, e.g. monetary: 1979M10+, 1990M12-")] = ""
    narrative_major: Annotated[str, Widget("text", advanced=True, show_if=("scheme", ("SignRestrictions",)),
                                           help="narrative restrictions: the variable the shock moved most, e.g. monetary: 1979M10 ff")] = ""

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"entity": self.entity.strip()} if self.entity.strip() else {}
        if not self.draws:
            return options
        options = {"draws": self.draws, **options}
        if self.coverage != 0.95:
            options["alpha"] = round(1 - self.coverage, 6)
        if self.seed:
            options["seed"] = self.seed
        return options

    def validate(self, session: Session) -> list[str]:
        problems = _scheme_problems(self, session, "historical decomposition")
        if self.var in session.objects:
            from ..panelvar import FittedPanelVAR
            from ..proxy import ProxySVAR
            from ..vecm import FittedVECM

            obj = session.objects[self.var]
            if self.draws and isinstance(obj, FittedVECM):
                problems.append("bands for a historical decomposition are for VARs (Bayesian, or sign-identified); set Draws to 0 for a VECM")
            if isinstance(obj, ProxySVAR) and (self.scheme != "Cholesky" or _names(self.ordering)):
                problems.append("a proxy SVAR's shock is identified by its proxy: leave the scheme at Cholesky with no ordering")
            if self.entity.strip() and not isinstance(obj, FittedPanelVAR):
                problems.append("Entity is for panel VARs; leave it empty")
        return problems

    def to_call(self, session: Session) -> Any:
        return _structural_call(session, self, "historical_decomposition", **self._options())

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        target, _ = _identified(self)
        opts = ", ".join(f"{k}={v!r}" for k, v in self._options().items())
        return f"{self.name} = {target}.historical_decomposition({opts})"


_DETERMINISTIC = {
    "none (case 1)": "n",
    "restricted constant (case 2)": "ci",
    "unrestricted constant (case 3)": "co",
    "unrestricted constant, restricted trend (case 4)": "coli",
    "unrestricted constant and trend (case 5)": "colo",
}
Case = Literal[
    "none (case 1)", "restricted constant (case 2)", "unrestricted constant (case 3)",
    "unrestricted constant, restricted trend (case 4)", "unrestricted constant and trend (case 5)",
]


def _vecm_spec(variables: str, deterministic: str, exog: str, exog_coint: str, seasons: int, **settings: Any) -> tuple[Any, str]:
    from ..vecm import VECM

    spec = VECM(_names(variables), deterministic=_DETERMINISTIC[deterministic], exog=_names(exog),  # type: ignore[arg-type]
                exog_coint=_names(exog_coint), seasons=seasons, **settings)
    return spec, _code(spec).replace("VECM(variables=", "VECM(", 1)


@command(menu="Estimate/VECM...")
@dataclass(frozen=True)
class FitVECM(Command):
    """Estimate a vector error-correction model for cointegrated series.

    Give the series in levels. The cointegrating rank can be fixed or chosen by Johansen's trace or maximum-
    eigenvalue test; choose where the constant and trend enter.

    The rank is the number of long-run (cointegrating) relations. If the tests find rank 0 the series are not
    cointegrated and a VAR in differences is the model; full rank means the series look stationary, so a VAR in
    levels is; both stop with a message. The deterministic case changes both the rank tests and the meaning of the
    relations: case 2 puts the constant only inside them (no drift in the levels), case 3, the usual choice for
    trending macro series, lets the levels drift. Check that the estimated adjustment coefficients have the signs
    that pull the variables back towards the relation; normalisation and the signs of the cointegrating vector
    decide which.
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import VECM, Dataset

        data = Dataset.example("sample_monthly").derive(lip="log(ip)", lemp="log(emp)", lret="log(retail)")
        vecm = VECM(["lip", "lemp", "lret"], lags=4).fit(data)   # the rank by Johansen's trace tests; case 3
        print(vecm.summary())      # beta: cointegrating vectors (normalised); alpha: adjustment speeds
        vecm.short_run()           # the coefficients of the lagged differences
        # a fixed rank, with the constant only inside the relation (case 2)
        VECM(["lip", "lemp", "lret"], lags=4, rank=1, deterministic="ci").fit(data)
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("beta", ("the cointegrating vectors, the long-run relations, normalised on the first rank variables (their "
                  "entries fixed, so without standard errors)")),
        ("alpha", ("adjustment speeds: a variable pulls the relation back if its alpha and beta entry have opposite "
                   "signs; alpha near zero, weakly exogenous")),
        ("Alpha, Beta, Short run", "alpha and beta as matrices, and the coefficients of the lagged differences"),
        ("Relations, Recursive eigenvalues", ("each relation over the sample, which should look stationary, and the "
                                              "eigenvalues estimated recursively, which should be stable")),
        ("Roots, Residual correlogram", ("as many unit roots as variables less the rank, the rest inside the unit "
                                         "circle; residual correlations against the ±1.96/sqrt(T) band")),
    )

    topics: ClassVar[tuple[str, ...]] = ('cointegration',)

    variables: Annotated[str, Widget("series", depends_on="data", multiple=True, help="levels of the (cointegrated) series")]
    lags: Annotated[int, Widget("number", min=0, max=24, help="lags in first differences")] = 1
    rank: Annotated[str, Widget("text", help="a number, or trace / maxeig to choose by Johansen tests")] = "trace"
    deterministic: Case = "unrestricted constant (case 3)"
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    start: Annotated[str, Widget("period")] = ""
    end: Annotated[str, Widget("period")] = ""
    name: str = "vecm"
    exog: Annotated[str, Widget("text", advanced=True, help="unrestricted exogenous expressions")] = ""
    exog_coint: Annotated[str, Widget("text", advanced=True, help="exogenous expressions inside the cointegrating relations")] = ""
    seasons: Annotated[int, Widget("number", min=0, max=52, advanced=True, help="seasonal dummies: 4 quarterly, 12 monthly; 0 = none")] = 0
    first_season: Annotated[int, Widget("number", min=0, max=51, advanced=True,
                                        help="season of the sample's first period, counting from 0: 2 for data starting in Q3 or March")] = 0
    signif: Annotated[Literal[0.05, 0.10, 0.01], Widget("choice", advanced=True, help="significance level for choosing the rank")] = 0.05

    def _spec(self) -> tuple[Any, str]:
        rank: Any = int(self.rank) if self.rank.strip().isdigit() else self.rank.strip().lower()
        return _vecm_spec(self.variables, self.deterministic, self.exog, self.exog_coint, self.seasons,
                          lags=self.lags, rank=rank, first_season=self.first_season, signif=self.signif)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:vecm.fit", self._spec()[0], output=self.name, recorded_by=self.command_id,
                         data=Ref(self.data), span=_span(self.start, self.end))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/VECM lag selection...")
@dataclass(frozen=True)
class VECMLagOrder(Command):
    """Information criteria for choosing a VECM's lag length (in differences).

    Lag lengths 0 to the maximum are compared on the same observations by AIC, BIC, Hannan-Quinn and FPE, in the
    deterministic case chosen here. The lags counted are those of the differences, so p lags in a VAR in levels
    correspond to p - 1 here. Choose the lags before the cointegrating rank: Johansen's tests depend on them, and
    too few lags leave serial correlation that distorts the tests. BIC tends to pick fewer lags than AIC; check the
    chosen model's residuals.
    """

    topics: ClassVar[tuple[str, ...]] = ('cointegration',)

    variables: Annotated[str, Widget("series", depends_on="data", multiple=True)]
    max_lags: Annotated[int, Widget("number", min=1, max=24,
                                    help="most lagged differences compared; all lengths use the sample this allows")] = 8
    deterministic: Case = "unrestricted constant (case 3)"
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    name: str = "vecm_lags"
    exog: Annotated[str, Widget("text", advanced=True)] = ""
    exog_coint: Annotated[str, Widget("text", advanced=True)] = ""
    seasons: Annotated[int, Widget("number", min=0, max=52, advanced=True)] = 0

    def _spec(self) -> tuple[Any, str]:
        return _vecm_spec(self.variables, self.deterministic, self.exog, self.exog_coint, self.seasons)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:vecm.select_order", self._spec()[0], output=self.name, recorded_by=self.command_id,
                         data=Ref(self.data), max_lags=self.max_lags)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Test/Hausman (compare two fits)...")
@dataclass(frozen=True)
class Hausman(Command):
    """Hausman test comparing a consistent and an efficient estimate (e.g. fixed against random effects).

    The efficient estimate must be consistent under the null; the test asks whether the two differ systematically.

    For fixed against random effects, a small p-value means the individual effects are correlated with the
    regressors, so random effects is inconsistent and fixed effects is preferred. A large p-value does not prove
    random effects right: the test may lack power. The classic test assumes the efficient estimator is fully
    efficient, which fails with heteroskedastic or clustered errors; then use a robust alternative, such as the
    Mundlak (correlated random effects) approach.
    """

    topics: ClassVar[tuple[str, ...]] = ('panel-data',)

    consistent: Annotated[str, Widget("object", source="FittedEquation", help="e.g. the fixed-effects fit")]
    efficient: Annotated[str, Widget("object", source="FittedEquation", help="e.g. the random-effects fit")]
    name: str = "hausman"
    sigma: Annotated[Literal["efficient", "separate"], Widget("choice", advanced=True,
                     help="efficient: both covariances use the efficient fit's error variance (Stata's sigmamore)")] = "efficient"

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:compare", output=self.name, recorded_by=self.command_id,
                         fits=(Ref(self.consistent), Ref(self.efficient)), test="hausman", sigma=self.sigma)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/Cointegrating regression...", id="pymetrix:fit_cointegrating_regression")
@dataclass(frozen=True)
class FitCointegratingRegression(Command):
    """Estimate a cointegrating relation by fully modified OLS, canonical cointegrating regression or dynamic OLS.

    All three correct least squares for endogenous regressors and serially correlated errors, so that the
    coefficients' z statistics are standard normal. Dynamic OLS adds leads and lags of the differenced regressors,
    chosen by BIC unless given. Start and End limit the sample; under Advanced, a condition on the rows (Where) can
    leave rows out, and the rows kept are then taken as consecutive periods. The automatic bandwidth is Newey and
    West's (1994), as in arch 9 and R's sandwich.
    """

    topics: ClassVar[tuple[str, ...]] = ("cointegration",)
    schema_version: ClassVar[int] = 2  # 2: bandwidth_rule removed (its "legacy" rule left after 0.1.0a20)

    y: Annotated[str, Widget("series", depends_on="data", help="the dependent series (in levels)")]
    x: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the cointegrated regressors (in levels)")]
    data: Annotated[str, Widget("object", source="Dataset",
                                help="time series in levels, I(1) and cointegrated: test first (Johansen, or Engle-Granger)")] = "data"
    method: Annotated[Literal["fmols", "ccr", "dols"], Widget("choice", help="fully modified OLS, canonical cointegrating regression, or dynamic OLS")] = "fmols"
    trend: Annotated[Literal["c", "ct", "n", "ctt"], Widget("choice", help="c: a constant; ct: and a trend; ctt: and its square; n: neither")] = "c"
    name: str = "coint"
    bandwidth: Annotated[float | None, Widget("number", min=0.1, max=1000, step=1, advanced=True,
                                              help="Bartlett bandwidth b of the long-run covariance, lag weights 1 - j/(b+1) (rounded up "
                                                   "for FMOLS and CCR); empty = automatic (Newey and West)")] = None
    leads: Annotated[int | None, Widget("number", min=0, max=24, advanced=True, help="dynamic OLS: leads of the differences; empty = by BIC")] = None
    lags: Annotated[int | None, Widget("number", min=0, max=24, advanced=True, help="dynamic OLS: lags of the differences; empty = by BIC")] = None
    start: Annotated[str, Widget("period", group="Sample", help="the first period used; empty = the first observed")] = ""
    end: Annotated[str, Widget("period", group="Sample", help="the last period used; empty = the last observed")] = ""
    where: Annotated[str, Widget("text", advanced=True, group="Sample",
                                 help="a condition on the rows, e.g. year >= 1990; the rows kept are taken as consecutive "
                                      "periods; empty = all rows")] = ""

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        extra: dict[str, Any] = {k: getattr(self, k) for k in ("bandwidth", "leads", "lags") if getattr(self, k) is not None}
        if self.start or self.end:
            extra["span"] = _span(self.start, self.end)
        if self.where.strip():
            extra["where"] = self.where.strip()
        return make_call("pymetrix:cointegrating_regression", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                         y=self.y, x=_names(self.x), method=self.method, trend=self.trend, **extra)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@migration("pymetrix:fit_cointegrating_regression", 1)
def _coint_bandwidth_rule(fields: dict[str, Any]) -> dict[str, Any]:  # schema 2: bandwidth_rule removed
    """A step saved by 0.1.0a19 or 0.1.0a20: "newey-west" is what every step now does; "legacy" cannot be reproduced,
    so the step stops with an error that says why rather than giving other numbers."""
    rule = fields.pop("bandwidth_rule", "newey-west")
    if rule != "newey-west":
        raise MigrationError(f"this cointegrating regression was saved with Bandwidth rule = {rule}, pymetrix 0.1.0a18's "
                             "rule, which was removed after 0.1.0a20; set the step again (Newey and West's rule) or "
                             "open the workfile with pymetrix 0.1.0a20 to reproduce its numbers")
    return fields


@command(menu="Test/Johansen cointegration...")
@dataclass(frozen=True)
class Johansen(Command):
    """Johansen cointegration tests: trace and maximum-eigenvalue statistics for each rank.

    Tick the series in levels and choose the lags (in differences) and deterministic terms.

    Read the tests in sequence from r0 = 0: the suggested rank is the first r0 at which the statistic falls below its
    critical value. Rank 0 means no cointegration (model the differences), full rank that the series look
    stationary; anything between is the number of long-run relations a VECM should have. Trace and maximum-
    eigenvalue tests can disagree, and both tend to over-reject in small samples. The critical values depend on the
    deterministic terms and the lags, so settle those first (VECM lag selection).
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import Dataset

        data = Dataset.example("sample_monthly").derive(lip="log(ip)", lemp="log(emp)", lret="log(retail)")
        # trace and maximum-eigenvalue statistics for each r0, their critical values, and the rank at 5%
        print(data.johansen(["lip", "lemp", "lret"], lags=4))
        data.johansen(["lip", "lemp", "lret"], lags=4, det_order=1)   # with a linear trend; -1 for none
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("trace, max-eig", ("statistics for rank r0 (trace: against full rank; max-eig: against r0 + 1), with 10%, 5% "
                            "and 1% critical values")),
        ("eigenvalue", ("the squared canonical correlations, largest first: a large one is a strongly stationary "
                        "combination")),
        ("Rank at 5%", ("in the notes: the rank the sequential trace and maximum-eigenvalue tests suggest (at the "
                        "significance level chosen); they can disagree")),
        ("rank_trace, rank_maxeig", "the two suggested ranks, for use in code, e.g. VECM(..., rank=...)"),
    )

    topics: ClassVar[tuple[str, ...]] = ('cointegration',)

    variables: Annotated[str, Widget("series", depends_on="data", multiple=True)]
    lags: Annotated[int, Widget("number", min=0, max=24, help="lags in first differences")] = 1
    trend: Annotated[Literal["none", "constant", "linear trend"],
                     Widget("", help="deterministic terms: none, a constant, or a linear trend; match how the series trend")] = "constant"
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    name: str = "johansen_result"
    signif: Annotated[Literal[0.05, 0.10, 0.01], Widget("choice", advanced=True,
                                                               help="significance level of the sequential tests that suggest the rank")] = 0.05
    start: Annotated[str, Widget("period", advanced=True)] = ""
    end: Annotated[str, Widget("period", advanced=True)] = ""

    def _det(self) -> int:
        return {"none": -1, "constant": 0, "linear trend": 1}[self.trend]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:dataset.johansen", Ref(self.data), output=self.name, recorded_by=self.command_id,
                         variables=_names(self.variables), lags=self.lags, det_order=self._det(), signif=self.signif,
                         span=_span(self.start, self.end))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Forecast/Forecast...")
@dataclass(frozen=True)
class Forecast(Command):
    """Forecast from a fitted equation, VAR, VECM or exponential smoothing model, with intervals.

    Static forecasts use actual lagged values (one step ahead); dynamic forecasts use their own earlier forecasts.
    VAR and VECM forecasts are always dynamic, and so are smooth-transition autoregressions', simulated as their
    nonlinearity requires (Draws and Seed under Advanced settings).

    A static forecast within the sample is a fit check, not a forecast: it uses data that would not have been known.
    For an honest test of forecasting ability, estimate up to a date and forecast the periods after it dynamically,
    or use Estimate > Rolling or recursive. Static forecasts of a single equation have intervals for shock and
    parameter uncertainty; a dynamic forecast of an equation without ARMA or GARCH errors has none (simulate it with
    Model > Solve model for that), and a least-squares VAR's intervals reflect shock uncertainty only, so they are too
    narrow in small samples.
    """

    maturity: ClassVar[str] = "stable"
    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation

        data = Dataset.example("sample_macro")
        fit = Equation("c ~ y + c[-1]").fit(data, span=("2000Q1", "2022Q4"))    # estimated up to 2022Q4
        static = fit.forecast(data, span=("2023Q1", "2024Q4"))    # one step ahead: the actual c[-1] each period
        dynamic = fit.forecast(data, span=("2023Q1", "2024Q4"), dynamic=True)   # from its own earlier forecasts
        dynamic.to_frame()         # forecast, se, lower, upper, actual; this dynamic forecast has no intervals
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("forecast, se, lower, upper", ("a single equation's forecast, its standard error and the interval at the "
                                        "coverage chosen; empty for a dynamic forecast without ARMA or GARCH errors")),
        ("actual", ("the data, where they exist: compare to judge the forecast; a static forecast in the sample is a "
                    "fit check")),
        ("{var}, {var}_se, {var}_lower, {var}_upper", ("VAR and VECM forecasts: these columns for each variable, always "
                                                       "dynamic")),
        ("Graph", "each forecast with its interval band, and the actual values where there are any"),
    )
    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)

    fit: Annotated[str, Widget("object", source="FittedEquation,FittedVAR,FittedVECM,FittedStateSpace,FittedExponentialSmoothing,FittedSARIMA,FittedTheta,FittedSTLForecast,FittedBMA,FittedThresholdAR,FittedSmoothTransitionAR,FittedLearner")]
    start: Annotated[str, Widget("period", help="the first forecast period; actual data before it supply the lags")]
    end: Annotated[str, Widget("period", help="the last forecast period; uncertainty grows with the horizon")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    dynamic: Annotated[bool, Widget("checkbox", help="single equations: multi-step forecasts (VAR and VECM forecasts are always dynamic)")] = False
    name: str = "forecast"
    coverage: Annotated[float, Widget("number", min=0.5, max=0.999, step=0.05, advanced=True, help="interval coverage, e.g. 0.95")] = 0.95
    draws: Annotated[int | None, Widget("number", min=1, max=100_000, step=100, advanced=True,
                                        help="Bayesian VARs, threshold and smooth-transition models: simulated paths; empty = the model's default")] = None
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True,
                                help="Bayesian VARs, threshold and smooth-transition models: random seed of the simulated paths")] = 0

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"dynamic": True} if self.dynamic else {}
        if self.coverage != 0.95:
            options["alpha"] = round(1 - self.coverage, 6)
        return options | _simulation(self)

    def validate(self, session: Session) -> list[str]:
        return _simulation_problems(self, session, "forecast")

    def to_call(self, session: Session) -> Any:
        from ..calls import method_call
        from ..codec import Ref

        return method_call(type(session[self.fit]), "forecast", Ref(self.fit), output=self.name, recorded_by=self.command_id,
                           data=Ref(self.data), span=(self.start, self.end), **self._options())

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        opts = "".join(f", {k}={v!r}" for k, v in self._options().items())
        return f"{self.name} = {self.fit}.forecast({self.data}, span=({self.start!r}, {self.end!r}){opts})"


@command(menu="Forecast/Conditional forecast...")
@dataclass(frozen=True)
class ConditionalForecast(Command):
    """Forecast from a VAR given conditions on some of its variables: an assumed policy-rate path, or a range such as
    inflation between 3 and 4%.

    Write a condition as r=2.0 (every period), r[2025Q1:2025Q4]=2.0 (some periods), pi[2026Q1]=3..4 (a range),
    pi>=3 or pi[2026Q1:2026Q4]<=4 (one-sided); several, separated by commas. With exact conditions, the other variables
    follow their expected paths given the conditions (Waggoner and Zha 1999): the most likely shocks that meet them,
    so the result does not depend on how shocks are identified. A range is met in every simulated path, drawn from
    the forecast distribution truncated to it. Name the identified shocks that may move to meet the conditions for a
    structural scenario (the others keep their usual distribution); conditions on the shocks themselves (e.g.
    y=0: no demand shocks) are written the same way. For a Bayesian VAR this is done for each posterior draw, so the
    intervals include parameter uncertainty; tick Update posterior to let the conditions inform the parameters too.
    """

    topics: ClassVar[tuple[str, ...]] = ("forecast-workflow",)

    fit: Annotated[str, Widget("object", source="FittedVAR", help="the estimated VAR or Bayesian VAR")]
    start: Annotated[str, Widget("period", help="the first forecast period")]
    end: Annotated[str, Widget("period", help="the last forecast period; conditions may cover all of the horizon or part")]
    conditions: Annotated[str, Widget("text", help="r=2.0 holds r every period; r[2025Q1:2025Q4]=2.0 some periods; pi=3..4 a range; "
                                                   "pi>=3 or pi<=4 one side; several, separated by commas")]
    data: Annotated[str, Widget("object", source="Dataset", help="the data the forecast starts from")] = "data"
    coverage: Annotated[float, Widget("number", min=0.5, max=0.999, step=0.05, advanced=True, help="interval coverage, e.g. 0.95")] = 0.95
    name: str = "conditional"
    draws: Annotated[int | None, Widget("number", min=1, max=100_000, step=100, advanced=True,
                                        help="Bayesian VARs: posterior draws used; VARs with ranges: simulated paths; empty = the default")] = None
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="random seed of the simulated paths")] = 0
    shocks: Annotated[str, Widget("text", advanced=True, group="Structural scenario",
                                  help="the identified shocks that may move to meet the conditions, e.g. r; empty = all")] = ""
    shock_conditions: Annotated[str, Widget("text", advanced=True, group="Structural scenario",
                                            help="conditions on the shocks themselves, written as the conditions, e.g. y=0 or "
                                                 "y[2025Q1:2025Q2]>=0")] = ""
    update_posterior: Annotated[bool, Widget("checkbox", advanced=True,
                                             help="Bayesian VARs: let the conditions inform the parameters too (Waggoner-Zha "
                                                  "Gibbs sampler; slower)")] = False

    def validate(self, session: Session) -> list[str]:
        try:
            paths = _conditions(self.conditions)
            _conditions(self.shock_conditions)
        except (ValueError, SpecificationError) as exc:
            return [f"conditions: {exc}"]
        if not paths:
            return ["give at least one condition, e.g. r[2025Q1:2025Q4]=2.0 or pi=3..4"]
        fit = session.objects.get(self.fit)
        if self.update_posterior and fit is not None and not hasattr(fit, "draws_B"):
            return ["Update posterior is for Bayesian VARs"]
        return _simulation_problems(self, session, "conditional_forecast")

    def _options(self) -> dict[str, Any]:
        out: dict[str, Any] = ({"alpha": round(1 - self.coverage, 6)} if self.coverage != 0.95 else {}) | _simulation(self)
        if _names(self.shocks):
            out["shocks"] = _names(self.shocks)
        if self.shock_conditions.strip():
            out["shock_conditions"] = _conditions(self.shock_conditions)
        if self.update_posterior:
            out["update_posterior"] = True
        return out

    def to_call(self, session: Session) -> Any:
        from ..calls import method_call
        from ..codec import Ref

        return method_call(type(session[self.fit]), "conditional_forecast", Ref(self.fit), output=self.name, recorded_by=self.command_id,
                           data=Ref(self.data), span=(self.start, self.end), conditions=_conditions(self.conditions), **self._options())

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        opts = "".join(f", {k}={v!r}" for k, v in self._options().items())
        return f"{self.name} = {self.fit}.conditional_forecast({self.data}, ({self.start!r}, {self.end!r}), {_conditions(self.conditions)!r}{opts})"


def _conditions(text: str) -> dict[str, Any]:
    """Conditions of a conditional forecast: as ``_paths`` (``name=value``, ``name[2025Q1:2025Q4]=value``), with ranges
    ``name=3..4``, ``name>=3`` and ``name<=4`` written as ``{"lower": 3.0, "upper": 4.0}`` (one side for >= and <=);
    a >= and a <= on the same periods make one range. A variable's conditions are all for every period or all for
    periods named: mixing the two is refused (one would otherwise replace the other)."""
    import re

    import pandas as pd

    out: dict[str, Any] = {}

    def put(name: str, key: str | None, cell: Any) -> None:
        current = out.get(name)
        if current is not None and _is_path(current) == (key is None):
            raise SpecificationError(f"{name} has a condition for every period and others for some periods: write them all "
                                     f"for the periods they cover, e.g. {name}[2025Q1:2025Q4]<=4")
        if key is None:
            out[name] = _merge(current, cell, name) if current is not None else cell
            return
        path = current if current is not None else {}
        path[key] = _merge(path[key], cell, f"{name}[{key}]") if key in path else cell
        out[name] = path

    for part in [p.strip() for p in text.split(",") if p.strip()]:
        m = re.fullmatch(r"([^\W\d]\w*)\s*(?:\[\s*([^\]:]+?)\s*(?::\s*([^\]]+?)\s*)?\])?\s*(>=|<=|=)\s*(.+)", part)
        if not m:
            raise SpecificationError(f"write name=value, name=low..high, name>=value or name<=value (with [2025Q1:2025Q4] for "
                                     f"some periods), not {part!r}")
        name, first, last, op, text_value = m.groups()
        if op == "=" and ".." in text_value:
            lo, _, hi = text_value.partition("..")
            low, high = float(lo), float(hi)
            if low > high:
                raise SpecificationError(f"{part!r}: the lower bound is above the upper one")
            cell: Any = {"lower": low, "upper": high}
        elif op == "=":
            cell = float(text_value)
        else:
            cell = {"lower" if op == ">=" else "upper": float(text_value)}
        if first is None:
            put(name, None, cell)
            continue
        start = pd.Period(first)
        periods = pd.period_range(start, pd.Period(last, freq=start.freq) if last else start, freq=start.freq)
        if not len(periods):
            raise SpecificationError(f"{part!r}: the range is empty")
        for q in periods:
            put(name, str(q), cell)
    return out


def _is_path(value: Any) -> bool:
    return isinstance(value, dict) and not set(value) <= {"lower", "upper"}


def _merge(old: Any, new: Any, where: str) -> Any:
    """Two conditions on the same cell: a lower and an upper bound make a range; anything else is a clash."""
    if isinstance(old, dict) and isinstance(new, dict) and not set(old) & set(new):
        merged = {**old, **new}
        if merged["lower"] > merged["upper"]:
            raise SpecificationError(f"{where}: the lower bound is above the upper one")
        return {"lower": merged["lower"], "upper": merged["upper"]}
    raise SpecificationError(f"{where} has two conditions that do not make one range")


def _simulation(cmd: Any) -> dict[str, Any]:
    """A forecast's draws and seed, when set."""
    return {k: getattr(cmd, k) for k, d in (("draws", None), ("seed", 0)) if getattr(cmd, k) != d}


def _simulation_problems(cmd: Any, session: Session, method: str) -> list[str]:
    """Draws or a seed for a model whose forecasts are not simulated."""
    import inspect

    fit = session.objects.get(cmd.fit)
    if fit is None or not hasattr(fit, method):
        return []
    taken = inspect.signature(getattr(fit, method)).parameters
    return [f"{type(fit).__name__} forecasts take no {k} (only Bayesian VARs and threshold models simulate)"
            for k in _simulation(cmd) if k not in taken]


@command(menu="Model/Solve model...")
@dataclass(frozen=True)
class SolveModel(Command):
    """Solve a model of fitted equations and identities over a period range.

    Choose the equations and the solution period; values before the period come from the data. Choose the solver
    and, if needed, its settings. In the scenario, targets hold endogenous variables, each with one free shock
    (exogenize and endogenize). With the first-order solver, shocks are known in advance unless listed as
    unanticipated. With Partial (advanced), a hold applies only in the periods it gives a value: elsewhere the
    variable's equation determines it as usual and its free variable keeps its data.

    The result's Convergence view shows, period by period, the solver's iterations and the largest residual left and
    its equation (Newton, Gauss-Seidel, perfect foresight). Before solving, Model.missing_inputs (or Model.check with
    data and a span) lists the inputs each equation lacks in each period. The solver's settings choose its convergence
    test (the residual, or the scaled step, which does not depend on the equations' units), Newton's line search and
    how often it evaluates the Jacobian; under Advanced, Fallback retries a period the solver cannot solve, and
    Ordering: blocks solves recursive equations once and iterates only on the simultaneous blocks.

    Solved over a historical period with no scenario, the solution is a dynamic simulation: compare it with the data
    to see how far the model drifts when its errors are left out. A policy experiment is best read as the difference
    between a scenario and a baseline solved over the same period, which removes the model's own tracking error.
    With draws, the stochastic simulation adds the equations' errors (and, with Coefficients, parameter
    uncertainty), so its quantiles show the uncertainty around the path; a model with expectations needs the
    first-order solver for that.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, Equation, Model, Scenario

        data = Dataset.example("sample_macro")
        cons = Equation("c ~ y + c[-1]").fit(data)
        inv = Equation("i ~ y[-1] + i[-1]").fit(data)
        model = Model([cons, inv, Equation("y = c + i + g")])   # two fitted equations and an identity
        base = model.solve(data, ("2020Q1", "2024Q4"))   # a dynamic simulation (Newton); the lags come from the data
        more_g = model.solve(data, ("2020Q1", "2024Q4"), scenario=Scenario(shift={"g": 5}))   # g 5 higher
        more_g["y"] - base["y"]                          # the effect on y, by period (zero before 2020Q1)
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Solution", ("the endogenous variables over the solved periods; the title gives the scenario and the solver's "
                      "iterations")),
        ("solution['y']", ("the whole history with the solved periods filled in; subtract a baseline solved over the "
                           "same periods to read a scenario's effect")),
        ("Fan chart, Quantiles", ("with Draws: the median and bands of the stochastic simulation, and columns "
                                  "{variable}_q5 to {variable}_q95")),
        ("Data, Graph", "every series of the dataset after solving, and their charts"),
        ("converged, iterations", "whether the solver converged and in how many iterations, for use in code"),
        ("Convergence", "by period: iterations, the largest absolute residual at the end and its equation, and whether it "
                        "converged (solution.convergence)"),
    )

    topics: ClassVar[tuple[str, ...]] = ('model-solution',)
    schema_version: ClassVar[int] = 2  # 2: target became hold

    equations: Annotated[str, Widget("object", source="FittedEquation,Equation,Model,EViewsXmlModel", multiple=True,
                                     help="fitted equations and identities, or a model read (FRB/US XML)")]
    start: Annotated[str, Widget("period", help="the first period solved; earlier values, the lags, come from the data",
                                 empty="the sample's first period")]
    end: Annotated[str, Widget("period", help="the last period solved; exogenous series need values up to it",
                               empty="the sample's last period")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    solver: Annotated[str, Widget("extension", source="solvers", help="Newton solves all equations jointly; Gauss-Seidel one at a time")] = "Newton"
    solver_settings: Annotated[str, Widget("options", depends_on="solver", source="solvers", group="Solver settings")] = ""
    ordering: Annotated[Literal["joint", "blocks"], Widget("choice", advanced=True, group="Solver settings",
                                                           help="joint: the solver takes all the equations each period; blocks: "
                                                                "recursive equations solved once, in order, the solver only on "
                                                                "the simultaneous blocks (Newton, Gauss-Seidel)")] = "joint"
    fallback: Annotated[bool, Widget("checkbox", advanced=True, group="Solver settings",
                                     help="a period the solver cannot solve is tried again with damped Gauss-Seidel (damping "
                                          "0.5, 400 iterations), then Newton with a line search (Newton, Gauss-Seidel)")] = False
    params: Annotated[str, Widget("text", help="values for the parameters of equations not estimated, e.g. b=0.99, k=0.1")] = ""
    shift: Annotated[str, Widget("text", group="Scenario", help="add to exogenous series, e.g. g=5 (every period) or g[2025Q1:2025Q4]=5")] = ""
    set_to: Annotated[str, Widget("text", group="Scenario", help="replace exogenous series, e.g. rate=2.5 or ey[2025Q1]=-1")] = ""
    hold: Annotated[str, Widget("text", group="Scenario", help="hold endogenous variables (exogenize), e.g. i[2025Q1:2025Q4]=0.5")] = ""
    free: Annotated[str, Widget("text", group="Scenario",
                                help="one shock or exogenous variable per held variable, in order, solved to hold it (endogenize); "
                                     "more free shocks than holds: a least-norm plan (first-order solver)")] = ""
    unanticipated: Annotated[str, Widget("text", advanced=True, group="Scenario", help="shocks that are surprises (first-order solver); others are known in advance")] = ""
    hold_sd: Annotated[str, Widget("text", advanced=True, group="Scenario",
                                   help="soft holds (first-order solver): a standard deviation, e.g. y[2025Q4]=0.25 or y=0.25; "
                                        "empty or 0 = exact")] = ""
    free_sd: Annotated[str, Widget("text", advanced=True, group="Scenario",
                                   help="least-norm plan (first-order solver): the free shocks' sizes, e.g. ei=0.2, ey=0.5; "
                                        "the smallest shocks in these units that meet the holds")] = ""
    name: str = "solution"
    draws: Annotated[int, Widget("number", min=0, max=100_000, step=100, advanced=True, group="Stochastic simulation",
                                 help="draws of a stochastic simulation (the fan, quantiles); 0 = none")] = 0
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, group="Stochastic simulation",
                                help="random seed of the simulation; the same seed gives the same quantiles")] = 0
    errors: Annotated[Literal["normal", "bootstrap", "sample"], Widget("choice", advanced=True, group="Stochastic simulation",
                                                                        help="fitted equations' errors: normal with their standard errors and "
                                                                             "correlations, or their residuals resampled; sample: a sample "
                                                                             "of residuals given (below) resampled, a period at a time")] = "normal"
    shock_sd: Annotated[str, Widget("text", advanced=True, group="Stochastic simulation",
                                    help="standard deviations: by equation (e.g. cons=0.5), or by shock with the first-order solver "
                                         "(e.g. ey=0.5, ei=0.2)")] = ""
    coefficients: Annotated[bool, Widget("checkbox", advanced=True, group="Stochastic simulation",
                                         help="draw the fitted equations' coefficients too (parameter uncertainty)")] = False
    over: Annotated[str, Widget("text", advanced=True, group="Panel",
                                help="a panel's entity axis (country...): the model solved for each entity on its own series")] = ""
    links: Annotated[str, Widget("text", advanced=True, group="Panel",
                                 help="variables each entity takes from the others, solved jointly: ys = trade @ y (trade: a "
                                      "matrix of weights, a row per entity; one row for an aggregate)")] = ""
    regimes: Annotated[str, Widget("text", advanced=True, group="Regimes",
                                   help="first-order solver or perfect foresight: parameters that change at known dates, known in advance, e.g. "
                                        "2025Q1..2026Q4: rho=1, phi_pi=0 (a while) or 2027Q1: phi_pi=2 (for good); "
                                        "2027Q1 surprise: phi_pi=2 is unknown until it starts")] = ""
    hold_lower: Annotated[str, Widget("text", advanced=True, group="Scenario",
                                      help="ranges (first-order solver): keep a variable at or above, e.g. pi[2025Q1:2025Q4]=1.5; "
                                           "the free shocks move only as far as needed")] = ""
    hold_upper: Annotated[str, Widget("text", advanced=True, group="Scenario",
                                      help="ranges (first-order solver): keep a variable at or below, e.g. pi=3")] = ""
    # -- model dialogs batch: residual samples and add factors (Model > Add factors) ---------------------------------
    residuals: Annotated[str, Widget("object", source="Dataset,dict", advanced=True, group="Stochastic simulation",
                                     empty="empty: none", show_if=("errors", ("sample",)),
                                     help="errors: sample: residuals by equation, a series named after each equation shocked "
                                          "(a dataset), or add factors over a historical span (Model > Add factors)")] = ""
    residual_series: Annotated[str, Widget("series", depends_on="residuals", multiple=True, advanced=True, group="Stochastic simulation",
                                           show_if=("errors", ("sample",)),
                                           help="the dataset's series to resample, each named after an equation; empty = all")] = ""
    residual_start: Annotated[str, Widget("period", advanced=True, group="Stochastic simulation", show_if=("errors", ("sample",)),
                                          help="the dataset's periods to resample from: first (empty = its active sample's)")] = ""
    residual_end: Annotated[str, Widget("period", advanced=True, group="Stochastic simulation", show_if=("errors", ("sample",)),
                                        help="the dataset's periods to resample from: last")] = ""
    error_scale: Annotated[str, Widget("text", advanced=True, group="Stochastic simulation",
                                       help="multiply the errors drawn in some periods, e.g. 2026Q2=0.5 (a backward-looking model)")] = ""
    workers: Annotated[int, Widget("number", min=0, max=256, advanced=True, group="Stochastic simulation",
                                   help="processes solving the draws (a backward-looking model); 0 = this one; the draws are "
                                        "the same for any number")] = 0
    adjust: Annotated[str, Widget("object", source="dict", advanced=True, group="Scenario", empty="empty: none",
                                  help="add factors (Model > Add factors) added to the equations, so the model tracks the data")] = ""
    partial: Annotated[bool, Widget("checkbox", advanced=True, group="Scenario",
                                    help="hold only in the periods given a value; elsewhere the variable's equation determines "
                                         "it and its free variable keeps its data (Newton, Gauss-Seidel)")] = False

    def _errors(self) -> Any:
        """The residual sample resampled (errors: sample): the dataset's series chosen, over the periods chosen, or
        the add factors given."""
        from ..calls import make_call
        from ..codec import Ref

        sample: Any = Ref(self.residuals)
        if self.residual_series.strip():
            sample = make_call("pymetrix:dataset.keep", sample, names=tuple(_names(self.residual_series)))
        if self.residual_start or self.residual_end:
            sample = make_call("pymetrix:dataset.over", sample, sample=(self.residual_start or None, self.residual_end or None))
        return sample

    def _error_scale(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for part in _names(self.error_scale):
            period, sep, value = part.partition("=")
            try:
                out[period.strip()] = float(value)
            except ValueError:
                raise SpecificationError(f"write each as period=multiplier, e.g. 2026Q2=0.5, not {part!r}") from None
            if not sep or not period.strip():
                raise SpecificationError(f"write each as period=multiplier, e.g. 2026Q2=0.5, not {part!r}")
        return out
    # -- end of the model dialogs batch's fields ---------------------------------------------------------------------

    def _stochastic(self) -> dict[str, Any]:
        if not self.draws:
            return {}
        out: dict[str, Any] = {"draws": self.draws, "seed": self.seed}
        if self.errors == "sample":
            out["errors"] = self._errors()
        elif self.errors != "normal":
            out["errors"] = self.errors
        if self.error_scale.strip():
            out["error_scale"] = self._error_scale()
        if self.workers:
            out["workers"] = self.workers
        if self.shock_sd.strip():
            out["shock_sd"] = _calibration(self.shock_sd)
        if self.coefficients:
            out["coefficients"] = True
        return out

    def _solver(self) -> Any:
        solver = build(registry("solvers").get(self.solver), self.solver_settings)
        if not self.fallback:
            return solver
        from ..models.solvers import Fallback, GaussSeidel, Newton

        return Fallback(solver, GaussSeidel(damping=0.5, max_iter=400), Newton(line_search=True))

    def _period_by_period(self) -> bool:
        from ..models.solvers import FirstOrder, PerfectForesight

        return not isinstance(build(registry("solvers").get(self.solver), self.solver_settings), (FirstOrder, PerfectForesight))

    def _scenario(self) -> Any:
        import dataclasses

        from ..codec import Fields, Ref
        from ..models import Scenario

        if not (self.shift or self.set_to or self.hold or self.free or self.unanticipated or self.hold_sd or self.free_sd
                or self.hold_lower or self.hold_upper):
            scenario = None
        else:
            scenario = Scenario(set=_paths(self.set_to), shift=_paths(self.shift), hold=_paths(self.hold),
                                free=tuple(_names(self.free)), unanticipated=tuple(_names(self.unanticipated)),
                                hold_sd=_paths(self.hold_sd), free_sd=_calibration(self.free_sd) if self.free_sd.strip() else {},
                                hold_lower=_paths(self.hold_lower), hold_upper=_paths(self.hold_upper),
                                **({"partial": True} if self.partial and self.hold else {}))
        if not self.adjust:
            return scenario
        given = {f.name: getattr(scenario, f.name) for f in dataclasses.fields(Scenario) if f.init} if scenario is not None else {}
        return Fields(Scenario, {**given, "adjust": Ref(self.adjust)})  # the add factors, by name

    def validate(self, session: Session) -> list[str]:
        try:
            self._scenario()
        except ValueError as exc:
            return [f"scenario: {exc}"]
        try:
            self._stochastic()
        except (ValueError, SpecificationError) as exc:
            return [f"stochastic simulation: {exc}"]
        if self.errors == "sample" and self.draws:
            if not self.residuals:
                return ["errors: sample: choose the residuals to resample"]
            if self.residuals not in session.objects:
                return [f"no residuals named {self.residuals!r}"]
            if isinstance(session[self.residuals], dict) and (self.residual_series.strip() or self.residual_start or self.residual_end):
                return ["the series and periods to resample are a dataset's: add factors are resampled as they are"]
        if self.adjust and self.adjust not in session.objects:
            return [f"no add factors named {self.adjust!r}"]
        if (self.fallback or self.ordering != "joint") and not self._period_by_period():
            return ["fallback and ordering by blocks are for the period-by-period solvers (Newton, Gauss-Seidel)"]
        try:
            links = _links(self.links)
        except SpecificationError as exc:
            return [f"links: {exc}"]
        if links and not self.over.strip():
            return ["links: give the panel's entity axis (over) too"]
        if self.regimes.strip():
            from ..models.solvers import FirstOrder, PerfectForesight

            if not isinstance(self._solver(), (FirstOrder, PerfectForesight)):
                return ["regimes: choose the first-order solver or perfect foresight"]
            problems = _regime_problems(self.regimes)
            if problems:
                return problems
        missing = [link.weights.name for link in links if link.weights.name not in session.objects]
        return [f"links: no {name} in the session (a matrix of weights)" for name in missing]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:model.solve", _model_fields(self.equations, self.params, self.over, self.links), output=self.name,
                         recorded_by=self.command_id, data=Ref(self.data), span=(self.start, self.end), scenario=self._scenario(),
                         solver=self._solver(), **self._stochastic(), **({"regimes": self.regimes.strip()} if self.regimes.strip() else {}),
                         **({"ordering": self.ordering} if self.ordering != "joint" else {}))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@migration("pymetrix:solve_model", 1)
def _solve_model_hold(fields: dict[str, Any]) -> dict[str, Any]:  # schema 2: target became hold
    if "target" in fields:
        fields["hold"] = fields.pop("target")
    return fields


@command(menu="Model/Forecast from estimated model...", id="pymetrix:forecast_model")
@dataclass(frozen=True)
class ForecastModel(Command):
    """Forecast from an estimated model (Model > Estimate model): its variables start at their Kalman-smoothed values at
    the end of the data and follow its solution, and the observed series come through the measurement equations.

    Shock paths (anticipated unless listed as unanticipated) and targets held by free shocks make scenarios. With
    draws, a stochastic simulation too: the estimated shocks drawn in every period, for fan charts and quantiles.
    Choose the posterior (Model > Metropolis-Hastings) instead of the estimated model for parameter uncertainty as
    well: the model solved and filtered again at each posterior draw. Newer data are filtered with the estimated model
    first.
    """

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models", "forecast-workflow")

    source: Annotated[str, Widget("object", source="FittedModel,Posterior",
                                  help="an estimated model, or its posterior (Metropolis-Hastings) for parameter uncertainty too")]
    start: Annotated[str, Widget("period", help="the first forecast period")]
    end: Annotated[str, Widget("period", help="the last forecast period; far ahead, a stationary model returns to its steady state")]
    data: Annotated[str, Widget("object", source="Dataset", empty="empty: the data the model was estimated on", advanced=True,
                                help="newer data, filtered with the estimated model; empty = the estimation's data")] = ""
    shift: Annotated[str, Widget("text", group="Scenario", help="add to shocks, e.g. ei[2025Q1]=0.25")] = ""
    set_to: Annotated[str, Widget("text", group="Scenario", help="set shocks, e.g. ey[2025Q1:2025Q2]=-1")] = ""
    hold: Annotated[str, Widget("text", group="Scenario", help="hold model variables (exogenize), e.g. i[2025Q1:2025Q4]=0.5")] = ""
    free: Annotated[str, Widget("text", group="Scenario", help="one shock per held variable, in order, solved to hold it (endogenize); "
                                                                "more free shocks than holds: a least-norm plan")] = ""
    unanticipated: Annotated[str, Widget("text", advanced=True, group="Scenario", help="shocks that are surprises; others are known in advance")] = ""
    hold_sd: Annotated[str, Widget("text", advanced=True, group="Scenario",
                                   help="soft holds: a standard deviation, e.g. y[2025Q4]=0.25 or y=0.25; empty or 0 = exact")] = ""
    free_sd: Annotated[str, Widget("text", advanced=True, group="Scenario",
                                   help="least-norm plan: the free shocks' sizes, e.g. ei=0.2, ey=0.5 (the estimated sizes are a "
                                        "natural choice); the smallest shocks in these units that meet the holds")] = ""
    draws: Annotated[int, Widget("number", advanced=True, min=0, max=100_000, step=100, group="Stochastic simulation",
                                 help="draws of the estimated shocks, for fan charts and quantiles; 0 = none")] = 0
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, group="Stochastic simulation",
                                help="random seed of the draws; the same seed gives the same fan")] = 0
    name: str = "model_forecast"
    shocks: Annotated[bool, Widget("checkbox", advanced=True, group="Stochastic simulation",
                                   help="from a posterior: draw shocks too (untick for parameter uncertainty alone)")] = True
    occbin: Annotated[bool | None, Widget("choice", advanced=True, group="Scenario",
                                          help="respect the model's occasionally binding constraints (its max or min) with OccBin; "
                                               "empty = when it was estimated with the OccBin filter")] = None
    regimes: Annotated[str, Widget("text", advanced=True, group="Regimes",
                                   help="parameters that change at known dates in the forecast, known in advance, after the model's "
                                        "own regimes: e.g. 2025Q1..2026Q4: rho=1, phi_pi=0 (a while), 2027Q1: phi_pi=2 (for good); "
                                        "2027Q1 surprise: phi_pi=2 is unknown until it starts")] = ""
    hold_lower: Annotated[str, Widget("text", advanced=True, group="Scenario",
                                      help="ranges keep a variable at or above, e.g. pi[2025Q1:2025Q4]=1.5; "
                                           "the free shocks move only as far as needed")] = ""
    hold_upper: Annotated[str, Widget("text", advanced=True, group="Scenario",
                                      help="ranges keep a variable at or below, e.g. pi=3")] = ""

    schema_version: ClassVar[int] = 2  # 2: target became hold

    def _scenario(self) -> Any:
        from ..models import Scenario

        if not (self.shift or self.set_to or self.hold or self.free or self.unanticipated or self.hold_sd or self.free_sd
                or self.hold_lower or self.hold_upper):
            return None
        return Scenario(set=_paths(self.set_to), shift=_paths(self.shift), hold=_paths(self.hold),
                        free=tuple(_names(self.free)), unanticipated=tuple(_names(self.unanticipated)),
                        hold_sd=_paths(self.hold_sd), free_sd=_calibration(self.free_sd) if self.free_sd.strip() else {},
                        hold_lower=_paths(self.hold_lower), hold_upper=_paths(self.hold_upper))

    def validate(self, session: Session) -> list[str]:
        from ..models.bayes import Posterior

        if self.data and self.data not in session.objects:
            return [f"no dataset named {self.data!r}"]
        try:
            self._scenario()
        except ValueError as exc:
            return [f"scenario: {exc}"]
        if self.source not in session.objects:
            return [f"no estimated model named {self.source!r}"]
        problems = _regime_problems(self.regimes)
        if problems:
            return problems
        if isinstance(session.objects[self.source], Posterior):
            return [] if self.draws >= 2 else ["give the number of posterior draws to forecast at (draws), e.g. 500"]
        return [] if self.shocks else ["untick shocks only for a posterior: choose one (Model > Metropolis-Hastings)"]

    def to_call(self, session: Session) -> Any:
        from ..calls import method_call
        from ..codec import Ref

        extra: dict[str, Any] = {"scenario": self._scenario()} if self._scenario() is not None else {}
        if self.data:
            extra["data"] = Ref(self.data)
        if self.draws:
            extra.update(draws=self.draws, seed=self.seed)
        if not self.shocks:
            extra["shocks"] = False
        if self.occbin is not None:
            extra["occbin"] = self.occbin
        if self.regimes.strip():
            extra["regimes"] = self.regimes.strip()
        return method_call(type(session[self.source]), "forecast", Ref(self.source), output=self.name, recorded_by=self.command_id,
                           span=(self.start, self.end), **extra)

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        args = [repr((self.start, self.end))] + ([f"data={self.data}"] if self.data else [])
        scenario = self._scenario()
        args += [f"scenario={_code(scenario)}"] if scenario is not None else []
        args += [f"draws={self.draws}"] if self.draws else []
        args += [] if self.shocks else ["shocks=False"]  # in the order of the call's arguments, as recorded
        args += [f"seed={self.seed}"] if self.draws else []
        args += [f"occbin={self.occbin!r}"] if self.occbin is not None else []
        args += [f"regimes={self.regimes.strip()!r}"] if self.regimes.strip() else []
        return f"{self.name} = {self.source}.forecast({', '.join(args)})"


@migration("pymetrix:forecast_model", 1)
def _forecast_model_hold(fields: dict[str, Any]) -> dict[str, Any]:  # schema 2: target became hold
    if "target" in fields:
        fields["hold"] = fields.pop("target")
    return fields


@command(menu="Model/Calibrate model...")
@dataclass(frozen=True)
class Calibrate(Command):
    """Choose parameters so that the model's steady state hits targets: levels, ratios or spreads of its variables.

    Give one target for each free parameter, e.g. targets "k/y = 10; c/y = 0.75" with free parameters "beta, delta".
    Tick growth for a trending model (the targets then apply on its balanced growth path). The result holds the model
    with the parameters found, its steady state, and each target with the value reached.

    Calibration makes the model match the chosen long-run facts exactly; it does not test them, and it says nothing
    about the model's dynamics. Pick targets that each free parameter clearly drives (the capital-output ratio for
    the depreciation rate, say): targets that depend on the same combination of parameters cannot be met
    separately, and the search fails. Only calibrated parameters can be free; estimated coefficients stay as they
    are.
    """

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models", "model-solution")

    equations: Annotated[str, Widget("object", source="FittedEquation,Equation", multiple=True,
                                     help="the model's equations and identities; their calibrated parameters can be free")]
    targets: Annotated[str, Widget("text", help="expressions of the variables and their values, separated by ; e.g. k/y = 10; c/y = 0.75")]
    free: Annotated[str, Widget("text", help="one parameter for each target, e.g. beta, delta")]
    params: Annotated[str, Widget("text", help="values of the model's parameters, e.g. alpha=0.33, beta=0.99, delta=0.025")] = ""
    growth: Annotated[bool, Widget("checkbox", help="a trending model: targets on its balanced growth path")] = False
    name: str = "calibration"

    def _targets(self) -> dict[str, float]:
        out = {}
        for part in [p.strip() for p in self.targets.replace("\n", ";").split(";") if p.strip()]:
            expr, eq, value = part.rpartition("=")
            if not eq or not expr.strip():
                raise SpecificationError(f"write each target as expression = value, not {part!r}")
            out[expr.strip()] = float(value)
        return out

    def validate(self, session: Session) -> list[str]:
        try:
            targets = self._targets()
        except ValueError as exc:
            return [str(exc)]
        if len(targets) != len(_names(self.free)):
            return [f"{len(targets)} targets but {len(_names(self.free))} free parameters: give one parameter for each target"]
        return []

    def to_call(self) -> Any:
        from ..calls import make_call

        return make_call("pymetrix:model.calibrate", _model_fields(self.equations, self.params), output=self.name, recorded_by=self.command_id,
                         targets=self._targets(), free=_names(self.free), growth=self.growth)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Model/Check model...")
@dataclass(frozen=True)
class CheckModel(Command):
    """Check a model before solving it: every equation contains the variable it determines, the simultaneous blocks,
    the parameters, the timing of lags, leads and shocks, and, given a period, whether the data supply the starting
    values and exogenous paths. For a forward-looking model, also the steady state and the Blanchard-Kahn conditions.
    The model can be a whole one however it was made (imported from an IRIS, Dynare or EViews file, or built): an
    imported model is checked with its own shocks and steady-state guesses, unless Shocks are given.

    Run it before solving: most failures to solve come from an equation that does not determine its variable, a
    parameter with no value, a missing exogenous path or a shock listed with the wrong timing. The Blanchard-Kahn
    conditions compare the stable roots with the predetermined variables: too few stable roots means no stable
    solution (often a sign error or a policy rule that responds too strongly), too many means the solution is not
    unique (indeterminacy, e.g. a Taylor rule responding less than one for one to inflation).
    """

    topics: ClassVar[tuple[str, ...]] = ('model-solution',)

    equations: Annotated[str, Widget("object", source="FittedEquation,Equation,Model,Template,DynareModel,EViewsXmlModel",
                                     multiple=True, help="fitted equations and identities, or a model, however it was made "
                                                         "(imported from IRIS, Dynare, EViews or built)")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    start: Annotated[str, Widget("period", group="Solution period", help="optional: check the data cover it")] = ""
    end: Annotated[str, Widget("period", group="Solution period", help="with Start, the solution period the data must cover")] = ""
    shocks: Annotated[str, Widget("text", help="random exogenous variables of a forward-looking model, e.g. ey, epi")] = ""
    params: Annotated[str, Widget("text", help="values for the parameters of equations not estimated, e.g. b=0.99, k=0.1")] = ""
    name: str = "model_check"
    steady: Annotated[bool | None, Widget("choice", advanced=True,
                                          help="check the steady state too: automatic (when the model has one), yes or no")] = None

    def _span(self) -> tuple[str, str] | None:
        return (self.start, self.end) if self.start and self.end else None

    def to_call(self, session: Session) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        data = Ref(self.data) if self.data in session.objects else None  # the equations alone can be checked
        shocks, guess = _imported_shocks(self.equations, session)
        return make_call("pymetrix:model.check", _model_fields(self.equations, self.params), output=self.name, recorded_by=self.command_id,
                         data=data, span=self._span(), shocks=_names(self.shocks) or shocks,
                         **({"steady": self.steady} if self.steady is not None else {}), **({"guess": guess} if guess else {}))

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        """The code without a session: as if the chosen dataset were there (none chosen, ``data=""``: no data). The
        dialog and the recorded step build it from the session instead, leaving out a dataset it does not have."""
        from .session import Session

        preview = Session()
        if self.data:
            preview.objects[self.data] = None  # as if the dataset were there
        return self.to_call(preview).to_code()


def _imported_shocks(equations: str, session: Any) -> tuple[tuple[str, ...], dict[str, float]]:
    """The shocks and steady-state guesses of the imported models among ``equations`` (an IRIS or EViews template, a
    Dynare model), as their own ``check`` uses them, so checking one in a dialog checks what the API does."""
    objects = getattr(session, "objects", None) or {}
    shocks: list[str] = []
    guess: dict[str, float] = {}
    for n in _names(equations):
        obj = objects.get(n)
        if type(obj).__name__ not in ("Template", "DynareModel"):
            continue
        own = obj.shocks if type(obj).__name__ == "Template" else obj.exogenous
        shocks += [s for s in own if s not in shocks]
        guess.update({k: float(v) for k, v in dict(obj.guess or {}).items()})
    return tuple(shocks), guess


def _paths(text: str) -> dict[str, Any]:
    """Scenario values: ``name=value`` for every period, ``name[2025Q1:2025Q4]=value`` for a range, ``name[2026Q1]=value``
    for one period; several, separated by commas."""
    import re

    import pandas as pd

    out: dict[str, Any] = {}
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        m = re.fullmatch(r"([^\W\d]\w*)\s*(?:\[\s*([^\]:]+?)\s*(?::\s*([^\]]+?)\s*)?\])?\s*=\s*(.+)", part)
        if not m:
            raise SpecificationError(f"write name=value or name[2025Q1:2025Q4]=value, not {part!r}")
        name, first, last, value = m.group(1), m.group(2), m.group(3), float(m.group(4))
        if first is None:
            out[name] = value
            continue
        start = pd.Period(first)
        periods = pd.period_range(start, pd.Period(last, freq=start.freq) if last else start, freq=start.freq)
        if not len(periods):
            raise SpecificationError(f"{part!r}: the range is empty")
        path = out.get(name) if isinstance(out.get(name), dict) else {}
        path.update({str(q): value for q in periods})
        out[name] = path
    return out


def _calibration(text: str) -> dict[str, float]:
    args, kw = parse_options(text)
    if args:
        raise SpecificationError(f"give parameter values as name=value, not {args[0]!r}")
    return {k: float(v) for k, v in kw.items()}


def _model_fields(equations: str, params: str, over: str = "", links: str = "") -> Any:
    """The model a dialog solves, checks or calibrates, naming its equations: ``Model([fc, fi], params=...)``; with
    ``over``, solved for each entity of a panel, and with ``links``, jointly."""
    from ..codec import Fields, Ref

    fields: dict[str, Any] = {"equations": tuple(Ref(n) for n in _names(equations)), "params": _calibration(params) or None}
    if over.strip():
        fields["over"] = over.strip()
    if links.strip():
        fields["links"] = _links(links)
    return Fields(Model, fields)


def _assignments(text: str) -> dict[str, str]:
    """``growth = 400*dlog(gdp); infl = 400*dlog(cpi)``: names and expressions, separated by semicolons."""
    out = {}
    for part in filter(None, (p.strip() for p in text.split(";"))):
        name, sep, expr = part.partition("=")
        if not sep or not name.strip().isidentifier() or not expr.strip():
            raise SpecificationError(f"{part!r}: write each as name = expression, separated by semicolons")
        out[name.strip()] = expr.strip()
    return out


def _links(text: str) -> tuple[Any, ...]:
    """``ys = trade @ y, world = shares @ y``: links made from a variable by a matrix of weights named in the session."""
    from ..codec import Ref
    from ..models import Link

    out = []
    for part in filter(None, (p.strip() for p in re.split(r"[;,]", text))):
        m = re.fullmatch(r"([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*@\s*([A-Za-z_]\w*)", part)
        if m is None:
            raise SpecificationError(f"{part!r}: write each link as name = weights @ variable, e.g. ys = trade @ y")
        out.append(Link(m.group(1), m.group(3), Ref(m.group(2))))
    return tuple(out)


@command(menu="Model/Import Dynare file...")
@dataclass(frozen=True)
class ImportDynare(Command):
    """Read a Dynare model file (.mod): its variables, parameters and values, model equations, steady-state guesses
    and shock standard deviations, as a PyMetrix model ready to solve.

    Commands in the file (steady, stoch_simul...) are listed, not run. Use Rational expectations to solve the model,
    or the imported object's solve() in code.
    """

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models",)

    path: Annotated[str, Widget("file", source="Dynare files (*.mod);;All files (*)", help="the Dynare .mod file; its commands (steady, stoch_simul...) are listed, not run")]
    name: str = "dynare"

    def to_call(self) -> Any:
        from ..calls import make_call

        return make_call("pymetrix:read_mod", output=self.name, recorded_by=self.command_id, source=self.path)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Model/Import IRIS file...")
@dataclass(frozen=True)
class ImportIris(Command):
    """Read an IRIS model file (.model): transition and measurement variables, shocks, parameters and their
    values, transition and measurement equations, log variables and !for loops, as a template model.

    Parameter values missing from the file (IRIS usually sets them in code) are given here. Shock sizes come from
    the std_ parameters. Measurement shocks become measurement errors. Solve, check and take the result to data like
    any template: Model > Analysis > State-space form, Model > Analysis > Kalman filter.
    """

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models",)

    path: Annotated[str, Widget("file", source="IRIS model files (*.model);;All files (*)", help="the IRIS .model file; values IRIS sets in code go under Parameters")]
    params: Annotated[str, Widget("text", help="parameter values, e.g. rho=0.8, std_eps_y=0.5")] = ""
    name: str = "iris"
    trend_base: Annotated[str, Widget("text", advanced=True, help="the base period of !dtrends' ttrend, e.g. 2000Q1; empty = IRIS's")] = ""

    def to_call(self) -> Any:
        from ..calls import make_call

        return make_call("pymetrix:read_iris", output=self.name, recorded_by=self.command_id, source=self.path,
                         params=_calibration(self.params) or None, **({"trend_base": self.trend_base} if self.trend_base.strip() else {}))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Model/Rational expectations...")
@dataclass(frozen=True)
class SolveRationalExpectations(Command):
    """Solve a forward-looking model under rational expectations, to first order.

    The model is linearised around its steady state and solved for its unique stable solution, the decision
    rules: each variable on the lagged variables and the current shocks. Shocks are exogenous variables of the
    model (zero in the steady state, usually). The Blanchard-Kahn conditions are checked: the dialog reports
    when there is no stable solution or more than one. The result gives impulse responses, stochastic
    simulations, and the Kalman filter for estimating unobserved variables from data.
    """

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models",)

    equations: Annotated[str, Widget("object", source="FittedEquation,Equation,Template,DynareModel", multiple=True,
                                     help="fitted or calibrated equations and identities, or a template or Dynare model "
                                          "(solved with its own shocks and parameters)")]
    shocks: Annotated[str, Widget("text", help="exogenous variables that are shocks, e.g. ey, epi, ei (equations only: a "
                                               "template or Dynare model has its own)")] = ""
    params: Annotated[str, Widget("text", help="values for the parameters of equations not estimated, e.g. b=0.99, k=0.1")] = ""
    name: str = "re"
    steady_state: Annotated[str, Widget("text", advanced=True, help="values of other exogenous variables in the steady state, e.g. pistar=2")] = ""
    guess: Annotated[str, Widget("text", advanced=True, help="starting values for finding the steady state, e.g. y=1, c=0.8")] = ""

    def _whole_model(self, session: Session | None = None) -> tuple[str, str] | None:
        """(name, "template" or "dynare_model") when a template or Dynare model is chosen, as ``to_call`` needs."""
        objects = getattr(session, "objects", None) or {}
        for n in _names(self.equations):
            kind = type(objects.get(n)).__name__ if n in objects else ""
            if kind in ("Template", "DynareModel"):
                return n, "template" if kind == "Template" else "dynare_model"
        return None

    def validate(self, session: Session) -> list[str]:
        missing = [n for n in _names(self.equations) if n not in session.objects]
        if missing:
            return [f"no equation named {n!r}" for n in missing]
        whole = self._whole_model(session)
        if whole is not None:  # a template or Dynare model: solved as it is
            if len(_names(self.equations)) > 1:
                return [f"{whole[0]} is a whole model: solve it on its own, without other equations"]
            extra = [label for label, value in (("Shocks", self.shocks), ("Params", self.params),
                                                ("Steady state", self.steady_state), ("Guess", self.guess)) if value.strip()]
            return [f"{whole[0]} has its own {', '.join(extra).lower()}: leave {' and '.join(extra)} empty"] if extra else []
        if not _names(self.shocks):
            return ["give at least one shock"]
        try:
            _calibration(self.params), _calibration(self.steady_state), _calibration(self.guess)
        except (ValueError, TypeError) as exc:
            return [str(exc)]
        return []

    def to_call(self, session: Session | None = None) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        whole = self._whole_model(session)
        if whole is not None:  # nk = Template(...): nk.solve()
            return make_call(f"pymetrix:{whole[1]}.solve", Ref(whole[0]), output=self.name, recorded_by=self.command_id)
        linear = make_call("pymetrix:model.linearize", _model_fields(self.equations, self.params), shocks=_names(self.shocks) or (),
                           exogenous=_calibration(self.steady_state) or None, guess=_calibration(self.guess) or None)
        return make_call("pymetrix:linear_model.solve", linear, output=self.name, recorded_by=self.command_id)

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()



_TEMPLATES = ("new_keynesian", "rbc", "gap_model")


@command(menu="Model/Template model...")
@dataclass(frozen=True)
class LoadTemplate(Command):
    """Load a template model: a small standard model with calibrated parameters, shocks and measurement equations.

    new_keynesian is the three-equation New Keynesian model; rbc the stochastic growth model; gap_model a small
    semi-structural projection model with an unobserved output gap. Solve it, check it, take it to data with
    Model > Analysis > State-space form and Model > Analysis > Kalman filter, or copy its equations to start your own.
    """

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models",)

    template: Annotated[Literal["new_keynesian", "rbc", "gap_model"], Widget("choice", help="new_keynesian (three-equation NK), rbc (stochastic growth) or gap_model "
                                                                                           "(semi-structural, unobserved output gap)")] = "new_keynesian"
    params: Annotated[str, Widget("text", help="parameters to change, e.g. phi_pi=1.2")] = ""
    name: str = "template"
    shock_sd: Annotated[str, Widget("text", advanced=True, help="shock standard deviations, e.g. ey=0.5, ei=0.2; empty = the template's")] = ""

    def validate(self, session: Session) -> list[str]:
        try:
            _calibration(self.params), _calibration(self.shock_sd)
        except (ValueError, SpecificationError) as exc:
            return [str(exc)]
        return []

    def to_call(self) -> Any:
        from ..calls import make_call

        sd = _calibration(self.shock_sd)
        return make_call(f"pymetrix:templates.{self.template}", output=self.name, recorded_by=self.command_id,
                         **_calibration(self.params), **({"shock_sd": sd} if sd else {}))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Model/State-space form...")
@dataclass(frozen=True)
class StateSpaceForm(Command):
    """Put a solved forward-looking model in state-space form, linked to data by measurement equations.

    Choose a template model (its measurement equations are used unless you write others) or a rational-expectations
    solution. Measurement equations give each observed series as an expression in the model's variables and their
    lags, one per line, e.g. infl = 2 + 4*pi or growth = 2 + 4*(y - y[-1]). Shock sizes and measurement errors are
    standard deviations.
    """

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models", "state-space")

    source: Annotated[str, Widget("object", source="Template,LinearSolution", help="a template model or a rational-expectations solution")]
    measurement: Annotated[str, Widget("multiline", help="one per line: observed = expression in model variables; empty = the template's")] = ""
    measurement_params: Annotated[str, Widget("text", help="values for parameters in the measurement equations, e.g. pi_ss=2")] = ""
    shock_sd: Annotated[str, Widget("text", help="shock standard deviations, e.g. ey=0.5, ei=0.2; empty = the template's (or 1)")] = ""
    errors: Annotated[str, Widget("text", help="measurement-error standard deviations, e.g. growth=0.5")] = ""
    name: str = "system"
    transforms: Annotated[str, Widget("text", advanced=True, group="Raw data",
                                      help="observed series made from raw data, rebuilt into it in forecasts: "
                                           "growth = 400*dlog(gdp); infl = 400*dlog(cpi)")] = ""
    reported: Annotated[str, Widget("multiline", advanced=True, group="Reported series",
                                    help="one per line: name = expression in model variables, reported with the smoothed "
                                         "history and forecasts but not fitted (e.g. gap = 100*y)")] = ""

    def _measurement(self) -> Any:
        from ..models import MeasurementMap

        lines = [x.strip() for x in self.measurement.splitlines() if x.strip()]
        reported = [x.strip() for x in self.reported.splitlines() if x.strip()]
        if not lines:
            if reported or self.transforms.strip():
                raise SpecificationError("reported series and transforms go with measurement equations: write them too")
            return None
        return MeasurementMap(lines, params=_calibration(self.measurement_params), errors=_calibration(self.errors),
                              transforms=_assignments(self.transforms), reported=reported)

    def validate(self, session: Session) -> list[str]:
        try:
            m = self._measurement()
            _calibration(self.shock_sd), _calibration(self.errors)
        except Exception as exc:
            return [str(exc)]
        from ..models.templates import Template

        if m is None and not isinstance(session.objects.get(self.source), Template):  # a solution needs equations
            return ["write measurement equations (a solution has none of its own)"]
        return []

    def to_call(self, session: Session) -> Any:
        from ..calls import method_call
        from ..codec import Ref

        return method_call(type(session[self.source]), "system", Ref(self.source), output=self.name, recorded_by=self.command_id,
                           measurement=self._measurement(), shock_sd=_calibration(self.shock_sd) or None,
                           measurement_sd=_calibration(self.errors) or None)

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        m = self._measurement()
        code_m = "None"
        if m is not None:
            parts = [repr(list(m.equations))] + ([f"params={dict(m.params)!r}"] if m.params else []) + ([f"errors={dict(m.errors)!r}"] if m.errors else [])
            parts += ([f"transforms={dict(m.transforms)!r}"] if m.transforms else []) + ([f"reported={list(m.reported)!r}"] if m.reported else [])
            code_m = f"MeasurementMap({', '.join(parts)})"
        sd, errors = _calibration(self.shock_sd), _calibration(self.errors)
        args = [code_m, repr(sd) if sd else "None", repr(errors) if errors and m is None else "None"]
        while args and args[-1] == "None":
            args.pop()
        return f"{self.name} = {self.source}.system({', '.join(args)})"


@command(menu="Model/Kalman filter...")
@dataclass(frozen=True)
class KalmanFilter(Command):
    """Estimate a model's variables from data with the Kalman smoother: e.g. the output gap behind observed inflation
    and interest rates. The result has every model variable in levels and each observed series as the model fits it.

    The parameters are those of the state-space form, calibrated or estimated earlier; nothing is estimated here.
    The smoother uses the whole sample, so estimates near its end rest on fewer later observations and are revised
    as new data arrive, as the output gap is in practice. Compare each ``<series>_fitted`` with the data: the gaps
    are the measurement errors, and large, persistent ones mean the model or its measurement equation fits that
    series poorly. The estimates depend on the shock and measurement-error sizes: bigger measurement errors let the
    filter follow the data less closely.
    """

    topics: ClassVar[tuple[str, ...]] = ("state-space",)

    system: Annotated[str, Widget("object", source="StateSpaceSystem", help="a model in state-space form (Model > Analysis > State-space form)")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "filtered"

    def _span(self) -> tuple[str | None, str | None] | None:
        return (self.start or None, self.end or None) if self.start or self.end else None

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:state_space_system.filter", Ref(self.system), output=self.name, recorded_by=self.command_id,
                         data=Ref(self.data), span=self._span())

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Model/Estimate model...")
@dataclass(frozen=True)
class EstimateModel(Command):
    """Estimate a model's parameters and shock sizes from data, by maximum likelihood with the Kalman filter: a template
    model, an IRIS model, or a Dynare file imported with Model > Import > Dynare file.

    The model is linearised and solved at each trial value and fitted to the observed series of its measurement
    equations. List the parameters to estimate (and shock sizes, sd_<shock>), or leave the list empty for every
    calibrated parameter and shock size. With priors (e.g. kappa=Gamma(0.3, 0.1); rho=Beta(0.8, 0.1); sd_ei=InvGamma1(0.2, inf),
    each given by its mean and standard deviation, as in Dynare; also Normal and Uniform(low, high)) the estimate is the
    posterior mode; then Model > Metropolis-Hastings draws from the whole posterior.

    A Dynare file brings its own estimation: leave the data, parameters and priors empty to use its datafile,
    estimated_params and options (presample, lik_init...), as Dynare's estimation command would. Medium-sized models
    can have more than one mode: Starts (under Advanced) also optimises from draws from the priors, keeps the best, and
    says in the results where the others stopped.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        import pandas as pd

        from pymetrix import Dataset, priors, templates

        nk = templates.new_keynesian()             # a template: the model and its measurement equations
        sim = nk.solve().simulate(120, nk.shock_sd, seed=11)
        sim.index = pd.period_range("1995Q1", periods=120, freq="Q", name="time")
        data = nk.measurement.apply(Dataset.from_frame(sim))   # observed: inflation, the policy rate, GDP growth
        fit = nk.estimate(data, estimate=["kappa", "rho"])     # maximum likelihood, Kalman filter; the rest calibrated
        print(fit.summary())       # estimates, standard errors, bounds, log likelihood
        mode = nk.estimate(data, estimate=["kappa", "rho"],    # with priors (mean, sd): the posterior mode
                           priors={"kappa": priors.Gamma(0.15, 0.05), "rho": priors.Beta(0.7, 0.1)})
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", ("the estimates with standard errors, priors and bounds; the notes give the method, log likelihood "
                     "(and log posterior), AIC, BIC and convergence")),
        ("Parameters", "every parameter with its status: estimated, or calibrated (kept at its value)"),
        ("Shock sizes", "the shocks' standard deviations, estimated where listed as sd_{shock}"),
        ("Smoothed variables", ("the model's variables estimated from the data by the Kalman smoother, with the "
                                "observed series' fitted values")),
        ("Moments, Variance decomposition, Decision rules", ("what the estimated model implies: standard deviations and "
                                                             "autocorrelations, each shock's share, the solved rules")),
        ("Several starts", ("a note says how many starts reached the best value; starts that stop elsewhere point to "
                            "several modes")),
    )

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models", "state-space", "likelihood-gradient", "particle-smoother")

    source: Annotated[str, Widget("object", source="Template, DynareModel",
                                  help="a template, IRIS or Dynare model (Model > Template model, Import IRIS file, Import Dynare file)")]
    data: Annotated[str, Widget("object", source="Dataset", empty="empty: a Dynare file's own datafile",
                                help="data with the model's observed series; leave empty for a Dynare file's own datafile")] = "data"
    estimate: Annotated[str, Widget("text", help="parameters and shock sizes to estimate, e.g. kappa, rho, sd_ei; empty = all (a Dynare file: its estimated_params)")] = ""
    priors: Annotated[str, Widget("text", help="e.g. kappa=Gamma(0.3, 0.1); rho=Beta(0.8, 0.1): mean and standard deviation; empty = none (a Dynare file: its own)")] = ""
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    starts: Annotated[int, Widget("number", min=1, max=64, advanced=True,
                                  help="optimise from this many points (the starting values, then draws from the priors) and keep the best")] = 1
    name: str = "estimated"
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="random seed of the draws from the priors (several starts)")] = 1
    workers: Annotated[int | None, Widget("number", min=1, max=64, advanced=True,
                                          help="starts run at once in other processes; empty = automatic")] = None
    initial: Annotated[Literal["default", "auto", "stationary", "diffuse", "scaled"], Widget("choice", advanced=True, group="Kalman filter",
                       help="the filter's start: stationary, diffuse, scaled (variance 10 I) or auto; default = auto (a Dynare file: its lik_init)")] = "default"
    presample: Annotated[int | None, Widget("number", min=0, max=400, advanced=True, group="Kalman filter",
                                            help="periods filtered but left out of the likelihood; empty = none (a Dynare file: its presample)")] = None
    max_iter: Annotated[int, Widget("number", min=10, max=100_000, step=100, advanced=True, help="the optimiser's iterations")] = 2000
    bounds: Annotated[str, Widget("text", advanced=True, help="parameter bounds, e.g. kappa=(0.01, 5); rho=(0, 0.99)")] = ""
    growth: Annotated[bool, Widget("checkbox", advanced=True, help="a balanced-growth steady state (variables trend at constant rates)")] = False
    occbin: Annotated[bool, Widget("checkbox", advanced=True, group="Kalman filter",
                                   help="the likelihood of the OccBin filter, for a model with occasionally binding constraints "
                                        "(its max or min, such as a lower bound on the policy rate); slower")] = False
    regimes: Annotated[str, Widget("text", advanced=True, group="Regimes",
                                   help="parameters that change at known dates in the sample, e.g. 2009Q1: phi_pi=1.2; each value "
                                        "is estimated as name@period (phi_pi@2009Q1), which estimate, priors and bounds can name")] = ""
    order: Annotated[Literal[1, 2], Widget("choice", advanced=True, group="Particle filter",
                                           help="the solution's order: 1 (linear, Kalman filter) or 2 (second order, pruned, "
                                                "estimated with the particle filter; every observed series needs a measurement error)")] = 1
    filter: Annotated[Literal["default", "kalman", "particle", "guided"], Widget("choice", advanced=True, group="Particle filter",
                      help="the likelihood's filter: Kalman, particle (bootstrap) or guided particle filter; default = Kalman at "
                           "order 1, particle at order 2")] = "default"
    particles: Annotated[int, Widget("number", min=10, max=1_000_000, step=500, advanced=True, group="Particle filter",
                                     help="particles of the particle filter (its seed: the seed above, with common random numbers)")] = 2000
    smooth_paths: Annotated[int, Widget("number", min=0, max=100_000, step=50, advanced=True, group="Particle filter",
                                        help="paths the particle smoother draws for the smoothed variables (backward simulation); "
                                             "0 = filtered means instead")] = 100
    gradient: Annotated[Literal["auto", "analytic", "numerical"], Widget("choice", advanced=True, group="Kalman filter",
                        help="the optimiser's gradient: analytic (the Kalman filter's score) where it applies with auto, "
                             "or central differences (numerical)")] = "auto"
    diffuse_terms: Annotated[Literal["include", "exclude"], Widget("choice", advanced=True, group="Kalman filter",
                             help="with diffuse states, the log likelihood's diffuse periods: included (the exact diffuse "
                                  "likelihood, Durbin and Koopman) or left out (as statsmodels reports it)")] = "include"

    def _span(self) -> tuple[str | None, str | None] | None:
        return (self.start or None, self.end or None) if self.start or self.end else None

    def to_call(self, session: Session) -> Any:
        from ..calls import method_call
        from ..codec import Ref

        from ..models.dynare import DynareModel

        model = session[self.source]
        if not self.data and not isinstance(model, DynareModel):
            raise SpecificationError("choose the data with the model's observed series")
        options: dict[str, Any] = {"estimate": _names(self.estimate) or None, "priors": _priors(self.priors) or None}
        if self._span():
            options["span"] = self._span()
        options.update(self._advanced())
        return method_call(type(model), "estimate", Ref(self.source), output=self.name, recorded_by=self.command_id,
                           data=Ref(self.data) if self.data else None, **options)

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        def number(v: float) -> str:
            return "float('inf')" if v == float("inf") else f"{v:g}"

        priors_code = ", ".join(f"{n!r}: priors.{kind}({', '.join(map(number, values))})"
                                for n, (kind, values) in _prior_specs(self.priors).items())
        args = (([self.data] if self.data else []) + ([f"estimate={_names(self.estimate)!r}"] if self.estimate else [])
                + ([f"priors={{{priors_code}}}"] if priors_code else []) + ([f"span={self._span()!r}"] if self._span() else [])
                + [f"{k}={v!r}" for k, v in self._advanced().items()])
        return f"{self.name} = {self.source}.estimate({', '.join(args)})"

    def _advanced(self) -> dict[str, Any]:
        options: dict[str, Any] = {"starts": int(self.starts), "seed": self.seed} if self.starts > 1 else {}
        particle = self.order == 2 or self.filter in ("particle", "guided")
        if self.order != 1:
            options["order"] = int(self.order)
        if self.filter != "default":
            options["filter"] = self.filter
        if particle:
            options["particles"] = int(self.particles)
            options["seed"] = self.seed
            if self.smooth_paths != 100:
                options["smooth_paths"] = int(self.smooth_paths)
        if self.gradient != "auto":
            options["gradient"] = self.gradient
        if self.diffuse_terms != "include":
            options["diffuse_terms"] = self.diffuse_terms
        if self.starts > 1 and self.workers is not None:
            options["workers"] = self.workers
        if self.initial != "default":
            options["start"] = self.initial
        for key, default in (("presample", None), ("max_iter", 2000), ("growth", False), ("occbin", False)):
            if getattr(self, key) != default:
                options[key] = getattr(self, key)
        if self.bounds.strip():
            options["bounds"] = _bounds(self.bounds)
        if self.regimes.strip():
            options["regimes"] = self.regimes.strip()
        return options

    def validate(self, session: Session) -> list[str]:
        try:
            _prior_specs(self.priors), _bounds(self.bounds)
        except (ValueError, SpecificationError) as exc:
            return [str(exc)]
        if self.order == 2 and self.filter == "kalman":
            return ["a second-order solution is estimated with the particle filter: choose particle or guided (or default)"]
        if (self.order == 2 or self.filter in ("particle", "guided")) and (self.occbin or self.regimes.strip()):
            return ["the particle filter does not take OccBin constraints or regimes"]
        return _regime_problems(self.regimes)


def _regime_problems(text: str) -> list[str]:
    """What is wrong with regimes typed as text (``2009Q1: phi_pi=1.2; 2025Q1..2026Q4: rho=1``), if anything."""
    from ..models.regimes import parse_regimes

    try:
        parse_regimes(text)
    except SpecificationError as exc:
        return [f"regimes: {exc}"]
    return []


def _bounds(text: str) -> dict[str, tuple[float, float]]:
    """``kappa=(0.01, 5); rho=(0, 0.99)`` -> {"kappa": (0.01, 5.0), "rho": (0.0, 0.99)}."""
    import re

    out = {}
    for item in [t.strip() for t in re.split(r"[;\n]", text) if t.strip()]:
        m = re.fullmatch(r"([\w.@]+)\s*=\s*\(?([^,()]+),([^,()]+)\)?", item)  # phi_pi@2009Q1: a regime's value
        try:
            low, high = float(m.group(2)), float(m.group(3))  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            raise SpecificationError(f"give bounds as name=(low, high): {item!r}") from None
        if not low < high:
            raise SpecificationError(f"the lower bound must be below the upper: {item!r}")
        out[m.group(1)] = (low, high)  # type: ignore[union-attr]
    return out


def _prior_specs(text: str) -> dict[str, tuple[str, list[float]]]:
    from ..estimation.priors import specs

    return specs(text)


def _priors(text: str) -> dict[str, Any]:
    from ..estimation.priors import parse

    return parse(text)


@command(menu="Model/Estimate DSGE-VAR...", id="pymetrix:estimate_dsge_var")
@dataclass(frozen=True)
class EstimateDSGEVAR(Command):
    """Estimate a DSGE-VAR(lambda) (Del Negro and Schorfheide 2004): a VAR in the model's observed series whose prior
    comes from the model, weighted as lambda times the sample's observations, with the model's parameters estimated by
    the VAR's marginal likelihood.

    Lambda measures how far the data trust the model: large values hold the VAR to the model's restrictions, small ones
    leave it close to the unrestricted VAR (least squares); it is at least (k + n) / T, the regressors and series per
    observation. Fix it, choose it on a grid by the marginal data density (lambda-hat: the top of the hump, "grid"), or
    estimate it with a prior (e.g. Gamma(1, 0.5)). A best lambda well inside the grid says the model's restrictions help
    but do not fit the data exactly; the largest lambda best says they fit; the smallest, that they do not.

    The model needs stationary observables (no unit roots), and its responses as many shocks as observables. Then
    Model > Metropolis-Hastings draws from the parameters' posterior; responses (identified by the model's rotation),
    variance decompositions and forecasts follow from the result.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        import pandas as pd

        from pymetrix import Dataset, priors, templates

        nk = templates.new_keynesian()
        sim = nk.solve().simulate(160, nk.shock_sd, seed=11)
        sim.index = pd.period_range("1980Q1", periods=160, freq="Q", name="time")
        data = nk.measurement.apply(Dataset.from_frame(sim))
        dv = nk.dsge_var(data, lags=2, lambda_="grid", estimate=["kappa", "rho"],
                         priors={"kappa": priors.Gamma(0.15, 0.05), "rho": priors.Beta(0.7, 0.1)})
        print(dv.lambdas())        # the marginal data density at each lambda: lambda-hat at the top
        print(dv.irf(12).table())  # responses identified by the model's rotation
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", "the mode of the model's parameters (and lambda), with priors and standard errors; lambda, the sample, "
                    "log p(Y | theta, lambda) and the marginal data density"),
        ("Marginal density by lambda", "with lambda chosen on a grid: the log marginal data density at each value (the hump)"),
        ("Responses", "impulse responses identified by the model's rotation, beside the model's own"),
        ("Variance decomposition", "each shock's share of the forecast-error variance"),
    )

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models", "bayesian-var")

    source: Annotated[str, Widget("object", source="Template, DynareModel",
                                  help="a template, IRIS or Dynare model (a Dynare file's estimation(dsge_var=..., dsge_varlag=...) is used)")]
    data: Annotated[str, Widget("object", source="Dataset", empty="empty: a Dynare file's own datafile",
                                help="data with the model's observed series; leave empty for a Dynare file's own datafile")] = "data"
    lags: Annotated[int, Widget("number", min=1, max=12, help="lags of the VAR")] = 4
    lambda_: Annotated[str, Widget("text", help="the prior's weight: a number, grid (lambda-hat) or a prior such as Gamma(1, 0.5); "
                                                "empty = a Dynare file's dsge_var, else 1")] = "grid"
    lambdas: Annotated[str, Widget("text", advanced=True, help="the grid of lambda for lambda-hat, e.g. 0.25, 0.5, 1, 2, 5; empty = the default")] = ""
    estimate: Annotated[str, Widget("text", help="parameters and shock sizes to estimate, e.g. kappa, rho, sd_ei; empty = all")] = ""
    priors: Annotated[str, Widget("text", help="e.g. kappa=Gamma(0.3, 0.1); rho=Beta(0.8, 0.1): mean and standard deviation")] = ""
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "dsge_var"
    max_iter: Annotated[int, Widget("number", min=10, max=100_000, step=100, advanced=True, help="the optimiser's iterations")] = 2000
    bounds: Annotated[str, Widget("text", advanced=True, help="parameter bounds, e.g. kappa=(0.01, 5); rho=(0, 0.99)")] = ""

    def _span(self) -> tuple[str | None, str | None] | None:
        return _span(self.start, self.end)

    def _lambda(self) -> Any:
        text = self.lambda_.strip()
        if not text:
            return None
        if text.lower() == "grid":
            return "grid"
        try:
            return float(text)
        except ValueError:
            return _priors(f"lambda={text}")["lambda"]

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"lags": int(self.lags)}
        lam = self._lambda()
        if lam is not None:
            options["lambda_"] = lam
        if self.lambdas.strip() and lam == "grid":
            options["lambdas"] = [float(v) for v in _names(self.lambdas)]
        if self.estimate.strip():
            options["estimate"] = _names(self.estimate)
        if self.priors.strip():
            options["priors"] = _priors(self.priors)
        if self._span():
            options["span"] = self._span()
        if self.max_iter != 2000:
            options["max_iter"] = self.max_iter
        if self.bounds.strip():
            options["bounds"] = _bounds(self.bounds)
        return options

    def to_call(self, session: Session) -> Any:
        from ..calls import method_call
        from ..codec import Ref

        from ..models.dynare import DynareModel

        model = session[self.source]
        if not self.data and not isinstance(model, DynareModel):
            raise SpecificationError("choose the data with the model's observed series")
        options = self._options()
        if options.get("lambda_") is None and not isinstance(model, DynareModel):
            options["lambda_"] = 1.0
        return method_call(type(model), "dsge_var", Ref(self.source), output=self.name, recorded_by=self.command_id,
                           data=Ref(self.data) if self.data else None, **options)

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        def number(v: float) -> str:
            return "float('inf')" if v == float("inf") else f"{v:g}"

        def prior_code(text: str) -> str:
            return ", ".join(f"{n!r}: priors.{kind}({', '.join(map(number, values))})" for n, (kind, values) in _prior_specs(text).items())

        options = self._options()
        args = [self.data] if self.data else []
        for key, value in options.items():
            if key == "priors":
                args.append(f"priors={{{prior_code(self.priors)}}}")
            elif key == "lambda_" and not isinstance(value, (str, float)):
                args.append(f"lambda_={prior_code(f'lambda={self.lambda_.strip()}').split(': ', 1)[1]}")
            else:
                args.append(f"{key}={value!r}")
        return f"{self.name} = {self.source}.dsge_var({', '.join(args)})"

    def validate(self, session: Session) -> list[str]:
        try:
            _prior_specs(self.priors), _bounds(self.bounds), self._lambda()
            [float(v) for v in _names(self.lambdas)]
        except (ValueError, SpecificationError, KeyError) as exc:
            return [f"lambda: give a number, grid or a prior such as Gamma(1, 0.5) ({exc})" if "lambda" in str(exc) else str(exc)]
        return []


@command(menu="Model/Metropolis-Hastings...")
@dataclass(frozen=True)
class RunMCMC(Command):
    """Draw from the posterior of an estimated model by random-walk Metropolis-Hastings, started around the posterior
    mode (Model > Estimate model, with a prior on every estimated parameter).

    Shows posterior means, medians and HPD intervals with convergence checks (R-hat, effective draws), the marginal
    data density for comparing models, prior and posterior densities, and the chains. More draws give steadier
    results; each draw solves the model once.

    Check convergence before reading the estimates: R-hat close to 1 for every parameter (above about 1.1 the
    chains disagree) and effective draws in the hundreds at least. If they fall short, run longer chains rather than
    reading the posterior. A posterior that looks like its prior means the data say little about that parameter
    (see Model > Identification). The marginal data density compares models estimated on the same data and
    observables; it is sensitive to the priors.

    Sampler NUTS (the No-U-Turn sampler) instead follows the posterior's gradient: analytic for linearised models
    filtered by the Kalman filter (the steady state, the solution and the filter differentiated exactly), by central
    differences otherwise (regimes, OccBin, diffuse starts; or Gradient numerical), when each leapfrog step costs two
    model solutions per parameter. Each draw costs many leapfrog steps but the draws are far less correlated, so a few
    hundred can do the work of many thousand Metropolis-Hastings draws; compare the effective draws. Its burn-in
    (warmup) adapts the step size and the mass matrix; divergent transitions are reported in the notes.
    """

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models", "state-space", "likelihood-gradient")

    source: Annotated[str, Widget("object", source="FittedModel, FittedDSGEVAR", help="a model (or DSGE-VAR) estimated with priors")]
    draws: Annotated[int, Widget("number", help="draws kept per chain after the burn-in; more if R-hat or effective draws fall short",
                                 min=10, max=1000000)] = 5000
    chains: Annotated[int, Widget("number", help="independent chains, for the convergence checks", min=1, max=16)] = 2
    seed: Annotated[int, Widget("number", advanced=True, help="random seed, for reproducible draws", min=0, max=2147483647)] = 1
    burn: Annotated[int, Widget("number", min=0, max=1000000, advanced=True,
                                help="draws discarded first, while the proposal is tuned; 0 = as many as kept")] = 0
    workers: Annotated[int, Widget("number", min=0, max=64, advanced=True,
                                   help="chains run at once in other processes; 0 = automatic (in parallel when it pays), 1 = here")] = 0
    name: str = "posterior"
    scale: Annotated[float | None, Widget("number", min=0.01, max=10, step=0.05, advanced=True, group="Proposal",
                                          help="the proposal's scale; empty = 2.38 / sqrt(parameters), tuned during the burn-in")] = None
    adapt: Annotated[bool, Widget("checkbox", advanced=True, group="Proposal",
                                  help="tune the proposal's scale during the burn-in towards a quarter of draws accepted")] = True
    jump: Annotated[float, Widget("number", min=0.0, max=0.5, step=0.01, advanced=True, group="Proposal",
                                  help="share of steps that propose a jump to another mode the estimation found; 0 = none")] = 0.05
    sampler: Annotated[Literal["rwmh", "nuts"], Widget("choice", help="rwmh: random-walk Metropolis-Hastings; nuts: the No-U-Turn "
                                                       "sampler (follows the gradient: fewer, costlier, less correlated draws)")] = "rwmh"
    target_accept: Annotated[float, Widget("number", min=0.5, max=0.99, step=0.05, advanced=True, show_if=("sampler", ("nuts",)),
                                           help="NUTS: the acceptance statistic its step size is tuned towards; higher if divergences")] = 0.8
    gradient: Annotated[Literal["auto", "analytic", "numerical"], Widget("choice", advanced=True, show_if=("sampler", ("nuts",)),
                        help="NUTS: the gradient, analytic (Kalman-filter score) where it applies with auto, or central "
                             "differences (numerical)")] = "auto"
    metric: Annotated[Literal["dense", "diag", "mode"], Widget("choice", advanced=True, show_if=("sampler", ("nuts",)),
                      help="NUTS: the mass matrix, adapted in the burn-in; dense undoes correlations between parameters, "
                           "diag only scales them; mode keeps the inverse Hessian at the mode (a short burn-in)")] = "dense"

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {k: v for k, v in (("burn", self.burn), ("workers", self.workers)) if v}
        options.update({k: getattr(self, k) for k, d in (("scale", None), ("adapt", True), ("jump", 0.05), ("sampler", "rwmh"))
                        if getattr(self, k) != d})
        if self.sampler == "nuts" and self.target_accept != 0.8:
            options["target_accept"] = self.target_accept
        if self.sampler == "nuts" and self.gradient != "auto":
            options["gradient"] = self.gradient
        if self.sampler == "nuts" and self.metric != "dense":
            options["metric"] = self.metric
        return options

    def to_call(self, session: Session) -> Any:
        from ..calls import method_call
        from ..codec import Ref

        return method_call(type(session[self.source]), "mcmc", Ref(self.source), output=self.name, recorded_by=self.command_id,
                           draws=self.draws, chains=self.chains, seed=self.seed, **self._options())

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        extra = "".join(f", {k}={v!r}" for k, v in self._options().items())
        return f"{self.name} = {self.source}.mcmc(draws={self.draws}, chains={self.chains}, seed={self.seed}{extra})"


@command(menu="Model/Sequential Monte Carlo...", id="pymetrix:run_smc")
@dataclass(frozen=True)
class RunSMC(Command):
    """Draw from the posterior of an estimated model by sequential Monte Carlo (Herbst and Schorfheide), as Dynare 6
    does: particles drawn from the prior move to the posterior through the likelihood raised to rising powers, each
    stage reweighting, resampling and moving them by Metropolis-Hastings.

    It needs no starting point and finds posteriors with several modes or long ridges, where a Metropolis-Hastings chain
    can stay in one; it also gives the marginal data density from the same run. The result is a posterior like
    Metropolis-Hastings's (means, HPD intervals, densities, forecasts and responses with parameter uncertainty).
    More particles give steadier estimates; more steps a stage move them further, at a cost of one model solution each.
    """

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models", "state-space")
    schema_version: ClassVar[int] = 2  # 2: target became ess_fraction

    source: Annotated[str, Widget("object", source="FittedModel", help="a model estimated with priors")]
    particles: Annotated[int, Widget("number", min=50, max=1000000, step=500,
                                     help="particles drawn from the prior; more give steadier means and marginal density")] = 2000
    seed: Annotated[int, Widget("number", advanced=True, help="random seed, for reproducible draws", min=0, max=2147483647)] = 1
    name: str = "posterior"
    steps: Annotated[int, Widget("number", min=1, max=100, advanced=True,
                                 help="Metropolis-Hastings steps moving the particles at each stage")] = 2
    ess_fraction: Annotated[float, Widget("number", min=0.05, max=0.95, step=0.05, advanced=True,
                                          help="the particles' effective sample size kept at each tempering stage, as a share "
                                               "of the particles: higher means more, smaller stages")] = 0.5
    workers: Annotated[int, Widget("number", min=0, max=64, advanced=True,
                                   help="processes moving the particles at once; 0 = automatic (when it pays), 1 = here")] = 0

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {k: getattr(self, k) for k, d in (("steps", 2), ("ess_fraction", 0.5)) if getattr(self, k) != d}
        if self.workers:
            options["workers"] = self.workers
        return options

    def to_call(self, session: Session) -> Any:
        from ..calls import method_call
        from ..codec import Ref

        return method_call(type(session[self.source]), "smc", Ref(self.source), output=self.name, recorded_by=self.command_id,
                           particles=self.particles, seed=self.seed, **self._options())

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        extra = "".join(f", {k}={v!r}" for k, v in self._options().items())
        return f"{self.name} = {self.source}.smc(particles={self.particles}, seed={self.seed}{extra})"


@migration("pymetrix:run_smc", 1)
def _run_smc_ess_fraction(fields: dict[str, Any]) -> dict[str, Any]:  # schema 2: target became ess_fraction
    if "target" in fields:
        fields["ess_fraction"] = fields.pop("target")
    return fields


@command(menu="Model/Identification...")
@dataclass(frozen=True)
class CheckIdentification(Command):
    """Check whether the data's means and autocovariances can tell an estimated model's parameters apart (Iskrev's
    test): the rank of their Jacobian, each parameter's sensitivity, and the parameters whose effects look alike.

    The check is local, at the estimates or at the values given under At. Full rank means the parameters can be
    told apart from these moments; a lower rank means some parameter, or combination of parameters, cannot. A
    parameter with low sensitivity barely moves the moments, so it is weakly identified and its posterior will
    largely repeat its prior. Collinearity near 1 means its effect can be reproduced by the others, mostly by the
    one named as closest: fix one of the pair, or add observables that separate them. More lags add moments, so a
    parameter that fails with few lags may pass with more.
    """

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models",)

    source: Annotated[str, Widget("object", source="FittedModel",
                                  help="a model estimated with Model > Estimate model; checked at its estimates unless At parameter values are given")]
    lags: Annotated[int, Widget("number", help="autocovariances up to this lag enter the moments; more lags reveal persistence parameters",
                                min=0, max=40)] = 4
    name: str = "identification"
    at: Annotated[str, Widget("text", advanced=True, help="check at other parameter values, e.g. kappa=0.2, rho=0.5; empty = the estimates")] = ""
    elasticity: Annotated[bool, Widget("checkbox", advanced=True,
                                       help="measure each moment's change relative to its size, so the largest variances do not "
                                            "make every pair of parameters look alike (useful on large models)")] = False

    def validate(self, session: Session) -> list[str]:
        try:
            _calibration(self.at)
        except (ValueError, SpecificationError) as exc:
            return [str(exc)]
        return []

    def _options(self) -> dict[str, Any]:
        at = _calibration(self.at)
        return {"lags": self.lags, **({"at": at} if at else {}), **({"elasticity": True} if self.elasticity else {})}

    def to_call(self, session: Session) -> Any:
        from ..calls import method_call
        from ..codec import Ref

        return method_call(type(session[self.source]), "identification", Ref(self.source), output=self.name,
                           recorded_by=self.command_id, **self._options())

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        return f"{self.name} = {self.source}.identification({', '.join(f'{k}={v!r}' for k, v in self._options().items())})"


_parse_options = parse_options


@command(menu="Test/Run test...")
@dataclass(frozen=True)
class RunTest(Command):
    """Run a specification or diagnostic test on a fit, a VAR or VECM, or a dataset.

    The list shows only tests that apply to the chosen object; the test's settings appear below it.
    """

    topics: ClassVar[tuple[str, ...]] = ('diagnostic-tests', 'unit-roots')
    schema_version: ClassVar[int] = 2  # 2: target became source

    source: Annotated[str, Widget("object", source="FittedEquation,FittedVAR,FittedVECM,Dataset",
                                  help="the object to examine: a fit, a VAR or VECM, or a dataset")]
    test: Annotated[str, Widget("test", depends_on="source")]
    options: Annotated[str, Widget("options", depends_on="test", group="Test settings")] = ""
    name: str = "test"

    def validate(self, session: Session) -> list[str]:
        if self.source not in session.objects:
            return [f"no object named {self.source!r}"]
        allowed = available_tests(session[self.source])
        return [] if self.test in allowed else [f"{self.test!r} does not apply to {self.source!r}; choose one of {allowed}"]

    def to_call(self, session: Session) -> Any:
        from ..calls import method_call
        from ..codec import Ref

        args, kw = _parse_options(self.options)
        return method_call(type(session[self.source]), "test", Ref(self.source), output=self.name, recorded_by=self.command_id,
                           name=self.test, args=tuple(args), **kw)

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        args, kw = _parse_options(self.options)
        extra = "".join(f", {a!r}" for a in args) + "".join(f", {k}={v!r}" for k, v in kw.items())
        return f"{self.name} = {self.source}.test({self.test!r}{extra})"


@migration("pymetrix:run_test", 1)
def _run_test_source(fields: dict[str, Any]) -> dict[str, Any]:  # schema 2: target became source
    if "target" in fields:
        fields["source"] = fields.pop("target")
    return fields
