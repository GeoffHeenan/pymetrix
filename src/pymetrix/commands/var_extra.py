"""More VAR and time-series dialogs: Bayesian VARs with stochastic volatility, Markov-switching and quantile VARs,
restriction and stability tests on a VECM, spectra and long memory, VARMA, threshold VARs with Tsay's test, and
bilinear models.

Each builds the API call it stands for (``make_call`` / ``method_call``), so the step replays and its script is that
call's code. Impulse responses of the new VAR types go through Estimate → VARs and local projections → Impulse
responses…, which takes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from ..errors import SpecificationError
from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session

__all__ = ["FitBVARSV", "FitBilinear", "FitMSVAR", "FitQuantileVAR", "FitThresholdVAR", "FitVARMA", "LongMemory", "SpectralAnalysis",
           "ThresholdLinearityTest", "VECMRestrictionTest"]


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _span(start: str, end: str) -> tuple[str | None, str | None] | None:
    return (start or None, end or None) if start or end else None


def _need_data(session: Session, data: str) -> list[str]:
    return [] if data in session.objects else [f"no dataset named {data!r}"]


class _CallCommand(Command):
    """A dialog that is one API call: run it, and write its code."""

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)  # type: ignore[attr-defined]

    def to_code(self) -> str:
        return self.to_call().to_code()  # type: ignore[attr-defined]


# --- estimation -------------------------------------------------------------------------------------------------------


@command(menu="Estimate/BVAR with stochastic volatility...")
@dataclass(frozen=True)
class FitBVARSV(_CallCommand):
    """A Bayesian VAR with stochastic volatility (Cogley and Sargent; Carriero, Clark and Marcellino).

    The coefficients are constant, with a Minnesota prior; each shock's standard deviation drifts as a random walk in
    logs, so a larger VAR stays estimable where a TVP-VAR would not. Estimated by Gibbs sampling: start with a few
    hundred draws to see that it runs, then use thousands. The window shows each shock's volatility over time with
    68% bands and the responses at the last period's volatilities; Impulse responses… gives them at any period.

    Tightness shrinks the coefficients towards Own mean (1 for series in levels, 0 for growth rates); KW, under
    Advanced, is the prior scale of the volatilities' drift: larger values let them move more. Shocks are identified
    recursively, in the order the series are listed.
    """

    maturity: ClassVar[str] = "beta"
    topics: ClassVar[tuple[str, ...]] = ("large-and-switching-vars", "bayesian-var", "var")

    variables: Annotated[str, Widget("series", depends_on="data", multiple=True,
                                     help="in the list's order: the recursive ordering of the shocks")]
    data: Annotated[str, Widget("object", source="Dataset", help="a single time series (not a panel)")] = "data"
    lags: Annotated[int, Widget("number", min=1, max=12, help="lags of every variable; 4 is usual for quarterly data")] = 4
    draws: Annotated[int, Widget("number", advanced=True, min=10, max=100_000, step=500,
                                 help="Gibbs draws kept after the burn-in; thousands for stable bands, a few hundred to try a run")] = 2000
    burn: Annotated[int, Widget("number", advanced=True, min=0, max=100_000, step=500, help="Gibbs draws discarded first")] = 1000
    tightness: Annotated[float, Widget("number", min=0.001, max=10, step=0.05, help="Minnesota prior: overall tightness; smaller shrinks more")] = 0.2
    own_mean: Annotated[float, Widget("number", min=-2, max=2, step=0.1, help="prior mean of each own first lag: 1 for levels, 0 for growth rates")] = 1.0
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "bvarsv"
    decay: Annotated[float, Widget("number", min=0, max=4, step=0.5, advanced=True, help="Minnesota prior: lag l's prior sd falls as 1 / l^decay")] = 1.0
    kw: Annotated[float, Widget("number", min=1e-4, max=2, step=0.05, advanced=True, help="prior scale of the log volatilities' drift")] = 0.1
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True,
                                help="random seed of the sampler; if results move with it, the chain is too short")] = 0

    def validate(self, session: Session) -> list[str]:
        return _need_data(session, self.data) + ([] if _names(self.variables) else ["choose the VAR's series"])

    def to_call(self) -> Any:
        from ..bvar_sv import BVARSV
        from ..calls import make_call
        from ..codec import Ref

        options = {k: getattr(self, k) for k, d in (("tightness", 0.2), ("decay", 1.0), ("own_mean", 1.0), ("kw", 0.1), ("seed", 0))
                   if getattr(self, k) != d}
        return make_call("pymetrix:bvar_sv.fit", BVARSV(_names(self.variables), self.lags), output=self.name, recorded_by=self.command_id,
                         data=Ref(self.data), draws=self.draws, burn=self.burn, span=_span(self.start, self.end), **options)


@command(menu="Estimate/Markov-switching VAR...")
@dataclass(frozen=True)
class FitMSVAR(_CallCommand):
    """A Markov-switching VAR (Hamilton; Krolzig), by the EM algorithm and maximum likelihood.

    Choose what switches with the unobserved regime: the intercepts (MSI), the intercepts and the covariance (MSIH),
    or everything, the lag coefficients too (MSIAH). The window shows each regime's estimates, the transition
    probabilities and expected durations, and the smoothed probability of each regime in each period. Regimes are
    ordered by the first variable's intercept, so with GDP growth first regime 0 is the low-growth one.

    The likelihood has local maxima: EM starts from regime paths read off the data (splits by levels and residual
    variances, clustering, one-variable fits) and from random splits, and the best few starts run to convergence; if
    the estimates change with Starts or the seed, try more. Switching everything with several variables and lags needs a long sample. Impulse responses… gives the
    responses within a regime, as if it lasted.
    """

    maturity: ClassVar[str] = "beta"
    topics: ClassVar[tuple[str, ...]] = ("large-and-switching-vars", "markov-switching", "var")

    variables: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the VAR's series; the first orders the regimes")]
    data: Annotated[str, Widget("object", source="Dataset", help="a single time series (not a panel)")] = "data"
    lags: Annotated[int, Widget("number", min=0, max=8, help="lags of every variable (0: switching means and covariances only)")] = 1
    regimes: Annotated[int, Widget("number", min=2, max=5, help="number of regimes; two (e.g. recession and expansion) is usual")] = 2
    switching: Annotated[Literal["intercept", "intercept_variance", "all"],
                         Widget("choice", help="what depends on the regime: intercept (MSI), intercept_variance (MSIH) or all (MSIAH)")] = "intercept_variance"
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "msvar"
    starts: Annotated[int, Widget("number", min=1, max=500, advanced=True, help="random starting splits, besides the starts read off the data; more guards against a local maximum")] = 10
    max_iter: Annotated[int, Widget("number", min=10, max=100_000, step=500, advanced=True, help="EM iterations allowed in each run")] = 5000
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="random seed of the starting splits")] = 0

    def validate(self, session: Session) -> list[str]:
        return _need_data(session, self.data) + ([] if _names(self.variables) else ["choose the VAR's series"])

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref
        from ..ms_var import MSVAR

        options = {k: getattr(self, k) for k, d in (("starts", 10), ("max_iter", 5000), ("seed", 0)) if getattr(self, k) != d}
        return make_call("pymetrix:msvar.fit", MSVAR(_names(self.variables), self.lags, self.regimes, self.switching), output=self.name,
                         recorded_by=self.command_id, data=Ref(self.data), span=_span(self.start, self.end), **options)


@command(menu="Estimate/Quantile VAR...")
@dataclass(frozen=True)
class FitQuantileVAR(_CallCommand):
    """A recursive quantile VAR (White, Kim and Manganelli; Chavleishvili and Manganelli).

    Each equation is a quantile regression, at the same quantile, of a variable on those listed before it in the
    same period and on lags of all of them, so the system is recursive like a Cholesky VAR. Its impulse responses
    (Impulse responses…, or the window's Responses) trace the system's quantile after a unit shock: fit it at 0.1 and
    at 0.5 to see whether bad outcomes respond differently from typical ones. No standard errors are reported.
    """

    maturity: ClassVar[str] = "beta"
    topics: ClassVar[tuple[str, ...]] = ("quantile-regression", "var")

    variables: Annotated[str, Widget("series", depends_on="data", multiple=True,
                                     help="in the list's order: each equation includes the ones before it in the same period")]
    data: Annotated[str, Widget("object", source="Dataset", help="a single time series (not a panel)")] = "data"
    lags: Annotated[int, Widget("number", min=1, max=12, help="lags of every variable")] = 1
    quantile: Annotated[float, Widget("number", min=0.01, max=0.99, step=0.05, help="the quantile of every equation, e.g. 0.1 for the lower tail")] = 0.5
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "qvar"

    def validate(self, session: Session) -> list[str]:
        return _need_data(session, self.data) + ([] if _names(self.variables) else ["choose the VAR's series"])

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref
        from ..quantile_var import QuantileVAR

        return make_call("pymetrix:quantile_var.fit", QuantileVAR(_names(self.variables), self.lags, self.quantile), output=self.name,
                         recorded_by=self.command_id, data=Ref(self.data), span=_span(self.start, self.end))


@command(menu="Estimate/VARMA...")
@dataclass(frozen=True)
class FitVARMA(_CallCommand):
    """A vector ARMA(p, q) by conditional maximum likelihood, started from the VAR(p) least-squares estimates.

    A VARMA can describe with few parameters what a VAR needs many lags for, but its parameters are identified only
    up to the usual VARMA ambiguity: keep p and q small. The window shows the estimates, the largest AR and MA root
    moduli (below 1: stationary and invertible), Cholesky responses in the listed order and the residual covariance.
    """

    maturity: ClassVar[str] = "beta"
    topics: ClassVar[tuple[str, ...]] = ("var",)

    variables: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the series, in the order of the Cholesky responses")]
    data: Annotated[str, Widget("object", source="Dataset", help="a single time series (not a panel)")] = "data"
    p: Annotated[int, Widget("number", min=0, max=8, help="autoregressive order")] = 1
    q: Annotated[int, Widget("number", min=0, max=8, help="moving-average order")] = 1
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "varma_fit"

    def validate(self, session: Session) -> list[str]:
        problems = _need_data(session, self.data) + ([] if _names(self.variables) else ["choose the series"])
        return problems + (["p and q cannot both be 0"] if self.p + self.q == 0 else [])

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:varma", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                         variables=_names(self.variables), p=self.p, q=self.q, span=_span(self.start, self.end))


@command(menu="Estimate/Threshold VAR...")
@dataclass(frozen=True)
class FitThresholdVAR(_CallCommand):
    """A two-regime threshold VAR (Tsay 1998): one VAR when the threshold variable, Delay periods back, is at or
    below the threshold, another above it.

    The threshold is the value that minimises the pooled residual covariance's determinant, searched over the middle
    of the threshold variable's values (Trim leaves out each tail). The window shows the threshold, each regime's
    coefficients and the regime of each period. The likelihood-ratio statistic against a linear VAR has no standard
    distribution, since the threshold is unidentified without one: Test → Threshold VAR linearity (Tsay)…
    bootstraps it.
    """

    maturity: ClassVar[str] = "beta"
    topics: ClassVar[tuple[str, ...]] = ("var", "smooth-transition")

    variables: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the VAR's series")]
    data: Annotated[str, Widget("object", source="Dataset", help="a single time series (not a panel)")] = "data"
    lags: Annotated[int, Widget("number", min=1, max=8, help="lags of every variable, in both regimes")] = 1
    threshold: Annotated[str, Widget("series", depends_on="data", help="the series that splits the regimes (empty: the first variable)")] = ""
    delay: Annotated[int, Widget("number", min=1, max=12, help="the threshold variable is taken this many periods back")] = 1
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "tvar"
    trim: Annotated[float, Widget("number", min=0.05, max=0.45, step=0.05, advanced=True,
                                  help="share of the threshold variable's values left out at each end of the search")] = 0.15

    def validate(self, session: Session) -> list[str]:
        return _need_data(session, self.data) + ([] if _names(self.variables) else ["choose the VAR's series"])

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:threshold_var", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                         variables=_names(self.variables), lags=self.lags, threshold=self.threshold.strip() or None, delay=self.delay,
                         trim=self.trim, span=_span(self.start, self.end))


@command(menu="Test/Threshold VAR linearity (Tsay)...")
@dataclass(frozen=True)
class ThresholdLinearityTest(Command):
    """Tsay's sup-LR test that a threshold VAR is in fact linear (one regime).

    The p-value comes from a residual bootstrap of the linear VAR, with the threshold searched again in each draw
    (Hansen 1996), because the threshold is not identified under the null. A small p-value favours two regimes.
    """

    maturity: ClassVar[str] = "beta"
    topics: ClassVar[tuple[str, ...]] = ("var", "smooth-transition")

    tvar: Annotated[str, Widget("object", source="FittedThresholdVAR", help="a fitted threshold VAR")]
    draws: Annotated[int, Widget("number", advanced=True, min=19, max=10_000, step=100, help="bootstrap draws; each searches the threshold again")] = 200
    name: str = "tsay"
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="random seed of the bootstrap")] = 0

    def validate(self, session: Session) -> list[str]:
        return [] if self.tvar in session.objects else [f"no threshold VAR named {self.tvar!r}"]

    def to_call(self, session: Session) -> Any:
        from ..calls import method_call
        from ..codec import Ref

        return method_call(type(session[self.tvar]), "test", Ref(self.tvar), output=self.name, recorded_by=self.command_id,
                           draws=self.draws, seed=self.seed)

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        seed = f", seed={self.seed}" if self.seed else ""
        return f"{self.name} = {self.tvar}.test(draws={self.draws}{seed})"


@command(menu="Estimate/Bilinear model...")
@dataclass(frozen=True)
class FitBilinear(_CallCommand):
    """The bilinear model y(t) = mu + phi y(t-1) + b y(t-1) e(t-1) + e(t), by conditional least squares.

    The b term lets the effect of last period's shock depend on last period's level, a simple nonlinearity that
    linear ARMA models miss. The window shows the estimates with Gauss-Newton standard errors, and the conditional
    mean against the series; b near 0 means an AR(1) would do.
    """

    maturity: ClassVar[str] = "beta"
    topics: ClassVar[tuple[str, ...]] = ("count-series",)

    variable: Annotated[str, Widget("series", depends_on="data", help="the series modelled")]
    data: Annotated[str, Widget("object", source="Dataset", help="a single time series")] = "data"
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "bl"

    def validate(self, session: Session) -> list[str]:
        return _need_data(session, self.data)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:bilinear", output=self.name, recorded_by=self.command_id, data=Ref(self.data), variable=self.variable,
                         span=_span(self.start, self.end))


# --- tests on a VECM --------------------------------------------------------------------------------------------------

_VECM_TESTS = {"cointegrating vectors": "test_beta", "weak exogeneity": "test_weak_exogeneity", "eigenvalue fluctuation": "fluctuation_test"}


def _h_matrix(text: str) -> list[list[float]]:
    from .builtin import _matrix_text

    rows = _matrix_text(text, "H")
    if not rows:
        raise SpecificationError("give H: a row for each variable (then the restricted constant or trend), e.g. 1 0; -1 0; 0 1")
    if any(v == "*" for row in rows for v in row):
        raise SpecificationError("H: every entry is a number")
    if len({len(r) for r in rows}) > 1:
        raise SpecificationError("H: every row needs the same number of columns")
    return rows


@command(menu="Test/VECM restrictions and stability...")
@dataclass(frozen=True)
class VECMRestrictionTest(Command):
    """Tests on a fitted VECM: Johansen's likelihood-ratio tests of restrictions on the cointegrating vectors and of
    weak exogeneity, and Hansen and Johansen's fluctuation test of the eigenvalues' constancy.

    Cointegrating vectors tests beta = H phi: H has a row for each variable (then one for the restricted constant or
    trend, if the case has one) and a column for each free direction, so 1 0; -1 0; 0 1 says the first two variables
    enter only as their difference. Weak exogeneity tests that the variables named do not adjust to the cointegrating
    relations (their rows of alpha are zero), so they can be conditioned on. Both are chi-squared tests that re-solve
    the reduced-rank regression under the restriction; the VECM must have no exogenous terms or seasonal dummies.

    Eigenvalue fluctuation re-estimates the first rank eigenvalues recursively from Start, with the short run held at
    the full-sample estimates, and compares the largest deviation with bootstrap samples of the fitted VECM: a small
    p-value says the cointegrating relations changed over the sample. Its details hold each eigenvalue's statistic
    and path.
    """

    maturity: ClassVar[str] = "beta"
    topics: ClassVar[tuple[str, ...]] = ("cointegration",)

    vecm: Annotated[str, Widget("object", source="FittedVECM", help="a fitted VECM")]
    test: Annotated[Literal["cointegrating vectors", "weak exogeneity", "eigenvalue fluctuation"],
                    Widget("choice", help="restrictions on beta (H), zero rows of alpha (weak exogeneity), or the eigenvalues' constancy")] = "cointegrating vectors"
    h_matrix: Annotated[str, Widget("multiline", label="H", show_if=("test", ("cointegrating vectors",)),
                             help="beta = H phi: a row per variable (then the restricted constant or trend), columns separated by spaces, e.g. 1 0; -1 0; 0 1")] = ""
    variables: Annotated[str, Widget("text", show_if=("test", ("weak exogeneity",)),
                                     help="the variables tested for weak exogeneity, e.g. li, lm")] = ""
    start: Annotated[str, Widget("period", show_if=("test", ("eigenvalue fluctuation",)),
                                 help="the first period of the recursions (empty: chosen from the sample size)")] = ""
    draws: Annotated[int, Widget("number", advanced=True, min=19, max=10_000, step=100, show_if=("test", ("eigenvalue fluctuation",)),
                                 help="bootstrap samples for the p-value")] = 199
    name: str = "vecm_test"
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, show_if=("test", ("eigenvalue fluctuation",)),
                                help="random seed of the bootstrap")] = 0

    def _arguments(self) -> dict[str, Any]:
        if self.test == "cointegrating vectors":
            return {"H": _h_matrix(self.h_matrix)}
        if self.test == "weak exogeneity":
            if not _names(self.variables):
                raise SpecificationError("name the variables tested for weak exogeneity")
            return {"variables": _names(self.variables)}
        return {"start": self.start.strip() or None, "draws": self.draws, "seed": self.seed}

    def validate(self, session: Session) -> list[str]:
        if self.vecm not in session.objects:
            return [f"no VECM named {self.vecm!r}"]
        try:
            arguments = self._arguments()
        except SpecificationError as exc:
            return [str(exc)]
        unknown = [v for v in arguments.get("variables", ()) if v not in session[self.vecm].variables]
        return [f"{', '.join(unknown)}: not variables of {self.vecm}"] if unknown else []

    def to_call(self, session: Session) -> Any:
        from ..calls import method_call
        from ..codec import Ref

        cls, common = type(session[self.vecm]), {"output": self.name, "recorded_by": self.command_id, **self._arguments()}
        if self.test == "cointegrating vectors":
            return method_call(cls, "test_beta", Ref(self.vecm), **common)
        if self.test == "weak exogeneity":
            return method_call(cls, "test_weak_exogeneity", Ref(self.vecm), **common)
        return method_call(cls, "fluctuation_test", Ref(self.vecm), **common)

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        arguments = {k: v for k, v in self._arguments().items() if (k, v) not in (("start", None), ("draws", 199), ("seed", 0))}
        return f"{self.name} = {self.vecm}.{_VECM_TESTS[self.test]}({', '.join(f'{k}={v!r}' for k, v in arguments.items())})"


# --- spectra and long memory ------------------------------------------------------------------------------------------


@command(menu="Test/Spectrum...")
@dataclass(frozen=True)
class SpectralAnalysis(_CallCommand):
    """The spectrum of a series, or the cross-spectrum of two: how their variance and comovement split by frequency.

    The periodogram is the raw estimate at the Fourier frequencies; the smoothed spectrum averages 2 x Bandwidth + 1
    neighbouring ordinates (a Daniell window), trading resolution for precision. The cross-spectrum gives, by
    frequency, the squared coherence (0 to 1: how closely the two move together at that frequency), the phase (which
    leads) and the gain (how much of Other's movement passes to Variable). Peaks at business-cycle periods, or a
    spectrum rising towards frequency zero (a persistent or long-memory series), are what to look for; the window
    shows the period in time units next to each frequency.
    """

    maturity: ClassVar[str] = "beta"
    topics: ClassVar[tuple[str, ...]] = ("spectral-analysis",)

    variable: Annotated[str, Widget("series", depends_on="data", help="the series; for a cross-spectrum, the one responding (y)")]
    data: Annotated[str, Widget("object", source="Dataset", help="a single time series")] = "data"
    kind: Annotated[Literal["periodogram", "smoothed spectrum", "cross-spectrum"],
                    Widget("choice", help="the raw periodogram, a smoothed spectrum, or the cross-spectrum of the Series (y) and the Other series (x)")] = "smoothed spectrum"
    other: Annotated[str, Widget("series", depends_on="data", show_if=("kind", ("cross-spectrum",)),
                                 help="the second series (x): the gain is the response of Series (y) to it")] = ""
    bandwidth: Annotated[int, Widget("number", min=1, max=200, show_if=("kind", ("smoothed spectrum", "cross-spectrum")),
                                     help="ordinates averaged on each side (Daniell window); wider is smoother")] = 5
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "spec"

    def validate(self, session: Session) -> list[str]:
        problems = _need_data(session, self.data)
        if self.kind == "cross-spectrum" and not self.other.strip():
            problems.append("choose the second series (Other) for the cross-spectrum")
        return problems

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        common = {"output": self.name, "recorded_by": self.command_id, "data": Ref(self.data), "span": _span(self.start, self.end)}
        if self.kind == "periodogram":
            return make_call("pymetrix:periodogram", variable=self.variable, **common)
        if self.kind == "smoothed spectrum":
            return make_call("pymetrix:spectrum", variable=self.variable, bandwidth=self.bandwidth, **common)
        return make_call("pymetrix:cross_spectrum", y=self.variable, x=self.other.strip(), bandwidth=self.bandwidth, **common)


@command(menu="Estimate/Long memory...")
@dataclass(frozen=True)
class LongMemory(_CallCommand):
    """The fractional difference d of a series: the GPH and local Whittle estimators, or an ARFIMA(p, d, q) model.

    d = 0 is short memory; between 0 and 0.5 the series is stationary but its autocorrelations die out slowly;
    between 0.5 and 1 it is mean-reverting but not stationary; 1 is a unit root. GPH (Geweke and Porter-Hudak)
    regresses the log periodogram on the lowest T^Power frequencies; the local Whittle (Robinson) is more efficient.
    Both use only those frequencies, so short-run dynamics bias them less, at the cost of precision: try a few
    powers. ARFIMA estimates d together with ARMA(p, q) short-run dynamics, by conditional sum of squares.
    """

    maturity: ClassVar[str] = "beta"
    topics: ClassVar[tuple[str, ...]] = ("spectral-analysis", "unit-roots")

    variable: Annotated[str, Widget("series", depends_on="data", help="the series")]
    data: Annotated[str, Widget("object", source="Dataset", help="a single time series")] = "data"
    method: Annotated[Literal["GPH", "local Whittle", "ARFIMA"],
                      Widget("choice", help="GPH (log-periodogram regression), local Whittle, or an ARFIMA(p, d, q) model")] = "local Whittle"
    gph_power: Annotated[float, Widget("number", advanced=True, min=0.2, max=0.9, step=0.05, show_if=("method", ("GPH",)),
                                       help="frequencies used: T^power; 0.5 is usual")] = 0.5
    whittle_power: Annotated[float, Widget("number", advanced=True, min=0.2, max=0.9, step=0.05, show_if=("method", ("local Whittle",)),
                                           help="frequencies used: T^power; 0.65 is usual")] = 0.65
    p: Annotated[int, Widget("number", min=0, max=6, show_if=("method", ("ARFIMA",)), help="ARFIMA: autoregressive order")] = 0
    q: Annotated[int, Widget("number", min=0, max=6, show_if=("method", ("ARFIMA",)), help="ARFIMA: moving-average order")] = 0
    start: Annotated[str, Widget("period", group="Sample")] = ""
    end: Annotated[str, Widget("period", group="Sample")] = ""
    name: str = "memory"

    def validate(self, session: Session) -> list[str]:
        return _need_data(session, self.data)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        common = {"output": self.name, "recorded_by": self.command_id, "data": Ref(self.data), "variable": self.variable,
                  "span": _span(self.start, self.end)}
        if self.method == "GPH":
            return make_call("pymetrix:gph", power=self.gph_power, **common)
        if self.method == "local Whittle":
            return make_call("pymetrix:local_whittle", power=self.whittle_power, **common)
        return make_call("pymetrix:arfima", p=self.p, q=self.q, **common)
