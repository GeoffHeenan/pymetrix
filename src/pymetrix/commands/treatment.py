"""Dialogs for treatment effects: difference-in-differences, event studies, regression discontinuity, propensity-score
weighting and synthetic control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session

_TOPICS = ("treatment-effects",)


def _names(text: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in text.replace("\n", ",").split(",") if s.strip())


class _Call(Command):
    call_id: ClassVar[str] = ""

    def _arguments(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call(self.call_id, output=self.name, recorded_by=self.command_id, data=Ref(self.data),  # type: ignore[attr-defined]
                         **self._arguments())

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/Difference-in-differences...")
@dataclass(frozen=True)
class DifferenceInDifferences(_Call):
    """Difference-in-differences: the outcome on a 0/1 treatment (1 where the policy is in force) with unit and
    period fixed effects, errors clustered by unit. The window shows the two groups' means by period.

    The coefficient on the treatment is the average effect on the treated only if, without the policy, treated and
    untreated units would have moved in parallel (parallel trends); the fixed effects remove level differences between
    units and shocks common to all units in a period, nothing more. Check the plot of the two groups' means before
    adoption, or run an event study. With one adoption date the estimate equals the two-by-two difference of means
    (in a balanced panel without controls). When units adopt at different dates and effects change over time, two-way
    fixed effects can weight some comparisons negatively: use Staggered DiD or the Bacon decomposition instead.
    Clustering by unit allows for serial correlation within units, but needs a reasonable number of units.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, did

        panel = Dataset.example("policy_panel")    # 50 states, 2000-2019; d is 1 where the policy is in force
        result = did(panel, "y", "d")
        print(result.summary())    # the effect of d with state and year fixed effects, errors clustered by state
        did(panel, "y", "d", controls=["x"])       # with a time-varying control
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", "the treatment's effect with its clustered standard error, z, p-value and 95% interval"),
        ("Means", ("the mean outcome of treated and controls before and after; with one adoption date the notes show "
                   "the effect as their difference in differences")),
        ("Graph", ("the mean outcome by period of the ever treated and the never treated: look for parallel paths "
                   "before adoption")),
        ("Regression", "the two-way fixed-effects regression behind the effect, with the within R-squared"),
    )

    topics: ClassVar[tuple[str, ...]] = _TOPICS
    call_id: ClassVar[str] = "pymetrix:did"

    outcome: Annotated[str, Widget("series", depends_on="data", help="the series the policy may affect; a level, rate or log, as the effect should be measured")]
    treatment: Annotated[str, Widget("series", depends_on="data", help="1 where and when the policy is in force, else 0")]
    data: Annotated[str, Widget("object", source="Dataset", help="a panel of units and periods")] = "data"
    controls: Annotated[str, Widget("series", depends_on="data", multiple=True, help="time-varying controls; leave out anything the policy itself may change (bad controls)")] = ""
    name: str = "did_result"

    def _arguments(self) -> dict[str, Any]:
        return {"outcome": self.outcome, "treatment": self.treatment, "controls": _names(self.controls)}


@command(menu="Estimate/Event study...")
@dataclass(frozen=True)
class FitEventStudy(_Call):
    """Event study: the effect in each period before and after each unit's adoption, against the period before it,
    with unit and period fixed effects. Effects before adoption near zero support parallel trends.

    Each coefficient is the difference between treated and never-treated units in that period relative to adoption,
    measured against the reference period (shown as zero). Periods beyond the window are gathered into its end points,
    so the first and last coefficients average everything earlier or later. Leads (negative periods) near zero are
    consistent with parallel trends but do not prove them, and they have little power against gradual divergence.
    Lags trace how the effect builds up or fades. With adoption at different dates and effects that change over time,
    these two-way fixed effects coefficients can mix comparisons (Sun and Abraham 2021): prefer Staggered DiD then.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, event_study

        panel = Dataset.example("policy_panel")    # 50 states, 2000-2019; d is 1 where the policy is in force
        es = event_study(panel, "y", "d")          # periods -5 to 5 around adoption, against period -1
        print(es.summary())        # an effect per period; leads (negative periods) near zero support parallel trends
        event_study(panel, "y", "d", window=(-3, 3), controls=["x"])
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", ("an effect per period relative to adoption, with standard error, z, p-value and 95% interval; the "
                     "reference period is zero")),
        ("Leads (negative periods)", ("the pre-trend check: near zero and inside their intervals is consistent with "
                                      "parallel trends, but with little power")),
        ("Graph", "the effects with their 95% band, a zero line and a dashed line at adoption"),
        ("es.fit", ("the regression; es.fit.test('wald', 'rel_m5 = 0, rel_m4 = 0, rel_m3 = 0, rel_m2 = 0') tests "
                    "the leads jointly")),
    )

    topics: ClassVar[tuple[str, ...]] = _TOPICS
    call_id: ClassVar[str] = "pymetrix:event_study"

    outcome: Annotated[str, Widget("series", depends_on="data", help="the series the policy may affect")]
    treatment: Annotated[str, Widget("series", depends_on="data", help="1 where and when the policy is in force, else 0")]
    data: Annotated[str, Widget("object", source="Dataset", help="a panel of units and periods")] = "data"
    first: Annotated[int, Widget("number", min=-40, max=-1, help="earliest lead estimated; earlier periods are pooled into it. A few leads test pre-trends")] = -5
    last: Annotated[int, Widget("number", min=0, max=40, help="latest lag estimated; later periods are pooled into it. Long enough to see the effect settle")] = 5
    controls: Annotated[str, Widget("series", depends_on="data", multiple=True, help="time-varying controls; leave out anything the policy itself may change (bad controls)")] = ""
    name: str = "event"
    reference: Annotated[int, Widget("number", min=-40, max=-1, advanced=True,
                                     help="the period before adoption the effects are measured against (left out)")] = -1

    def _arguments(self) -> dict[str, Any]:
        extra = {"reference": self.reference} if self.reference != -1 else {}
        return {"outcome": self.outcome, "treatment": self.treatment, "window": (self.first, self.last), "controls": _names(self.controls),
                **extra}


@command(menu="Estimate/Staggered DiD...", id="pymetrix:fit_staggered_did")
@dataclass(frozen=True)
class FitStaggeredDiD(_Call):
    """Staggered difference-in-differences: each adoption cohort compared only with units not (yet) treated.

    Callaway and Sant'Anna's group-time effects (against the never treated, or the not yet treated), aggregated by time
    since adoption, by cohort and overall, with a bootstrap over units; or Sun and Abraham's interaction-weighted event
    study. Use these rather than two-way fixed effects when units adopt at different times and effects may change.
    """

    topics: ClassVar[tuple[str, ...]] = ("staggered-did",)
    call_id: ClassVar[str] = "pymetrix:staggered_did"

    outcome: Annotated[str, Widget("series", depends_on="data", help="the series the policy may affect")]
    treatment: Annotated[str, Widget("series", depends_on="data", help="0 before a unit adopts, 1 from then on")]
    data: Annotated[str, Widget("object", source="Dataset", help="a balanced panel of units and periods")] = "data"
    method: Annotated[Literal["callaway_santanna", "sun_abraham"], Widget("choice", help="Callaway and Sant'Anna (group-time effects, aggregated) or Sun and Abraham (interaction-weighted event study)")] = "callaway_santanna"
    control: Annotated[Literal["never", "notyet"], Widget("choice", help="the comparison units: never treated, or not yet treated")] = "never"
    draws: Annotated[int, Widget("number", min=0, max=10_000, step=100, advanced=True, help="bootstrap draws over units for Callaway and Sant'Anna's standard errors; more give steadier ones")] = 499
    covariates: Annotated[str, Widget("series", depends_on="data", multiple=True,
                                      help="for conditional parallel trends: each unit's values in the first period (Callaway and Sant'Anna)")] = ""
    estimator: Annotated[Literal["dr", "or", "ipw"], Widget("choice", advanced=True,
                                                            help="with covariates: doubly robust, outcome regression or inverse probability weighting")] = "dr"
    name: str = "staggered"
    first: Annotated[int | None, Widget("number", min=-40, max=-1, advanced=True,
                                        help="the earliest period before adoption shown; empty = all")] = None
    last: Annotated[int | None, Widget("number", min=0, max=40, advanced=True, help="the latest period after adoption shown; empty = all")] = None
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="random seed for the bootstrap")] = 0

    def validate(self, session: Session) -> list[str]:
        if (self.first is None) != (self.last is None):
            return ["give both the earliest and the latest period shown, or neither"]
        return []

    def _arguments(self) -> dict[str, Any]:
        extra: dict[str, Any] = {"covariates": list(_names(self.covariates)), "estimator": self.estimator} if self.covariates.strip() else {}
        if self.first is not None and self.last is not None:
            extra["window"] = (self.first, self.last)
        if self.seed:
            extra["seed"] = self.seed
        return {"outcome": self.outcome, "treatment": self.treatment, "method": self.method, "control": self.control, "draws": self.draws,
                **extra}


@command(menu="Estimate/Bacon decomposition...")
@dataclass(frozen=True)
class BaconDecompose(_Call):
    """Goodman-Bacon decomposition: the two-way fixed effects DiD estimate as a weighted average of its 2x2 comparisons.

    It shows how much of the estimate comes from comparing later adopters with units already treated, which misleads
    when effects grow over time.
    """

    topics: ClassVar[tuple[str, ...]] = ("staggered-did",)
    call_id: ClassVar[str] = "pymetrix:bacon_decomposition"

    outcome: Annotated[str, Widget("series", depends_on="data", help="the series the policy may affect; as in the two-way fixed effects DiD being decomposed")]
    treatment: Annotated[str, Widget("series", depends_on="data", help="0 before a unit adopts, 1 from then on")]
    data: Annotated[str, Widget("object", source="Dataset", help="a balanced panel of units and periods")] = "data"
    name: str = "bacon"

    def _arguments(self) -> dict[str, Any]:
        return {"outcome": self.outcome, "treatment": self.treatment}


@command(menu="Estimate/Regression discontinuity...")
@dataclass(frozen=True)
class RegressionDiscontinuity(_Call):
    """Sharp regression discontinuity: the jump in the outcome where the running variable reaches the cutoff, by
    local linear regression with a triangular kernel; the bandwidth by cross-validation unless given.

    The jump identifies the effect at the cutoff only, for units whose score is near it, and only if nothing else
    changes discontinuously there and units cannot sort precisely to one side (check for bunching in the running
    variable). The design is sharp: everyone at or above the cutoff is treated. The bandwidth trades bias (too wide)
    against variance (too narrow); the second table re-estimates at half to twice the chosen bandwidth, and a jump
    that moves a lot across it is fragile. Standard errors are robust (HC1) but conventional, not bias-corrected, so
    intervals at a cross-validated bandwidth can be too narrow. The plot shows binned means either side of the cutoff
    with the local linear fits.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS
    call_id: ClassVar[str] = "pymetrix:rdd"

    outcome: Annotated[str, Widget("series", depends_on="data", help="the series measured either side of the cutoff")]
    running: Annotated[str, Widget("series", depends_on="data", help="the running variable (the score)")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    cutoff: Annotated[float, Widget("number", help="value of the running variable at which treatment switches on; units at or above it are treated")] = 0.0
    bandwidth: Annotated[float, Widget("number", min=0.0, help="half-width of the window around the cutoff; wider is less noisy but more biased. 0 = cross-validation")] = 0.0
    name: str = "rd"

    def _arguments(self) -> dict[str, Any]:
        return {"outcome": self.outcome, "running": self.running, "cutoff": self.cutoff, "bandwidth": self.bandwidth or None}


@command(menu="Estimate/Propensity score weighting...")
@dataclass(frozen=True)
class PropensityWeighting(_Call):
    """Average treatment effects under selection on observables: inverse probability weighting on a logit
    propensity score, or doubly robust (AIPW). Bootstrap standard errors; the window shows the overlap.

    Identification rests on unconfoundedness: given the covariates, treatment is as good as random, so every
    confounder must be among them; no weighting can fix an omitted one. It also needs overlap: every unit must have
    some chance of being treated and of not being. Check the histogram of propensity scores: scores near 0 or 1 give
    a few units huge weights and an unstable estimate (the summary warns of this). IPW is consistent if the logit for
    the propensity score is right; AIPW adds linear regressions of the outcome on the covariates and is consistent if
    either model is right. The bootstrap re-estimates the propensity score in each replication.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS
    call_id: ClassVar[str] = "pymetrix:treatment_effect"

    outcome: Annotated[str, Widget("series", depends_on="data", help="the series whose average the treatment may change")]
    treatment: Annotated[str, Widget("series", depends_on="data", help="0/1 indicator of treatment; both groups must be present")]
    covariates: Annotated[str, Widget("series", depends_on="data", multiple=True, help="pre-treatment confounders the logit uses; all of them must be measured, none affected by treatment")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    estimand: Annotated[Literal["ATE", "ATT"], Widget("choice", help="the average effect on everyone (ATE) or on the treated (ATT)")] = "ATE"
    method: Annotated[Literal["ipw", "aipw"], Widget("choice", help="ipw: weighting only; aipw: adds outcome regressions, consistent if either model is right")] = "ipw"
    reps: Annotated[int, Widget("number", min=0, max=5000, advanced=True, help="bootstrap resamples for the standard error; 200 is fine for a rough s.e., use more for publication")] = 200
    name: str = "effect"
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="random seed for the bootstrap")] = 0

    def _arguments(self) -> dict[str, Any]:
        return {"outcome": self.outcome, "treatment": self.treatment, "covariates": _names(self.covariates), "estimand": self.estimand,
                "method": self.method, "reps": self.reps, **({"seed": self.seed} if self.seed else {})}


@command(menu="Estimate/Synthetic DiD...", id="pymetrix:fit_synthetic_did")
@dataclass(frozen=True)
class FitSyntheticDiD(_Call):
    """Synthetic difference-in-differences: unit and time weights make DiD's parallel trends plausible (Arkhangelsky et al.).

    Control units are weighted so that their average tracks the treated units' path before treatment, and the
    pre-treatment periods are weighted towards those most like the treated periods; a DiD on the weighted data then
    gives the average effect on the treated. It needs a balanced panel in which every treated unit adopts at the same
    period and stays treated; for staggered adoption use Staggered DiD. The jackknife standard error needs at least
    two treated units; with one, the placebo method reassigns treatment at random among the controls, which needs more
    controls than treated units and assumes the controls' errors resemble the treated ones'.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS
    call_id: ClassVar[str] = "pymetrix:synthetic_did"

    outcome: Annotated[str, Widget("series", depends_on="data", help="the series the policy may affect")]
    treatment: Annotated[str, Widget("series", depends_on="data", help="0/1, every treated unit switching on at the same period")]
    data: Annotated[str, Widget("object", source="Dataset", help="a balanced panel of units and periods")] = "data"
    name: str = "sdid"
    se: Annotated[Literal["auto", "jackknife", "placebo"], Widget("choice", advanced=True,
                                                                  help="the standard error: jackknife, placebo, or automatic (jackknife with "
                                                                       "two or more treated units, else placebo)")] = "auto"
    placebos: Annotated[int, Widget("number", min=10, max=10_000, step=50, advanced=True, help="random placebo assignments for the placebo s.e.; more gives a steadier s.e. but takes longer")] = 200
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="random seed for the placebos")] = 0

    def _arguments(self) -> dict[str, Any]:
        extra = {k: getattr(self, k) for k, d in (("se", "auto"), ("placebos", 200), ("seed", 0)) if getattr(self, k) != d}
        return {"outcome": self.outcome, "treatment": self.treatment, **extra}


@command(menu="Estimate/Synthetic control...")
@dataclass(frozen=True)
class FitSyntheticControl(_Call):
    """Synthetic control: one treated unit against a weighted average of untreated units matching its path before
    treatment; in-space placebos give a p-value.

    The donor weights are non-negative and sum to one, chosen so that the synthetic unit reproduces the treated unit's
    outcome before the start; the gap afterwards is the estimated effect. It is credible only if the pre-treatment
    fit is close (see the RMSPE before) and the donors were not affected by the policy or similar shocks, so drop
    donors that were. Inference is by in-space placebos: each donor is treated in turn, and the p-value is the
    treated unit's rank in the ratio of post- to pre-treatment RMSPE; with few donors the smallest attainable p-value
    is large. The lower plot shows the treated unit's gap against the placebo gaps.
    """

    topics: ClassVar[tuple[str, ...]] = _TOPICS
    call_id: ClassVar[str] = "pymetrix:synthetic_control"

    outcome: Annotated[str, Widget("series", depends_on="data", help="the series whose pre-treatment path the donor weights match")]
    treated: Annotated[str, Widget("text", help="the one treated unit, as named on the panel's unit axis")]
    start: Annotated[str, Widget("text", help="first period the policy applies, e.g. 2005; the weights are fitted on the periods before it")]
    data: Annotated[str, Widget("object", source="Dataset", help="a panel of units and periods")] = "data"
    donors: Annotated[str, Widget("text", help="the donor units, comma-separated (empty: every other unit)")] = ""
    name: str = "synth"

    def _arguments(self) -> dict[str, Any]:
        treated: Any = int(self.treated) if self.treated.strip().lstrip("-").isdigit() else self.treated
        donors = tuple(int(d) if d.lstrip("-").isdigit() else d for d in _names(self.donors)) or None
        return {"outcome": self.outcome, "treated": treated, "start": self.start, "donors": donors}
