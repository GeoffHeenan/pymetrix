"""Dialogs for applied microeconometrics: double machine learning, shift-share IV, interactive fixed effects and
conditional logit (fixed effects for panels, McFadden's for choices)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from ..errors import SpecificationError
from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.replace("\n", ",").split(",") if s.strip()]


def _pairs(text: str) -> dict[str, str]:
    """"a: 1, b: 2" as {"a": "1", "b": "2"}."""
    out = {}
    for part in _names(text):
        key, sep, value = part.partition(":")
        if not sep:
            raise SpecificationError(f"{part!r} is not 'name: value'")
        out[key.strip()] = value.strip()
    return out


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


_LEARNER_HELP = ("ols: least squares (a logit for a propensity score), no extra library; lasso, forest, boosting: scikit-learn "
                 "(pip install pymetrix[ml]); lightgbm, xgboost: LightGBM or XGBoost (pip install pymetrix[lightgbm], pymetrix[xgboost])")


@command(menu="Estimate/Double machine learning...")
@dataclass(frozen=True)
class DoubleMachineLearning(_Call):
    """Double/debiased machine learning: the effect of a treatment with many controls, learned flexibly and cross-fitted.

    The controls enter through nuisance functions (the outcome's and the treatment's conditional means, and in the
    interactive model the propensity score) learned by the chosen learners on the other folds; the effect then solves
    a Neyman-orthogonal score, so mistakes in those functions affect it only at second order. The effect is causal
    only if the controls remove all confounding (or, for the partially linear IV model, the instrument is valid given
    them). The partially linear model assumes a constant effect; the interactive model allows any heterogeneity for a
    0/1 treatment and needs overlap (propensities away from 0 and 1: see the trimming). Repeating the sample split and
    taking the median makes the estimate depend less on one random split. With OLS learners the partially linear
    estimate is a cross-fitted Frisch-Waugh-Lovell regression; without cross-fitting (one fold) it is exactly OLS.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, double_ml

        data = Dataset.example("card")      # wages, schooling and family background
        fit = double_ml(data, "lwage", "educ", ["exper", "expersq", "black", "south", "smsa", "fatheduc", "motheduc"])
        print(fit.summary())                # the partially linear coefficient on schooling, cross-fitted
        double_ml(data, "lwage", "educ", ["exper", "expersq", "black", "south", "smsa"], model="pliv", instrument="nearc4")
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", "the effect with its standard error, z, p-value and 95% interval, and the learners and folds used"),
        ("Graph", "each sample split's estimate and interval, and the median aggregate"),
        ("Splits", "the estimate and standard error from each repetition of the sample split"),
        ("Nuisance fit", "the out-of-fold root mean squared error of each nuisance function: compare learners by it"),
    )
    topics: ClassVar[tuple[str, ...]] = ("double-ml",)
    call_id: ClassVar[str] = "pymetrix:double_ml"

    outcome: Annotated[str, Widget("series", depends_on="data", help="the outcome Y")]
    treatment: Annotated[str, Widget("series", depends_on="data", help="the treatment D (0/1 for the interactive model)")]
    controls: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the controls X the nuisance functions depend on: confounders, measured before treatment")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    model: Annotated[Literal["plr", "pliv", "irm"], Widget("choice", help="plr: partially linear; pliv: partially linear IV (needs an instrument); irm: interactive, a 0/1 treatment")] = "plr"
    instrument: Annotated[str, Widget("series", depends_on="data", help="the instrument Z (partially linear IV only)")] = ""
    score: Annotated[Literal["ATE", "ATTE"], Widget("choice", help="interactive model: the average effect (ATE) or the average effect on the treated (ATTE)")] = "ATE"
    learner: Annotated[Literal["ols", "lasso", "forest", "boosting", "lightgbm", "xgboost"], Widget("choice", help=_LEARNER_HELP)] = "ols"
    learner_treatment: Annotated[Literal["", "ols", "lasso", "forest", "boosting", "lightgbm", "xgboost"], Widget("choice", advanced=True, help="the learner for the treatment (and instrument); empty: the same as the outcome's")] = ""
    folds: Annotated[int, Widget("number", advanced=True, min=1, max=20, help="cross-fitting folds; 1 fits the nuisance functions on the whole sample (no cross-fitting)")] = 5
    repeats: Annotated[int, Widget("number", advanced=True, min=1, max=100, help="repetitions of the sample split, aggregated by the median")] = 1
    cluster: Annotated[str, Widget("level", depends_on="data", advanced=True, help="a series or axis to cluster the standard errors (and keep together in folds) by; empty: none")] = ""
    trim: Annotated[float, Widget("number", min=0.0, max=0.2, step=0.01, advanced=True, help="interactive model: propensity scores are kept within [trim, 1 - trim]")] = 0.01
    seed: Annotated[int, Widget("number", min=0, max=2**31 - 1, advanced=True, help="random seed for the folds and the learners")] = 0
    name: str = "dml"

    def validate(self, session: Session) -> list[str]:
        if self.model == "pliv" and not self.instrument:
            return ["the partially linear IV model needs an instrument"]
        if self.model != "pliv" and self.instrument:
            return ["an instrument goes with the partially linear IV model"]
        return []

    def _arguments(self) -> dict[str, Any]:
        out: dict[str, Any] = {"outcome": self.outcome, "treatment": self.treatment, "controls": _names(self.controls)}
        if self.model != "plr":
            out["model"] = self.model
        if self.instrument:
            out["instrument"] = self.instrument
        if self.model == "irm" and self.score != "ATE":
            out["score"] = self.score
        if self.learner != "ols":
            out["learner"] = self.learner
        if self.learner_treatment:
            out["learner_treatment"] = self.learner_treatment
        for key, default in (("folds", 5), ("repeats", 1), ("trim", 0.01), ("seed", 0)):
            if getattr(self, key) != default:
                out[key] = getattr(self, key)
        if self.cluster:
            out["cluster"] = self.cluster
        return out


@command(menu="Estimate/Shift-share IV...")
@dataclass(frozen=True)
class ShiftShareIV(_Call):
    """Shift-share (Bartik) IV: an endogenous treatment instrumented by regions' exposure shares times industry shocks.

    Regions exposed to the same industries share their shocks, so their errors can be correlated in ways
    region-level robust errors miss: the AKM and BHJ standard errors allow for it (and for clusters of related
    industries), and AKM0 inverts the test with the null imposed, which behaves better with few large shocks or a
    weak instrument. Identification can come from the shocks being as good as random (Borusyak, Hull and Jaravel; the
    shock-level regression shows it) or from the shares being exogenous (Goldsmith-Pinkham, Sorkin and Swift): the
    Rotemberg weights say which industries' shares drive the estimate, to check those first. The 2SLS includes a
    constant and the controls; with shares that do not sum to one, control for their sum.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        import numpy as np
        import pandas as pd
        from pymetrix import Dataset, shift_share

        rng = np.random.default_rng(0)      # 200 regions' employment shares in 5 industries, and the industries' shocks
        shares = pd.DataFrame(rng.dirichlet(np.ones(5), size=200), columns=["agr", "man", "con", "trd", "svc"])
        shocks = dict(zip(shares.columns, rng.normal(size=5)))
        dx = shares.to_numpy() @ list(shocks.values()) + rng.normal(size=200)
        regions = Dataset.from_frame(shares.assign(dx=dx, dy=0.5 * dx + rng.normal(size=200), region=range(200)), entity="region")
        ss = shift_share(regions, "dy", "dx", list(shocks), shocks)
        print(ss.summary())             # robust, AKM, BHJ standard errors and the AKM0 interval
        print(ss.rotemberg_table())     # the industries that drive the estimate
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", "the 2SLS estimate with robust, AKM and BHJ standard errors, the AKM0 interval and the first stage"),
        ("Rotemberg weights", "each industry's weight in the estimate and its just-identified estimate; weights sum to one"),
        ("Shock level", "Borusyak, Hull and Jaravel's equivalent IV across industries: the same coefficient"),
    )
    topics: ClassVar[tuple[str, ...]] = ("shift-share",)
    call_id: ClassVar[str] = "pymetrix:shift_share"

    outcome: Annotated[str, Widget("series", depends_on="data", help="the regional outcome")]
    treatment: Annotated[str, Widget("series", depends_on="data", help="the endogenous regional treatment the instrument moves")]
    shares: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the exposure shares, one series per industry")]
    shocks: Annotated[str, Widget("text", help="one shock per share: 'share: value, ...', or values in the order of the shares")]
    data: Annotated[str, Widget("object", source="Dataset", help="one row per region")] = "data"
    controls: Annotated[str, Widget("series", depends_on="data", multiple=True, help="regional controls (a constant is always included); add the sum of shares when they do not sum to one")] = ""
    weights: Annotated[str, Widget("series", depends_on="data", advanced=True, help="regression weights, e.g. population; empty: equal")] = ""
    shock_clusters: Annotated[str, Widget("text", advanced=True, help="clusters of industries for the AKM and BHJ errors: 'share: cluster, ...'; empty: none")] = ""
    se: Annotated[Literal["akm", "bhj", "robust"], Widget("choice", advanced=True, help="the standard error the coefficient carries (all are shown)")] = "akm"
    name: str = "bartik"

    def _shocks(self) -> dict[str, float]:
        shares = _names(self.shares)
        if ":" in self.shocks:
            return {k: float(v) for k, v in _pairs(self.shocks).items()}
        values = [float(v) for v in _names(self.shocks)]
        if len(values) != len(shares):
            raise SpecificationError(f"{len(values)} shocks for {len(shares)} shares")
        return dict(zip(shares, values))

    def validate(self, session: Session) -> list[str]:
        try:
            self._shocks()
            _pairs(self.shock_clusters)
        except ValueError as exc:  # SpecificationError, or a value that is not a number
            return [f"shocks and clusters are 'share: value' pairs or numbers: {exc}"]
        return []

    def _arguments(self) -> dict[str, Any]:
        out: dict[str, Any] = {"outcome": self.outcome, "treatment": self.treatment, "shares": _names(self.shares), "shocks": self._shocks()}
        if self.controls.strip():
            out["controls"] = _names(self.controls)
        if self.weights:
            out["weights"] = self.weights
        if self.shock_clusters.strip():
            out["shock_clusters"] = _pairs(self.shock_clusters)
        if self.se != "akm":
            out["se"] = self.se
        return out


@command(menu="Estimate/Interactive fixed effects...")
@dataclass(frozen=True)
class FitInteractiveFE(_Call):
    """Interactive fixed effects (Bai 2009): a panel regression with unobserved common factors whose loadings differ by unit.

    Two-way fixed effects allow for a level per unit and a shock per period common to all units; interactive effects
    let each unit respond differently to common shocks (lambda_i' F_t), and stay consistent when the regressors are
    correlated with them. The factors are estimated with the coefficients by iterating least squares and principal
    components; the number of factors can be chosen by Bai and Ng's criteria. It needs a balanced panel with both
    many units and many periods; with few periods the estimates are biased in 1/T, which the bias correction reduces
    for heteroskedastic errors. Coefficients on regressors that are constant within units or periods, or are
    themselves low-rank, are not identified.
    """

    maturity: ClassVar[str] = "experimental"
    example: ClassVar[str] = """
        from pymetrix import Dataset, interactive_fe

        panel = Dataset.example("grunfeld")     # 10 firms, 20 years
        ife = interactive_fe(panel, "invest", ["value", "capital"], factors=1, effects="twoway")
        print(ife.summary())
        interactive_fe(panel, "invest", ["value", "capital"], factors="auto", max_factors=3)   # Bai and Ng's ICp1
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", "the coefficients with Bai's standard errors"),
        ("Graph", "the estimated factors over time"),
        ("Factors and Loadings", "the factors by period and each unit's loadings"),
        ("Number of factors", "Bai and Ng's criteria for 0 to the maximum number of factors, when they chose it"),
    )
    topics: ClassVar[tuple[str, ...]] = ("interactive-fixed-effects",)
    call_id: ClassVar[str] = "pymetrix:interactive_fe"

    outcome: Annotated[str, Widget("series", depends_on="data", help="the dependent series")]
    regressors: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the regressors; they may be correlated with the factors")]
    data: Annotated[str, Widget("object", source="Dataset", help="a balanced panel of units and periods")] = "data"
    factors: Annotated[int, Widget("number", min=0, max=20, help="the number of common factors (ignored when chosen by criteria)")] = 1
    choose: Annotated[bool, Widget("checkbox", help="choose the number of factors by Bai and Ng's criterion")] = False
    effects: Annotated[Literal["none", "unit", "time", "twoway"], Widget("choice", help="additive effects removed first: none (a constant), unit, time or both")] = "none"
    cov: Annotated[Literal["robust", "homoskedastic", "cluster"], Widget("choice", help="standard errors: robust to heteroskedasticity, homoskedastic, or clustered by unit")] = "robust"
    bias_correction: Annotated[bool, Widget("checkbox", advanced=True, help="subtract Bai's estimated bias for heteroskedastic errors (B/N + C/T)")] = False
    max_factors: Annotated[int, Widget("number", min=1, max=20, advanced=True, help="the most factors the criteria consider")] = 8
    criterion: Annotated[Literal["icp1", "icp2", "icp3", "pcp1", "pcp2", "pcp3"], Widget("choice", advanced=True, help="Bai and Ng's criterion")] = "icp1"
    name: str = "ife"

    def _arguments(self) -> dict[str, Any]:
        out: dict[str, Any] = {"outcome": self.outcome, "regressors": _names(self.regressors), "factors": "auto" if self.choose else self.factors}
        if self.effects != "none":
            out["effects"] = self.effects
        if self.cov != "robust":
            out["cov"] = self.cov
        if self.bias_correction:
            out["bias_correction"] = True
        if self.choose:
            if self.max_factors != 8:
                out["max_factors"] = self.max_factors
            if self.criterion != "icp1":
                out["criterion"] = self.criterion
        return out


@command(menu="Estimate/Conditional logit (fixed effects)...")
@dataclass(frozen=True)
class FitConditionalLogit(_Call):
    """Chamberlain's conditional logit: a 0/1 outcome on regressors with a fixed effect per unit, conditioned away.

    Conditioning on each unit's number of ones removes its fixed effect from the likelihood, so the coefficients are
    consistent with few periods, where a logit with unit dummies is not (the incidental parameters problem). Only
    units whose outcome changes contribute; regressors that do not vary within units are not identified. The
    coefficients are log odds ratios within a unit; marginal effects on probabilities are not identified because they
    depend on the fixed effects.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        from pymetrix import Dataset, conditional_logit

        panel = Dataset.example("wage_panel")       # 545 men, 1980-87
        fit = conditional_logit(panel, "union", ["married", "exper"])
        print(fit.summary())                         # log odds ratios, men whose membership changes
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", "coefficients (log odds ratios) with standard errors; the groups used and dropped"),
        ("Odds ratios", "exp(coefficient) with 95% intervals"),
    )
    topics: ClassVar[tuple[str, ...]] = ("conditional-logit",)
    call_id: ClassVar[str] = "pymetrix:conditional_logit"

    outcome: Annotated[str, Widget("series", depends_on="data", help="the 0/1 outcome")]
    regressors: Annotated[str, Widget("series", depends_on="data", multiple=True, help="regressors that vary within units")]
    data: Annotated[str, Widget("object", source="Dataset", help="a panel (or grouped data)")] = "data"
    group: Annotated[str, Widget("level", depends_on="data", advanced=True, help="the groups whose fixed effects are conditioned away; empty: the panel's units")] = ""
    cov: Annotated[Literal["hessian", "cluster"], Widget("choice", help="standard errors from the inverse Hessian, or clustered by group")] = "hessian"
    name: str = "clogit"

    def _arguments(self) -> dict[str, Any]:
        out: dict[str, Any] = {"outcome": self.outcome, "regressors": _names(self.regressors)}
        if self.group:
            out["group"] = self.group
        if self.cov != "hessian":
            out["cov"] = self.cov
        return out


@command(menu="Estimate/Choice model (conditional logit)...")
@dataclass(frozen=True)
class FitChoiceLogit(_Call):
    """McFadden's conditional logit: which alternative each decision maker chooses, from attributes of the alternatives.

    The data are long: one row per case and alternative it could choose, with a 0/1 series marking the choice.
    Attributes that vary across alternatives (price, time) have one coefficient each; alternative constants and
    characteristics of the decision maker (income) need one coefficient per alternative but the base. Choice sets may
    differ across cases. The model assumes independence of irrelevant alternatives: the odds between two alternatives
    do not depend on what else is available.
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        import numpy as np
        import pandas as pd
        from pymetrix import Dataset, choice_logit

        rng = np.random.default_rng(0)      # 300 trips by bus, car or train: cost by mode, income by trip
        cost, income = rng.normal(5, 2, (300, 3)), rng.normal(3, 1, (300, 1))
        utility = -0.5 * cost + [0.0, -1.0, 0.2] + [0.0, 0.4, 0.0] * income + rng.gumbel(size=(300, 3))
        trips = Dataset.from_frame(pd.DataFrame({
            "trip": np.repeat(range(300), 3), "mode": ["bus", "car", "train"] * 300, "cost": cost.ravel(),
            "income": np.repeat(income, 3), "chosen": (utility == utility.max(1, keepdims=True)).ravel() * 1.0}), entity=["trip", "mode"])
        fit = choice_logit(trips, "chosen", ["cost"], case="trip", alternative="mode",
                           alternative_constants=True, case_variables=["income"])
        print(fit.summary())
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", "coefficients with standard errors, the log likelihood and McFadden's pseudo R-squared"),
        ("Probabilities", "each alternative's fitted probability in each case"),
    )
    topics: ClassVar[tuple[str, ...]] = ("conditional-logit",)
    call_id: ClassVar[str] = "pymetrix:choice_logit"

    choice: Annotated[str, Widget("series", depends_on="data", help="1 for the chosen alternative, 0 for the others")]
    regressors: Annotated[str, Widget("series", depends_on="data", multiple=True, help="attributes that vary across alternatives")]
    data: Annotated[str, Widget("object", source="Dataset", help="long choice data: one row per case and alternative")] = "data"
    case: Annotated[str, Widget("level", depends_on="data", help="the series or axis identifying each choice situation; empty: the first entity axis")] = ""
    alternative: Annotated[str, Widget("level", depends_on="data", help="the series or axis naming the alternatives; empty: the second entity axis")] = ""
    alternative_constants: Annotated[bool, Widget("checkbox", help="a constant for each alternative but the base")] = False
    case_variables: Annotated[str, Widget("series", depends_on="data", multiple=True, help="characteristics of the decision maker, one coefficient per alternative but the base")] = ""
    base: Annotated[str, Widget("text", advanced=True, help="the base alternative; empty: the first")] = ""
    cov: Annotated[Literal["hessian", "cluster"], Widget("choice", advanced=True, help="standard errors from the inverse Hessian, or clustered by case")] = "hessian"
    name: str = "choice"

    def _arguments(self) -> dict[str, Any]:
        out: dict[str, Any] = {"choice": self.choice, "regressors": _names(self.regressors)}
        for key in ("case", "alternative"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        if self.alternative_constants:
            out["alternative_constants"] = True
        if self.case_variables.strip():
            out["case_variables"] = _names(self.case_variables)
        if self.base:
            out["base"] = self.base
        if self.cov != "hessian":
            out["cov"] = self.cov
        return out


@command(menu="Estimate/Conditional logit marginal effects...")
@dataclass(frozen=True)
class ClogitMarginalEffects(Command):
    """Marginal effects and elasticities after a conditional logit: how the choice probabilities move with each variable.

    After McFadden's choice logit: the derivative of each alternative's probability in each attribute of each
    alternative (own effects on the diagonal, cross effects off it) and in each characteristic of the decision maker,
    with the same as elasticities, averaged over the cases or at the means, with delta-method standard errors. After
    Chamberlain's panel logit the probabilities depend on the fixed effects, which are conditioned away and not
    estimated, so there are no marginal effects: the result gives odds ratios, and, if you give a value for the fixed
    effect, the effects for a unit with that fixed effect (0 is Stata's pu0), labelled as such.

    Discrete changes: tick 0/1 variables for the change in the probabilities when each variable whose values are all
    0 or 1 goes from 0 to 1 (for an attribute, one alternative's at a time: own and cross changes), and list others
    under Changes with the amount, e.g. cost: 1, income: 0.5, for the change from each case's value to that value plus
    the amount. They are computed from the probabilities themselves, with delta-method standard errors, and replace
    the derivatives in the summary (the derivatives stay in the tables).
    """

    maturity: ClassVar[str] = "beta"
    example: ClassVar[str] = """
        import numpy as np
        import pandas as pd
        from pymetrix import Dataset, choice_logit, clogit_effects

        rng = np.random.default_rng(0)      # 300 trips by bus, car or train
        cost = rng.normal(5, 2, (300, 3))
        utility = -0.5 * cost + [0.0, -1.0, 0.2] + rng.gumbel(size=(300, 3))
        trips = Dataset.from_frame(pd.DataFrame({
            "trip": np.repeat(range(300), 3), "mode": ["bus", "car", "train"] * 300, "cost": cost.ravel(),
            "chosen": (utility == utility.max(1, keepdims=True)).ravel() * 1.0}), entity=["trip", "mode"])
        fit = choice_logit(trips, "chosen", ["cost"], alternative_constants=True)
        me = clogit_effects(fit)            # averaged over the trips
        print(me.table("cost"))             # dP(row)/dcost(column)
        print(me.table("cost", "elasticity"))
    """
    results: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Summary", "own effects of each attribute and the effects of each characteristic (choice); odds ratios (panel)"),
        ("Graph", "own elasticities with intervals (choice); odds ratios (panel)"),
        ("Effects and Elasticities", "for each attribute, alternatives by alternatives: the change in P(row) for the attribute of the column"),
        ("All", "every effect with its standard error, z, p-value and interval"),
    )
    topics: ClassVar[tuple[str, ...]] = ("conditional-logit",)

    fit: Annotated[str, Widget("object", source="FittedConditionalLogit", help="a conditional logit or choice logit")]
    at: Annotated[Literal["average", "mean"], Widget("choice", help="average the effects over the cases, or evaluate them at the means")] = "average"
    fixed_effect: Annotated[str, Widget("text", advanced=True, help="panel logit only: a value of the fixed effect to show effects at (e.g. 0); empty: odds ratios only")] = ""
    dummy: Annotated[bool, Widget("checkbox", label="0/1 variables", help="discrete changes from 0 to 1 for every variable whose values are all 0 or 1")] = False
    changes: Annotated[str, Widget("text", help="discrete changes by an amount, e.g. cost: 1, income: 0.5 (from each value to the value plus the amount)")] = ""
    name: str = "effects"

    def _changes(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for part in self.changes.replace(";", ",").split(","):
            if not part.strip():
                continue
            name, sep, amount = part.partition(":")
            if not sep or not name.strip():
                raise SpecificationError(f"changes are name: amount, e.g. cost: 1; not {part.strip()!r}")
            out[name.strip()] = float(amount)
        return out

    def validate(self, session: Session) -> list[str]:
        if self.fixed_effect.strip():
            try:
                float(self.fixed_effect)
            except ValueError:
                return [f"the fixed effect is a number, not {self.fixed_effect!r}"]
        try:
            changes = self._changes()
        except ValueError as exc:
            return [f"changes: {exc}"]
        if any(v == 0 for v in changes.values()):
            return ["changes: each amount is a nonzero number"]
        return []

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        extra: dict[str, Any] = {}
        if self.at != "average":
            extra["at"] = self.at
        if self.fixed_effect.strip():
            extra["fixed_effect"] = float(self.fixed_effect)
        if self.dummy:
            extra["dummy"] = True
        if self.changes.strip():
            extra["changes"] = self._changes()
        return make_call("pymetrix:clogit_effects", output=self.name, recorded_by=self.command_id, fit=Ref(self.fit), **extra)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
