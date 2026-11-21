"""Dialogs for models: Model > Shock decomposition, Model > Export model, Model > Import > FRB/US XML file and
EViews state-space object, Model > Add factors and Model > Policy > Optimal control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session

# the objects a backward-looking model is built from: equations (fitted or not), or a model already built or read
MODEL_PARTS = "FittedEquation,Equation,Model,EViewsXmlModel"


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _model(equations: str, params: str) -> Any:
    from .builtin import _model_fields

    return _model_fields(equations, params)


def _missing(session: Session, *names: str) -> list[str]:
    return [f"no object named {n!r}" for n in names if n and n not in session.objects]


@command(menu="Model/Shock decomposition...")
@dataclass(frozen=True)
class ShockDecomposition(Command):
    """Split an estimated model's Kalman-smoothed history into the contributions of each shock and of the starting
    point: which shocks drove output, inflation or the policy rate in each period? The parts add up exactly to the
    smoothed path (in deviations from the steady state).

    Start (first period) counts the first period's state, its shocks included, as the starting point; Start
    (presample) takes the first period as the state before the sample and decomposes from the next, so that period's
    shocks go to the shocks. Early in the sample the starting point carries much of the movement, as in a VAR's
    historical decomposition; read the decomposition from a few years in.

    A model estimated with the OccBin filter (Model > Estimate model, Advanced, OccBin) is decomposed along its
    OccBin-smoothed path, with the regimes held at the smoothed ones: the binding constraints' constants (a rate held
    at its floor) are a part of their own, and the parts still add up. OccBin decompositions start from the first
    period. A model estimated with known-date regimes is decomposed through the regimes' time-varying rules, their
    constants a part of their own ("regimes"). With correlated shocks, the parts are of the shocks orthogonalised by
    Cholesky in the model's order, as impulse responses are.
    """

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models", "state-space")

    source: Annotated[str, Widget("object", source="FittedModel", help="an estimated model (Model > Estimate model)")]
    start: Annotated[Literal["first", "presample"], Widget("choice", help="first: the first period's state is the starting point; "
                                                                           "presample: decompose from the second period, the first "
                                                                           "period's shocks going to the shocks")] = "first"
    name: str = "shock_decomposition"
    occbin: Annotated[bool | None, Widget("choice", advanced=True,
                                          help="decompose the OccBin-smoothed path, regimes held at the smoothed ones; "
                                               "empty = when the model was estimated with the OccBin filter")] = None

    def validate(self, session: Session) -> list[str]:
        if self.source not in session.objects:
            return [f"no estimated model named {self.source!r}"]
        fit = session[self.source]
        occbin = self.occbin
        if occbin is None:
            method = fit.provenance.method if getattr(fit, "provenance", None) else {}
            occbin = str(method.get("filter", "")).startswith("OccBin")
        if occbin and self.start != "first":
            return ["OccBin decompositions start from the first period's state: choose start first"]
        return []

    def to_call(self, session: Session) -> Any:
        from ..calls import method_call
        from ..codec import Ref

        return method_call(type(session[self.source]), "decompose", Ref(self.source), output=self.name, recorded_by=self.command_id,
                           start=self.start, occbin=self.occbin)

    def run(self, session: Session) -> Any:
        return self.to_call(session).run(session)

    def to_code(self) -> str:
        args = ([repr(self.start)] if self.start != "first" else []) + ([f"occbin={self.occbin!r}"] if self.occbin is not None else [])
        return f"{self.name} = {self.source}.decompose({', '.join(args)})"


@command(menu="Model/Export model...")
@dataclass(frozen=True)
class ExportModel(Command):
    """Write a model in another syntax: pymetrix's, Dynare's (.mod), IRIS's (.model) or EViews's (a program, or the
    model's text). The result has the text, notes on what did not carry across (priors IRIS keeps elsewhere, regimes
    Dynare has no place for, names EViews would read differently) and a check: the text is read back in and solved,
    and its responses to each shock and its steady state compared with the original's.

    The text is generated from the model, not from any text it was read from: it means the same, but comments,
    layout, loops and model-local variables are not reproduced. Copy it from the Text tab, or open it in
    Model > Model editor to change it. The editor's Write as button does the same from the text being edited.
    """

    topics: ClassVar[tuple[str, ...]] = ("forward-looking-models",)

    source: Annotated[str, Widget("object", source="Template,DynareModel,Model", help="a template, a Dynare model or a Model")]
    syntax: Annotated[Literal["pymetrix", "dynare", "iris", "eviews"], Widget("choice", help="the language to write it in")] = "dynare"
    form: Annotated[Literal["program", "text"], Widget("choice", show_if=("syntax", ("eviews",)),
                                                       help="program: a .prg (model m, m.append lines and the values); "
                                                            "text: the model's text alone, its values in a comment")] = "program"
    name: str = "translation"
    origin: Annotated[str, Widget("text", advanced=True, help="where the model came from, said in the first comment line")] = ""

    def validate(self, session: Session) -> list[str]:
        return [] if self.source in session.objects else [f"no model named {self.source!r}"]

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:export_model", output=self.name, recorded_by=self.command_id, obj=Ref(self.source),
                         syntax=self.syntax, origin=self.origin.strip(), form=self.form if self.syntax == "eviews" else "program")

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Model/Import FRB/US XML file...")
@dataclass(frozen=True)
class ImportFRBUS(Command):
    """Read an XML model file in the format of the Federal Reserve's FRB/US package (mods/model.xml): its equations,
    coefficients and each variable's kind, sector, definition and stochastic type, as a model ready to solve.

    The equations are EViews's, read by the EViews reader; coefficients such as y_x(1) become parameters y_x_1.
    Variables with model-consistent-expectations versions use the standard (VAR-expectations) equations unless MCE
    names their groups (e.g. mcap, mcwp) or is all. The result lists the variables; solve it with Model > Solve model (its
    name among the equations), make it track the data with Model > Add factors, and control it with
    Model > Policy > Optimal control. The package itself is the Fed's and is not shipped with pymetrix.
    """

    topics: ClassVar[tuple[str, ...]] = ("model-solution",)

    path: Annotated[str, Widget("file", source="XML model files (*.xml);;All files (*)", help="the model file, e.g. mods/model.xml of the FRB/US package")]
    mce: Annotated[str, Widget("text", help="model-consistent expectations: all, or the groups to use them for, e.g. mcap, mcwp; "
                                            "empty = VAR expectations throughout")] = ""
    name: str = "frbus"

    def _mce(self) -> bool | list[str]:
        text = self.mce.strip()
        if text.lower() in ("all", "true", "yes"):
            return True
        return _names(text) or False

    def to_call(self) -> Any:
        from ..calls import make_call

        return make_call("pymetrix:read_eviews_xml", output=self.name, recorded_by=self.command_id, path=self.path,
                         mce=self._mce(), name=None)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Model/Import EViews state-space object...", id="pymetrix:import_eviews_sspace")
@dataclass(frozen=True)
class ImportEViewsStateSpace(Command):
    """Read an EViews state-space object (sspace: @signal, @state, @ename, @evar; bare lines or after ss.append) as a
    state-space model ready to estimate with Estimate > Time series > State space.

    Signals become measurement equations (terms in data series moved to the left, the observed side), states
    transition equations, and an error's variance written as a coefficient squared (c(2)^2) the shock's standard
    deviation, the parameter c_2. Coefficients c(i) become parameters c_i, with starting values from c(i) = ... lines
    or given here. What does not map is refused with its line and column: covariances of errors, variances not of
    that form, terms mixing data and states, data in state equations, and EViews's programming language.
    """

    topics: ClassVar[tuple[str, ...]] = ("state-space",)

    path: Annotated[str, Widget("file", source="EViews programs (*.prg);;All files (*)", help="a .prg holding the sspace object's lines")]
    params: Annotated[str, Widget("text", help="starting values, e.g. c_1=0.3, c_2=0.8 (or c(1)=0.3), over the file's")] = ""
    name: str = "sspace"

    def _params(self) -> dict[str, float] | None:
        import re

        from .builtin import _calibration

        text = re.sub(r"\b([A-Za-z_]\w*)\(\s*(\d+)\s*\)", r"\1_\2", self.params)  # c(1)=0.3 -> c_1=0.3
        return _calibration(text) or None

    def validate(self, session: Session) -> list[str]:
        try:
            self._params()
        except ValueError as exc:
            return [str(exc)]
        return []

    def to_call(self) -> Any:
        from ..calls import make_call

        return make_call("pymetrix:read_eviews_sspace", output=self.name, recorded_by=self.command_id, source=self.path,
                         params=self._params())

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Model/Add factors...")
@dataclass(frozen=True)
class AddFactors(Command):
    """Tracking add factors: each equation's residual at the data in each period of a span, as EViews's addinit sets
    them. Solved with them (Model > Solve model, Scenario, Add factors) the model reproduces the data exactly, under any
    solver, so a scenario's other changes read as deviations from that baseline.

    Over a projection's span (a baseline such as the Fed's LONGBASE), the add factors make the model track it; over a
    historical span, they are the equations' historical errors, by period, which a stochastic simulation can resample
    (Model > Solve model, Advanced, Errors: sample). Every variable of the model needs values in the span; an equation's
    add factor is missing where its lags reach before the data.
    """

    topics: ClassVar[tuple[str, ...]] = ("model-solution",)

    equations: Annotated[str, Widget("object", source=MODEL_PARTS, multiple=True,
                                     help="the model: fitted equations and identities, or a model read (FRB/US XML)")]
    data: Annotated[str, Widget("object", source="Dataset", help="the data to track: every variable of the model")] = "data"
    start: Annotated[str, Widget("period", group="Span", help="the first period; empty = from where they can be computed")] = ""
    end: Annotated[str, Widget("period", group="Span", help="the last period; empty = to the end of the data")] = ""
    name: str = "add_factors"
    params: Annotated[str, Widget("text", advanced=True, help="values for the parameters of equations not estimated, e.g. b=0.99")] = ""

    def validate(self, session: Session) -> list[str]:
        if not _names(self.equations):
            return ["choose the model's equations"]
        return _missing(session, *_names(self.equations), self.data)

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        span = (self.start or None, self.end or None) if self.start or self.end else None
        return make_call("pymetrix:model.add_factors", _model(self.equations, self.params), output=self.name,
                         recorded_by=self.command_id, data=Ref(self.data), span=span)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Model/Optimal control...", id="pymetrix:optimal_control")
@dataclass(frozen=True)
class OptimalControlCommand(Command):
    """The paths of some exogenous instruments that minimise a discounted quadratic loss over a span, with a
    backward-looking model solved as the constraint: the optimal-control simulation of the Federal Reserve's
    ocpolicy.prg on FRB/US, where the instrument is an add factor on the policy rule (rfftay_aerr).

    The loss is the sum, over the span's periods t (counted from 0) and the variables v named, of
    discount^t * weight_v * (v(t) - desired_v)^2; desired values are zero unless given. The instruments move in each
    period of the control span (the whole span by default), and the policymaker commits to the whole path at the
    start. Penalise the instrument's changes (e.g. delrff=1) or the path may be implausibly aggressive. The model is
    solved period by period (Newton or Gauss-Seidel); a model with leads (model-consistent expectations) is not
    controlled here. Add factors (Model > Add factors) make the baseline track the data first. The result shows the
    loss with and without the moves, the moves, and the paths of the loss's variables.
    """

    topics: ClassVar[tuple[str, ...]] = ("model-solution",)

    equations: Annotated[str, Widget("object", source=MODEL_PARTS, multiple=True,
                                     help="the model: fitted equations and identities, or a model read (FRB/US XML)")]
    start: Annotated[str, Widget("period", help="the first period of the loss and of the solution")]
    end: Annotated[str, Widget("period", help="the last period of the loss and of the solution")]
    instruments: Annotated[str, Widget("text", help="exogenous series moved optimally, e.g. rfftay_aerr")]
    loss: Annotated[str, Widget("text", help="weights of the loss, e.g. ugap=1, pic4=1, delrff=1 (penalise the instrument's moves too)")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    desired: Annotated[str, Widget("text", help="desired values, e.g. pic4=2; the others zero")] = ""
    discount: Annotated[float, Widget("number", advanced=True, min=0.0001, max=1.0, step=0.01, help="discount factor of the loss; 1 = none")] = 1.0
    name: str = "control"
    control_start: Annotated[str, Widget("period", advanced=True, group="Control",
                                         help="the first period the instruments move; empty = the span's")] = ""
    control_end: Annotated[str, Widget("period", advanced=True, group="Control",
                                       help="the last period the instruments move; empty = the span's")] = ""
    adjust: Annotated[str, Widget("object", source="dict", advanced=True, empty="empty: none",
                                  help="add factors (Model > Add factors) added to the equations, so the baseline tracks the data")] = ""
    solver: Annotated[Literal["Newton", "GaussSeidel"], Widget("choice", advanced=True,
                                                              help="how the model is solved each period")] = "Newton"
    params: Annotated[str, Widget("text", advanced=True, help="values for the parameters of equations not estimated, e.g. b=0.99")] = ""
    tol: Annotated[float, Widget("number", min=1e-12, max=0.1, advanced=True,
                                 help="stop when the gradient falls below this share of its size at the start")] = 1e-6
    max_iter: Annotated[int, Widget("number", min=1, max=1000, advanced=True, help="Gauss-Newton iterations at most")] = 20

    def _weights(self, text: str) -> dict[str, float]:
        from .builtin import _calibration

        return _calibration(text)

    def validate(self, session: Session) -> list[str]:
        if not _names(self.equations):
            return ["choose the model's equations"]
        problems = _missing(session, *_names(self.equations), self.data, self.adjust)
        if problems:
            return problems
        if not _names(self.instruments):
            return ["name at least one instrument: an exogenous series to move"]
        try:
            weights, desired = self._weights(self.loss), self._weights(self.desired)
        except ValueError as exc:
            return [str(exc)]
        if not weights:
            return ["give the loss weights, e.g. ugap=1, pic4=1"]
        extra = [v for v in desired if v not in weights]
        return [f"desired values for {', '.join(extra)}, which are not in the loss"] if extra else []

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Fields, Ref
        from ..models import Scenario
        from ..models.solvers import GaussSeidel

        control = (self.control_start or self.start, self.control_end or self.end) if self.control_start or self.control_end else None
        extra: dict[str, Any] = {}
        if self.adjust:
            extra["scenario"] = Fields(Scenario, {"adjust": Ref(self.adjust)})
        if self.solver == "GaussSeidel":
            extra["solver"] = GaussSeidel()
        return make_call("pymetrix:optimal_control", output=self.name, recorded_by=self.command_id,
                         model=_model(self.equations, self.params), data=Ref(self.data), span=(self.start, self.end),
                         instruments=_names(self.instruments), loss=self._weights(self.loss),
                         desired=self._weights(self.desired) or None, control=control, discount=self.discount,
                         tol=self.tol, max_iter=self.max_iter,
                         **extra)

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
