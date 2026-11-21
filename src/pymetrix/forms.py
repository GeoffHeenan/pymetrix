"""Form metadata: how a setting appears in a generated dialog.

This module has no UI dependencies, so specifications, methods and tests in
the core can describe their own settings (``Annotated[int, Widget(...)]``);
the Qt and Jupyter frontends read the descriptions through ``form_spec``.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints

from .errors import SpecificationError


@dataclass(frozen=True)
class Widget:
    """How a field appears in a form.

    kind: "text", "multiline" (several lines, e.g. one equation per line), "formula", "hidden" (not in forms: set in code only), "period", "file", "choice", "number", "checkbox",
          "object" (a workspace object of the types in ``source``),
          "series" (series of the dataset named by field ``depends_on``),
          "level" (axes and series of that dataset: what to group or pivot by),
          "entity" (the values of that dataset's entity axis, e.g. its countries),
          "test" (tests valid for the object named by field ``depends_on``),
          "method" (estimation methods whose library is installed),
          "file" (a path with a Browse button; ``source`` is the file-type filter),
          "savefile" (the same, for a file to write: Browse opens a Save dialog),
          "column" (columns of the file named by field ``depends_on``, plus a blank choice),
          "item" (results that can be added to a dataset from the object named by ``depends_on``),
          "shock" (the shocks of the model or solution named by field ``depends_on``),
          "layer" (the series a figure draws, as "gdp (line)", from the figure or result named by ``depends_on``),
          "figurepart" (those series and the figure's panels, "panel 0: title (axes)", for the property editor),
          "axesmethod" (the matplotlib Axes methods that fig.call records),
          "stylestep" (the style calls recorded on the figure named by field ``depends_on``),
          "dataset" (the example datasets installed with pymetrix and its plugins),
          "named" (the names pymetrix knows now of the kind in ``source``: "report_templates", the built-in
          templates and plugins'; "themes", the built-in themes and those added at runtime),
          "extension" (the available extensions of the registry group in ``source``, e.g.
          "variances"; the field's default is always offered too),
          "options" (a sub-form for the settings of the extension named by field ``depends_on``;
          ``source`` is its registry group: "tests" (the default), "methods", "variances", "filters"...;
          fields named in ``exclude`` are set by the main form and left out).
    ``advanced`` fields go in the dialog's collapsed Advanced settings section (with a sub-form's own advanced settings).
    """

    kind: str
    source: str | None = None
    help: str = ""
    label: str = ""  # shown instead of the field name, e.g. "Standard errors" for cov
    depends_on: str | None = None
    multiple: bool = False  # "series"/"object": pick several (the value is comma-separated)
    show_if: tuple[str, Any] | None = None  # (field, values): shown only when field has one of values;
    # values may be a function returning them, e.g. the labels of the IV methods registered now
    group: str = ""  # section heading in the form
    min: float | None = None  # "number" bounds and step
    max: float | None = None
    step: float | None = None
    advanced: bool = False
    exclude: tuple[str, ...] = ()  # "options": settings the main form sets itself, left out of the sub-form
    empty: str = ""  # "object": what leaving it empty means (offered as a blank choice), e.g. "the file's own data"


def advanced(**kw: Any) -> Widget:
    """Shorthand for an advanced field whose kind follows from its type."""
    return Widget(kind="", advanced=True, **kw)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    kind: str
    default: Any
    required: bool
    choices: tuple[Any, ...] | None = None
    source: str | None = None
    help: str = ""
    type: Any = None
    depends_on: str | None = None
    multiple: bool = False
    show_if: tuple[str, Any] | None = None
    group: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    advanced: bool = False
    exclude: tuple[str, ...] = ()
    empty: str = ""
    owner: str = ""  # the command (or settings class) whose field it is: its own labels for the field and choices

    @property
    def dynamic(self) -> bool:
        """True when the UI fills the choices from the session (see ``Session.choices``)."""
        return self.kind in ("test", "series", "level", "entity", "method", "extension", "column", "item", "shock", "dataset", "layer", "figurepart", "axesmethod", "stylestep", "named") or (self.kind == "object" and bool(self.source))

    @property
    def scientific(self) -> bool:
        """Small floats (tolerances) are typed as text such as 1e-10 rather than spun."""
        if self.type is not float:
            return False
        tiny = [v for v in (self.default, self.min) if isinstance(v, (int, float)) and v != 0]
        return any(abs(v) < 1e-3 for v in tiny)

    def display(self, choice: Any) -> str:
        """How a fixed choice is shown in a form: "Constant and trend" for trend 'ct'; the value itself otherwise."""
        return _shown_choice(self, choice)

    @property
    def radio(self) -> bool:
        """Short fixed choice lists show as radio buttons, longer ones as dropdowns."""
        return self.kind == "choice" and self.choices is not None and len(self.choices) <= 3

    def visible(self, values: dict[str, Any]) -> bool:
        if self.show_if is None:
            return True
        name, allowed = self.show_if
        return values.get(name) in (allowed() if callable(allowed) else allowed)

    def bounds(self) -> tuple[float, float, float]:
        is_int = self.type is int
        low = self.min if self.min is not None else (0 if is_int else -1e9)
        high = self.max if self.max is not None else (100_000 if is_int else 1e9)
        return low, high, self.step if self.step is not None else (1 if is_int else 0.1)


from .wording import CHOICE_LABELS, FIELD_LABELS as LABELS  # noqa: E402, F401  (the words shown: one table, in wording)

VOCABULARY = {
    "cov": "coefficient covariance: classic, white (robust), hac (Newey-West) or Cluster(...)",
    "hac_lags": "HAC truncation lag; empty or 0 = automatic",
    "endog": "regressors correlated with the error, to be instrumented; list them in the formula too",
    "instruments": "excluded instruments, e.g. g, y[-1]",
    "lags": "lag length: too few leaves dynamics in the errors, too many costs degrees of freedom",
    "max_lags": "longest lag the search considers; the criterion picks the length up to this",
    "trend": "deterministic terms: n = none, c = constant, ct = constant and trend",
    "dist": "error distribution: normal, t, skewt (skewed t) or ged",
    "alpha": "significance level, e.g. 0.05",
    "signif": "significance level, e.g. 0.05",
    "seed": "random-number seed: the same seed gives the same result",
    "start": "first period; empty = earliest available",
    "end": "last period; empty = latest available",
    "name": "name of the result in the workspace",
    "data": "the dataset holding the series; its active sample and frequency apply",
    "variable": "the series to examine, by name: make a transformation such as log(y) a series first (Data → Generate series…)",
    "variables": "the series included, comma-separated, from the dataset",
    "tol": "convergence tolerance of the optimiser: smaller is more precise and slower",
    "max_iter": "maximum iterations of the optimiser before it stops",
    "debiased": "small-sample (n-k) correction of the covariance",
    "weights": "series of weights (optional)",
    "missing": "rows with missing values: drop them, or stop with an error",
    "formula": "e.g. c ~ y + c[-1]   (use = for identities)",
    "params": "comma-separated parameter names; empty for the linear form",
    "test": "the tests that apply to the chosen object",
    "var": "the fitted VAR or VECM whose coefficients and residual covariance are used",
    "fit": "the fitted model to forecast from",
    "path": "the data file to read",
    "periods": "horizon: number of periods after the shock or forecast origin",
    "scheme": "identification: Cholesky (recursive, in variable order) or LongRun (Blanchard-Quah)",
    "cumulative": "cumulate the responses (for variables in differences)",
    "squared": "use squared values (looks for volatility clustering)",
    "deterministic": "deterministic terms: constant and trend, inside or outside the cointegrating relations",
    "exog": "exogenous expressions (optional)",
    "exog_coint": "exogenous expressions inside the cointegrating relations",
    "seasons": "seasonal dummies: 4 quarterly, 12 monthly; 0 = none",
    "autolag": "how lags are chosen: information criterion (aic, bic) or significance of the last lag (t-stat)",
}
"""Help for settings that use pymetrix's standard names, so extensions need not repeat it."""


def option_string(specs: list[FieldSpec], values: dict[str, Any]) -> str:
    """``"lags=4, squared=True"`` for the entries that differ from their defaults.

    Text typed into a field that is not a plain string (``float | None``,
    ``int | Sequence[int]``...) is passed as written, so ``0.5`` stays a number.
    """
    parts = []
    for spec in specs:
        value = values.get(spec.name)
        if value in (None, "") or (not spec.required and value == spec.default):
            continue
        if not spec.required and isinstance(value, str) and spec.default is not None and not isinstance(spec.default, str) \
                and value.strip() == str(spec.default):
            continue  # a text box showing a default that is not text ("{}", "()"): still the default
        typed_text = isinstance(value, str) and spec.type is not str and get_origin(spec.type) is not Literal and _expression(value)
        parts.append(f"{spec.name}={value}" if typed_text and "," not in value else
                     f"{spec.name}=({value})" if typed_text else f"{spec.name}={value!r}")
    return ", ".join(parts)


def _expression(text: str) -> bool:
    import ast

    try:
        ast.parse(text, mode="eval")
    except SyntaxError:
        return False
    return True


def parse_options(text: str) -> tuple[list[Any], dict[str, Any]]:
    """``"y, lags=2, cause='b, c'"`` -> (["y"], {"lags": 2, "cause": "b, c"}).
    Values are Python literals; anything else (a bare name such as ``y``) is kept as text."""
    import ast

    if not text.strip():
        return [], {}
    try:
        call = ast.parse(f"f({text})", mode="eval").body
    except SyntaxError:
        raise SpecificationError(f"could not read the options {text!r}") from None

    def value(node: ast.AST) -> Any:
        try:
            return ast.literal_eval(node)
        except ValueError:
            return ast.unparse(node)

    return [value(a) for a in call.args], {k.arg: value(k.value) for k in call.keywords}  # type: ignore[attr-defined]


def settings_form(entry: Any, exclude: Any = frozenset()) -> list[FieldSpec]:
    """Form fields for a registered extension's settings: its fields less ``exclude``,
    with the entry's preset values as defaults."""
    from dataclasses import replace

    preset = getattr(entry, "preset", {}) or {}
    return [replace(s, default=preset.get(s.name, s.default)) for s in form_spec(entry.obj) if s.name not in exclude]


def build(entry: Any, options: str = "", **main: Any) -> Any:
    """An instance of a registered extension from its settings text plus main-form values.

    Main-form values apply only to fields the class has; the entry's preset
    settings fill in the rest.
    """
    names = {f.name for f in fields(entry.obj)}
    _, kw = parse_options(options)
    kw.update({k: v for k, v in main.items() if k in names})
    return entry.build(**kw)


def _kind(widget: Widget | None, base: Any, choices: tuple | None) -> str:
    if widget is not None and widget.kind:
        return widget.kind
    if choices:
        return "choice"
    if base is bool:
        return "checkbox"
    if base in (int, float):
        return "number"
    return "text"


def form_spec(cls: type) -> list[FieldSpec]:
    """Form fields for a command, or for any dataclass of settings (tests, methods)."""
    hints = get_type_hints(cls, include_extras=True)
    specs = []
    for f in fields(cls):  # type: ignore[arg-type]
        if not f.init:
            continue
        hint = hints[f.name]
        widget = None
        base = hint
        if get_origin(hint) is Annotated:
            base, *extras = get_args(hint)
            widget = next((e for e in extras if isinstance(e, Widget)), None)
        if widget is not None and widget.kind == "hidden":  # set in code only (e.g. planned features)
            continue
        choices = get_args(base) if get_origin(base) is Literal else None
        if choices is None and set(get_args(base)) == {bool, type(None)}:  # yes, no, or let pymetrix decide
            choices = (None, True, False)
        if f.default is not MISSING:
            default, required = f.default, False
        elif f.default_factory is not MISSING:
            default, required = f.default_factory(), False
        else:
            default, required = None, True
        w = widget or Widget(kind="")
        specs.append(
            FieldSpec(
                name=f.name,
                label=_label(cls, f.name, w.label),
                kind=_kind(widget, base, choices),
                default=default,
                required=required,
                choices=choices,
                source=w.source,
                help=w.help or VOCABULARY.get(f.name, ""),
                type=base,
                depends_on=w.depends_on,
                multiple=w.multiple,
                show_if=w.show_if,
                group=w.group,
                min=w.min,
                max=w.max,
                step=w.step,
                advanced=w.advanced,
                exclude=w.exclude,
                empty=w.empty,
                owner=cls.__name__,
            )
        )
    return specs


# --- what a form needs before it can run (validation only: kept apart from labels and choices) -----------------------

_ALWAYS_SET = ("checkbox", "number", "options", "hidden")  # a tick box, a spin box, a settings sub-form: always a value


def needs_value(spec: FieldSpec) -> bool:
    """Whether the step cannot run with this field left empty: it has no default, leaving it empty means nothing
    (``empty`` is not set), and its widget can be left empty (a tick box, a number or a fixed choice always has a
    value). The form's required marker (*) shows exactly these fields."""
    if not spec.required or spec.empty or spec.kind in _ALWAYS_SET:
        return False
    return not (spec.kind == "choice" and spec.choices)


def is_blank(value: Any) -> bool:
    """An empty field: nothing typed, chosen or ticked."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (tuple, list, set, dict)):
        return not value
    return False


def missing_fields(specs: list[FieldSpec], values: dict[str, Any]) -> list[FieldSpec]:
    """The fields shown in the form (``show_if``) that need a value (``needs_value``) and have none."""
    return [s for s in specs if needs_value(s) and s.visible(values) and is_blank(values.get(s.name))]


def missing_message(missing: list[FieldSpec]) -> str:
    """"Fill in Start and End." / "Choose Fit." for the fields ``missing_fields`` found; "" when none."""
    if not missing:
        return ""
    picks = [s.label for s in missing if s.kind in ("object", "series", "choice", "file", "savefile", "extension", "method",
                                                     "test", "dataset", "named", "level", "entity", "column", "item", "shock")]
    typed = [s.label for s in missing if s.label not in picks]

    def listed(names: list[str]) -> str:
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]

    parts = ([f"choose {listed(picks)}"] if picks else []) + ([f"fill in {listed(typed)}"] if typed else [])
    text = "; ".join(parts)
    return text[0].upper() + text[1:] + " to run this step."


# ----------------------------------------------------------------- presentation: labels and choice names (wording)

def _label(cls: type, name: str, explicit: str) -> str:
    """A field's label: the command's override, its Widget's label, the shared table, or its name's words."""
    from .wording import field_label

    return field_label(cls.__name__, name, explicit)


def _shown_choice(spec: FieldSpec, choice: Any) -> str:
    """How a fixed choice is shown in a form, in sentence case: "Constant and trend" for trend 'ct', "Drop rows with
    missing values" for missing 'drop'; the value itself otherwise. The stored value stays the code."""
    from .wording import choice_label, sentence

    words = choice_label(spec.owner, spec.name, choice)
    if words is None:
        if choice is None or isinstance(choice, bool):  # bool | None settings: automatic, yes, no
            words = {None: "automatic", True: "yes", False: "no"}[choice]
        elif choice == "" and spec.empty:  # e.g. (unchanged)
            return f"({spec.empty})"
        else:
            words = str(choice)
    return sentence(words) if isinstance(choice, str) or choice is None or isinstance(choice, bool) else words


def access_keys(labels: list[str], taken: Any = ()) -> list[str]:
    """Each display label with an access key: ``&`` before a letter no other label (nor ``taken``: the dialog's
    buttons' keys) uses, preferring the first letter of a word, then any letter; where every letter of a label is
    another's, the others move to a letter of theirs still free, so as many labels as possible have a key (a label
    is left without one only when that cannot be done). Ampersands already in a label are doubled (shown, not keys).
    "Standard errors", "Sample" -> "&Standard errors", "S&ample"."""
    used = {str(k).lower() for k in taken}
    texts = [str(label) for label in labels]

    def candidates(text: str) -> list[int]:
        starts = [i for i, ch in enumerate(text) if ch.isalnum() and (i == 0 or not text[i - 1].isalnum())]
        others = [i for i, ch in enumerate(text) if ch.isalnum() and i not in starts]
        seen, out = set(), []
        for i in starts + others:
            key = text[i].lower()
            if text[i].isascii() and key not in used and key not in seen:
                seen.add(key)
                out.append(i)
        return out

    options = [candidates(t) for t in texts]
    at: list[int | None] = [None] * len(texts)
    owner: dict[str, int] = {}  # key -> the label that has it

    def move(n: int, visited: set[str]) -> bool:  # give label n a key, moving others along (an augmenting path)
        for i in options[n]:
            key = texts[n][i].lower()
            if key in visited:
                continue
            visited.add(key)
            other = owner.get(key)
            if other is None or move(other, visited):
                at[n], owner[key] = i, n
                return True
        return False

    # In order, each label its first free choice, else a key moved along from labels before it. A label with a key
    # keeps one, so the labels first in the list (a dialog's fields in view) are never left without a key while a
    # later one has one they could have had.
    for n, text in enumerate(texts):
        i = next((i for i in options[n] if text[i].lower() not in owner), None)
        if i is not None:
            at[n], owner[text[i].lower()] = i, n
        else:
            move(n, set())
    out = []
    for text, i in zip(texts, at):
        escaped = [ch.replace("&", "&&") for ch in text]
        if i is not None:
            escaped[i] = "&" + escaped[i]
        out.append("".join(escaped))
    return out


def access_key(text: str) -> str:
    """The key ``&`` marks in ``text`` (lower case), or "" (``&&`` is a shown ampersand, not a key)."""
    i = 0
    while i < len(text) - 1:
        if text[i] == "&":
            if text[i + 1] != "&":
                return text[i + 1].lower()
            i += 1
        i += 1
    return ""
