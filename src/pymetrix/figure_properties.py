"""Every style property matplotlib can set on what a figure draws, from matplotlib's own description of its artists.

``ArtistInspector`` lists each artist type's settable properties and their valid values. This module turns that
into a catalogue (``properties``) and a form (``property_form``), so the app's property editor covers a line's, a
bar's, a band's and an axes' properties without pymetrix listing them by hand (layer 2 of
docs/notes/figure-styling.md). Properties that hold the plotted data or positions (``xdata``, ``xy``, ``width``,
``paths``...) are left out, so styling cannot change what is plotted; so are those that need Python objects or
concern interaction (``transform``, ``picker``, ``path_effects``, ``cmap``...).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .errors import SpecificationError

__all__ = ["PARTS", "PropertyInfo", "axes_methods", "properties", "property_form", "property_table"]

# Axes methods that style or annotate a panel: recorded by fig.call. Methods that plot more data (plot, bar,
# scatter...), clear or remove things, or need Python objects are not among them, so styling stays honest and safe.
_METHOD_PREFIXES = ("set_", "tick_", "invert_", "minorticks_", "label_", "locator_")
_METHODS = {"grid", "legend", "margins", "annotate", "text", "axhline", "axvline", "axhspan", "axvspan", "hlines",
            "vlines", "axline", "arrow", "autoscale", "ticklabel_format", "set", "tick_params"}
_NOT_METHODS = {"set_figure", "set_axes_locator", "set_prop_cycle", "set_subplotspec", "set_transform", "set_clip_path",
                "set_clip_box", "set_agg_filter", "set_path_effects", "set_picker", "set_navigate", "set_navigate_mode",
                "set_mouseover", "set_animated", "set_in_layout", "set_label", "set_url", "set_gid", "set_sketch_params",
                "set_snap", "set_rasterized", "set_rasterization_zorder", "set_position", "set_box_aspect",
                "set_forward_navigation_events", "set_xticklabels", "set_yticklabels"}


def axes_methods() -> list[str]:
    """The ``matplotlib.axes.Axes`` methods ``fig.call`` records: those that style or annotate a panel."""
    from matplotlib.axes import Axes

    return sorted(name for name in dir(Axes) if not name.startswith("_") and callable(getattr(Axes, name))
                  and (name in _METHODS or name.startswith(_METHOD_PREFIXES)) and name not in _NOT_METHODS)

#: what a figure draws, by the name the style calls use, and matplotlib's class for it
PARTS = ("line", "bars", "band", "axes")

# the plotted data and positions, Python objects, interaction and rendering internals: not style
_LEFT_OUT = frozenset({
    "agg_filter", "animated", "clip_box", "clip_path", "data", "xdata", "ydata", "figure", "gid", "in_layout", "label",
    "mouseover", "path_effects", "picker", "pickradius", "rasterized", "sketch_params", "snap", "transform", "url", "urls",
    "offset_transform", "offsets", "paths", "verts", "verts_and_codes", "array", "clim", "cmap", "norm", "sizes",
    "angle", "bounds", "height", "width", "x", "y", "xy", "rotation_point", "axes_locator", "navigate", "navigate_mode",
    "position", "prop_cycle", "subplotspec", "forward_navigation_events", "rasterization_zorder", "autoscale_on",
    "autoscalex_on", "autoscaley_on", "xticklabels", "yticklabels", "xticks", "yticks", "clip_on",
})
_OBJECT_WORDS = ("Transform", "Callable", "callable", "Colormap", "Normalize", "PathEffect", "Cycler", "Bbox", "Figure",
                 "array", "unknown")
_KEPT_ANYWAY = {"marker", "xscale", "yscale", "xinverted", "yinverted", "hatch_linewidth"}  # useful, though described loosely
_SCALES = ("", "linear", "log", "symlog", "logit")
_BOOLS = {"xinverted", "yinverted"}


@dataclass(frozen=True)
class PropertyInfo:
    """One settable property: its name (matplotlib's), how it is entered (``choice``, ``bool``, ``number`` or
    ``text``), the fixed choices if any, and matplotlib's own description of its valid values."""

    name: str
    kind: str
    choices: tuple[str, ...] = ()
    help: str = ""


def _artist(part: str) -> type:
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    classes = {"line": Line2D, "bars": Rectangle, "band": PolyCollection, "axes": Axes}
    if part not in classes:
        raise SpecificationError(f"no figure part {part!r}; the parts are {', '.join(PARTS)}")
    return classes[part]


def _clean(text: str) -> str:
    """matplotlib's reST markup as plain text: :mpltype:`color` -> color, `~matplotlib.markers.MarkerStyle` -> MarkerStyle."""
    text = re.sub(r":mpltype:`(\w+)`", r"\1", " ".join(text.split()))
    return re.sub(r"`~?\.?([\w.]+)`", lambda m: m.group(1).split(".")[-1], text)


def _choices(text: str) -> tuple[str, ...] | None:
    """The values of a property described only by a set of them: ``{'default', 'steps'}`` (or an enum class or that set)."""
    m = re.fullmatch(r"(?:`\.\w+` or )?\{(.*)\}(?:, default: .*)?", text.strip())
    if not m or "..." in m.group(1):
        return None
    try:
        values = ast.literal_eval(f"({m.group(1)},)")
    except (ValueError, SyntaxError):
        return None
    return tuple(str(v) for v in values)


@lru_cache(maxsize=None)
def properties(part: str) -> tuple[PropertyInfo, ...]:
    """The style properties matplotlib can set on ``part`` (line, bars, band or axes), by name."""
    from matplotlib.artist import ArtistInspector

    inspector = ArtistInspector(_artist(part))
    out = []
    for name in inspector.get_setters():
        if name in _LEFT_OUT:
            continue
        raw = inspector.get_valid_values(name) or ""
        if name not in _KEPT_ANYWAY and any(word in raw for word in _OBJECT_WORDS):
            continue
        help_text = _clean(raw)
        if name in ("xscale", "yscale"):
            out.append(PropertyInfo(name, "choice", _SCALES, "linear, log, symlog or logit"))
        elif name in _BOOLS or raw.startswith("bool"):
            out.append(PropertyInfo(name, "bool", ("", "True", "False"), help_text))
        elif (fixed := _choices(raw)) is not None:
            out.append(PropertyInfo(name, "choice", ("", *fixed), help_text))
        elif raw.startswith("float") or name == "hatch_linewidth":
            out.append(PropertyInfo(name, "number", (), help_text))
        else:
            out.append(PropertyInfo(name, "text", (), help_text))
    return tuple(out)


def property_table(part: str) -> Any:
    """The catalogue as a table: each property, how it is entered and its valid values (matplotlib's words)."""
    from .output import Table

    rows = [[p.kind, ", ".join(c for c in p.choices if c) or p.help] for p in properties(part)]
    return Table(["entered as", "valid values"], rows, index=[p.name for p in properties(part)], index_name="property",
                 title=f"Style properties of a {part}", notes=("From matplotlib's ArtistInspector; data and positions left out.",))


def property_form(part: str) -> list[Any]:
    """Form fields for ``part``'s properties, each empty (unchanged) unless set; labelled with matplotlib's names."""
    from typing import Literal  # noqa: F401 - the choices are typed as text

    from .forms import FieldSpec

    fields = []
    for p in properties(part):
        # colours and words stay text ('0.5' is a grey, '2020' a title); everything else is read as written: 2, True,
        # (0, 10), dashed (option_string quotes what is not a Python expression, such as #aa0000)
        text = ("color" in p.help or p.help == "str" or p.name in ("title", "xlabel", "ylabel")
                or p.kind == "choice")  # 'steps-post' is a word from the list, not steps minus post
        fields.append(FieldSpec(p.name, p.name, "choice" if p.choices else "text", "", False,
                                choices=p.choices or None, help=p.help, type=str if text else Any, empty="unchanged"))
    return fields
