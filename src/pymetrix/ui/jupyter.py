"""Jupyter frontend for teaching: forms generated from the command registry.

The same rules as the desktop dialogs: radio buttons for short choice lists,
dropdowns, bounded number boxes, series checklists, fields shown only when
relevant, section headings, and a settings sub-form for the chosen test. Each
form shows the Python it ran, so students see the code behind the menu::

    from pymetrix.ui.jupyter import form, menu
    session = Session(); session.objects["data"] = data
    form(FitEquation, session)      # one form
    menu(session)                   # every registered command, in an accordion
"""

from __future__ import annotations

import html
from typing import Any

from .._api import renamed_arguments
from ..commands import COMMANDS, Command, FieldSpec, Session, form_spec, option_string
from .render import to_html


def _widgets():
    try:
        import ipywidgets
    except ImportError as exc:
        raise ImportError("Jupyter forms need ipywidgets: pip install pymetrix[jupyter]") from exc
    return ipywidgets


def _style() -> dict[str, str]:
    return {"description_width": "initial"}


def _widget(w: Any, spec: FieldSpec) -> Any:
    from ..forms import needs_value

    desc = spec.label + (" *" if needs_value(spec) else "")  # the same marks as the desktop dialogs
    if spec.multiple:
        return w.SelectMultiple(options=[], description=desc, style=_style(), rows=6)
    if spec.dynamic:
        return w.Dropdown(options=[], description=desc, style=_style())
    if spec.radio:
        return w.RadioButtons(options=[(spec.display(c), c) for c in spec.choices], value=spec.default, description=desc, style=_style())
    if spec.kind == "choice":
        return w.Dropdown(options=[(spec.display(c), c) for c in spec.choices], value=spec.default, description=desc, style=_style())
    if spec.kind == "checkbox":
        return w.Checkbox(value=bool(spec.default), description=desc)
    if spec.kind == "number" and _optional_number(spec):  # empty means its default meaning (automatic): text, read as a number
        return w.Text(value="" if spec.default is None else str(spec.default), description=desc, placeholder=spec.help,
                      style=_style(), layout=w.Layout(width="60%"))
    if spec.kind == "number":
        low, high, step = spec.bounds()
        cls = w.BoundedIntText if spec.type is int else w.BoundedFloatText
        return cls(value=spec.default or 0, min=low, max=high, step=step, description=desc, style=_style())
    if spec.kind == "multiline":
        return w.Textarea(value="" if spec.default is None else str(spec.default), description=desc, placeholder=spec.help,
                          style=_style(), layout=w.Layout(width="80%", height="90px"))
    return w.Text(value="" if spec.default is None else str(spec.default), description=desc, placeholder=spec.help,
                  style=_style(), layout=w.Layout(width="60%"))


def _optional_number(spec: FieldSpec) -> bool:
    import types
    import typing

    hint = spec.type
    return (isinstance(hint, types.UnionType) or typing.get_origin(hint) is typing.Union) and type(None) in typing.get_args(hint)


def _value(widget: Any, spec: FieldSpec) -> Any:
    if spec.kind == "number" and _optional_number(spec):
        text = str(widget.value).strip()
        if not text:
            return None
        import typing

        number = float(text)
        return int(number) if int in typing.get_args(spec.type) and number.is_integer() else number
    if spec.multiple:
        return ", ".join(widget.value)
    value = widget.value
    return "" if value is None and spec.kind in ("text", "series", "level", "entity", "object", "test", "formula", "period", "layer", "figurepart", "axesmethod", "stylestep", "named") else value


def _advanced_section(w: Any, rows: list[Any]) -> Any:
    """The form's one place for advanced settings, as in the desktop dialogs: a collapsed "Advanced settings" section
    at the bottom, holding the command's advanced fields and those of a sub-form's chosen extension."""
    section = w.Accordion(children=[w.VBox(rows)], selected_index=None)
    section.set_title(0, "Advanced settings")
    return section


def form(command_cls: type[Command], session: Session) -> Any:
    w = _widgets()
    from IPython.display import HTML, Markdown, display

    specs = form_spec(command_cls)
    inputs: dict[str, Any] = {}
    sub_forms: dict[str, tuple[Any, list[FieldSpec], dict[str, Any], list[Any]]] = {}
    rows: dict[str, Any] = {}
    headings: dict[tuple[bool, str], Any] = {}
    pages: dict[bool, list[Any]] = {False: [], True: []}
    sub_advanced: dict[str, Any] = {}
    for spec in specs:
        page = pages[spec.advanced]
        if spec.group and (spec.advanced, spec.group) not in headings:
            headings[(spec.advanced, spec.group)] = w.HTML(f"<b>{html.escape(spec.group)}</b>")
            page.append(headings[(spec.advanced, spec.group)])
        if spec.kind == "options":
            box = w.VBox(layout=w.Layout(margin="0 0 0 20px"))
            sub_forms[spec.name] = (box, [], {}, [None])
            rows[spec.name] = box
            sub_advanced[spec.name] = w.VBox(layout=w.Layout(margin="0 0 0 20px"))  # its advanced settings, in the section
        else:
            inputs[spec.name] = _widget(w, spec)
            rows[spec.name] = inputs[spec.name]
        page.append(rows[spec.name])
    for name, part in sub_advanced.items():  # a sub-form's advanced settings: in the one section, under its label
        heading = w.HTML(f"<b>{html.escape(next(s.label for s in specs if s.name == name))}</b>")
        part.heading = heading
        pages[True] += [heading, part]
    advanced_section = _advanced_section(w, pages[True]) if pages[True] else None
    children = pages[False] + ([advanced_section] if advanced_section is not None else [])

    def values() -> dict[str, Any]:
        out = {name: _value(widget, next(s for s in specs if s.name == name)) for name, widget in inputs.items()}
        for name, (_, sub_specs, sub_inputs, _) in sub_forms.items():
            out[name] = option_string(sub_specs, {k: _value(v, s) for (k, v), s in zip(sub_inputs.items(), sub_specs)})
        return out

    busy = [False]

    def refresh(*_: Any) -> None:
        if busy[0]:
            return
        busy[0] = True
        try:
            for _ in range(2):
                for spec in specs:
                    if spec.dynamic:
                        widget = inputs[spec.name]
                        options = session.choices(spec, values())
                        if spec.multiple:
                            keep = tuple(v for v in widget.value if v in options)
                            widget.options = options
                            widget.value = keep
                        else:
                            current = widget.value if widget.value in options else (spec.default if spec.default in options else None)
                            widget.options = options
                            widget.value = current if current is not None else (options[0] if options else None)
                    if spec.kind == "options":
                        box, _, sub_inputs, last = sub_forms[spec.name]
                        key = values().get(spec.depends_on)
                        if key != last[0]:
                            last[0] = key
                            sub_specs = session.options_form(spec, values())
                            new_inputs = {s.name: _widget(w, s) for s in sub_specs}
                            sub_forms[spec.name] = (box, sub_specs, new_inputs, last)
                            basic = [new_inputs[s.name] for s in sub_specs if not s.advanced]
                            extra = [new_inputs[s.name] for s in sub_specs if s.advanced]
                            sub_advanced[spec.name].children = extra
                            box.children = basic or [w.HTML("<i>(no settings)</i>")]
            current = values()
            shown = {s.name: s.visible(current) for s in specs}
            for spec in specs:
                rows[spec.name].layout.display = None if shown[spec.name] else "none"
            for (is_adv, group), heading in headings.items():
                members = [s for s in specs if s.group == group and s.advanced == is_adv]
                heading.layout.display = None if any(shown[s.name] for s in members) else "none"
            for name, part in sub_advanced.items():
                on = shown[name] and bool(part.children)
                part.heading.layout.display = part.layout.display = None if on else "none"
            if advanced_section is not None:  # shown while some advanced setting applies
                applies = any(shown[s.name] for s in specs if s.advanced) or any(
                    shown[n] and p.children for n, p in sub_advanced.items())
                advanced_section.layout.display = None if applies else "none"
        finally:
            busy[0] = False

    for widget in inputs.values():
        widget.observe(refresh, names="value")
    refresh()

    button = w.Button(description="Run", button_style="primary")
    code = w.HTML()
    out = w.Output()

    def on_click(_: Any) -> None:
        out.clear_output()
        with out:
            try:
                cmd = command_cls(**values())
                result = session.run(cmd)
                recorded = session.history[-1].code if session.history else cmd.to_code()  # as recorded, from the session
                code.value = f"<pre>{html.escape(recorded)}</pre>"
                display(HTML(to_html(result)))
            except Exception as exc:  # shown to the student, not raised
                print(f"Error: {exc}")

    button.on_click(on_click)
    help_button = w.ToggleButton(description="Help", tooltip="help for this form and what is chosen in it")
    help_out = w.Output()

    def show_help(*_: Any) -> None:
        from ..help import dialog_markdown

        help_out.clear_output()
        box.help_text = _notebook_links(dialog_markdown(command_cls, values())) if help_button.value else ""
        if box.help_text:
            with help_out:
                display(Markdown(box.help_text))

    help_button.observe(show_help, names="value")
    for widget in inputs.values():
        widget.observe(lambda change: show_help() if help_button.value else None, names="value")
    title = next((i.menu for i in COMMANDS.values() if i.cls is command_cls), command_cls.__name__)
    box = w.VBox([w.HTML(f"<b>{html.escape(title)}</b>"), *children, w.HBox([button, help_button]), code, out, help_out])
    box.inputs, box.values, box.help_button, box.help_out, box.help_text = inputs, values, help_button, help_out, ""  # for tests
    return box


def _notebook_links(markdown: str) -> str:
    """Notebooks cannot follow ``pymetrix:`` links: show their titles as plain text (use ``show_help`` for those pages)."""
    import re

    return re.sub(r"\[([^\]]*)\]\(pymetrix:[^)]*\)", r"**\1**", markdown)


@renamed_arguments(target="topic")
def show_help(topic: Any = "home") -> Any:
    """Help in a notebook: ``show_help(OLS)``, ``show_help("garch")`` (a topic), ``show_help("pymetrix:methods/linearmodels:liml")``
    or ``show_help()`` for the contents."""
    from IPython.display import Markdown

    from .. import help as hp

    if isinstance(topic, str):
        if topic in ("home", hp.link("home")):  # the contents: sections a notebook folds, one page per line
            return Markdown(_notebook_links(hp.home_markdown(details=True)))
        url = topic if topic.startswith(hp.SCHEME) else hp.link("topic", topic)
        return Markdown(_notebook_links(hp.page_for_link(url)))
    return Markdown(_notebook_links(hp.help_page(topic).to_markdown()))


def menu(session: Session) -> Any:
    w = _widgets()
    from ..plugins import load

    load("commands")
    from ..commands.menus import menu_order

    ordered = menu_order([i for i in COMMANDS.values() if i.menu])  # as the app's menus lay them out
    acc = w.Accordion(children=[form(i.cls, session) for i, _ in ordered])
    for n, (_, path) in enumerate(ordered):
        acc.set_title(n, path)
    return acc
