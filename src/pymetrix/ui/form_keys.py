"""Access keys on a command dialog's field labels: Alt and a letter goes to a field, as in any desktop dialog.

The letters are given by ``pymetrix.forms.access_keys`` from the labels' own words (the wording table's display
labels), unique within the dialog and none of the keys its buttons (OK, Cancel, Help) already use. Each label's buddy
is the field's input itself (the text box in a field with a message line, the path in a file field, the chosen radio
button), not the box around it, so Alt+letter puts the cursor in the field. Visible fields take letters first; the
letters are given again when fields show or hide (a sub-form's settings change with the method chosen). The
Advanced settings header has a key of its own, and a field hidden in that closed section keeps its key: pressing it
opens the section and goes to the field.
"""

from __future__ import annotations

from typing import Any

__all__ = ["assign_keys", "input_widget"]

BUTTON_KEYS = {"o", "c", "h"}  # OK, Cancel and Help: their keys where the platform's style gives them (&OK...)
PLAIN = "plain_label"  # the label's text without its key, kept on the label (a Qt property)


def input_widget(editor: Any) -> Any:
    """The widget that takes the keys for ``editor`` (a ``_Editor``), or None for a sub-form (its settings have their
    own labels)."""
    spec = editor.spec
    if spec.kind == "options":
        return None
    if spec.kind in ("file", "savefile"):
        return editor.path
    if spec.radio:
        buttons = editor.group.buttons()
        return editor.group.checkedButton() or (buttons[0] if buttons else None)
    return editor.widget


def _rows(editors: dict[str, Any], forms: dict[str, Any]) -> list[tuple[Any, Any, Any]]:
    """(label, input, editor) for each field with a label, sub-forms' settings after the field that holds them."""
    out = []
    for name, editor in editors.items():
        form = forms.get(name)
        label = form.labelForField(editor.outer) if form is not None else None
        target = input_widget(editor)
        if label is not None and target is not None and hasattr(label, "setBuddy"):
            out.append((label, target, editor))
        if editor.spec.kind == "options":
            subs = {n: e for n, e in getattr(editor, "sub", {}).items()}
            sub_forms = {n: (editor.advanced_form if e.spec.advanced else editor.inner) for n, e in subs.items()}
            out += _rows(subs, sub_forms)
    return out


def assign_keys(dialog: Any, editors: dict[str, Any], forms: dict[str, Any], buttons: Any,
                section: Any = None) -> list[tuple[Any, Any]]:
    """Give every field label of ``dialog`` an access key and its input as buddy; ``forms``: each field's form layout;
    ``buttons``: the dialog's QDialogButtonBox (its buttons' keys are left to them). Returns (label, input) pairs.

    ``section``: the dialog's Advanced settings section (``pymetrix.ui.advanced_section``), or None. Its header takes
    a key too, after the fields in view and before the fields in it, and the fields in it keep their keys whether it
    is open or closed: while it is closed (their labels hidden), a field's key opens it and goes to the field."""
    from ..forms import access_key, access_keys

    taken = ({access_key(b.text()) for b in buttons.buttons()} | BUTTON_KEYS) - {""}
    rows = _rows(editors, forms)
    for label, _, _ in rows:
        if label.property(PLAIN) is None:
            label.setProperty(PLAIN, label.text())
    with_section = section is not None and section.applies()

    def rank(i: int) -> int:  # fields in view, the header, the fields in the section, then the fields not in view
        editor = rows[i][2]
        if editor.spec.advanced and with_section:
            return 2
        return 0 if editor.outer.isVisibleTo(dialog) else 3

    order = sorted(range(len(rows)), key=rank)
    plain = [rows[i][0].property(PLAIN) for i in order]
    at = sum(1 for i in order if rank(i) == 0)
    if with_section:
        plain.insert(at, section.TEXT)
    texts = access_keys(plain, taken)
    if with_section:
        section.set_key(texts.pop(at))
    for i, text in zip(order, texts):
        label, target, _ = rows[i]
        if label.text() != text:
            label.setText(text)
        if label.buddy() is not target:
            label.setBuddy(target)
    if section is not None:
        section.reveal_keys([(rows[i][0].text(), rows[i][1]) for i in order if rank(i) == 2])
    return [(label, target) for label, target, _ in rows]
