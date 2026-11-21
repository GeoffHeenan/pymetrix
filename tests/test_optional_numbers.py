"""Optional number fields (``int | None``: empty means automatic) in the desktop dialogs and Jupyter forms: left empty
they give None, not the spin box's minimum (STL's trend length came back as 3); typed values come back as typed,
whole numbers as int; a step reopened with None shows empty again."""

import os

import pytest

from pymetrix.forms import settings_form
from pymetrix.plugins import registry


def _specs(method):
    return {s.name: s for s in settings_form(registry("seasonal").get(method))}


def test_the_desktop_dialog():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import _Editor, _qt

    stl = _specs("STL")
    trend = _Editor(_qt(), stl["trend"])
    assert trend.get() is None and trend.widget.text() == "automatic"  # its help: "empty = automatic"
    trend.set(25)
    assert trend.get() == 25 and isinstance(trend.get(), int)
    trend.set(None)
    assert trend.get() is None
    seasonal = _Editor(_qt(), stl["seasonal"])  # not optional: unchanged, a whole number
    assert seasonal.get() == 7 and isinstance(seasonal.get(), int)
    if registry("seasonal").find("X13") is None:
        return  # X-13's optional decimal and its "none": with the seasonal plugin installed
    x13 = _specs("X13")
    critical = _Editor(_qt(), x13["critical"])
    assert critical.get() is None
    critical.set(3.5)
    assert critical.get() == pytest.approx(3.5)
    easter = _Editor(_qt(), x13["easter"])  # 0 = none, as before
    assert easter.get() == 0 and easter.widget.text() == "none"


def test_the_jupyter_form():
    pytest.importorskip("ipywidgets")
    from pymetrix.ui.jupyter import _value, _widget, _widgets

    w = _widgets()
    spec = _specs("STL")["trend"]
    field = _widget(w, spec)
    assert _value(field, spec) is None
    field.value = "25"
    assert _value(field, spec) == 25 and isinstance(_value(field, spec), int)
