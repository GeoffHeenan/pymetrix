"""Every kind of object that has a window opens in one, tab by tab, figures drawn (off-screen Qt). The objects come
from tests/_gallery.py and the demo projects: one of each of the 54 kinds the views know. A view that crashes, as
temporal disaggregation's Fields tab did, fails here rather than in front of a reviewer."""

import os
import importlib.util
import warnings

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("qtconsole")

import _gallery  # noqa: E402


@pytest.fixture(scope="module")
def demo_objects():
    from pymetrix import demo

    found = {}
    for name in ("consumption", "forecast_round", "nk_estimation", "policy_and_nonlinear"):
        for obj in demo(name).objects.values():
            found.setdefault(type(obj).__name__, obj)
    return found


def _open(obj):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.ui.qt import object_view

    view = object_view(obj, name="x", code="x = ...")
    tabs = view.titles()  # the views listed down the window's side
    assert tabs, f"{type(obj).__name__}: no tabs"
    view.deleteLater()
    return tabs


@pytest.mark.parametrize("kind", sorted(_gallery.BUILDERS))
def test_built_objects_open_in_a_window(kind):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence chatter from small examples
        obj = _gallery.build(kind)
    assert type(obj).__name__ == kind
    _open(obj)


def test_demo_objects_open_in_a_window(demo_objects):
    for kind, obj in demo_objects.items():
        _open(obj)


def test_every_kind_with_a_window_is_covered(demo_objects):
    """A new kind of result with views but no example here would go unchecked: add it to tests/_gallery.py."""
    from pymetrix.views import sections

    known = {t.__name__ for t in sections.registry if t is not object}
    if importlib.util.find_spec("sklearn") is None:  # the scikit-learn plugin's results need it to be built
        known.discard("FittedLearner")
    covered = set(_gallery.BUILDERS) | set(demo_objects)
    assert not known - covered, sorted(known - covered)
