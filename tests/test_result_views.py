"""Every kind of result reaches each of its views from Python, as the app's tabs show them: ``to_frame(view)`` for each
table view, a clear error for a figure, and a notebook display. The objects come from tests/_gallery.py and the demo
projects (the same 54 kinds test_result_windows.py opens in the app). No Qt needed."""


import pytest

import _gallery
from pymetrix.errors import NotFoundError
from pymetrix.output import Table
from pymetrix.views import view_sections


@pytest.fixture(scope="module")
def objects():
    return dict(_gallery.every_object())  # shared with test_saved_results.py: both only read the objects


def _results(objects):
    """pymetrix's own kinds, less figures and tables (a figure's numbers are .data(); a table is its own view) and
    data (whose to_frame takes series names: the NumPy/pandas bridge)."""
    return {k: o for k, o in objects.items() if type(o).__module__.startswith("pymetrix")
            and k not in ("Figure", "Table") and not any(b.__name__ == "Dataset" for b in type(o).__mro__)}


def test_every_result_gives_each_table_view_as_a_frame(objects):
    failed = []
    for kind, obj in _results(objects).items():
        try:
            obj.to_frame()  # the default: its own, or the first table view
            for title, part in view_sections(obj):
                if isinstance(part, Table):
                    assert obj.to_frame(title.upper()).shape == part.to_frame().shape  # any case
                else:
                    with pytest.raises(NotFoundError, match="figure"):
                        obj.to_frame(title)
        except Exception as exc:  # noqa: BLE001 - the test lists every failure
            failed.append(f"{kind}: {type(exc).__name__}: {exc}")
    assert not failed, failed


def test_every_result_and_dataset_shows_in_a_notebook(objects):
    for kind, obj in objects.items():
        if not type(obj).__module__.startswith("pymetrix"):
            continue
        if hasattr(obj, "_repr_png_"):  # a figure shows as a picture
            assert obj._repr_png_()[1:4] == b"PNG", kind  # the PNG signature
            continue
        html = obj._repr_html_()
        assert isinstance(html, str) and "<" in html, kind


def test_an_unknown_view_lists_the_views(objects):
    with pytest.raises(NotFoundError, match="views:.*Summary"):
        objects["FittedVECM"].to_frame("no such view")


def test_every_result_copies_its_summary_as_text(objects):
    """Copy summary in a result window: the result as plain text, from its summary() or its first table."""
    from pymetrix.views import summary_text

    for kind, obj in _results(objects).items():
        text = summary_text(obj)
        assert isinstance(text, str) and len(text.strip()) > 10 and text != repr(obj), kind  # a summary, not the bare repr
