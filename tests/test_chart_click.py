"""Click on a chart to identify what is there (off-screen Qt, matplotlib's own mouse and key events): a series' name,
or a data point's period and value with the point marked; a bar's period and value; a heatmap cell's row, column and
value. Empty space or Esc takes the box away; one box at a time; never recorded, never exported. The running x, y
readout is off unless Values is switched on."""

import os

import numpy as np
import pandas as pd
import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def data():
    from pymetrix import Dataset

    rng = np.random.default_rng(3)
    idx = pd.period_range("2015Q1", periods=24, freq="Q", name="time")
    return Dataset.from_frame(pd.DataFrame({"c": 100 + np.cumsum(rng.normal(1, 1, 24)),
                                            "y": 150 + np.cumsum(rng.normal(1, 1, 24))}, index=idx))


def _window(figure, **kw):
    _app()
    from pymetrix.ui.qt import figure_view

    page = figure_view(figure, **kw)
    page.resize(800, 600)
    page.show()
    _app().processEvents()
    page.canvas.draw()
    return page


def _click(canvas, px, py, button=1):
    from matplotlib.backend_bases import MouseEvent

    MouseEvent("button_press_event", canvas, px, py, button=button)._process()


def _key(canvas, key):
    from matplotlib.backend_bases import KeyEvent

    KeyEvent("key_press_event", canvas, key)._process()


def _line(page, label):
    return next(l for ax in page.canvas.figure.axes for l in ax.get_lines() if l.get_label() == label)


def _pixel(line, k):
    """The pixel of the line's k-th data point."""
    return line.axes.transData.transform((line.get_xdata(orig=False)[k], line.get_ydata(orig=False)[k]))


def test_clicking_a_point_gives_its_period_and_value_and_marks_it(data):
    page = _window(data.plot(["c", "y"]))
    line = _line(page, "c")
    px, py = _pixel(line, 6)
    _click(page.canvas, px + 2, py + 1)  # a couple of pixels off: still the point
    text = page.identify.text()
    assert text.splitlines()[0] == "c"
    assert text.splitlines()[1] == f"2016Q3: {data['c'].loc['2016Q3']:.4g}"
    assert page.identify.marker is not None
    mx, my = page.identify.marker.get_xdata()[0], page.identify.marker.get_ydata()[0]
    assert (mx, my) == (line.get_xdata(orig=False)[6], line.get_ydata(orig=False)[6])


def test_clicking_a_line_between_points_names_the_series(data):
    page = _window(data.plot(["c", "y"]))
    line = _line(page, "y")
    a, b = _pixel(line, 10), _pixel(line, 11)
    middle = (a + b) / 2
    _click(page.canvas, *middle)
    assert page.identify.text() == "y"
    assert page.identify.marker is None


def test_empty_space_and_escape_dismiss_and_only_one_box_shows(data):
    page = _window(data.plot(["c", "y"]))
    ax = _line(page, "c").axes
    _click(page.canvas, *_pixel(_line(page, "c"), 3))
    _click(page.canvas, *_pixel(_line(page, "y"), 20))
    boxes = [t for t in ax.texts if t.get_gid() == "pymetrix-identify"]
    assert len(boxes) == 1 and page.identify.text().startswith("y")
    box = ax.get_window_extent()
    _click(page.canvas, box.x0 + 3, box.y1 - 3)  # a corner of the panel, far from both lines
    assert page.identify.text() == "" and not [t for t in ax.texts if t.get_gid() == "pymetrix-identify"]
    _click(page.canvas, *_pixel(_line(page, "c"), 3))
    assert page.identify.text()
    _key(page.canvas, "escape")
    assert page.identify.text() == "" and page.identify.marker is None


def test_the_box_stays_inside_the_panel_near_its_edges(data):
    page = _window(data.plot(["c", "y"]))
    line = _line(page, "y")
    _click(page.canvas, *_pixel(line, 23))  # the last point: at the right edge
    page.canvas.draw()
    box = page.identify.box.get_window_extent()
    panel = line.axes.get_window_extent()
    assert box.x1 <= panel.x1 + 1 and box.x0 >= panel.x0 - 1


def test_bars_and_heatmaps(data):
    from pymetrix.figures import heatmap_figure

    page = _window(data.plot(["c"], kind="bar"))
    ax = page.canvas.figure.axes[0]
    patch = ax.containers[0].patches[5]
    cx, cy = ax.transData.transform((patch.get_x() + patch.get_width() / 2, patch.get_y() + patch.get_height() / 2))
    _click(page.canvas, cx, cy)
    assert page.identify.text().splitlines()[-1] == f"2016Q2: {data['c'].loc['2016Q2']:.4g}"
    table = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["a", "b"], columns=["a", "b"])
    page = _window(heatmap_figure(table, title="corr"))
    ax = next(a for a in page.canvas.figure.axes if a.get_images())
    _click(page.canvas, *ax.transData.transform((1, 0)))
    assert page.identify.text() == "a, b\n0.5"


def test_every_panel_answers(data):
    page = _window(data.plot(["c", "y"], layout="each"))
    for label in ("c", "y"):
        line = _line(page, label)
        _click(page.canvas, *_pixel(line, 4))
        assert page.identify.text().startswith(label) and page.identify.box.axes is line.axes


def test_identifying_records_nothing_and_exports_without_the_box(data):
    figure = data.plot(["c", "y"])
    before = figure.png()
    page = _window(figure, on_edit=lambda kind: None)
    _click(page.canvas, *_pixel(_line(page, "c"), 6))
    assert page.identify.text()
    assert figure.style == () and figure.style_steps() == []
    assert figure.png() == before  # drawn afresh from the figure: no box, no marker
    page.copy = next(b for b in page.findChildren(QtWidgets.QPushButton) if b.text() == "Copy image")
    page.copy.click()
    assert not _app().clipboard().image().isNull()
    lines = [l.get_label() for ax in figure.draw().axes for l in ax.get_lines()]
    assert "_identify" not in lines


def test_add_a_note_and_pan_take_the_click_instead(data):
    placed = []
    page = _window(data.plot(["c", "y"]), on_place=lambda *a: placed.append(a))
    page.place.setChecked(True)
    _click(page.canvas, *_pixel(_line(page, "c"), 6))
    assert placed and page.identify.text() == ""


def test_the_running_readout_is_off_until_values_is_switched_on(data):
    from matplotlib.backend_bases import MouseEvent

    page = _window(data.plot(["c", "y"]))
    assert not page.values_action.isChecked() and not page.readout.isVisible()
    assert not getattr(page.toolbar, "coordinates", False)  # matplotlib's own x, y readout is off too
    px, py = _pixel(_line(page, "c"), 6)
    MouseEvent("motion_notify_event", page.canvas, px, py)._process()
    assert page.readout.text() == " "
    page.values_action.setChecked(True)
    MouseEvent("motion_notify_event", page.canvas, px, py)._process()
    assert page.readout.isVisible() and page.readout.text().startswith("c  2016Q3")
    page.values_action.setChecked(False)
    assert not page.readout.isVisible()


def test_areas_and_scatter_points(data):
    from matplotlib.collections import PathCollection, PolyCollection

    page = _window(data.plot(["c", "y"], kind="area"))
    ax = page.canvas.figure.axes[0]
    area = next(c for c in ax.collections if isinstance(c, PolyCollection) and c.get_label() == "c")
    inside = area.get_paths()[0].vertices.mean(axis=0)
    _click(page.canvas, *ax.transData.transform(inside))
    assert page.identify.text().splitlines()[0] == "c"
    page = _window(data.plot(["c", "y"], kind="scatter"))
    ax = page.canvas.figure.axes[0]
    points = next(c for c in ax.collections if isinstance(c, PathCollection))
    x, y = points.get_offsets()[4]
    _click(page.canvas, *ax.transData.transform((x, y)))
    lines = page.identify.text().splitlines()
    assert len(lines) == 2 and lines[1].endswith(f"{y:.4g}") and page.identify.marker is not None
