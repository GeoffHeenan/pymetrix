"""A result table's columns are sized, and its rows fitted, once, whatever its number of columns.

Sizing each column (``_size_columns``) resizes a header section, and a resized section asks for the table's frame to
follow (``_fit_height``, which fits every row). One fit per column made a wide table quadratic: the solution of a
200-equation model (200 columns, 40 rows) took half a minute to open, and as long again on each change of the text
size. Counted here, not timed: the row fits asked for while a table is filled, for 10 and for 300 columns, are the
same few; a column widened by hand still makes the frame follow."""

import os

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")


def _app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def fits(monkeypatch) -> list[int]:
    """Each row fit asked for from Python (``resizeRowsToContents``), counted."""
    _app()
    asked: list[int] = []
    original = QtWidgets.QTableWidget.resizeRowsToContents
    monkeypatch.setattr(QtWidgets.QTableWidget, "resizeRowsToContents", lambda self: (asked.append(1), original(self))[1])
    return asked


def _filled(columns: int, fits: list[int]) -> tuple[object, int]:
    """A table of 10 rows and ``columns`` columns filled and shown, and the row fits that asked for."""
    from pymetrix.output import Table
    from pymetrix.ui.qt import _fill_grid, _qt

    fits.clear()
    table = Table([f"v{c}" for c in range(columns)], [[r + c / 7 for c in range(columns)] for r in range(10)],
                  index=[str(r) for r in range(10)], title="wide")
    grid = QtWidgets.QTableWidget(10, columns)
    grid.setHorizontalHeaderLabels(table.columns)
    _fill_grid(_qt(), grid, table.to_frame(), table)
    grid.resize(600, 300)
    grid.show()
    _app().processEvents()
    return grid, len(fits)


def test_the_rows_are_fitted_a_few_times_whatever_the_columns(fits):
    from pymetrix.ui.qt import _fit_height

    narrow, few = _filled(10, fits)
    wide, many = _filled(300, fits)
    assert many == few <= 6  # the same few for 300 columns as for 10 (it was one a column, and more)
    for grid in (narrow, wide):
        width, height = grid.maximumWidth(), grid.maximumHeight()
        _fit_height(grid)  # the frame was fitted to the final columns: fitting again changes nothing
        assert (grid.maximumWidth(), grid.maximumHeight()) == (width, height)


def test_a_column_widened_by_hand_still_moves_the_frame(fits):
    grid, _ = _filled(3, fits)
    before, width = len(fits), grid.maximumWidth()
    grid.horizontalHeader().resizeSection(1, grid.columnWidth(1) + 40)
    _app().processEvents()
    assert len(fits) > before and grid.maximumWidth() == width + 40
