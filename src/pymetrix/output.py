"""Plain output objects. Frontends and exporters decide how to render them."""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def _fmt(v: Any, digits: int = 4) -> str:
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "" if np.isnan(v) else str(v)
        if v != 0 and (abs(v) >= 1e6 or abs(v) < 10.0 ** -digits):
            return f"{v:.{digits}e}"
        return f"{v:.{digits}f}"
    return "" if v is None else str(v)


def bounds_text(low: float, high: float) -> str:
    """A parameter's bounds as read: [0, 1], [0, ∞), (-∞, 2]; empty when unbounded both ways. Bounds of 1e8 or more
    (how some models write "no upper bound") count as infinite."""
    lo = -np.inf if low <= -1e8 else low
    hi = np.inf if high >= 1e8 else high
    if np.isinf(lo) and np.isinf(hi):
        return ""
    return ("(-∞" if np.isinf(lo) else f"[{lo:g}") + ", " + ("∞)" if np.isinf(hi) else f"{hi:g}]")


@dataclass(frozen=True, eq=False)
class Table:
    columns: Sequence[str]
    rows: Sequence[Sequence[Any]]
    index: Sequence[str] | None = None
    title: str = ""
    notes: Sequence[str] = ()
    index_name: str = ""

    @classmethod
    def from_frame(cls, df: pd.DataFrame, title: str = "", notes: Sequence[str] = ()) -> Table:
        return cls(
            columns=[str(c) for c in df.columns],
            rows=df.to_numpy().tolist(),
            index=[str(i) for i in df.index],
            title=title,
            notes=tuple(notes),
            index_name=str(df.index.name or ""),
        )

    def to_frame(self) -> pd.DataFrame:
        """The table as a pandas DataFrame: one column per table column, indexed by the row labels if it has them."""
        df = pd.DataFrame(list(self.rows), columns=list(self.columns))
        if self.index is not None:
            df.index = pd.Index(list(self.index), name=self.index_name or None)
        return df

    def _flat(self) -> pd.DataFrame:
        """The table with its row labels as the first column."""
        frame = self.to_frame()
        if self.index is not None:
            frame = frame.reset_index().rename(columns={"index": self.index_name or ""})
        return frame

    def to_csv(self, sep: str = ",", header: bool = True) -> str:
        """The numbers as text, at full precision: ``sep="\\t"`` pastes into Excel. Missing values are blank."""
        return self._flat().to_csv(sep=sep, header=header, index=False, lineterminator="\n")

    def to_markdown(self, digits: int = 4) -> str:
        """A Markdown table (numbers right-aligned, at ``digits`` decimals), followed by the notes."""
        def cell(v: Any) -> str:
            return _fmt(v, digits).replace("|", "\\|").replace("\n", " ")

        def number(v: Any) -> bool:
            return isinstance(v, (int, float, np.number)) and not isinstance(v, bool)

        frame = self._flat()
        numeric = [len(frame) > 0 and frame.iloc[:, j].map(number).all() for j in range(frame.shape[1])]  # by position: names can repeat
        lines = ["| " + " | ".join(cell(c) for c in frame.columns) + " |",
                 "|" + "|".join("---:" if n else "---" for n in numeric) + "|"]
        lines += ["| " + " | ".join(cell(v) for v in values) + " |" for values in frame.itertuples(index=False)]
        if self.notes:
            lines += [""] + [f"*{n}*  " for n in self.notes]
        return "\n".join(lines)

    def _cells(self) -> list[list[str]]:
        head = ([self.index_name] if self.index is not None else []) + list(self.columns)
        body = []
        for i, row in enumerate(self.rows):
            label = [str(self.index[i])] if self.index is not None else []
            body.append(label + [_fmt(v) for v in row])
        return [head, *body]

    def __str__(self) -> str:
        cells = self._cells()
        widths = [max(len(r[j]) for r in cells) for j in range(len(cells[0]))]
        lines = []
        if self.title:
            lines.append(self.title)
        for n, r in enumerate(cells):
            first = r[0].ljust(widths[0]) if self.index is not None else r[0].rjust(widths[0])
            rest = [c.rjust(w) for c, w in zip(r[1:], widths[1:])]
            lines.append("  ".join([first, *rest]))
            if n == 0:
                lines.append("-" * len(lines[-1]))
        lines.extend(self.notes)
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """The table as HTML, as wide as its content (numbers right-aligned, every other row shaded), then its notes:
        "Label: value" statistics grouped under headings, the other notes as sentences (``pymetrix.summary_layout``)."""
        from .summary_layout import notes_html, numeric_columns, table_html

        cells = self._cells()
        numeric = numeric_columns(self.rows, len(self.columns))
        if self.index is not None:
            numeric = [False, *numeric]
        out = []
        if self.title:
            out.append(f"<p><b>{html.escape(self.title)}</b></p>")
        out.append(table_html(cells[0], cells[1:], numeric))
        if self.notes:
            out.append(notes_html(self.notes))
        return "\n".join(out)


from .figures import Figure  # noqa: E402  (charts live in figures.py)


# show(): results on screen, wherever the program runs

_show_handler: Any = None
_quiet = 0


def show(obj: Any, title: str | None = None) -> None:
    """Show a result: its window in the desktop app, rich output in a notebook, text in a terminal.

    ``show(fit)``, ``show(fits.coefficients())``, ``show(data, "Revised data")``. Nothing is shown
    while a project is being opened or results are being recomputed.
    """
    if _quiet:
        return
    if _show_handler is not None:
        _show_handler(obj, title)
        return
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            from IPython.display import display

            display(obj)
            return
    except ImportError:
        pass
    if title:
        print(title)
    summary = getattr(obj, "summary", None)
    print(summary() if callable(summary) else obj)


def set_show_handler(handler: Any) -> Any:
    """Route ``show`` to a frontend (the desktop app opens windows); returns the previous handler."""
    global _show_handler
    previous, _show_handler = _show_handler, handler
    return previous


class quiet_show:
    """``with quiet_show():`` -- no windows from ``show`` (replaying a project, recomputing)."""

    def __enter__(self) -> None:
        global _quiet
        _quiet += 1

    def __exit__(self, *exc: Any) -> None:
        global _quiet
        _quiet -= 1
