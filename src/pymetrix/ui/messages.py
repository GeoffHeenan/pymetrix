"""What a dialog says when its step fails: a plain message, a next step, and the technical text kept for Details.

Errors from deep inside the libraries pymetrix uses read as their internals ("zero-size array to reduction operation
maximum which has no identity", "[Errno 21] Is a directory: '.'", "Item wrong length 100 instead of 0."). At the dialog
boundary ``explain`` turns the common ones into what went wrong in the reader's terms and what to try; pymetrix's own
errors (``PyMetrixError``) are already written for the reader and are kept as they are, only sentence-cased. The
original text always stays available, under Details. No Qt here: the Qt dialog shows what this returns.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = ["EMPTY_STATES", "TITLES_SHOWN", "Explained", "explain", "failure_title", "plural", "titled"]

# What each panel says while it has nothing to show, in one pattern: "No X yet: how to get some". Each shows inside
# its own view, at the top; the short form is what a narrow panel shows, the long one its tooltip.
EMPTY_STATES = {
    "objects": "No objects yet: open a demo project or load data (File menu) to begin.",
    "history": "No steps yet: dialogs and typed code are listed here.",
    "history_tip": "No steps yet: each dialog's step, and each line run in the Immediate window or a Code window, is "
                   "listed here, with the Python it recorded.",
    "problems": "No problems: every result is up to date, and no step failed or warned.",
    "report": "No report items yet: use <b>Add to report</b> under a figure or a table, or in a result window.",
    "script": "No steps yet: everything done in the dialogs and the Immediate window appears here as Python, and replays "
              "without the app (File → Export script…).",
    "model_problems": "No problems: the model's text checks out (errors and warnings are listed here, at their line).",
}


def plural(count: int, word: str, many: str | None = None) -> str:
    """"1 equation", "2 equations": the count and the word, made plural (``many``, or with an s) unless it is 1."""
    return f"{count:,} {word if count == 1 else (many or word + 's')}"


@dataclass(frozen=True)
class Explained:
    """``text``: what went wrong, in plain words; ``hint``: what to try next ("" if nothing to add); ``details``: the
    technical text (the exception's type and message)."""

    text: str
    hint: str
    details: str


# macOS draws message boxes without a title, as Apple's guidelines ask: Qt's QMessageBox.setWindowTitle does nothing
# there (its windowTitle() stays ""). A title that says what failed ("Could not save the project") would be lost, so
# on a Mac it leads the box's text instead (tests/test_ui_dialog_checks.py checks this against Qt on each platform).
TITLES_SHOWN = sys.platform != "darwin"


def titled(title: str, text: str) -> tuple[str, str]:
    """(title, text) for a message box whose title carries information: unchanged where message boxes show their
    title; on a Mac, where they do not, the title is also the first line of the text."""
    return title, text if TITLES_SHOWN else f"{title}\n\n{text}"


def failure_title(title: str) -> str:
    """A failed step's message box title: the dialog's name, then what happened ("Forecast: could not run")."""
    title = (title or "").strip().rstrip(".…") or "Step"
    return f"{title}: could not run"


def _sentence(text: str) -> str:
    text = text.strip()
    return text[:1].upper() + text[1:] if text else text


def _message(exc: BaseException) -> str:
    """The exception's own words (a KeyError's without the quotes str() adds)."""
    if isinstance(exc, KeyError) and len(exc.args) == 1 and isinstance(exc.args[0], str):
        return exc.args[0]
    return str(exc)


def _sample(values: Mapping[str, Any]) -> str:
    start, end = str(values.get("start") or "").strip(), str(values.get("end") or "").strip()
    if start or end:
        return f" ({start or 'the first period'}–{end or 'the last period'})"
    return ""


def _path(values: Mapping[str, Any], exc: BaseException) -> str:
    found = getattr(exc, "filename", None)
    if found:
        return str(found)
    for key in ("path", "file", "filename"):
        if values.get(key):
            return str(values[key])
    return ""


def explain(exc: BaseException, values: Mapping[str, Any] | None = None) -> Explained:
    """``exc`` as the reader of a dialog should see it, given the dialog's ``values`` (to name the sample or file)."""
    from ..errors import PyMetrixError

    values = values or {}
    raw = _message(exc)
    details = f"{type(exc).__name__}: {raw}" if raw else type(exc).__name__
    low = raw.lower()
    sample = _sample(values)
    where = _path(values, exc) if isinstance(exc, OSError) else ""
    if (isinstance(exc, IsADirectoryError) or "is a directory" in low
            or (isinstance(exc, PermissionError) and where and os.path.isdir(where))):
        # a folder where a file was wanted: on Windows, opening a folder as a file is refused (PermissionError, errno
        # 13) rather than IsADirectoryError, so the path itself is asked
        where = where or _path(values, exc)
        return Explained(f"Choose a file, not a folder{f' ({where})' if where else ''}.",
                         "Use Browse… to pick the file itself.", details)
    if isinstance(exc, FileNotFoundError) or "no such file or directory" in low:
        where = _path(values, exc)
        return Explained(f"There is no file{f' at {where}' if where else ' with that name'}.",
                         "Check the name and folder, or use Browse… to pick the file.", details)
    if isinstance(exc, PermissionError) or "permission denied" in low:
        return Explained("The file could not be opened: permission was refused.",
                         "Choose a file you can read (or a folder you can write to), or close it in other programs.", details)
    if "must not be nat" in low or ("nat" in low.split() and ("start" in low or "end" in low)):
        return Explained(f"No observations in the chosen sample{sample}: a start or end period is missing or not a period of the data.",
                         "Choose Start and End from the periods offered (the arrows step through them).", details)
    if re.search(r"wrong length:? \d+ instead of \d+|length of values \(\d+\) does not match|length mismatch|unalignable|"
                 r"operands could not be broadcast|shapes? \(\d+,?\) .*not aligned", low):
        return Explained(f"The series don't cover the sample{sample}: they have a different number of values than the "
                         "periods the step uses.",
                         "Check Start and End and the series chosen (Data → Set sample… sets the sample; the spreadsheet "
                         "shows where the series have values).", details)
    if "zero-size array" in low or "need at least one array to concatenate" in low or "no observations" in low \
            or "empty sample" in low or "found array with 0 sample" in low:
        return Explained(f"No data left after removing missing values in the chosen sample{sample}.",
                         "Check that the series chosen have values in the sample (the spreadsheet shows them), or widen "
                         "the sample.", details)
    if isinstance(exc, ZeroDivisionError) or "division by zero" in low:
        return Explained("A calculation divided by zero: a series chosen may be constant, all zero, or have too few values.",
                         "Check the series and the sample chosen.", details)
    if isinstance(exc, (MemoryError,)):
        return Explained("There is not enough memory for this step.", "Try a smaller sample, fewer draws or fewer series.", details)
    known = re.search(r"unknown function '([^']+)' \(known: [^)]*\)", raw)
    if known:  # the full list of functions (about 150 names) is for the details, not the box
        return Explained(_sentence(raw[:known.start()] + f"unknown function '{known.group(1)}'" + raw[known.end():]),
                         "Help → Help contents → Derived series and Distribution functions and moments list the functions "
                         "a formula can use.", details)
    if isinstance(exc, PyMetrixError):  # pymetrix's own words, written for the reader: kept
        return Explained(_sentence(raw) or "The step stopped.", "", details)
    if isinstance(exc, (ValueError, TypeError, KeyError, IndexError, AttributeError, ArithmeticError, LookupError)):
        return Explained("The step stopped: " + (_sentence(raw) or type(exc).__name__),
                         "Check the fields' values; Details has the technical message.", details)
    return Explained(_sentence(raw) or f"The step stopped ({type(exc).__name__}).", "", details)
