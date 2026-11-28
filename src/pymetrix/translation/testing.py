"""Checks for a front end's tests: every line has exactly one row and every row left out or in error a reason
(``assert_rows_cover``); the status counts, to fix per fixture program (``status_counts``); and the script run as
Python and the steps run in a session giving equal objects (``assert_steps_equal_script``, with ``assert_same``).

    from pymetrix.translation.testing import assert_rows_cover, assert_steps_equal_script, status_counts

    t = translate_toy(TEXT)
    assert_rows_cover(t)
    assert status_counts(t) == {"translated": 2, "translated with a note": 0, "left out": 1, "error": 0}
    namespace, out = assert_steps_equal_script(t)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from .report import ERROR, LEFT

__all__ = ["assert_rows_cover", "assert_same", "assert_steps_equal_script", "status_counts"]


def status_counts(translation: Any) -> dict[str, int]:
    """How many lines have each status, every status listed (``report.counts()``)."""
    return translation.report.counts()


def assert_rows_cover(translation: Any) -> None:
    """Each line with text of the program, and of each included one (``translation.sources``), has exactly one row,
    in order, with the line's text; each row left out or in error has a reason."""
    report = translation.report
    by_file: dict[str, list[Any]] = {}
    for row in report:
        by_file.setdefault(row.file, []).append(row)
        assert row.status not in (LEFT, ERROR) or row.reason, f"{row.file or 'the program'} line {row.line}: {row.status} without a reason"
    sources = dict(translation.sources) or {translation.program: ""}
    for file, text in sources.items():
        rows = by_file.pop(file, [])
        if not text and not rows:
            continue
        expected = [(n, raw) for n, raw in enumerate(text.split("\n"), start=1) if raw.strip()]
        got = [(r.line, r.source) for r in rows]
        assert got == expected, f"{file or 'the program'}: the rows are not one per line with text, in order"
    assert not by_file, f"rows of files the translation has no text of: {sorted(by_file)}"


def assert_same(a: Any, b: Any) -> None:
    """Two objects equal as the script and the steps make them: datasets by their frames and samples, matrices by
    their values, results by their summaries (an estimate's parameters exactly), NaN equal to NaN, the rest by ``==``."""
    import numpy as np
    import pandas as pd

    from pymetrix import Dataset, Matrix

    if isinstance(a, Dataset):
        assert isinstance(b, Dataset)
        pd.testing.assert_frame_equal(a.frame, b.frame)
        assert a.sample == b.sample
    elif isinstance(a, Matrix):
        np.testing.assert_array_equal(a.values, b.values)
    elif isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float) and math.isnan(b)
    elif hasattr(a, "names") and hasattr(a, "data"):  # a Selection
        assert list(a.names) == list(b.names)
    elif hasattr(a, "params") and hasattr(a, "summary"):  # a fitted estimate
        assert type(a) is type(b)
        np.testing.assert_array_equal(np.asarray(a.params), np.asarray(b.params))
        assert str(a.summary()) == str(b.summary())
    else:
        assert a == b, (a, b)


def assert_steps_equal_script(translation: Any, session: Any = None, same: Callable[[Any, Any], None] = assert_same,
                              **run: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the script as Python (with the objects given to the translation; a ``stop``, ``SystemExit``, ends it) and
    the steps in ``session`` (``translation.run(session, **run)``): every value and object the steps make must be
    made by the script, and equal (``same``). Returns the script's namespace and what ``run`` returned."""
    namespace: dict[str, Any] = dict(translation.objects)
    try:
        exec(compile(translation.script, "<script>", "exec"), namespace)  # noqa: S102 - the translation's own script
    except SystemExit:
        pass
    out = translation.run(session, **run)
    s = out["session"]
    made = {**{k: v for k, v in out.items() if k != "session"}, **s.objects}
    for name, value in made.items():
        assert name in namespace, f"{name} is made by the steps but not by the script"
        same(value, namespace[name])
    return namespace, out
