"""EViews's date numbers and date strings, for translated programs: ``@dateval``, ``@datestr``, ``@dateadd``,
``@datediff``, ``@dtoo`` and ``@otod`` act on a program's ``%`` strings and ``!`` numbers, so the translation
(``translate_program``) works them out as Python values, as it does the string functions: when the program's own
values are known before it runs, at translation time; otherwise when the script runs, with these functions
(``import pymetrix_eviews.dates as eviews_dates`` in the script).

A date number is EViews's: the number of days since Monday 1 January A.D. 1 in the proleptic Gregorian calendar,
that day being 0 (EViews's Command Reference, "Dates"; its example, ``@dateval("12/1/1999", "dd/mm/yyyy")`` =
729765, is 12 January 1999). It is Python's ``date.toordinal()`` minus 1. Only whole days are read (EViews keeps the
time of day in the fraction).

Formats: ``yyyy`` and ``yy`` (two-digit years from 30 are 19xx, others 20xx), ``q`` (the quarter), ``mm`` and ``m``
(the month), ``dd`` and ``d`` (the day), ``f``/``F`` (the period's letter: ``q`` in a format with a quarter, ``m``
in one with a month only; lower or upper case as written), text in square brackets as it is (``yyyy[Q]q``), and
spaces and ``: - / .`` as they are. A period is its first day: ``"2001q3"`` is 1 July 2001. Without a format, a
date string is read as ``2001q3``, ``2001m7``, ``2001``, ``2001-07-01`` or ``7/1/2001`` (month first).
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta
from typing import Any

__all__ = ["dateadd", "datediff", "datestr", "dateval", "dtoo", "otod"]

_TOKENS = re.compile(r"\[[^\]]*\]|yyyy|yy|mm|dd|[qmdf]|.", re.I | re.S)


def _number(d: date) -> float:
    return float(d.toordinal() - 1)


def _date(number: Any) -> date:
    n = float(number)
    if n != n:
        raise ValueError("an NA date number")
    return date.fromordinal(int(n // 1) + 1)


def _year(text: str) -> int:
    y = int(text)
    return y if len(text) > 2 else (1900 + y if y >= 30 else 2000 + y)


def _tokens(fmt: str) -> list[str]:
    return _TOKENS.findall(fmt)


def dateval(text: str, fmt: str | None = None) -> float:
    """``@dateval``: the date number of a date string, read with ``fmt`` (EViews's date format) or, without one, as
    a period or a calendar date (the module's docstring)."""
    s = str(text).strip()
    if fmt is None:
        m = re.fullmatch(r"(\d{4})\s*([qQmM])\s*(\d{1,2})", s)
        if m:
            year, kind, n = int(m.group(1)), m.group(2).lower(), int(m.group(3))
            if kind == "q" and 1 <= n <= 4:
                return _number(date(year, 3 * n - 2, 1))
            if kind == "m" and 1 <= n <= 12:
                return _number(date(year, n, 1))
        if re.fullmatch(r"\d{4}", s):
            return _number(date(int(s), 1, 1))
        m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
        if m:
            return _number(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})", s)
        if m:
            return _number(date(_year(m.group(3)), int(m.group(1)), int(m.group(2))))
        raise ValueError(f"{s!r} is not a date read without a format: 2001q3, 2001m7, 2001, 2001-07-01 or 7/1/2001")
    pattern, fields = "", []
    for tok in _tokens(fmt):
        low = tok.lower()
        if tok.startswith("["):
            pattern += re.escape(tok[1:-1])
        elif low in ("yyyy", "yy", "mm", "dd", "q", "m", "d"):
            pattern += {"yyyy": r"(\d{4})", "yy": r"(\d{2})", "mm": r"(\d{1,2})", "dd": r"(\d{1,2})", "q": r"(\d)",
                        "m": r"(\d{1,2})", "d": r"(\d{1,2})"}[low]
            fields.append(low)
        elif low == "f":
            pattern += r"[a-z]"
        elif tok.isspace():
            pattern += r"\s*"
        else:
            pattern += re.escape(tok)
    m = re.fullmatch(pattern, s, re.I)
    if m is None:
        raise ValueError(f"{s!r} does not read with the format {fmt!r}")
    got = dict(zip(fields, m.groups()))
    year = _year(got.get("yyyy") or got["yy"]) if ("yyyy" in got or "yy" in got) else 1
    month = int(got.get("mm") or got.get("m") or 1)
    if "q" in got:
        month = 3 * int(got["q"]) - 2
    day = int(got.get("dd") or got.get("d") or 1)
    return _number(date(year, month, day))


def datestr(number: Any, fmt: str) -> str:
    """``@datestr``: the date number as text in EViews's date format ``fmt``."""
    d = _date(number)
    tokens = _tokens(fmt)
    lows = [t.lower() for t in tokens]
    letter = "q" if "q" in lows else "m" if ("m" in lows or "mm" in lows) else "a"
    out = []
    for tok, low in zip(tokens, lows):
        if tok.startswith("["):
            out.append(tok[1:-1])
        elif low == "yyyy":
            out.append(f"{d.year:04d}")
        elif low == "yy":
            out.append(f"{d.year % 100:02d}")
        elif low == "q":
            out.append(str((d.month - 1) // 3 + 1))
        elif low == "mm":
            out.append(f"{d.month:02d}")
        elif low == "m":
            out.append(str(d.month))
        elif low == "dd":
            out.append(f"{d.day:02d}")
        elif low == "d":
            out.append(str(d.day))
        elif low == "f":
            out.append(letter.upper() if tok == "F" else letter)
        else:
            out.append(tok)
    return "".join(out)


_PER_YEAR = {"a": 1, "y": 1, "s": 2, "q": 4, "m": 12}


def _unit(u: str) -> str:
    low = str(u).strip().lower()
    if low in _PER_YEAR or low in ("w", "d"):
        return low
    raise ValueError(f"{u!r} is not a unit read here: a (or y), s, q, m, w or d")


def dateadd(number: Any, n: Any, u: str) -> float:
    """``@dateadd``: the date number ``n`` units ``u`` later (a, y, s, q, m: calendar periods, the day of the month
    kept or, past the month's end, its last day; w: weeks; d: days)."""
    d, unit, k = _date(number), _unit(u), int(float(n))
    if unit == "d":
        return _number(d + timedelta(days=k))
    if unit == "w":
        return _number(d + timedelta(weeks=k))
    months = d.year * 12 + d.month - 1 + k * 12 // _PER_YEAR[unit]
    year, month = divmod(months, 12)
    return _number(date(year, month + 1, min(d.day, calendar.monthrange(year, month + 1)[1])))


def datediff(first: Any, second: Any, u: str) -> float:
    """``@datediff``: how many units ``u`` the first date number is after the second (negative when before): whole
    calendar periods between the periods they fall in for a, y, s, q, m; whole weeks or days for w and d."""
    a, b, unit = _date(first), _date(second), _unit(u)
    if unit == "d":
        return float((a - b).days)
    if unit == "w":
        days = (a - b).days
        return float(days // 7 if days >= 0 else -((-days) // 7))
    per = _PER_YEAR[unit]

    def period(d: date) -> int:
        return d.year * per + (d.month - 1) * per // 12
    return float(period(a) - period(b))


def _periods(data: Any) -> Any:
    t = data.time_axis
    if t is None:
        raise ValueError("@dtoo and @otod need a dated page")
    return data.frame.index.get_level_values(t.name).unique().sort_values()


def dtoo(data: Any, text: str) -> float:
    """``@dtoo``: the observation number (from 1) of a date in the page ``data`` (a Dataset): its period's place
    among the page's periods."""
    import pandas as pd

    periods = _periods(data)
    when = pd.Timestamp(_date(dateval(text)))
    hits = [i for i, p in enumerate(periods) if p.start_time <= when <= p.end_time]
    if not hits:
        raise ValueError(f"{text!r} is not a period of the page ({periods[0]} to {periods[-1]})")
    return float(hits[0] + 1)


def otod(data: Any, n: Any) -> str:
    """``@otod``: the date of observation ``n`` (from 1) of the page ``data``, as pandas writes the period
    (``2001Q3``, ``2001-07``, ``2001``)."""
    periods = _periods(data)
    i = int(float(n))
    if not 1 <= i <= len(periods):
        raise ValueError(f"observation {i} is outside the page (1 to {len(periods)})")
    return str(periods[i - 1])
