"""Conversion of specification and configuration objects to JSON-ready values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from math import inf
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Return a JSON-ready copy of ``obj``.

    Dataclasses become ``{"type": <class name>, <init fields>...}``; enums
    become their values; dates become ISO strings.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {"type": type(obj).__name__}
        for f in fields(obj):
            if f.init:
                out[f.name] = to_jsonable(getattr(obj, f.name))
        return out
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and obj in (inf, -inf):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return repr(obj)
