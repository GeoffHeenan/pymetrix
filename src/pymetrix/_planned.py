"""Marker for components defined in the design but not yet implemented."""

from __future__ import annotations

from typing import NoReturn

from .errors import PlannedError


def planned(what: str, stage: int) -> NoReturn:
    raise PlannedError(
        f"{what} is planned for build stage {stage} (see the design doc, 'Build plan')."
    )
