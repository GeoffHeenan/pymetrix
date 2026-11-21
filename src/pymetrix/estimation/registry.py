"""Estimation methods for the Estimate dialog: a view of ``registry("methods")``.

Each method class declares ``label`` and ``kind`` ("ls", "iv", "ml",
"panel"), and ``requires`` when it needs another library. The dialog lists
the methods whose libraries are installed; the method's own fields (other
than those on the main form) become its settings sub-form.
"""

from __future__ import annotations

from ..forms import FieldSpec
from ..plugins import Entry, registry


def methods() -> list[Entry]:
    return registry("methods").entries()


def available_methods() -> list[str]:
    return registry("methods").labels()


def labels_of(kind: str) -> tuple[str, ...]:
    """Labels of the methods of one kind, for conditional form fields."""
    reg = registry("methods")
    return tuple(reg.display(e) for e in reg.entries(kind=kind))


MAIN_FORM_FIELDS = ("endog", "instruments", "cov", "hac_lags", "hac_correction")  # set on the Estimate form itself


def method_settings(label: str, exclude: tuple[str, ...] = MAIN_FORM_FIELDS) -> list[FieldSpec]:
    """The settings of a method that the main form does not cover, for its sub-form."""
    from ..forms import settings_form

    entry = registry("methods").find(label)
    return settings_form(entry, exclude) if entry is not None else []
