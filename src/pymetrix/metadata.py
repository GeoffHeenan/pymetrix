"""Metadata for variables, datasets, parameters and results.

Fields that drive behaviour are typed; anything else goes in ``extra``.
Every field has a default and ``from_dict`` ignores keys it does not
recognise, so fields can be added later without breaking stored files.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import date, datetime
from enum import Enum
from math import inf
from typing import Any, Literal

from ._serial import to_jsonable


class Measure(str, Enum):
    """What a series measures; sets default aggregation."""

    STOCK = "stock"
    FLOW = "flow"
    RATE = "rate"
    INDEX = "index"
    RATIO = "ratio"
    COUNT = "count"


class Adjustment(str, Enum):
    NSA = "nsa"
    SA = "sa"
    SWDA = "swda"
    TREND = "trend"


class Prices(str, Enum):
    CURRENT = "current"
    CONSTANT = "constant"
    CHAINED = "chained"
    NONE = "none"


class Status(str, Enum):
    """Observation status, stored in ``Dataset.flags``."""

    NORMAL = "normal"
    ESTIMATED = "estimated"
    PROVISIONAL = "provisional"
    FORECAST = "forecast"
    BREAK = "break"
    MISSING = "missing"
    EDITED = "edited"  # changed by hand (the series' vintage is unchanged)


class Role(str, Enum):
    """A variable's role in a model."""

    ENDOGENOUS = "endogenous"
    EXOGENOUS = "exogenous"
    INSTRUMENT = "instrument"
    SHOCK = "shock"
    ADD_FACTOR = "add_factor"


@dataclass(frozen=True)
class Vintage:
    """When data was released. ``released`` is the release date, never the load date."""

    released: date
    retrieved: datetime | None = None
    revision: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out = to_jsonable(self)
        out.pop("type")
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Vintage:
        retrieved = d.get("retrieved")
        return cls(
            released=date.fromisoformat(d["released"]),
            retrieved=datetime.fromisoformat(retrieved) if retrieved else None,
            revision=d.get("revision"),
        )


@dataclass(frozen=True)
class Variable:
    """Metadata for one series (one column of a ``Dataset``).

    Where a series came from, the minimum pymetrix keeps itself: ``source`` (the provider or file, in words),
    ``source_key`` (the series' ID there, also ``source_id``), ``vintage`` (when the provider released the data;
    never when they were loaded), ``retrieved`` (when pymetrix fetched or imported them, ISO 8601), and the units,
    frequency and recipe (derived or not). ``external_ref`` is for a plugin with a richer source system (e.g. a
    statistical data management system): an opaque reference to its own record of the series, which pymetrix keeps,
    saves and shows but does not interpret.
    """

    name: str
    label: str = ""
    description: str = ""
    units: str = ""
    scale: int = 0
    measure: Measure | None = None
    prices: Prices = Prices.NONE
    base_period: str | None = None
    adjustment: Adjustment = Adjustment.NSA
    source: str = ""
    source_key: str = ""  # the series' ID at its source, e.g. GDPC1 at FRED
    vintage: Vintage | None = None
    release_lag: int | None = None  # days after the end of a period until its value is published (nowcasting, as_of)
    derived_from: str | None = None
    symbol: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    recipe: Any = None  # a Formula or Filtered (pymetrix.data.derived): the series is recomputed when its inputs change
    retrieved: str = ""  # when pymetrix fetched or imported the values (ISO 8601); not a release date
    external_ref: str = ""  # a plugin's own reference for the series (e.g. "psets://..."): kept, saved and shown

    @property
    def source_id(self) -> str:
        """The series' ID at its source (``source_key``)."""
        return self.source_key

    def to_dict(self) -> dict[str, Any]:
        out = to_jsonable(self)
        out.pop("type")
        out["vintage"] = self.vintage.to_dict() if self.vintage else None
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Variable:
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        if kw.get("measure") is not None:
            kw["measure"] = Measure(kw["measure"])
        if "prices" in kw:
            kw["prices"] = Prices(kw["prices"])
        if "adjustment" in kw:
            kw["adjustment"] = Adjustment(kw["adjustment"])
        if kw.get("vintage"):
            kw["vintage"] = Vintage.from_dict(kw["vintage"])
        if kw.get("recipe"):
            from .data.derived import recipe_from_dict

            kw["recipe"] = recipe_from_dict(kw["recipe"])
        return cls(**kw)


@dataclass(frozen=True)
class DatasetMeta:
    title: str = ""
    description: str = ""
    license: str = ""
    contact: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)
    sample: Mapping[str, Any] | None = None  # the active sample (Sample.to_dict()); None: all rows
    # which data was asked for: {"policy": "latest", "retrieved": ...}, {"policy": "as_of", "date": ...},
    # {"policy": "file", "modified": ...}, {"policy": "example", ...} or {"policy": "user"}; None: not known
    vintage: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Param:
    """Metadata for a model or equation parameter, or a model's shock size (``sd_<shock>``) or measurement error
    (``sd_<series>``): its description (``label``), its symbol in algebraic notation (``"φ_π"``), bounds, the
    transform estimation works on (``"log"``: positive; ``"logit"``: between 0 and 1, when no bounds are given), a
    starting value for estimation, a prior, and whether it is ``fixed`` (held at its value when the model is
    estimated)."""

    name: str
    label: str = ""
    symbol: str | None = None
    bounds: tuple[float, float] = (-inf, inf)
    transform: Literal["none", "log", "logit"] = "none"
    start: float | None = None
    prior: Any = None
    fixed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", tuple(self.bounds))  # a list from recorded code: the same parameter
        if self.transform not in ("none", "log", "logit"):
            from .errors import SpecificationError

            raise SpecificationError(f"a parameter's transform is 'none', 'log' or 'logit', not {self.transform!r}")

    @property
    def limits(self) -> tuple[float, float]:
        """The bounds, or those the transform implies when none are given: (0, inf) for log, (0, 1) for logit."""
        low, high = self.bounds
        if low == -inf and high == inf:
            return {"log": (0.0, inf), "logit": (0.0, 1.0)}.get(self.transform, (low, high))
        return low, high

    def merged(self, other: Param | None) -> Param:
        """This spec, with ``other``'s fields where this one leaves them at their defaults (a model's spec over an
        equation's)."""
        if other is None:
            return self
        out = {}
        for f in fields(self):
            mine = getattr(self, f.name)
            out[f.name] = getattr(other, f.name) if mine == f.default else mine
        return Param(**out)


@dataclass(frozen=True)
class Provenance:
    """Where a result came from: data vintages, specification, method and sample."""

    data_vintages: Mapping[str, Vintage | None]
    spec: Mapping[str, Any]
    method: Mapping[str, Any]
    sample: tuple[str, str] | None
    nobs: int
    dropped: int
    seed: int | None = None
    package_version: str = ""
    created: datetime = field(default_factory=datetime.now)
    data_fingerprint: str = ""  # a hash of the data used: the same inputs give the same fingerprint
    warnings: tuple[str, ...] = ()  # warnings raised while making the result ("Category: message")
    converged: bool | None = None  # for iterative estimators: whether the optimiser converged (None: not iterative, or not known)
    # what was used, where it was chosen automatically or inherited (plain defaults, so results saved before load):
    resolved: Mapping[str, Any] | None = None  # automatic choices as made, e.g. {"hac_lags": 4}
    sample_source: str = ""  # which rule chose the rows: "explicit", "active sample of the dataset" or "all observations"
    sample_definition: Mapping[str, Any] | None = None  # that rule in full (a Sample's dict, or the span and condition)
    # the library that did the numbers (pymetrix.adapters), when one did:
    backend: Mapping[str, Any] | None = None  # {"name": "linearmodels", "version": "7.1", "wraps": "linearmodels.iv.IVLIML"}
    options: Mapping[str, Any] | None = None  # the arguments each stage of the library call was given, and the user's backend_options
    messages: tuple[str, ...] = ()  # convergence and optimiser messages the library returned (warnings are in ``warnings``)
    # the rows actually used, after missing values, lags and any presample the library took (pymetrix._capture.effective_sample):
    rows: Any = None  # runs of consecutive periods used: [["1990Q2", "2007Q4"], ...]; for panels {entity: runs}; None without a time axis
    presample: tuple[str, ...] = ()  # rows of the sample used only for lags or initial values by the library (VECM, ARDL, first differences)
    dropped_rows: Mapping[str, Any] | None = None  # {"by_reason": {"missing": 3, "lag": 1}, "rows": [[row, reason], ...] (first 50)}
    vintage_request: Mapping[str, Any] | None = None  # which data was asked for (the dataset's meta.vintage); {"policy": "unknown"} if not said
    # the bit generator the seed drove ("pcg64", "mt19937", "philox", "sfc64": pymetrix.set_random_generator), when there is a seed
    random_generator: str | None = None
    # the pymetrix that made it, beside package_version (pymetrix.build): {"commit": sha, "dirty": bool} from a git
    # checkout, {"commit": None, "note": ...} from an installed package; None: not recorded (made before it was)
    build: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.seed is not None and self.random_generator is None:  # the draws were made with the generator in use now
            from ._random import get_random_generator

            object.__setattr__(self, "random_generator", get_random_generator())
        if self.build is None and self.package_version:  # made now, by the pymetrix running now
            from .build import build_info

            object.__setattr__(self, "build", build_info())

    @classmethod
    def _saved_fields(cls, fields: dict[str, Any]) -> dict[str, Any]:
        """A provenance saved before a field existed, as it was then: a seeded result saved before the random
        generator was recorded was drawn with PCG64 (the only generator then); one saved before the commit was
        recorded does not say it (rather than taking the running pymetrix's)."""
        if fields.get("seed") is not None and "random_generator" not in fields:
            fields = {**fields, "random_generator": "pcg64"}
        if "build" not in fields and fields.get("package_version"):
            fields = {**fields, "build": {"commit": None, "note": "not recorded (saved before pymetrix recorded its commit)"}}
        return fields
