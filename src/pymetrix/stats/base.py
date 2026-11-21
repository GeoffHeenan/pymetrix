"""Tests as objects: a registry by name and one result type for all of them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace, dataclass, field
from typing import Any, ClassVar

from .._api import renamed_arguments
from ..errors import NotFoundError, SpecificationError
from ..output import Table


@dataclass(frozen=True, eq=False)
class TestResult:
    __test__ = False  # not a pytest test class

    name: str
    null: str
    statistic: float
    df: Any
    p_value: float | None
    distribution: str = ""
    critical_values: Mapping[str, float] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)
    provenance: Any = None  # what was tested: the fit's provenance, or the series of a dataset, with the test's settings
    backend_call: Any = field(default=None, repr=False)  # a test run through an adapter: its BackendCall, moved into provenance by run_test

    def reject(self, alpha: float = 0.05) -> bool:
        if self.p_value is None:
            raise SpecificationError(f"{self.name} has no p-value; compare the statistic with critical_values")
        return self.p_value < alpha

    def to_table(self) -> Table:
        rows: list[list[Any]] = [["statistic", self.statistic]]
        if self.df is not None:
            rows.append(["df", str(self.df)])
        if self.p_value is not None:
            rows.append(["p-value", self.p_value])
        rows += [[f"critical value {k}", v] for k, v in self.critical_values.items()]
        rows += [[k, v] for k, v in self.details.items()]
        return Table(["", "value"], rows, title=self.name, notes=[f"Null hypothesis: {self.null}"] + ([f"Distribution: {self.distribution}"] if self.distribution else []))

    def __str__(self) -> str:
        return str(self.to_table())

    def _repr_html_(self) -> str:
        return self.to_table()._repr_html_()


class Test:
    """Base class. Subclasses are frozen dataclasses holding the test's options."""

    __test__ = False

    name: ClassVar[str] = ""
    null: ClassVar[str] = ""
    applies_to: ClassVar[tuple[type, ...]] = ()
    topics: ClassVar[tuple[str, ...]] = ()  # background help topics (see pymetrix.help)

    def run(self, obj: Any) -> TestResult:
        raise NotImplementedError


def register(cls: type[Test] | None = None, *, plugin: Any = None) -> Any:
    """Register a test class (a decorator); its ``name`` is how scripts and menus refer to it."""
    from .. import plugins

    if cls is None:
        return lambda c: register(c, plugin=plugin)
    if not cls.name:
        raise SpecificationError(f"{cls.__name__} needs a name")
    plugins.register("tests", cls, name=cls.name, label=cls.name, plugin=plugin)
    return cls


def test_class(name: str) -> type[Test] | None:
    from ..plugins import registry

    entry = registry("tests").find(name)
    return entry.obj if entry is not None else None


def all_tests() -> list[str]:
    from ..plugins import registry

    return sorted(registry("tests").labels(available_only=False))


@renamed_arguments(target="obj")
def available_tests(obj: Any) -> list[str]:
    """The names of the tests that apply to a result, as `run_test` takes them."""
    from ..plugins import registry

    return sorted(registry("tests").display(e) for e in registry("tests").entries() if isinstance(obj, e.obj.applies_to))


def _missing_settings(cls: type, args: tuple[Any, ...], opts: Mapping[str, Any]) -> list[str]:
    """The required settings of a test class that neither ``args`` nor ``opts`` gives."""
    import inspect

    try:
        params = list(inspect.signature(cls).parameters.values())
    except (TypeError, ValueError):
        return []
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return []
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    given = {p.name for p in positional[:len(args)]} | set(opts)
    return [p.name for p in params if p.default is p.empty and p.kind is not p.VAR_KEYWORD and p.name not in given]


@renamed_arguments(target="obj")
def run_test(obj: Any, test: str | Test, *args: Any, **opts: Any) -> TestResult:
    """Run a test on a result, by name with its options or as a test object."""
    if isinstance(test, str):
        cls = test_class(test)
        if cls is None:
            raise NotFoundError(f"unknown test {test!r}; available here: {available_tests(obj)}")
        missing = _missing_settings(cls, args, opts)
        if missing:  # say which settings, rather than Python's "__init__() missing 1 required positional argument"
            raise SpecificationError(f"the {test} test needs {' and '.join(missing)}: give {', '.join(f'{m}=...' for m in missing)} "
                                     "(in the Run test dialog, under Test settings)")
        test = cls(*args, **opts)
    elif args or opts:
        raise TypeError("options go on the test object when one is passed")
    if test.applies_to and not isinstance(obj, test.applies_to):
        raise TypeError(f"{test.name} does not apply to {type(obj).__name__}")
    result = test.run(obj)
    if isinstance(result, TestResult) and result.provenance is None:
        result = replace(result, provenance=_test_provenance(obj, test))
    if isinstance(result, TestResult) and result.backend_call is not None:
        from ..adapters.base import with_backend

        if result.provenance is not None:
            result = replace(result, provenance=with_backend(result.provenance, result.backend_call))
        result = replace(result, backend_call=None)
    if isinstance(result, TestResult) and result.provenance is not None:
        chosen = _resolved(result.details)
        if chosen:  # e.g. the ADF lag chosen by AIC: what was used, next to what was asked for
            result = replace(result, provenance=replace(result.provenance, resolved={**(result.provenance.resolved or {}), **chosen}))
    return result


def _resolved(details: Any) -> dict[str, Any]:
    """The choices a test made itself, from its details: lags, bandwidths and orders."""
    import numbers

    return {k: (int(v) if float(v).is_integer() else float(v)) for k, v in dict(details or {}).items()
            if isinstance(v, numbers.Number) and not isinstance(v, bool) and any(w in str(k).lower() for w in ("lag", "bandwidth", "order"))}


def _test_provenance(target: Any, test: Any) -> Any:
    """Where a test's numbers come from: the tested result's provenance, or the dataset's series, with the test."""
    from dataclasses import replace as _replace

    from .._serial import to_jsonable
    from ..data import Dataset

    method = {"test": getattr(test, "name", type(test).__name__), "settings": to_jsonable(test)}
    prov = getattr(target, "provenance", None)
    if prov is not None:
        return _replace(prov, method={**(prov.method or {}), **method})
    if isinstance(target, Dataset):
        from .._capture import data_fingerprint
        from .._version import __version__
        from ..metadata import Provenance

        from .._capture import with_effective_sample

        candidates = [getattr(test, "variable", None), *(getattr(test, "variables", None) or ())]
        names = [n for n in candidates if isinstance(n, str) and n in target.names]
        rows = target.frame.index[target.sample_mask()]
        if names:  # the tests use the observations where the series are all observed (missing values are skipped)
            rows = rows[target.frame.loc[rows, names].notna().all(axis=1).to_numpy()]
        source, rule = target.sample_record()
        prov = Provenance(data_vintages={n: target.variables[n].vintage for n in names if n in target.variables},
                          spec={"series": names}, method=method, sample_source=source, sample_definition=rule,
                          sample=None, nobs=len(rows), dropped=0, package_version=__version__,
                          data_fingerprint=data_fingerprint(target, names or None, rows), backend=_test_backend(test))
        prov = with_effective_sample(prov, target, rows, variables=names)
        if prov.rows and isinstance(prov.rows, list) and len(prov.rows) > 1:  # gaps inside the sample: joined up by the test
            prov = _replace(prov, messages=(*prov.messages, f"{', '.join(names)} has gaps inside the sample "
                                            f"({len(prov.rows)} runs of periods); the test treats the observations as consecutive"))
        return prov
    return None


def _test_backend(test: Any) -> dict[str, Any]:
    """The library a test wraps (from its ``wraps``), or pymetrix for its own tests."""
    from ..adapters.native import backend_record
    from ..plugins import installed_version

    wraps = getattr(test, "wraps", "") or ""
    library = wraps.split(".")[0]
    if library and library != "pymetrix":
        return {"name": library, "version": installed_version(library) or "", "wraps": wraps}
    return backend_record(f"pymetrix.{type(test).__name__}")
