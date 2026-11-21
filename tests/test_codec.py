"""The typed JSON codec, command JSON and JSON Schemas: what projects, UIs and AI tools rely on (plan item 2)."""

import dataclasses
import json
import typing
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymetrix
from pymetrix import ARMA, OLS, VAR, Cholesky, Cluster, Dataset, Equation, Sample, Selection
from pymetrix.codec import Fields, Ref, SerialisationError, decode, encode, type_name
from pymetrix.commands import COMMANDS, form_spec
from pymetrix.commands.registry import COMMAND_IDS, from_json, to_json
from pymetrix.metadata import Measure
from pymetrix.plugins import GROUPS, registry
from pymetrix.schema import catalogue, json_schema

FIXTURES = Path(__file__).parent / "fixtures"


def _via_json(value):
    return json.loads(json.dumps(encode(value), allow_nan=False))


@pytest.mark.parametrize("value", [
    None, True, 3, -2.5, "text", [1, "a", None], (1, 2), {"a": 1, "b": [1, 2]}, {1: "x", 2: "y"}, {"$odd": 1},
    {"a", "b"}, frozenset({1, 2}), float("inf"), -float("inf"), date(2020, 1, 30), datetime(2026, 10, 24, 9, 12, 3),
    pd.Period("1990Q1", freq="Q"), pd.Period("2001-07", freq="M"), Measure.FLOW, np.float64(1.5), np.int32(7), np.bool_(True),
])
def test_values_round_trip_through_json(value):
    back = decode(_via_json(value))
    assert back == value
    assert type(back) is type(value) or isinstance(value, np.generic)


def test_nan_and_arrays_round_trip():
    assert np.isnan(decode(_via_json(float("nan"))))
    a = np.arange(6, dtype=float).reshape(2, 3)
    back = decode(_via_json(a))
    assert back.dtype == a.dtype and back.shape == a.shape and (back == a).all()


@pytest.mark.parametrize("obj", [
    OLS(cov="hac", hac_lags=4), OLS(cov=Cluster("country", "year")), OLS(start={"b0": 0.5}),
    Equation("c ~ y + c[-1]"), Equation("y ~ x", errors=ARMA(ar=[1, 4], ma=1)), Equation("dlog(c) ~ b0 + b1*dlog(y)", params=["b0", "b1"]),
    VAR(["a", "b"], lags=2), Cholesky(), Sample(("1990Q1", "2019Q4"), country=["DEU", "FRA"]), Sample(where="x > 0") & Sample(end="2007Q4"),
    Selection(("c", "i", "g"), name="demand"),
], ids=lambda o: type(o).__name__)
def test_settings_round_trip(obj):
    value = _via_json(obj)
    back = decode(value)
    assert type(back) is type(obj)
    assert encode(back) == value  # equal by value, also for classes that compare by identity (Equation, VAR)
    if type(obj).__eq__ is not object.__eq__:
        assert back == obj


def _settings_entries():
    out = []
    for group in GROUPS:
        for entry in registry(group).entries():
            if isinstance(entry.obj, type) and dataclasses.is_dataclass(entry.obj):
                try:
                    entry.build()
                except TypeError:  # needs arguments (a specification or result type, not settings)
                    continue
                out.append(entry)
    return out


@pytest.mark.parametrize("entry", _settings_entries(), ids=lambda e: f"{e.group}/{e.qualified}")
def test_every_registered_setting_round_trips_and_matches_its_schema(entry):
    obj = entry.build()
    value = _via_json(obj)
    assert encode(decode(value)) == value
    jsonschema = pytest.importorskip("jsonschema")
    schema = json_schema(type(obj))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate({k: v for k, v in value.items() if k != "$type"}, schema)


def test_defaults_are_recorded():
    assert encode(OLS()) == {"$type": "pymetrix:OLS", "cov": "classic", "hac_lags": None, "start": None, "hac_correction": True,
                             "restrictions": None}
    # a record written before hac_correction (or restrictions) existed reads with its default, so it computes as it did
    old = decode({"$type": "pymetrix:OLS", "cov": "hac", "hac_lags": 4, "start": None})
    assert old.hac_correction is True and old.restrictions is None
    assert decode(encode(OLS(restrictions="a + b = 1, c = 0"))).restrictions == ("a + b = 1", "c = 0")


def test_type_names_are_stable_and_public_where_possible():
    assert type_name(OLS) == "pymetrix:OLS"
    from pymetrix.estimation.families import Gaussian

    assert type_name(Gaussian) in ("pymetrix:Gaussian", "pymetrix.estimation.families:Gaussian")


def test_references_stay_references_or_resolve():
    value = _via_json({"data": Ref("data"), "method": OLS()})
    assert value["data"] == {"$ref": "data"}
    assert decode(value)["data"] == Ref("data")
    ds = object()
    assert decode(value, {"data": ds})["data"] is ds
    with pytest.raises(SerialisationError, match="no object named 'other'"):
        decode({"$ref": "other"}, {})


def test_named_objects_inside_a_setting_are_written_by_name():
    from pymetrix import System

    ce, ie = Equation("c ~ y"), Equation("i ~ y")
    system = System([ce, ie], instruments=("g",))
    value = json.loads(json.dumps(encode(system, {id(ce): "ce"})))
    assert value["equations"]["$tuple"][0] == {"$ref": "ce"}
    assert value["equations"]["$tuple"][1]["$type"] == "pymetrix:Equation"  # not named: by value
    assert encode(decode(value, {"ce": ce})) == encode(system)
    assert encode(ce, {id(ce): "ce"})["$type"] == "pymetrix:Equation"  # the object itself is a value; only what it holds is named
    assert encode(system) == encode(System([ce, ie], instruments=("g",)))  # without names, as before
    fit = Equation("c ~ y").fit(Dataset.example())
    assert encode((fit,), {id(fit): "fit"}) == {"$tuple": [{"$ref": "fit"}]}  # a named result in a tuple: compare(fe, re)
    with pytest.raises(SerialisationError, match="is data or a result"):
        encode((fit,))


def test_fields_describe_a_setting_holding_references():
    from pymetrix import System

    ce, ie = Equation("c ~ y"), Equation("i ~ y")
    value = json.loads(json.dumps(encode(Fields(System, {"equations": (Ref("ce"), Ref("ie")), "instruments": ("g",)}))))
    assert value == encode(System([ce, ie], instruments=("g",)), {id(ce): "ce", id(ie): "ie"})  # defaults filled in
    assert encode(decode(value, {"ce": ce, "ie": ie})) == encode(System([ce, ie], instruments=("g",)))
    with pytest.raises(SerialisationError, match="no field order"):
        encode(Fields(System, {"order": 1}))


def test_what_cannot_be_saved_is_refused_not_repr():
    data = Dataset.from_frame(pd.DataFrame({"x": [1.0, 2.0]}, index=pd.period_range("2000Q1", periods=2, freq="Q")))
    with pytest.raises(SerialisationError, match="by name"):
        encode(data)
    with pytest.raises(SerialisationError, match="method.start"):
        encode({"method": {"start": object()}})
    with pytest.raises(SerialisationError):
        encode(lambda x: x)


def test_decoding_does_not_import_arbitrary_modules():
    with pytest.raises(SerialisationError, match="not read from saved files"):
        decode({"$type": "subprocess:Popen", "args": "ls"})
    with pytest.raises(SerialisationError, match="no field"):
        decode({"$type": "pymetrix:OLS", "covariance": "hac"})
    with pytest.raises(SerialisationError, match="newer pymetrix"):
        decode({"$future": 1})
    with pytest.raises(SerialisationError, match="not installed"):
        decode({"$type": "pymetrix:NoSuchThing"})


def test_aliases_find_classes_that_moved(monkeypatch):
    from pymetrix import codec

    monkeypatch.setitem(codec.TYPE_ALIASES, "pymetrix.old_place:LeastSquares", "pymetrix:OLS")
    assert decode({"$type": "pymetrix.old_place:LeastSquares", "cov": "white"}) == OLS(cov="white")


def _example(cls):
    """A command with its required fields filled in with simple values of the right type."""
    kwargs = {}
    hints = typing.get_type_hints(cls, include_extras=True)
    for spec in form_spec(cls):
        if spec.required:
            base = typing.get_args(hints[spec.name])[0] if typing.get_origin(hints[spec.name]) is typing.Annotated else hints[spec.name]
            kwargs[spec.name] = {int: 1, float: 1.0, bool: True}.get(base, "x")
    for f in dataclasses.fields(cls):  # hidden fields without defaults
        if f.init and f.name not in kwargs and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            kwargs[f.name] = "x"
    return cls(**kwargs)


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_every_command_round_trips_as_json_and_matches_its_schema(name):
    cmd = _example(COMMANDS[name].cls)
    value = json.loads(json.dumps(to_json(cmd), allow_nan=False))
    assert value["id"] == type(cmd).command_id and value["version"] == type(cmd).schema_version
    assert from_json(value) == cmd
    jsonschema = pytest.importorskip("jsonschema")
    schema = json_schema(type(cmd))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(value["fields"], schema)


def test_command_json_from_an_older_schema_is_migrated():
    old = {"kind": "command", "id": "pymetrix:add_to_dataset", "version": 1, "fields": {"source": "fit", "item": "resid"}}
    cmd = from_json(old)
    assert cmd.replace is True  # schema 1 steps replaced existing series silently, and still do


def test_the_catalogue_lists_every_command_with_its_frozen_id_and_version():
    frozen = {k: v for k, v in json.loads((FIXTURES / "command_ids.json").read_text()).items() if not k.startswith("_")}
    cat = json.loads(json.dumps(catalogue()))
    commands = {e["id"]: e["version"] for e in cat["entries"] if e["kind"] == "command" and e["plugin"] == "pymetrix"}
    assert commands == frozen
    entry = next(e for e in cat["entries"] if e["id"] == "pymetrix:fit_equation")
    assert entry["schema"]["properties"]["cov"]["enum"] == ["classic", "white", "hac", "cluster"]
    assert entry["help"].startswith("Estimate a single equation")


def test_pymetrix_exports_the_codec_names():
    assert pymetrix.codec.encode is encode
