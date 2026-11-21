# Serialisation: the typed codec and command schemas

Plan item 2 · built 24 October 2026 · modules `pymetrix.codec`, `pymetrix.schema`, `pymetrix.commands.registry`

## The problem

Commands could already be saved in projects: they had stable IDs, schema versions and migrations. But they
round-tripped only because their fields were plain strings. The other pieces had gaps:

- `_serial.to_jsonable` wrote unqualified class names and fell back to `repr` for anything it did not know, and
  nothing could read its output back;
- no settings object (`OLS(cov=Cluster(...))`, `ARMA(ar=[1, 4])`, a `Sample`) could be stored as itself;
- UIs and AI tools had no machine-readable description of what a command accepts. The MCP server built its own.

## The decisions

1. **One codec, both ways.** `codec.encode(obj)` gives plain JSON, and `codec.decode(value)` gives back an equal
   object. The guarantee, tested for every registered settings class and every command, is
   `encode(decode(v)) == v`. Many specification classes compare by identity (`eq=False`), so equality is checked
   on the encoded value.
2. **Explicit tags, never `repr`.**
   - Tuples, sets, frozensets, non-string or `$`-prefixed dictionary keys, non-finite floats, dates, datetimes,
     pandas periods, enums and arrays each have a `$` tag.
   - Dataclasses are `{"$type": ..., fields...}` with **every** init field, defaults included. A decoded object
     therefore never silently picks up a default that changed in a later release.
   - Anything else raises `SerialisationError` naming the path to the value (`method.start: ...`).
3. **Data and results are references, not values.** A `Dataset` or a result raises an error that says to pass
   it by name. `Ref("data")` encodes as `{"$ref": "data"}` and is resolved by `decode(value, refs)`, for example
   with a session's objects. Calls (item 1) use this.
   - A named object inside a setting is a reference too: `encode(obj, names)`, with `names` mapping `id(object)`
     to its name, writes the equations of `System([ce, ie])` as `{"$ref": "ce"}`, so a recorded step follows the
     equations it names. Objects without a name stay values. Named results inside a tuple are references the same
     way, so `compare(fe, re)` records `fits` as two references.
   - `Fields(cls, {...})` describes a setting that holds references and so cannot be built until they are resolved
     (a `System` checks its equations). It encodes exactly as the object would; dialogs use it to build their call.
4. **Stable, public type names.** A class exported by `pymetrix` is `"pymetrix:OLS"`; anything else is
   `"module.path:QualName"`. A class that moves or is renamed keeps old files readable through
   `codec.TYPE_ALIASES`. Classes whose constructor does not take their fields (`Cluster(*levels)`, `Sample`)
   register an encoder and decoder with `register_codec`.
5. **Decoding cannot import arbitrary code.** `decode` imports classes only from `pymetrix`, from packages that
   registered extensions, and from packages added with `allow_module`. `{"$type": "subprocess:Popen"}` is
   refused.
6. **Schemas from the code.** `schema.json_schema(cls)` gives a JSON Schema (draft 2020-12) for a command or
   settings class, describing values *as the codec encodes them*. It is generated from the type hints, the
   `Widget` help, minimum and maximum, and the defaults, so it cannot drift. `schema.catalogue()` lists every
   command (and, from item 1, every API call) with its ID, schema version, plugin, menu, help and schema. The
   MCP server's `describe_command` now includes the schema.
7. **Command JSON.** `to_json(cmd)` gives `{"kind": "command", "id", "version", "fields"}`, and `from_json` reads
   it back through `upgrade`, so a step saved at an older schema is migrated exactly as in a project.

## Compatibility

- **Nothing existing changed meaning.** `to_jsonable` still writes provenance and display records, so saved
  results (`.pymxr`) and project format 2 are unchanged.
- **No new dependencies.** The tests validate schemas with `jsonschema` when it is installed (it usually is, as a
  dependency of Jupyter) and skip that check otherwise.

## Tests

`tests/test_codec.py` covers:
- every tagged value;
- every registered settings class and every command (round trip, and validation against its own schema);
- migration of an old command's JSON;
- references, refusals (data, unknown types, disallowed modules, unknown fields and tags) and aliases;
- the catalogue against the frozen command IDs in `tests/fixtures/command_ids.json`.
