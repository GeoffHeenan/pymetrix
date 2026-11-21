"""Exercise packs: guided estimation walkthroughs that ship on their own, as a folder or in a plugin (provisional).

A pack is a folder (the format is in ``docs/exercise-packs.md``)::

    <pack>/
      pack.toml          [pack] name, title, version, description, requires_pymetrix, audience (optional)
      data/              optional: .parquet or .csv files the exercises load by name
      models/            optional: model files (.mod Dynare, .model IRIS, .pymodel pymetrix syntax), by name; pack.toml's
                         [models.<file stem>] name = "..." gives the session name one is imported under
      exercises/
        01-<slug>/
          exercise.md    front matter (title, level, minutes, data), then "## Step N: <title>" sections, each with
                         the steps in the app, an optional ```python block and optional "> Hint:" blockquotes
          checks.toml    [[check]] id, step, object, value, expected, abs_tol / rel_tol, message
          solution.py    optional: a script that completes the exercise in a fresh session

    from pymetrix.exercises import check, installed_packs, load_pack

    pack = load_pack("my-pack")                     # validated: a bad pack raises SpecificationError, every problem listed
    ex = pack.exercise("ols-mroz")                  # by folder name, slug or number
    for r in check(ex, session, step=2):            # the checks of step 2 against the session's objects
        print(r.passed, r.actual, r.expected, r.message)

**Checks are data, not code.** A check's ``value`` is a dotted path into a session object, resolved by attribute and
key lookup only (``params.educ``, ``stats.r2``, ``details.lags used``): nothing is called and no name starting with
``__`` is followed, so a pack's checks are safe to load from any folder. For estimates, ``se`` (or ``bse``),
``tvalues``, ``pvalues`` (by parameter name) and ``nobs`` are read from the result's own tables when it has no attribute of that
name. ``solution.py`` is code: only the command-line checker runs it (``pymetrix-exercises check``), on the pack it is
given, with the pack's folder as its working directory and ``PACK_DIR`` (that folder, a ``Path``) defined; the
app never runs anything in a pack.

**Plugins** ship a pack through the entry-point group ``pymetrix.exercises``, whose value is a callable returning the
pack's folder (``importlib.resources.files(__package__) / "pack"``); a disabled plugin's packs are not listed, and a
new plugin's wait for the app's answer, as its other extensions do. A plugin's pack and a folder are the same thing.

This interface is public but provisional before pymetrix 1.0: it may change in a minor release.
"""

from __future__ import annotations

import math
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import NotFoundError, SpecificationError

__all__ = ["LEVELS", "MODEL_SUFFIXES", "Check", "CheckResult", "Exercise", "Pack", "Step", "check", "data_commands",
           "installed_packs", "load_data", "load_model", "load_pack", "model_command", "problems", "resolve"]

GROUP = "pymetrix.exercises"  # the entry-point group of plugins' packs
DATA_SUFFIXES = (".parquet", ".csv")
MODEL_SUFFIXES = {".mod": "dynare", ".model": "iris", ".pymodel": "pymetrix"}  # models/: Dynare, IRIS, pymetrix's syntax
LEVELS = ("beginner", "intermediate", "advanced")
DEFAULT_REL_TOL = 1e-6  # a number with neither tolerance: equal to 6 significant digits

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_FOLDER = re.compile(r"^(\d+)-([a-z0-9][a-z0-9-]*)$")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_STEP = re.compile(r"^##\s+Step\s+(\d+)\s*:\s*(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^(```+|~~~+)\s*([^`\s]*)")
_PACK_KEYS = {"name", "title", "version", "description", "requires_pymetrix", "audience", "classification"}
_PACK_REQUIRED = ("name", "title", "version", "description", "requires_pymetrix")
_CHECK_KEYS = {"id", "step", "object", "value", "expected", "abs_tol", "rel_tol", "message"}
_FRONT_KEYS = {"title", "level", "minutes", "data", "models"}


# -- the format ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One ``## Step N: <title>`` section."""

    number: int
    title: str
    text: str  # the steps in the app (Markdown), without the code and the hints
    code: str = ""  # the ```python block: the same step as code ("" if none)
    hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class Check:
    """One ``[[check]]`` of ``checks.toml``: the value at ``value`` in the session object ``object`` should be
    ``expected`` (numbers within ``abs_tol`` or ``rel_tol``)."""

    id: str
    step: int
    object: str
    value: str
    expected: float | int | str | bool
    abs_tol: float | None = None
    rel_tol: float | None = None
    message: str = ""


@dataclass(frozen=True)
class Exercise:
    """One exercise of a pack: its folder ``01-<slug>``."""

    id: str  # the folder's name: "01-ols-mroz"
    title: str
    steps: tuple[Step, ...]
    checks: tuple[Check, ...]
    path: Path
    level: str = ""
    minutes: int | None = None
    data: tuple[str, ...] = ()  # the datasets it loads, by name
    models: tuple[str, ...] = ()  # the models of the pack's models/ it uses, by name
    intro: str = ""  # the Markdown before step 1
    closing: str = ""  # the Markdown of any sections after the last step
    solution: Path | None = None
    pack: Pack | None = field(default=None, compare=False, repr=False)

    @property
    def number(self) -> int:
        return int(self.id.split("-", 1)[0])

    @property
    def slug(self) -> str:
        return self.id.split("-", 1)[1]

    def step(self, number: int) -> Step:
        for s in self.steps:
            if s.number == number:
                return s
        raise NotFoundError(f"exercise {self.id} has no step {number}; it has steps 1 to {len(self.steps)}")

    def checks_for(self, step: int | None = None) -> tuple[Check, ...]:
        return self.checks if step is None else tuple(c for c in self.checks if c.step == step)


@dataclass(frozen=True)
class Pack:
    """An exercise pack, loaded and validated (``load_pack``)."""

    name: str
    title: str
    version: str
    description: str
    requires_pymetrix: str
    path: Path
    exercises: tuple[Exercise, ...]
    data: Mapping[str, Path] = field(default_factory=dict)  # the files of data/, by name
    audience: str = ""  # free text: who it is for, shown in the list
    source: str = "folder"  # "folder", or the distribution of the plugin that ships it
    data_options: Mapping[str, Mapping[str, str]] = field(default_factory=dict)  # name -> time, entity, freq
    models: Mapping[str, Path] = field(default_factory=dict)  # the files of models/, by name
    classification: str = ""  # free text ("internal", "public"...): shown, never enforced
    model_names: Mapping[str, str] = field(default_factory=dict)  # file stem -> session name ([models.<stem>] name)

    def model_name(self, model: str) -> str:
        """The name the model ``model`` of models/ is imported under in a session: its ``[models.<stem>]`` table's
        ``name``, or else the file's stem (``rbc`` for ``models/rbc.mod``). ``model`` is the stem or that name."""
        stem = self._model_stem(model)
        return self.model_names.get(stem, stem)

    def _model_stem(self, model: str) -> str:
        """The file stem of the model named ``model``: a stem of models/ first, then a session name a table gives."""
        if model in self.models:
            return model
        for stem, name in self.model_names.items():
            if name == model and stem in self.models:
                return stem
        raise NotFoundError(f"the pack {self.name} has no model {model!r}: its models/ has {', '.join(self.models) or 'none'}")

    @property
    def requirement_problem(self) -> str | None:
        """Why this pymetrix does not meet ``requires_pymetrix`` (None when it does)."""
        from packaging.specifiers import SpecifierSet

        from .._version import __version__

        spec = SpecifierSet(self.requires_pymetrix)
        if spec.contains(__version__, prereleases=True):
            return None
        return f"the pack {self.name} needs pymetrix {self.requires_pymetrix}; this is pymetrix {__version__}"

    @property
    def compatible(self) -> bool:
        return self.requirement_problem is None

    def exercise(self, key: str | int) -> Exercise:
        """An exercise by its folder name (``01-ols-mroz``), its slug (``ols-mroz``) or its number (1)."""
        text = str(key).strip()
        for ex in self.exercises:
            if text in (ex.id, ex.slug) or (text.isdigit() and int(text) == ex.number):
                return ex
        raise NotFoundError(f"the pack {self.name} has no exercise {text!r}; it has {', '.join(e.id for e in self.exercises)}")


class _Problems:
    """The problems found in a pack, each with the file it is in."""

    def __init__(self, root: Path) -> None:
        self.root, self.items = root, []

    def add(self, where: Path | str, text: str) -> None:
        try:
            shown = Path(where).relative_to(self.root).as_posix() if isinstance(where, Path) else where
        except ValueError:
            shown = str(where)
        self.items.append(f"{shown}: {text}")

    def raise_if_any(self) -> None:
        if self.items:
            n = len(self.items)
            err = SpecificationError(f"the exercise pack {self.root} has {n} problem{'s' * (n != 1)}:\n"
                                     + "\n".join(f"- {p}" for p in self.items))
            err.problems = list(self.items)  # type: ignore[attr-defined]
            raise err


def _read_toml(path: Path, problems: _Problems) -> dict[str, Any] | None:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        problems.add(path, "missing")
    except (OSError, UnicodeDecodeError) as exc:
        problems.add(path, f"cannot be read: {exc}")
    except tomllib.TOMLDecodeError as exc:
        problems.add(path, f"is not valid TOML: {exc}")
    return None


def _pack_meta(root: Path, problems: _Problems) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    """[pack], the [data.<name>] tables and the [models.<file stem>] tables."""
    path = root / "pack.toml"
    raw = _read_toml(path, problems)
    if raw is None:
        return {}, {}, {}
    meta = raw.get("pack")
    for key in raw:
        if key not in ("pack", "data", "models"):
            problems.add(path, f"unknown table or key {key!r}: the file has a [pack] table, [data.<name>] tables and "
                               "[models.<file stem>] tables")
    tables = raw.get("data", {})
    if not isinstance(tables, dict) or not all(isinstance(v, dict) for v in tables.values()):
        problems.add(path, "data options are written [data.<name>], with time, entity and freq")
        tables = {}
    model_tables = raw.get("models", {})
    if not isinstance(model_tables, dict) or not all(isinstance(v, dict) for v in model_tables.values()):
        problems.add(path, "a model's session name is written [models.<file stem>], with name")
        model_tables = {}
    if not isinstance(meta, dict):
        problems.add(path, "needs a [pack] table")
        return {}, tables, model_tables
    out: dict[str, str] = {}
    for key in sorted(set(meta) - _PACK_KEYS):
        problems.add(path, f"unknown key {key!r} in [pack]; the keys are {', '.join(sorted(_PACK_KEYS))}")
    for key in sorted(_PACK_KEYS):
        value = meta.get(key)
        if value is None:
            if key in _PACK_REQUIRED:
                problems.add(path, f"[pack] needs {key!r}")
            continue
        if not isinstance(value, str) or not value.strip():
            problems.add(path, f"[pack] {key} should be a non-empty string, not {value!r}")
            continue
        out[key] = value.strip()
    if "name" in out and not _NAME.match(out["name"]):
        problems.add(path, f"[pack] name {out['name']!r} should be lower case letters, digits, '-' and '_'")
    if "requires_pymetrix" in out:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet

        try:
            SpecifierSet(out["requires_pymetrix"])
        except InvalidSpecifier:
            problems.add(path, f"[pack] requires_pymetrix {out['requires_pymetrix']!r} is not a version specifier "
                               "(PEP 440, e.g. \">=0.1.0a18\")")
    return out, tables, model_tables


def _data_files(root: Path, problems: _Problems, tables: dict[str, Any]) -> tuple[dict[str, Path], dict[str, dict[str, str]]]:
    """The files of data/ by name, and the options to read each with (pack.toml's [data.<name>] tables)."""
    out, options = _data_folder(root, problems)
    pack_toml = root / "pack.toml"
    for name, table in tables.items():
        if name not in out:
            problems.add(pack_toml, f"[data.{name}] names no file of data/")
            continue
        for key in sorted(set(table) - {"time", "entity", "freq"}):
            problems.add(pack_toml, f"[data.{name}]: unknown key {key!r}; the keys are entity, freq and time")
        for key in ("time", "entity", "freq"):
            value = table.get(key)
            if value is not None and (not isinstance(value, str) or not value):
                problems.add(pack_toml, f"[data.{name}] {key} should be a non-empty string, not {value!r}")
            elif value is not None:
                options[name][key] = value
        if options[name].get("freq", "") not in DATA_FREQUENCIES:
            problems.add(pack_toml, f"[data.{name}] freq should be one of {', '.join(f for f in DATA_FREQUENCIES if f)}")
    for name, path in out.items():
        if path.suffix.lower() == ".csv" and not ({"time", "entity"} & set(options[name])):
            problems.add(path, f"a CSV file needs [data.{name}] in pack.toml with its time and/or entity column")
    return out, options


DATA_FREQUENCIES = ("", "Y", "Y-JUN", "Q", "M", "W", "D")  # as File → Load data… offers them


def _data_folder(root: Path, problems: _Problems) -> tuple[dict[str, Path], dict[str, dict[str, str]]]:
    folder = root / "data"
    out: dict[str, Path] = {}
    if not folder.exists():
        return out, {}
    if not folder.is_dir():
        problems.add(folder, "should be a folder")
        return out, {}
    for path in sorted(folder.iterdir()):
        if path.name.startswith(".") or path.stem.upper() in ("README", "LICENSE", "LICENCE", "NOTICE"):
            continue
        if path.suffix.lower() not in DATA_SUFFIXES or not path.is_file():
            problems.add(path, "is not a data file (.parquet or .csv)")
        elif path.stem in out:
            problems.add(path, f"two data files are called {path.stem!r}")
        else:
            out[path.stem] = path
    return out, {name: {} for name in out}


def _model_files(root: Path, problems: _Problems) -> dict[str, Path]:
    folder = root / "models"
    out: dict[str, Path] = {}
    if not folder.exists():
        return out
    if not folder.is_dir():
        problems.add(folder, "should be a folder")
        return out
    for path in sorted(folder.iterdir()):
        if path.name.startswith(".") or path.stem.upper() in ("README", "LICENSE", "LICENCE", "NOTICE"):
            continue
        if path.suffix.lower() not in MODEL_SUFFIXES or not path.is_file():
            problems.add(path, f"is not a model file ({', '.join(MODEL_SUFFIXES)})")
        elif path.stem in out:
            problems.add(path, f"two model files are called {path.stem!r}")
        else:
            out[path.stem] = path
    return out


def _model_names(root: Path, problems: _Problems, tables: dict[str, Any], models: dict[str, Path]) -> dict[str, str]:
    """The session name of each model that pack.toml renames: ``[models.<file stem>] name = "..."``."""
    import keyword

    pack_toml = root / "pack.toml"
    out: dict[str, str] = {}
    for stem, table in tables.items():
        if stem not in models:
            problems.add(pack_toml, f"[models.{stem}] names no file of models/ (by its name without the extension)")
            continue
        for key in sorted(set(table) - {"name"}):
            problems.add(pack_toml, f"[models.{stem}]: unknown key {key!r}; the key is name")
        name = table.get("name")
        if name is None:
            problems.add(pack_toml, f"[models.{stem}] needs name: the name the model is imported under")
        elif not isinstance(name, str) or not _IDENT.match(name) or keyword.iskeyword(name):
            problems.add(pack_toml, f"[models.{stem}] name should be a Python name (letters, digits and _), not {name!r}")
        else:
            out[stem] = name
    seen: dict[str, str] = {}
    for stem in models:
        name = out.get(stem, stem)
        if name in seen:
            problems.add(pack_toml, f"the models {seen[name]} and {stem} would both be imported as {name!r}: "
                                    "give one another name in [models.<file stem>]")
        seen.setdefault(name, stem)
    return out


def _split_steps(text: str) -> tuple[list[tuple[str, list[str]]], int]:
    """The Markdown in sections: ("", intro lines), then (heading, lines) for each ``## `` heading outside code."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    fence = ""
    for line in text.splitlines():
        m = _FENCE.match(line.strip())
        if m and (not fence or m.group(1).startswith(fence)):
            fence = "" if fence else m.group(1)[:3]
        if not fence and re.match(r"^##\s", line):
            sections.append((line.strip(), []))
            continue
        sections[-1][1].append(line)
    return sections, len(sections)


def _step_parts(lines: list[str]) -> tuple[str, list[str], list[str]]:
    """A step's lines as (text, python blocks, hints)."""
    text: list[str] = []
    code: list[str] = []
    hints: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _FENCE.match(line.strip())
        if m:
            fence, info = m.group(1), m.group(2).lower()
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(fence[:3]):
                body.append(lines[i])
                i += 1
            i += 1  # the closing fence
            if info in ("python", "py"):
                code.append("\n".join(body).strip("\n"))
            else:
                text += [line, *body, fence]
            continue
        if re.match(r"^\s*>\s*Hint\s*:", line, re.IGNORECASE):
            hint = [re.sub(r"^\s*>\s*Hint\s*:\s*", "", line, flags=re.IGNORECASE)]
            i += 1
            while i < len(lines) and re.match(r"^\s*>", lines[i]) and not re.match(r"^\s*>\s*Hint\s*:", lines[i], re.IGNORECASE):
                hint.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            hints.append("\n".join(hint).strip())
            continue
        text.append(line)
        i += 1
    return "\n".join(text).strip(), code, hints


def _exercise_md(path: Path, problems: _Problems, data_names: set[str], model_names: Mapping[str, str]) -> dict[str, Any] | None:
    from ..help import _front_matter

    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        problems.add(path, "missing")
        return None
    except (OSError, UnicodeDecodeError) as exc:
        problems.add(path, f"cannot be read: {exc}")
        return None
    meta, body = _front_matter(raw.lstrip("﻿"))
    if not meta:
        problems.add(path, "needs front matter between --- lines (title, level, minutes, data)")
    for key in sorted(set(meta) - _FRONT_KEYS):
        problems.add(path, f"unknown front-matter key {key!r}; the keys are {', '.join(sorted(_FRONT_KEYS))}")
    out: dict[str, Any] = {"title": str(meta.get("title", "")).strip(), "level": str(meta.get("level", "") or "").strip()}
    if not out["title"]:
        problems.add(path, "the front matter needs a title")
    if out["level"] and out["level"] not in LEVELS:
        problems.add(path, f"level should be {', '.join(LEVELS[:-1])} or {LEVELS[-1]}, not {out['level']!r}")
    minutes = meta.get("minutes")
    out["minutes"] = None
    if minutes not in (None, ""):
        if isinstance(minutes, str) and minutes.isdigit():
            out["minutes"] = int(minutes)
        else:
            problems.add(path, f"minutes should be a whole number, not {minutes!r}")
    data = meta.get("data", [])
    data = [data] if isinstance(data, str) and data else list(data or [])
    for name in data:
        if name not in data_names:
            problems.add(path, f"data names {name!r}, which is neither in the pack's data/ nor an example dataset")
    out["data"] = tuple(data)
    models = meta.get("models", [])
    models = [models] if isinstance(models, str) and models else list(models or [])
    for name in models:
        if name not in model_names:
            problems.add(path, f"models names {name!r}, which is not a file of the pack's models/")
    out["models"] = tuple(model_names.get(name, name) for name in models)  # by file stem
    body = re.sub(r"\A\s*#\s+[^\n]*\n", "", body)  # a "# Title" line: the front matter's title is the one shown
    sections, _ = _split_steps(body)
    steps: list[Step] = []
    closing: list[str] = []
    for heading, lines in sections[1:]:
        m = _STEP.match(heading)
        if m is None:
            if re.match(r"^##\s+Step\b", heading, re.IGNORECASE):
                problems.add(path, f"{heading!r} should read '## Step N: <title>'")
            elif steps:
                closing += [heading, *lines]
            else:
                sections[0][1].extend([heading, *lines])  # before the first step: part of the introduction
            continue
        number, title = int(m.group(1)), m.group(2).strip()
        if closing:
            problems.add(path, f"step {number} comes after a section that is not a step")
        if number != len(steps) + 1:
            problems.add(path, f"step {number} should be step {len(steps) + 1}: steps are numbered 1, 2, 3… in order")
        text, code, hints = _step_parts(lines)
        if len(code) > 1:
            problems.add(path, f"step {number} has {len(code)} python blocks; a step has at most one")
        steps.append(Step(number, title, text, code[0] if code else "", tuple(hints)))
    if not steps:
        problems.add(path, "has no steps: sections '## Step 1: <title>', '## Step 2: …'")
    out["steps"] = tuple(steps)
    out["intro"] = "\n".join(sections[0][1]).strip()
    out["closing"] = "\n".join(closing).strip()
    return out


def _check_path(value: str) -> str | None:
    """Why a check's value path cannot be used (None if it can)."""
    if not value:
        return "is empty"
    for part in value.split("."):
        if not part:
            return "has an empty part (two dots, or a dot at an end)"
        if part.startswith("__"):
            return f"names {part!r}: names starting with __ are not followed"
    return None


def _checks(path: Path, problems: _Problems, steps: int) -> tuple[Check, ...]:
    if not path.exists():
        return ()
    raw = _read_toml(path, problems)
    if raw is None:
        return ()
    for key in raw:
        if key != "check":
            problems.add(path, f"unknown table or key {key!r}: the file holds [[check]] tables")
    entries = raw.get("check", [])
    if not isinstance(entries, list):
        problems.add(path, "checks are written [[check]] (an array of tables)")
        return ()
    out: list[Check] = []
    seen: set[str] = set()
    for n, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            problems.add(path, f"check {n} is not a table")
            continue
        where = f"check {entry.get('id', n)!s}"
        bad = len(problems.items)
        for key in sorted(set(entry) - _CHECK_KEYS):
            problems.add(path, f"{where}: unknown key {key!r}; the keys are {', '.join(sorted(_CHECK_KEYS))}")
        for key in ("id", "step", "object", "value", "expected"):
            if key not in entry:
                problems.add(path, f"{where}: needs {key!r}")
        cid, step, obj, value = entry.get("id"), entry.get("step"), entry.get("object"), entry.get("value")
        expected = entry.get("expected")
        if cid is not None and (not isinstance(cid, str) or not cid.strip()):
            problems.add(path, f"{where}: id should be a non-empty string")
        elif isinstance(cid, str):
            if cid in seen:
                problems.add(path, f"{where}: two checks have the id {cid!r}")
            seen.add(cid)
        if step is not None and (not isinstance(step, int) or isinstance(step, bool) or not 1 <= step <= steps):
            problems.add(path, f"{where}: step should be the number of a step of the exercise (1 to {steps}), not {step!r}")
        if obj is not None and (not isinstance(obj, str) or not _IDENT.match(obj)):
            problems.add(path, f"{where}: object should be the name of a session object, not {obj!r}")
        if value is not None:
            why = _check_path(value) if isinstance(value, str) else "should be a string"
            if why:
                problems.add(path, f"{where}: value {value!r} {why}")
        if "expected" in entry and not isinstance(expected, (int, float, str, bool)):
            problems.add(path, f"{where}: expected should be a number, a string or true/false, not {expected!r}")
        numeric = isinstance(expected, (int, float)) and not isinstance(expected, bool)
        for tol in ("abs_tol", "rel_tol"):
            if tol in entry:
                t = entry[tol]
                if not numeric:
                    problems.add(path, f"{where}: {tol} is for numbers; expected is {expected!r}")
                elif not isinstance(t, (int, float)) or isinstance(t, bool) or not t >= 0:
                    problems.add(path, f"{where}: {tol} should be a number at least 0, not {t!r}")
        if numeric and isinstance(expected, float) and math.isnan(expected):
            problems.add(path, f"{where}: expected is nan, which no value equals")
        if "message" in entry and not isinstance(entry["message"], str):
            problems.add(path, f"{where}: message should be a string")
        if len(problems.items) == bad:
            out.append(Check(cid, step, obj, value, expected,
                             float(entry["abs_tol"]) if "abs_tol" in entry else None,
                             float(entry["rel_tol"]) if "rel_tol" in entry else None,
                             entry.get("message", "")))
    return tuple(out)


def load_pack(path: str | Path, source: str = "folder") -> Pack:
    """Read and validate the exercise pack in the folder ``path``.

    Every problem found (a missing or unknown key, a step out of order, a check naming no step, a dataset that does
    not exist...) is listed in one ``SpecificationError``, each with its file; its ``problems`` attribute holds them.
    A pack written for another pymetrix loads: ``Pack.compatible`` and ``requirement_problem`` say so."""
    from .. import datasets

    root = Path(path).expanduser()
    if not root.is_dir():
        raise NotFoundError(f"no exercise pack at {root}: it should be a folder with pack.toml and exercises/")
    root = root.resolve()
    problems = _Problems(root)
    meta, tables, model_tables = _pack_meta(root, problems)
    data, data_options = _data_files(root, problems, tables)
    models = _model_files(root, problems)
    model_names = _model_names(root, problems, model_tables, models)
    known_models = {**{name: stem for stem, name in model_names.items()}, **{stem: stem for stem in models}}  # a stem first
    names = set(data) | set(datasets.available())
    exercises: list[Exercise] = []
    folder = root / "exercises"
    if not folder.is_dir():
        problems.add(folder, "missing: exercises go in exercises/01-<slug>/")
    else:
        numbers: dict[int, str] = {}
        for sub in sorted(folder.iterdir(), key=lambda p: p.name):
            if sub.name.startswith((".", "_")):
                continue
            m = _FOLDER.match(sub.name)
            if not sub.is_dir() or m is None:
                problems.add(sub, "is not an exercise folder: name them 01-<slug> (digits, '-', lower case letters, "
                                  "digits and '-')")
                continue
            if int(m.group(1)) in numbers:
                problems.add(sub, f"has the number of {numbers[int(m.group(1))]}")
            numbers[int(m.group(1))] = sub.name
            md = _exercise_md(sub / "exercise.md", problems, names, known_models)
            if md is None:
                continue
            checks = _checks(sub / "checks.toml", problems, len(md["steps"]))
            solution = sub / "solution.py"
            exercises.append(Exercise(sub.name, md["title"], md["steps"], checks, sub, md["level"], md["minutes"],
                                      md["data"], md["models"], md["intro"], md["closing"],
                                      solution if solution.is_file() else None))
        if not exercises and not problems.items:
            problems.add(folder, "has no exercises")
    problems.raise_if_any()
    pack = Pack(meta["name"], meta["title"], meta["version"], meta["description"], meta["requires_pymetrix"], root,
                tuple(exercises), data, meta.get("audience", ""), source, data_options, models,
                meta.get("classification", ""), model_names)
    for ex in exercises:
        object.__setattr__(ex, "pack", pack)
    return pack


# -- installed packs --------------------------------------------------------------------------------------------------

_PROBLEMS: dict[str, str] = {}


def installed_packs() -> list[Pack]:
    """The packs that installed plugins ship (entry-point group ``pymetrix.exercises``), by title.

    A disabled plugin's packs are left out, and so are a new plugin's while the app waits for the user's answer
    (Help → Plugins), as with its other extensions. A pack that cannot be found or read is left out too, and why is
    in ``problems()``."""
    from ..plugins import _admit, _dist_name, _entry_points

    _PROBLEMS.clear()
    out: list[Pack] = []
    for ep in _entry_points(GROUP):
        dist = _dist_name(ep)
        label = f"{dist}: {ep.name}"
        if not _admit(ep):
            continue
        try:
            obj = ep.load()
            where = obj() if callable(obj) else obj
            out.append(load_pack(Path(str(where)), source=dist))
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not stop the list
            _PROBLEMS[label] = str(exc)
    return sorted(out, key=lambda p: (p.title.lower(), p.name))


def problems() -> dict[str, str]:
    """Plugins' packs that ``installed_packs`` could not load: "distribution: entry point" -> why."""
    return dict(_PROBLEMS)


# -- data ---------------------------------------------------------------------------------------------------------


def _pack(pack: Pack | str | Path) -> Pack:
    return pack if isinstance(pack, Pack) else load_pack(pack)


def load_data(pack: Pack | str | Path, name: str) -> Any:
    """The dataset ``name`` an exercise loads: the pack's ``data/<name>.parquet`` or ``.csv`` (read with its
    ``[data.<name>]`` options), or else pymetrix's example dataset of that name (``Dataset.example``). ``pack`` is a
    ``Pack`` or its folder (``PACK_DIR`` in a solution)."""
    from ..data import Dataset

    pack = _pack(pack)
    if name in pack.data:
        return Dataset.read(str(pack.data[name]), **dict(pack.data_options.get(name, {})))
    from .. import datasets

    if name in datasets.available():
        return datasets.load(name)
    raise NotFoundError(f"the pack {pack.name} has no dataset {name!r}: its data/ has {', '.join(pack.data) or 'none'}")


def load_model(pack: Pack | str | Path, name: str) -> Any:
    """The model ``models/<name>.<ext>`` of a pack, read by its extension: ``.mod`` with ``read_mod`` (a
    ``DynareModel``), ``.model`` with ``read_iris`` (a ``Template``), ``.pymodel`` (pymetrix's model syntax) with
    ``ModelSource(text).build()`` (a ``Template`` named by the model's session name, ``Pack.model_name``): what Model →
    Import Dynare file…, Import IRIS file… and the Model editor make of the same file. ``name`` is the file's stem or
    the session name pack.toml gives it (``[models.<stem>] name``). ``pack`` is a ``Pack`` or its folder (``PACK_DIR``
    in a solution)."""
    pack = _pack(pack)
    stem = pack._model_stem(name)
    path = pack.models[stem]
    syntax = MODEL_SUFFIXES[path.suffix.lower()]
    if syntax == "dynare":
        from ..models.dynare import read_mod

        return read_mod(path)
    if syntax == "iris":
        from ..models.iris import read_iris

        return read_iris(path)
    from ..models.source import ModelSource

    return ModelSource(path.read_text(encoding="utf-8"), "pymetrix", pack.model_name(stem)).build()


def model_command(pack: Pack, name: str) -> Any:
    """The dialog step that imports the pack's model ``name`` (its file stem, or the session name pack.toml gives it)
    under its session name (``Pack.model_name``: ``[models.<stem>] name``, else the stem), filled in with its file:
    Model → Import Dynare file…, Import IRIS file…, or the Model editor with the file's text. The app opens it for the
    user to run."""
    from ..commands.builtin import ImportDynare, ImportIris
    from ..commands.model_source import DefineModel

    stem = pack._model_stem(name)
    path, session_name = pack.models[stem], pack.model_name(stem)
    syntax = MODEL_SUFFIXES[path.suffix.lower()]
    if syntax == "dynare":
        return ImportDynare(path=str(path), name=session_name)
    if syntax == "iris":
        return ImportIris(path=str(path), name=session_name)
    return DefineModel(text=path.read_text(encoding="utf-8"), syntax="pymetrix", name=session_name)


def data_commands(exercise: Exercise) -> list[Any]:
    """The steps that load an exercise's data into a session, each dataset under its own name: File → Load data… for
    a file of the pack, File → Load example data… for an example dataset. Run them with ``session.run``: they are
    recorded, and replay, as any step."""
    from ..commands.builtin import LoadData, LoadExample

    pack = exercise.pack
    out: list[Any] = []
    for name in exercise.data:
        if pack is not None and name in pack.data:
            out.append(LoadData(path=str(pack.data[name]), name=name, **dict(pack.data_options.get(name, {}))))
        else:
            out.append(LoadExample(dataset=name, name=name))
    return out


# -- checking ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """A check against a session: ``passed``, the value found (``actual``, None if it could not be read), the value
    wanted, and the check's ``message``; ``detail`` says what was compared, or why the value could not be read."""

    id: str
    step: int
    passed: bool
    actual: Any
    expected: Any
    message: str
    detail: str = ""


def _derived(obj: Any, name: str) -> Any:
    """``se`` (or ``bse``), ``tvalues``, ``pvalues`` (by parameter name) and ``nobs`` of an estimate, from its own
    tables."""
    if name == "nobs":
        stats = getattr(obj, "stats", None)
        if isinstance(stats, Mapping) and "nobs" in stats:
            n = stats["nobs"]
            return int(n) if isinstance(n, float) and n.is_integer() else n
        raise LookupError
    column = {"bse": "std_error", "se": "std_error", "tvalues": "statistic", "pvalues": "p_value"}.get(name)
    params = getattr(obj, "params", None) if column else None
    if params is None or not hasattr(params, "names"):
        raise LookupError
    from ..results import coefficients

    table = coefficients(obj)
    if column not in table or len(table) != len(params.names):
        raise LookupError
    return dict(zip(params.names, table[column].tolist()))


def _whole(part: str) -> int | None:
    """A path part that is a whole number ("1", "2001", "-1"), as an int; None for any other part."""
    return int(part) if part.lstrip("-").isdigit() else None


def _key(obj: Any, part: str) -> Any:
    """``part`` as a key of ``obj``: the string first, then, for a part that is a whole number, the int key (a label
    of a Series with an integer index, a dict's int key), then, for a list, tuple or array, the position."""
    if isinstance(obj, (str, bytes)):
        raise LookupError
    number = _whole(part)
    if isinstance(obj, Mapping):
        if part in obj:
            return obj[part]
        if number is not None and number in obj:
            return obj[number]
        raise LookupError
    import pandas as pd

    if isinstance(obj, (pd.Series, pd.DataFrame)):
        labels = obj.index if isinstance(obj, pd.Series) else obj.columns
        if part in labels:
            return obj[part]
        if number is not None and number in labels:  # a label, never a position: a Series is read by its index
            return obj.loc[number] if isinstance(obj, pd.Series) else obj[number]
        raise LookupError
    if number is not None and isinstance(obj, Sequence):
        return obj[number]
    import numpy as np

    if number is not None and isinstance(obj, np.ndarray):
        return obj[number]
    if hasattr(type(obj), "__getitem__"):
        return obj[part]
    raise LookupError


def resolve(obj: Any, path: str) -> Any:
    """The value at the dotted ``path`` in ``obj``, by attribute and key lookup only: no part is called, and no part
    starting with ``__`` is followed. Each part is a key of a mapping or a pandas object; otherwise an attribute (a
    property counts), then, for an estimate, ``se``/``bse``/``tvalues``/``pvalues``/``nobs``, then a key (``params.educ``)
    or, for a sequence or array, a whole-number index. A key is tried as the string, then, when the part is a whole
    number, as an int (``2001`` on a Series indexed by year, a dict's int key), then as a position (a list, tuple or
    array). A method is refused, wherever it is in the path.

    Raises ``SpecificationError`` for a path that cannot be used and ``NotFoundError`` for one that leads nowhere."""
    import inspect

    why = _check_path(path) if isinstance(path, str) else "should be a string"
    if why:
        raise SpecificationError(f"the value path {path!r} {why}")
    current = obj
    done: list[str] = []
    for part in path.split("."):
        place = ".".join(done) or "the object"
        import pandas as pd

        found = False
        if isinstance(current, (Mapping, pd.Series, pd.DataFrame)):
            try:
                current, found = _key(current, part), True
            except (LookupError, TypeError, ValueError):
                found = False
        if not found:
            try:
                inspect.getattr_static(current, part)
                has = True
            except AttributeError:
                has = False
            if has:
                try:
                    current, found = getattr(current, part), True
                except Exception as exc:  # noqa: BLE001 - a property that fails: the value cannot be read
                    raise NotFoundError(f"{place}.{part} could not be read: {exc}") from None
        if not found:
            try:
                current, found = _derived(current, part), True
            except LookupError:
                pass
        if not found:
            try:
                current, found = _key(current, part), True
            except (LookupError, TypeError, ValueError, IndexError):
                pass
            except Exception as exc:  # noqa: BLE001 - a key lookup that fails in the object's own code
                raise NotFoundError(f"{place} has no {part!r}: {exc}") from None
        if not found:
            raise NotFoundError(f"{place} has no attribute or key {part!r}")
        if inspect.isroutine(current) or inspect.ismethoddescriptor(current):
            raise SpecificationError(f"{'.'.join([*done, part])} is a method: checks read values and never call")
        done.append(part)
    return current


def _plain(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compare(c: Check, actual: Any) -> tuple[bool, str]:
    actual = _plain(actual)
    if isinstance(c.expected, bool):
        if not isinstance(actual, bool):
            return False, f"expected true or false, found {type(actual).__name__}"
        return actual is c.expected, f"{actual} vs {c.expected}"
    if isinstance(c.expected, str):
        if not isinstance(actual, str):
            return False, f"expected text, found {type(actual).__name__}"
        return actual == c.expected, f"{actual!r} vs {c.expected!r}"
    if not _number(actual):
        return False, f"expected a number, found {type(actual).__name__}"
    if math.isnan(actual):
        return False, "the value is nan"
    rel = c.rel_tol if c.rel_tol is not None else (0.0 if c.abs_tol is not None else DEFAULT_REL_TOL)
    ok = math.isclose(actual, c.expected, rel_tol=rel, abs_tol=c.abs_tol or 0.0)
    tol = " and ".join(t for t in (f"abs_tol {c.abs_tol:g}" if c.abs_tol is not None else "",
                                   f"rel_tol {rel:g}" if rel else "") if t) or "exactly"
    return ok, f"{actual:.6g} vs {c.expected:.6g} ({tol})"


def _objects(session: Any) -> Mapping[str, Any]:
    objects = getattr(session, "objects", session)
    if not isinstance(objects, Mapping):
        raise SpecificationError("check needs a Session (or a mapping of names to objects)")
    return objects


def check(exercise: Exercise, session: Any, step: int | None = None) -> list[CheckResult]:
    """Run the exercise's checks (those of ``step``, or all) against the session's objects. Nothing is called: each
    value is read by ``resolve``. A check whose object or value is missing fails, saying why in ``detail``."""
    if step is not None:
        exercise.step(step)  # a step the exercise has
    objects = _objects(session)
    out: list[CheckResult] = []
    for c in exercise.checks_for(step):
        if c.object not in objects:
            out.append(CheckResult(c.id, c.step, False, None, c.expected, c.message, f"no object named {c.object!r} yet"))
            continue
        try:
            actual = _plain(resolve(objects[c.object], c.value))
        except (NotFoundError, SpecificationError) as exc:
            out.append(CheckResult(c.id, c.step, False, None, c.expected, c.message, str(exc)))
            continue
        passed, detail = _compare(c, actual)
        out.append(CheckResult(c.id, c.step, passed, actual, c.expected, c.message, f"{c.object}.{c.value}: {detail}"))
    return out


def __getattr__(name: str) -> Any:
    if name == "main":
        from .cli import main

        return main
    raise AttributeError(name)


HELP_TOPIC = """
    An exercise is a short guided task: each step says what to do in the app (the menu path and the dialog's
    fields), with hints, the same step as Python, and a **Check** that reads your session's results and says, step by
    step, whether they are right. Exercises come in packs, from a plugin or a folder.

    **Help → Exercises…** lists the packs; **Open folder…** adds one from a folder (it has `pack.toml` and
    `exercises/`). Choose an exercise and **Start**: its data load into the session once you agree, as File → Load
    data… steps (in History and the script, and Undo takes them back), and the **Exercise** panel opens on the right
    with every step. In each step, **Show hint** gives a hint (press again for the next), **Show the code** the step
    as Python, and **Check** runs the step's checks against the session: green when a value is as expected, red with
    a message when it is not, or when the object it reads does not exist yet.

    Each pack shows who it is for and, if it has one, its classification (such as "internal"): information only. An
    exercise can also come with models (Dynare, IRIS or pymetrix files): the panel's **Import…** opens the model's
    import dialog filled in with the pack's file and the name the pack gives the model, to run as any step.

    Checks only read values (a coefficient, a statistic, a setting); nothing in a pack runs in the app. A pack's
    reference solution runs only from the command line: `pymetrix-exercises check <pack>` loads each exercise's data
    in a fresh session, runs its solution and every check, and exits non-zero when one fails, so a pack's own CI can
    catch drift. `pymetrix-exercises new <folder>` starts a pack. The format, and how a plugin ships a pack, are in
    the guide (Writing an exercise pack) and `docs/exercise-packs.md`. The interface is provisional before 1.0.
"""


def add_help_topic() -> None:
    """The help topic ``exercises``."""
    from ..help import add_topic

    add_topic("exercises", "Exercises and exercise packs", HELP_TOPIC, (), ("how-do-i",))
