"""Projects: a session saved as one ``.pymx`` file (a zip archive).

Contents::

    manifest.json     format version, package version, saved time, objects
    commands.json     the steps: each command's stable ID, schema version and field values
    session.py        the same commands as a Python script
    data/<name>.parquet   every dataset the session started from (with metadata)
    opened/<n>.pymxr  every saved result an Open result step read (pymetrix.saved), as it was read
    results/<step>.pkl    results of slow steps, so opening does not compute them again
    course/<n>/...    course material attached to the session: notes, notebooks and the files they use
    attachments/<key>/<name>  plugins' own files (``session.attachments``), kept as they are even without the plugin

The manifest's ``save_log`` lists every save of the project: when, by which pymetrix (its version and, from a
checkout of its source, the commit and whether the code differed from it: ``pymetrix.build``), and which observers that keep
a record of the work (a lab book plugin, see ``pymetrix.commands.events``) were active, so work done without them
shows as a gap.

Opening a project restores the embedded datasets and replays the commands,
which rebuilds every estimate, test and forecast exactly (random seeds are
fixed). Data are embedded so a project still opens when the original files
have moved or been revised; loads (Load data, Load from connector and typed
``Dataset.read``) are taken from the embedded copy, so data from a service comes
back as it was read, not as revised since; ``Session.refresh`` reads it again. Course
material is unpacked to a folder in the user's cache and added to the help,
so an assignment's data, commands and notes travel as one file.

Opening does only the work needed. Dead ends (steps whose results were later replaced or
deleted, and nothing current uses) stay in the history and the script but are not run.
Results of slow steps (``EXPENSIVE`` seconds or more) are saved with the project and taken
from there, without running their steps again, when the project is trusted
(saved by you, or opened with ``trust=True``) and opened by the same version of pymetrix.
Otherwise, in Manual calculation mode, slow steps (and the steps that use their results) are
deferred, unless a later step changes their inputs (they then run in their place): listed as not computed until ``session.compute()`` (the app's Compute); in Automatic
mode they run on opening. A step that uses results not computed yet waits with them, and steps saved deferred
(``Session.add(cmd, deferred=True)``) stay deferred in Manual mode, computed later against their inputs as they are at
their place. A deferred step whose inputs are missing is kept deferred with a note (``CommandRecord.note``), in either
mode: opening never fails because of a pending step. The calculation mode and which results are frozen are saved with the project
(replaying the steps in order gives a frozen result its frozen value).

Steps typed in the Code or Immediate windows are Python, so they run on opening
only when the project is trusted (see ``pymetrix.trust``): projects you saved
yourself, or ones you chose to trust after reading their code
(``project_code(path)``; ``load_project(path, trust=True)``). Otherwise the data
and dialog steps are restored, and the typed code is kept but not run.
"""

from __future__ import annotations

import hashlib
import io
import json
import warnings
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from ..errors import SpecificationError
from .._version import __version__
from .registry import COMMAND_IDS, COMMANDS, MigrationError, upgrade
from .session import CommandRecord, Session, _deferred, _mentioned, load_target

EXPENSIVE = 2.0  # seconds: slower steps have their results saved with the project, and are deferred on opening in Manual mode

__all__ = ["SUFFIX", "ProjectCode", "ProjectTrustWarning", "load_project", "project_code", "save_project"]


class ProjectTrustWarning(UserWarning):
    """A project was opened without running its typed code, because the code is not trusted."""


@dataclass(frozen=True)
class ProjectCode:
    """The typed code a project would run when opened, and whether it is trusted."""

    steps: tuple[str, ...]
    fingerprint: str
    trusted: bool

    @property
    def needs_trust(self) -> bool:
        return bool(self.steps) and not self.trusted


def _code_steps(commands: list[dict[str, Any]]) -> list[str]:
    """Typed code that replay would run (IPython-only syntax is recorded but never run)."""
    from .code import _python

    return [e["fields"]["code"] for e in commands if (e.get("id") == "pymetrix:run_code" or e.get("command") == "RunCode")
            and _python(e["fields"]["code"]) is not None]


def _entry(command: Any, call: Any = None) -> dict[str, Any]:
    """A step as saved: stable ID, schema version, class name (for people reading the file) and fields; for a dialog
    built on an API call, also that call (for readers and tools: opening a project replays the dialog's fields). A
    step whose plugin is missing is written back as it was read."""
    from .missing import MissingStep

    if isinstance(command, MissingStep):
        return {k: v for k, v in command.entry.items() if k not in ("seconds", "outputs", "dead", "deferred", "proposed_by")}
    entry = {"id": command.command_id, "version": command.schema_version, "command": type(command).__name__,
             "fields": asdict(command)}
    if call is not None and call is not command:
        entry["call"] = {"function": call.function, "version": call.version, "receiver": call.receiver,
                         "arguments": dict(call.arguments), "output": call.output}
    return entry


def project_code(path: str | Path) -> ProjectCode:
    """What typed code opening the project would run: read it before trusting the project."""
    from ..trust import code_fingerprint, is_trusted

    with zipfile.ZipFile(Path(path)) as z:
        steps = _code_steps(json.loads(z.read("commands.json")))
    fp = code_fingerprint(steps)
    return ProjectCode(tuple(steps), fp, not steps or is_trusted(fp))

_NOT_RUN = "not run: the project's typed code is not trusted"

FORMAT_VERSION = 2  # 1: steps by class name (to October 2026); 2: by stable ID and schema version
SUFFIX = ".pymx"
_SAFE_NAME = __import__("re").compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")  # an attachment's key or file name: no paths


def _starting_points(session: Session) -> tuple[dict[str, Any], dict[str, Any]]:
    """Datasets replay cannot recreate, as they were before any command changed them:
    those placed in the session directly, and those read by Load data."""
    from ..data import Dataset

    placed = dict(session.initial)
    for name, obj in session.objects.items():  # objects added directly after the last command
        if name not in session.made and name not in placed:
            placed[name] = obj
    datasets = {n: o for n, o in placed.items() if isinstance(o, Dataset)}
    datasets.update(session.loaded)
    unsaved = sorted(n for n, o in placed.items() if not isinstance(o, Dataset))
    return datasets, {"not_saved": unsaved, "placed": sorted(n for n in datasets if n not in session.loaded)}


def save_project(session: Session, path: str | Path, as_copy: bool = False) -> Path:
    """Save the session as a project. ``as_copy=True`` writes a copy (an autosave) and leaves the session's own path
    and unsaved-changes flag as they were."""
    from ..io.parquet import write_parquet

    path = Path(path)
    if path.suffix != SUFFIX:
        path = path.with_suffix(SUFFIX)
    session._emit("saving", path=str(path), detail={"as_copy": as_copy})  # observers may update their attachments now
    attachments = _attachments(session)
    from .events import recorders

    from ..build import build_info

    save_log = [*session.save_log, {"saved": datetime.now().astimezone().isoformat(timespec="seconds"), "pymetrix": __version__,
                                    **build_info(), "recorders": recorders(), **({"copy": True} if as_copy else {})}]
    needed = dict(zip(map(id, session.history), session.needed()))
    records = [r for r in session.history if r.ok or r.skipped or r.pruned or r.deferred]
    roots, info = _starting_points(session)
    cache = _cache(session, records, needed)
    manifest = {
        "format_version": FORMAT_VERSION,
        "package_version": __version__,
        "saved": datetime.now().isoformat(timespec="seconds"),
        "datasets": sorted(roots),
        "placed": info["placed"],  # datasets put into the session directly rather than loaded
        "objects": sorted(session.objects),
        "not_saved": info["not_saved"],  # objects that neither data nor commands can rebuild
        "courses": [{"name": _course_name(folder), "dir": f"course/{i}"} for i, folder in enumerate(session.courses)],
        "plugins": _plugins_used(records),  # who provides the steps, so a missing plugin can be named on opening
        "mode": session.mode,
        "frozen": sorted(session.frozen & set(session.objects)),
        "working_sets": {k: v for k, v in session.working_sets.items() if k in session.objects},  # the app's view of large datasets
        "report_outline": session.report_outline,  # the report being put together
        "save_log": save_log,  # every save: when, by which pymetrix, with which recorders active
    }
    if attachments:
        manifest["attachments"] = {key: sorted(files) for key, files in attachments.items()}
    opened = _opened(session, records)
    if opened:
        manifest["opened"] = {p: f"opened/{i}.pymxr" for i, p in enumerate(opened)}  # copies of the saved results opened
    commands = [_saved_entry(r, needed.get(id(r), False)) for r in records]
    if cache:
        from ..trust import code_fingerprint

        manifest["cache"] = {"package_version": __version__, "steps": {str(i): f"results/{i}.pkl" for i in cache},
                             "fingerprint": code_fingerprint([hashlib.sha256(b).hexdigest() for _, b in sorted(cache.items())])}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        z.writestr("commands.json", json.dumps(commands, indent=2))
        z.writestr("session.py", session.script())
        for name, ds in roots.items():
            buffer = io.BytesIO()
            write_parquet(ds, buffer)
            z.writestr(f"data/{name}.parquet", buffer.getvalue())
        for i, blob in cache.items():
            z.writestr(f"results/{i}.pkl", blob)
        for p, blob in opened.items():
            z.writestr(manifest["opened"][p], blob)
        for i, folder in enumerate(session.courses):
            for file in _course_files(Path(folder)):
                z.write(file, f"course/{i}/{file.relative_to(folder).as_posix()}")
        for key, files in attachments.items():
            for name, blob in files.items():
                z.writestr(f"attachments/{key}/{name}", blob)
    if not as_copy:
        session.path = path
        session.dirty = False
        session.save_log = save_log
    _trust_own_code(session, records, path)
    if cache:
        from ..trust import remember

        remember(manifest["cache"]["fingerprint"], f"{path.name} (saved results)")
    return path


def _attachments(session: Session) -> dict[str, dict[str, bytes]]:
    """The plugins' files to save, checked: plain names, and bytes."""
    out: dict[str, dict[str, bytes]] = {}
    for key, files in session.attachments.items():
        if not _SAFE_NAME.fullmatch(str(key)):
            raise SpecificationError(f"an attachment's key is a plain name (letters, digits, '_', '.', '-'): {key!r}")
        for name, blob in files.items():
            if not _SAFE_NAME.fullmatch(str(name)):
                raise SpecificationError(f"an attachment's file name is a plain name, with no folders: {key}/{name!r}")
            if not isinstance(blob, (bytes, bytearray)):
                raise SpecificationError(f"attachment {key}/{name} must be bytes, not {type(blob).__name__}")
        if files:
            out[str(key)] = {str(n): bytes(b) for n, b in files.items()}
    return out


def _read_attachments(z: zipfile.ZipFile, manifest: dict[str, Any]) -> dict[str, dict[str, bytes]]:
    """The plugins' files a project keeps (plain files: no trust needed, as nothing in them runs)."""
    out: dict[str, dict[str, bytes]] = {}
    for key, names in manifest.get("attachments", {}).items():
        if not _SAFE_NAME.fullmatch(str(key)):
            continue
        out[key] = {n: z.read(f"attachments/{key}/{n}") for n in names if _SAFE_NAME.fullmatch(str(n))}
    return out


def _opened(session: Session, records: list[CommandRecord]) -> dict[str, bytes]:
    """The saved results the project's Open result steps read, as read: the project opens without the files."""
    from .builtin import OpenResult

    paths = dict.fromkeys(r.command.path for r in records if isinstance(r.command, OpenResult))
    return {p: session.opened[p] for p in paths if p in session.opened}


def _saved_entry(record: CommandRecord, needed: bool) -> dict[str, Any]:
    entry = _entry(record.command, record.call)
    entry["seconds"] = round(record.seconds, 3)
    entry["outputs"] = list(record.outputs)
    if record.pruned or not needed:
        entry["dead"] = True
    if record.deferred:
        entry["deferred"] = True
    if record.random_generator != "pcg64":  # the default is not written: projects saved before it existed used it
        entry["random_generator"] = record.random_generator
    if record.proposed_by:
        entry["proposed_by"] = record.proposed_by
    return entry


def _cache(session: Session, records: list[CommandRecord], needed: dict[int, bool]) -> dict[int, bytes]:
    """Pickled results to save, by step: those of slow steps, still current."""
    import pickle

    out: dict[int, bytes] = {}
    for i, r in enumerate(records):
        if not r.ok or r.typed or not needed.get(id(r), False):
            continue
        current = {n: session.objects[n] for n in r.outputs if n in session.objects and session.versions.get(n) == r.writes.get(n)}
        if not current or r.seconds < EXPENSIVE:
            continue
        if any(getattr(obj, "needs_backend", False) for obj in current.values()):
            continue  # it needs a library's own object, which projects never contain: computed again on opening
        try:
            out[i] = pickle.dumps(current, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:  # noqa: BLE001 - a result that cannot be saved is computed again on opening
            continue
    return out


def _trust_own_code(session: Session, records: list[CommandRecord], path: Path) -> None:
    """Code that ran in this session is the user's own: trust the saved project's code."""
    from ..trust import code_fingerprint, remember

    if any(r.skipped for r in records):
        return
    steps = _code_steps([_entry(r.command) for r in records])
    if steps:
        remember(code_fingerprint(steps), path.name)


def load_project(path: str | Path, trust: bool | None = None) -> Session:
    """Open a project: restore its data and replay its steps.

    ``trust``: None (default) runs the typed code only if it is trusted; True runs it and
    trusts it from now on; False never runs it. Code that is not run is kept in the
    history, marked as not run, and saved again with the project."""
    from ..figures import set_house_style

    set_house_style(None)  # a project's house style is the one its own steps set
    from ..trust import code_fingerprint, is_trusted, remember

    from ..io.parquet import read_parquet

    path = Path(path)
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("manifest.json"))
        if manifest.get("format_version", 1) > FORMAT_VERSION:
            raise SpecificationError(f"{path.name} was saved by a newer version of pymetrix ({manifest.get('package_version', '?')}); "
                             f"upgrade pymetrix to open it")
        commands = json.loads(z.read("commands.json"))
        datasets = {name: read_parquet(io.BytesIO(z.read(f"data/{name}.parquet"))) for name in manifest.get("datasets", [])}
        course_dirs = _unpack_courses(z, manifest.get("courses", []))
        opened = {p: z.read(name) for p, name in manifest.get("opened", {}).items()}  # plain files: no trust needed
        attachments = _read_attachments(z, manifest)
        cache = _usable_cache(z, manifest, path, trust)
    steps = _code_steps(commands)
    fingerprint = code_fingerprint(steps)
    run_code = not steps or (is_trusted(fingerprint) if trust is None else trust)
    if steps and trust is True:
        remember(fingerprint, path.name)
    if steps and not run_code:
        warnings.warn(ProjectTrustWarning(
            f"{path.name}: {len(steps)} step(s) of typed code were not run, because the project's code is not trusted. "
            "Read it with project_code(path).steps; to run it, open the project with load_project(path, trust=True)."), stacklevel=2)
    from ..output import quiet_show
    from ..plugins import load

    load("commands")
    with quiet_show():
        return _replay(path, manifest, commands, datasets, course_dirs, run_code, cache, opened, attachments)


def _usable_cache(z: zipfile.ZipFile, manifest: dict[str, Any], path: Path, trust: bool | None) -> dict[str, bytes]:
    """The saved results, if they may be used: saved by this version of pymetrix, and trusted (unpickling runs code)."""
    from ..trust import code_fingerprint, is_trusted, remember

    info = manifest.get("cache")
    if not info or trust is False or info.get("package_version") != __version__:
        return {}
    blobs = {step: z.read(name) for step, name in info.get("steps", {}).items()}
    fingerprint = code_fingerprint([hashlib.sha256(b).hexdigest() for _, b in sorted(blobs.items(), key=lambda kv: int(kv[0]))])
    if fingerprint != info.get("fingerprint"):
        return {}
    if trust is True:
        remember(fingerprint, f"{path.name} (saved results)")
    elif not is_trusted(fingerprint):
        return {}
    return blobs


def _replay(path: Path, manifest: dict[str, Any], commands: list[Any], datasets: dict[str, Any], course_dirs: list[Path],
            run_code: bool = True, cache: dict[str, bytes] | None = None, opened: dict[str, bytes] | None = None,
            attachments: dict[str, dict[str, bytes]] | None = None) -> Session:
    session = Session()
    session.opened.update(opened or {})  # Open result steps read these copies, not the files
    session.attachments = {k: dict(v) for k, v in (attachments or {}).items()}  # before the steps: observers may read them
    session.save_log = list(manifest.get("save_log", []))
    session._opening = True
    try:
        _replay_steps(session, path, manifest, commands, datasets, course_dirs, run_code, cache)
    finally:
        session._opening = False
    session._emit("opened", path=str(path), records=tuple(session.history))
    return session


def _replay_steps(session: Session, path: Path, manifest: dict[str, Any], commands: list[Any], datasets: dict[str, Any],
                  course_dirs: list[Path], run_code: bool, cache: dict[str, bytes] | None) -> None:
    mode = manifest.get("mode", "manual")
    frozen = set(manifest.get("frozen", []))
    cache = cache or {}
    deferred: set[str] = set()  # results of deferred steps: steps that use them wait too
    for name in manifest.get("placed", []):
        session.objects[name] = datasets[name]
        session.initial[name] = datasets[name]
    unavailable: set[str] = set()  # what steps that could not run would have made: steps using it cannot run either
    session.missing_plugins = {}
    from .._random import get_random_generator, set_random_generator

    before = get_random_generator()
    try:
        _replay_entries(session, path, manifest, commands, datasets, run_code, cache, mode, deferred, unavailable)
    finally:
        set_random_generator(before)  # opening a project leaves the user's own setting as it was
    for folder in course_dirs:
        session.attach_course(folder)
    session.frozen = frozen & set(session.objects)
    session.working_sets = {k: v for k, v in manifest.get("working_sets", {}).items() if k in session.objects}
    session.report_outline = list(manifest.get("report_outline", []))
    session.mode = mode
    session.path = path
    session.dirty = False
    session._undo.clear()  # an opened project starts afresh: its replayed steps are not undone one by one


def _use_generator(entry: dict[str, Any], path: Path) -> None:
    """Replay a step with the random generator it was run with (saved with the step when not the default PCG64)."""
    from .._random import GENERATORS, set_random_generator

    name = entry.get("random_generator", "pcg64")
    if name not in GENERATORS:
        warnings.warn(f"{path.name}: a step was run with the random generator {name!r}, which this pymetrix does not have; "
                      "it is replayed with PCG64, so its random draws differ", stacklevel=3)
        name = "pcg64"
    set_random_generator(name)


def _replay_entries(session: Session, path: Path, manifest: dict[str, Any], commands: list[Any], datasets: dict[str, Any],
                    run_code: bool, cache: dict[str, bytes], mode: str, deferred: set[str], unavailable: set[str]) -> None:
    """The steps of a project, replayed in order (see ``_replay_steps``)."""
    commands = _revive_used(commands)
    for i, entry in enumerate(commands):
        first = len(session.history)
        _replay_entry(session, path, manifest, commands, datasets, run_code, cache, mode, deferred, unavailable, i, entry)
        by = entry.get("proposed_by")
        if by and len(session.history) > first:  # the step an assistant proposed: marked again, as saved
            session.history[first] = replace(session.history[first], proposed_by=" ".join(str(by).split())[:60])


def _revive_used(commands: list[Any]) -> list[Any]:
    """Steps saved as dead ends that a deferred step after them uses are run again on opening: projects saved by
    pymetrix 0.1.0a16 and earlier marked them dead (a deferred step had no recorded inputs), and the deferred step
    could then not run. Which objects a step uses is told from its saved fields, as ``_mentioned`` does."""
    import re

    out = [dict(e) for e in commands]
    queue = [j for j, e in enumerate(out) if e.get("deferred") and not e.get("dead")]
    while queue:
        j = queue.pop()
        text = json.dumps(out[j].get("fields", {}))
        made: dict[str, int] = {}  # each name -> the last step before j that makes it
        for k in range(j):
            made.update(dict.fromkeys(out[k].get("outputs", ()), k))
        for name, k in made.items():
            if out[k].get("dead") and re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)", text):
                del out[k]["dead"]
                queue.append(k)
    return out


def _replay_entry(session: Session, path: Path, manifest: dict[str, Any], commands: list[Any], datasets: dict[str, Any],
                  run_code: bool, cache: dict[str, bytes], mode: str, deferred: set[str], unavailable: set[str], i: int,
                  entry: dict[str, Any]) -> None:
    """One saved step, replayed (or kept unrun, or restored from the cache)."""
    import pickle

    from .code import RunCode

    _use_generator(entry, path)
    outputs, seconds = tuple(entry.get("outputs", ())), float(entry.get("seconds", 0.0))
    missing = _missing_step(entry, manifest)
    if missing is not None:  # its plugin is not here: kept, not run, saved again as it was
        session.history.append(CommandRecord(missing, missing.to_code(), ok=False, skipped=True, outputs=outputs, seconds=seconds,
                                             error=f"not run: {missing.reason}"))
        session.missing_plugins.setdefault(missing.plugin_label, missing.reason)
        unavailable |= set(outputs)
        return
    info, fields = _resolve(entry, manifest, path)
    command = info.cls(**fields)
    if entry.get("dead"):
        session.history.append(CommandRecord(command, command.to_code(), ok=False, pruned=True, outputs=outputs, seconds=seconds))
        return
    if str(i) in cache:
        session.restore(command, pickle.loads(cache[str(i)]), seconds)  # noqa: S301 - trusted: saved by this user
        return
    from ..errors import PluginUnavailable

    try:
        code = command.to_code()
    except (PluginUnavailable, ImportError) as exc:  # even writing its code needs a method that is missing or disabled
        session.history.append(CommandRecord(command, f"# not run: {exc}", ok=False, skipped=True, outputs=outputs, seconds=seconds,
                                             error=f"not run: {exc}"))
        session.missing_plugins.setdefault(str(exc).split(":")[0], str(exc))
        unavailable |= set(outputs)
        return
    stub = CommandRecord(command, code, ok=False, outputs=outputs, seconds=seconds)
    needs = _mentioned(stub, sorted(unavailable))
    if needs:  # it uses what a step from a missing plugin would have made
        session.history.append(replace(stub, skipped=True, error=f"not run: it uses {', '.join(sorted(needs))}, made by a step "
                                                                 "that could not run (its plugin is not available, or its "
                                                                 "row condition was refused)"))
        unavailable |= set(outputs)
        return
    later = {n for e in commands[i + 1:] if not e.get("dead") for n in e.get("outputs", ())}
    added = bool(entry.get("deferred"))  # deferred where it was saved (Session.add, or deferred on opening, not computed)
    if (_mentioned(stub, sorted(deferred))  # it uses results not computed yet: it cannot run in its place
            or (added and mode == "manual")  # computed later against its inputs as they are here (Session.compute)
            or (mode == "manual" and seconds >= EXPENSIVE and not _mentioned(stub, sorted(later - set(outputs))))):
        session.history.append(_deferred(session, stub))
        deferred |= set(outputs)
        return
    if added and (record := _deferred(session, stub)).note:  # Automatic mode, an input missing: kept, never an error
        session.history.append(record)
        deferred |= set(outputs)
        return
    read_into = load_target(command)
    if read_into and read_into in datasets:
        # the embedded copy stands in for the original file or connector: the data as it was read
        session.objects[read_into] = session.loaded[read_into] = datasets[read_into]
        session.made.add(read_into)
        session.history.append(CommandRecord(command, command.to_code(), ok=True, writes={read_into: 0}, outputs=(read_into,),
                                             call=command if type(command).__name__ == "Call" else None))
        return
    if not run_code and isinstance(command, RunCode) and command.replayable:
        session.history.append(CommandRecord(command, command.to_code(), ok=False, error=_NOT_RUN, skipped=True))
        return
    from ..data.conditions import ConditionError, check_condition

    try:
        where = getattr(command, "where", None)  # a row condition: checked before the step's own validation sees it
        if isinstance(where, str) and where.strip():
            check_condition(where.strip())
        session.run(command)
    except Exception as exc:
        from ..errors import PluginUnavailable

        if isinstance(exc, (PluginUnavailable, ImportError)):  # a method or library it needs is missing or disabled
            if session.history and session.history[-1].command is command:
                session.history.pop()
            session.history.append(CommandRecord(command, command.to_code(), ok=False, skipped=True, outputs=outputs, seconds=seconds,
                                                 error=f"not run: {exc}"))
            session.missing_plugins.setdefault(str(exc).split(":")[0], str(exc))
            unavailable |= set(outputs)
            return
        if isinstance(exc, ConditionError):  # a row condition that could run code (or is not one): kept, never run
            if session.history and session.history[-1].command is command:
                session.history.pop()
            session.history.append(CommandRecord(command, command.to_code(), ok=False, skipped=True, outputs=outputs,
                                                 seconds=seconds, error=f"not run: {exc}"))
            unavailable |= set(outputs)
            return
        if run_code:
            raise
        # it may need objects the code that was not run would have made: keep it, unrun
        if session.history and session.history[-1].command is command:
            session.history.pop()
        session.history.append(CommandRecord(command, command.to_code(), ok=False, skipped=True,
                                             error=f"not run: {exc} (it may need typed code that was not run)"))


def _plugins_used(records: list[CommandRecord]) -> dict[str, str]:
    from .missing import MissingStep

    used: dict[str, str] = {}
    for r in records:
        if isinstance(r.command, MissingStep):  # still needed by the project: keep naming it
            if r.command.plugin_name:
                used[r.command.plugin_name] = r.command.plugin_version
            continue
        info = COMMAND_IDS.get(r.command.command_id)
        if info is not None:
            used[info.plugin] = info.plugin_version
    return dict(sorted(used.items()))


def _missing_step(entry: dict[str, Any], manifest: dict[str, Any]) -> Any:
    """A placeholder for a saved step this installation cannot run (its plugin is not installed, or is disabled),
    or None when it can."""
    from ..calls import API_CALLS
    from ..plugins import PluginInfo, is_disabled
    from .missing import MissingStep

    saved = manifest.get("plugins", {})

    def owner(prefix: str) -> tuple[str, str]:
        for n, v in saved.items():
            if PluginInfo(n, v).prefix.replace("-", "_") == prefix.replace("-", "_"):
                return n, v
        return f"pymetrix-{prefix}" if prefix and prefix != "pymetrix" else prefix, ""

    if "id" in entry:
        cid = entry["id"]
        if cid == "pymetrix:call":  # a recorded API call: its function must be installed
            function = (entry.get("fields") or {}).get("function", "")
            if function and function not in API_CALLS:
                name, version = owner(function.partition(":")[0])
                why = "disabled" if is_disabled(name) else "not installed"
                return MissingStep(entry, name, version, f"{function} comes from {name}, which is {why}{_hint(name, why)}")
            return None
        if cid in COMMAND_IDS:
            return None
        name, version = owner(cid.partition(":")[0])
    else:
        if _LEGACY_COMMANDS.get(entry["command"], entry["command"]) in COMMANDS:
            return None
        name, version = "", ""
    why = "disabled" if name and is_disabled(name) else "not installed"
    what = entry.get("id") or entry.get("command")
    return MissingStep(entry, name, version, f"the step {what} comes from {name or 'a plugin'}, which is {why}{_hint(name, why)}")


def _hint(name: str, why: str) -> str:
    if not name:
        return ""
    return ": enable it in Help → Plugins" if why == "disabled" else f": pip install {name}"


def _resolve(entry: dict[str, Any], manifest: dict[str, Any], path: Path) -> tuple[Any, dict[str, Any]]:
    """The installed command for a saved step, and its fields brought up to the command's schema."""
    from ..plugins import PluginInfo

    fields = dict(entry["fields"])
    if "id" in entry:  # format 2
        cid, version = entry["id"], int(entry.get("version", 1))
        info = COMMAND_IDS.get(cid)
        if info is None:
            prefix = cid.partition(":")[0]
            saved = manifest.get("plugins", {})
            source = next((f"{n} {v}" for n, v in saved.items() if PluginInfo(n, v).prefix.replace("-", "_") == prefix), prefix)
            raise SpecificationError(f"{path.name} uses the step {cid} from {source}, which is not installed (a missing plugin?)")
    else:  # format 1: steps by class name, with the renames made since
        name = _LEGACY_COMMANDS.get(entry["command"], entry["command"])
        info = COMMANDS.get(name)
        if info is None:
            raise SpecificationError(f"{path.name} uses a command that is not installed: {entry['command']} (a missing plugin?)")
        for old, new in _LEGACY_FIELDS.get(name, {}).items():
            if old in fields:
                fields[new] = fields.pop(old)
        cid, version = info.id, 1
    try:
        return info, upgrade(cid, version, fields)
    except MigrationError as exc:
        raise SpecificationError(f"{path.name}: {exc}") from None


# Format-1 projects named steps by class; these renames happened before steps had IDs and schema
# versions. Later changes are migrations (see pymetrix.commands.registry.migration).
_LEGACY_COMMANDS = {"DefineGroup": "SelectSeries"}
_LEGACY_FIELDS = {name: {"group": "selection"} for name in ("DeriveEach", "FitEach", "TestEach")}

_SKIP = {".ipynb_checkpoints", "__pycache__", ".git"}


def _course_files(folder: Path) -> list[Path]:
    """Everything in a course folder except checkpoints, caches and hidden files."""
    return sorted(f for f in folder.rglob("*") if f.is_file() and not (_SKIP & set(f.relative_to(folder).parts))
                  and not any(part.startswith(".") for part in f.relative_to(folder).parts))


def _course_name(folder: str) -> str:
    from ..help import courses

    return next((c.name for c in courses() if str(c.path) == folder), Path(folder).name)


def course_cache() -> Path:
    """Where course material from projects is unpacked: one folder per distinct content."""
    import os
    import tempfile

    base = os.environ.get("PYMETRIX_CACHE") or os.path.join(tempfile.gettempdir(), "pymetrix")
    return Path(base) / "courses"


def _unpack_courses(z: zipfile.ZipFile, entries: list[dict[str, Any]]) -> list[Path]:
    """Unpack each course into the cache (reused when the content is unchanged); refuses paths that escape it."""
    import hashlib

    out = []
    for entry in entries:
        prefix = entry["dir"].rstrip("/") + "/"
        members = sorted(n for n in z.namelist() if n.startswith(prefix) and not n.endswith("/"))
        digest = hashlib.sha256()
        for name in members:
            digest.update(name.encode())
            digest.update(z.read(name))
        target = (course_cache() / f"{_slug(entry.get('name', 'course'))}-{digest.hexdigest()[:12]}").resolve()
        if not target.exists():
            import shutil

            staging = target.with_name(target.name + ".partial")
            shutil.rmtree(staging, ignore_errors=True)  # left by an interrupted unpack
            for name in members:
                dest = (staging / name[len(prefix):]).resolve()
                if not dest.is_relative_to(staging.resolve()):
                    raise SpecificationError(f"unsafe path in project: {name}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(name))
            staging.mkdir(parents=True, exist_ok=True)
            _move_into_place(staging, target)
        out.append(target)
    return out


def _move_into_place(staging: Path, target: Path) -> None:
    """Rename the unpacked folder into place. On Windows, OneDrive and antivirus scanners briefly lock
    new files, so retry, then fall back to copying."""
    import shutil
    import time

    for attempt in range(5):
        try:
            staging.replace(target)
            return
        except PermissionError:
            if target.exists():  # another process (a second window) unpacked the same content
                shutil.rmtree(staging, ignore_errors=True)
                return
            time.sleep(0.2 * (attempt + 1))
    shutil.copytree(staging, target, dirs_exist_ok=True)
    shutil.rmtree(staging, ignore_errors=True)


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "course"
