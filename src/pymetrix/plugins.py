"""Extension API v1: plugins and the registries that menus, dialogs and scripts read.

Every extension point has a registry (``registry("methods")``, ...). Built-in
components and the bundled reference plugins register on import; other
packages register through entry points in their own ``pyproject.toml``::

    [project.entry-points."pymetrix.methods"]
    liml = "pymetrix_linearmodels:LIML"

    [project.entry-points."pymetrix.io"]
    sdmx = "pymetrix_sdmx:register"        # a function: called, registers itself

An entry point naming a class registers that class under the entry point's
name; one naming a function calls it (it registers through ``register`` or
the reader, writer and view functions). The package that owns the entry
point describes itself with a module-level ``PLUGIN = PluginInfo(...)``.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import json
import os
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

from .errors import NotFoundError, PluginUnavailable
from ._version import __version__

API = "1.10"  # 1.1 (additive): the app's workspaces, menus and panels; results' stage and next steps; 1.2: report templates, example datasets;
# 1.3: session observers (pymetrix.commands.events) and project attachments; 1.4: migration, call_migration and
# renamed_arguments, for renaming a call's arguments or a dialog's fields; 1.5: who acted (SessionEvent.actor, Actor); 1.6: report notes
# (ReportNote: a line per reported result, under the plugin's heading); 1.7 (additive): AppInterface.open_demo(name), a panel
# opens a demo project as File → Open demo project does
# 1.8 (additive): build_mcp_server, pymetrix's MCP tools over a session that lives elsewhere (the app's)
# 1.9 (additive): Session.proposing(command_cls, by) and CommandRecord.proposed_by: a step an assistant proposed and the user ran
# 1.10 (additive): the core EViews reader's text helpers (eviews_statements, parse_eviews_expression, Diagnostic...), for
# a plugin that translates EViews programs
SUPPORTED_APIS = ("1",)  # the current major version and, once there is one, the previous
GROUPS = ("methods", "systems", "models", "variances", "tests", "filters", "seasonal", "solvers", "io", "views", "commands", "courses",
          "workspaces", "menus", "panels", "report_templates", "datasets", "observers", "report_notes")

__all__ = ["API", "GROUPS", "CORE", "Entry", "PluginInfo", "Registry", "disable", "enable", "is_disabled", "load", "pending",
           "load_problems", "plugin_of_command", "plugin_table", "plugins", "register", "registry", "set_mode"]


def installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


@functools.lru_cache(maxsize=1)
def _distributions() -> dict[str, list[str]]:
    from importlib.metadata import packages_distributions

    return dict(packages_distributions())


def installed_version(module: str) -> str | None:
    """The installed version of the distribution that provides ``module`` (None if unknown)."""
    from importlib.metadata import PackageNotFoundError, version

    top = module.split(".")[0]
    for dist in _distributions().get(top, [top]):
        try:
            return version(dist)
        except PackageNotFoundError:
            continue
    return None


def version_problem(module: str, spec: str) -> str | None:
    """Why the installed ``module`` does not meet ``spec`` (e.g. ">=6"), or None when it does
    (or when its version cannot be told, or the spec is empty)."""
    if not spec or not spec.strip():
        return None
    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.version import InvalidVersion, Version
    except ImportError:  # packaging is a dependency; without it, versions go unchecked
        return None
    found = installed_version(module)
    if found is None:
        return None
    try:
        ok = Version(found) in SpecifierSet(spec)
    except (InvalidSpecifier, InvalidVersion):
        return None
    return None if ok else f"{module} {found} is installed, but {spec.strip()} is needed"


@dataclass(frozen=True)
class PluginInfo:
    """Who provides a set of extensions, what they need, and under which licence.

    ``requires`` maps importable library names to the versions supported, e.g.
    ``{"linearmodels": ">=6"}``. ``install`` is the command shown when they are missing.
    """

    name: str
    version: str
    api: str = API
    requires: Mapping[str, str] = field(default_factory=dict)
    license: str = ""
    maintainer: str = ""
    description: str = ""
    install: str = ""

    @property
    def prefix(self) -> str:
        """The plugin part of qualified identifiers: ``pymetrix-linearmodels`` -> ``linearmodels``."""
        return self.name.removeprefix("pymetrix-").removeprefix("pymetrix_")

    @property
    def missing(self) -> list[str]:
        return [lib for lib in self.requires if not installed(lib)]

    @property
    def unsupported(self) -> list[str]:
        """Libraries that are installed in a version the plugin does not support, with what is needed."""
        return [p for lib, spec in self.requires.items() if installed(lib) and (p := version_problem(lib, spec))]

    @property
    def available(self) -> bool:
        return not self.missing and not self.unsupported

    @property
    def problem(self) -> str | None:
        """Why the plugin cannot be used, with what to do about it; None when it can."""
        if self.missing:
            return f"needs {', '.join(self.missing)}: {self.install_hint}"
        if self.unsupported:
            wanted = " ".join(f'"{lib}{spec.strip()}"' for lib, spec in self.requires.items() if installed(lib) and version_problem(lib, spec))
            return "; ".join(self.unsupported) + f": pip install {wanted}"
        return None

    @property
    def install_hint(self) -> str:
        return self.install or f"pip install {self.name}"

    @property
    def supported(self) -> bool:
        return self.api.split(".")[0] in SUPPORTED_APIS


CORE = PluginInfo("pymetrix", __version__, license="BSD-3-Clause", maintainer="Geoffrey Heenan",
                  description="Core estimation, testing, filtering and solving")


@dataclass(frozen=True)
class Entry:
    """One registered extension: an object (usually a class) with its menu label and plugin."""

    group: str
    name: str
    label: str
    obj: Any
    plugin: PluginInfo
    kind: str = ""
    requires: tuple[str, ...] = ()
    preset: Mapping[str, Any] = field(default_factory=dict)  # settings this entry fixes, e.g. GJR-GARCH: asym=1

    @property
    def qualified(self) -> str:
        return f"{self.plugin.prefix}:{self.name}"

    @property
    def missing(self) -> list[str]:
        return [lib for lib in dict.fromkeys((*self.requires, *self.plugin.requires)) if not installed(lib)]

    @property
    def available(self) -> bool:
        return not self.missing and not self.plugin.unsupported and not is_disabled(self.plugin.name)

    @property
    def install_hint(self) -> str:
        return self.plugin.install_hint if self.plugin.requires else f"pip install {' '.join(self.missing)}"

    @property
    def problem(self) -> str | None:
        """Why the extension cannot be used, with what to do about it; None when it can."""
        if is_disabled(self.plugin.name):
            return f"{self.plugin.name} is disabled: enable it in Help → Plugins (or pymetrix.plugins.enable({self.plugin.name!r}))"
        if self.missing:
            return f"needs {', '.join(self.missing)}: {self.install_hint}"
        return self.plugin.problem

    def build(self, **settings: Any) -> Any:
        """An instance with the entry's preset settings, overridden by ``settings``."""
        return self.obj(**{**self.preset, **settings})


class Registry:
    """The extensions of one group, looked up by label, short name or qualified name."""

    def __init__(self, group: str) -> None:
        self.group = group
        self._entries: dict[str, Entry] = {}

    def add(self, entry: Entry) -> Entry:
        self._entries[entry.qualified] = entry
        return entry

    def entries(self, kind: str | None = None, available: bool | None = None) -> list[Entry]:
        load(self.group)
        return [e for e in self._entries.values()
                if (kind is None or e.kind == kind) and (available is None or e.available == available)]

    def __iter__(self) -> Iterator[Entry]:
        return iter(self.entries())

    def __len__(self) -> int:
        return len(self.entries())

    def display(self, entry: Entry) -> str:
        """The menu label; qualified only when two plugins use the same label."""
        clash = sum(e.label == entry.label for e in self._entries.values()) > 1
        return f"{entry.label} ({entry.qualified})" if clash else entry.label

    def labels(self, kind: str | None = None, available_only: bool = True) -> list[str]:
        return [self.display(e) for e in self.entries(kind, True if available_only else None)]

    def find(self, key: str) -> Entry | None:
        load(self.group)
        if key in self._entries:
            return self._entries[key]
        for test in (lambda e: self.display(e) == key, lambda e: e.label == key, lambda e: e.name == key,
                     lambda e: e.name.lower() == key.lower() or e.label.lower() == key.lower()):
            found = [e for e in self._entries.values() if test(e)]
            if len(found) == 1:
                return found[0]
            if len(found) > 1:
                raise NotFoundError(f"{key!r} is ambiguous in {self.group}: use one of {[e.qualified for e in found]}")
        return None

    def get(self, key: str) -> Entry:
        entry = self.find(key)
        if entry is not None and is_disabled(entry.plugin.name):
            raise PluginUnavailable(f"{key!r} comes from {entry.plugin.name}, which is disabled: enable it in Help → Plugins")
        if entry is None:
            raise NotFoundError(f"no {self.group[:-1] if self.group.endswith('s') else self.group} {key!r}; known: {self.labels(available_only=False)}")
        return entry

    def __contains__(self, key: object) -> bool:
        try:
            return isinstance(key, str) and self.find(key) is not None
        except KeyError:
            return True

    def of(self, obj: Any) -> Entry | None:
        """The entry for an object or class (the first registered, if several share the class)."""
        cls = obj if isinstance(obj, type) else type(obj)
        return next((e for e in self._entries.values() if e.obj is cls), None)


_REGISTRIES: dict[str, Registry] = {g: Registry(g) for g in GROUPS}
_PLUGINS: dict[str, PluginInfo] = {CORE.name: CORE}
_PROBLEMS: dict[str, str] = {}  # plugin name -> why it was not loaded
_current: list[PluginInfo] = []
_loaded: set[str] = set()


def registry(group: str) -> Registry:
    try:
        return _REGISTRIES[group.removeprefix("pymetrix.")]
    except KeyError:
        raise NotFoundError(f"no extension group {group!r}; groups: {GROUPS}") from None


def add_plugin(info: PluginInfo) -> PluginInfo:
    _PLUGINS.setdefault(info.name, info)
    return _PLUGINS[info.name]


def register(group: str, obj: Any = None, *, name: str | None = None, label: str | None = None, kind: str | None = None,
             requires: tuple[str, ...] | None = None, preset: Mapping[str, Any] | None = None, plugin: PluginInfo | None = None) -> Any:
    """Register ``obj`` in ``group``; usable as a decorator (``@register("filters")``).

    Defaults come from the class: ``label`` (else the class name), ``kind`` and
    ``requires`` class attributes. ``name`` (the short identifier) defaults to
    the lower-case class name.
    """
    if obj is None:
        return lambda o: register(group, o, name=name, label=label, kind=kind, requires=requires, preset=preset, plugin=plugin)
    info = add_plugin(plugin or (_current[-1] if _current else CORE))
    reg = registry(group)
    if reg.group == "tests":  # tests are named by their ``name`` in scripts: fit.test("breusch_godfrey")
        name = label = getattr(obj, "name", None) or name
    reg.add(Entry(
        group=reg.group,
        name=name or getattr(obj, "__name__", str(obj)).lower(),
        label=label or getattr(obj, "label", None) or getattr(obj, "__name__", str(obj)),
        obj=obj,
        plugin=info,
        kind=kind if kind is not None else getattr(obj, "kind", "") or "",
        requires=tuple(requires if requires is not None else getattr(obj, "requires", ())),
        preset=dict(preset or {}),
    ))
    return obj


def current_plugin() -> PluginInfo:
    """The plugin that registrations made now belong to (pymetrix itself outside a plugin)."""
    return _current[-1] if _current else add_plugin(CORE)


@contextmanager
def plugin_context(info: PluginInfo) -> Iterator[PluginInfo]:
    """Registrations inside the block belong to ``info``."""
    _current.append(add_plugin(info))
    try:
        yield _current[-1]
    finally:
        _current.pop()


def _plugin_of(ep: Any) -> PluginInfo:
    """The ``PLUGIN`` declared by the entry point's module or one of its parent packages."""
    module = ep.value.split(":")[0]
    parts = module.split(".")
    for i in range(len(parts), 0, -1):
        try:
            found = getattr(importlib.import_module(".".join(parts[:i])), "PLUGIN", None)
        except ImportError:
            break
        if isinstance(found, PluginInfo):
            return found
    dist = getattr(ep, "dist", None)
    return PluginInfo(dist.name if dist else parts[0], dist.version if dist else "")


def _entry_points(group: str) -> list[Any]:
    return list(entry_points(group=group))


# -- enabled, disabled and new plugins --------------------------------------------------------------------------------
#
# ~/.pymetrix/plugins.json (or $PYMETRIX_HOME/plugins.json) records the plugins enabled, with the version enabled,
# and those disabled; the app and scripts read the same file. A disabled plugin stays out of both: an installed
# one's entry points are not loaded, a bundled one's registrations are hidden. A plugin that is neither (new) is
# loaded with a warning in scripts, which can never stop to ask; the app asks first (``set_mode("app")``), from the
# package's metadata alone, since importing the plugin would run its code.

_STATE: dict[str, Any] | None = None
_MODE = {"mode": "script"}
_SKIPPED: set[str] = set()  # new plugins the app was told to leave out for now (this session)
_WARNED: set[str] = set()
FIRST_PARTY = ("pymetrix", "pymetrix-arch", "pymetrix-linearmodels", "pymetrix-sklearn", "pymetrix-connectors", "pymetrix-finance",
               "pymetrix-labbook", "pymetrix-psets", "pymetrix-guide",
               "pymetrix-assistant", "pymetrix-seasonal", "pymetrix-eviews", "pymetrix-ha")


def _state_path() -> str:
    from pathlib import Path

    base = os.environ.get("PYMETRIX_HOME") or os.path.join(Path.home(), ".pymetrix")
    return os.path.join(base, "plugins.json")


def plugin_state() -> dict[str, Any]:
    """The enabled plugins (name -> the version enabled) and the disabled ones (names)."""
    global _STATE
    if _STATE is None:
        try:
            with open(_state_path(), encoding="utf-8") as f:
                raw = json.load(f)
            _STATE = {"enabled": dict(raw.get("enabled", {})), "disabled": list(raw.get("disabled", []))}
        except (OSError, ValueError):
            _STATE = {"enabled": {}, "disabled": []}
    return _STATE


def _save_state() -> None:
    path = _state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plugin_state(), f, indent=2)


def _canonical(name: str) -> str:
    return name.lower().replace("_", "-")


def is_disabled(name: str) -> bool:
    """Whether the plugin is disabled (pymetrix itself never is)."""
    name = _canonical(name)
    return name != "pymetrix" and name in {_canonical(n) for n in plugin_state()["disabled"]}


def is_known(name: str) -> bool:
    """Whether the plugin has been enabled or disabled before (the bundled ones count as enabled)."""
    name = _canonical(name)
    state = plugin_state()
    return name in FIRST_PARTY or name in {_canonical(n) for n in (*state["enabled"], *state["disabled"])}


def enable(name: str, version: str = "") -> None:
    """Enable a plugin, here and in scripts; an installed one's extensions load from the next start (or ``load``)."""
    state = plugin_state()
    state["disabled"] = [n for n in state["disabled"] if _canonical(n) != _canonical(name)]
    state["enabled"][name] = version
    _SKIPPED.discard(_canonical(name))
    _save_state()


def disable(name: str) -> None:
    """Disable a plugin, here and in scripts. A bundled one's extensions are hidden at once; an installed one is not
    loaded from the next start."""
    if _canonical(name) == "pymetrix":
        raise NotFoundError("pymetrix itself cannot be disabled")
    state = plugin_state()
    if _canonical(name) not in {_canonical(n) for n in state["disabled"]}:
        state["disabled"].append(name)
    state["enabled"].pop(name, None)
    _save_state()


def set_mode(mode: str) -> None:
    """``"app"``: new plugins wait for the user's answer (``pending``); ``"script"`` (the default): they load, warned."""
    if mode not in ("app", "script"):
        raise NotFoundError("mode 'app' or 'script'")
    _MODE["mode"] = mode


def skip_for_now(name: str) -> None:
    """Leave a new plugin out for this session without enabling or disabling it."""
    _SKIPPED.add(_canonical(name))


@dataclass(frozen=True)
class PendingPlugin:
    """A new plugin as its package's metadata describes it, read without importing (running) it."""

    name: str
    version: str
    author: str
    license: str
    summary: str
    groups: tuple[str, ...]  # the extension groups it registers in: commands, methods, io...
    entry_points: tuple[str, ...]


def _dist_name(ep: Any) -> str:
    dist = getattr(ep, "dist", None)
    return dist.name if dist is not None and getattr(dist, "name", None) else ep.value.split(":")[0].split(".")[0]


def pending() -> list[PendingPlugin]:
    """Installed plugins neither enabled nor disabled yet: what the app asks about before loading them."""
    found: dict[str, dict[str, Any]] = {}
    for group in (*GROUPS, "plugins", "exercises"):  # exercises: packs (pymetrix.exercises), read only when listed
        for ep in _entry_points(f"pymetrix.{group}"):
            name = _dist_name(ep)
            if is_known(name) or _canonical(name) in _SKIPPED:
                continue
            info = found.setdefault(name, {"dist": getattr(ep, "dist", None), "groups": [], "eps": []})
            if group != "plugins" and group not in info["groups"]:
                info["groups"].append(group)
            info["eps"].append(f"{group}: {ep.name}")
    out = []
    for name, info in found.items():
        meta = getattr(info["dist"], "metadata", None)

        def field_(key: str, meta: Any = meta) -> str:
            try:
                return str(meta.get(key) or "") if meta is not None else ""
            except Exception:  # noqa: BLE001 - metadata is only read, and may be odd
                return ""

        out.append(PendingPlugin(name, getattr(info["dist"], "version", "") or "", field_("Author") or field_("Author-email"),
                                 field_("License-Expression") or field_("License"), field_("Summary"),
                                 tuple(info["groups"]), tuple(info["eps"])))
    return out


def _admit(ep: Any) -> bool:
    """Whether to load an entry point: not if its plugin is disabled, or new and waiting for the app's answer."""
    name = _dist_name(ep)
    if is_disabled(name):
        _PROBLEMS.setdefault(name, "disabled")
        return False
    if is_known(name):
        return True
    if _MODE["mode"] == "app" or _canonical(name) in _SKIPPED:
        return False
    if name not in _WARNED:
        _WARNED.add(name)
        warnings.warn(f"pymetrix plugin {name} is new: loaded, but not yet enabled in the app (Help → Plugins), "
                      f"or pymetrix.plugins.enable({name!r}) / disable({name!r})", stacklevel=3)
    return True


# -- why an installed plugin cannot be imported ------------------------------------------------------------------------
#
# A plugin's metadata (its entry points) can be installed while its code is not importable: most often an editable
# install (``pip install -e packages/...``) whose folder was moved, deleted or checked out at a branch without it.
# Python then says only "No module named 'pymetrix_finance'"; ``import_problem`` says which distribution, where it was
# installed from, and the command that fixes it, for this Python.

_DETAILS: dict[str, dict[str, str]] = {}  # plugin name -> {"summary", "detail", "command"}: an import failure explained


def _shell_quote(text: str) -> str:
    return f'"{text}"' if any(c in text for c in " \t&()'") else text


def _pip(*args: str) -> str:
    """A pip command for the Python running pymetrix (not whichever ``python`` comes first on the PATH)."""
    import sys

    return " ".join([_shell_quote(sys.executable or "python"), "-m", "pip", *(_shell_quote(a) for a in args)])


def _direct_url(dist: Any) -> dict[str, Any]:
    """The distribution's ``direct_url.json`` (PEP 610): where it was installed from; {} if none."""
    try:
        text = dist.read_text("direct_url.json")
        return dict(json.loads(text)) if text else {}
    except Exception:  # noqa: BLE001 - metadata is only read, and may be odd
        return {}


def _url_path(url: str) -> str:
    """A ``file://`` URL as a local path (``file:///C:/x`` -> ``C:\\x`` on Windows)."""
    from urllib.parse import unquote, urlparse
    from urllib.request import url2pathname

    parsed = urlparse(url)
    if parsed.scheme != "file":
        return url
    path = url2pathname(unquote(parsed.path))
    return f"//{parsed.netloc}{path}" if parsed.netloc and parsed.netloc != "localhost" else path


def _requirement_for(dist: Any, module: str) -> str | None:
    """The distribution among the plugin's ``Requires-Dist`` that provides ``module``, when its name says so."""
    import re

    want = re.sub(r"[-_.]+", "_", module.split(".")[0]).lower()
    for req in (getattr(dist, "requires", None) or []):
        name = re.split(r"[\s;\[<>=!~(]", req.strip(), maxsplit=1)[0]
        if name and re.sub(r"[-_.]+", "_", name).lower() in (want, f"py{want}", f"python_{want}"):
            return name
    return None


def import_problem(ep: Any, exc: BaseException) -> dict[str, str] | None:
    """Explain a ``ModuleNotFoundError`` raised importing an entry point: ``{"summary", "detail", "command"}``, or None
    for any other error. ``summary`` is one short line; ``detail`` names the distribution, its version, whether it is
    an editable install and from where (and whether that folder exists), and the Python in use; ``command`` is the fix."""
    import sys

    if not isinstance(exc, ModuleNotFoundError) or not exc.name:
        return None
    root = ep.value.split(":")[0].strip().split(".")[0]
    dist = getattr(ep, "dist", None)
    name = _dist_name(ep)
    version = getattr(dist, "version", "") or ""
    who = f"{name} {version}".strip()
    python = f"Python in use: {sys.executable}."
    if exc.name.split(".")[0] != root:  # the plugin's own code is there; something it imports is not
        package = _requirement_for(dist, exc.name) if dist is not None else None
        command = _pip("install", package or exc.name.split(".")[0])
        plugin = name.removeprefix("pymetrix-").removeprefix("pymetrix_")
        return {"summary": f"needs {exc.name}, which isn't installed",
                "detail": f"the {plugin} plugin needs {exc.name}, which isn't installed: {command}. {python}",
                "command": command}
    remove = _pip("uninstall", name)
    if dist is None:
        command = _pip("install", "--force-reinstall", name)
        return {"summary": f"its code ({root}) can't be found",
                "detail": f"{name}'s entry points are installed but its code ({root}) can't be found; reinstall it with "
                          f"{command}, or remove it: {remove}. {python}", "command": command}
    info = _direct_url(dist)
    editable = bool((info.get("dir_info") or {}).get("editable"))
    if editable:
        source = _url_path(str(info.get("url", ""))) if info.get("url") else ""
        exists = bool(source) and os.path.isdir(source)
        if exists:
            command = _pip("install", "-e", source)
            return {"summary": f"its code can't be found: an editable install whose folder has no {root}",
                    "detail": f"{who} is installed (editable, from {source}) but its code can't be found there — the folder "
                              f"exists but has no {root} (is the clone checked out at a branch without it?). Check out a "
                              f"branch that has it and reinstall: {command}, or remove it: {remove}. {python}",
                    "command": command}
        folder = os.path.basename(source.rstrip("/\\")) or name
        target = f"<clone>/packages/{folder}" if _canonical(name) in FIRST_PARTY else f"<the {folder} folder>"
        command = _pip("install", "-e", target)
        where = f"from {source}" if source else "from a folder not recorded"
        return {"summary": "its code can't be found: an editable install whose folder was moved or deleted",
                "detail": f"{who} is installed (editable, {where}) but its code can't be found there — the folder was moved "
                          f"or deleted. Reinstall it from your clone: {command}, or remove it: {remove}. {python}",
                "command": command}
    command = _pip("install", "--force-reinstall", name)
    return {"summary": "its files are missing: reinstall it",
            "detail": f"{who} is installed but its files are missing; reinstall with {command}, or remove it: {remove}. {python}",
            "command": command}


def _record_import_failure(ep: Any, exc: BaseException, fallback_name: str, fallback: str) -> str:
    """Record why an entry point could not be imported, explained when it can be; returns the plugin name used."""
    explained = import_problem(ep, exc)
    if explained is None:
        _PROBLEMS[fallback_name] = fallback
        return fallback_name
    name = _dist_name(ep)
    _PROBLEMS[name] = f"could not be imported: {explained['summary']}"
    _DETAILS[name] = explained
    return name


def problem_detail(name: str) -> dict[str, str] | None:
    """The full explanation of a plugin's load problem (``summary``, ``detail``, ``command``), when there is one."""
    found = _DETAILS.get(name)
    return found if found is not None and name in _PROBLEMS else None


def load(group: str) -> None:
    """Load the entry points of a group (once). Problems are recorded, not raised: see ``plugins()``."""
    group = group.removeprefix("pymetrix.")
    if group in _loaded:
        return
    _loaded.add(group)
    for ep in _entry_points(f"pymetrix.{group}"):
        label = getattr(getattr(ep, "dist", None), "name", None) or ep.value
        if not _admit(ep):  # decided from the package's metadata: nothing of the plugin has run
            continue
        try:
            info = _plugin_of(ep)
        except Exception as exc:  # a broken plugin must not stop the program
            _record_import_failure(ep, exc, label, f"could not be imported: {exc}")
            continue
        if not info.supported:
            _PROBLEMS[info.name] = f"written for extension API {info.api}; this pymetrix supports {', '.join(SUPPORTED_APIS)}"
            continue
        with plugin_context(info):
            try:
                obj = ep.load()
                if isinstance(obj, type):
                    if group == "commands":
                        from .commands.registry import command

                        command(getattr(obj, "menu", f"Plugins/{obj.__name__}"))(obj)
                    else:
                        register(group, obj, name=ep.name)
                elif callable(obj):
                    obj()
                else:
                    register(group, obj, name=ep.name)
            except Exception as exc:
                name = _record_import_failure(ep, exc, info.name, f"{ep.name} failed to load: {exc}")
                detail = problem_detail(name)
                warnings.warn(f"pymetrix plugin {name}: {detail['detail'] if detail else _PROBLEMS[name]}", stacklevel=2)


def load_all() -> None:
    for group in GROUPS:
        load(group)
    for ep in _entry_points("pymetrix.plugins"):  # plugins that only describe themselves
        if not _admit(ep):
            continue
        try:
            obj = ep.load()
            if isinstance(obj, PluginInfo):
                add_plugin(obj)
        except Exception as exc:
            _record_import_failure(ep, exc, ep.name, f"could not be imported: {exc}")


def load_problems() -> dict[str, str]:
    """Plugins that could not be loaded (import errors, an unsupported API, a failing registration): name -> why. A
    disabled plugin is not a problem, nor a bundled one whose optional library is not installed."""
    load_all()
    return {name: why for name, why in _PROBLEMS.items() if why != "disabled"}


def plugin_of_command(cls: type) -> PluginInfo | None:
    """The plugin that provides a dialog command (None for pymetrix's own)."""
    from .commands.registry import COMMAND_IDS

    info = next((i for i in COMMAND_IDS.values() if i.cls is cls), None)
    if info is None or info.plugin == CORE.name:
        return None
    return _PLUGINS.get(info.plugin) or PluginInfo(info.plugin, info.plugin_version)


def plugins() -> list[PluginInfo]:
    load_all()
    return list(_PLUGINS.values())


def plugin_table() -> Any:
    """Installed plugins, their status and what they provide, as a ``Table``."""
    from .output import Table

    rows = []
    for info in plugins():
        provided = sum(e.plugin.name == info.name for reg in _REGISTRIES.values() for e in reg._entries.values())
        if is_disabled(info.name):
            status = "disabled"
        elif _PROBLEMS.get(info.name):
            status = _PROBLEMS[info.name]
        elif not info.supported:
            status = f"unsupported API {info.api}"
        else:
            status = info.problem or "available"
        requires = ", ".join(f"{k} {v}".strip() for k, v in info.requires.items())
        rows.append([info.name, info.version, info.api, info.license, requires, provided, status])
    rows += [[name, "", "", "", "", 0, problem] for name, problem in _PROBLEMS.items() if name not in _PLUGINS]
    return Table(["plugin", "version", "API", "licence", "requires", "extensions", "status"], rows, title="Plugins")
