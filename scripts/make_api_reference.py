"""Build the PyMetrix command and function reference (docs/reference) from the code itself.

    python scripts/make_api_reference.py              # into docs/reference
    python scripts/make_api_reference.py <folder>     # elsewhere

Writes reference.md (the readable reference) and six CSV files (commands, calls, settings, other, library, extension), from:
the dialog registry (menus, settings), the @api call registry (signatures, return types), the plugin registries
(methods, tests, filters, ...), the adapter specs (which library function each wrapper calls), the programs PyMetrix
runs outside Python (``PROGRAMS``), the extension API's public names (``pymetrix.extension``), and a scan of the source for calls into statsmodels, SciPy, linearmodels, arch
and scikit-learn, followed two steps through PyMetrix's own functions and methods. Installed plugins' commands are
included, so run it with the first-party plugins installed (as CI does). Rerun it when the API changes.
"""

from __future__ import annotations

import ast
import csv
import dataclasses
import importlib
import inspect
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO / "docs" / "reference"
sys.path.insert(0, str(REPO / "src"))

import pymetrix
import pymetrix.commands.builtin
from pymetrix import _api
from pymetrix.adapters.base import AdapterSpec
from pymetrix.commands import COMMANDS, form_spec
from pymetrix.commands.menus import shown
from pymetrix.plugins import _REGISTRIES, load_all

load_all()  # installed plugins' dialogs, calls and methods too (the finance plugin's, say)

LIBS = ("statsmodels", "scipy", "linearmodels", "arch", "sklearn")
SKIP = ("scipy.linalg.",)  # linear-algebra building blocks (like numpy.linalg), not analytical functions
SETTINGS_GROUPS = ("methods", "variances", "systems", "models", "tests", "filters", "seasonal", "solvers")
# programs outside Python that PyMetrix runs (no adapter spec or import to find them by): class -> (call, "library")
PROGRAMS = {"X13": ("x13as (US Census Bureau X-13ARIMA-SEATS), run as a separate program on a generated .spc file", "x13as"),
            "TramoSeats": ("jdplus.tramoseats.base.r.TramoSeats.process (JDemetra+ 3.9, in Java through JPype)", "JDemetra+")}
GROUP_TITLES = {"methods": "Estimation methods", "variances": "Variance models", "systems": "System estimators",
                "models": "Model types", "tests": "Tests", "filters": "Filters", "seasonal": "Seasonal adjustment",
                "solvers": "Model solvers"}


def first_line(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    text = " ".join(doc.split("\n\n")[0].split())
    return text if len(text) <= 260 else text[:257].rstrip() + "…"


def sig(fn, drop_self=True) -> str:
    try:
        s = inspect.signature(fn)
    except (TypeError, ValueError):
        return ""
    params = [p for p in s.parameters.values() if not (drop_self and p.name in ("self", "cls"))]
    out = []
    for p in params:
        text = p.name if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) else ("*" if p.kind == p.VAR_POSITIONAL else "**") + p.name
        if p.default is not p.empty:
            d = repr(p.default)
            text += "=" + (d if len(d) <= 40 else d[:37] + "…")
        out.append(text)
    return ", ".join(out)


def returns(fn) -> str:
    try:
        r = inspect.signature(fn).return_annotation
    except (TypeError, ValueError):
        return ""
    if r is inspect.Signature.empty:
        return ""
    return r if isinstance(r, str) else getattr(r, "__name__", str(r))


# ------------------------------------------------------------------ source scan: calls into external libraries

def canonical(dotted: str) -> str:
    """The library's own name for an object (statsmodels.api.OLS -> statsmodels.regression.linear_model.OLS)."""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except Exception:  # noqa: BLE001, S112 - not a module at this length: try a shorter prefix
            continue
        try:
            for name in parts[cut:]:
                obj = getattr(obj, name)
        except AttributeError:
            return dotted
        mod, qual = getattr(obj, "__module__", None), getattr(obj, "__qualname__", None)
        if isinstance(mod, str) and isinstance(qual, str) and mod.split(".")[0] == parts[0] and "<" not in qual:
            return f"{mod}.{qual}"
        return dotted
    return dotted


def wanted(name: str) -> bool:
    return name.startswith(tuple(lib + "." for lib in LIBS)) and not name.startswith(SKIP)


class Scan(ast.NodeVisitor):
    """Calls into external libraries, and references to adapter specs, by enclosing pymetrix definition."""

    def __init__(self, module: str, aliases: dict[str, str]):
        self.module, self.aliases, self.stack = module, aliases, []
        self.calls: dict[str, set[str]] = defaultdict(set)
        self.specs: dict[str, set[str]] = defaultdict(set)
        self.inner: dict[str, set[str]] = defaultdict(set)  # pymetrix functions and methods it calls, as scan keys
        self.local: set[str] = set()  # this module's top-level function names

    def where(self) -> str:
        return f"{self.module}:{'.'.join(self.stack) or '<module>'}"

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        if len(self.stack) >= 2:  # nested helpers count towards their enclosing method or function
            self.generic_visit(node)
            return
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def dotted(self, node) -> str | None:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name) or node.id not in self.aliases:
            return None
        return ".".join([self.aliases[node.id], *reversed(parts)])

    def visit_Call(self, node):
        name = self.dotted(node.func)
        if name and wanted(name):
            self.calls[self.where()].add(name)
        elif name and name.startswith("pymetrix."):
            mod, _, fn = name.rpartition(".")
            self.inner[self.where()].add(f"{mod}:{fn}")
        elif isinstance(node.func, ast.Name) and node.func.id in self.local:
            self.inner[self.where()].add(f"{self.module}:{node.func.id}")
        elif (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
              and node.func.value.id == "self" and self.stack):
            self.inner[self.where()].add(f"{self.module}:{self.stack[0]}.{node.func.attr}")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        name = self.dotted(node)
        if name and name.startswith("pymetrix.adapters.") and name.count(".") == 3:
            self.specs[self.where()].add(name)
        self.generic_visit(node)


# pymetrix._lazy's stand-ins, read as the modules they stand for
LAZY = {"pymetrix._lazy.stats": "scipy.stats", "pymetrix._lazy.signal": "scipy.signal"}


def aliases_of(tree: ast.AST, module: str) -> dict[str, str]:
    out: dict[str, str] = {}
    package = module if module.endswith("__init__") else module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                up = package.split(".")
                up = up[: len(up) - node.level + 1]
                base = ".".join([*up, base]) if base else ".".join(up)
            for a in node.names:
                out[a.asname or a.name] = f"{base}.{a.name}"
    return {k: LAZY.get(v, v) for k, v in out.items()}


def scan_sources():
    calls, specs = {}, {}
    for path in sorted((REPO / "src" / "pymetrix").rglob("*.py")):
        module = ".".join(path.relative_to(REPO / "src").with_suffix("").parts)
        if module.startswith(("pymetrix.adapters", "pymetrix.ui")):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        s = Scan(module.removesuffix(".__init__"), aliases_of(tree, module))
        s.local = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        s.visit(tree)
        calls.update(s.calls)
        specs.update(s.specs)
        INNER.update(s.inner)
    return calls, specs


INNER: dict[str, set[str]] = {}
EXTERNAL, SPEC_REFS = scan_sources()


def reach(keys: list[str], depth: int = 2) -> list[str]:
    """The keys, and the pymetrix functions and methods they call, ``depth`` steps on."""
    seen, frontier = list(keys), list(keys)
    for _ in range(depth):
        frontier = [k for f in frontier for k in INNER.get(f, ()) if k not in seen]
        seen += frontier
    return seen
CANON = {}


def canon(name: str) -> str:
    """The library's own public name where it is one (statsmodels.api.OLS -> statsmodels.regression.linear_model.OLS);
    the name as PyMetrix writes it where the library's is private (scipy.optimize._minimize) or names something else
    (stats.norm.cdf is a method of the generic continuous distribution)."""
    if name not in CANON:
        full = canonical(name)
        private = any(part.startswith("_") and not part.startswith("__") for part in full.split("."))
        CANON[name] = full if not private and full.rpartition(".")[2] == name.rpartition(".")[2] else name
    return CANON[name]


def spec_object(dotted: str):
    mod, _, attr = dotted.rpartition(".")
    try:
        obj = getattr(importlib.import_module(mod), attr)
    except Exception:  # noqa: BLE001
        return None
    return obj if isinstance(obj, AdapterSpec) else None


def where_of(obj, method: str | None = None) -> list[str]:
    """The scan keys for a class (all its methods), a method or a function."""
    mod = getattr(obj, "__module__", "")
    qual = getattr(obj, "__qualname__", "")
    if method:
        return [f"{mod}:{qual}.{method}"]
    if inspect.isclass(obj):
        keys = [k for k in EXTERNAL.keys() | SPEC_REFS.keys() if k.startswith(f"{mod}:{qual}.") or k == f"{mod}:{qual}"]
        for base in obj.__mro__[1:]:  # inherited methods count too
            if base.__module__.startswith("pymetrix"):
                keys += [k for k in EXTERNAL.keys() | SPEC_REFS.keys() if k.startswith(f"{base.__module__}:{base.__qualname__}.")]
        return keys
    return [f"{mod}:{qual}"]


def implementation(obj, method: str | None = None) -> tuple[str, str, str]:
    """(kind, library function, library) for a class, method or function: a wrapper of one library call, or PyMetrix's
    own code, with the library functions it uses."""
    if method is None and getattr(obj, "__name__", "") in PROGRAMS and str(getattr(obj, "__module__", "")).startswith("pymetrix"):
        return ("wrapper", *PROGRAMS[obj.__name__])
    spec = getattr(obj, "adapter_spec", None) if method is None else None
    wraps = getattr(obj, "wraps", None) if method is None else None
    own = where_of(obj, method)
    keys = own if any(SPEC_REFS.get(k) for k in own) else reach(own)
    refs = {s for k in keys for s in SPEC_REFS.get(k, ())}
    spec_objs = [spec] if isinstance(spec, AdapterSpec) else [o for o in (spec_object(r) for r in sorted(refs)) if o]
    if spec_objs:
        return "wrapper", "; ".join(dict.fromkeys(s.wraps for s in spec_objs)), "; ".join(dict.fromkeys(s.backend for s in spec_objs))
    if isinstance(wraps, str) and wraps:
        return "wrapper", wraps, wraps.split(".")[0]
    if method is not None and inspect.isclass(obj):  # a method of a wrapper class: the class's library
        whole = implementation(obj)
        if whole[0] == "wrapper":
            return whole
    used = sorted({canon(c) for k in own for c in EXTERNAL.get(k, ())})
    if used:
        return "pymetrix", "; ".join(used), "; ".join(sorted({u.split(".")[0] for u in used}))
    return "pymetrix", "", ""


# ------------------------------------------------------------------ the four tables

def fields_of(cls) -> str:
    if dataclasses.is_dataclass(cls):
        out = []
        for f in dataclasses.fields(cls):
            if f.name.startswith("_") or not f.init:
                continue
            if f.default is not dataclasses.MISSING:
                d = repr(f.default)
                out.append(f"{f.name}={d if len(d) <= 30 else d[:27] + '…'}")
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                out.append(f"{f.name}=…")
            else:
                out.append(f.name)
        return ", ".join(out)
    return sig(cls, drop_self=True)


def call_ids_of(cls, spec_fields) -> list[str]:
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return []
    ids = re.findall(r'make_call\(\s*"([a-z_][\w.:-]*)"', src)
    for pattern in re.findall(r'make_call\(\s*f"([a-z_][\w.:{}-]*)"', src):  # f"pymetrix:dataset.{verb}"
        rx = re.compile(re.sub(r"\\{[^}]*\\}", r"(\\w+)", re.escape(pattern)) + "$")
        ids += [c for c in _api.API_CALLS if (m := rx.match(c)) and all(f'"{g}"' in src for g in m.groups())]
    objects = {f.name: [c.strip() for c in f.source.split(",")] for f in spec_fields
               if getattr(f, "kind", "") == "object" and isinstance(getattr(f, "source", None), str)}
    every_object = [c for classes in objects.values() for c in classes]
    for owner, method in re.findall(r'method_call\(\s*([^,]+?)\s*,\s*"(\w+)"', src):
        named = re.fullmatch(r"[A-Za-z_][\w.]*", owner)
        field = re.search(r"self\.(\w+)", owner)
        if named and owner.rpartition(".")[2][:1].isupper():
            classes = [owner.rpartition(".")[2]]
        elif field and field.group(1) in objects:  # type(session[self.fit]): the classes the field accepts
            classes = objects[field.group(1)]
        else:  # a class chosen in code: any the dialog's objects could be
            classes = every_object
        ids += [f"pymetrix:{_api._snake(c)}.{method}" for c in classes]
    for method in re.findall(r'_structural_call\(\s*session\s*,\s*self\s*,\s*"(\w+)"', src):  # the VAR dialogs' identification
        ids += [f"pymetrix:{_api._snake(c)}.{method}" for c in every_object]
        if "Heteroskedasticity" in src:
            ids += ["pymetrix:fitted_var.heteroskedasticity_svar", f"pymetrix:heteroskedasticity_svar.{method}"]
    for cid in re.findall(r'call_id[^=\n]*=\s*"([a-z_][\w.:-]*)"', src):
        ids.append(cid)
    return [i for i in dict.fromkeys(ids) if i in _api.API_CALLS]


calls = _api.API_CALLS
command_rows, recorded_by = [], defaultdict(list)
for cid, info in sorted(COMMANDS.items(), key=lambda kv: (kv[1].menu or "~", kv[0])):
    cls = info.cls
    try:
        spec_fields = form_spec(cls)
    except Exception:  # noqa: BLE001
        spec_fields = []
    settings = ", ".join(f"{f.name}" + (f"={f.default!r}" if f.default not in (None, "", [], ()) else "") for f in spec_fields)
    sources = sorted({f.source for f in spec_fields if getattr(f, "source", None) in SETTINGS_GROUPS}
                     | {"tests" for f in spec_fields if getattr(f, "kind", "") == "test"})
    made = call_ids_of(cls, spec_fields)
    for m in made:
        recorded_by[m].append(" → ".join(shown(info.menu).split("/")) if info.menu else f"{cls.__name__} (button)")
    out = ", ".join(dict.fromkeys(returns(calls[m].function) for m in made if returns(calls[m].function)))
    kinds = [implementation(calls[m].owner or calls[m].function, calls[m].function.__name__ if calls[m].owner else None) for m in made]
    wrap = "; ".join(dict.fromkeys(k[1] for k in kinds if k[0] == "wrapper"))
    command_rows.append({
        "menu": " → ".join(shown(info.menu).split("/")) if info.menu else "(button or panel)",
        "command": cls.__name__, "id": cid, "plugin": info.plugin or "pymetrix", "settings": settings,
        "choices_from": ", ".join(sources),
        "api_calls": ", ".join(m.split(":", 1)[1] for m in made) or ("figure methods, chained" if "figure" in cls.__module__
                                                                    else "(no single API call: the dialog's own step)"),
        "returns": out,
        "wraps": wrap or ("depends on the method chosen" if sources else ""), "summary": first_line(cls)})

call_rows = []
for cid, spec in sorted(calls.items()):
    owner, fn = spec.owner, spec.function
    name = f"{owner.__name__}.{fn.__name__}" if owner is not None else fn.__name__
    kind, lib_fn, lib = implementation(owner if owner is not None else fn, fn.__name__ if owner is not None else None)
    takes_settings = any(p in ("method", "test", "filter", "solver", "variance", "errors") for p in spec.signature.parameters)
    call_rows.append({
        "call": name, "id": cid, "plugin": spec.plugin, "kind": spec.kind, "signature": f"{name}({sig(fn)})",
        "returns": returns(fn), "ui": "; ".join(dict.fromkeys(recorded_by.get(cid, []))) or "",
        "implementation": kind if lib_fn or not takes_settings else "pymetrix (dispatches to the method chosen)",
        "library_function": lib_fn, "library": lib, "summary": first_line(fn)})

ui_groups = defaultdict(list)
for row in command_rows:
    for g in filter(None, row["choices_from"].split(", ")):
        ui_groups[g].append(row["menu"])
by_owner = defaultdict(list)  # a model class's own dialogs: those recording its calls (VAR.fit: Estimate → VAR…)
for cid, spec in calls.items():
    if spec.owner is not None:
        by_owner[spec.owner].extend(recorded_by.get(cid, []))
settings_rows = []
for group in SETTINGS_GROUPS:
    seen = set()
    for entry in sorted(_REGISTRIES[group]._entries.values(), key=lambda e: e.label.lower()):
        cls = entry.obj
        if cls in seen:  # one class registered under two names (an alias)
            continue
        seen.add(cls)
        kind, lib_fn, lib = implementation(cls)
        settings_rows.append({
            "group": GROUP_TITLES[group], "name": cls.__name__, "label": entry.label, "plugin": entry.plugin.name,
            "settings": f"{cls.__name__}({fields_of(cls)})", "ui": "; ".join(dict.fromkeys([*ui_groups.get(group, []), *by_owner.get(cls, [])])),
            "implementation": kind, "library_function": lib_fn, "library": lib, "summary": first_line(cls)})

covered = {r["name"] for r in settings_rows} | {r["call"].split(".")[0] for r in call_rows} | {r["call"] for r in call_rows}
other_rows = []
for name in sorted(pymetrix.__all__, key=str.lower):
    obj = getattr(pymetrix, name, None)
    if obj is None or name in covered or inspect.ismodule(obj):
        continue
    kind = "class" if inspect.isclass(obj) else "function" if callable(obj) else "value"
    impl = implementation(obj) if kind != "value" else ("", "", "")
    other_rows.append({"name": name, "kind": kind, "signature": f"{name}({sig(obj)})" if kind != "value" else name,
                       "implementation": impl[0], "library_function": impl[1], "library": impl[2],
                       "summary": first_line(obj) if kind != "value" else ""})

# the extension API: what a plugin imports from pymetrix.extension, and where each name lives
import pymetrix.extension as extension
from pymetrix.plugins import API

extension_rows = []
for name in sorted(extension.__all__, key=str.lower):
    if name == "STABLE":
        continue
    obj = getattr(extension, name)
    module, attr = extension._PUBLIC.get(name, (extension.__name__, name))
    kind = "class" if inspect.isclass(obj) else "function" if callable(obj) else "value"
    extension_rows.append({"name": name, "kind": kind, "signature": f"{name}({sig(obj)})" if kind != "value" else name,
                           "defined_in": f"{module}.{attr}", "summary": first_line(obj) if kind != "value" else ""})

# the library side: each external function and the pymetrix code that calls it
public_of = {}
for row in call_rows:
    spec = calls[row["id"]]
    public_of[f"{spec.function.__module__}:{spec.function.__qualname__}"] = row["call"]
for group in SETTINGS_GROUPS:
    for entry in _REGISTRIES[group]._entries.values():
        public_of[f"{entry.obj.__module__}:{entry.obj.__qualname__}"] = f"{entry.obj.__name__} ({GROUP_TITLES[group].lower()})"
for name in pymetrix.__all__:
    obj = getattr(pymetrix, name, None)
    if obj is not None and hasattr(obj, "__qualname__") and hasattr(obj, "__module__"):
        public_of.setdefault(f"{obj.__module__}:{obj.__qualname__}", name)


def public_name(key: str) -> str:
    """The public pymetrix name a scan key belongs to: the method or function itself, else its class."""
    if key in public_of:
        return public_of[key]
    mod, _, qual = key.partition(":")
    head = qual.split(".")[0]
    if f"{mod}:{head}" in public_of:
        return f"{public_of[f'{mod}:{head}']}" + (f" ({qual.split('.', 1)[1]})" if "." in qual else "")
    return f"{mod.removeprefix('pymetrix.')}.{qual} (internal)"


library = defaultdict(set)
lib_of: dict[str, str] = {}
for key, names in EXTERNAL.items():
    for n in names:
        library[canon(n)].add(public_name(key))
for key, refs in SPEC_REFS.items():
    for r in refs:
        s = spec_object(r)
        if s is not None:
            library[s.wraps].add(public_name(key) + " [adapter]")
for row in settings_rows:  # classes that declare what they wrap
    if row["implementation"] == "wrapper":
        for w, lib in zip(row["library_function"].split("; "), row["library"].split("; ") * 9):
            library[w].add(f"{row['name']} ({row['group'].lower()})")
            lib_of[w] = lib
for fn, users in library.items():  # one entry per caller: "[adapter]" only where the plain name is not listed already
    library[fn] = {u for u in users if not (u.endswith(" [adapter]") and u.removesuffix(" [adapter]") in users)}
def users_text(users: set[str]) -> str:
    public = sorted(u for u in users if "(internal)" not in u)
    inner = len(users) - len(public)
    return "; ".join(public) + (f"{'; ' if public else ''}and {inner} internal helper{'s' if inner > 1 else ''}" if inner else "")


library_rows = [{"library": lib_of.get(fn, fn.split(".")[0]), "function": fn, "used_by": users_text(users),
                 "internal_helpers": "; ".join(sorted(u.removesuffix(" (internal)") for u in users if "(internal)" in u))}
                for fn, users in sorted(library.items(), key=lambda kv: kv[0].lower()) if not fn.startswith(SKIP)]

# ------------------------------------------------------------------ write

OUT.mkdir(parents=True, exist_ok=True)
for name, rows in (("commands", command_rows), ("calls", call_rows), ("settings", settings_rows), ("other", other_rows),
                   ("library", library_rows), ("extension", extension_rows)):
    with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def esc(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def impl_text(row) -> str:
    if row.get("implementation") == "wrapper":
        return f"**Wrapper** of `{row['library_function']}` ({row['library']})"
    if row.get("library_function"):
        return f"PyMetrix; uses {', '.join(f'`{x}`' for x in row['library_function'].split('; '))}"
    return esc(row.get("implementation") or "PyMetrix")


md = ["# PyMetrix command and function reference\n",
      (f"Generated from the code at version {pymetrix.__version__}: {len(command_rows)} dialog commands, {len(call_rows)} "
      f"API calls, {len(settings_rows)} settings classes, {len(other_rows)} other public names and {len(library_rows)} "
      "external library functions (`python scripts/make_api_reference.py`).\n"),
      ("**How to read it.** *UI* means a dialog in the desktop app records it; everything here can also be used "
      "headlessly (scripts, notebooks, the MCP server), since every dialog step is recorded as the Python call shown. "
      "*Wrapper* means PyMetrix passes the work to one library function, mapping its settings to that function's "
      "arguments (the adapter); *PyMetrix* means PyMetrix's own code, with the library functions it calls listed. "
      "The library functions are found by scanning the source, following calls up to two steps through PyMetrix's "
      "own functions and methods.\n")]
md += ["## 1. Dialog commands (UI)\n",
       ("Each dialog runs the API call(s) shown, so the same step can be written in Python. Methods, tests and filters "
       "offered as choices are in section 3.\n"),
       "| Menu | Command | Settings | API call | Returns | Library | Summary |", "|---|---|---|---|---|---|---|"]
md += [f"| {esc(r['menu'])} | `{r['command']}` | {esc(r['settings'])} | {esc(r['api_calls'])} | {esc(r['returns'])} | "
       f"{esc(r['wraps'])} | {esc(r['summary'])} |" for r in command_rows]
md += ["\n## 2. API functions and methods (headless; UI where a dialog records them)\n",
       "| Call | Signature | Returns | UI (dialog) | Implementation | Summary |", "|---|---|---|---|---|---|"]
md += [f"| `{r['call']}` | `{esc(r['signature'])}` | {esc(r['returns'])} | {esc(r['ui']) or 'script only'} | "
       f"{impl_text(r)} | {esc(r['summary'])} |" for r in call_rows]
md += ["\n## 3. Methods, tests, filters and solvers (settings classes)\n",
       "Passed to a call (`Equation(...).fit(data, method=GMM(...))`) or chosen in a dialog.\n"]
for group in SETTINGS_GROUPS:
    rows = [r for r in settings_rows if r["group"] == GROUP_TITLES[group]]
    if not rows:
        continue
    md += [f"### {GROUP_TITLES[group]}\n", "| Name | Settings | UI (dialogs) | Implementation | Summary |", "|---|---|---|---|---|"]
    md += [f"| `{r['name']}` | `{esc(r['settings'])}` | {esc(r['ui']) or 'script only'} | {impl_text(r)} | {esc(r['summary'])} |" for r in rows]
    md.append("")
md += ["## 4. Other public names (`from pymetrix import *`)\n", "| Name | Kind | Signature | Implementation | Summary |",
       "|---|---|---|---|---|"]
md += [f"| `{r['name']}` | {r['kind']} | `{esc(r['signature'])}` | {impl_text(r) if r['kind'] != 'value' else ''} | "
       f"{esc(r['summary'])} |" for r in other_rows]
md += ["\n## 5. External library functions, and what calls them\n",
       ("Every function or class of statsmodels, SciPy, linearmodels, arch and scikit-learn that PyMetrix calls, by "
       "the library's own name, with the public PyMetrix commands, functions or classes calling it directly "
       "(`[adapter]`: through an adapter spec, which records the call in the result's provenance; internal helpers are "
       "counted, and named in library.csv). Linear algebra (NumPy's and SciPy's `linalg`: solve, inverse, "
       "decompositions) is left out: it is arithmetic PyMetrix's own estimators are built from, not analysis "
       "delegated to a library.\n")]
for lib in sorted({r["library"] for r in library_rows}):
    md += [f"### {lib}\n", "| Function | Called by (PyMetrix) |", "|---|---|"]
    md += [f"| `{r['function']}` | {esc(r['used_by'])} |" for r in library_rows if r["library"] == lib]
    md.append("")
md += [f"## 6. Extension API {API} (`pymetrix.extension`)\n",
       ("What a plugin imports from pymetrix besides the stable modules (`pymetrix.extension.STABLE`), under names "
       "that stay put as pymetrix changes (docs/design/plugins.md). *Defined in* is where each name lives now; a "
       "plugin never imports it from there.\n"),
       "| Name | Kind | Signature | Defined in | Summary |", "|---|---|---|---|---|"]
md += [f"| `{r['name']}` | {r['kind']} | `{esc(r['signature'])}` | `{r['defined_in']}` | {esc(r['summary'])} |"
       for r in extension_rows]
md.append("")
(OUT / "reference.md").write_text("\n".join(md), encoding="utf-8")
print(f"{len(command_rows)} commands, {len(call_rows)} calls, {len(settings_rows)} settings, {len(other_rows)} other, "
      f"{len(library_rows)} library functions -> {OUT}")
