"""``pymetrix-exercises``: check an exercise pack against its reference solutions, or start a new one.

    pymetrix-exercises check <pack-or-folder> [--exercise ID]
    pymetrix-exercises new <folder> [--name NAME] [--title TITLE]
    pymetrix-exercises list

``check`` validates the pack, then, for each exercise, loads its data into a fresh session, runs its ``solution.py``
(or, without one, the steps' python blocks in order) and every check. It prints each check and exits 0 when all pass,
1 when one fails or a solution raises, and 2 when the pack cannot be read or does not suit this pymetrix. A pack's own
CI runs it to catch drift. ``solution.py`` is code: it runs here because you named the pack; the app never runs it.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

from ..errors import NotFoundError, PyMetrixError

__all__ = ["main", "new_pack", "run_exercise"]


def _find(where: str) -> Any:
    """A pack folder, or the name of an installed pack."""
    from . import installed_packs, load_pack

    path = Path(where).expanduser()
    if path.exists():
        return load_pack(path)
    for pack in installed_packs():
        if pack.name == where:
            return pack
    raise NotFoundError(f"{where} is neither a folder nor the name of an installed exercise pack")


def run_exercise(exercise: Any) -> tuple[Any, list[Any], str]:
    """Complete ``exercise`` in a fresh session: its data (each dataset of its front matter's ``data``, under its name),
    then its solution (or its steps' code), run as typed code with the pack's folder as the working directory and
    ``PACK_DIR`` (that folder, a ``Path``) defined. Returns the session, the check results, and the error the solution
    raised ("" if none)."""
    import os
    from ..commands import Session
    from . import check, data_commands

    session = Session()
    for cmd in data_commands(exercise):
        session.run(cmd)
    if exercise.solution is not None:
        code, source = exercise.solution.read_text(encoding="utf-8"), str(exercise.solution)
    else:
        code, source = "\n\n".join(s.code for s in exercise.steps if s.code), "the steps' code"
    error = ""
    if code.strip():
        folder = exercise.pack.path if exercise.pack is not None else exercise.path.parent.parent
        before = os.getcwd()
        session.objects["PACK_DIR"] = folder  # the namespace takes the session's objects as the code starts
        try:
            os.chdir(folder)
            session.run_code(code, source="script")
        except Exception:  # noqa: BLE001 - reported, and the exercise fails
            error = f"{source} raised:\n{traceback.format_exc(limit=3)}"
        finally:
            os.chdir(before)
            session.namespace.pop("PACK_DIR", None)
            session.objects.pop("PACK_DIR", None)
    return session, check(exercise, session), error


def _check(args: argparse.Namespace) -> int:
    try:
        pack = _find(args.pack)
    except (PyMetrixError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    why = pack.requirement_problem
    if why:
        print(f"error: {why}", file=sys.stderr)
        return 2
    try:
        exercises = [pack.exercise(args.exercise)] if args.exercise else list(pack.exercises)
    except NotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{pack.title} ({pack.name} {pack.version}), {pack.path}")
    if pack.classification:
        print(f"Classification: {pack.classification}")
    failed = 0
    for ex in exercises:
        _session, results, error = run_exercise(ex)
        how = "solution.py" if ex.solution is not None else ("the steps' code" if any(s.code for s in ex.steps) else "no code")
        print(f"\n{ex.id}: {ex.title} ({how})")
        if error:
            print("  " + error.rstrip().replace("\n", "\n  "))
            failed += 1
        if not results:
            print("  no checks")
        for r in results:
            mark = "ok  " if r.passed else "FAIL"
            print(f"  {mark} step {r.step} {r.id}: {r.detail}" + ("" if r.passed or not r.message else f"\n       {r.message}"))
        failed += sum(not r.passed for r in results)
    total = sum(len(ex.checks) for ex in exercises)
    print(f"\n{'All' if not failed else 'Not all'} checks passed: {failed} problem{'s' * (failed != 1)} in {len(exercises)} "
          f"exercise{'s' * (len(exercises) != 1)} with {total} check{'s' * (total != 1)}.")
    return 1 if failed else 0


def _list(_args: argparse.Namespace) -> int:
    from . import installed_packs, problems

    packs = installed_packs()
    for pack in packs:
        print(f"{pack.name} {pack.version}: {pack.title} ({len(pack.exercises)} exercises, from {pack.source})"
              + (f" [{pack.classification}]" if pack.classification else ""))
    for label, why in problems().items():
        print(f"could not load {label}: {why}", file=sys.stderr)
    if not packs:
        print("No exercise packs are installed.")
    return 0


_PACK_TOML = '''[pack]
name = "{name}"
title = "{title}"
version = "0.1.0"
description = "Exercises in pymetrix: replace this with what the pack teaches."
requires_pymetrix = ">={version}"
audience = "students meeting regression for the first time"
# classification = "internal"    # optional free text, shown in the app and by pymetrix-exercises; never enforced
'''

_EXERCISE_MD = '''---
title: A first regression
level: beginner
minutes: 10
data: [sample_macro]
---

Consumption against output in a simulated quarterly economy (the example dataset `sample_macro`, loaded as
`sample_macro` when the exercise starts).

## Step 1: Fit the consumption function

Choose Estimate → Equation…, with Data: sample_macro, Formula: `c ~ y`, Method: OLS and Save as: fit.

```python
fit = Equation("c ~ y").fit(sample_macro)
```

> Hint: a constant is included unless the formula says `- 1`.

## Step 2: Read the marginal propensity to consume

The coefficient on `y`, in the Summary, is the marginal propensity to consume: about 0.75.

> Hint: the coefficient table is in the result's Summary view.
'''

_CHECKS_TOML = '''[[check]]
id = "sample"
step = 1
object = "fit"
value = "nobs"
expected = 100
message = "Use the whole sample: 2000Q1 to 2024Q4, 100 quarters."

[[check]]
id = "mpc"
step = 2
object = "fit"
value = "params.y"
expected = 0.749
abs_tol = 0.001
message = "Regress c on y, with a constant: the formula c ~ y."
'''

_SOLUTION = '''# The reference solution: pymetrix-exercises check runs it in a fresh session, after the exercise's data are
# loaded under their names, then runs the checks. It runs as typed code does in the Immediate window.
fit = Equation("c ~ y").fit(sample_macro)
'''

_README = '''# {title}

An exercise pack for pymetrix. Check it with

    pymetrix-exercises check .

Open it in the app with Help → Exercises… → Open folder…, or ship it in a plugin: in the plugin's pyproject.toml,

    [project.entry-points."pymetrix.exercises"]
    {name} = "your_package.exercises:pack_folder"

with ``pack_folder`` returning this folder, e.g. ``importlib.resources.files(__package__) / "{name}"``.
'''


def new_pack(folder: str | Path, name: str | None = None, title: str | None = None) -> Path:
    """Write a new pack in ``folder`` (which must not exist, or be empty) with one exercise that passes its checks."""
    from .._version import __version__

    root = Path(folder).expanduser()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"{root} is not empty")
    name = name or root.resolve().name.lower().replace(" ", "-")
    title = title or name.replace("-", " ").replace("_", " ").capitalize()
    ex = root / "exercises" / "01-first-regression"
    ex.mkdir(parents=True)
    (root / "data").mkdir()
    (root / "pack.toml").write_text(_PACK_TOML.format(name=name, title=title, version=__version__), encoding="utf-8")
    (root / "README.md").write_text(_README.format(name=name, title=title), encoding="utf-8")
    (ex / "exercise.md").write_text(_EXERCISE_MD, encoding="utf-8")
    (ex / "checks.toml").write_text(_CHECKS_TOML, encoding="utf-8")
    (ex / "solution.py").write_text(_SOLUTION, encoding="utf-8")
    return root


def _new(args: argparse.Namespace) -> int:
    from . import load_pack

    try:
        root = new_pack(args.folder, args.name, args.title)
        load_pack(root)
    except (OSError, PyMetrixError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"A new exercise pack in {root}. Check it with: pymetrix-exercises check {root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pymetrix-exercises", description="Check or start pymetrix exercise packs.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check", help="run each exercise's solution in a fresh session, then its checks")
    p.add_argument("pack", help="a pack's folder, or the name of an installed pack")
    p.add_argument("--exercise", help="one exercise: its folder name, slug or number")
    p.set_defaults(handler=_check)
    p = sub.add_parser("new", help="write a new pack with one exercise")
    p.add_argument("folder")
    p.add_argument("--name", help="the pack's name (default: the folder's)")
    p.add_argument("--title", help="the pack's title")
    p.set_defaults(handler=_new)
    p = sub.add_parser("list", help="the installed packs")
    p.set_defaults(handler=_list)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
