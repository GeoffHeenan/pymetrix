"""The dependency map behind docs/design/plugin-split.md.

    python scripts/dependency_map.py deps.json

For every module of pymetrix: its lines, the third-party packages it imports and the pymetrix modules it imports,
each marked lazy (only inside a function) or eager (at module level). Imports under ``TYPE_CHECKING`` are left out.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "pymetrix"


def modules() -> dict[str, Path]:
    found = {}
    for path in PACKAGE.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(ROOT / "src").with_suffix("")
        found[".".join(rel.parts[:-1] if rel.name == "__init__" else rel.parts)] = path
    return found


def imports(name: str, path: Path, known: dict[str, Path]) -> dict:
    """One module's lines and imports: {"lines", "ext": {package: lazy}, "int": {module: lazy}}."""
    source = path.read_text(encoding="utf-8")
    package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
    ext: dict[str, bool] = {}
    internal: dict[str, bool] = {}

    def record(module: str, lazy: bool) -> None:
        if module.startswith("pymetrix"):
            target = module
            while target and target not in known:  # "pymetrix.var.VAR" -> "pymetrix.var"
                target = target.rsplit(".", 1)[0] if "." in target else ""
            if target and target != name:
                internal[target] = internal.get(target, True) and lazy  # eager if any import is eager
        else:
            top = module.split(".")[0]
            ext[top] = ext.get(top, True) and lazy

    def visit(node: ast.AST, lazy: bool) -> None:
        for child in ast.iter_child_nodes(node):
            inner = lazy or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if isinstance(child, ast.If) and "TYPE_CHECKING" in ast.unparse(child.test):
                continue
            if isinstance(child, ast.Import):
                for alias in child.names:
                    record(alias.name, inner)
            elif isinstance(child, ast.ImportFrom):
                if child.level:  # relative: resolve against the module's package
                    base = package.split(".")[: len(package.split(".")) - (child.level - 1)]
                    target = ".".join(base + ([child.module] if child.module else []))
                    for alias in child.names if child.module is None else [None]:
                        record(f"{target}.{alias.name}" if alias else target, inner)
                elif child.module:
                    record(child.module, inner)
            visit(child, inner)

    visit(ast.parse(source), False)
    return {"lines": source.count("\n"), "ext": ext, "int": internal}


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    known = modules()
    result = {name: imports(name, path, known) for name, path in sorted(known.items())}
    Path(argv[0]).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{len(result)} modules, {sum(m['lines'] for m in result.values())} lines -> {argv[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
