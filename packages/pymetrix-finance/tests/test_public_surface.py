"""The finance plugin imports from pymetrix only through the extension API and the stable modules, and no private
name: so pymetrix's internals can change without breaking it."""

import ast
from pathlib import Path

from pymetrix.extension import STABLE

SRC = Path(__file__).resolve().parents[1] / "src" / "pymetrix_finance"


def test_only_the_public_surface():
    bad = []
    for f in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 and node.module.split(".")[0] == "pymetrix":
                if node.module not in STABLE:
                    bad.append(f"{f.name}: from {node.module}")
                bad += [f"{f.name}: {node.module}.{a.name}" for a in node.names if a.name.startswith("_")]
            elif isinstance(node, ast.Import):
                bad += [f"{f.name}: import {a.name}" for a in node.names if a.name.split(".")[0] == "pymetrix" and a.name not in STABLE]
    assert not bad, "imports outside the public surface:\n" + "\n".join(bad)
