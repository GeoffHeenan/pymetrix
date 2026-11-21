"""The README's quick start runs as written, and the project documents' links point at files that exist."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_quick_start_runs(capsys):
    pytest.importorskip("statsmodels")  # the Breusch-Godfrey test
    code = re.search(r"## Quick start\s+```python\n(.*?)```", README, re.S).group(1)
    exec(compile(code, "README quick start", "exec"), {"__name__": "__main__"})  # noqa: S102 - our own example
    out = capsys.readouterr().out
    assert "HAC standard errors" in out and "Breusch-Godfrey" in out


def test_links_exist():
    """Relative links in the project's own documents point at files that exist."""
    files = [ROOT / n for n in ("README.md", "GOVERNANCE.md", "CONTRIBUTING.md", "SECURITY.md")]
    files += sorted((ROOT / "docs").rglob("*.md"))
    missing = []
    for f in files:
        for target in re.findall(r"\]\(([^)\s]+)\)", f.read_text(encoding="utf-8")):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            if not (f.parent / target.split("#")[0]).exists():
                missing.append(f"{f.relative_to(ROOT)}: {target}")
    assert not missing
