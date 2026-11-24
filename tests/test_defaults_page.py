"""docs/defaults.md is the in-app help page ``defaults`` (scripts/make_defaults_page.py writes it from the page):
the two must not drift apart."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_docs_page_is_the_help_page():
    spec = importlib.util.spec_from_file_location("make_defaults_page", ROOT / "scripts" / "make_defaults_page.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    written = (ROOT / "docs" / "defaults.md").read_text(encoding="utf-8")
    assert written == module.defaults_markdown(), "docs/defaults.md is out of date: run python scripts/make_defaults_page.py"
    from pymetrix.help import topic

    assert topic("defaults").text in written
