"""Every help page's example runs as written, so the examples cannot drift from the code."""

from __future__ import annotations

import matplotlib
import pytest

from pymetrix import help as hp
from pymetrix.plugins import registry

matplotlib.use("Agg")
hp._load_all()

PAGES = {f"{g}/{e.qualified}": e for g in hp.HELP_GROUPS for e in registry(g).entries()
         if getattr(e.obj, "example", "") and registry(g).of(e.obj) is e}


def test_examples_exist():
    assert len(PAGES) >= 2


@pytest.mark.parametrize("key", sorted(PAGES))
def test_example_runs(key):
    page = hp.help_page(PAGES[key])
    assert f"```python\n{page.example}\n```" in page.to_markdown()
    try:
        exec(compile(page.example, f"<example {key}>", "exec"), {"__name__": "__example__"})
    except ImportError as exc:  # an example of a wrapper needs its library: skipped where the extra is not installed
        if "pip install pymetrix[" not in str(exc):
            raise
        pytest.skip(str(exc))


def test_dialogs_show_their_menu():
    page = hp.help_page(next(e for e in registry("commands").entries() if e.name.lower() == "fitequation"))
    assert page.menu.startswith("Estimate → ") and f"**{page.menu}**" in page.to_markdown()


RESULTS = {f"{g}/{e.qualified}": e for g in hp.HELP_GROUPS for e in registry(g).entries()
           if getattr(e.obj, "results", ()) and registry(g).of(e.obj) is e}


@pytest.mark.parametrize("key", sorted(RESULTS))
def test_results_tables_render(key):
    page = hp.help_page(RESULTS[key])
    assert all(len(row) == 2 and all(isinstance(x, str) and x for x in row) for row in page.results)
    assert "## Results" in page.to_markdown() and all(item in page.to_markdown() for item, _ in page.results)
