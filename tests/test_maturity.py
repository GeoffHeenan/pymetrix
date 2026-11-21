"""Maturity tiers on help pages: a class's ``maturity`` is one of the three tiers defined in docs/features.md, reaches
its help page and is shown under the summary; pages without one show nothing."""

from dataclasses import fields, is_dataclass

import pytest

from pymetrix.help import HELP_GROUPS, MATURITY, _load_all, help_page, maturity_line
from pymetrix.plugins import registry

TIERS = ("stable", "beta", "experimental")


def _entries():
    _load_all()
    return [e for g in HELP_GROUPS for e in registry(g).entries()]


def test_the_tiers_are_the_three_words():
    assert tuple(MATURITY) == TIERS


def test_every_declared_tier_is_one_of_the_three_and_is_rendered():
    tagged = 0
    for entry in _entries():
        tier = getattr(entry.obj, "maturity", None)
        page = help_page(entry)
        if tier is None:
            assert page.maturity == "" and "Maturity:" not in page.to_markdown(), entry.qualified
            continue
        tagged += 1
        assert tier in TIERS, f"{entry.group}/{entry.qualified}: maturity {tier!r}"
        assert not is_dataclass(entry.obj) or "maturity" not in {f.name for f in fields(entry.obj)}, "declare it as a ClassVar"
        assert page.maturity == tier
        md = page.to_markdown()
        assert f"*Maturity: {tier}: " in md
        if page.usage:  # one line under the summary, before the rest of the page
            assert md.index("Maturity:") < md.index(page.usage[:40])
    assert tagged >= 90  # methods, variances, tests, filters and the dialogs with examples


@pytest.mark.parametrize("group", ["methods", "variances", "tests", "filters"])
def test_every_built_in_page_in_these_groups_has_a_tier(group):
    _load_all()
    builtin = {"pymetrix", "pymetrix-linearmodels", "pymetrix-arch", "pymetrix-sklearn"}
    missing = [e.qualified for e in registry(group).entries() if e.plugin.name in builtin and not getattr(e.obj, "maturity", "")]
    assert not missing


def test_dialogs_with_examples_have_a_tier():
    missing = [e.qualified for e in _entries() if e.group == "commands" and getattr(e.obj, "example", "") and not getattr(e.obj, "maturity", "")]
    assert not missing


def test_unknown_tiers_show_nothing():
    assert maturity_line("beta") == "*Maturity: beta: validated in tests, little use so far.*"
    assert maturity_line("") == "" and maturity_line("alpha") == ""
