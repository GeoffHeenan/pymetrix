"""The words the dialogs show (pymetrix.wording): field labels from one table, choice names with the codes kept in
scripts, dialog titles from the menus, one Advanced settings section per dialog, and the glossary.

Every field of every registered dialog is checked, the core's and the bundled plugins' (the finance plugin's commands;
the guide, lab book and problem-set plugins add panels and courses, not dialogs), and so are the settings sub-forms of
every registered test, method, filter, variance model, solver and seasonal adjustment."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from pymetrix.commands import COMMANDS, Session, form_spec
from pymetrix.wording import (CHOICE_LABELS, COMMAND_LABELS, FIELD_LABELS, GLOSSARY, TITLE_WORDS_OK, TITLES, dialog_title,
                              fallback_label, field_label, menu_label)

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]

SETTINGS_GROUPS = ("tests", "methods", "filters", "variances", "solvers", "seasonal", "systems")

# Words that are cryptic as a label on their own: argument names the audit found shown raw ("Var", "Exog", "Reps"...)
ABBREVIATIONS = {"var", "exog", "endog", "dist", "garch", "reps", "tau", "rf", "kw", "kq", "ks", "mce", "lam", "v0", "gof",
                 "agg", "of", "g", "h", "x", "y", "low", "high", "how", "freq", "cov", "vcov", "sep", "cols", "fod", "ar",
                 "ma", "se", "dpi", "ha", "va", "deg", "iter", "obs", "loc", "params", "lamb", "asym", "arch", "xl", "pad",
                 "lags_", "max", "min", "n", "p", "q", "k", "a", "b", "fill", "rho", "sigma", "theta", "kappa", "delta",
                 "beta", "alpha"}
SHORT_WORDS = {"a", "an", "as", "at", "by", "for", "in", "of", "on", "or", "to", "the", "and", "per", "new", "old", "day",
               "log", "end", "set", "add", "use", "lag", "tab", "run", "own", "one", "two", "all", "who", "how", "why",
               "key", "row", "map", "fit", "cut", "box", "up", "if", "its", "too", "bar", "pi", "be", "is", "it", "no",
               "not", "out", "off", "via"}


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def registered():
    """Every command (core and plugins) and every registered settings class, with their fields."""
    from pymetrix.plugins import load_all, registry

    load_all()
    owners = [(info.cls, info.menu) for info in COMMANDS.values()]
    for group in SETTINGS_GROUPS:
        owners += [(entry.obj, None) for entry in registry(group).entries()]
    return [(cls, menu, form_spec(cls)) for cls, menu in owners]


def _human(label: str) -> bool:
    """A label read as words: no underscores or trailing colon, no word that is a cryptic abbreviation, short
    words only where they are plain English."""
    if label.rstrip().endswith(":") or not (label[:1].isupper() or label[:1].isdigit()):
        return False
    plain = re.sub(r"\([^)]*\)", "", label)  # symbols and units in brackets are fine: "Order (p, d, q)", "(k_Q)"
    if "_" in plain:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", plain)
    for word in words:
        low = word.lower()
        if low in ABBREVIATIONS and len(words) == 1:
            return False
        if len(word) < 4 and low not in SHORT_WORDS and not word.isupper() and low not in ABBREVIATIONS:
            return False
    return True


def test_every_field_label_is_a_display_name_or_reads_as_words(registered, capsys):
    """1,284 of 1,493 labels were raw argument names ("Var", "Exog", "Reps"): now each comes from the table (or the
    field's own Widget label), and the few that fall back to their name's words read as words."""
    total = mapped = 0
    raw = []
    for cls, _, specs in registered:
        for spec in specs:
            total += 1
            from_table = spec.name in COMMAND_LABELS.get(cls.__name__, {}) or spec.name in FIELD_LABELS
            mapped += from_table or spec.label != fallback_label(spec.name)  # or a Widget(label=...) of its own
            if spec.name in COMMAND_LABELS.get(cls.__name__, {}):  # a command's own override comes first
                assert spec.label == field_label(cls.__name__, spec.name)
            if not _human(spec.label):
                raw.append(f"{cls.__name__}.{spec.name}: {spec.label!r}")
            if len(spec.name) > 1 and spec.name.lower() in ABBREVIATIONS:
                assert spec.label != spec.name.replace("_", " ").title(), f"{cls.__name__}.{spec.name} shown raw"
    assert raw == []
    with capsys.disabled():
        print(f"\nfield labels: {mapped} of {total} from the display-name table or the field itself ({mapped / total:.0%}), "
              f"{total - mapped} from their argument's words; table: {len(FIELD_LABELS)} names, "
              f"{sum(len(v) for v in COMMAND_LABELS.values())} per-command overrides in {len(COMMAND_LABELS)} commands")
    assert mapped / total > 0.98


def test_the_audit_s_cryptic_labels_are_spelt_out(registered):
    by = {cls.__name__: {s.name: s.label for s in specs} for cls, _, specs in registered}
    assert by["ImpulseResponse"]["var"] == "Estimated VAR"
    assert by["FitVAR"]["exog"] == "Exogenous series"
    assert by["FitEquation"]["dist"] == "Error distribution"
    assert by["ImportFRBUS"]["mce"] == "Model-consistent expectations"
    assert by["FitMIDAS"]["low"] == "Low-frequency data" and by["FitMIDAS"]["high"] == "High-frequency data"
    assert by["FitCointegratingRegression"]["y"] == "Dependent series (levels)"
    assert by["PropensityWeighting"]["reps"] == "Bootstrap replications"
    assert by["ImpulseResponse"]["periods"] == "Horizon"  # one word for steps ahead
    assert by["FitVAR"]["variables"] == "Series" and by["MakeRound"]["variables"] == "Variables"  # columns vs model variables
    assert by["LoadData"]["path"] == "Data file"


def test_the_finance_plugins_labels_and_choices_are_spelt_out(registered):
    """The finance plugin's dialogs, checked only where it is installed (the installed-wheel job has the core alone)."""
    by = {cls.__name__: {s.name: s for s in specs} for cls, _, specs in registered}
    if "MeasurePerformance" not in by:
        pytest.skip("the finance plugin is not installed")
    assert by["MeasurePerformance"]["rf"].label == "Risk-free rate"
    assert by["FitCopula"]["garch"].label == "GARCH filter first"
    assert by["FitCopula"]["family"].display("clayton180") == "Clayton, rotated 180° (survival)"


def test_labels_are_unique_within_each_dialog(registered):
    for cls, _, specs in registered:
        labels = [s.label for s in specs]
        assert len(labels) == len(set(labels)), (cls.__name__, sorted(x for x in labels if labels.count(x) > 1))


def test_choices_show_names_and_keep_their_codes(registered):
    """Raw codes (fmols/ccr/dols, drop/raise...) show as names, in sentence case; the stored value is the code."""
    unnamed = []
    for cls, _, specs in registered:
        for spec in specs:
            if spec.kind != "choice" or not spec.choices:
                continue
            for choice in spec.choices:
                shown = spec.display(choice)
                if isinstance(choice, str) and choice and shown[:1].islower() and not shown.startswith(("pymetrix", "t-")):
                    unnamed.append(f"{cls.__name__}.{spec.name}={choice!r}: {shown!r}")
                if isinstance(choice, str) and "_" in choice and shown.lower() == choice.lower():
                    unnamed.append(f"{cls.__name__}.{spec.name}={choice!r}: {shown!r}")
    assert unnamed == []
    by = {cls.__name__: {s.name: s for s in specs} for cls, _, specs in registered}
    method = by["FitCointegratingRegression"]["method"]
    assert [method.display(c) for c in method.choices] == ["FM-OLS (fully modified)", "CCR (canonical)", "DOLS (dynamic)"]
    missing = by["FitEquation"]["missing"]
    assert [missing.display(c) for c in missing.choices] == ["Drop rows with missing values", "Stop with an error"]
    from pymetrix.wording import choice_label

    assert choice_label("Variable", "adjustment", "nsa") == "not seasonally adjusted"


def test_a_choice_picked_by_its_name_records_its_code(app, ts):
    """The dialog shows "DOLS (dynamic)"; the step, and so the script, says method='dols', as before."""
    from pymetrix.commands import FitEquation
    from pymetrix.ui.qt import command_dialog

    FitCointegratingRegression = COMMANDS["FitCointegratingRegression"].cls

    s = Session()
    s.objects["data"] = ts
    dialog = command_dialog(FitCointegratingRegression, s)
    box = dialog.editors["method"]
    buttons = {b.text(): b for b in box.group.buttons()}
    buttons["DOLS (dynamic)"].setChecked(True)
    assert box.get() == "dols" and "Python: method='dols'" in buttons["DOLS (dynamic)"].toolTip()
    dialog.editors["y"].set("yd")
    dialog.editors["x"].set("c")
    picked = FitCointegratingRegression(**{k: e.get() for k, e in dialog.editors.items()})
    assert picked == FitCointegratingRegression(y="yd", x="c", method="dols")
    assert picked.to_code() == FitCointegratingRegression(y="yd", x="c", method="dols").to_code()
    assert "method='dols'" in picked.to_code()
    eq = command_dialog(FitEquation, s)
    eq.editors["missing"].set("raise")
    assert eq.editors["missing"].get() == "raise"
    combo = eq.editors["cov"].widget
    combo.setCurrentIndex(combo.findText("HAC (Newey-West)"))
    assert eq.editors["cov"].get() == "hac"
    for d in (dialog, eq):
        d.close()


def test_tooltips_give_the_python_argument(app, ts):
    from pymetrix.commands import FitVAR
    from pymetrix.ui.qt import command_dialog

    s = Session()
    s.objects["data"] = ts
    dialog = command_dialog(FitVAR, s)
    ed = dialog.editors["exog"]
    assert ed.widget.toolTip().endswith("Python: exog=")
    labels = [w for w in dialog.findChildren(QtWidgets.QLabel) if w.text().replace("&", "") == "Exogenous series"]  # the row's label too (with its access key)
    assert labels and labels[0].toolTip().endswith("Python: exog=")
    dialog.close()


@pytest.fixture(scope="module")
def every_dialog(app, registered):
    """Every registered command's dialog, opened on the sample economy, with its title and Advanced section."""
    from pymetrix import Dataset
    from pymetrix.ui.qt import command_dialog

    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    out = {}
    for info in COMMANDS.values():
        dialog = command_dialog(info.cls, s)
        tabs = dialog.findChildren(QtWidgets.QTabWidget)
        out[info.cls.__name__] = {
            "menu": info.menu, "title": dialog.windowTitle(),
            "advanced_tabs": [t.tabText(i) for t in tabs for i in range(t.count()) if "advanced" in t.tabText(i).lower()],
            "toggles": [b for b in dialog.findChildren(QtWidgets.QToolButton) if "advanced" in b.text().replace("&", "").lower()],
            "dialog": dialog}
    yield out
    for rec in out.values():
        rec["dialog"].close()
        rec["dialog"].deleteLater()


def test_every_dialog_has_at_most_one_advanced_mechanism(every_dialog, registered):
    """There were two: an Advanced tab and a sub-form's "▸ Advanced settings" expander on Main. Now one collapsed
    section at the bottom of the form holds both kinds of advanced setting."""
    has_advanced = {cls.__name__ for cls, menu, specs in registered if any(s.advanced for s in specs) and cls.__name__ in COMMANDS}
    for name, rec in every_dialog.items():
        assert rec["advanced_tabs"] == [], name
        assert len(rec["toggles"]) <= 1, name
        if name in has_advanced:
            assert rec["toggles"] and rec["dialog"].advanced_toggle is rec["toggles"][0], name


def test_sub_form_advanced_settings_join_the_one_section(app, ts):
    """Each method of Equation and each test of Run test: their advanced settings go in the dialog's section."""
    from pymetrix.commands import FitEquation, RunTest
    from pymetrix.ui.qt import command_dialog

    s = Session()
    s.objects["data"] = ts
    for cls, field, sub in ((FitEquation, "method", "method_settings"), (RunTest, "test", "options")):
        dialog = command_dialog(cls, s)
        combo = dialog.editors[field].widget
        seen_advanced = 0
        for i in range(combo.count()):
            combo.setCurrentIndex(i)
            toggles = [b for b in dialog.findChildren(QtWidgets.QToolButton) if "advanced" in b.text().replace("&", "").lower()]
            assert len(toggles) <= 1
            part = dialog.editors[sub]
            if part.advanced_form.rowCount():
                seen_advanced += 1
                assert dialog.advanced_toggle is not None
                assert not part.advanced_part.isVisibleTo(dialog)  # collapsed until opened
                dialog.show_advanced(True)
                assert part.advanced_part.isVisibleTo(dialog)
                dialog.show_advanced(False)
        assert seen_advanced, cls.__name__
        dialog.close()


def test_a_reopened_step_with_advanced_settings_opens_the_section(app, ts):
    from pymetrix.commands import FitEquation
    from pymetrix.ui.qt import command_dialog

    s = Session()
    s.objects["data"] = ts
    plain = command_dialog(FitEquation, s, initial=FitEquation("y ~ x"))
    assert not plain.advanced_toggle.isChecked()
    weighted = command_dialog(FitEquation, s, initial=FitEquation("y ~ x", missing="raise"))
    assert weighted.advanced_toggle.isChecked() and weighted.row_visible("missing")
    for d in (plain, weighted):
        d.close()


def test_dialog_titles_are_full_and_descriptive(every_dialog):
    """Titles come from the menu item, less its "…", unless the table says more; none is cut at a "/" ("US XML file")
    and none is a lone word but the name of a method."""
    for name, rec in every_dialog.items():
        title, menu = rec["title"], rec["menu"]
        cls = COMMANDS[name].cls
        assert title == dialog_title(cls, menu)
        if menu and name not in TITLES and not getattr(cls, "dialog_title", ""):
            assert title == menu_label(menu).rstrip(".… ")  # the whole label, slashes and all
        words = title.split()
        assert len(words) >= 2 or title in TITLE_WORDS_OK, (name, title)
        assert not title.endswith(("…", "...", "/")) and title[:1].isupper() or title.startswith("matplotlib"), (name, title)
    assert every_dialog["ImportFRBUS"]["title"] == "Import FRB/US XML file"
    assert every_dialog["CheckModel"]["title"] == "Check model" and every_dialog["SolveModel"]["title"] == "Solve model"
    assert every_dialog["DeriveEach"]["title"] == "Derived series for each selected series"
    assert every_dialog["Calibrate"]["title"] == "Calibrate model"


def test_menu_labels_keep_slashes():
    assert menu_label("Model/Import FRB/US XML file...") == "Import FRB/US XML file..."
    assert menu_label("Estimate/Equation...") == "Equation..."
    from pymetrix.plugins import registry

    assert registry("commands").get("importfrbus").label == "Import FRB/US XML file"


def test_choice_tables_have_no_raw_codes_left_in_their_names():
    for name, table in CHOICE_LABELS.items():
        for code, words in table.items():
            assert words and words != code or not isinstance(code, str) or "_" not in code, (name, code, words)
            assert not words.endswith(":") and not (isinstance(code, str) and "_" in code and words.lower() == code.lower())


def test_the_glossary_is_in_the_help_and_the_guide():
    from pymetrix.help import topic, topic_markdown

    page = topic_markdown(topic("terms"))
    guide = (ROOT / "docs" / "guide.md").read_text(encoding="utf-8")
    assert "## Terms used in pymetrix" in guide
    for term, meaning, _ in GLOSSARY:
        assert f"**{term}**" in page and f"**{term}**" in guide and meaning in page
