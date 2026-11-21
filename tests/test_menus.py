"""The app's menus: every command placed once, by task, with separators and submenus; others added at the end."""

import os

import pytest

from pymetrix.commands import COMMANDS
from pymetrix.commands.menus import MENU_LAYOUT, menu_entries, menu_order


def _names(entries):
    out = []
    for entry in entries:
        if entry is None or (isinstance(entry, str) and entry.startswith("@")):
            continue
        if isinstance(entry, tuple) and isinstance(entry[1], list):
            out += _names(entry[1])
        else:
            out.append(entry[0] if isinstance(entry, tuple) else entry)
    return out


def _flat(entries):
    for e in entries:
        if e.kind == "submenu":
            yield from _flat(e.entries)
        elif e.kind == "command":
            yield e


def test_every_command_of_a_laid_out_menu_has_one_place():
    infos = [i for i in COMMANDS.values() if i.menu and i.plugin == "pymetrix"]
    for top, layout in MENU_LAYOUT.items():
        listed = _names(layout)
        assert len(listed) == len(set(listed)), f"{top}: a command listed twice"
        own = {i.cls.__name__ for i in infos if i.menu.partition("/")[0] == top}
        assert set(listed) == own, f"{top}: not laid out {sorted(own - set(listed))}; not in {top} {sorted(set(listed) - own)}"
    assert len(menu_order(infos)) == len(infos)  # every menu command once, menus not laid out (Forecast) included


def test_menus_group_by_task_and_tidy_their_separators():
    infos = [i for i in COMMANDS.values() if i.menu]
    data = menu_entries("Data", infos)
    assert [e.label if e.kind != "action" else "@" + e.action for e in data[:2]] == ["Plot series…", "@spreadsheet"]
    submenus = {e.label: [c.info.cls.__name__ for c in _flat(e.entries)] for e in data if e.kind == "submenu"}
    assert submenus == {"Sample": ["SetSample", "NewSample"], "Statistics": ["DescriptiveStatistics", "Correlations"], "Transform": ["Filter", "SeasonalAdjust", "CalendarRegressors", "ConvertFrequency", "Disaggregate", "Reconcile"],
                        "Panel": ["PanelStatistics", "BalancePanel", "Slice", "ToWide", "ToLong", "PivotTable", "AggregateEntities"]}
    assert sum(e.kind != "separator" for e in data) == 12  # 21 lines before
    for top in ("Data", "Estimate", "Test", "Model", "File", "Forecast"):
        entries = menu_entries(top, infos)
        assert entries[0].kind != "separator" and entries[-1].kind != "separator"
        assert not any(a.kind == b.kind == "separator" for a, b in zip(entries, entries[1:]))
    paths = dict((i.cls.__name__, p) for i, p in menu_order(infos))
    assert paths["Filter"] == "Data → Transform → Filter (trend and cycle)…"
    assert paths["ImportDynare"] == "Model → Import → Dynare file…" and paths["Correlogram"] == "Test → Correlogram…"
    assert paths["DescriptiveStatistics"] == "Data → Statistics → Descriptive statistics…" and paths["CrossCorrelogram"] == "Test → Cross-correlogram…"
    assert list(paths)[0] == "LoadData"  # the menu bar's order: File first


def test_a_command_the_layout_does_not_list_goes_at_the_end_of_its_menu():
    from types import SimpleNamespace

    infos = [i for i in COMMANDS.values() if i.menu]
    extra = SimpleNamespace(menu="Estimate/My plugin model...", cls=type("MyPluginModel", (), {}))
    entries = menu_entries("Estimate", [*infos, extra])
    assert entries[-1].info is extra and entries[-1].label == "My plugin model…" and entries[-2].kind == "separator"
    lonely = menu_entries("Extras", [SimpleNamespace(menu="Extras/Thing...", cls=type("Thing", (), {}))])
    assert [e.label for e in lonely] == ["Thing…"]  # a menu with no layout: its commands as registered


def test_the_window_builds_the_menus_as_laid_out():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.commands import Session
    from pymetrix.ui.qt import main_window

    window = main_window(Session())
    try:
        data = window.command_menus["Data"]
        lines = [(a.text().replace("&", ""), a.menu().title().replace("&", "") if a.menu() else None) for a in data.actions() if not a.isSeparator()]
        assert lines[0] == ("Plot series…", None) and lines[1][0] == "Spreadsheet editor…"
        assert [a for a in data.actions() if a.text().replace("&", "") == "Spreadsheet editor…"][0].shortcut().toString() == "Ctrl+E"
        transform = next(a.menu() for a in data.actions() if a.menu() and a.menu().title().replace("&", "") == "Transform")
        assert [a.text().replace("&", "") for a in transform.actions()] == ["Filter (trend and cycle)…", "Seasonal adjustment…",
                                                           "Calendar regressors…", "Convert frequency…", "Temporal disaggregation…",
                                                           "Reconcile components…"]
        titles = [a.text().replace("&", "") for a in window.menuBar().actions()]
        assert titles[:7] == ["File", "Edit", "Data", "Estimate", "Test", "Model", "Forecast"]  # Edit (Undo) after File
        assert "Correlogram…" in [a.text().replace("&", "") for a in window.command_menus["Test"].actions()]
    finally:
        window.close()


def test_the_notebook_lists_commands_in_the_same_order():
    pytest.importorskip("ipywidgets")
    from pymetrix.commands import Session
    from pymetrix.ui.jupyter import menu

    acc = menu(Session())
    titles = [acc.get_title(n) for n in range(len(acc.children))]
    assert titles[0] == "File → Load data…" and "Data → Transform → Seasonal adjustment…" in titles


def test_model_follows_the_workflow_and_vars_put_estimation_first():
    infos = [i for i in COMMANDS.values() if i.menu]
    model = menu_entries("Model", infos)
    top = [e.label for e in model if e.kind != "separator"]
    assert top == ["Model editor…", "Define equation…", "Template model…", "Import", "Export model…", "Check model…",
                   "Solution", "Estimation", "Analysis", "Policy", "Scenarios", "Matrix"]
    stages = {e.label: [c.info.cls.__name__ for c in _flat(e.entries)] for e in model if e.kind == "submenu"}
    assert stages["Solution"] == ["SolveModel", "AddFactors", "SolveRationalExpectations", "SolveSecondOrder",
                                  "OccasionallyBinding", "OccBinFilterCommand"]
    assert stages["Estimation"] == ["Calibrate", "EstimateModel", "RunMCMC", "RunSMC", "EstimateDSGEVAR"]
    assert stages["Analysis"] == ["ModelImpulseResponses", "CheckIdentification", "StateSpaceForm", "KalmanFilter",
                                  "DataContributions", "ShockDecomposition", "ForecastModel"]
    paths = dict((i.cls.__name__, p) for i, p in menu_order(infos))
    assert paths["RunMCMC"] == "Model → Estimation → Metropolis-Hastings…"
    assert paths["KalmanFilter"] == "Model → Analysis → Kalman filter…"
    assert paths["ModelImpulseResponses"] == "Model → Analysis → Impulse responses…"
    estimate = menu_entries("Estimate", infos)
    vars_ = next(e for e in estimate if e.kind == "submenu" and e.label == "VARs and local projections")
    names = [c.info.cls.__name__ for c in _flat(vars_.entries)]
    fits = [n for n in names if n.startswith("Fit")]
    assert names[:len(fits)] == fits  # every estimator before identification, responses and decompositions
    assert names.index("IdentifyStructuralVAR") < names.index("ImpulseResponse") < names.index("HistoricalDecomposition")
    assert names[-2:] == ["VARLagOrder", "VECMLagOrder"]  # the lag-order searches last
