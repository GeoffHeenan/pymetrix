"""The order of the app's menus: commands grouped by task, with separators and submenus.

Each command names its menu in its decorator (``@command(menu="Data/Filter...")``); this layout says where in that
menu it goes, so related steps sit together in the order they are used. A command the layout does not list (a
plugin's, or a new one) is added at the end of its menu, after a separator, so nothing is ever left out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = ["MENU_LAYOUT", "MENU_TOPS", "MenuEntry", "access_key", "menu_entries", "menu_order", "with_access_keys"]

MENU_TOPS = ("File", "Data", "Estimate", "Test", "Model", "Forecast")  # the menu bar, left to right; plugins' menus follow

SEPARATOR = None
SPREADSHEET = "@spreadsheet"  # the app's spreadsheet editor: an action of the window, not a command
SERIES_BROWSER = "@series-browser"  # File → Browse series…: finds codes; what it loads is a Load from connector step
REFRESH_DATA = "@refresh-data"  # File → Refresh data…: reads loaded datasets again (Session.refresh), not a command

# An entry is a command's class name, SEPARATOR, an app action ("@..."), or (submenu title, [entries]). A command's
# label is its own (the part of its menu path after the "/"), unless given as (class name, label).
MENU_LAYOUT: dict[str, list[Any]] = {
    "File": ["LoadData", "LoadFromConnector", SERIES_BROWSER, REFRESH_DATA, "LoadExample", "NewDataset", "OpenResult", SEPARATOR, "ExportReport"],
    "Data": [
        "PlotSeries", SPREADSHEET, SEPARATOR,
        "SelectSeries", "GenerateSeries", "DeriveSeries", "DeriveEach", "AddToDataset", SEPARATOR,
        "CheckData", ("Sample", ["SetSample", "NewSample"]), ("Statistics", ["DescriptiveStatistics", "Correlations"]), SEPARATOR,
        ("Transform", [("Filter", "Filter (trend and cycle)..."), "SeasonalAdjust", "CalendarRegressors", "ConvertFrequency", "Disaggregate", "Reconcile"]),
        ("Panel", ["PanelStatistics", "BalancePanel", "Slice", "ToWide", "ToLong", "PivotTable", "AggregateEntities"]),
    ],
    "Estimate": [
        "FitEquation", "FitEach", "FitSystem", "FitRolling", "FitBMA", SEPARATOR,
        ("Time series", ["SelectOrder", "EquationLagSearch", "FitARDL", "FitSARIMA", "FitExponentialSmoothing", "FitTheta", "FitSTLForecast", "FitUnobservedComponents", "FitStateSpace", "LongMemory"]),
        ("Nonlinear", ["FitMarkovSwitching", "FitThresholdAR", "FitSmoothTransitionAR", "FitThresholdVAR", "FitBilinear"]),
        ("Nonparametric", ["FitKernelDensity", "FitKernelRegression"]),
        ("Mixed frequency", [("FitMIDAS", "MIDAS..."), "FitBridge", "FitDynamicFactor", SEPARATOR,
                             "RaggedEdge", "DataAsOf", "FillEdge", "RealignEdge", SEPARATOR, "CombineNowcasts"]),
        ("Factors", ["FitPCA", "FitFactorModel"]),
        ("Treatment effects", ["DifferenceInDifferences", "FitEventStudy", "FitStaggeredDiD", "BaconDecompose",
                               "RegressionDiscontinuity", "PropensityWeighting", "FitSyntheticControl", "FitSyntheticDiD"]),
        ("Microeconometrics", ["DoubleMachineLearning", "ShiftShareIV", "FitInteractiveFE", "FitConditionalLogit", "FitChoiceLogit", "ClogitMarginalEffects",
                               "FitNestedLogit", "FitMixedLogit", SEPARATOR, "FitKaplanMeier", "FitCoxPH"]),
        # estimation first (reduced-form VARs, then cointegrated systems, then local projections), then identification
        # and what an identified VAR shows (responses, decompositions), then the lag-order searches
        ("VARs and local projections", ["FitVAR", "FitPanelVAR", "FitPanelMeanGroup", "FitFAVAR", "FitBVARSV", "FitTVPVAR",
                                        "FitMSVAR", "FitQuantileVAR", "FitVARMA", SEPARATOR,
                                        "FitVECM", "FitCointegratingRegression", SEPARATOR,
                                        "FitLocalProjection", "FitGrowthAtRisk", SEPARATOR,
                                        "IdentifyStructuralVAR", "ProxySVARCommand", "ImpulseResponse", "VarianceDecomposition",
                                        "HistoricalDecomposition", SEPARATOR,
                                        "VARLagOrder", "VECMLagOrder"]),
        SEPARATOR,
        "CoefficientPlot", "QuantileProcessPlot",
    ],
    "Test": [
        "RunTest", "TestEach", "FindBreaks", "InstrumentTests", "NonNestedTest", SEPARATOR,
        "Correlogram", "CrossCorrelogram", "SpectralAnalysis", "Johansen", "GregoryHansenTest", "VECMRestrictionTest", "ThresholdLinearityTest",
        "Hausman",
    ],
    # the modelling workflow, top to bottom: write the model, check and solve it, estimate it, analyse it, use it
    # for policy and scenarios; each stage a submenu named by a noun ("Model → Estimation → Estimate model…"), so a
    # path never reads like the Estimate menu's
    "Model": [
        "DefineModel", "DefineEquation", "LoadTemplate",
        ("Import", [("ImportDynare", "Dynare file..."), ("ImportIris", "IRIS file..."), ("ImportFRBUS", "FRB/US XML file..."),
                    ("ImportEViewsStateSpace", "EViews state-space object...")]),
        "ExportModel", SEPARATOR,
        "CheckModel",
        ("Solution", ["SolveModel", "AddFactors", SEPARATOR, "SolveRationalExpectations", "SolveSecondOrder", SEPARATOR,
                      "OccasionallyBinding", "OccBinFilterCommand"]),
        SEPARATOR,
        ("Estimation", ["Calibrate", SEPARATOR, "EstimateModel", "RunMCMC", "RunSMC", SEPARATOR, "EstimateDSGEVAR"]),
        ("Analysis", [("ModelImpulseResponses", "Impulse responses..."), SEPARATOR, "CheckIdentification", SEPARATOR, "StateSpaceForm", "KalmanFilter",
                      "DataContributions", "ShockDecomposition", SEPARATOR, "ForecastModel"]),
        SEPARATOR,
        ("Policy", ["SolveOptimalPolicy", "OptimalSimpleRule", "OptimalControlCommand"]),
        ("Scenarios", ["ParameterSweep", "CompareScenarios"]),
        SEPARATOR,
        ("Matrix", [("DefineMatrix", "Expression..."), ("MatrixFrom", "From data or a result..."), ("EnterMatrix", "Enter values...")]),
    ],
    "Forecast": [  # a forecast from an estimated model first, then forecast rounds, then the evaluation of forecasts
        "Forecast", "ConditionalForecast", SEPARATOR,
        "MakeRound", "AdjustRound", "SaveRound", "CompareRounds", SEPARATOR,
        "RunBacktest", "CombineForecasts", "CompareBacktests", "CompareManyForecasts", "SuperiorPredictiveAbility", "BinaryForecastScores",
    ],
}


@dataclass
class MenuEntry:
    """One line of a menu: a command (``info``) with its ``label``, a separator, an app ``action``, or a submenu
    (``label`` and its ``entries``)."""

    kind: str  # "command", "separator", "action" or "submenu"
    label: str = ""
    info: Any = None
    action: str = ""
    entries: list[MenuEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.label = shown(self.label)


def shown(label: str) -> str:
    """A menu label as the menus show it: a trailing "..." (as commands and plugins write their menu paths) becomes
    the ellipsis "…", so every entry that opens a dialog ends the same way."""
    return label[:-3].rstrip() + "…" if label.endswith("...") else label


def _label(info: Any) -> str:
    return info.menu.partition("/")[2] or info.menu


def menu_entries(top: str, infos: list[Any]) -> list[MenuEntry]:
    """The menu ``top`` as laid out: the commands of ``infos`` whose menu path starts with it, in the layout's order,
    then any the layout does not list, after a separator. Separators never start, end or double up."""
    mine = [i for i in infos if i.menu and i.menu.partition("/")[0] == top]
    by_name = {i.cls.__name__: i for i in mine}
    placed: set[str] = set()

    def build(entries: list[Any]) -> list[MenuEntry]:
        out: list[MenuEntry] = []
        for entry in entries:
            if entry is SEPARATOR:
                out.append(MenuEntry("separator"))
            elif isinstance(entry, str) and entry.startswith("@"):
                out.append(MenuEntry("action", action=entry[1:]))
            elif isinstance(entry, tuple) and isinstance(entry[1], list):
                inner = build(entry[1])
                if any(e.kind == "command" for e in inner):
                    out.append(MenuEntry("submenu", entry[0], entries=_tidy(inner)))
            else:
                name, label = entry if isinstance(entry, tuple) else (entry, None)
                info = by_name.get(name)
                if info is not None:
                    placed.add(name)
                    out.append(MenuEntry("command", label or _label(info), info))
        return out

    laid_out = build(MENU_LAYOUT.get(top) or _plugin_layout(top))
    rest = [MenuEntry("command", _label(i), i) for i in mine if i.cls.__name__ not in placed]
    return _tidy(laid_out + ([MenuEntry("separator")] + rest if rest else []))


def _plugin_layout(top: str) -> list[Any]:
    """A plugin's layout of its own top menu (``menus`` registrations; the core menus keep theirs)."""
    from ..plugins import registry

    for entry in registry("menus").entries(available=True):
        if getattr(entry.obj, "top", None) == top and top not in MENU_LAYOUT:
            return list(entry.obj.entries)
    return []


def _tidy(entries: list[MenuEntry]) -> list[MenuEntry]:
    out: list[MenuEntry] = []
    for entry in entries:
        if entry.kind == "separator" and (not out or out[-1].kind == "separator"):
            continue
        out.append(entry)
    while out and out[-1].kind == "separator":
        out.pop()
    return out


def menu_order(infos: list[Any]) -> list[tuple[Any, str]]:
    """Every menu command of ``infos`` with its place written out ("Data → Transform → Seasonal adjustment..."), menu
    by menu in the app's order, for lists of the commands (the notebook's and the help's)."""
    tops = [t for t in MENU_TOPS if any(i.menu and i.menu.partition("/")[0] == t for i in infos)]
    tops += [t for t in dict.fromkeys(i.menu.partition("/")[0] for i in infos if i.menu) if t not in tops]
    out: list[tuple[Any, str]] = []

    def walk(entries: list[MenuEntry], path: str) -> None:
        for e in entries:
            if e.kind == "command":
                out.append((e.info, f"{path} → {e.label}"))
            elif e.kind == "submenu":
                walk(e.entries, f"{path} → {e.label}")

    for top in tops:
        walk(menu_entries(top, infos), top)
    return out


# Words an access key avoids when the label has a better one: "Data contributions" takes D or C, not the "of" of
# "Measure of fit"
_MINOR_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of", "on", "or", "per", "the", "to",
                "vs", "with"}
_WORD = re.compile(r"[A-Za-z0-9]+")


def access_key(text: str) -> str:
    """The access key ``text`` marks with "&" (lower case), or "" when it marks none ("&&" is a literal "&")."""
    plain = text.replace("&&", "")
    at = plain.find("&")
    return plain[at + 1].lower() if 0 <= at < len(plain) - 1 else ""


def _key_places(text: str) -> list[int]:
    """Where in ``text`` an access key may go, best first: the first letter of each significant word, then of the
    minor words, then the other letters and digits, left to right."""
    words = [m for m in _WORD.finditer(text) if not (m.start() and text[m.start() - 1] == "&")]
    first = [m.start() for m in words if m.group().lower() not in _MINOR_WORDS]
    minor = [m.start() for m in words if m.group().lower() in _MINOR_WORDS]
    rest = [m.start() + i for m in words for i in range(1, len(m.group()))]
    return first + minor + rest


def with_access_keys(texts: list[str], taken: str = "") -> list[str]:
    """``texts`` (one menu's lines, top to bottom) each with an access key ("&Kalman filter…"), every key used once in
    the menu. A line that marks its own keeps it; the others take, in order, the first free letter of their first
    significant word, then of the next word, and so on, then any free letter of theirs. When a line finds none free,
    an earlier line gives up its letter for its next choice if that frees one (so a long menu still keys every line
    it can); a line no rearrangement can serve is left as it is. ``taken``: letters already used (by lines not
    passed)."""
    used = set(taken.lower()) | {access_key(t) for t in texts} - {""}
    choices = {n: list(dict.fromkeys(t[i].lower() for i in _key_places(t))) for n, t in enumerate(texts) if not access_key(t)}
    owner: dict[str, int] = {}  # letter -> the line given it
    given: dict[int, str] = {}

    def place(n: int, seen: set[str]) -> bool:
        free = [c for c in choices[n] if c not in used and c not in owner]
        if free:
            owner[free[0]], given[n] = n, free[0]
            return True
        for c in choices[n]:  # take a letter whose line can move to another of its own
            if c in used or c in seen:
                continue
            seen.add(c)
            if place_other(owner[c], c, seen):
                owner[c], given[n] = n, c
                return True
        return False

    def place_other(m: int, losing: str, seen: set[str]) -> bool:
        free = [c for c in choices[m] if c != losing and c not in used and c not in owner]
        if free:
            owner[free[0]], given[m] = m, free[0]
            return True
        for c in choices[m]:
            if c == losing or c in used or c in seen:
                continue
            seen.add(c)
            if place_other(owner[c], c, seen):
                owner[c], given[m] = m, c
                return True
        return False

    for n in choices:
        place(n, set())
    out = []
    for n, text in enumerate(texts):
        if n in given:
            at = next(i for i in _key_places(text) if text[i].lower() == given[n])
            text = text[:at] + "&" + text[at:]
        out.append(text)
    return out
