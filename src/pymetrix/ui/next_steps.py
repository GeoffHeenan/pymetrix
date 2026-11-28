"""Next steps: from a result's window to the next stage of the work, with the result filled in.

The stages connect through their objects more than through menus: a dataset is explored and estimated on, a fitted
equation is tested, plotted, forecast with and backtested, a VAR gives impulse responses, a model is estimated and
analysed. Each result window's **Next** menu offers those steps; choosing one opens its dialog with the result filled
in, in the workspace of that stage (unless All is in use). ``NEXT_STEPS`` is the table, by type.

Below a separator, the same menu offers the task pages ("How do I…", ``pymetrix.howto``) that carry on from that
type of result: ``NEXT_PAGES``, by type, opens each in the help.
"""

from __future__ import annotations

import dataclasses
from typing import Any

__all__ = ["NEXT_PAGES", "NEXT_STEPS", "pages_for", "prefilled", "steps_for"]

# result type -> (label, command class, the field that takes the result, workspace)
NEXT_STEPS: dict[str, tuple[tuple[str, str, str, str], ...]] = {
    "Dataset": (("Plot series…", "PlotSeries", "data", "explore"), ("Check data…", "CheckData", "data", "explore"),
                ("Select series…", "SelectSeries", "data", "explore"), ("Seasonal adjustment…", "SeasonalAdjust", "data", "explore"),
                ("Test each series…", "TestEach", "data", "explore"), ("Estimate an equation…", "FitEquation", "data", "estimate")),
    "FittedEquation": (("Test…", "RunTest", "source", "estimate"), ("Coefficient plot…", "CoefficientPlot", "results", "estimate"),
                       ("Forecast…", "Forecast", "fit", "forecast"), ("Backtest…", "RunBacktest", "fit", "forecast"),
                       ("Solve as a model…", "SolveModel", "equations", "model"), ("As a matrix…", "MatrixFrom", "result", "estimate")),
    "FittedVAR": (("Impulse responses…", "ImpulseResponse", "var", "estimate"),
                  ("Variance decomposition…", "VarianceDecomposition", "var", "estimate"),
                  ("Historical decomposition…", "HistoricalDecomposition", "var", "estimate"), ("Test…", "RunTest", "source", "estimate"),
                  ("Forecast…", "Forecast", "fit", "forecast"), ("Conditional forecast…", "ConditionalForecast", "fit", "forecast"),
                  ("Backtest…", "RunBacktest", "fit", "forecast"), ("As a matrix…", "MatrixFrom", "result", "estimate")),
    "FittedVECM": (("Impulse responses…", "ImpulseResponse", "var", "estimate"),
                   ("Variance decomposition…", "VarianceDecomposition", "var", "estimate"),
                   ("Historical decomposition…", "HistoricalDecomposition", "var", "estimate"), ("Test…", "RunTest", "source", "estimate"),
                   ("VECM restrictions and stability…", "VECMRestrictionTest", "vecm", "estimate"),
                   ("Forecast…", "Forecast", "fit", "forecast"), ("As a matrix…", "MatrixFrom", "result", "estimate")),
    # more VAR types: their own (recursive) responses and decompositions; a threshold VAR's linearity test
    "FittedBVARSV": (("Impulse responses…", "ImpulseResponse", "var", "estimate"),
                     ("Variance decomposition…", "VarianceDecomposition", "var", "estimate"),
                     ("Historical decomposition…", "HistoricalDecomposition", "var", "estimate")),
    "FittedMSVAR": (("Impulse responses…", "ImpulseResponse", "var", "estimate"),
                    ("Variance decomposition…", "VarianceDecomposition", "var", "estimate"),
                    ("Historical decomposition…", "HistoricalDecomposition", "var", "estimate")),
    "FittedQuantileVAR": (("Impulse responses…", "ImpulseResponse", "var", "estimate"),
                          ("Variance decomposition…", "VarianceDecomposition", "var", "estimate")),
    "FittedThresholdVAR": (("Threshold VAR linearity (Tsay)…", "ThresholdLinearityTest", "tvar", "estimate"),),
    "Equation": (("Check…", "CheckModel", "equations", "model"), ("Solve…", "SolveModel", "equations", "model"),
                 ("Rational expectations…", "SolveRationalExpectations", "equations", "model")),
    "Template": (("Estimate model…", "EstimateModel", "source", "model"), ("State-space form…", "StateSpaceForm", "source", "model"),
                 ("Parameter sweep…", "ParameterSweep", "source", "model"), ("Export model…", "ExportModel", "source", "model")),
    "DynareModel": (("Estimate model…", "EstimateModel", "source", "model"), ("Parameter sweep…", "ParameterSweep", "source", "model"),
                    ("Export model…", "ExportModel", "source", "model")),
    "EViewsXmlModel": (("Add factors…", "AddFactors", "equations", "model"), ("Solve…", "SolveModel", "equations", "model"),
                       ("Optimal control…", "OptimalControlCommand", "equations", "model")),
    "FittedModel": (("Forecast…", "ForecastModel", "source", "model"), ("Data contributions…", "DataContributions", "source", "model"),
                    ("Shock decomposition…", "ShockDecomposition", "source", "model"),
                    ("Identification…", "CheckIdentification", "source", "model"), ("Metropolis-Hastings…", "RunMCMC", "source", "model")),
    "Posterior": (("Forecast with parameter uncertainty…", "ForecastModel", "source", "model"),),
    "StateSpaceSystem": (("Kalman filter…", "KalmanFilter", "system", "model"),),
    "Solution": (("Make a forecast round…", "MakeRound", "forecast", "forecast"),
                 ("Compare scenarios…", "CompareScenarios", "solutions", "model")),
    "ForecastRound": (("Adjust round…", "AdjustRound", "round", "forecast"), ("Save round…", "SaveRound", "round", "forecast")),
    # nowcasting and factors
    "FittedBridge": (("Combine nowcasts…", "CombineNowcasts", "models", "estimate"),),
    "FittedDynamicFactor": (("Combine nowcasts…", "CombineNowcasts", "models", "estimate"),),
    "FittedPCA": (("Add components to a dataset…", "AddToDataset", "source", "explore"),),
    "FittedFactorModel": (("Add factor scores to a dataset…", "AddToDataset", "source", "explore"),),
}


# steps added with the dialogs of 1 November 2026 (forecast evaluation; forecasts from smooth-transition models)
NEXT_STEPS["Backtest"] = (("Superior predictive ability…", "SuperiorPredictiveAbility", "benchmark", "forecast"),
                          ("Compare backtests…", "CompareBacktests", "first", "forecast"))
NEXT_STEPS["FittedSmoothTransitionAR"] = (("Forecast…", "Forecast", "fit", "forecast"),)
# a solved linear (rational-expectations) model: its impulse responses (Model > Analysis > Impulse responses)
NEXT_STEPS["LinearSolution"] = (("Impulse responses…", "ModelImpulseResponses", "solution", "model"),)

# result type -> the task pages that carry on from it (their file names in pymetrix/howto), most likely first
NEXT_PAGES: dict[str, tuple[str, ...]] = {
    "Dataset": ("estimate-and-test-an-equation", "estimate-a-var", "seasonally-adjust-a-series",
                "cointegration-in-a-short-sample", "reconcile-components-to-a-total"),
    "FittedEquation": ("estimate-and-test-an-equation", "check-a-forecasts-record", "make-a-report"),
    "FittedVAR": ("estimate-a-var", "run-a-forecast-round", "check-a-forecasts-record"),
    "TestResult": ("cointegration-in-a-short-sample",),
    "Equation": ("bring-in-an-eviews-model", "solve-a-large-backward-looking-model"),
    "EViewsXmlModel": ("solve-a-large-backward-looking-model",),
    "Template": ("bring-in-an-iris-or-dynare-model",),
    "DynareModel": ("bring-in-an-iris-or-dynare-model",),
    "Solution": ("run-a-forecast-round", "solve-a-large-backward-looking-model"),
    "ForecastRound": ("run-a-forecast-round",),
    "Backtest": ("check-a-forecasts-record",),
}


def pages_for(obj: Any) -> list[tuple[str, str]]:
    """The task pages for ``obj``'s Next menu: (title, help link), by its type (or a base type's)."""
    from ..help import link
    from ..howto import PREFIX, pages

    known = {p.id: p.title for p in pages()}
    for kind in (c.__name__ for c in type(obj).__mro__):
        if kind in NEXT_PAGES:
            return [(known[PREFIX + n], link("topic", PREFIX + n)) for n in NEXT_PAGES[kind] if PREFIX + n in known]
    return []


def steps_for(obj: Any) -> list[tuple[str, type, str, str]]:
    """The next steps for ``obj``: (label, command class, field, workspace), by its type (or a base type's)."""
    from ..commands import COMMANDS

    by_class = {i.cls.__name__: i.cls for i in COMMANDS.values()}
    own = getattr(type(obj), "next_steps", None)  # a plugin's result says its own
    if own:
        return [(label, by_class[cls], fld, ws) for label, cls, fld, ws in own if cls in by_class]
    for kind in (c.__name__ for c in type(obj).__mro__):
        if kind in NEXT_STEPS:
            return [(label, by_class[cls], field, ws) for label, cls, field, ws in NEXT_STEPS[kind] if cls in by_class]
    return []


def prefilled(cls: type, **values: Any) -> Any:
    """The dialog's command with ``values`` filled in and its other required fields empty, to finish in the dialog;
    None if it cannot be made that way."""
    filled: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        if f.name in values:
            filled[f.name] = values[f.name]
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            filled[f.name] = ""  # a required field: left for the dialog
    try:
        return cls(**filled)
    except Exception:  # noqa: BLE001 - a command that will not take empty fields: the dialog opens unfilled
        return None
