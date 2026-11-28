from .builtin import AddToDataset, ConditionalForecast, Correlogram, DefineEquation, DeriveSeries, Filter, FitEquation, FitRolling, FitVAR, FitVECM, Forecast, Hausman, Johansen, VECMLagOrder, ImpulseResponse, VARLagOrder, EquationLagSearch, FitFAVAR, VarianceDecomposition, HistoricalDecomposition, LoadData, LoadExample, OpenResult, RunTest, SelectOrder, SolveModel, CheckModel, LoadTemplate, StateSpaceForm, KalmanFilter, EstimateModel, Calibrate
from .code import RunCode
from .. import calls as _calls  # noqa: F401 - registers Call, the command that replays API calls
from .selection import DeriveEach, FitEach, SelectSeries, TestEach
from .statespace import FitBMA, FitExponentialSmoothing, FitStateSpace, FitThresholdAR, FitUnobservedComponents
from .projections import FitLocalProjection
from . import nonparametric as _nonparametric  # noqa: F401 - registers the dialogs
from .arima import FitSARIMA, FitSTLForecast, FitTheta
from .breaks import FindBreaks
from . import micro as _micro  # noqa: F401 - registers the dialogs
from .learners import ConformalForecast, ExplainLearner  # noqa: F401 - Estimate > Machine learning
from .treatment import DifferenceInDifferences, FitEventStudy, FitSyntheticControl, PropensityWeighting, RegressionDiscontinuity
from .systems import FitSystem
from .panel import BalancePanel, PanelStatistics, PivotTable, Slice, ToLong, ToWide
from .seasonal import CalendarRegressors, SeasonalAdjust
from .forecasting import AdjustRound, CombineForecasts, CompareBacktests, CompareManyForecasts, CompareRounds, MakeRound, RunBacktest, SaveRound
from .report import ExportReport
from .disaggregation import Disaggregate, Reconcile
from .midas import FitMIDAS
from .nowcast import CombineNowcasts, DataAsOf, FillEdge, FitBridge, FitDynamicFactor, RaggedEdge, RealignEdge
# data and nowcasting dialogs: frequency conversion, aggregation, connectors, factors (with the ragged-edge tools above)
from .data_tools import AggregateEntities, ConvertFrequency, LoadFromConnector
from .factors import FitFactorModel, FitPCA
from .ardl import FitARDL
from .markov import FitMarkovSwitching
from .figures import (AnnotateFigure, MarkChange, ArrangeStyles, CallAxesMethod, ChangeStyle, CoefficientPlot, CopyStyle, FigureOptions,
                      PlotSeries, SaveFigure, SetProperties, StyleFigure, TableHeatmap, UseHouseStyle)
from .samples import CheckData, NewSample, SetSample
from .sweeps import CompareScenarios, ParameterSweep
from .occbin import OccasionallyBinding, OccBinFilterCommand
from .model_source import DefineModel
from .model_tools import AddFactors, ExportModel, ImportEViewsStateSpace, ImportFRBUS, OptimalControlCommand, ShockDecomposition  # model dialogs (Model menu)
from .matrices import DefineMatrix, EnterMatrix, MatrixFrom
from .perturbation import SolveSecondOrder
from .sensitivity import MapStability  # noqa: F401 - Model > Analysis > Stability mapping
from .model_irf import ModelImpulseResponses  # noqa: F401 - Model > Analysis > Impulse responses
from .policy import OptimalSimpleRule, SolveOptimalPolicy
from .contributions import DataContributions
from .var_extra import (FitBilinear, FitBVARSV, FitMSVAR, FitQuantileVAR, FitThresholdVAR, FitVARMA, LongMemory,  # noqa: F401 - more VAR
                        SpectralAnalysis, ThresholdLinearityTest, VECMRestrictionTest)  # and time-series dialogs (dlg/var)
from .data_edit import DropSeries, EditData, FreezeSeries, GenerateSeries, RenameSeries, SeriesInfo
# dialogs added 1 November 2026: weak-instrument, non-nested and Gregory-Hansen tests, the quantile process, panel mean
# group and CCE; superior predictive ability and binary forecast scores
from .inference import FitPanelMeanGroup, GregoryHansenTest, InstrumentTests, NonNestedTest, QuantileProcessPlot
from .forecast_evaluation import BinaryForecastScores, SuperiorPredictiveAbility
from .registry import COMMANDS, Command, CommandInfo, FieldSpec, Widget, command, form_spec, option_string
from .project import load_project, save_project
from .session import CommandError, CommandRecord, Session

__all__ = [
    "COMMANDS", "AddToDataset", "ConditionalForecast", "load_project", "save_project", "Command", "Correlogram", "FitLocalProjection", "FitStateSpace", "FitSystem", "FitARDL", "FitMarkovSwitching", "FitMIDAS", "FitBridge", "FitDynamicFactor", "RaggedEdge", "DataAsOf", "FillEdge", "RealignEdge", "CombineNowcasts", "ConvertFrequency", "AggregateEntities", "LoadFromConnector", "FitPCA", "FitFactorModel", "Disaggregate", "Reconcile", "ExportReport", "AdjustRound", "CombineForecasts", "CompareBacktests", "CompareManyForecasts", "CompareRounds", "MakeRound", "RunBacktest", "SaveRound", "CalendarRegressors", "SeasonalAdjust", "BalancePanel", "PanelStatistics", "PivotTable", "Slice", "ToLong", "ToWide", "FitUnobservedComponents", "FitExponentialSmoothing", "FitBMA", "FitThresholdAR", "DeriveEach", "DeriveSeries", "FitEach", "AnnotateFigure", "MarkChange", "ArrangeStyles", "CallAxesMethod", "ChangeStyle", "CoefficientPlot", "CopyStyle", "FigureOptions", "PlotSeries", "SaveFigure", "SetProperties", "StyleFigure", "TableHeatmap", "UseHouseStyle", "SelectSeries", "TestEach", "DropSeries", "EditData", "FreezeSeries", "GenerateSeries", "RenameSeries", "SeriesInfo", "Filter", "FitRolling", "FitVAR", "FitVECM", "Hausman", "Johansen", "VECMLagOrder", "ImpulseResponse", "VARLagOrder", "EquationLagSearch", "FitFAVAR", "VarianceDecomposition", "HistoricalDecomposition", "SelectOrder", "CommandError", "CommandInfo", "CommandRecord", "DefineEquation", "DefineMatrix", "EnterMatrix", "MatrixFrom",
    "FieldSpec", "FitEquation", "LoadData", "LoadExample", "OpenResult", "RunCode", "RunTest", "Session", "SolveModel", "CheckModel", "LoadTemplate", "StateSpaceForm", "KalmanFilter", "EstimateModel", "Calibrate", "Widget",
    "command", "form_spec", "option_string",
    "AddFactors", "ExportModel", "ImportEViewsStateSpace", "ImportFRBUS", "OptimalControlCommand", "ShockDecomposition",
    "FitPanelMeanGroup", "GregoryHansenTest", "InstrumentTests", "NonNestedTest", "QuantileProcessPlot", "BinaryForecastScores",
    "SuperiorPredictiveAbility",
]
